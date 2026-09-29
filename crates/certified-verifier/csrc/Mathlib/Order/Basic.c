// Lean compiler output
// Module: Mathlib.Order.Basic
// Imports: public import Init public meta import Init public import Mathlib.Data.Subtype public import Mathlib.Order.Defs.LinearOrder public import Mathlib.Order.Defs.Prop public import Mathlib.Order.Notation public import Mathlib.Tactic.Spread public import Mathlib.Tactic.Convert public import Mathlib.Tactic.Inhabit public import Mathlib.Tactic.SimpRw public import Mathlib.Tactic.GCongr public import Mathlib.Tactic.Attr.Register public import Mathlib.Tactic.FastInstance
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
lean_object* lp_mathlib_decidableEqOfDecidableLE___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_decidableLTOfDecidableLE___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lp_mathlib_decidableLTOfDecidableLE___redArg(lean_object*, lean_object*, lean_object*);
uint8_t lp_mathlib_decidableEqOfDecidableLE___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkFreshLevelMVar(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_forallMetaTelescope(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_mvarId_x21(lean_object*);
lean_object* lp_batteries_Lean_MVarId_assignIfDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkAppN(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_instInhabitedExpr;
lean_object* l_instOrdSubtype___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term___u207b_xb9_x27o___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 10, .m_data = "term_⁻¹'o_"};
static const lean_object* lp_mathlib_term___u207b_xb9_x27o___00__closed__0 = (const lean_object*)&lp_mathlib_term___u207b_xb9_x27o___00__closed__0_value;
static const lean_ctor_object lp_mathlib_term___u207b_xb9_x27o___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u207b_xb9_x27o___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(120, 188, 102, 51, 227, 78, 49, 38)}};
static const lean_object* lp_mathlib_term___u207b_xb9_x27o___00__closed__1 = (const lean_object*)&lp_mathlib_term___u207b_xb9_x27o___00__closed__1_value;
static const lean_string_object lp_mathlib_term___u207b_xb9_x27o___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_term___u207b_xb9_x27o___00__closed__2 = (const lean_object*)&lp_mathlib_term___u207b_xb9_x27o___00__closed__2_value;
static const lean_ctor_object lp_mathlib_term___u207b_xb9_x27o___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u207b_xb9_x27o___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_term___u207b_xb9_x27o___00__closed__3 = (const lean_object*)&lp_mathlib_term___u207b_xb9_x27o___00__closed__3_value;
static const lean_string_object lp_mathlib_term___u207b_xb9_x27o___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 6, .m_data = " ⁻¹'o "};
static const lean_object* lp_mathlib_term___u207b_xb9_x27o___00__closed__4 = (const lean_object*)&lp_mathlib_term___u207b_xb9_x27o___00__closed__4_value;
static const lean_ctor_object lp_mathlib_term___u207b_xb9_x27o___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u207b_xb9_x27o___00__closed__4_value)}};
static const lean_object* lp_mathlib_term___u207b_xb9_x27o___00__closed__5 = (const lean_object*)&lp_mathlib_term___u207b_xb9_x27o___00__closed__5_value;
static const lean_string_object lp_mathlib_term___u207b_xb9_x27o___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_term___u207b_xb9_x27o___00__closed__6 = (const lean_object*)&lp_mathlib_term___u207b_xb9_x27o___00__closed__6_value;
static const lean_ctor_object lp_mathlib_term___u207b_xb9_x27o___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u207b_xb9_x27o___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_term___u207b_xb9_x27o___00__closed__7 = (const lean_object*)&lp_mathlib_term___u207b_xb9_x27o___00__closed__7_value;
static const lean_ctor_object lp_mathlib_term___u207b_xb9_x27o___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_term___u207b_xb9_x27o___00__closed__7_value),((lean_object*)(((size_t)(81) << 1) | 1))}};
static const lean_object* lp_mathlib_term___u207b_xb9_x27o___00__closed__8 = (const lean_object*)&lp_mathlib_term___u207b_xb9_x27o___00__closed__8_value;
static const lean_ctor_object lp_mathlib_term___u207b_xb9_x27o___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u207b_xb9_x27o___00__closed__3_value),((lean_object*)&lp_mathlib_term___u207b_xb9_x27o___00__closed__5_value),((lean_object*)&lp_mathlib_term___u207b_xb9_x27o___00__closed__8_value)}};
static const lean_object* lp_mathlib_term___u207b_xb9_x27o___00__closed__9 = (const lean_object*)&lp_mathlib_term___u207b_xb9_x27o___00__closed__9_value;
static const lean_ctor_object lp_mathlib_term___u207b_xb9_x27o___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___u207b_xb9_x27o___00__closed__1_value),((lean_object*)(((size_t)(80) << 1) | 1)),((lean_object*)(((size_t)(80) << 1) | 1)),((lean_object*)&lp_mathlib_term___u207b_xb9_x27o___00__closed__9_value)}};
static const lean_object* lp_mathlib_term___u207b_xb9_x27o___00__closed__10 = (const lean_object*)&lp_mathlib_term___u207b_xb9_x27o___00__closed__10_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___u207b_xb9_x27o__ = (const lean_object*)&lp_mathlib_term___u207b_xb9_x27o___00__closed__10_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__0_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__1_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__2_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "Order.Preimage"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__5_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__6;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Order"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__7 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__7_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Preimage"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__8 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__8_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(102, 52, 160, 208, 138, 190, 17, 238)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__9_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(239, 180, 27, 139, 134, 251, 96, 240)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__9 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__9_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__10 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__10_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__10_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__11 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__11_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__12 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__12_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__13 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__13_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Basic______unexpand__Order__Preimage__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Basic______unexpand__Order__Preimage__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Basic______unexpand__Order__Preimage__1___closed__0_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Basic______unexpand__Order__Preimage__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Basic______unexpand__Order__Preimage__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Basic______unexpand__Order__Preimage__1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Basic______unexpand__Order__Preimage__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Basic______unexpand__Order__Preimage__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Basic______unexpand__Order__Preimage__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Order_Preimage_decidable___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Order_Preimage_decidable___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Order_Preimage_decidable(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Order_Preimage_decidable___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_GCongr_exactLeOfLt___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "le_of_lt"};
static const lean_object* lp_mathlib_Mathlib_Tactic_GCongr_exactLeOfLt___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_GCongr_exactLeOfLt___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_GCongr_exactLeOfLt___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_GCongr_exactLeOfLt___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(26, 46, 54, 245, 81, 108, 136, 63)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_GCongr_exactLeOfLt___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_GCongr_exactLeOfLt___lam__0___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GCongr_exactLeOfLt___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GCongr_exactLeOfLt___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_GCongr_exactLeOfLt___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_GCongr_exactLeOfLt___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GCongr_exactLeOfLt;
LEAN_EXPORT lean_object* lp_mathlib_Prop_instCompl;
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompl___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompl___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompl(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instHNot___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instHNot(lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Pi_preorder___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Pi_preorder___closed__0 = (const lean_object*)&lp_mathlib_Pi_preorder___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Pi_preorder(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_preorder___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_partialOrder___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_partialOrder___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_partialOrder(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__3_value),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(11, 240, 1, 62, 92, 163, 173, 149)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Basic"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__5_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__4_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__5_value),LEAN_SCALAR_PTR_LITERAL(84, 87, 176, 83, 227, 131, 36, 254)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__6_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(85, 175, 18, 46, 68, 51, 101, 35)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__7_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 7, .m_data = "term_≺_"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__7_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__8_value),LEAN_SCALAR_PTR_LITERAL(109, 16, 232, 12, 34, 36, 220, 84)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__9_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 3, .m_data = " ≺ "};
static const lean_object* lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__10_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_term___u207b_xb9_x27o___00__closed__7_value),((lean_object*)(((size_t)(51) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__12_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u207b_xb9_x27o___00__closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__11_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__12_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__13_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__9_value),((lean_object*)(((size_t)(50) << 1) | 1)),((lean_object*)(((size_t)(50) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__13_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__14_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a__ = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__14_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Basic_0____aux__Mathlib__Order__Basic______macroRules____private__Mathlib__Order__Basic__0__term___u227a____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "StrongLT"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Basic_0____aux__Mathlib__Order__Basic______macroRules____private__Mathlib__Order__Basic__0__term___u227a____1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Basic_0____aux__Mathlib__Order__Basic______macroRules____private__Mathlib__Order__Basic__0__term___u227a____1___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Order_Basic_0____aux__Mathlib__Order__Basic______macroRules____private__Mathlib__Order__Basic__0__term___u227a____1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Order_Basic_0____aux__Mathlib__Order__Basic______macroRules____private__Mathlib__Order__Basic__0__term___u227a____1___closed__1;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Basic_0____aux__Mathlib__Order__Basic______macroRules____private__Mathlib__Order__Basic__0__term___u227a____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Order_Basic_0____aux__Mathlib__Order__Basic______macroRules____private__Mathlib__Order__Basic__0__term___u227a____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(129, 66, 62, 1, 107, 224, 114, 233)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Basic_0____aux__Mathlib__Order__Basic______macroRules____private__Mathlib__Order__Basic__0__term___u227a____1___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Basic_0____aux__Mathlib__Order__Basic______macroRules____private__Mathlib__Order__Basic__0__term___u227a____1___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Basic_0____aux__Mathlib__Order__Basic______macroRules____private__Mathlib__Order__Basic__0__term___u227a____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Basic_0____aux__Mathlib__Order__Basic______macroRules____private__Mathlib__Order__Basic__0__term___u227a____1___closed__2_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Basic_0____aux__Mathlib__Order__Basic______macroRules____private__Mathlib__Order__Basic__0__term___u227a____1___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Basic_0____aux__Mathlib__Order__Basic______macroRules____private__Mathlib__Order__Basic__0__term___u227a____1___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Basic_0____aux__Mathlib__Order__Basic______macroRules____private__Mathlib__Order__Basic__0__term___u227a____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Basic_0____aux__Mathlib__Order__Basic______macroRules____private__Mathlib__Order__Basic__0__term___u227a____1___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Basic_0____aux__Mathlib__Order__Basic______macroRules____private__Mathlib__Order__Basic__0__term___u227a____1___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Basic_0____aux__Mathlib__Order__Basic______macroRules____private__Mathlib__Order__Basic__0__term___u227a____1___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Basic_0____aux__Mathlib__Order__Basic______macroRules____private__Mathlib__Order__Basic__0__term___u227a____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Basic_0____aux__Mathlib__Order__Basic______macroRules____private__Mathlib__Order__Basic__0__term___u227a____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Basic_0____aux__Mathlib__Order__Basic______unexpand__StrongLT__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Basic_0____aux__Mathlib__Order__Basic______unexpand__StrongLT__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instSDiff___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instSDiff___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instSDiff(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instHImp___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instHImp___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instHImp(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_preorder___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_preorder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_preorder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_partialOrder___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_partialOrder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_partialOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_linearOrder___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_linearOrder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_linearOrder___boxed(lean_object**);
static const lean_ctor_object lp_mathlib_Preorder_lift___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Preorder_lift___closed__0 = (const lean_object*)&lp_mathlib_Preorder_lift___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Preorder_lift(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Preorder_lift___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PartialOrder_lift(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PartialOrder_lift___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_LinearOrder_lift___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_lift___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_LinearOrder_lift___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_lift___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_LinearOrder_lift___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_lift___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_LinearOrder_lift___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_lift___redArg___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_lift___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_lift(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_lift_x27___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_lift_x27___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_lift_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_lift_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_LinearOrder_liftWithOrd___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_liftWithOrd___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_LinearOrder_liftWithOrd___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_liftWithOrd___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_LinearOrder_liftWithOrd___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_liftWithOrd___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_liftWithOrd___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_liftWithOrd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_liftWithOrd_x27___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_liftWithOrd_x27___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_liftWithOrd_x27___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_liftWithOrd_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_preorder(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_preorder___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_partialOrder___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_partialOrder___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_partialOrder(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_partialOrder___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Subtype_decidableLE___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_decidableLE___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Subtype_decidableLE(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_decidableLE___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Subtype_decidableLT___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_decidableLT___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Subtype_decidableLT(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_decidableLT___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Subtype_instLinearOrder___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLinearOrder___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Subtype_instLinearOrder___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLinearOrder___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Subtype_instLinearOrder___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLinearOrder___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLinearOrder___redArg___lam__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLinearOrder___redArg___lam__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLinearOrder___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLinearOrder(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLE__mathlib(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Prod_instDecidableLE___redArg(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instDecidableLE___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Prod_instDecidableLE(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instDecidableLE___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Prod_instPreorder___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Prod_instPreorder___closed__0 = (const lean_object*)&lp_mathlib_Prod_instPreorder___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Prod_instPreorder(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instPreorder___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instPartialOrder___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instPartialOrder___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instPartialOrder(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instPartialOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_ofSubsingleton___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_ofSubsingleton___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_ofSubsingleton___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_ofSubsingleton___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_LinearOrder_ofSubsingleton___lam__2(uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_ofSubsingleton___lam__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_LinearOrder_ofSubsingleton___lam__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_ofSubsingleton___lam__4___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_LinearOrder_ofSubsingleton___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LinearOrder_ofSubsingleton___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_LinearOrder_ofSubsingleton___closed__0 = (const lean_object*)&lp_mathlib_LinearOrder_ofSubsingleton___closed__0_value;
static const lean_closure_object lp_mathlib_LinearOrder_ofSubsingleton___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LinearOrder_ofSubsingleton___lam__1___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_LinearOrder_ofSubsingleton___closed__1 = (const lean_object*)&lp_mathlib_LinearOrder_ofSubsingleton___closed__1_value;
static const lean_closure_object lp_mathlib_LinearOrder_ofSubsingleton___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LinearOrder_ofSubsingleton___lam__2___boxed, .m_arity = 3, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1))} };
static const lean_object* lp_mathlib_LinearOrder_ofSubsingleton___closed__2 = (const lean_object*)&lp_mathlib_LinearOrder_ofSubsingleton___closed__2_value;
static const lean_closure_object lp_mathlib_LinearOrder_ofSubsingleton___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LinearOrder_ofSubsingleton___lam__4___boxed, .m_arity = 3, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_LinearOrder_ofSubsingleton___closed__2_value)} };
static const lean_object* lp_mathlib_LinearOrder_ofSubsingleton___closed__3 = (const lean_object*)&lp_mathlib_LinearOrder_ofSubsingleton___closed__3_value;
static const lean_closure_object lp_mathlib_LinearOrder_ofSubsingleton___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_decidableEqOfDecidableLE___boxed, .m_arity = 5, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Preorder_lift___closed__0_value),((lean_object*)&lp_mathlib_LinearOrder_ofSubsingleton___closed__2_value)} };
static const lean_object* lp_mathlib_LinearOrder_ofSubsingleton___closed__4 = (const lean_object*)&lp_mathlib_LinearOrder_ofSubsingleton___closed__4_value;
static const lean_closure_object lp_mathlib_LinearOrder_ofSubsingleton___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_decidableLTOfDecidableLE___boxed, .m_arity = 5, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Preorder_lift___closed__0_value),((lean_object*)&lp_mathlib_LinearOrder_ofSubsingleton___closed__2_value)} };
static const lean_object* lp_mathlib_LinearOrder_ofSubsingleton___closed__5 = (const lean_object*)&lp_mathlib_LinearOrder_ofSubsingleton___closed__5_value;
static const lean_ctor_object lp_mathlib_LinearOrder_ofSubsingleton___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*7 + 0, .m_other = 7, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Preorder_lift___closed__0_value),((lean_object*)&lp_mathlib_LinearOrder_ofSubsingleton___closed__0_value),((lean_object*)&lp_mathlib_LinearOrder_ofSubsingleton___closed__1_value),((lean_object*)&lp_mathlib_LinearOrder_ofSubsingleton___closed__3_value),((lean_object*)&lp_mathlib_LinearOrder_ofSubsingleton___closed__2_value),((lean_object*)&lp_mathlib_LinearOrder_ofSubsingleton___closed__4_value),((lean_object*)&lp_mathlib_LinearOrder_ofSubsingleton___closed__5_value)}};
static const lean_object* lp_mathlib_LinearOrder_ofSubsingleton___closed__6 = (const lean_object*)&lp_mathlib_LinearOrder_ofSubsingleton___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_ofSubsingleton(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instLinearOrderEmpty___lam__0(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderEmpty___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instLinearOrderEmpty___lam__1(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderEmpty___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instLinearOrderEmpty___lam__2(uint8_t, uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderEmpty___lam__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instLinearOrderEmpty___lam__4(lean_object*, uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderEmpty___lam__4___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_instLinearOrderEmpty___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_instLinearOrderEmpty___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instLinearOrderEmpty___closed__0 = (const lean_object*)&lp_mathlib_instLinearOrderEmpty___closed__0_value;
static const lean_closure_object lp_mathlib_instLinearOrderEmpty___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_instLinearOrderEmpty___lam__1___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instLinearOrderEmpty___closed__1 = (const lean_object*)&lp_mathlib_instLinearOrderEmpty___closed__1_value;
static const lean_closure_object lp_mathlib_instLinearOrderEmpty___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_instLinearOrderEmpty___lam__2___boxed, .m_arity = 3, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1))} };
static const lean_object* lp_mathlib_instLinearOrderEmpty___closed__2 = (const lean_object*)&lp_mathlib_instLinearOrderEmpty___closed__2_value;
static const lean_closure_object lp_mathlib_instLinearOrderEmpty___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_instLinearOrderEmpty___lam__4___boxed, .m_arity = 3, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_instLinearOrderEmpty___closed__2_value)} };
static const lean_object* lp_mathlib_instLinearOrderEmpty___closed__3 = (const lean_object*)&lp_mathlib_instLinearOrderEmpty___closed__3_value;
static const lean_ctor_object lp_mathlib_instLinearOrderEmpty___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_instLinearOrderEmpty___closed__4 = (const lean_object*)&lp_mathlib_instLinearOrderEmpty___closed__4_value;
static const lean_closure_object lp_mathlib_instLinearOrderEmpty___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_decidableEqOfDecidableLE___boxed, .m_arity = 5, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_instLinearOrderEmpty___closed__4_value),((lean_object*)&lp_mathlib_instLinearOrderEmpty___closed__2_value)} };
static const lean_object* lp_mathlib_instLinearOrderEmpty___closed__5 = (const lean_object*)&lp_mathlib_instLinearOrderEmpty___closed__5_value;
static const lean_closure_object lp_mathlib_instLinearOrderEmpty___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_decidableLTOfDecidableLE___boxed, .m_arity = 5, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_instLinearOrderEmpty___closed__4_value),((lean_object*)&lp_mathlib_instLinearOrderEmpty___closed__2_value)} };
static const lean_object* lp_mathlib_instLinearOrderEmpty___closed__6 = (const lean_object*)&lp_mathlib_instLinearOrderEmpty___closed__6_value;
static const lean_ctor_object lp_mathlib_instLinearOrderEmpty___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*7 + 0, .m_other = 7, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_instLinearOrderEmpty___closed__4_value),((lean_object*)&lp_mathlib_instLinearOrderEmpty___closed__0_value),((lean_object*)&lp_mathlib_instLinearOrderEmpty___closed__1_value),((lean_object*)&lp_mathlib_instLinearOrderEmpty___closed__3_value),((lean_object*)&lp_mathlib_instLinearOrderEmpty___closed__2_value),((lean_object*)&lp_mathlib_instLinearOrderEmpty___closed__5_value),((lean_object*)&lp_mathlib_instLinearOrderEmpty___closed__6_value)}};
static const lean_object* lp_mathlib_instLinearOrderEmpty___closed__7 = (const lean_object*)&lp_mathlib_instLinearOrderEmpty___closed__7_value;
LEAN_EXPORT const lean_object* lp_mathlib_instLinearOrderEmpty = (const lean_object*)&lp_mathlib_instLinearOrderEmpty___closed__7_value;
LEAN_EXPORT uint8_t lp_mathlib_instLinearOrderPEmpty___lam__0(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderPEmpty___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instLinearOrderPEmpty___lam__1(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderPEmpty___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instLinearOrderPEmpty___lam__2(uint8_t, uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderPEmpty___lam__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instLinearOrderPEmpty___lam__4(lean_object*, uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderPEmpty___lam__4___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_instLinearOrderPEmpty___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_instLinearOrderPEmpty___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instLinearOrderPEmpty___closed__0 = (const lean_object*)&lp_mathlib_instLinearOrderPEmpty___closed__0_value;
static const lean_closure_object lp_mathlib_instLinearOrderPEmpty___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_instLinearOrderPEmpty___lam__1___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instLinearOrderPEmpty___closed__1 = (const lean_object*)&lp_mathlib_instLinearOrderPEmpty___closed__1_value;
static const lean_closure_object lp_mathlib_instLinearOrderPEmpty___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_instLinearOrderPEmpty___lam__2___boxed, .m_arity = 3, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1))} };
static const lean_object* lp_mathlib_instLinearOrderPEmpty___closed__2 = (const lean_object*)&lp_mathlib_instLinearOrderPEmpty___closed__2_value;
static const lean_closure_object lp_mathlib_instLinearOrderPEmpty___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_instLinearOrderPEmpty___lam__4___boxed, .m_arity = 3, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_instLinearOrderPEmpty___closed__2_value)} };
static const lean_object* lp_mathlib_instLinearOrderPEmpty___closed__3 = (const lean_object*)&lp_mathlib_instLinearOrderPEmpty___closed__3_value;
static const lean_ctor_object lp_mathlib_instLinearOrderPEmpty___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_instLinearOrderPEmpty___closed__4 = (const lean_object*)&lp_mathlib_instLinearOrderPEmpty___closed__4_value;
static const lean_closure_object lp_mathlib_instLinearOrderPEmpty___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_decidableEqOfDecidableLE___boxed, .m_arity = 5, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_instLinearOrderPEmpty___closed__4_value),((lean_object*)&lp_mathlib_instLinearOrderPEmpty___closed__2_value)} };
static const lean_object* lp_mathlib_instLinearOrderPEmpty___closed__5 = (const lean_object*)&lp_mathlib_instLinearOrderPEmpty___closed__5_value;
static const lean_closure_object lp_mathlib_instLinearOrderPEmpty___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_decidableLTOfDecidableLE___boxed, .m_arity = 5, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_instLinearOrderPEmpty___closed__4_value),((lean_object*)&lp_mathlib_instLinearOrderPEmpty___closed__2_value)} };
static const lean_object* lp_mathlib_instLinearOrderPEmpty___closed__6 = (const lean_object*)&lp_mathlib_instLinearOrderPEmpty___closed__6_value;
static const lean_ctor_object lp_mathlib_instLinearOrderPEmpty___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*7 + 0, .m_other = 7, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_instLinearOrderPEmpty___closed__4_value),((lean_object*)&lp_mathlib_instLinearOrderPEmpty___closed__0_value),((lean_object*)&lp_mathlib_instLinearOrderPEmpty___closed__1_value),((lean_object*)&lp_mathlib_instLinearOrderPEmpty___closed__3_value),((lean_object*)&lp_mathlib_instLinearOrderPEmpty___closed__2_value),((lean_object*)&lp_mathlib_instLinearOrderPEmpty___closed__5_value),((lean_object*)&lp_mathlib_instLinearOrderPEmpty___closed__6_value)}};
static const lean_object* lp_mathlib_instLinearOrderPEmpty___closed__7 = (const lean_object*)&lp_mathlib_instLinearOrderPEmpty___closed__7_value;
LEAN_EXPORT const lean_object* lp_mathlib_instLinearOrderPEmpty = (const lean_object*)&lp_mathlib_instLinearOrderPEmpty___closed__7_value;
LEAN_EXPORT lean_object* lp_mathlib_PUnit_instLinearOrder___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_instLinearOrder___lam__1(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_PUnit_instLinearOrder___lam__2(uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_instLinearOrder___lam__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_PUnit_instLinearOrder___lam__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_instLinearOrder___lam__4___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_PUnit_instLinearOrder___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_PUnit_instLinearOrder___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_PUnit_instLinearOrder___closed__0 = (const lean_object*)&lp_mathlib_PUnit_instLinearOrder___closed__0_value;
static const lean_closure_object lp_mathlib_PUnit_instLinearOrder___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_PUnit_instLinearOrder___lam__1, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_PUnit_instLinearOrder___closed__1 = (const lean_object*)&lp_mathlib_PUnit_instLinearOrder___closed__1_value;
static const lean_closure_object lp_mathlib_PUnit_instLinearOrder___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_PUnit_instLinearOrder___lam__2___boxed, .m_arity = 3, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1))} };
static const lean_object* lp_mathlib_PUnit_instLinearOrder___closed__2 = (const lean_object*)&lp_mathlib_PUnit_instLinearOrder___closed__2_value;
static const lean_closure_object lp_mathlib_PUnit_instLinearOrder___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_PUnit_instLinearOrder___lam__4___boxed, .m_arity = 3, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_PUnit_instLinearOrder___closed__2_value)} };
static const lean_object* lp_mathlib_PUnit_instLinearOrder___closed__3 = (const lean_object*)&lp_mathlib_PUnit_instLinearOrder___closed__3_value;
static const lean_ctor_object lp_mathlib_PUnit_instLinearOrder___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_PUnit_instLinearOrder___closed__4 = (const lean_object*)&lp_mathlib_PUnit_instLinearOrder___closed__4_value;
static const lean_closure_object lp_mathlib_PUnit_instLinearOrder___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_decidableEqOfDecidableLE___boxed, .m_arity = 5, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_PUnit_instLinearOrder___closed__4_value),((lean_object*)&lp_mathlib_PUnit_instLinearOrder___closed__2_value)} };
static const lean_object* lp_mathlib_PUnit_instLinearOrder___closed__5 = (const lean_object*)&lp_mathlib_PUnit_instLinearOrder___closed__5_value;
static const lean_closure_object lp_mathlib_PUnit_instLinearOrder___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_decidableLTOfDecidableLE___boxed, .m_arity = 5, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_PUnit_instLinearOrder___closed__4_value),((lean_object*)&lp_mathlib_PUnit_instLinearOrder___closed__2_value)} };
static const lean_object* lp_mathlib_PUnit_instLinearOrder___closed__6 = (const lean_object*)&lp_mathlib_PUnit_instLinearOrder___closed__6_value;
static const lean_ctor_object lp_mathlib_PUnit_instLinearOrder___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*7 + 0, .m_other = 7, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_PUnit_instLinearOrder___closed__4_value),((lean_object*)&lp_mathlib_PUnit_instLinearOrder___closed__0_value),((lean_object*)&lp_mathlib_PUnit_instLinearOrder___closed__1_value),((lean_object*)&lp_mathlib_PUnit_instLinearOrder___closed__3_value),((lean_object*)&lp_mathlib_PUnit_instLinearOrder___closed__2_value),((lean_object*)&lp_mathlib_PUnit_instLinearOrder___closed__5_value),((lean_object*)&lp_mathlib_PUnit_instLinearOrder___closed__6_value)}};
static const lean_object* lp_mathlib_PUnit_instLinearOrder___closed__7 = (const lean_object*)&lp_mathlib_PUnit_instLinearOrder___closed__7_value;
LEAN_EXPORT const lean_object* lp_mathlib_PUnit_instLinearOrder = (const lean_object*)&lp_mathlib_PUnit_instLinearOrder___closed__7_value;
static const lean_ctor_object lp_mathlib_Prop_partialOrder___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Prop_partialOrder___closed__0 = (const lean_object*)&lp_mathlib_Prop_partialOrder___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Prop_partialOrder = (const lean_object*)&lp_mathlib_Prop_partialOrder___closed__0_value;
static lean_object* _init_lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__6(void){
_start:
{
lean_object* v___x_35_; lean_object* v___x_36_; 
v___x_35_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__5));
v___x_36_ = l_String_toRawSubstring_x27(v___x_35_);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1(lean_object* v_x_51_, lean_object* v_a_52_, lean_object* v_a_53_){
_start:
{
lean_object* v___x_54_; uint8_t v___x_55_; 
v___x_54_ = ((lean_object*)(lp_mathlib_term___u207b_xb9_x27o___00__closed__1));
lean_inc(v_x_51_);
v___x_55_ = l_Lean_Syntax_isOfKind(v_x_51_, v___x_54_);
if (v___x_55_ == 0)
{
lean_object* v___x_56_; lean_object* v___x_57_; 
lean_dec(v_x_51_);
v___x_56_ = lean_box(1);
v___x_57_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_57_, 0, v___x_56_);
lean_ctor_set(v___x_57_, 1, v_a_53_);
return v___x_57_;
}
else
{
lean_object* v_quotContext_58_; lean_object* v_currMacroScope_59_; lean_object* v_ref_60_; lean_object* v___x_61_; lean_object* v___x_62_; lean_object* v___x_63_; lean_object* v___x_64_; uint8_t v___x_65_; lean_object* v___x_66_; lean_object* v___x_67_; lean_object* v___x_68_; lean_object* v___x_69_; lean_object* v___x_70_; lean_object* v___x_71_; lean_object* v___x_72_; lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v___x_75_; lean_object* v___x_76_; 
v_quotContext_58_ = lean_ctor_get(v_a_52_, 1);
v_currMacroScope_59_ = lean_ctor_get(v_a_52_, 2);
v_ref_60_ = lean_ctor_get(v_a_52_, 5);
v___x_61_ = lean_unsigned_to_nat(0u);
v___x_62_ = l_Lean_Syntax_getArg(v_x_51_, v___x_61_);
v___x_63_ = lean_unsigned_to_nat(2u);
v___x_64_ = l_Lean_Syntax_getArg(v_x_51_, v___x_63_);
lean_dec(v_x_51_);
v___x_65_ = 0;
v___x_66_ = l_Lean_SourceInfo_fromRef(v_ref_60_, v___x_65_);
v___x_67_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__4));
v___x_68_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__6, &lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__6_once, _init_lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__6);
v___x_69_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__9));
lean_inc(v_currMacroScope_59_);
lean_inc(v_quotContext_58_);
v___x_70_ = l_Lean_addMacroScope(v_quotContext_58_, v___x_69_, v_currMacroScope_59_);
v___x_71_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__11));
lean_inc_n(v___x_66_, 2);
v___x_72_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_72_, 0, v___x_66_);
lean_ctor_set(v___x_72_, 1, v___x_68_);
lean_ctor_set(v___x_72_, 2, v___x_70_);
lean_ctor_set(v___x_72_, 3, v___x_71_);
v___x_73_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__13));
v___x_74_ = l_Lean_Syntax_node2(v___x_66_, v___x_73_, v___x_62_, v___x_64_);
v___x_75_ = l_Lean_Syntax_node2(v___x_66_, v___x_67_, v___x_72_, v___x_74_);
v___x_76_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_76_, 0, v___x_75_);
lean_ctor_set(v___x_76_, 1, v_a_53_);
return v___x_76_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___boxed(lean_object* v_x_77_, lean_object* v_a_78_, lean_object* v_a_79_){
_start:
{
lean_object* v_res_80_; 
v_res_80_ = lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1(v_x_77_, v_a_78_, v_a_79_);
lean_dec_ref(v_a_78_);
return v_res_80_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Basic______unexpand__Order__Preimage__1(lean_object* v_x_84_, lean_object* v_a_85_, lean_object* v_a_86_){
_start:
{
lean_object* v___x_87_; uint8_t v___x_88_; 
v___x_87_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__4));
lean_inc(v_x_84_);
v___x_88_ = l_Lean_Syntax_isOfKind(v_x_84_, v___x_87_);
if (v___x_88_ == 0)
{
lean_object* v___x_89_; lean_object* v___x_90_; 
lean_dec(v_x_84_);
v___x_89_ = lean_box(0);
v___x_90_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_90_, 0, v___x_89_);
lean_ctor_set(v___x_90_, 1, v_a_86_);
return v___x_90_;
}
else
{
lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; uint8_t v___x_94_; 
v___x_91_ = lean_unsigned_to_nat(0u);
v___x_92_ = l_Lean_Syntax_getArg(v_x_84_, v___x_91_);
v___x_93_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Basic______unexpand__Order__Preimage__1___closed__1));
lean_inc(v___x_92_);
v___x_94_ = l_Lean_Syntax_isOfKind(v___x_92_, v___x_93_);
if (v___x_94_ == 0)
{
lean_object* v___x_95_; lean_object* v___x_96_; 
lean_dec(v___x_92_);
lean_dec(v_x_84_);
v___x_95_ = lean_box(0);
v___x_96_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_96_, 0, v___x_95_);
lean_ctor_set(v___x_96_, 1, v_a_86_);
return v___x_96_;
}
else
{
lean_object* v___x_97_; lean_object* v___x_98_; lean_object* v___x_99_; uint8_t v___x_100_; 
v___x_97_ = lean_unsigned_to_nat(1u);
v___x_98_ = l_Lean_Syntax_getArg(v_x_84_, v___x_97_);
lean_dec(v_x_84_);
v___x_99_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_98_);
v___x_100_ = l_Lean_Syntax_matchesNull(v___x_98_, v___x_99_);
if (v___x_100_ == 0)
{
lean_object* v___x_101_; lean_object* v___x_102_; 
lean_dec(v___x_98_);
lean_dec(v___x_92_);
v___x_101_ = lean_box(0);
v___x_102_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_102_, 0, v___x_101_);
lean_ctor_set(v___x_102_, 1, v_a_86_);
return v___x_102_;
}
else
{
lean_object* v___x_103_; lean_object* v___x_104_; lean_object* v_ref_105_; uint8_t v___x_106_; lean_object* v___x_107_; lean_object* v___x_108_; lean_object* v___x_109_; lean_object* v___x_110_; lean_object* v___x_111_; lean_object* v___x_112_; 
v___x_103_ = l_Lean_Syntax_getArg(v___x_98_, v___x_91_);
v___x_104_ = l_Lean_Syntax_getArg(v___x_98_, v___x_97_);
lean_dec(v___x_98_);
v_ref_105_ = l_Lean_replaceRef(v___x_92_, v_a_85_);
lean_dec(v___x_92_);
v___x_106_ = 0;
v___x_107_ = l_Lean_SourceInfo_fromRef(v_ref_105_, v___x_106_);
lean_dec(v_ref_105_);
v___x_108_ = ((lean_object*)(lp_mathlib_term___u207b_xb9_x27o___00__closed__1));
v___x_109_ = ((lean_object*)(lp_mathlib_term___u207b_xb9_x27o___00__closed__4));
lean_inc(v___x_107_);
v___x_110_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_110_, 0, v___x_107_);
lean_ctor_set(v___x_110_, 1, v___x_109_);
v___x_111_ = l_Lean_Syntax_node3(v___x_107_, v___x_108_, v___x_103_, v___x_110_, v___x_104_);
v___x_112_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_112_, 0, v___x_111_);
lean_ctor_set(v___x_112_, 1, v_a_86_);
return v___x_112_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Basic______unexpand__Order__Preimage__1___boxed(lean_object* v_x_113_, lean_object* v_a_114_, lean_object* v_a_115_){
_start:
{
lean_object* v_res_116_; 
v_res_116_ = lp_mathlib___aux__Mathlib__Order__Basic______unexpand__Order__Preimage__1(v_x_113_, v_a_114_, v_a_115_);
lean_dec(v_a_114_);
return v_res_116_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Order_Preimage_decidable___redArg(lean_object* v_f_117_, lean_object* v_H_118_, lean_object* v_x_119_, lean_object* v_x_120_){
_start:
{
lean_object* v___x_121_; lean_object* v___x_122_; lean_object* v___x_123_; uint8_t v___x_124_; 
lean_inc(v_f_117_);
v___x_121_ = lean_apply_1(v_f_117_, v_x_119_);
v___x_122_ = lean_apply_1(v_f_117_, v_x_120_);
v___x_123_ = lean_apply_2(v_H_118_, v___x_121_, v___x_122_);
v___x_124_ = lean_unbox(v___x_123_);
return v___x_124_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Order_Preimage_decidable___redArg___boxed(lean_object* v_f_125_, lean_object* v_H_126_, lean_object* v_x_127_, lean_object* v_x_128_){
_start:
{
uint8_t v_res_129_; lean_object* v_r_130_; 
v_res_129_ = lp_mathlib_Order_Preimage_decidable___redArg(v_f_125_, v_H_126_, v_x_127_, v_x_128_);
v_r_130_ = lean_box(v_res_129_);
return v_r_130_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Order_Preimage_decidable(lean_object* v_00_u03b1_131_, lean_object* v_00_u03b2_132_, lean_object* v_f_133_, lean_object* v_s_134_, lean_object* v_H_135_, lean_object* v_x_136_, lean_object* v_x_137_){
_start:
{
uint8_t v___x_138_; 
v___x_138_ = lp_mathlib_Order_Preimage_decidable___redArg(v_f_133_, v_H_135_, v_x_136_, v_x_137_);
return v___x_138_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Order_Preimage_decidable___boxed(lean_object* v_00_u03b1_139_, lean_object* v_00_u03b2_140_, lean_object* v_f_141_, lean_object* v_s_142_, lean_object* v_H_143_, lean_object* v_x_144_, lean_object* v_x_145_){
_start:
{
uint8_t v_res_146_; lean_object* v_r_147_; 
v_res_146_ = lp_mathlib_Order_Preimage_decidable(v_00_u03b1_139_, v_00_u03b2_140_, v_f_141_, v_s_142_, v_H_143_, v_x_144_, v_x_145_);
v_r_147_ = lean_box(v_res_146_);
return v_r_147_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GCongr_exactLeOfLt___lam__0(lean_object* v___x_151_, lean_object* v_h_152_, lean_object* v_goal_153_, lean_object* v___y_154_, lean_object* v___y_155_, lean_object* v___y_156_, lean_object* v___y_157_){
_start:
{
lean_object* v___x_159_; 
v___x_159_ = l_Lean_Meta_mkFreshLevelMVar(v___y_154_, v___y_155_, v___y_156_, v___y_157_);
if (lean_obj_tag(v___x_159_) == 0)
{
lean_object* v_a_160_; lean_object* v___x_161_; lean_object* v___x_162_; lean_object* v___x_163_; lean_object* v___x_164_; lean_object* v___x_165_; 
v_a_160_ = lean_ctor_get(v___x_159_, 0);
lean_inc(v_a_160_);
lean_dec_ref_known(v___x_159_, 1);
v___x_161_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_GCongr_exactLeOfLt___lam__0___closed__1));
v___x_162_ = lean_box(0);
v___x_163_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_163_, 0, v_a_160_);
lean_ctor_set(v___x_163_, 1, v___x_162_);
v___x_164_ = l_Lean_Expr_const___override(v___x_161_, v___x_163_);
lean_inc(v___y_157_);
lean_inc_ref(v___y_156_);
lean_inc(v___y_155_);
lean_inc_ref(v___y_154_);
lean_inc_ref(v___x_164_);
v___x_165_ = lean_infer_type(v___x_164_, v___y_154_, v___y_155_, v___y_156_, v___y_157_);
if (lean_obj_tag(v___x_165_) == 0)
{
lean_object* v_a_166_; uint8_t v___x_167_; lean_object* v___x_168_; 
v_a_166_ = lean_ctor_get(v___x_165_, 0);
lean_inc(v_a_166_);
lean_dec_ref_known(v___x_165_, 1);
v___x_167_ = 0;
v___x_168_ = l_Lean_Meta_forallMetaTelescope(v_a_166_, v___x_167_, v___y_154_, v___y_155_, v___y_156_, v___y_157_);
if (lean_obj_tag(v___x_168_) == 0)
{
lean_object* v_a_169_; lean_object* v_fst_170_; lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; lean_object* v___x_174_; 
v_a_169_ = lean_ctor_get(v___x_168_, 0);
lean_inc(v_a_169_);
lean_dec_ref_known(v___x_168_, 1);
v_fst_170_ = lean_ctor_get(v_a_169_, 0);
lean_inc(v_fst_170_);
lean_dec(v_a_169_);
v___x_171_ = lean_unsigned_to_nat(4u);
v___x_172_ = lean_array_get_borrowed(v___x_151_, v_fst_170_, v___x_171_);
v___x_173_ = l_Lean_Expr_mvarId_x21(v___x_172_);
v___x_174_ = lp_batteries_Lean_MVarId_assignIfDefEq(v___x_173_, v_h_152_, v___y_154_, v___y_155_, v___y_156_, v___y_157_);
if (lean_obj_tag(v___x_174_) == 0)
{
lean_object* v___x_175_; lean_object* v___x_176_; 
lean_dec_ref_known(v___x_174_, 1);
v___x_175_ = l_Lean_mkAppN(v___x_164_, v_fst_170_);
lean_dec(v_fst_170_);
v___x_176_ = lp_batteries_Lean_MVarId_assignIfDefEq(v_goal_153_, v___x_175_, v___y_154_, v___y_155_, v___y_156_, v___y_157_);
return v___x_176_;
}
else
{
lean_dec(v_fst_170_);
lean_dec_ref(v___x_164_);
lean_dec(v_goal_153_);
return v___x_174_;
}
}
else
{
lean_object* v_a_177_; lean_object* v___x_179_; uint8_t v_isShared_180_; uint8_t v_isSharedCheck_184_; 
lean_dec_ref(v___x_164_);
lean_dec(v_goal_153_);
lean_dec_ref(v_h_152_);
v_a_177_ = lean_ctor_get(v___x_168_, 0);
v_isSharedCheck_184_ = !lean_is_exclusive(v___x_168_);
if (v_isSharedCheck_184_ == 0)
{
v___x_179_ = v___x_168_;
v_isShared_180_ = v_isSharedCheck_184_;
goto v_resetjp_178_;
}
else
{
lean_inc(v_a_177_);
lean_dec(v___x_168_);
v___x_179_ = lean_box(0);
v_isShared_180_ = v_isSharedCheck_184_;
goto v_resetjp_178_;
}
v_resetjp_178_:
{
lean_object* v___x_182_; 
if (v_isShared_180_ == 0)
{
v___x_182_ = v___x_179_;
goto v_reusejp_181_;
}
else
{
lean_object* v_reuseFailAlloc_183_; 
v_reuseFailAlloc_183_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_183_, 0, v_a_177_);
v___x_182_ = v_reuseFailAlloc_183_;
goto v_reusejp_181_;
}
v_reusejp_181_:
{
return v___x_182_;
}
}
}
}
else
{
lean_object* v_a_185_; lean_object* v___x_187_; uint8_t v_isShared_188_; uint8_t v_isSharedCheck_192_; 
lean_dec_ref(v___x_164_);
lean_dec(v_goal_153_);
lean_dec_ref(v_h_152_);
v_a_185_ = lean_ctor_get(v___x_165_, 0);
v_isSharedCheck_192_ = !lean_is_exclusive(v___x_165_);
if (v_isSharedCheck_192_ == 0)
{
v___x_187_ = v___x_165_;
v_isShared_188_ = v_isSharedCheck_192_;
goto v_resetjp_186_;
}
else
{
lean_inc(v_a_185_);
lean_dec(v___x_165_);
v___x_187_ = lean_box(0);
v_isShared_188_ = v_isSharedCheck_192_;
goto v_resetjp_186_;
}
v_resetjp_186_:
{
lean_object* v___x_190_; 
if (v_isShared_188_ == 0)
{
v___x_190_ = v___x_187_;
goto v_reusejp_189_;
}
else
{
lean_object* v_reuseFailAlloc_191_; 
v_reuseFailAlloc_191_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_191_, 0, v_a_185_);
v___x_190_ = v_reuseFailAlloc_191_;
goto v_reusejp_189_;
}
v_reusejp_189_:
{
return v___x_190_;
}
}
}
}
else
{
lean_object* v_a_193_; lean_object* v___x_195_; uint8_t v_isShared_196_; uint8_t v_isSharedCheck_200_; 
lean_dec(v_goal_153_);
lean_dec_ref(v_h_152_);
v_a_193_ = lean_ctor_get(v___x_159_, 0);
v_isSharedCheck_200_ = !lean_is_exclusive(v___x_159_);
if (v_isSharedCheck_200_ == 0)
{
v___x_195_ = v___x_159_;
v_isShared_196_ = v_isSharedCheck_200_;
goto v_resetjp_194_;
}
else
{
lean_inc(v_a_193_);
lean_dec(v___x_159_);
v___x_195_ = lean_box(0);
v_isShared_196_ = v_isSharedCheck_200_;
goto v_resetjp_194_;
}
v_resetjp_194_:
{
lean_object* v___x_198_; 
if (v_isShared_196_ == 0)
{
v___x_198_ = v___x_195_;
goto v_reusejp_197_;
}
else
{
lean_object* v_reuseFailAlloc_199_; 
v_reuseFailAlloc_199_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_199_, 0, v_a_193_);
v___x_198_ = v_reuseFailAlloc_199_;
goto v_reusejp_197_;
}
v_reusejp_197_:
{
return v___x_198_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GCongr_exactLeOfLt___lam__0___boxed(lean_object* v___x_201_, lean_object* v_h_202_, lean_object* v_goal_203_, lean_object* v___y_204_, lean_object* v___y_205_, lean_object* v___y_206_, lean_object* v___y_207_, lean_object* v___y_208_){
_start:
{
lean_object* v_res_209_; 
v_res_209_ = lp_mathlib_Mathlib_Tactic_GCongr_exactLeOfLt___lam__0(v___x_201_, v_h_202_, v_goal_203_, v___y_204_, v___y_205_, v___y_206_, v___y_207_);
lean_dec(v___y_207_);
lean_dec_ref(v___y_206_);
lean_dec(v___y_205_);
lean_dec_ref(v___y_204_);
lean_dec_ref(v___x_201_);
return v_res_209_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_GCongr_exactLeOfLt___closed__0(void){
_start:
{
lean_object* v___x_210_; lean_object* v___f_211_; 
v___x_210_ = l_Lean_instInhabitedExpr;
v___f_211_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_GCongr_exactLeOfLt___lam__0___boxed), 8, 1);
lean_closure_set(v___f_211_, 0, v___x_210_);
return v___f_211_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_GCongr_exactLeOfLt(void){
_start:
{
lean_object* v___f_212_; 
v___f_212_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_GCongr_exactLeOfLt___closed__0, &lp_mathlib_Mathlib_Tactic_GCongr_exactLeOfLt___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_GCongr_exactLeOfLt___closed__0);
return v___f_212_;
}
}
static lean_object* _init_lp_mathlib_Prop_instCompl(void){
_start:
{
lean_object* v___x_213_; 
v___x_213_ = lean_box(0);
return v___x_213_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompl___redArg___lam__0(lean_object* v_inst_214_, lean_object* v_x_215_, lean_object* v_i_216_){
_start:
{
lean_object* v___x_217_; lean_object* v___x_218_; 
lean_inc(v_i_216_);
v___x_217_ = lean_apply_1(v_x_215_, v_i_216_);
v___x_218_ = lean_apply_2(v_inst_214_, v_i_216_, v___x_217_);
return v___x_218_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompl___redArg(lean_object* v_inst_219_){
_start:
{
lean_object* v___f_220_; 
v___f_220_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instCompl___redArg___lam__0), 3, 1);
lean_closure_set(v___f_220_, 0, v_inst_219_);
return v___f_220_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompl(lean_object* v_00_u03b9_221_, lean_object* v_00_u03c0_222_, lean_object* v_inst_223_){
_start:
{
lean_object* v___f_224_; 
v___f_224_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instCompl___redArg___lam__0), 3, 1);
lean_closure_set(v___f_224_, 0, v_inst_223_);
return v___f_224_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instHNot___redArg(lean_object* v_inst_225_){
_start:
{
lean_object* v___f_226_; 
v___f_226_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instCompl___redArg___lam__0), 3, 1);
lean_closure_set(v___f_226_, 0, v_inst_225_);
return v___f_226_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instHNot(lean_object* v_00_u03b9_227_, lean_object* v_00_u03c0_228_, lean_object* v_inst_229_){
_start:
{
lean_object* v___f_230_; 
v___f_230_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instCompl___redArg___lam__0), 3, 1);
lean_closure_set(v___f_230_, 0, v_inst_229_);
return v___f_230_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_preorder(lean_object* v_00_u03b9_234_, lean_object* v_00_u03c0_235_, lean_object* v_inst_236_){
_start:
{
lean_object* v___x_237_; 
v___x_237_ = ((lean_object*)(lp_mathlib_Pi_preorder___closed__0));
return v___x_237_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_preorder___boxed(lean_object* v_00_u03b9_238_, lean_object* v_00_u03c0_239_, lean_object* v_inst_240_){
_start:
{
lean_object* v_res_241_; 
v_res_241_ = lp_mathlib_Pi_preorder(v_00_u03b9_238_, v_00_u03c0_239_, v_inst_240_);
lean_dec_ref(v_inst_240_);
return v_res_241_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_partialOrder___redArg___lam__0(lean_object* v_inst_242_, lean_object* v_i_243_){
_start:
{
lean_object* v___x_244_; 
v___x_244_ = lean_apply_1(v_inst_242_, v_i_243_);
return v___x_244_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_partialOrder___redArg(lean_object* v_inst_245_){
_start:
{
lean_object* v___f_246_; lean_object* v___x_247_; 
v___f_246_ = lean_alloc_closure((void*)(lp_mathlib_Pi_partialOrder___redArg___lam__0), 2, 1);
lean_closure_set(v___f_246_, 0, v_inst_245_);
v___x_247_ = lp_mathlib_Pi_preorder(lean_box(0), lean_box(0), v___f_246_);
lean_dec_ref(v___f_246_);
return v___x_247_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_partialOrder(lean_object* v_00_u03b9_248_, lean_object* v_00_u03c0_249_, lean_object* v_inst_250_){
_start:
{
lean_object* v___x_251_; 
v___x_251_ = lp_mathlib_Pi_partialOrder___redArg(v_inst_250_);
return v___x_251_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Order_Basic_0____aux__Mathlib__Order__Basic______macroRules____private__Mathlib__Order__Basic__0__term___u227a____1___closed__1(void){
_start:
{
lean_object* v___x_290_; lean_object* v___x_291_; 
v___x_290_ = ((lean_object*)(lp_mathlib___private_Mathlib_Order_Basic_0____aux__Mathlib__Order__Basic______macroRules____private__Mathlib__Order__Basic__0__term___u227a____1___closed__0));
v___x_291_ = l_String_toRawSubstring_x27(v___x_290_);
return v___x_291_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Basic_0____aux__Mathlib__Order__Basic______macroRules____private__Mathlib__Order__Basic__0__term___u227a____1(lean_object* v_x_300_, lean_object* v_a_301_, lean_object* v_a_302_){
_start:
{
lean_object* v___x_303_; lean_object* v___x_304_; uint8_t v___x_305_; 
v___x_303_ = lean_unsigned_to_nat(0u);
v___x_304_ = ((lean_object*)(lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__9));
lean_inc(v_x_300_);
v___x_305_ = l_Lean_Syntax_isOfKind(v_x_300_, v___x_304_);
if (v___x_305_ == 0)
{
lean_object* v___x_306_; lean_object* v___x_307_; 
lean_dec(v_x_300_);
v___x_306_ = lean_box(1);
v___x_307_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_307_, 0, v___x_306_);
lean_ctor_set(v___x_307_, 1, v_a_302_);
return v___x_307_;
}
else
{
lean_object* v_quotContext_308_; lean_object* v_currMacroScope_309_; lean_object* v_ref_310_; lean_object* v___x_311_; lean_object* v___x_312_; lean_object* v___x_313_; uint8_t v___x_314_; lean_object* v___x_315_; lean_object* v___x_316_; lean_object* v___x_317_; lean_object* v___x_318_; lean_object* v___x_319_; lean_object* v___x_320_; lean_object* v___x_321_; lean_object* v___x_322_; lean_object* v___x_323_; lean_object* v___x_324_; lean_object* v___x_325_; 
v_quotContext_308_ = lean_ctor_get(v_a_301_, 1);
v_currMacroScope_309_ = lean_ctor_get(v_a_301_, 2);
v_ref_310_ = lean_ctor_get(v_a_301_, 5);
v___x_311_ = l_Lean_Syntax_getArg(v_x_300_, v___x_303_);
v___x_312_ = lean_unsigned_to_nat(2u);
v___x_313_ = l_Lean_Syntax_getArg(v_x_300_, v___x_312_);
lean_dec(v_x_300_);
v___x_314_ = 0;
v___x_315_ = l_Lean_SourceInfo_fromRef(v_ref_310_, v___x_314_);
v___x_316_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__4));
v___x_317_ = lean_obj_once(&lp_mathlib___private_Mathlib_Order_Basic_0____aux__Mathlib__Order__Basic______macroRules____private__Mathlib__Order__Basic__0__term___u227a____1___closed__1, &lp_mathlib___private_Mathlib_Order_Basic_0____aux__Mathlib__Order__Basic______macroRules____private__Mathlib__Order__Basic__0__term___u227a____1___closed__1_once, _init_lp_mathlib___private_Mathlib_Order_Basic_0____aux__Mathlib__Order__Basic______macroRules____private__Mathlib__Order__Basic__0__term___u227a____1___closed__1);
v___x_318_ = ((lean_object*)(lp_mathlib___private_Mathlib_Order_Basic_0____aux__Mathlib__Order__Basic______macroRules____private__Mathlib__Order__Basic__0__term___u227a____1___closed__2));
lean_inc(v_currMacroScope_309_);
lean_inc(v_quotContext_308_);
v___x_319_ = l_Lean_addMacroScope(v_quotContext_308_, v___x_318_, v_currMacroScope_309_);
v___x_320_ = ((lean_object*)(lp_mathlib___private_Mathlib_Order_Basic_0____aux__Mathlib__Order__Basic______macroRules____private__Mathlib__Order__Basic__0__term___u227a____1___closed__4));
lean_inc_n(v___x_315_, 2);
v___x_321_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_321_, 0, v___x_315_);
lean_ctor_set(v___x_321_, 1, v___x_317_);
lean_ctor_set(v___x_321_, 2, v___x_319_);
lean_ctor_set(v___x_321_, 3, v___x_320_);
v___x_322_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__13));
v___x_323_ = l_Lean_Syntax_node2(v___x_315_, v___x_322_, v___x_311_, v___x_313_);
v___x_324_ = l_Lean_Syntax_node2(v___x_315_, v___x_316_, v___x_321_, v___x_323_);
v___x_325_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_325_, 0, v___x_324_);
lean_ctor_set(v___x_325_, 1, v_a_302_);
return v___x_325_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Basic_0____aux__Mathlib__Order__Basic______macroRules____private__Mathlib__Order__Basic__0__term___u227a____1___boxed(lean_object* v_x_326_, lean_object* v_a_327_, lean_object* v_a_328_){
_start:
{
lean_object* v_res_329_; 
v_res_329_ = lp_mathlib___private_Mathlib_Order_Basic_0____aux__Mathlib__Order__Basic______macroRules____private__Mathlib__Order__Basic__0__term___u227a____1(v_x_326_, v_a_327_, v_a_328_);
lean_dec_ref(v_a_327_);
return v_res_329_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Basic_0____aux__Mathlib__Order__Basic______unexpand__StrongLT__1(lean_object* v_x_330_, lean_object* v_a_331_, lean_object* v_a_332_){
_start:
{
lean_object* v___x_333_; uint8_t v___x_334_; 
v___x_333_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Basic______macroRules__term___u207b_xb9_x27o____1___closed__4));
lean_inc(v_x_330_);
v___x_334_ = l_Lean_Syntax_isOfKind(v_x_330_, v___x_333_);
if (v___x_334_ == 0)
{
lean_object* v___x_335_; lean_object* v___x_336_; 
lean_dec(v_x_330_);
v___x_335_ = lean_box(0);
v___x_336_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_336_, 0, v___x_335_);
lean_ctor_set(v___x_336_, 1, v_a_332_);
return v___x_336_;
}
else
{
lean_object* v___x_337_; lean_object* v___x_338_; lean_object* v___x_339_; uint8_t v___x_340_; 
v___x_337_ = lean_unsigned_to_nat(0u);
v___x_338_ = l_Lean_Syntax_getArg(v_x_330_, v___x_337_);
v___x_339_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Basic______unexpand__Order__Preimage__1___closed__1));
lean_inc(v___x_338_);
v___x_340_ = l_Lean_Syntax_isOfKind(v___x_338_, v___x_339_);
if (v___x_340_ == 0)
{
lean_object* v___x_341_; lean_object* v___x_342_; 
lean_dec(v___x_338_);
lean_dec(v_x_330_);
v___x_341_ = lean_box(0);
v___x_342_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_342_, 0, v___x_341_);
lean_ctor_set(v___x_342_, 1, v_a_332_);
return v___x_342_;
}
else
{
lean_object* v___x_343_; lean_object* v___x_344_; lean_object* v___x_345_; uint8_t v___x_346_; 
v___x_343_ = lean_unsigned_to_nat(1u);
v___x_344_ = l_Lean_Syntax_getArg(v_x_330_, v___x_343_);
lean_dec(v_x_330_);
v___x_345_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_344_);
v___x_346_ = l_Lean_Syntax_matchesNull(v___x_344_, v___x_345_);
if (v___x_346_ == 0)
{
lean_object* v___x_347_; lean_object* v___x_348_; 
lean_dec(v___x_344_);
lean_dec(v___x_338_);
v___x_347_ = lean_box(0);
v___x_348_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_348_, 0, v___x_347_);
lean_ctor_set(v___x_348_, 1, v_a_332_);
return v___x_348_;
}
else
{
lean_object* v___x_349_; lean_object* v___x_350_; lean_object* v_ref_351_; uint8_t v___x_352_; lean_object* v___x_353_; lean_object* v___x_354_; lean_object* v___x_355_; lean_object* v___x_356_; lean_object* v___x_357_; lean_object* v___x_358_; 
v___x_349_ = l_Lean_Syntax_getArg(v___x_344_, v___x_337_);
v___x_350_ = l_Lean_Syntax_getArg(v___x_344_, v___x_343_);
lean_dec(v___x_344_);
v_ref_351_ = l_Lean_replaceRef(v___x_338_, v_a_331_);
lean_dec(v___x_338_);
v___x_352_ = 0;
v___x_353_ = l_Lean_SourceInfo_fromRef(v_ref_351_, v___x_352_);
lean_dec(v_ref_351_);
v___x_354_ = ((lean_object*)(lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__9));
v___x_355_ = ((lean_object*)(lp_mathlib___private_Mathlib_Order_Basic_0__term___u227a___00__closed__10));
lean_inc(v___x_353_);
v___x_356_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_356_, 0, v___x_353_);
lean_ctor_set(v___x_356_, 1, v___x_355_);
v___x_357_ = l_Lean_Syntax_node3(v___x_353_, v___x_354_, v___x_349_, v___x_356_, v___x_350_);
v___x_358_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_358_, 0, v___x_357_);
lean_ctor_set(v___x_358_, 1, v_a_332_);
return v___x_358_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Basic_0____aux__Mathlib__Order__Basic______unexpand__StrongLT__1___boxed(lean_object* v_x_359_, lean_object* v_a_360_, lean_object* v_a_361_){
_start:
{
lean_object* v_res_362_; 
v_res_362_ = lp_mathlib___private_Mathlib_Order_Basic_0____aux__Mathlib__Order__Basic______unexpand__StrongLT__1(v_x_359_, v_a_360_, v_a_361_);
lean_dec(v_a_360_);
return v_res_362_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instSDiff___redArg___lam__0(lean_object* v_inst_363_, lean_object* v_x_364_, lean_object* v_y_365_, lean_object* v_i_366_){
_start:
{
lean_object* v___x_367_; lean_object* v___x_368_; lean_object* v___x_369_; 
lean_inc_n(v_i_366_, 2);
v___x_367_ = lean_apply_1(v_x_364_, v_i_366_);
v___x_368_ = lean_apply_1(v_y_365_, v_i_366_);
v___x_369_ = lean_apply_3(v_inst_363_, v_i_366_, v___x_367_, v___x_368_);
return v___x_369_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instSDiff___redArg(lean_object* v_inst_370_){
_start:
{
lean_object* v___f_371_; 
v___f_371_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instSDiff___redArg___lam__0), 4, 1);
lean_closure_set(v___f_371_, 0, v_inst_370_);
return v___f_371_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instSDiff(lean_object* v_00_u03b9_372_, lean_object* v_00_u03c0_373_, lean_object* v_inst_374_){
_start:
{
lean_object* v___f_375_; 
v___f_375_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instSDiff___redArg___lam__0), 4, 1);
lean_closure_set(v___f_375_, 0, v_inst_374_);
return v___f_375_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instHImp___redArg___lam__0(lean_object* v_inst_376_, lean_object* v_y_377_, lean_object* v_x_378_, lean_object* v_i_379_){
_start:
{
lean_object* v___x_380_; lean_object* v___x_381_; lean_object* v___x_382_; 
lean_inc_n(v_i_379_, 2);
v___x_380_ = lean_apply_1(v_y_377_, v_i_379_);
v___x_381_ = lean_apply_1(v_x_378_, v_i_379_);
v___x_382_ = lean_apply_3(v_inst_376_, v_i_379_, v___x_380_, v___x_381_);
return v___x_382_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instHImp___redArg(lean_object* v_inst_383_){
_start:
{
lean_object* v___f_384_; 
v___f_384_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instHImp___redArg___lam__0), 4, 1);
lean_closure_set(v___f_384_, 0, v_inst_383_);
return v___f_384_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instHImp(lean_object* v_00_u03b9_385_, lean_object* v_00_u03c0_386_, lean_object* v_inst_387_){
_start:
{
lean_object* v___f_388_; 
v___f_388_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instHImp___redArg___lam__0), 4, 1);
lean_closure_set(v___f_388_, 0, v_inst_387_);
return v___f_388_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_preorder___redArg(lean_object* v_inst_389_, lean_object* v_inst_390_){
_start:
{
lean_object* v___x_391_; 
v___x_391_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_391_, 0, v_inst_389_);
lean_ctor_set(v___x_391_, 1, v_inst_390_);
return v___x_391_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_preorder(lean_object* v_00_u03b1_392_, lean_object* v_00_u03b2_393_, lean_object* v_inst_394_, lean_object* v_inst_395_, lean_object* v_inst_396_, lean_object* v_f_397_, lean_object* v_le_398_, lean_object* v_lt_399_){
_start:
{
lean_object* v___x_400_; 
v___x_400_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_400_, 0, v_inst_395_);
lean_ctor_set(v___x_400_, 1, v_inst_396_);
return v___x_400_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_preorder___boxed(lean_object* v_00_u03b1_401_, lean_object* v_00_u03b2_402_, lean_object* v_inst_403_, lean_object* v_inst_404_, lean_object* v_inst_405_, lean_object* v_f_406_, lean_object* v_le_407_, lean_object* v_lt_408_){
_start:
{
lean_object* v_res_409_; 
v_res_409_ = lp_mathlib_Function_Injective_preorder(v_00_u03b1_401_, v_00_u03b2_402_, v_inst_403_, v_inst_404_, v_inst_405_, v_f_406_, v_le_407_, v_lt_408_);
lean_dec(v_f_406_);
lean_dec_ref(v_inst_403_);
return v_res_409_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_partialOrder___redArg(lean_object* v_inst_410_, lean_object* v_inst_411_){
_start:
{
lean_object* v___x_412_; 
v___x_412_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_412_, 0, v_inst_410_);
lean_ctor_set(v___x_412_, 1, v_inst_411_);
return v___x_412_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_partialOrder(lean_object* v_00_u03b1_413_, lean_object* v_00_u03b2_414_, lean_object* v_inst_415_, lean_object* v_inst_416_, lean_object* v_inst_417_, lean_object* v_f_418_, lean_object* v_hf_419_, lean_object* v_le_420_, lean_object* v_lt_421_){
_start:
{
lean_object* v___x_422_; 
v___x_422_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_422_, 0, v_inst_416_);
lean_ctor_set(v___x_422_, 1, v_inst_417_);
return v___x_422_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_partialOrder___boxed(lean_object* v_00_u03b1_423_, lean_object* v_00_u03b2_424_, lean_object* v_inst_425_, lean_object* v_inst_426_, lean_object* v_inst_427_, lean_object* v_f_428_, lean_object* v_hf_429_, lean_object* v_le_430_, lean_object* v_lt_431_){
_start:
{
lean_object* v_res_432_; 
v_res_432_ = lp_mathlib_Function_Injective_partialOrder(v_00_u03b1_423_, v_00_u03b2_424_, v_inst_425_, v_inst_426_, v_inst_427_, v_f_428_, v_hf_429_, v_le_430_, v_lt_431_);
lean_dec(v_f_428_);
lean_dec_ref(v_inst_425_);
return v_res_432_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_linearOrder___redArg(lean_object* v_inst_433_, lean_object* v_inst_434_, lean_object* v_inst_435_, lean_object* v_inst_436_, lean_object* v_inst_437_, lean_object* v_inst_438_, lean_object* v_inst_439_, lean_object* v_inst_440_){
_start:
{
lean_object* v___x_441_; lean_object* v___x_442_; 
v___x_441_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_441_, 0, v_inst_433_);
lean_ctor_set(v___x_441_, 1, v_inst_434_);
v___x_442_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v___x_442_, 0, v___x_441_);
lean_ctor_set(v___x_442_, 1, v_inst_436_);
lean_ctor_set(v___x_442_, 2, v_inst_435_);
lean_ctor_set(v___x_442_, 3, v_inst_437_);
lean_ctor_set(v___x_442_, 4, v_inst_439_);
lean_ctor_set(v___x_442_, 5, v_inst_438_);
lean_ctor_set(v___x_442_, 6, v_inst_440_);
return v___x_442_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_linearOrder(lean_object* v_00_u03b1_443_, lean_object* v_00_u03b2_444_, lean_object* v_inst_445_, lean_object* v_inst_446_, lean_object* v_inst_447_, lean_object* v_inst_448_, lean_object* v_inst_449_, lean_object* v_inst_450_, lean_object* v_inst_451_, lean_object* v_inst_452_, lean_object* v_inst_453_, lean_object* v_f_454_, lean_object* v_hf_455_, lean_object* v_le_456_, lean_object* v_lt_457_, lean_object* v_min_458_, lean_object* v_max_459_, lean_object* v_compare_460_){
_start:
{
lean_object* v___x_461_; lean_object* v___x_462_; 
v___x_461_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_461_, 0, v_inst_446_);
lean_ctor_set(v___x_461_, 1, v_inst_447_);
v___x_462_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v___x_462_, 0, v___x_461_);
lean_ctor_set(v___x_462_, 1, v_inst_449_);
lean_ctor_set(v___x_462_, 2, v_inst_448_);
lean_ctor_set(v___x_462_, 3, v_inst_450_);
lean_ctor_set(v___x_462_, 4, v_inst_452_);
lean_ctor_set(v___x_462_, 5, v_inst_451_);
lean_ctor_set(v___x_462_, 6, v_inst_453_);
return v___x_462_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_linearOrder___boxed(lean_object** _args){
lean_object* v_00_u03b1_463_ = _args[0];
lean_object* v_00_u03b2_464_ = _args[1];
lean_object* v_inst_465_ = _args[2];
lean_object* v_inst_466_ = _args[3];
lean_object* v_inst_467_ = _args[4];
lean_object* v_inst_468_ = _args[5];
lean_object* v_inst_469_ = _args[6];
lean_object* v_inst_470_ = _args[7];
lean_object* v_inst_471_ = _args[8];
lean_object* v_inst_472_ = _args[9];
lean_object* v_inst_473_ = _args[10];
lean_object* v_f_474_ = _args[11];
lean_object* v_hf_475_ = _args[12];
lean_object* v_le_476_ = _args[13];
lean_object* v_lt_477_ = _args[14];
lean_object* v_min_478_ = _args[15];
lean_object* v_max_479_ = _args[16];
lean_object* v_compare_480_ = _args[17];
_start:
{
lean_object* v_res_481_; 
v_res_481_ = lp_mathlib_Function_Injective_linearOrder(v_00_u03b1_463_, v_00_u03b2_464_, v_inst_465_, v_inst_466_, v_inst_467_, v_inst_468_, v_inst_469_, v_inst_470_, v_inst_471_, v_inst_472_, v_inst_473_, v_f_474_, v_hf_475_, v_le_476_, v_lt_477_, v_min_478_, v_max_479_, v_compare_480_);
lean_dec(v_f_474_);
lean_dec_ref(v_inst_465_);
return v_res_481_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Preorder_lift(lean_object* v_00_u03b1_485_, lean_object* v_00_u03b2_486_, lean_object* v_inst_487_, lean_object* v_f_488_){
_start:
{
lean_object* v___x_489_; 
v___x_489_ = ((lean_object*)(lp_mathlib_Preorder_lift___closed__0));
return v___x_489_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Preorder_lift___boxed(lean_object* v_00_u03b1_490_, lean_object* v_00_u03b2_491_, lean_object* v_inst_492_, lean_object* v_f_493_){
_start:
{
lean_object* v_res_494_; 
v_res_494_ = lp_mathlib_Preorder_lift(v_00_u03b1_490_, v_00_u03b2_491_, v_inst_492_, v_f_493_);
lean_dec(v_f_493_);
lean_dec_ref(v_inst_492_);
return v_res_494_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PartialOrder_lift(lean_object* v_00_u03b1_495_, lean_object* v_00_u03b2_496_, lean_object* v_inst_497_, lean_object* v_f_498_, lean_object* v_inj_499_){
_start:
{
lean_object* v___x_500_; 
v___x_500_ = ((lean_object*)(lp_mathlib_Preorder_lift___closed__0));
return v___x_500_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PartialOrder_lift___boxed(lean_object* v_00_u03b1_501_, lean_object* v_00_u03b2_502_, lean_object* v_inst_503_, lean_object* v_f_504_, lean_object* v_inj_505_){
_start:
{
lean_object* v_res_506_; 
v_res_506_ = lp_mathlib_PartialOrder_lift(v_00_u03b1_501_, v_00_u03b2_502_, v_inst_503_, v_f_504_, v_inj_505_);
lean_dec(v_f_504_);
lean_dec_ref(v_inst_503_);
return v_res_506_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_LinearOrder_lift___redArg___lam__0(lean_object* v_f_507_, lean_object* v_toDecidableEq_508_, lean_object* v_x_509_, lean_object* v_y_510_){
_start:
{
lean_object* v___x_511_; lean_object* v___x_512_; lean_object* v___x_513_; uint8_t v___x_514_; 
lean_inc(v_f_507_);
v___x_511_ = lean_apply_1(v_f_507_, v_x_509_);
v___x_512_ = lean_apply_1(v_f_507_, v_y_510_);
v___x_513_ = lean_apply_2(v_toDecidableEq_508_, v___x_511_, v___x_512_);
v___x_514_ = lean_unbox(v___x_513_);
return v___x_514_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_lift___redArg___lam__0___boxed(lean_object* v_f_515_, lean_object* v_toDecidableEq_516_, lean_object* v_x_517_, lean_object* v_y_518_){
_start:
{
uint8_t v_res_519_; lean_object* v_r_520_; 
v_res_519_ = lp_mathlib_LinearOrder_lift___redArg___lam__0(v_f_515_, v_toDecidableEq_516_, v_x_517_, v_y_518_);
v_r_520_ = lean_box(v_res_519_);
return v_r_520_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_LinearOrder_lift___redArg___lam__1(lean_object* v_f_521_, lean_object* v_toDecidableLE_522_, lean_object* v_x_523_, lean_object* v_y_524_){
_start:
{
lean_object* v___x_525_; lean_object* v___x_526_; lean_object* v___x_527_; uint8_t v___x_528_; 
lean_inc(v_f_521_);
v___x_525_ = lean_apply_1(v_f_521_, v_x_523_);
v___x_526_ = lean_apply_1(v_f_521_, v_y_524_);
v___x_527_ = lean_apply_2(v_toDecidableLE_522_, v___x_525_, v___x_526_);
v___x_528_ = lean_unbox(v___x_527_);
return v___x_528_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_lift___redArg___lam__1___boxed(lean_object* v_f_529_, lean_object* v_toDecidableLE_530_, lean_object* v_x_531_, lean_object* v_y_532_){
_start:
{
uint8_t v_res_533_; lean_object* v_r_534_; 
v_res_533_ = lp_mathlib_LinearOrder_lift___redArg___lam__1(v_f_529_, v_toDecidableLE_530_, v_x_531_, v_y_532_);
v_r_534_ = lean_box(v_res_533_);
return v_r_534_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_LinearOrder_lift___redArg___lam__2(lean_object* v_f_535_, lean_object* v_toDecidableLT_536_, lean_object* v_x_537_, lean_object* v_y_538_){
_start:
{
lean_object* v___x_539_; lean_object* v___x_540_; lean_object* v___x_541_; uint8_t v___x_542_; 
lean_inc(v_f_535_);
v___x_539_ = lean_apply_1(v_f_535_, v_x_537_);
v___x_540_ = lean_apply_1(v_f_535_, v_y_538_);
v___x_541_ = lean_apply_2(v_toDecidableLT_536_, v___x_539_, v___x_540_);
v___x_542_ = lean_unbox(v___x_541_);
return v___x_542_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_lift___redArg___lam__2___boxed(lean_object* v_f_543_, lean_object* v_toDecidableLT_544_, lean_object* v_x_545_, lean_object* v_y_546_){
_start:
{
uint8_t v_res_547_; lean_object* v_r_548_; 
v_res_547_ = lp_mathlib_LinearOrder_lift___redArg___lam__2(v_f_543_, v_toDecidableLT_544_, v_x_545_, v_y_546_);
v_r_548_ = lean_box(v_res_547_);
return v_r_548_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_LinearOrder_lift___redArg___lam__3(lean_object* v_f_549_, lean_object* v_toOrd_550_, lean_object* v_a_551_, lean_object* v_b_552_){
_start:
{
lean_object* v___x_553_; lean_object* v___x_554_; lean_object* v___x_555_; uint8_t v___x_556_; 
lean_inc(v_f_549_);
v___x_553_ = lean_apply_1(v_f_549_, v_a_551_);
v___x_554_ = lean_apply_1(v_f_549_, v_b_552_);
v___x_555_ = lean_apply_2(v_toOrd_550_, v___x_553_, v___x_554_);
v___x_556_ = lean_unbox(v___x_555_);
return v___x_556_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_lift___redArg___lam__3___boxed(lean_object* v_f_557_, lean_object* v_toOrd_558_, lean_object* v_a_559_, lean_object* v_b_560_){
_start:
{
uint8_t v_res_561_; lean_object* v_r_562_; 
v_res_561_ = lp_mathlib_LinearOrder_lift___redArg___lam__3(v_f_557_, v_toOrd_558_, v_a_559_, v_b_560_);
v_r_562_ = lean_box(v_res_561_);
return v_r_562_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_lift___redArg(lean_object* v_inst_563_, lean_object* v_inst_564_, lean_object* v_inst_565_, lean_object* v_f_566_){
_start:
{
lean_object* v_toOrd_567_; lean_object* v_toDecidableLE_568_; lean_object* v_toDecidableEq_569_; lean_object* v_toDecidableLT_570_; lean_object* v___x_572_; uint8_t v_isShared_573_; uint8_t v_isSharedCheck_582_; 
v_toOrd_567_ = lean_ctor_get(v_inst_563_, 3);
v_toDecidableLE_568_ = lean_ctor_get(v_inst_563_, 4);
v_toDecidableEq_569_ = lean_ctor_get(v_inst_563_, 5);
v_toDecidableLT_570_ = lean_ctor_get(v_inst_563_, 6);
v_isSharedCheck_582_ = !lean_is_exclusive(v_inst_563_);
if (v_isSharedCheck_582_ == 0)
{
lean_object* v_unused_583_; lean_object* v_unused_584_; lean_object* v_unused_585_; 
v_unused_583_ = lean_ctor_get(v_inst_563_, 2);
lean_dec(v_unused_583_);
v_unused_584_ = lean_ctor_get(v_inst_563_, 1);
lean_dec(v_unused_584_);
v_unused_585_ = lean_ctor_get(v_inst_563_, 0);
lean_dec(v_unused_585_);
v___x_572_ = v_inst_563_;
v_isShared_573_ = v_isSharedCheck_582_;
goto v_resetjp_571_;
}
else
{
lean_inc(v_toDecidableLT_570_);
lean_inc(v_toDecidableEq_569_);
lean_inc(v_toDecidableLE_568_);
lean_inc(v_toOrd_567_);
lean_dec(v_inst_563_);
v___x_572_ = lean_box(0);
v_isShared_573_ = v_isSharedCheck_582_;
goto v_resetjp_571_;
}
v_resetjp_571_:
{
lean_object* v___f_574_; lean_object* v___f_575_; lean_object* v___f_576_; lean_object* v___f_577_; lean_object* v___x_578_; lean_object* v___x_580_; 
lean_inc_n(v_f_566_, 3);
v___f_574_ = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_lift___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_574_, 0, v_f_566_);
lean_closure_set(v___f_574_, 1, v_toDecidableEq_569_);
v___f_575_ = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_lift___redArg___lam__1___boxed), 4, 2);
lean_closure_set(v___f_575_, 0, v_f_566_);
lean_closure_set(v___f_575_, 1, v_toDecidableLE_568_);
v___f_576_ = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_lift___redArg___lam__2___boxed), 4, 2);
lean_closure_set(v___f_576_, 0, v_f_566_);
lean_closure_set(v___f_576_, 1, v_toDecidableLT_570_);
v___f_577_ = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_lift___redArg___lam__3___boxed), 4, 2);
lean_closure_set(v___f_577_, 0, v_f_566_);
lean_closure_set(v___f_577_, 1, v_toOrd_567_);
v___x_578_ = ((lean_object*)(lp_mathlib_Preorder_lift___closed__0));
if (v_isShared_573_ == 0)
{
lean_ctor_set(v___x_572_, 6, v___f_576_);
lean_ctor_set(v___x_572_, 5, v___f_574_);
lean_ctor_set(v___x_572_, 4, v___f_575_);
lean_ctor_set(v___x_572_, 3, v___f_577_);
lean_ctor_set(v___x_572_, 2, v_inst_564_);
lean_ctor_set(v___x_572_, 1, v_inst_565_);
lean_ctor_set(v___x_572_, 0, v___x_578_);
v___x_580_ = v___x_572_;
goto v_reusejp_579_;
}
else
{
lean_object* v_reuseFailAlloc_581_; 
v_reuseFailAlloc_581_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v_reuseFailAlloc_581_, 0, v___x_578_);
lean_ctor_set(v_reuseFailAlloc_581_, 1, v_inst_565_);
lean_ctor_set(v_reuseFailAlloc_581_, 2, v_inst_564_);
lean_ctor_set(v_reuseFailAlloc_581_, 3, v___f_577_);
lean_ctor_set(v_reuseFailAlloc_581_, 4, v___f_575_);
lean_ctor_set(v_reuseFailAlloc_581_, 5, v___f_574_);
lean_ctor_set(v_reuseFailAlloc_581_, 6, v___f_576_);
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
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_lift(lean_object* v_00_u03b1_586_, lean_object* v_00_u03b2_587_, lean_object* v_inst_588_, lean_object* v_inst_589_, lean_object* v_inst_590_, lean_object* v_f_591_, lean_object* v_inj_592_, lean_object* v_hsup_593_, lean_object* v_hinf_594_){
_start:
{
lean_object* v_toOrd_595_; lean_object* v_toDecidableLE_596_; lean_object* v_toDecidableEq_597_; lean_object* v_toDecidableLT_598_; lean_object* v___x_600_; uint8_t v_isShared_601_; uint8_t v_isSharedCheck_610_; 
v_toOrd_595_ = lean_ctor_get(v_inst_588_, 3);
v_toDecidableLE_596_ = lean_ctor_get(v_inst_588_, 4);
v_toDecidableEq_597_ = lean_ctor_get(v_inst_588_, 5);
v_toDecidableLT_598_ = lean_ctor_get(v_inst_588_, 6);
v_isSharedCheck_610_ = !lean_is_exclusive(v_inst_588_);
if (v_isSharedCheck_610_ == 0)
{
lean_object* v_unused_611_; lean_object* v_unused_612_; lean_object* v_unused_613_; 
v_unused_611_ = lean_ctor_get(v_inst_588_, 2);
lean_dec(v_unused_611_);
v_unused_612_ = lean_ctor_get(v_inst_588_, 1);
lean_dec(v_unused_612_);
v_unused_613_ = lean_ctor_get(v_inst_588_, 0);
lean_dec(v_unused_613_);
v___x_600_ = v_inst_588_;
v_isShared_601_ = v_isSharedCheck_610_;
goto v_resetjp_599_;
}
else
{
lean_inc(v_toDecidableLT_598_);
lean_inc(v_toDecidableEq_597_);
lean_inc(v_toDecidableLE_596_);
lean_inc(v_toOrd_595_);
lean_dec(v_inst_588_);
v___x_600_ = lean_box(0);
v_isShared_601_ = v_isSharedCheck_610_;
goto v_resetjp_599_;
}
v_resetjp_599_:
{
lean_object* v___f_602_; lean_object* v___f_603_; lean_object* v___f_604_; lean_object* v___f_605_; lean_object* v___x_606_; lean_object* v___x_608_; 
lean_inc_n(v_f_591_, 3);
v___f_602_ = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_lift___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_602_, 0, v_f_591_);
lean_closure_set(v___f_602_, 1, v_toDecidableEq_597_);
v___f_603_ = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_lift___redArg___lam__1___boxed), 4, 2);
lean_closure_set(v___f_603_, 0, v_f_591_);
lean_closure_set(v___f_603_, 1, v_toDecidableLE_596_);
v___f_604_ = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_lift___redArg___lam__2___boxed), 4, 2);
lean_closure_set(v___f_604_, 0, v_f_591_);
lean_closure_set(v___f_604_, 1, v_toDecidableLT_598_);
v___f_605_ = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_lift___redArg___lam__3___boxed), 4, 2);
lean_closure_set(v___f_605_, 0, v_f_591_);
lean_closure_set(v___f_605_, 1, v_toOrd_595_);
v___x_606_ = ((lean_object*)(lp_mathlib_Preorder_lift___closed__0));
if (v_isShared_601_ == 0)
{
lean_ctor_set(v___x_600_, 6, v___f_604_);
lean_ctor_set(v___x_600_, 5, v___f_602_);
lean_ctor_set(v___x_600_, 4, v___f_603_);
lean_ctor_set(v___x_600_, 3, v___f_605_);
lean_ctor_set(v___x_600_, 2, v_inst_589_);
lean_ctor_set(v___x_600_, 1, v_inst_590_);
lean_ctor_set(v___x_600_, 0, v___x_606_);
v___x_608_ = v___x_600_;
goto v_reusejp_607_;
}
else
{
lean_object* v_reuseFailAlloc_609_; 
v_reuseFailAlloc_609_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v_reuseFailAlloc_609_, 0, v___x_606_);
lean_ctor_set(v_reuseFailAlloc_609_, 1, v_inst_590_);
lean_ctor_set(v_reuseFailAlloc_609_, 2, v_inst_589_);
lean_ctor_set(v_reuseFailAlloc_609_, 3, v___f_605_);
lean_ctor_set(v_reuseFailAlloc_609_, 4, v___f_603_);
lean_ctor_set(v_reuseFailAlloc_609_, 5, v___f_602_);
lean_ctor_set(v_reuseFailAlloc_609_, 6, v___f_604_);
v___x_608_ = v_reuseFailAlloc_609_;
goto v_reusejp_607_;
}
v_reusejp_607_:
{
return v___x_608_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_lift_x27___redArg___lam__0(lean_object* v_f_614_, lean_object* v_toDecidableLE_615_, lean_object* v_x_616_, lean_object* v_y_617_){
_start:
{
lean_object* v___x_618_; lean_object* v___x_619_; lean_object* v___x_620_; uint8_t v___x_621_; 
lean_inc(v_f_614_);
lean_inc(v_x_616_);
v___x_618_ = lean_apply_1(v_f_614_, v_x_616_);
lean_inc(v_y_617_);
v___x_619_ = lean_apply_1(v_f_614_, v_y_617_);
v___x_620_ = lean_apply_2(v_toDecidableLE_615_, v___x_618_, v___x_619_);
v___x_621_ = lean_unbox(v___x_620_);
if (v___x_621_ == 0)
{
lean_dec(v_y_617_);
return v_x_616_;
}
else
{
lean_dec(v_x_616_);
return v_y_617_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_lift_x27___redArg___lam__1(lean_object* v_f_622_, lean_object* v_toDecidableLE_623_, lean_object* v_x_624_, lean_object* v_y_625_){
_start:
{
lean_object* v___x_626_; lean_object* v___x_627_; lean_object* v___x_628_; uint8_t v___x_629_; 
lean_inc(v_f_622_);
lean_inc(v_x_624_);
v___x_626_ = lean_apply_1(v_f_622_, v_x_624_);
lean_inc(v_y_625_);
v___x_627_ = lean_apply_1(v_f_622_, v_y_625_);
v___x_628_ = lean_apply_2(v_toDecidableLE_623_, v___x_626_, v___x_627_);
v___x_629_ = lean_unbox(v___x_628_);
if (v___x_629_ == 0)
{
lean_dec(v_x_624_);
return v_y_625_;
}
else
{
lean_dec(v_y_625_);
return v_x_624_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_lift_x27___redArg(lean_object* v_inst_630_, lean_object* v_f_631_){
_start:
{
lean_object* v_toOrd_632_; lean_object* v_toDecidableLE_633_; lean_object* v_toDecidableEq_634_; lean_object* v_toDecidableLT_635_; lean_object* v___x_637_; uint8_t v_isShared_638_; uint8_t v_isSharedCheck_649_; 
v_toOrd_632_ = lean_ctor_get(v_inst_630_, 3);
v_toDecidableLE_633_ = lean_ctor_get(v_inst_630_, 4);
v_toDecidableEq_634_ = lean_ctor_get(v_inst_630_, 5);
v_toDecidableLT_635_ = lean_ctor_get(v_inst_630_, 6);
v_isSharedCheck_649_ = !lean_is_exclusive(v_inst_630_);
if (v_isSharedCheck_649_ == 0)
{
lean_object* v_unused_650_; lean_object* v_unused_651_; lean_object* v_unused_652_; 
v_unused_650_ = lean_ctor_get(v_inst_630_, 2);
lean_dec(v_unused_650_);
v_unused_651_ = lean_ctor_get(v_inst_630_, 1);
lean_dec(v_unused_651_);
v_unused_652_ = lean_ctor_get(v_inst_630_, 0);
lean_dec(v_unused_652_);
v___x_637_ = v_inst_630_;
v_isShared_638_ = v_isSharedCheck_649_;
goto v_resetjp_636_;
}
else
{
lean_inc(v_toDecidableLT_635_);
lean_inc(v_toDecidableEq_634_);
lean_inc(v_toDecidableLE_633_);
lean_inc(v_toOrd_632_);
lean_dec(v_inst_630_);
v___x_637_ = lean_box(0);
v_isShared_638_ = v_isSharedCheck_649_;
goto v_resetjp_636_;
}
v_resetjp_636_:
{
lean_object* v___f_639_; lean_object* v___f_640_; lean_object* v___f_641_; lean_object* v___f_642_; lean_object* v___f_643_; lean_object* v___f_644_; lean_object* v___x_645_; lean_object* v___x_647_; 
lean_inc_ref_n(v_toDecidableLE_633_, 2);
lean_inc_n(v_f_631_, 5);
v___f_639_ = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_lift_x27___redArg___lam__0), 4, 2);
lean_closure_set(v___f_639_, 0, v_f_631_);
lean_closure_set(v___f_639_, 1, v_toDecidableLE_633_);
v___f_640_ = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_lift_x27___redArg___lam__1), 4, 2);
lean_closure_set(v___f_640_, 0, v_f_631_);
lean_closure_set(v___f_640_, 1, v_toDecidableLE_633_);
v___f_641_ = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_lift___redArg___lam__2___boxed), 4, 2);
lean_closure_set(v___f_641_, 0, v_f_631_);
lean_closure_set(v___f_641_, 1, v_toDecidableLT_635_);
v___f_642_ = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_lift___redArg___lam__1___boxed), 4, 2);
lean_closure_set(v___f_642_, 0, v_f_631_);
lean_closure_set(v___f_642_, 1, v_toDecidableLE_633_);
v___f_643_ = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_lift___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_643_, 0, v_f_631_);
lean_closure_set(v___f_643_, 1, v_toDecidableEq_634_);
v___f_644_ = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_lift___redArg___lam__3___boxed), 4, 2);
lean_closure_set(v___f_644_, 0, v_f_631_);
lean_closure_set(v___f_644_, 1, v_toOrd_632_);
v___x_645_ = ((lean_object*)(lp_mathlib_Preorder_lift___closed__0));
if (v_isShared_638_ == 0)
{
lean_ctor_set(v___x_637_, 6, v___f_641_);
lean_ctor_set(v___x_637_, 5, v___f_643_);
lean_ctor_set(v___x_637_, 4, v___f_642_);
lean_ctor_set(v___x_637_, 3, v___f_644_);
lean_ctor_set(v___x_637_, 2, v___f_639_);
lean_ctor_set(v___x_637_, 1, v___f_640_);
lean_ctor_set(v___x_637_, 0, v___x_645_);
v___x_647_ = v___x_637_;
goto v_reusejp_646_;
}
else
{
lean_object* v_reuseFailAlloc_648_; 
v_reuseFailAlloc_648_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v_reuseFailAlloc_648_, 0, v___x_645_);
lean_ctor_set(v_reuseFailAlloc_648_, 1, v___f_640_);
lean_ctor_set(v_reuseFailAlloc_648_, 2, v___f_639_);
lean_ctor_set(v_reuseFailAlloc_648_, 3, v___f_644_);
lean_ctor_set(v_reuseFailAlloc_648_, 4, v___f_642_);
lean_ctor_set(v_reuseFailAlloc_648_, 5, v___f_643_);
lean_ctor_set(v_reuseFailAlloc_648_, 6, v___f_641_);
v___x_647_ = v_reuseFailAlloc_648_;
goto v_reusejp_646_;
}
v_reusejp_646_:
{
return v___x_647_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_lift_x27(lean_object* v_00_u03b1_653_, lean_object* v_00_u03b2_654_, lean_object* v_inst_655_, lean_object* v_f_656_, lean_object* v_inj_657_){
_start:
{
lean_object* v_toOrd_658_; lean_object* v_toDecidableLE_659_; lean_object* v_toDecidableEq_660_; lean_object* v_toDecidableLT_661_; lean_object* v___x_663_; uint8_t v_isShared_664_; uint8_t v_isSharedCheck_675_; 
v_toOrd_658_ = lean_ctor_get(v_inst_655_, 3);
v_toDecidableLE_659_ = lean_ctor_get(v_inst_655_, 4);
v_toDecidableEq_660_ = lean_ctor_get(v_inst_655_, 5);
v_toDecidableLT_661_ = lean_ctor_get(v_inst_655_, 6);
v_isSharedCheck_675_ = !lean_is_exclusive(v_inst_655_);
if (v_isSharedCheck_675_ == 0)
{
lean_object* v_unused_676_; lean_object* v_unused_677_; lean_object* v_unused_678_; 
v_unused_676_ = lean_ctor_get(v_inst_655_, 2);
lean_dec(v_unused_676_);
v_unused_677_ = lean_ctor_get(v_inst_655_, 1);
lean_dec(v_unused_677_);
v_unused_678_ = lean_ctor_get(v_inst_655_, 0);
lean_dec(v_unused_678_);
v___x_663_ = v_inst_655_;
v_isShared_664_ = v_isSharedCheck_675_;
goto v_resetjp_662_;
}
else
{
lean_inc(v_toDecidableLT_661_);
lean_inc(v_toDecidableEq_660_);
lean_inc(v_toDecidableLE_659_);
lean_inc(v_toOrd_658_);
lean_dec(v_inst_655_);
v___x_663_ = lean_box(0);
v_isShared_664_ = v_isSharedCheck_675_;
goto v_resetjp_662_;
}
v_resetjp_662_:
{
lean_object* v___f_665_; lean_object* v___f_666_; lean_object* v___f_667_; lean_object* v___f_668_; lean_object* v___f_669_; lean_object* v___f_670_; lean_object* v___x_671_; lean_object* v___x_673_; 
lean_inc_ref_n(v_toDecidableLE_659_, 2);
lean_inc_n(v_f_656_, 5);
v___f_665_ = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_lift_x27___redArg___lam__0), 4, 2);
lean_closure_set(v___f_665_, 0, v_f_656_);
lean_closure_set(v___f_665_, 1, v_toDecidableLE_659_);
v___f_666_ = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_lift_x27___redArg___lam__1), 4, 2);
lean_closure_set(v___f_666_, 0, v_f_656_);
lean_closure_set(v___f_666_, 1, v_toDecidableLE_659_);
v___f_667_ = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_lift___redArg___lam__2___boxed), 4, 2);
lean_closure_set(v___f_667_, 0, v_f_656_);
lean_closure_set(v___f_667_, 1, v_toDecidableLT_661_);
v___f_668_ = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_lift___redArg___lam__1___boxed), 4, 2);
lean_closure_set(v___f_668_, 0, v_f_656_);
lean_closure_set(v___f_668_, 1, v_toDecidableLE_659_);
v___f_669_ = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_lift___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_669_, 0, v_f_656_);
lean_closure_set(v___f_669_, 1, v_toDecidableEq_660_);
v___f_670_ = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_lift___redArg___lam__3___boxed), 4, 2);
lean_closure_set(v___f_670_, 0, v_f_656_);
lean_closure_set(v___f_670_, 1, v_toOrd_658_);
v___x_671_ = ((lean_object*)(lp_mathlib_Preorder_lift___closed__0));
if (v_isShared_664_ == 0)
{
lean_ctor_set(v___x_663_, 6, v___f_667_);
lean_ctor_set(v___x_663_, 5, v___f_669_);
lean_ctor_set(v___x_663_, 4, v___f_668_);
lean_ctor_set(v___x_663_, 3, v___f_670_);
lean_ctor_set(v___x_663_, 2, v___f_665_);
lean_ctor_set(v___x_663_, 1, v___f_666_);
lean_ctor_set(v___x_663_, 0, v___x_671_);
v___x_673_ = v___x_663_;
goto v_reusejp_672_;
}
else
{
lean_object* v_reuseFailAlloc_674_; 
v_reuseFailAlloc_674_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v_reuseFailAlloc_674_, 0, v___x_671_);
lean_ctor_set(v_reuseFailAlloc_674_, 1, v___f_666_);
lean_ctor_set(v_reuseFailAlloc_674_, 2, v___f_665_);
lean_ctor_set(v_reuseFailAlloc_674_, 3, v___f_670_);
lean_ctor_set(v_reuseFailAlloc_674_, 4, v___f_668_);
lean_ctor_set(v_reuseFailAlloc_674_, 5, v___f_669_);
lean_ctor_set(v_reuseFailAlloc_674_, 6, v___f_667_);
v___x_673_ = v_reuseFailAlloc_674_;
goto v_reusejp_672_;
}
v_reusejp_672_:
{
return v___x_673_;
}
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_LinearOrder_liftWithOrd___redArg___lam__0(lean_object* v_inst_679_, lean_object* v_f_680_, lean_object* v_x_681_, lean_object* v_y_682_){
_start:
{
lean_object* v_toDecidableEq_683_; lean_object* v___x_684_; lean_object* v___x_685_; lean_object* v___x_686_; uint8_t v___x_687_; 
v_toDecidableEq_683_ = lean_ctor_get(v_inst_679_, 5);
lean_inc_ref(v_toDecidableEq_683_);
lean_dec_ref(v_inst_679_);
lean_inc(v_f_680_);
v___x_684_ = lean_apply_1(v_f_680_, v_x_681_);
v___x_685_ = lean_apply_1(v_f_680_, v_y_682_);
v___x_686_ = lean_apply_2(v_toDecidableEq_683_, v___x_684_, v___x_685_);
v___x_687_ = lean_unbox(v___x_686_);
return v___x_687_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_liftWithOrd___redArg___lam__0___boxed(lean_object* v_inst_688_, lean_object* v_f_689_, lean_object* v_x_690_, lean_object* v_y_691_){
_start:
{
uint8_t v_res_692_; lean_object* v_r_693_; 
v_res_692_ = lp_mathlib_LinearOrder_liftWithOrd___redArg___lam__0(v_inst_688_, v_f_689_, v_x_690_, v_y_691_);
v_r_693_ = lean_box(v_res_692_);
return v_r_693_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_LinearOrder_liftWithOrd___redArg___lam__1(lean_object* v_inst_694_, lean_object* v_f_695_, lean_object* v_x_696_, lean_object* v_y_697_){
_start:
{
lean_object* v_toDecidableLE_698_; lean_object* v___x_699_; lean_object* v___x_700_; lean_object* v___x_701_; uint8_t v___x_702_; 
v_toDecidableLE_698_ = lean_ctor_get(v_inst_694_, 4);
lean_inc_ref(v_toDecidableLE_698_);
lean_dec_ref(v_inst_694_);
lean_inc(v_f_695_);
v___x_699_ = lean_apply_1(v_f_695_, v_x_696_);
v___x_700_ = lean_apply_1(v_f_695_, v_y_697_);
v___x_701_ = lean_apply_2(v_toDecidableLE_698_, v___x_699_, v___x_700_);
v___x_702_ = lean_unbox(v___x_701_);
return v___x_702_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_liftWithOrd___redArg___lam__1___boxed(lean_object* v_inst_703_, lean_object* v_f_704_, lean_object* v_x_705_, lean_object* v_y_706_){
_start:
{
uint8_t v_res_707_; lean_object* v_r_708_; 
v_res_707_ = lp_mathlib_LinearOrder_liftWithOrd___redArg___lam__1(v_inst_703_, v_f_704_, v_x_705_, v_y_706_);
v_r_708_ = lean_box(v_res_707_);
return v_r_708_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_LinearOrder_liftWithOrd___redArg___lam__2(lean_object* v_inst_709_, lean_object* v_f_710_, lean_object* v_x_711_, lean_object* v_y_712_){
_start:
{
lean_object* v_toDecidableLT_713_; lean_object* v___x_714_; lean_object* v___x_715_; lean_object* v___x_716_; uint8_t v___x_717_; 
v_toDecidableLT_713_ = lean_ctor_get(v_inst_709_, 6);
lean_inc_ref(v_toDecidableLT_713_);
lean_dec_ref(v_inst_709_);
lean_inc(v_f_710_);
v___x_714_ = lean_apply_1(v_f_710_, v_x_711_);
v___x_715_ = lean_apply_1(v_f_710_, v_y_712_);
v___x_716_ = lean_apply_2(v_toDecidableLT_713_, v___x_714_, v___x_715_);
v___x_717_ = lean_unbox(v___x_716_);
return v___x_717_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_liftWithOrd___redArg___lam__2___boxed(lean_object* v_inst_718_, lean_object* v_f_719_, lean_object* v_x_720_, lean_object* v_y_721_){
_start:
{
uint8_t v_res_722_; lean_object* v_r_723_; 
v_res_722_ = lp_mathlib_LinearOrder_liftWithOrd___redArg___lam__2(v_inst_718_, v_f_719_, v_x_720_, v_y_721_);
v_r_723_ = lean_box(v_res_722_);
return v_r_723_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_liftWithOrd___redArg(lean_object* v_inst_724_, lean_object* v_inst_725_, lean_object* v_inst_726_, lean_object* v_inst_727_, lean_object* v_f_728_){
_start:
{
lean_object* v___f_729_; lean_object* v___f_730_; lean_object* v___f_731_; lean_object* v___x_732_; lean_object* v___x_733_; 
lean_inc_n(v_f_728_, 2);
lean_inc_ref_n(v_inst_724_, 2);
v___f_729_ = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_liftWithOrd___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_729_, 0, v_inst_724_);
lean_closure_set(v___f_729_, 1, v_f_728_);
v___f_730_ = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_liftWithOrd___redArg___lam__1___boxed), 4, 2);
lean_closure_set(v___f_730_, 0, v_inst_724_);
lean_closure_set(v___f_730_, 1, v_f_728_);
v___f_731_ = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_liftWithOrd___redArg___lam__2___boxed), 4, 2);
lean_closure_set(v___f_731_, 0, v_inst_724_);
lean_closure_set(v___f_731_, 1, v_f_728_);
v___x_732_ = ((lean_object*)(lp_mathlib_Preorder_lift___closed__0));
v___x_733_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v___x_733_, 0, v___x_732_);
lean_ctor_set(v___x_733_, 1, v_inst_726_);
lean_ctor_set(v___x_733_, 2, v_inst_725_);
lean_ctor_set(v___x_733_, 3, v_inst_727_);
lean_ctor_set(v___x_733_, 4, v___f_730_);
lean_ctor_set(v___x_733_, 5, v___f_729_);
lean_ctor_set(v___x_733_, 6, v___f_731_);
return v___x_733_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_liftWithOrd(lean_object* v_00_u03b1_734_, lean_object* v_00_u03b2_735_, lean_object* v_inst_736_, lean_object* v_inst_737_, lean_object* v_inst_738_, lean_object* v_inst_739_, lean_object* v_f_740_, lean_object* v_inj_741_, lean_object* v_hsup_742_, lean_object* v_hinf_743_, lean_object* v_compare__f_744_){
_start:
{
lean_object* v___f_745_; lean_object* v___f_746_; lean_object* v___f_747_; lean_object* v___x_748_; lean_object* v___x_749_; 
lean_inc_n(v_f_740_, 2);
lean_inc_ref_n(v_inst_736_, 2);
v___f_745_ = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_liftWithOrd___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_745_, 0, v_inst_736_);
lean_closure_set(v___f_745_, 1, v_f_740_);
v___f_746_ = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_liftWithOrd___redArg___lam__1___boxed), 4, 2);
lean_closure_set(v___f_746_, 0, v_inst_736_);
lean_closure_set(v___f_746_, 1, v_f_740_);
v___f_747_ = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_liftWithOrd___redArg___lam__2___boxed), 4, 2);
lean_closure_set(v___f_747_, 0, v_inst_736_);
lean_closure_set(v___f_747_, 1, v_f_740_);
v___x_748_ = ((lean_object*)(lp_mathlib_Preorder_lift___closed__0));
v___x_749_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v___x_749_, 0, v___x_748_);
lean_ctor_set(v___x_749_, 1, v_inst_738_);
lean_ctor_set(v___x_749_, 2, v_inst_737_);
lean_ctor_set(v___x_749_, 3, v_inst_739_);
lean_ctor_set(v___x_749_, 4, v___f_746_);
lean_ctor_set(v___x_749_, 5, v___f_745_);
lean_ctor_set(v___x_749_, 6, v___f_747_);
return v___x_749_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_liftWithOrd_x27___redArg___lam__0(lean_object* v_inst_750_, lean_object* v_f_751_, lean_object* v_x_752_, lean_object* v_y_753_){
_start:
{
lean_object* v_toDecidableLE_754_; lean_object* v___x_755_; lean_object* v___x_756_; lean_object* v___x_757_; uint8_t v___x_758_; 
v_toDecidableLE_754_ = lean_ctor_get(v_inst_750_, 4);
lean_inc_ref(v_toDecidableLE_754_);
lean_dec_ref(v_inst_750_);
lean_inc(v_f_751_);
lean_inc(v_x_752_);
v___x_755_ = lean_apply_1(v_f_751_, v_x_752_);
lean_inc(v_y_753_);
v___x_756_ = lean_apply_1(v_f_751_, v_y_753_);
v___x_757_ = lean_apply_2(v_toDecidableLE_754_, v___x_755_, v___x_756_);
v___x_758_ = lean_unbox(v___x_757_);
if (v___x_758_ == 0)
{
lean_dec(v_y_753_);
return v_x_752_;
}
else
{
lean_dec(v_x_752_);
return v_y_753_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_liftWithOrd_x27___redArg___lam__1(lean_object* v_inst_759_, lean_object* v_f_760_, lean_object* v_x_761_, lean_object* v_y_762_){
_start:
{
lean_object* v_toDecidableLE_763_; lean_object* v___x_764_; lean_object* v___x_765_; lean_object* v___x_766_; uint8_t v___x_767_; 
v_toDecidableLE_763_ = lean_ctor_get(v_inst_759_, 4);
lean_inc_ref(v_toDecidableLE_763_);
lean_dec_ref(v_inst_759_);
lean_inc(v_f_760_);
lean_inc(v_x_761_);
v___x_764_ = lean_apply_1(v_f_760_, v_x_761_);
lean_inc(v_y_762_);
v___x_765_ = lean_apply_1(v_f_760_, v_y_762_);
v___x_766_ = lean_apply_2(v_toDecidableLE_763_, v___x_764_, v___x_765_);
v___x_767_ = lean_unbox(v___x_766_);
if (v___x_767_ == 0)
{
lean_dec(v_x_761_);
return v_y_762_;
}
else
{
lean_dec(v_y_762_);
return v_x_761_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_liftWithOrd_x27___redArg(lean_object* v_inst_768_, lean_object* v_inst_769_, lean_object* v_f_770_){
_start:
{
lean_object* v___f_771_; lean_object* v___f_772_; lean_object* v___f_773_; lean_object* v___f_774_; lean_object* v___f_775_; lean_object* v___x_776_; lean_object* v___x_777_; 
lean_inc_n(v_f_770_, 4);
lean_inc_ref_n(v_inst_768_, 4);
v___f_771_ = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_liftWithOrd_x27___redArg___lam__0), 4, 2);
lean_closure_set(v___f_771_, 0, v_inst_768_);
lean_closure_set(v___f_771_, 1, v_f_770_);
v___f_772_ = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_liftWithOrd_x27___redArg___lam__1), 4, 2);
lean_closure_set(v___f_772_, 0, v_inst_768_);
lean_closure_set(v___f_772_, 1, v_f_770_);
v___f_773_ = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_liftWithOrd___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_773_, 0, v_inst_768_);
lean_closure_set(v___f_773_, 1, v_f_770_);
v___f_774_ = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_liftWithOrd___redArg___lam__1___boxed), 4, 2);
lean_closure_set(v___f_774_, 0, v_inst_768_);
lean_closure_set(v___f_774_, 1, v_f_770_);
v___f_775_ = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_liftWithOrd___redArg___lam__2___boxed), 4, 2);
lean_closure_set(v___f_775_, 0, v_inst_768_);
lean_closure_set(v___f_775_, 1, v_f_770_);
v___x_776_ = ((lean_object*)(lp_mathlib_Preorder_lift___closed__0));
v___x_777_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v___x_777_, 0, v___x_776_);
lean_ctor_set(v___x_777_, 1, v___f_772_);
lean_ctor_set(v___x_777_, 2, v___f_771_);
lean_ctor_set(v___x_777_, 3, v_inst_769_);
lean_ctor_set(v___x_777_, 4, v___f_774_);
lean_ctor_set(v___x_777_, 5, v___f_773_);
lean_ctor_set(v___x_777_, 6, v___f_775_);
return v___x_777_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_liftWithOrd_x27(lean_object* v_00_u03b1_778_, lean_object* v_00_u03b2_779_, lean_object* v_inst_780_, lean_object* v_inst_781_, lean_object* v_f_782_, lean_object* v_inj_783_, lean_object* v_compare__f_784_){
_start:
{
lean_object* v___f_785_; lean_object* v___f_786_; lean_object* v___f_787_; lean_object* v___f_788_; lean_object* v___f_789_; lean_object* v___x_790_; lean_object* v___x_791_; 
lean_inc_n(v_f_782_, 4);
lean_inc_ref_n(v_inst_780_, 4);
v___f_785_ = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_liftWithOrd_x27___redArg___lam__0), 4, 2);
lean_closure_set(v___f_785_, 0, v_inst_780_);
lean_closure_set(v___f_785_, 1, v_f_782_);
v___f_786_ = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_liftWithOrd_x27___redArg___lam__1), 4, 2);
lean_closure_set(v___f_786_, 0, v_inst_780_);
lean_closure_set(v___f_786_, 1, v_f_782_);
v___f_787_ = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_liftWithOrd___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_787_, 0, v_inst_780_);
lean_closure_set(v___f_787_, 1, v_f_782_);
v___f_788_ = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_liftWithOrd___redArg___lam__1___boxed), 4, 2);
lean_closure_set(v___f_788_, 0, v_inst_780_);
lean_closure_set(v___f_788_, 1, v_f_782_);
v___f_789_ = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_liftWithOrd___redArg___lam__2___boxed), 4, 2);
lean_closure_set(v___f_789_, 0, v_inst_780_);
lean_closure_set(v___f_789_, 1, v_f_782_);
v___x_790_ = ((lean_object*)(lp_mathlib_Preorder_lift___closed__0));
v___x_791_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v___x_791_, 0, v___x_790_);
lean_ctor_set(v___x_791_, 1, v___f_786_);
lean_ctor_set(v___x_791_, 2, v___f_785_);
lean_ctor_set(v___x_791_, 3, v_inst_781_);
lean_ctor_set(v___x_791_, 4, v___f_788_);
lean_ctor_set(v___x_791_, 5, v___f_787_);
lean_ctor_set(v___x_791_, 6, v___f_789_);
return v___x_791_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_preorder(lean_object* v_00_u03b1_792_, lean_object* v_inst_793_, lean_object* v_p_794_){
_start:
{
lean_object* v___x_795_; 
v___x_795_ = ((lean_object*)(lp_mathlib_Preorder_lift___closed__0));
return v___x_795_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_preorder___boxed(lean_object* v_00_u03b1_796_, lean_object* v_inst_797_, lean_object* v_p_798_){
_start:
{
lean_object* v_res_799_; 
v_res_799_ = lp_mathlib_Subtype_preorder(v_00_u03b1_796_, v_inst_797_, v_p_798_);
lean_dec_ref(v_inst_797_);
return v_res_799_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_partialOrder___redArg(lean_object* v_inst_800_){
_start:
{
lean_object* v___x_801_; 
v___x_801_ = lp_mathlib_Subtype_preorder(lean_box(0), v_inst_800_, lean_box(0));
return v___x_801_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_partialOrder___redArg___boxed(lean_object* v_inst_802_){
_start:
{
lean_object* v_res_803_; 
v_res_803_ = lp_mathlib_Subtype_partialOrder___redArg(v_inst_802_);
lean_dec_ref(v_inst_802_);
return v_res_803_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_partialOrder(lean_object* v_00_u03b1_804_, lean_object* v_inst_805_, lean_object* v_p_806_){
_start:
{
lean_object* v___x_807_; 
v___x_807_ = lp_mathlib_Subtype_preorder(lean_box(0), v_inst_805_, lean_box(0));
return v___x_807_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_partialOrder___boxed(lean_object* v_00_u03b1_808_, lean_object* v_inst_809_, lean_object* v_p_810_){
_start:
{
lean_object* v_res_811_; 
v_res_811_ = lp_mathlib_Subtype_partialOrder(v_00_u03b1_808_, v_inst_809_, v_p_810_);
lean_dec_ref(v_inst_809_);
return v_res_811_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Subtype_decidableLE___redArg(lean_object* v_h_812_, lean_object* v_a_813_, lean_object* v_b_814_){
_start:
{
lean_object* v___x_815_; uint8_t v___x_816_; 
v___x_815_ = lean_apply_2(v_h_812_, v_a_813_, v_b_814_);
v___x_816_ = lean_unbox(v___x_815_);
return v___x_816_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_decidableLE___redArg___boxed(lean_object* v_h_817_, lean_object* v_a_818_, lean_object* v_b_819_){
_start:
{
uint8_t v_res_820_; lean_object* v_r_821_; 
v_res_820_ = lp_mathlib_Subtype_decidableLE___redArg(v_h_817_, v_a_818_, v_b_819_);
v_r_821_ = lean_box(v_res_820_);
return v_r_821_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Subtype_decidableLE(lean_object* v_00_u03b1_822_, lean_object* v_inst_823_, lean_object* v_h_824_, lean_object* v_p_825_, lean_object* v_a_826_, lean_object* v_b_827_){
_start:
{
lean_object* v___x_828_; uint8_t v___x_829_; 
v___x_828_ = lean_apply_2(v_h_824_, v_a_826_, v_b_827_);
v___x_829_ = lean_unbox(v___x_828_);
return v___x_829_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_decidableLE___boxed(lean_object* v_00_u03b1_830_, lean_object* v_inst_831_, lean_object* v_h_832_, lean_object* v_p_833_, lean_object* v_a_834_, lean_object* v_b_835_){
_start:
{
uint8_t v_res_836_; lean_object* v_r_837_; 
v_res_836_ = lp_mathlib_Subtype_decidableLE(v_00_u03b1_830_, v_inst_831_, v_h_832_, v_p_833_, v_a_834_, v_b_835_);
lean_dec_ref(v_inst_831_);
v_r_837_ = lean_box(v_res_836_);
return v_r_837_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Subtype_decidableLT___redArg(lean_object* v_h_838_, lean_object* v_a_839_, lean_object* v_b_840_){
_start:
{
lean_object* v___x_841_; uint8_t v___x_842_; 
v___x_841_ = lean_apply_2(v_h_838_, v_a_839_, v_b_840_);
v___x_842_ = lean_unbox(v___x_841_);
return v___x_842_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_decidableLT___redArg___boxed(lean_object* v_h_843_, lean_object* v_a_844_, lean_object* v_b_845_){
_start:
{
uint8_t v_res_846_; lean_object* v_r_847_; 
v_res_846_ = lp_mathlib_Subtype_decidableLT___redArg(v_h_843_, v_a_844_, v_b_845_);
v_r_847_ = lean_box(v_res_846_);
return v_r_847_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Subtype_decidableLT(lean_object* v_00_u03b1_848_, lean_object* v_inst_849_, lean_object* v_h_850_, lean_object* v_p_851_, lean_object* v_a_852_, lean_object* v_b_853_){
_start:
{
lean_object* v___x_854_; uint8_t v___x_855_; 
v___x_854_ = lean_apply_2(v_h_850_, v_a_852_, v_b_853_);
v___x_855_ = lean_unbox(v___x_854_);
return v___x_855_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_decidableLT___boxed(lean_object* v_00_u03b1_856_, lean_object* v_inst_857_, lean_object* v_h_858_, lean_object* v_p_859_, lean_object* v_a_860_, lean_object* v_b_861_){
_start:
{
uint8_t v_res_862_; lean_object* v_r_863_; 
v_res_862_ = lp_mathlib_Subtype_decidableLT(v_00_u03b1_856_, v_inst_857_, v_h_858_, v_p_859_, v_a_860_, v_b_861_);
lean_dec_ref(v_inst_857_);
v_r_863_ = lean_box(v_res_862_);
return v_r_863_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Subtype_instLinearOrder___redArg___lam__0(lean_object* v_toDecidableLE_864_, lean_object* v_a_865_, lean_object* v_b_866_){
_start:
{
lean_object* v___x_867_; uint8_t v___x_868_; 
v___x_867_ = lean_apply_2(v_toDecidableLE_864_, v_a_865_, v_b_866_);
v___x_868_ = lean_unbox(v___x_867_);
return v___x_868_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLinearOrder___redArg___lam__0___boxed(lean_object* v_toDecidableLE_869_, lean_object* v_a_870_, lean_object* v_b_871_){
_start:
{
uint8_t v_res_872_; lean_object* v_r_873_; 
v_res_872_ = lp_mathlib_Subtype_instLinearOrder___redArg___lam__0(v_toDecidableLE_869_, v_a_870_, v_b_871_);
v_r_873_ = lean_box(v_res_872_);
return v_r_873_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Subtype_instLinearOrder___redArg___lam__1(lean_object* v_toDecidableEq_874_, lean_object* v_a_875_, lean_object* v_b_876_){
_start:
{
lean_object* v___x_877_; uint8_t v___x_878_; 
v___x_877_ = lean_apply_2(v_toDecidableEq_874_, v_a_875_, v_b_876_);
v___x_878_ = lean_unbox(v___x_877_);
return v___x_878_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLinearOrder___redArg___lam__1___boxed(lean_object* v_toDecidableEq_879_, lean_object* v_a_880_, lean_object* v_b_881_){
_start:
{
uint8_t v_res_882_; lean_object* v_r_883_; 
v_res_882_ = lp_mathlib_Subtype_instLinearOrder___redArg___lam__1(v_toDecidableEq_879_, v_a_880_, v_b_881_);
v_r_883_ = lean_box(v_res_882_);
return v_r_883_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Subtype_instLinearOrder___redArg___lam__2(lean_object* v_toDecidableLT_884_, lean_object* v_a_885_, lean_object* v_b_886_){
_start:
{
lean_object* v___x_887_; uint8_t v___x_888_; 
v___x_887_ = lean_apply_2(v_toDecidableLT_884_, v_a_885_, v_b_886_);
v___x_888_ = lean_unbox(v___x_887_);
return v___x_888_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLinearOrder___redArg___lam__2___boxed(lean_object* v_toDecidableLT_889_, lean_object* v_a_890_, lean_object* v_b_891_){
_start:
{
uint8_t v_res_892_; lean_object* v_r_893_; 
v_res_892_ = lp_mathlib_Subtype_instLinearOrder___redArg___lam__2(v_toDecidableLT_889_, v_a_890_, v_b_891_);
v_r_893_ = lean_box(v_res_892_);
return v_r_893_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLinearOrder___redArg___lam__3(lean_object* v_toMax_894_, lean_object* v_a_895_, lean_object* v_a_896_){
_start:
{
lean_object* v___x_897_; 
v___x_897_ = lean_apply_2(v_toMax_894_, v_a_895_, v_a_896_);
return v___x_897_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLinearOrder___redArg___lam__4(lean_object* v_toMin_898_, lean_object* v_a_899_, lean_object* v_a_900_){
_start:
{
lean_object* v___x_901_; 
v___x_901_ = lean_apply_2(v_toMin_898_, v_a_899_, v_a_900_);
return v___x_901_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLinearOrder___redArg(lean_object* v_inst_902_){
_start:
{
lean_object* v_toPartialOrder_903_; lean_object* v_toMin_904_; lean_object* v_toMax_905_; lean_object* v_toOrd_906_; lean_object* v_toDecidableLE_907_; lean_object* v_toDecidableEq_908_; lean_object* v_toDecidableLT_909_; lean_object* v___x_911_; uint8_t v_isShared_912_; uint8_t v_isSharedCheck_923_; 
v_toPartialOrder_903_ = lean_ctor_get(v_inst_902_, 0);
v_toMin_904_ = lean_ctor_get(v_inst_902_, 1);
v_toMax_905_ = lean_ctor_get(v_inst_902_, 2);
v_toOrd_906_ = lean_ctor_get(v_inst_902_, 3);
v_toDecidableLE_907_ = lean_ctor_get(v_inst_902_, 4);
v_toDecidableEq_908_ = lean_ctor_get(v_inst_902_, 5);
v_toDecidableLT_909_ = lean_ctor_get(v_inst_902_, 6);
v_isSharedCheck_923_ = !lean_is_exclusive(v_inst_902_);
if (v_isSharedCheck_923_ == 0)
{
v___x_911_ = v_inst_902_;
v_isShared_912_ = v_isSharedCheck_923_;
goto v_resetjp_910_;
}
else
{
lean_inc(v_toDecidableLT_909_);
lean_inc(v_toDecidableEq_908_);
lean_inc(v_toDecidableLE_907_);
lean_inc(v_toOrd_906_);
lean_inc(v_toMax_905_);
lean_inc(v_toMin_904_);
lean_inc(v_toPartialOrder_903_);
lean_dec(v_inst_902_);
v___x_911_ = lean_box(0);
v_isShared_912_ = v_isSharedCheck_923_;
goto v_resetjp_910_;
}
v_resetjp_910_:
{
lean_object* v___f_913_; lean_object* v___f_914_; lean_object* v___f_915_; lean_object* v___f_916_; lean_object* v___f_917_; lean_object* v___x_918_; lean_object* v___f_919_; lean_object* v___x_921_; 
v___f_913_ = lean_alloc_closure((void*)(lp_mathlib_Subtype_instLinearOrder___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_913_, 0, v_toDecidableLE_907_);
v___f_914_ = lean_alloc_closure((void*)(lp_mathlib_Subtype_instLinearOrder___redArg___lam__1___boxed), 3, 1);
lean_closure_set(v___f_914_, 0, v_toDecidableEq_908_);
v___f_915_ = lean_alloc_closure((void*)(lp_mathlib_Subtype_instLinearOrder___redArg___lam__2___boxed), 3, 1);
lean_closure_set(v___f_915_, 0, v_toDecidableLT_909_);
v___f_916_ = lean_alloc_closure((void*)(lp_mathlib_Subtype_instLinearOrder___redArg___lam__3), 3, 1);
lean_closure_set(v___f_916_, 0, v_toMax_905_);
v___f_917_ = lean_alloc_closure((void*)(lp_mathlib_Subtype_instLinearOrder___redArg___lam__4), 3, 1);
lean_closure_set(v___f_917_, 0, v_toMin_904_);
v___x_918_ = lp_mathlib_Subtype_preorder(lean_box(0), v_toPartialOrder_903_, lean_box(0));
lean_dec_ref(v_toPartialOrder_903_);
v___f_919_ = lean_alloc_closure((void*)(l_instOrdSubtype___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_919_, 0, v_toOrd_906_);
if (v_isShared_912_ == 0)
{
lean_ctor_set(v___x_911_, 6, v___f_915_);
lean_ctor_set(v___x_911_, 5, v___f_914_);
lean_ctor_set(v___x_911_, 4, v___f_913_);
lean_ctor_set(v___x_911_, 3, v___f_919_);
lean_ctor_set(v___x_911_, 2, v___f_916_);
lean_ctor_set(v___x_911_, 1, v___f_917_);
lean_ctor_set(v___x_911_, 0, v___x_918_);
v___x_921_ = v___x_911_;
goto v_reusejp_920_;
}
else
{
lean_object* v_reuseFailAlloc_922_; 
v_reuseFailAlloc_922_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v_reuseFailAlloc_922_, 0, v___x_918_);
lean_ctor_set(v_reuseFailAlloc_922_, 1, v___f_917_);
lean_ctor_set(v_reuseFailAlloc_922_, 2, v___f_916_);
lean_ctor_set(v_reuseFailAlloc_922_, 3, v___f_919_);
lean_ctor_set(v_reuseFailAlloc_922_, 4, v___f_913_);
lean_ctor_set(v_reuseFailAlloc_922_, 5, v___f_914_);
lean_ctor_set(v_reuseFailAlloc_922_, 6, v___f_915_);
v___x_921_ = v_reuseFailAlloc_922_;
goto v_reusejp_920_;
}
v_reusejp_920_:
{
return v___x_921_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLinearOrder(lean_object* v_00_u03b1_924_, lean_object* v_inst_925_, lean_object* v_p_926_){
_start:
{
lean_object* v___x_927_; 
v___x_927_ = lp_mathlib_Subtype_instLinearOrder___redArg(v_inst_925_);
return v___x_927_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLE__mathlib(lean_object* v_00_u03b1_928_, lean_object* v_00_u03b2_929_, lean_object* v_inst_930_, lean_object* v_inst_931_){
_start:
{
lean_object* v___x_932_; 
v___x_932_ = lean_box(0);
return v___x_932_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Prod_instDecidableLE___redArg(uint8_t v_inst_933_, uint8_t v_inst_934_){
_start:
{
if (v_inst_933_ == 0)
{
return v_inst_933_;
}
else
{
return v_inst_934_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instDecidableLE___redArg___boxed(lean_object* v_inst_935_, lean_object* v_inst_936_){
_start:
{
uint8_t v_inst_39__boxed_937_; uint8_t v_inst_40__boxed_938_; uint8_t v_res_939_; lean_object* v_r_940_; 
v_inst_39__boxed_937_ = lean_unbox(v_inst_935_);
v_inst_40__boxed_938_ = lean_unbox(v_inst_936_);
v_res_939_ = lp_mathlib_Prod_instDecidableLE___redArg(v_inst_39__boxed_937_, v_inst_40__boxed_938_);
v_r_940_ = lean_box(v_res_939_);
return v_r_940_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Prod_instDecidableLE(lean_object* v_00_u03b1_941_, lean_object* v_00_u03b2_942_, lean_object* v_inst_943_, lean_object* v_inst_944_, lean_object* v_x_945_, lean_object* v_y_946_, uint8_t v_inst_947_, uint8_t v_inst_948_){
_start:
{
if (v_inst_947_ == 0)
{
return v_inst_947_;
}
else
{
return v_inst_948_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instDecidableLE___boxed(lean_object* v_00_u03b1_949_, lean_object* v_00_u03b2_950_, lean_object* v_inst_951_, lean_object* v_inst_952_, lean_object* v_x_953_, lean_object* v_y_954_, lean_object* v_inst_955_, lean_object* v_inst_956_){
_start:
{
uint8_t v_inst_47__boxed_957_; uint8_t v_inst_48__boxed_958_; uint8_t v_res_959_; lean_object* v_r_960_; 
v_inst_47__boxed_957_ = lean_unbox(v_inst_955_);
v_inst_48__boxed_958_ = lean_unbox(v_inst_956_);
v_res_959_ = lp_mathlib_Prod_instDecidableLE(v_00_u03b1_949_, v_00_u03b2_950_, v_inst_951_, v_inst_952_, v_x_953_, v_y_954_, v_inst_47__boxed_957_, v_inst_48__boxed_958_);
lean_dec_ref(v_y_954_);
lean_dec_ref(v_x_953_);
v_r_960_ = lean_box(v_res_959_);
return v_r_960_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instPreorder(lean_object* v_00_u03b1_964_, lean_object* v_00_u03b2_965_, lean_object* v_inst_966_, lean_object* v_inst_967_){
_start:
{
lean_object* v___x_968_; 
v___x_968_ = ((lean_object*)(lp_mathlib_Prod_instPreorder___closed__0));
return v___x_968_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instPreorder___boxed(lean_object* v_00_u03b1_969_, lean_object* v_00_u03b2_970_, lean_object* v_inst_971_, lean_object* v_inst_972_){
_start:
{
lean_object* v_res_973_; 
v_res_973_ = lp_mathlib_Prod_instPreorder(v_00_u03b1_969_, v_00_u03b2_970_, v_inst_971_, v_inst_972_);
lean_dec_ref(v_inst_972_);
lean_dec_ref(v_inst_971_);
return v_res_973_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instPartialOrder___redArg(lean_object* v_inst_974_, lean_object* v_inst_975_){
_start:
{
lean_object* v___x_976_; 
v___x_976_ = lp_mathlib_Prod_instPreorder(lean_box(0), lean_box(0), v_inst_974_, v_inst_975_);
return v___x_976_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instPartialOrder___redArg___boxed(lean_object* v_inst_977_, lean_object* v_inst_978_){
_start:
{
lean_object* v_res_979_; 
v_res_979_ = lp_mathlib_Prod_instPartialOrder___redArg(v_inst_977_, v_inst_978_);
lean_dec_ref(v_inst_978_);
lean_dec_ref(v_inst_977_);
return v_res_979_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instPartialOrder(lean_object* v_00_u03b1_980_, lean_object* v_00_u03b2_981_, lean_object* v_inst_982_, lean_object* v_inst_983_){
_start:
{
lean_object* v___x_984_; 
v___x_984_ = lp_mathlib_Prod_instPreorder(lean_box(0), lean_box(0), v_inst_982_, v_inst_983_);
return v___x_984_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instPartialOrder___boxed(lean_object* v_00_u03b1_985_, lean_object* v_00_u03b2_986_, lean_object* v_inst_987_, lean_object* v_inst_988_){
_start:
{
lean_object* v_res_989_; 
v_res_989_ = lp_mathlib_Prod_instPartialOrder(v_00_u03b1_985_, v_00_u03b2_986_, v_inst_987_, v_inst_988_);
lean_dec_ref(v_inst_988_);
lean_dec_ref(v_inst_987_);
return v_res_989_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_ofSubsingleton___lam__0(lean_object* v_a_990_, lean_object* v_b_991_){
_start:
{
lean_inc(v_a_990_);
return v_a_990_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_ofSubsingleton___lam__0___boxed(lean_object* v_a_992_, lean_object* v_b_993_){
_start:
{
lean_object* v_res_994_; 
v_res_994_ = lp_mathlib_LinearOrder_ofSubsingleton___lam__0(v_a_992_, v_b_993_);
lean_dec(v_b_993_);
lean_dec(v_a_992_);
return v_res_994_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_ofSubsingleton___lam__1(lean_object* v_a_995_, lean_object* v_b_996_){
_start:
{
lean_inc(v_b_996_);
return v_b_996_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_ofSubsingleton___lam__1___boxed(lean_object* v_a_997_, lean_object* v_b_998_){
_start:
{
lean_object* v_res_999_; 
v_res_999_ = lp_mathlib_LinearOrder_ofSubsingleton___lam__1(v_a_997_, v_b_998_);
lean_dec(v_b_998_);
lean_dec(v_a_997_);
return v_res_999_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_LinearOrder_ofSubsingleton___lam__2(uint8_t v___x_1000_, lean_object* v_x_1001_, lean_object* v_x_1002_){
_start:
{
return v___x_1000_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_ofSubsingleton___lam__2___boxed(lean_object* v___x_1003_, lean_object* v_x_1004_, lean_object* v_x_1005_){
_start:
{
uint8_t v___x_64__boxed_1006_; uint8_t v_res_1007_; lean_object* v_r_1008_; 
v___x_64__boxed_1006_ = lean_unbox(v___x_1003_);
v_res_1007_ = lp_mathlib_LinearOrder_ofSubsingleton___lam__2(v___x_64__boxed_1006_, v_x_1004_, v_x_1005_);
lean_dec(v_x_1005_);
lean_dec(v_x_1004_);
v_r_1008_ = lean_box(v_res_1007_);
return v_r_1008_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_LinearOrder_ofSubsingleton___lam__4(lean_object* v___f_1009_, lean_object* v_a_1010_, lean_object* v_b_1011_){
_start:
{
uint8_t v___x_1012_; 
lean_inc(v_b_1011_);
lean_inc(v_a_1010_);
lean_inc_ref(v___f_1009_);
v___x_1012_ = lp_mathlib_decidableLTOfDecidableLE___redArg(v___f_1009_, v_a_1010_, v_b_1011_);
if (v___x_1012_ == 0)
{
uint8_t v___x_1013_; 
v___x_1013_ = lp_mathlib_decidableEqOfDecidableLE___redArg(v___f_1009_, v_a_1010_, v_b_1011_);
if (v___x_1013_ == 0)
{
uint8_t v___x_1014_; 
v___x_1014_ = 2;
return v___x_1014_;
}
else
{
uint8_t v___x_1015_; 
v___x_1015_ = 1;
return v___x_1015_;
}
}
else
{
uint8_t v___x_1016_; 
lean_dec(v_b_1011_);
lean_dec(v_a_1010_);
lean_dec_ref(v___f_1009_);
v___x_1016_ = 0;
return v___x_1016_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_ofSubsingleton___lam__4___boxed(lean_object* v___f_1017_, lean_object* v_a_1018_, lean_object* v_b_1019_){
_start:
{
uint8_t v_res_1020_; lean_object* v_r_1021_; 
v_res_1020_ = lp_mathlib_LinearOrder_ofSubsingleton___lam__4(v___f_1017_, v_a_1018_, v_b_1019_);
v_r_1021_ = lean_box(v_res_1020_);
return v_r_1021_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_ofSubsingleton(lean_object* v_00_u03b1_1043_, lean_object* v_inst_1044_){
_start:
{
lean_object* v___x_1045_; 
v___x_1045_ = ((lean_object*)(lp_mathlib_LinearOrder_ofSubsingleton___closed__6));
return v___x_1045_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instLinearOrderEmpty___lam__0(uint8_t v_a_1046_, uint8_t v_b_1047_){
_start:
{
return v_a_1046_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderEmpty___lam__0___boxed(lean_object* v_a_1048_, lean_object* v_b_1049_){
_start:
{
uint8_t v_a_boxed_1050_; uint8_t v_b_boxed_1051_; uint8_t v_res_1052_; lean_object* v_r_1053_; 
v_a_boxed_1050_ = lean_unbox(v_a_1048_);
v_b_boxed_1051_ = lean_unbox(v_b_1049_);
v_res_1052_ = lp_mathlib_instLinearOrderEmpty___lam__0(v_a_boxed_1050_, v_b_boxed_1051_);
v_r_1053_ = lean_box(v_res_1052_);
return v_r_1053_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instLinearOrderEmpty___lam__1(uint8_t v_a_1054_, uint8_t v_b_1055_){
_start:
{
return v_b_1055_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderEmpty___lam__1___boxed(lean_object* v_a_1056_, lean_object* v_b_1057_){
_start:
{
uint8_t v_a_boxed_1058_; uint8_t v_b_boxed_1059_; uint8_t v_res_1060_; lean_object* v_r_1061_; 
v_a_boxed_1058_ = lean_unbox(v_a_1056_);
v_b_boxed_1059_ = lean_unbox(v_b_1057_);
v_res_1060_ = lp_mathlib_instLinearOrderEmpty___lam__1(v_a_boxed_1058_, v_b_boxed_1059_);
v_r_1061_ = lean_box(v_res_1060_);
return v_r_1061_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instLinearOrderEmpty___lam__2(uint8_t v___x_1062_, uint8_t v_x_1063_, uint8_t v_x_1064_){
_start:
{
return v___x_1062_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderEmpty___lam__2___boxed(lean_object* v___x_1065_, lean_object* v_x_1066_, lean_object* v_x_1067_){
_start:
{
uint8_t v___x_48__boxed_1068_; uint8_t v_x_49__boxed_1069_; uint8_t v_x_50__boxed_1070_; uint8_t v_res_1071_; lean_object* v_r_1072_; 
v___x_48__boxed_1068_ = lean_unbox(v___x_1065_);
v_x_49__boxed_1069_ = lean_unbox(v_x_1066_);
v_x_50__boxed_1070_ = lean_unbox(v_x_1067_);
v_res_1071_ = lp_mathlib_instLinearOrderEmpty___lam__2(v___x_48__boxed_1068_, v_x_49__boxed_1069_, v_x_50__boxed_1070_);
v_r_1072_ = lean_box(v_res_1071_);
return v_r_1072_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instLinearOrderEmpty___lam__4(lean_object* v___f_1073_, uint8_t v_a_1074_, uint8_t v_b_1075_){
_start:
{
lean_object* v___x_1076_; lean_object* v___x_1077_; uint8_t v___x_1078_; 
v___x_1076_ = lean_box(v_a_1074_);
v___x_1077_ = lean_box(v_b_1075_);
lean_inc_ref(v___f_1073_);
v___x_1078_ = lp_mathlib_decidableLTOfDecidableLE___redArg(v___f_1073_, v___x_1076_, v___x_1077_);
if (v___x_1078_ == 0)
{
lean_object* v___x_1079_; lean_object* v___x_1080_; uint8_t v___x_1081_; 
v___x_1079_ = lean_box(v_a_1074_);
v___x_1080_ = lean_box(v_b_1075_);
v___x_1081_ = lp_mathlib_decidableEqOfDecidableLE___redArg(v___f_1073_, v___x_1079_, v___x_1080_);
if (v___x_1081_ == 0)
{
uint8_t v___x_1082_; 
v___x_1082_ = 2;
return v___x_1082_;
}
else
{
uint8_t v___x_1083_; 
v___x_1083_ = 1;
return v___x_1083_;
}
}
else
{
uint8_t v___x_1084_; 
lean_dec_ref(v___f_1073_);
v___x_1084_ = 0;
return v___x_1084_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderEmpty___lam__4___boxed(lean_object* v___f_1085_, lean_object* v_a_1086_, lean_object* v_b_1087_){
_start:
{
uint8_t v_a_boxed_1088_; uint8_t v_b_boxed_1089_; uint8_t v_res_1090_; lean_object* v_r_1091_; 
v_a_boxed_1088_ = lean_unbox(v_a_1086_);
v_b_boxed_1089_ = lean_unbox(v_b_1087_);
v_res_1090_ = lp_mathlib_instLinearOrderEmpty___lam__4(v___f_1085_, v_a_boxed_1088_, v_b_boxed_1089_);
v_r_1091_ = lean_box(v_res_1090_);
return v_r_1091_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instLinearOrderPEmpty___lam__0(uint8_t v_a_1117_, uint8_t v_b_1118_){
_start:
{
return v_a_1117_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderPEmpty___lam__0___boxed(lean_object* v_a_1119_, lean_object* v_b_1120_){
_start:
{
uint8_t v_a_boxed_1121_; uint8_t v_b_boxed_1122_; uint8_t v_res_1123_; lean_object* v_r_1124_; 
v_a_boxed_1121_ = lean_unbox(v_a_1119_);
v_b_boxed_1122_ = lean_unbox(v_b_1120_);
v_res_1123_ = lp_mathlib_instLinearOrderPEmpty___lam__0(v_a_boxed_1121_, v_b_boxed_1122_);
v_r_1124_ = lean_box(v_res_1123_);
return v_r_1124_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instLinearOrderPEmpty___lam__1(uint8_t v_a_1125_, uint8_t v_b_1126_){
_start:
{
return v_b_1126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderPEmpty___lam__1___boxed(lean_object* v_a_1127_, lean_object* v_b_1128_){
_start:
{
uint8_t v_a_boxed_1129_; uint8_t v_b_boxed_1130_; uint8_t v_res_1131_; lean_object* v_r_1132_; 
v_a_boxed_1129_ = lean_unbox(v_a_1127_);
v_b_boxed_1130_ = lean_unbox(v_b_1128_);
v_res_1131_ = lp_mathlib_instLinearOrderPEmpty___lam__1(v_a_boxed_1129_, v_b_boxed_1130_);
v_r_1132_ = lean_box(v_res_1131_);
return v_r_1132_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instLinearOrderPEmpty___lam__2(uint8_t v___x_1133_, uint8_t v_x_1134_, uint8_t v_x_1135_){
_start:
{
return v___x_1133_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderPEmpty___lam__2___boxed(lean_object* v___x_1136_, lean_object* v_x_1137_, lean_object* v_x_1138_){
_start:
{
uint8_t v___x_48__boxed_1139_; uint8_t v_x_49__boxed_1140_; uint8_t v_x_50__boxed_1141_; uint8_t v_res_1142_; lean_object* v_r_1143_; 
v___x_48__boxed_1139_ = lean_unbox(v___x_1136_);
v_x_49__boxed_1140_ = lean_unbox(v_x_1137_);
v_x_50__boxed_1141_ = lean_unbox(v_x_1138_);
v_res_1142_ = lp_mathlib_instLinearOrderPEmpty___lam__2(v___x_48__boxed_1139_, v_x_49__boxed_1140_, v_x_50__boxed_1141_);
v_r_1143_ = lean_box(v_res_1142_);
return v_r_1143_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instLinearOrderPEmpty___lam__4(lean_object* v___f_1144_, uint8_t v_a_1145_, uint8_t v_b_1146_){
_start:
{
lean_object* v___x_1147_; lean_object* v___x_1148_; uint8_t v___x_1149_; 
v___x_1147_ = lean_box(v_a_1145_);
v___x_1148_ = lean_box(v_b_1146_);
lean_inc_ref(v___f_1144_);
v___x_1149_ = lp_mathlib_decidableLTOfDecidableLE___redArg(v___f_1144_, v___x_1147_, v___x_1148_);
if (v___x_1149_ == 0)
{
lean_object* v___x_1150_; lean_object* v___x_1151_; uint8_t v___x_1152_; 
v___x_1150_ = lean_box(v_a_1145_);
v___x_1151_ = lean_box(v_b_1146_);
v___x_1152_ = lp_mathlib_decidableEqOfDecidableLE___redArg(v___f_1144_, v___x_1150_, v___x_1151_);
if (v___x_1152_ == 0)
{
uint8_t v___x_1153_; 
v___x_1153_ = 2;
return v___x_1153_;
}
else
{
uint8_t v___x_1154_; 
v___x_1154_ = 1;
return v___x_1154_;
}
}
else
{
uint8_t v___x_1155_; 
lean_dec_ref(v___f_1144_);
v___x_1155_ = 0;
return v___x_1155_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderPEmpty___lam__4___boxed(lean_object* v___f_1156_, lean_object* v_a_1157_, lean_object* v_b_1158_){
_start:
{
uint8_t v_a_boxed_1159_; uint8_t v_b_boxed_1160_; uint8_t v_res_1161_; lean_object* v_r_1162_; 
v_a_boxed_1159_ = lean_unbox(v_a_1157_);
v_b_boxed_1160_ = lean_unbox(v_b_1158_);
v_res_1161_ = lp_mathlib_instLinearOrderPEmpty___lam__4(v___f_1156_, v_a_boxed_1159_, v_b_boxed_1160_);
v_r_1162_ = lean_box(v_res_1161_);
return v_r_1162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit_instLinearOrder___lam__0(lean_object* v_a_1188_, lean_object* v_b_1189_){
_start:
{
return v_a_1188_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit_instLinearOrder___lam__1(lean_object* v_a_1190_, lean_object* v_b_1191_){
_start:
{
return v_b_1191_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_PUnit_instLinearOrder___lam__2(uint8_t v___x_1192_, lean_object* v_x_1193_, lean_object* v_x_1194_){
_start:
{
return v___x_1192_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit_instLinearOrder___lam__2___boxed(lean_object* v___x_1195_, lean_object* v_x_1196_, lean_object* v_x_1197_){
_start:
{
uint8_t v___x_48__boxed_1198_; uint8_t v_res_1199_; lean_object* v_r_1200_; 
v___x_48__boxed_1198_ = lean_unbox(v___x_1195_);
v_res_1199_ = lp_mathlib_PUnit_instLinearOrder___lam__2(v___x_48__boxed_1198_, v_x_1196_, v_x_1197_);
v_r_1200_ = lean_box(v_res_1199_);
return v_r_1200_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_PUnit_instLinearOrder___lam__4(lean_object* v___f_1201_, lean_object* v_a_1202_, lean_object* v_b_1203_){
_start:
{
uint8_t v___x_1204_; 
lean_inc_ref(v___f_1201_);
v___x_1204_ = lp_mathlib_decidableLTOfDecidableLE___redArg(v___f_1201_, v_a_1202_, v_b_1203_);
if (v___x_1204_ == 0)
{
uint8_t v___x_1205_; 
v___x_1205_ = lp_mathlib_decidableEqOfDecidableLE___redArg(v___f_1201_, v_a_1202_, v_b_1203_);
if (v___x_1205_ == 0)
{
uint8_t v___x_1206_; 
v___x_1206_ = 2;
return v___x_1206_;
}
else
{
uint8_t v___x_1207_; 
v___x_1207_ = 1;
return v___x_1207_;
}
}
else
{
uint8_t v___x_1208_; 
lean_dec_ref(v___f_1201_);
v___x_1208_ = 0;
return v___x_1208_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit_instLinearOrder___lam__4___boxed(lean_object* v___f_1209_, lean_object* v_a_1210_, lean_object* v_b_1211_){
_start:
{
uint8_t v_res_1212_; lean_object* v_r_1213_; 
v_res_1212_ = lp_mathlib_PUnit_instLinearOrder___lam__4(v___f_1209_, v_a_1210_, v_b_1211_);
v_r_1213_ = lean_box(v_res_1212_);
return v_r_1213_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Subtype(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Defs_LinearOrder(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Defs_Prop(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Notation(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Spread(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Convert(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Inhabit(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_SimpRw(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_GCongr(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Attr_Register(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_FastInstance(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Subtype(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Defs_LinearOrder(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Defs_Prop(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Spread(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Convert(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Inhabit(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_SimpRw(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_GCongr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Attr_Register(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_FastInstance(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Prop_instCompl = _init_lp_mathlib_Prop_instCompl();
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_Basic(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Tactic_GCongr_exactLeOfLt = _init_lp_mathlib_Mathlib_Tactic_GCongr_exactLeOfLt();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_GCongr_exactLeOfLt);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Subtype(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Defs_LinearOrder(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Defs_Prop(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Notation(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Spread(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Convert(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Inhabit(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_SimpRw(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_GCongr(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Attr_Register(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_FastInstance(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Subtype(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Defs_LinearOrder(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Defs_Prop(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Spread(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Convert(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Inhabit(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_SimpRw(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_GCongr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Attr_Register(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_FastInstance(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
