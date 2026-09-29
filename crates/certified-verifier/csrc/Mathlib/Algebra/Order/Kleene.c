// Lean compiler output
// Module: Mathlib.Algebra.Order.Kleene
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Order.Monoid.Canonical.Defs public import Mathlib.Algebra.Ring.InjSurj public import Mathlib.Algebra.Ring.Pi public import Mathlib.Algebra.Ring.Prod public import Mathlib.Tactic.Monotonicity.Attr
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
lean_object* l_Nat_cast(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Injective_addMonoid___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkAtom(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lp_mathlib_Prod_instSemiring___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Prod_instSemilatticeSup___redArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* lp_mathlib_Pi_semiring___redArg(lean_object*);
lean_object* lp_mathlib_Pi_instSemilatticeSup___redArg(lean_object*);
lean_object* lp_mathlib_Pi_instOrderBot___redArg(lean_object*);
lean_object* lp_mathlib_Pi_commSemiring___redArg(lean_object*);
lean_object* lp_mathlib_instDistribOfSemiring___redArg(lean_object*);
lean_object* lp_mathlib_instMulZeroClassOfSemiring___redArg(lean_object*);
static const lean_string_object lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__0 = (const lean_object*)&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__0_value;
static const lean_string_object lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__1 = (const lean_object*)&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__1_value;
static const lean_string_object lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__2 = (const lean_object*)&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__2_value;
static const lean_string_object lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__3 = (const lean_object*)&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__3_value;
static const lean_ctor_object lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__4_value_aux_0),((lean_object*)&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__4_value_aux_1),((lean_object*)&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__4_value_aux_2),((lean_object*)&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__3_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__4 = (const lean_object*)&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__4_value;
static const lean_array_object lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__5 = (const lean_object*)&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__5_value;
static const lean_string_object lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__6 = (const lean_object*)&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__6_value;
static const lean_ctor_object lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__7_value_aux_0),((lean_object*)&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__7_value_aux_1),((lean_object*)&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__7_value_aux_2),((lean_object*)&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__6_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__7 = (const lean_object*)&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__7_value;
static const lean_string_object lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__8 = (const lean_object*)&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__8_value;
static const lean_ctor_object lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__8_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__9 = (const lean_object*)&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__9_value;
static const lean_string_object lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "intros"};
static const lean_object* lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__10 = (const lean_object*)&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__10_value;
static const lean_ctor_object lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__11_value_aux_0),((lean_object*)&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__11_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__11_value_aux_1),((lean_object*)&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__11_value_aux_2),((lean_object*)&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__10_value),LEAN_SCALAR_PTR_LITERAL(26, 175, 18, 116, 252, 50, 128, 45)}};
static const lean_object* lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__11 = (const lean_object*)&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__11_value;
static lean_once_cell_t lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__12;
static lean_once_cell_t lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__13;
static const lean_ctor_object lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__9_value),((lean_object*)&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__5_value)}};
static const lean_object* lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__14 = (const lean_object*)&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__14_value;
static lean_once_cell_t lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__15;
static lean_once_cell_t lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__16;
static lean_once_cell_t lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__17;
static const lean_string_object lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ";"};
static const lean_object* lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__18 = (const lean_object*)&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__18_value;
static lean_once_cell_t lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__19;
static lean_once_cell_t lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__20;
static const lean_string_object lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticRfl"};
static const lean_object* lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__21 = (const lean_object*)&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__21_value;
static const lean_ctor_object lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__22_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__22_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__22_value_aux_0),((lean_object*)&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__22_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__22_value_aux_1),((lean_object*)&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__22_value_aux_2),((lean_object*)&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__21_value),LEAN_SCALAR_PTR_LITERAL(201, 188, 173, 198, 169, 252, 183, 45)}};
static const lean_object* lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__22 = (const lean_object*)&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__22_value;
static const lean_string_object lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "rfl"};
static const lean_object* lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__23 = (const lean_object*)&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__23_value;
static lean_once_cell_t lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__24;
static lean_once_cell_t lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__25_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__25;
static lean_once_cell_t lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__26_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__26;
static lean_once_cell_t lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__27_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__27;
static lean_once_cell_t lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__28_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__28;
static lean_once_cell_t lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__29_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__29;
static lean_once_cell_t lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__30_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__30;
static lean_once_cell_t lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__31_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__31;
static lean_once_cell_t lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__32_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__32;
LEAN_EXPORT lean_object* lp_mathlib_IdemSemiring_add__eq__sup___autoParam;
LEAN_EXPORT lean_object* lp_mathlib_IdemCommSemiring_toIdemSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IdemCommSemiring_toIdemSemiring(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Computability_term___u2217___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "Computability"};
static const lean_object* lp_mathlib_Computability_term___u2217___closed__0 = (const lean_object*)&lp_mathlib_Computability_term___u2217___closed__0_value;
static const lean_string_object lp_mathlib_Computability_term___u2217___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 6, .m_data = "term_∗"};
static const lean_object* lp_mathlib_Computability_term___u2217___closed__1 = (const lean_object*)&lp_mathlib_Computability_term___u2217___closed__1_value;
static const lean_ctor_object lp_mathlib_Computability_term___u2217___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Computability_term___u2217___closed__0_value),LEAN_SCALAR_PTR_LITERAL(255, 107, 224, 199, 110, 65, 247, 71)}};
static const lean_ctor_object lp_mathlib_Computability_term___u2217___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Computability_term___u2217___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Computability_term___u2217___closed__1_value),LEAN_SCALAR_PTR_LITERAL(176, 88, 93, 189, 121, 85, 156, 201)}};
static const lean_object* lp_mathlib_Computability_term___u2217___closed__2 = (const lean_object*)&lp_mathlib_Computability_term___u2217___closed__2_value;
static const lean_string_object lp_mathlib_Computability_term___u2217___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "∗"};
static const lean_object* lp_mathlib_Computability_term___u2217___closed__3 = (const lean_object*)&lp_mathlib_Computability_term___u2217___closed__3_value;
static const lean_ctor_object lp_mathlib_Computability_term___u2217___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Computability_term___u2217___closed__3_value)}};
static const lean_object* lp_mathlib_Computability_term___u2217___closed__4 = (const lean_object*)&lp_mathlib_Computability_term___u2217___closed__4_value;
static const lean_ctor_object lp_mathlib_Computability_term___u2217___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_Computability_term___u2217___closed__2_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Computability_term___u2217___closed__4_value)}};
static const lean_object* lp_mathlib_Computability_term___u2217___closed__5 = (const lean_object*)&lp_mathlib_Computability_term___u2217___closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_Computability_term___u2217 = (const lean_object*)&lp_mathlib_Computability_term___u2217___closed__5_value;
static const lean_string_object lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______macroRules__Computability__term___u2217__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______macroRules__Computability__term___u2217__1___closed__0 = (const lean_object*)&lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______macroRules__Computability__term___u2217__1___closed__0_value;
static const lean_string_object lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______macroRules__Computability__term___u2217__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______macroRules__Computability__term___u2217__1___closed__1 = (const lean_object*)&lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______macroRules__Computability__term___u2217__1___closed__1_value;
static const lean_ctor_object lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______macroRules__Computability__term___u2217__1___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______macroRules__Computability__term___u2217__1___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______macroRules__Computability__term___u2217__1___closed__2_value_aux_0),((lean_object*)&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______macroRules__Computability__term___u2217__1___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______macroRules__Computability__term___u2217__1___closed__2_value_aux_1),((lean_object*)&lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______macroRules__Computability__term___u2217__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______macroRules__Computability__term___u2217__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______macroRules__Computability__term___u2217__1___closed__2_value_aux_2),((lean_object*)&lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______macroRules__Computability__term___u2217__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______macroRules__Computability__term___u2217__1___closed__2 = (const lean_object*)&lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______macroRules__Computability__term___u2217__1___closed__2_value;
static const lean_string_object lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______macroRules__Computability__term___u2217__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "KStar.kstar"};
static const lean_object* lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______macroRules__Computability__term___u2217__1___closed__3 = (const lean_object*)&lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______macroRules__Computability__term___u2217__1___closed__3_value;
static lean_once_cell_t lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______macroRules__Computability__term___u2217__1___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______macroRules__Computability__term___u2217__1___closed__4;
static const lean_string_object lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______macroRules__Computability__term___u2217__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "KStar"};
static const lean_object* lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______macroRules__Computability__term___u2217__1___closed__5 = (const lean_object*)&lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______macroRules__Computability__term___u2217__1___closed__5_value;
static const lean_string_object lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______macroRules__Computability__term___u2217__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "kstar"};
static const lean_object* lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______macroRules__Computability__term___u2217__1___closed__6 = (const lean_object*)&lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______macroRules__Computability__term___u2217__1___closed__6_value;
static const lean_ctor_object lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______macroRules__Computability__term___u2217__1___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______macroRules__Computability__term___u2217__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(199, 120, 173, 213, 168, 66, 129, 119)}};
static const lean_ctor_object lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______macroRules__Computability__term___u2217__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______macroRules__Computability__term___u2217__1___closed__7_value_aux_0),((lean_object*)&lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______macroRules__Computability__term___u2217__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(139, 79, 235, 142, 11, 153, 105, 42)}};
static const lean_object* lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______macroRules__Computability__term___u2217__1___closed__7 = (const lean_object*)&lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______macroRules__Computability__term___u2217__1___closed__7_value;
static const lean_ctor_object lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______macroRules__Computability__term___u2217__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______macroRules__Computability__term___u2217__1___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______macroRules__Computability__term___u2217__1___closed__8 = (const lean_object*)&lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______macroRules__Computability__term___u2217__1___closed__8_value;
static const lean_ctor_object lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______macroRules__Computability__term___u2217__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______macroRules__Computability__term___u2217__1___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______macroRules__Computability__term___u2217__1___closed__9 = (const lean_object*)&lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______macroRules__Computability__term___u2217__1___closed__9_value;
LEAN_EXPORT lean_object* lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______macroRules__Computability__term___u2217__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______macroRules__Computability__term___u2217__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______unexpand__KStar__kstar__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______unexpand__KStar__kstar__1___closed__0 = (const lean_object*)&lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______unexpand__KStar__kstar__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______unexpand__KStar__kstar__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______unexpand__KStar__kstar__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______unexpand__KStar__kstar__1___closed__1 = (const lean_object*)&lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______unexpand__KStar__kstar__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______unexpand__KStar__kstar__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______unexpand__KStar__kstar__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IdemSemiring_ofSemiring___redArg___lam__0(lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_IdemSemiring_ofSemiring___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_IdemSemiring_ofSemiring___redArg___closed__0 = (const lean_object*)&lp_mathlib_IdemSemiring_ofSemiring___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_IdemSemiring_ofSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IdemSemiring_ofSemiring(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instIdemSemiring___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instIdemSemiring(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instIdemCommSemiring___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instIdemCommSemiring(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instKleeneAlgebra___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instKleeneAlgebra___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instKleeneAlgebra(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instIdemSemiring___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instIdemSemiring___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instIdemSemiring___redArg___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instIdemSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instIdemSemiring(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instIdemCommSemiringForall___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instIdemCommSemiringForall___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instIdemCommSemiringForall___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instIdemCommSemiringForall(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instKleeneAlgebraForall___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instKleeneAlgebraForall___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instKleeneAlgebraForall___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instKleeneAlgebraForall(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_idemSemiring___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_idemSemiring___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_idemSemiring___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_idemSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_idemSemiring___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_idemCommSemiring___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_idemCommSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_idemCommSemiring___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_kleeneAlgebra___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_kleeneAlgebra(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_kleeneAlgebra___boxed(lean_object**);
static lean_object* _init_lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__12(void){
_start:
{
lean_object* v___x_27_; lean_object* v___x_28_; 
v___x_27_ = ((lean_object*)(lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__10));
v___x_28_ = l_Lean_mkAtom(v___x_27_);
return v___x_28_;
}
}
static lean_object* _init_lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__13(void){
_start:
{
lean_object* v___x_29_; lean_object* v___x_30_; lean_object* v___x_31_; 
v___x_29_ = lean_obj_once(&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__12, &lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__12_once, _init_lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__12);
v___x_30_ = ((lean_object*)(lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__5));
v___x_31_ = lean_array_push(v___x_30_, v___x_29_);
return v___x_31_;
}
}
static lean_object* _init_lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__15(void){
_start:
{
lean_object* v___x_36_; lean_object* v___x_37_; lean_object* v___x_38_; 
v___x_36_ = ((lean_object*)(lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__14));
v___x_37_ = lean_obj_once(&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__13, &lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__13_once, _init_lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__13);
v___x_38_ = lean_array_push(v___x_37_, v___x_36_);
return v___x_38_;
}
}
static lean_object* _init_lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__16(void){
_start:
{
lean_object* v___x_39_; lean_object* v___x_40_; lean_object* v___x_41_; lean_object* v___x_42_; 
v___x_39_ = lean_obj_once(&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__15, &lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__15_once, _init_lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__15);
v___x_40_ = ((lean_object*)(lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__11));
v___x_41_ = lean_box(2);
v___x_42_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_42_, 0, v___x_41_);
lean_ctor_set(v___x_42_, 1, v___x_40_);
lean_ctor_set(v___x_42_, 2, v___x_39_);
return v___x_42_;
}
}
static lean_object* _init_lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__17(void){
_start:
{
lean_object* v___x_43_; lean_object* v___x_44_; lean_object* v___x_45_; 
v___x_43_ = lean_obj_once(&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__16, &lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__16_once, _init_lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__16);
v___x_44_ = ((lean_object*)(lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__5));
v___x_45_ = lean_array_push(v___x_44_, v___x_43_);
return v___x_45_;
}
}
static lean_object* _init_lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__19(void){
_start:
{
lean_object* v___x_47_; lean_object* v___x_48_; 
v___x_47_ = ((lean_object*)(lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__18));
v___x_48_ = l_Lean_mkAtom(v___x_47_);
return v___x_48_;
}
}
static lean_object* _init_lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__20(void){
_start:
{
lean_object* v___x_49_; lean_object* v___x_50_; lean_object* v___x_51_; 
v___x_49_ = lean_obj_once(&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__19, &lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__19_once, _init_lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__19);
v___x_50_ = lean_obj_once(&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__17, &lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__17_once, _init_lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__17);
v___x_51_ = lean_array_push(v___x_50_, v___x_49_);
return v___x_51_;
}
}
static lean_object* _init_lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__24(void){
_start:
{
lean_object* v___x_59_; lean_object* v___x_60_; 
v___x_59_ = ((lean_object*)(lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__23));
v___x_60_ = l_Lean_mkAtom(v___x_59_);
return v___x_60_;
}
}
static lean_object* _init_lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__25(void){
_start:
{
lean_object* v___x_61_; lean_object* v___x_62_; lean_object* v___x_63_; 
v___x_61_ = lean_obj_once(&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__24, &lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__24_once, _init_lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__24);
v___x_62_ = ((lean_object*)(lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__5));
v___x_63_ = lean_array_push(v___x_62_, v___x_61_);
return v___x_63_;
}
}
static lean_object* _init_lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__26(void){
_start:
{
lean_object* v___x_64_; lean_object* v___x_65_; lean_object* v___x_66_; lean_object* v___x_67_; 
v___x_64_ = lean_obj_once(&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__25, &lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__25_once, _init_lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__25);
v___x_65_ = ((lean_object*)(lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__22));
v___x_66_ = lean_box(2);
v___x_67_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_67_, 0, v___x_66_);
lean_ctor_set(v___x_67_, 1, v___x_65_);
lean_ctor_set(v___x_67_, 2, v___x_64_);
return v___x_67_;
}
}
static lean_object* _init_lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__27(void){
_start:
{
lean_object* v___x_68_; lean_object* v___x_69_; lean_object* v___x_70_; 
v___x_68_ = lean_obj_once(&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__26, &lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__26_once, _init_lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__26);
v___x_69_ = lean_obj_once(&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__20, &lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__20_once, _init_lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__20);
v___x_70_ = lean_array_push(v___x_69_, v___x_68_);
return v___x_70_;
}
}
static lean_object* _init_lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__28(void){
_start:
{
lean_object* v___x_71_; lean_object* v___x_72_; lean_object* v___x_73_; lean_object* v___x_74_; 
v___x_71_ = lean_obj_once(&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__27, &lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__27_once, _init_lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__27);
v___x_72_ = ((lean_object*)(lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__9));
v___x_73_ = lean_box(2);
v___x_74_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_74_, 0, v___x_73_);
lean_ctor_set(v___x_74_, 1, v___x_72_);
lean_ctor_set(v___x_74_, 2, v___x_71_);
return v___x_74_;
}
}
static lean_object* _init_lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__29(void){
_start:
{
lean_object* v___x_75_; lean_object* v___x_76_; lean_object* v___x_77_; 
v___x_75_ = lean_obj_once(&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__28, &lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__28_once, _init_lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__28);
v___x_76_ = ((lean_object*)(lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__5));
v___x_77_ = lean_array_push(v___x_76_, v___x_75_);
return v___x_77_;
}
}
static lean_object* _init_lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__30(void){
_start:
{
lean_object* v___x_78_; lean_object* v___x_79_; lean_object* v___x_80_; lean_object* v___x_81_; 
v___x_78_ = lean_obj_once(&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__29, &lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__29_once, _init_lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__29);
v___x_79_ = ((lean_object*)(lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__7));
v___x_80_ = lean_box(2);
v___x_81_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_81_, 0, v___x_80_);
lean_ctor_set(v___x_81_, 1, v___x_79_);
lean_ctor_set(v___x_81_, 2, v___x_78_);
return v___x_81_;
}
}
static lean_object* _init_lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__31(void){
_start:
{
lean_object* v___x_82_; lean_object* v___x_83_; lean_object* v___x_84_; 
v___x_82_ = lean_obj_once(&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__30, &lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__30_once, _init_lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__30);
v___x_83_ = ((lean_object*)(lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__5));
v___x_84_ = lean_array_push(v___x_83_, v___x_82_);
return v___x_84_;
}
}
static lean_object* _init_lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__32(void){
_start:
{
lean_object* v___x_85_; lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; 
v___x_85_ = lean_obj_once(&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__31, &lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__31_once, _init_lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__31);
v___x_86_ = ((lean_object*)(lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__4));
v___x_87_ = lean_box(2);
v___x_88_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_88_, 0, v___x_87_);
lean_ctor_set(v___x_88_, 1, v___x_86_);
lean_ctor_set(v___x_88_, 2, v___x_85_);
return v___x_88_;
}
}
static lean_object* _init_lp_mathlib_IdemSemiring_add__eq__sup___autoParam(void){
_start:
{
lean_object* v___x_89_; 
v___x_89_ = lean_obj_once(&lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__32, &lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__32_once, _init_lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__32);
return v___x_89_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IdemCommSemiring_toIdemSemiring___redArg(lean_object* v_self_90_){
_start:
{
lean_object* v_toCommSemiring_91_; lean_object* v_toSemilatticeSup_92_; lean_object* v_toOrderBot_93_; lean_object* v___x_95_; uint8_t v_isShared_96_; uint8_t v_isSharedCheck_100_; 
v_toCommSemiring_91_ = lean_ctor_get(v_self_90_, 0);
v_toSemilatticeSup_92_ = lean_ctor_get(v_self_90_, 1);
v_toOrderBot_93_ = lean_ctor_get(v_self_90_, 2);
v_isSharedCheck_100_ = !lean_is_exclusive(v_self_90_);
if (v_isSharedCheck_100_ == 0)
{
v___x_95_ = v_self_90_;
v_isShared_96_ = v_isSharedCheck_100_;
goto v_resetjp_94_;
}
else
{
lean_inc(v_toOrderBot_93_);
lean_inc(v_toSemilatticeSup_92_);
lean_inc(v_toCommSemiring_91_);
lean_dec(v_self_90_);
v___x_95_ = lean_box(0);
v_isShared_96_ = v_isSharedCheck_100_;
goto v_resetjp_94_;
}
v_resetjp_94_:
{
lean_object* v___x_98_; 
if (v_isShared_96_ == 0)
{
v___x_98_ = v___x_95_;
goto v_reusejp_97_;
}
else
{
lean_object* v_reuseFailAlloc_99_; 
v_reuseFailAlloc_99_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_99_, 0, v_toCommSemiring_91_);
lean_ctor_set(v_reuseFailAlloc_99_, 1, v_toSemilatticeSup_92_);
lean_ctor_set(v_reuseFailAlloc_99_, 2, v_toOrderBot_93_);
v___x_98_ = v_reuseFailAlloc_99_;
goto v_reusejp_97_;
}
v_reusejp_97_:
{
return v___x_98_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_IdemCommSemiring_toIdemSemiring(lean_object* v_00_u03b1_101_, lean_object* v_self_102_){
_start:
{
lean_object* v___x_103_; 
v___x_103_ = lp_mathlib_IdemCommSemiring_toIdemSemiring___redArg(v_self_102_);
return v___x_103_;
}
}
static lean_object* _init_lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______macroRules__Computability__term___u2217__1___closed__4(void){
_start:
{
lean_object* v___x_125_; lean_object* v___x_126_; 
v___x_125_ = ((lean_object*)(lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______macroRules__Computability__term___u2217__1___closed__3));
v___x_126_ = l_String_toRawSubstring_x27(v___x_125_);
return v___x_126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______macroRules__Computability__term___u2217__1(lean_object* v_x_138_, lean_object* v_a_139_, lean_object* v_a_140_){
_start:
{
lean_object* v___x_141_; uint8_t v___x_142_; 
v___x_141_ = ((lean_object*)(lp_mathlib_Computability_term___u2217___closed__2));
lean_inc(v_x_138_);
v___x_142_ = l_Lean_Syntax_isOfKind(v_x_138_, v___x_141_);
if (v___x_142_ == 0)
{
lean_object* v___x_143_; lean_object* v___x_144_; 
lean_dec(v_x_138_);
v___x_143_ = lean_box(1);
v___x_144_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_144_, 0, v___x_143_);
lean_ctor_set(v___x_144_, 1, v_a_140_);
return v___x_144_;
}
else
{
lean_object* v_quotContext_145_; lean_object* v_currMacroScope_146_; lean_object* v_ref_147_; lean_object* v___x_148_; lean_object* v___x_149_; uint8_t v___x_150_; lean_object* v___x_151_; lean_object* v___x_152_; lean_object* v___x_153_; lean_object* v___x_154_; lean_object* v___x_155_; lean_object* v___x_156_; lean_object* v___x_157_; lean_object* v___x_158_; lean_object* v___x_159_; lean_object* v___x_160_; lean_object* v___x_161_; 
v_quotContext_145_ = lean_ctor_get(v_a_139_, 1);
v_currMacroScope_146_ = lean_ctor_get(v_a_139_, 2);
v_ref_147_ = lean_ctor_get(v_a_139_, 5);
v___x_148_ = lean_unsigned_to_nat(0u);
v___x_149_ = l_Lean_Syntax_getArg(v_x_138_, v___x_148_);
lean_dec(v_x_138_);
v___x_150_ = 0;
v___x_151_ = l_Lean_SourceInfo_fromRef(v_ref_147_, v___x_150_);
v___x_152_ = ((lean_object*)(lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______macroRules__Computability__term___u2217__1___closed__2));
v___x_153_ = lean_obj_once(&lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______macroRules__Computability__term___u2217__1___closed__4, &lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______macroRules__Computability__term___u2217__1___closed__4_once, _init_lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______macroRules__Computability__term___u2217__1___closed__4);
v___x_154_ = ((lean_object*)(lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______macroRules__Computability__term___u2217__1___closed__7));
lean_inc(v_currMacroScope_146_);
lean_inc(v_quotContext_145_);
v___x_155_ = l_Lean_addMacroScope(v_quotContext_145_, v___x_154_, v_currMacroScope_146_);
v___x_156_ = ((lean_object*)(lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______macroRules__Computability__term___u2217__1___closed__9));
lean_inc_n(v___x_151_, 2);
v___x_157_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_157_, 0, v___x_151_);
lean_ctor_set(v___x_157_, 1, v___x_153_);
lean_ctor_set(v___x_157_, 2, v___x_155_);
lean_ctor_set(v___x_157_, 3, v___x_156_);
v___x_158_ = ((lean_object*)(lp_mathlib_IdemSemiring_add__eq__sup___autoParam___closed__9));
v___x_159_ = l_Lean_Syntax_node1(v___x_151_, v___x_158_, v___x_149_);
v___x_160_ = l_Lean_Syntax_node2(v___x_151_, v___x_152_, v___x_157_, v___x_159_);
v___x_161_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_161_, 0, v___x_160_);
lean_ctor_set(v___x_161_, 1, v_a_140_);
return v___x_161_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______macroRules__Computability__term___u2217__1___boxed(lean_object* v_x_162_, lean_object* v_a_163_, lean_object* v_a_164_){
_start:
{
lean_object* v_res_165_; 
v_res_165_ = lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______macroRules__Computability__term___u2217__1(v_x_162_, v_a_163_, v_a_164_);
lean_dec_ref(v_a_163_);
return v_res_165_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______unexpand__KStar__kstar__1(lean_object* v_x_169_, lean_object* v_a_170_, lean_object* v_a_171_){
_start:
{
lean_object* v___x_172_; uint8_t v___x_173_; 
v___x_172_ = ((lean_object*)(lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______macroRules__Computability__term___u2217__1___closed__2));
lean_inc(v_x_169_);
v___x_173_ = l_Lean_Syntax_isOfKind(v_x_169_, v___x_172_);
if (v___x_173_ == 0)
{
lean_object* v___x_174_; lean_object* v___x_175_; 
lean_dec(v_x_169_);
v___x_174_ = lean_box(0);
v___x_175_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_175_, 0, v___x_174_);
lean_ctor_set(v___x_175_, 1, v_a_171_);
return v___x_175_;
}
else
{
lean_object* v___x_176_; lean_object* v___x_177_; lean_object* v___x_178_; uint8_t v___x_179_; 
v___x_176_ = lean_unsigned_to_nat(0u);
v___x_177_ = l_Lean_Syntax_getArg(v_x_169_, v___x_176_);
v___x_178_ = ((lean_object*)(lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______unexpand__KStar__kstar__1___closed__1));
lean_inc(v___x_177_);
v___x_179_ = l_Lean_Syntax_isOfKind(v___x_177_, v___x_178_);
if (v___x_179_ == 0)
{
lean_object* v___x_180_; lean_object* v___x_181_; 
lean_dec(v___x_177_);
lean_dec(v_x_169_);
v___x_180_ = lean_box(0);
v___x_181_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_181_, 0, v___x_180_);
lean_ctor_set(v___x_181_, 1, v_a_171_);
return v___x_181_;
}
else
{
lean_object* v___x_182_; lean_object* v___x_183_; uint8_t v___x_184_; 
v___x_182_ = lean_unsigned_to_nat(1u);
v___x_183_ = l_Lean_Syntax_getArg(v_x_169_, v___x_182_);
lean_dec(v_x_169_);
lean_inc(v___x_183_);
v___x_184_ = l_Lean_Syntax_matchesNull(v___x_183_, v___x_182_);
if (v___x_184_ == 0)
{
lean_object* v___x_185_; lean_object* v___x_186_; 
lean_dec(v___x_183_);
lean_dec(v___x_177_);
v___x_185_ = lean_box(0);
v___x_186_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_186_, 0, v___x_185_);
lean_ctor_set(v___x_186_, 1, v_a_171_);
return v___x_186_;
}
else
{
lean_object* v___x_187_; lean_object* v_ref_188_; uint8_t v___x_189_; lean_object* v___x_190_; lean_object* v___x_191_; lean_object* v___x_192_; lean_object* v___x_193_; lean_object* v___x_194_; lean_object* v___x_195_; 
v___x_187_ = l_Lean_Syntax_getArg(v___x_183_, v___x_176_);
lean_dec(v___x_183_);
v_ref_188_ = l_Lean_replaceRef(v___x_177_, v_a_170_);
lean_dec(v___x_177_);
v___x_189_ = 0;
v___x_190_ = l_Lean_SourceInfo_fromRef(v_ref_188_, v___x_189_);
lean_dec(v_ref_188_);
v___x_191_ = ((lean_object*)(lp_mathlib_Computability_term___u2217___closed__2));
v___x_192_ = ((lean_object*)(lp_mathlib_Computability_term___u2217___closed__3));
lean_inc(v___x_190_);
v___x_193_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_193_, 0, v___x_190_);
lean_ctor_set(v___x_193_, 1, v___x_192_);
v___x_194_ = l_Lean_Syntax_node2(v___x_190_, v___x_191_, v___x_187_, v___x_193_);
v___x_195_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_195_, 0, v___x_194_);
lean_ctor_set(v___x_195_, 1, v_a_171_);
return v___x_195_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______unexpand__KStar__kstar__1___boxed(lean_object* v_x_196_, lean_object* v_a_197_, lean_object* v_a_198_){
_start:
{
lean_object* v_res_199_; 
v_res_199_ = lp_mathlib_Computability___aux__Mathlib__Algebra__Order__Kleene______unexpand__KStar__kstar__1(v_x_196_, v_a_197_, v_a_198_);
lean_dec(v_a_197_);
return v_res_199_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IdemSemiring_ofSemiring___redArg___lam__0(lean_object* v_toAdd_200_, lean_object* v_x1_201_, lean_object* v_x2_202_){
_start:
{
lean_object* v___x_203_; 
v___x_203_ = lean_apply_2(v_toAdd_200_, v_x1_201_, v_x2_202_);
return v___x_203_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IdemSemiring_ofSemiring___redArg(lean_object* v_inst_207_){
_start:
{
lean_object* v___x_208_; lean_object* v___x_209_; lean_object* v_toAdd_210_; lean_object* v___x_211_; lean_object* v_toZero_212_; lean_object* v___x_214_; uint8_t v_isShared_215_; uint8_t v_isSharedCheck_221_; 
v___x_208_ = ((lean_object*)(lp_mathlib_IdemSemiring_ofSemiring___redArg___closed__0));
lean_inc_ref_n(v_inst_207_, 2);
v___x_209_ = lp_mathlib_instDistribOfSemiring___redArg(v_inst_207_);
v_toAdd_210_ = lean_ctor_get(v___x_209_, 1);
lean_inc(v_toAdd_210_);
lean_dec_ref(v___x_209_);
v___x_211_ = lp_mathlib_instMulZeroClassOfSemiring___redArg(v_inst_207_);
v_toZero_212_ = lean_ctor_get(v___x_211_, 1);
v_isSharedCheck_221_ = !lean_is_exclusive(v___x_211_);
if (v_isSharedCheck_221_ == 0)
{
lean_object* v_unused_222_; 
v_unused_222_ = lean_ctor_get(v___x_211_, 0);
lean_dec(v_unused_222_);
v___x_214_ = v___x_211_;
v_isShared_215_ = v_isSharedCheck_221_;
goto v_resetjp_213_;
}
else
{
lean_inc(v_toZero_212_);
lean_dec(v___x_211_);
v___x_214_ = lean_box(0);
v_isShared_215_ = v_isSharedCheck_221_;
goto v_resetjp_213_;
}
v_resetjp_213_:
{
lean_object* v___f_216_; lean_object* v___x_218_; 
v___f_216_ = lean_alloc_closure((void*)(lp_mathlib_IdemSemiring_ofSemiring___redArg___lam__0), 3, 1);
lean_closure_set(v___f_216_, 0, v_toAdd_210_);
if (v_isShared_215_ == 0)
{
lean_ctor_set(v___x_214_, 1, v___f_216_);
lean_ctor_set(v___x_214_, 0, v___x_208_);
v___x_218_ = v___x_214_;
goto v_reusejp_217_;
}
else
{
lean_object* v_reuseFailAlloc_220_; 
v_reuseFailAlloc_220_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_220_, 0, v___x_208_);
lean_ctor_set(v_reuseFailAlloc_220_, 1, v___f_216_);
v___x_218_ = v_reuseFailAlloc_220_;
goto v_reusejp_217_;
}
v_reusejp_217_:
{
lean_object* v___x_219_; 
v___x_219_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_219_, 0, v_inst_207_);
lean_ctor_set(v___x_219_, 1, v___x_218_);
lean_ctor_set(v___x_219_, 2, v_toZero_212_);
return v___x_219_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_IdemSemiring_ofSemiring(lean_object* v_00_u03b1_223_, lean_object* v_inst_224_, lean_object* v_h_225_){
_start:
{
lean_object* v___x_226_; lean_object* v___x_227_; lean_object* v_toAdd_228_; lean_object* v___x_229_; lean_object* v_toZero_230_; lean_object* v___x_232_; uint8_t v_isShared_233_; uint8_t v_isSharedCheck_239_; 
v___x_226_ = ((lean_object*)(lp_mathlib_IdemSemiring_ofSemiring___redArg___closed__0));
lean_inc_ref_n(v_inst_224_, 2);
v___x_227_ = lp_mathlib_instDistribOfSemiring___redArg(v_inst_224_);
v_toAdd_228_ = lean_ctor_get(v___x_227_, 1);
lean_inc(v_toAdd_228_);
lean_dec_ref(v___x_227_);
v___x_229_ = lp_mathlib_instMulZeroClassOfSemiring___redArg(v_inst_224_);
v_toZero_230_ = lean_ctor_get(v___x_229_, 1);
v_isSharedCheck_239_ = !lean_is_exclusive(v___x_229_);
if (v_isSharedCheck_239_ == 0)
{
lean_object* v_unused_240_; 
v_unused_240_ = lean_ctor_get(v___x_229_, 0);
lean_dec(v_unused_240_);
v___x_232_ = v___x_229_;
v_isShared_233_ = v_isSharedCheck_239_;
goto v_resetjp_231_;
}
else
{
lean_inc(v_toZero_230_);
lean_dec(v___x_229_);
v___x_232_ = lean_box(0);
v_isShared_233_ = v_isSharedCheck_239_;
goto v_resetjp_231_;
}
v_resetjp_231_:
{
lean_object* v___f_234_; lean_object* v___x_236_; 
v___f_234_ = lean_alloc_closure((void*)(lp_mathlib_IdemSemiring_ofSemiring___redArg___lam__0), 3, 1);
lean_closure_set(v___f_234_, 0, v_toAdd_228_);
if (v_isShared_233_ == 0)
{
lean_ctor_set(v___x_232_, 1, v___f_234_);
lean_ctor_set(v___x_232_, 0, v___x_226_);
v___x_236_ = v___x_232_;
goto v_reusejp_235_;
}
else
{
lean_object* v_reuseFailAlloc_238_; 
v_reuseFailAlloc_238_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_238_, 0, v___x_226_);
lean_ctor_set(v_reuseFailAlloc_238_, 1, v___f_234_);
v___x_236_ = v_reuseFailAlloc_238_;
goto v_reusejp_235_;
}
v_reusejp_235_:
{
lean_object* v___x_237_; 
v___x_237_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_237_, 0, v_inst_224_);
lean_ctor_set(v___x_237_, 1, v___x_236_);
lean_ctor_set(v___x_237_, 2, v_toZero_230_);
return v___x_237_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instIdemSemiring___redArg(lean_object* v_inst_241_, lean_object* v_inst_242_){
_start:
{
lean_object* v_toSemiring_243_; lean_object* v_toSemilatticeSup_244_; lean_object* v_toOrderBot_245_; lean_object* v_toSemiring_246_; lean_object* v_toSemilatticeSup_247_; lean_object* v_toOrderBot_248_; lean_object* v___x_250_; uint8_t v_isShared_251_; uint8_t v_isSharedCheck_258_; 
v_toSemiring_243_ = lean_ctor_get(v_inst_241_, 0);
lean_inc_ref(v_toSemiring_243_);
v_toSemilatticeSup_244_ = lean_ctor_get(v_inst_241_, 1);
lean_inc_ref(v_toSemilatticeSup_244_);
v_toOrderBot_245_ = lean_ctor_get(v_inst_241_, 2);
lean_inc(v_toOrderBot_245_);
lean_dec_ref(v_inst_241_);
v_toSemiring_246_ = lean_ctor_get(v_inst_242_, 0);
v_toSemilatticeSup_247_ = lean_ctor_get(v_inst_242_, 1);
v_toOrderBot_248_ = lean_ctor_get(v_inst_242_, 2);
v_isSharedCheck_258_ = !lean_is_exclusive(v_inst_242_);
if (v_isSharedCheck_258_ == 0)
{
v___x_250_ = v_inst_242_;
v_isShared_251_ = v_isSharedCheck_258_;
goto v_resetjp_249_;
}
else
{
lean_inc(v_toOrderBot_248_);
lean_inc(v_toSemilatticeSup_247_);
lean_inc(v_toSemiring_246_);
lean_dec(v_inst_242_);
v___x_250_ = lean_box(0);
v_isShared_251_ = v_isSharedCheck_258_;
goto v_resetjp_249_;
}
v_resetjp_249_:
{
lean_object* v___x_252_; lean_object* v___x_253_; lean_object* v___x_254_; lean_object* v___x_256_; 
v___x_252_ = lp_mathlib_Prod_instSemiring___redArg(v_toSemiring_243_, v_toSemiring_246_);
v___x_253_ = lp_mathlib_Prod_instSemilatticeSup___redArg(v_toSemilatticeSup_244_, v_toSemilatticeSup_247_);
v___x_254_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_254_, 0, v_toOrderBot_245_);
lean_ctor_set(v___x_254_, 1, v_toOrderBot_248_);
if (v_isShared_251_ == 0)
{
lean_ctor_set(v___x_250_, 2, v___x_254_);
lean_ctor_set(v___x_250_, 1, v___x_253_);
lean_ctor_set(v___x_250_, 0, v___x_252_);
v___x_256_ = v___x_250_;
goto v_reusejp_255_;
}
else
{
lean_object* v_reuseFailAlloc_257_; 
v_reuseFailAlloc_257_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_257_, 0, v___x_252_);
lean_ctor_set(v_reuseFailAlloc_257_, 1, v___x_253_);
lean_ctor_set(v_reuseFailAlloc_257_, 2, v___x_254_);
v___x_256_ = v_reuseFailAlloc_257_;
goto v_reusejp_255_;
}
v_reusejp_255_:
{
return v___x_256_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instIdemSemiring(lean_object* v_00_u03b1_259_, lean_object* v_00_u03b2_260_, lean_object* v_inst_261_, lean_object* v_inst_262_){
_start:
{
lean_object* v___x_263_; 
v___x_263_ = lp_mathlib_Prod_instIdemSemiring___redArg(v_inst_261_, v_inst_262_);
return v___x_263_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instIdemCommSemiring___redArg(lean_object* v_inst_264_, lean_object* v_inst_265_){
_start:
{
lean_object* v_toCommSemiring_266_; lean_object* v_toCommSemiring_267_; lean_object* v___x_268_; lean_object* v___x_269_; lean_object* v___x_270_; lean_object* v___x_271_; lean_object* v_toSemilatticeSup_272_; lean_object* v_toOrderBot_273_; lean_object* v___x_275_; uint8_t v_isShared_276_; uint8_t v_isSharedCheck_280_; 
v_toCommSemiring_266_ = lean_ctor_get(v_inst_264_, 0);
v_toCommSemiring_267_ = lean_ctor_get(v_inst_265_, 0);
lean_inc_ref(v_toCommSemiring_267_);
lean_inc_ref(v_toCommSemiring_266_);
v___x_268_ = lp_mathlib_Prod_instSemiring___redArg(v_toCommSemiring_266_, v_toCommSemiring_267_);
v___x_269_ = lp_mathlib_IdemCommSemiring_toIdemSemiring___redArg(v_inst_264_);
v___x_270_ = lp_mathlib_IdemCommSemiring_toIdemSemiring___redArg(v_inst_265_);
v___x_271_ = lp_mathlib_Prod_instIdemSemiring___redArg(v___x_269_, v___x_270_);
v_toSemilatticeSup_272_ = lean_ctor_get(v___x_271_, 1);
v_toOrderBot_273_ = lean_ctor_get(v___x_271_, 2);
v_isSharedCheck_280_ = !lean_is_exclusive(v___x_271_);
if (v_isSharedCheck_280_ == 0)
{
lean_object* v_unused_281_; 
v_unused_281_ = lean_ctor_get(v___x_271_, 0);
lean_dec(v_unused_281_);
v___x_275_ = v___x_271_;
v_isShared_276_ = v_isSharedCheck_280_;
goto v_resetjp_274_;
}
else
{
lean_inc(v_toOrderBot_273_);
lean_inc(v_toSemilatticeSup_272_);
lean_dec(v___x_271_);
v___x_275_ = lean_box(0);
v_isShared_276_ = v_isSharedCheck_280_;
goto v_resetjp_274_;
}
v_resetjp_274_:
{
lean_object* v___x_278_; 
if (v_isShared_276_ == 0)
{
lean_ctor_set(v___x_275_, 0, v___x_268_);
v___x_278_ = v___x_275_;
goto v_reusejp_277_;
}
else
{
lean_object* v_reuseFailAlloc_279_; 
v_reuseFailAlloc_279_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_279_, 0, v___x_268_);
lean_ctor_set(v_reuseFailAlloc_279_, 1, v_toSemilatticeSup_272_);
lean_ctor_set(v_reuseFailAlloc_279_, 2, v_toOrderBot_273_);
v___x_278_ = v_reuseFailAlloc_279_;
goto v_reusejp_277_;
}
v_reusejp_277_:
{
return v___x_278_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instIdemCommSemiring(lean_object* v_00_u03b1_282_, lean_object* v_00_u03b2_283_, lean_object* v_inst_284_, lean_object* v_inst_285_){
_start:
{
lean_object* v___x_286_; 
v___x_286_ = lp_mathlib_Prod_instIdemCommSemiring___redArg(v_inst_284_, v_inst_285_);
return v___x_286_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instKleeneAlgebra___redArg___lam__0(lean_object* v_toKStar_287_, lean_object* v_toKStar_288_, lean_object* v_a_289_){
_start:
{
lean_object* v_fst_290_; lean_object* v_snd_291_; lean_object* v___x_293_; uint8_t v_isShared_294_; uint8_t v_isSharedCheck_300_; 
v_fst_290_ = lean_ctor_get(v_a_289_, 0);
v_snd_291_ = lean_ctor_get(v_a_289_, 1);
v_isSharedCheck_300_ = !lean_is_exclusive(v_a_289_);
if (v_isSharedCheck_300_ == 0)
{
v___x_293_ = v_a_289_;
v_isShared_294_ = v_isSharedCheck_300_;
goto v_resetjp_292_;
}
else
{
lean_inc(v_snd_291_);
lean_inc(v_fst_290_);
lean_dec(v_a_289_);
v___x_293_ = lean_box(0);
v_isShared_294_ = v_isSharedCheck_300_;
goto v_resetjp_292_;
}
v_resetjp_292_:
{
lean_object* v___x_295_; lean_object* v___x_296_; lean_object* v___x_298_; 
v___x_295_ = lean_apply_1(v_toKStar_287_, v_fst_290_);
v___x_296_ = lean_apply_1(v_toKStar_288_, v_snd_291_);
if (v_isShared_294_ == 0)
{
lean_ctor_set(v___x_293_, 1, v___x_296_);
lean_ctor_set(v___x_293_, 0, v___x_295_);
v___x_298_ = v___x_293_;
goto v_reusejp_297_;
}
else
{
lean_object* v_reuseFailAlloc_299_; 
v_reuseFailAlloc_299_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_299_, 0, v___x_295_);
lean_ctor_set(v_reuseFailAlloc_299_, 1, v___x_296_);
v___x_298_ = v_reuseFailAlloc_299_;
goto v_reusejp_297_;
}
v_reusejp_297_:
{
return v___x_298_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instKleeneAlgebra___redArg(lean_object* v_inst_301_, lean_object* v_inst_302_){
_start:
{
lean_object* v_toIdemSemiring_303_; lean_object* v_toKStar_304_; lean_object* v_toIdemSemiring_305_; lean_object* v_toKStar_306_; lean_object* v___x_308_; uint8_t v_isShared_309_; uint8_t v_isSharedCheck_315_; 
v_toIdemSemiring_303_ = lean_ctor_get(v_inst_301_, 0);
lean_inc_ref(v_toIdemSemiring_303_);
v_toKStar_304_ = lean_ctor_get(v_inst_301_, 1);
lean_inc(v_toKStar_304_);
lean_dec_ref(v_inst_301_);
v_toIdemSemiring_305_ = lean_ctor_get(v_inst_302_, 0);
v_toKStar_306_ = lean_ctor_get(v_inst_302_, 1);
v_isSharedCheck_315_ = !lean_is_exclusive(v_inst_302_);
if (v_isSharedCheck_315_ == 0)
{
v___x_308_ = v_inst_302_;
v_isShared_309_ = v_isSharedCheck_315_;
goto v_resetjp_307_;
}
else
{
lean_inc(v_toKStar_306_);
lean_inc(v_toIdemSemiring_305_);
lean_dec(v_inst_302_);
v___x_308_ = lean_box(0);
v_isShared_309_ = v_isSharedCheck_315_;
goto v_resetjp_307_;
}
v_resetjp_307_:
{
lean_object* v___f_310_; lean_object* v___x_311_; lean_object* v___x_313_; 
v___f_310_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instKleeneAlgebra___redArg___lam__0), 3, 2);
lean_closure_set(v___f_310_, 0, v_toKStar_304_);
lean_closure_set(v___f_310_, 1, v_toKStar_306_);
v___x_311_ = lp_mathlib_Prod_instIdemSemiring___redArg(v_toIdemSemiring_303_, v_toIdemSemiring_305_);
if (v_isShared_309_ == 0)
{
lean_ctor_set(v___x_308_, 1, v___f_310_);
lean_ctor_set(v___x_308_, 0, v___x_311_);
v___x_313_ = v___x_308_;
goto v_reusejp_312_;
}
else
{
lean_object* v_reuseFailAlloc_314_; 
v_reuseFailAlloc_314_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_314_, 0, v___x_311_);
lean_ctor_set(v_reuseFailAlloc_314_, 1, v___f_310_);
v___x_313_ = v_reuseFailAlloc_314_;
goto v_reusejp_312_;
}
v_reusejp_312_:
{
return v___x_313_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instKleeneAlgebra(lean_object* v_00_u03b1_316_, lean_object* v_00_u03b2_317_, lean_object* v_inst_318_, lean_object* v_inst_319_){
_start:
{
lean_object* v___x_320_; 
v___x_320_ = lp_mathlib_Prod_instKleeneAlgebra___redArg(v_inst_318_, v_inst_319_);
return v___x_320_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instIdemSemiring___redArg___lam__0(lean_object* v_inst_321_, lean_object* v_i_322_){
_start:
{
lean_object* v___x_323_; lean_object* v_toSemiring_324_; 
v___x_323_ = lean_apply_1(v_inst_321_, v_i_322_);
v_toSemiring_324_ = lean_ctor_get(v___x_323_, 0);
lean_inc_ref(v_toSemiring_324_);
lean_dec_ref(v___x_323_);
return v_toSemiring_324_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instIdemSemiring___redArg___lam__1(lean_object* v_inst_325_, lean_object* v_i_326_){
_start:
{
lean_object* v___x_327_; lean_object* v_toSemilatticeSup_328_; 
v___x_327_ = lean_apply_1(v_inst_325_, v_i_326_);
v_toSemilatticeSup_328_ = lean_ctor_get(v___x_327_, 1);
lean_inc_ref(v_toSemilatticeSup_328_);
lean_dec_ref(v___x_327_);
return v_toSemilatticeSup_328_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instIdemSemiring___redArg___lam__2(lean_object* v_inst_329_, lean_object* v_i_330_){
_start:
{
lean_object* v___x_331_; lean_object* v_toOrderBot_332_; 
v___x_331_ = lean_apply_1(v_inst_329_, v_i_330_);
v_toOrderBot_332_ = lean_ctor_get(v___x_331_, 2);
lean_inc(v_toOrderBot_332_);
lean_dec_ref(v___x_331_);
return v_toOrderBot_332_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instIdemSemiring___redArg(lean_object* v_inst_333_){
_start:
{
lean_object* v___f_334_; lean_object* v___f_335_; lean_object* v___f_336_; lean_object* v___x_337_; lean_object* v___x_338_; lean_object* v___x_339_; lean_object* v___x_340_; 
lean_inc_ref_n(v_inst_333_, 2);
v___f_334_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instIdemSemiring___redArg___lam__0), 2, 1);
lean_closure_set(v___f_334_, 0, v_inst_333_);
v___f_335_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instIdemSemiring___redArg___lam__1), 2, 1);
lean_closure_set(v___f_335_, 0, v_inst_333_);
v___f_336_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instIdemSemiring___redArg___lam__2), 2, 1);
lean_closure_set(v___f_336_, 0, v_inst_333_);
v___x_337_ = lp_mathlib_Pi_semiring___redArg(v___f_334_);
v___x_338_ = lp_mathlib_Pi_instSemilatticeSup___redArg(v___f_335_);
v___x_339_ = lp_mathlib_Pi_instOrderBot___redArg(v___f_336_);
v___x_340_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_340_, 0, v___x_337_);
lean_ctor_set(v___x_340_, 1, v___x_338_);
lean_ctor_set(v___x_340_, 2, v___x_339_);
return v___x_340_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instIdemSemiring(lean_object* v_00_u03b9_341_, lean_object* v_00_u03c0_342_, lean_object* v_inst_343_){
_start:
{
lean_object* v___x_344_; 
v___x_344_ = lp_mathlib_Pi_instIdemSemiring___redArg(v_inst_343_);
return v___x_344_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instIdemCommSemiringForall___redArg___lam__0(lean_object* v_inst_345_, lean_object* v_i_346_){
_start:
{
lean_object* v___x_347_; lean_object* v_toCommSemiring_348_; 
v___x_347_ = lean_apply_1(v_inst_345_, v_i_346_);
v_toCommSemiring_348_ = lean_ctor_get(v___x_347_, 0);
lean_inc_ref(v_toCommSemiring_348_);
lean_dec_ref(v___x_347_);
return v_toCommSemiring_348_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instIdemCommSemiringForall___redArg___lam__1(lean_object* v_inst_349_, lean_object* v_i_350_){
_start:
{
lean_object* v___x_351_; lean_object* v___x_352_; 
v___x_351_ = lean_apply_1(v_inst_349_, v_i_350_);
v___x_352_ = lp_mathlib_IdemCommSemiring_toIdemSemiring___redArg(v___x_351_);
return v___x_352_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instIdemCommSemiringForall___redArg(lean_object* v_inst_353_){
_start:
{
lean_object* v___f_354_; lean_object* v___f_355_; lean_object* v___x_356_; lean_object* v___x_357_; lean_object* v_toSemilatticeSup_358_; lean_object* v___x_360_; uint8_t v_isShared_361_; uint8_t v_isSharedCheck_367_; 
lean_inc_ref(v_inst_353_);
v___f_354_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instIdemCommSemiringForall___redArg___lam__0), 2, 1);
lean_closure_set(v___f_354_, 0, v_inst_353_);
v___f_355_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instIdemCommSemiringForall___redArg___lam__1), 2, 1);
lean_closure_set(v___f_355_, 0, v_inst_353_);
v___x_356_ = lp_mathlib_Pi_commSemiring___redArg(v___f_354_);
lean_inc_ref(v___f_355_);
v___x_357_ = lp_mathlib_Pi_instIdemSemiring___redArg(v___f_355_);
v_toSemilatticeSup_358_ = lean_ctor_get(v___x_357_, 1);
v_isSharedCheck_367_ = !lean_is_exclusive(v___x_357_);
if (v_isSharedCheck_367_ == 0)
{
lean_object* v_unused_368_; lean_object* v_unused_369_; 
v_unused_368_ = lean_ctor_get(v___x_357_, 2);
lean_dec(v_unused_368_);
v_unused_369_ = lean_ctor_get(v___x_357_, 0);
lean_dec(v_unused_369_);
v___x_360_ = v___x_357_;
v_isShared_361_ = v_isSharedCheck_367_;
goto v_resetjp_359_;
}
else
{
lean_inc(v_toSemilatticeSup_358_);
lean_dec(v___x_357_);
v___x_360_ = lean_box(0);
v_isShared_361_ = v_isSharedCheck_367_;
goto v_resetjp_359_;
}
v_resetjp_359_:
{
lean_object* v___f_362_; lean_object* v___x_363_; lean_object* v___x_365_; 
v___f_362_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instIdemSemiring___redArg___lam__2), 2, 1);
lean_closure_set(v___f_362_, 0, v___f_355_);
v___x_363_ = lp_mathlib_Pi_instOrderBot___redArg(v___f_362_);
if (v_isShared_361_ == 0)
{
lean_ctor_set(v___x_360_, 2, v___x_363_);
lean_ctor_set(v___x_360_, 0, v___x_356_);
v___x_365_ = v___x_360_;
goto v_reusejp_364_;
}
else
{
lean_object* v_reuseFailAlloc_366_; 
v_reuseFailAlloc_366_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_366_, 0, v___x_356_);
lean_ctor_set(v_reuseFailAlloc_366_, 1, v_toSemilatticeSup_358_);
lean_ctor_set(v_reuseFailAlloc_366_, 2, v___x_363_);
v___x_365_ = v_reuseFailAlloc_366_;
goto v_reusejp_364_;
}
v_reusejp_364_:
{
return v___x_365_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instIdemCommSemiringForall(lean_object* v_00_u03b9_370_, lean_object* v_00_u03c0_371_, lean_object* v_inst_372_){
_start:
{
lean_object* v___x_373_; 
v___x_373_ = lp_mathlib_Pi_instIdemCommSemiringForall___redArg(v_inst_372_);
return v___x_373_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instKleeneAlgebraForall___redArg___lam__0(lean_object* v_inst_374_, lean_object* v_i_375_){
_start:
{
lean_object* v___x_376_; lean_object* v_toIdemSemiring_377_; 
v___x_376_ = lean_apply_1(v_inst_374_, v_i_375_);
v_toIdemSemiring_377_ = lean_ctor_get(v___x_376_, 0);
lean_inc_ref(v_toIdemSemiring_377_);
lean_dec_ref(v___x_376_);
return v_toIdemSemiring_377_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instKleeneAlgebraForall___redArg___lam__1(lean_object* v_inst_378_, lean_object* v_a_379_, lean_object* v_i_380_){
_start:
{
lean_object* v___x_381_; lean_object* v_toKStar_382_; lean_object* v___x_383_; lean_object* v___x_384_; 
lean_inc(v_i_380_);
v___x_381_ = lean_apply_1(v_inst_378_, v_i_380_);
v_toKStar_382_ = lean_ctor_get(v___x_381_, 1);
lean_inc(v_toKStar_382_);
lean_dec_ref(v___x_381_);
v___x_383_ = lean_apply_1(v_a_379_, v_i_380_);
v___x_384_ = lean_apply_1(v_toKStar_382_, v___x_383_);
return v___x_384_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instKleeneAlgebraForall___redArg(lean_object* v_inst_385_){
_start:
{
lean_object* v___f_386_; lean_object* v___f_387_; lean_object* v___x_388_; lean_object* v___x_389_; 
lean_inc_ref(v_inst_385_);
v___f_386_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instKleeneAlgebraForall___redArg___lam__0), 2, 1);
lean_closure_set(v___f_386_, 0, v_inst_385_);
v___f_387_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instKleeneAlgebraForall___redArg___lam__1), 3, 1);
lean_closure_set(v___f_387_, 0, v_inst_385_);
v___x_388_ = lp_mathlib_Pi_instIdemSemiring___redArg(v___f_386_);
v___x_389_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_389_, 0, v___x_388_);
lean_ctor_set(v___x_389_, 1, v___f_387_);
return v___x_389_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instKleeneAlgebraForall(lean_object* v_00_u03b9_390_, lean_object* v_00_u03c0_391_, lean_object* v_inst_392_){
_start:
{
lean_object* v___x_393_; 
v___x_393_ = lp_mathlib_Pi_instKleeneAlgebraForall___redArg(v_inst_392_);
return v___x_393_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_idemSemiring___redArg___lam__0(lean_object* v_inst_394_, lean_object* v_n_395_, lean_object* v_x_396_){
_start:
{
lean_object* v___x_397_; 
v___x_397_ = lean_apply_2(v_inst_394_, v_x_396_, v_n_395_);
return v___x_397_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_idemSemiring___redArg___lam__1(lean_object* v_inst_398_, lean_object* v_a_399_, lean_object* v_b_400_){
_start:
{
lean_object* v___x_401_; 
v___x_401_ = lean_apply_2(v_inst_398_, v_a_399_, v_b_400_);
return v___x_401_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_idemSemiring___redArg(lean_object* v_inst_402_, lean_object* v_inst_403_, lean_object* v_inst_404_, lean_object* v_inst_405_, lean_object* v_inst_406_, lean_object* v_inst_407_, lean_object* v_inst_408_, lean_object* v_inst_409_, lean_object* v_inst_410_, lean_object* v_inst_411_, lean_object* v_inst_412_){
_start:
{
lean_object* v___f_413_; lean_object* v___f_414_; lean_object* v___x_415_; lean_object* v___x_416_; lean_object* v___x_417_; lean_object* v___x_418_; lean_object* v___x_419_; lean_object* v___x_420_; lean_object* v___x_421_; 
v___f_413_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_idemSemiring___redArg___lam__0), 3, 1);
lean_closure_set(v___f_413_, 0, v_inst_408_);
v___f_414_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_idemSemiring___redArg___lam__1), 3, 1);
lean_closure_set(v___f_414_, 0, v_inst_411_);
v___x_415_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_415_, 0, lean_box(0));
lean_closure_set(v___x_415_, 1, v_inst_410_);
v___x_416_ = lp_mathlib_Function_Injective_addMonoid___redArg(v_inst_406_, v_inst_404_, v_inst_409_);
v___x_417_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_417_, 0, v_inst_405_);
lean_ctor_set(v___x_417_, 1, v_inst_407_);
lean_ctor_set(v___x_417_, 2, v___f_413_);
v___x_418_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_418_, 0, v___x_416_);
lean_ctor_set(v___x_418_, 1, v___x_417_);
lean_ctor_set(v___x_418_, 2, v___x_415_);
v___x_419_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_419_, 0, v_inst_402_);
lean_ctor_set(v___x_419_, 1, v_inst_403_);
v___x_420_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_420_, 0, v___x_419_);
lean_ctor_set(v___x_420_, 1, v___f_414_);
v___x_421_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_421_, 0, v___x_418_);
lean_ctor_set(v___x_421_, 1, v___x_420_);
lean_ctor_set(v___x_421_, 2, v_inst_412_);
return v___x_421_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_idemSemiring(lean_object* v_00_u03b1_422_, lean_object* v_00_u03b2_423_, lean_object* v_inst_424_, lean_object* v_inst_425_, lean_object* v_inst_426_, lean_object* v_inst_427_, lean_object* v_inst_428_, lean_object* v_inst_429_, lean_object* v_inst_430_, lean_object* v_inst_431_, lean_object* v_inst_432_, lean_object* v_inst_433_, lean_object* v_inst_434_, lean_object* v_inst_435_, lean_object* v_f_436_, lean_object* v_hf_437_, lean_object* v_le_438_, lean_object* v_lt_439_, lean_object* v_zero_440_, lean_object* v_one_441_, lean_object* v_add_442_, lean_object* v_mul_443_, lean_object* v_nsmul_444_, lean_object* v_npow_445_, lean_object* v_natCast_446_, lean_object* v_sup_447_, lean_object* v_bot_448_){
_start:
{
lean_object* v___f_449_; lean_object* v___f_450_; lean_object* v___x_451_; lean_object* v___x_452_; lean_object* v___x_453_; lean_object* v___x_454_; lean_object* v___x_455_; lean_object* v___x_456_; lean_object* v___x_457_; 
v___f_449_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_idemSemiring___redArg___lam__0), 3, 1);
lean_closure_set(v___f_449_, 0, v_inst_431_);
v___f_450_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_idemSemiring___redArg___lam__1), 3, 1);
lean_closure_set(v___f_450_, 0, v_inst_434_);
v___x_451_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_451_, 0, lean_box(0));
lean_closure_set(v___x_451_, 1, v_inst_433_);
v___x_452_ = lp_mathlib_Function_Injective_addMonoid___redArg(v_inst_429_, v_inst_427_, v_inst_432_);
v___x_453_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_453_, 0, v_inst_428_);
lean_ctor_set(v___x_453_, 1, v_inst_430_);
lean_ctor_set(v___x_453_, 2, v___f_449_);
v___x_454_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_454_, 0, v___x_452_);
lean_ctor_set(v___x_454_, 1, v___x_453_);
lean_ctor_set(v___x_454_, 2, v___x_451_);
v___x_455_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_455_, 0, v_inst_425_);
lean_ctor_set(v___x_455_, 1, v_inst_426_);
v___x_456_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_456_, 0, v___x_455_);
lean_ctor_set(v___x_456_, 1, v___f_450_);
v___x_457_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_457_, 0, v___x_454_);
lean_ctor_set(v___x_457_, 1, v___x_456_);
lean_ctor_set(v___x_457_, 2, v_inst_435_);
return v___x_457_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_idemSemiring___boxed(lean_object** _args){
lean_object* v_00_u03b1_458_ = _args[0];
lean_object* v_00_u03b2_459_ = _args[1];
lean_object* v_inst_460_ = _args[2];
lean_object* v_inst_461_ = _args[3];
lean_object* v_inst_462_ = _args[4];
lean_object* v_inst_463_ = _args[5];
lean_object* v_inst_464_ = _args[6];
lean_object* v_inst_465_ = _args[7];
lean_object* v_inst_466_ = _args[8];
lean_object* v_inst_467_ = _args[9];
lean_object* v_inst_468_ = _args[10];
lean_object* v_inst_469_ = _args[11];
lean_object* v_inst_470_ = _args[12];
lean_object* v_inst_471_ = _args[13];
lean_object* v_f_472_ = _args[14];
lean_object* v_hf_473_ = _args[15];
lean_object* v_le_474_ = _args[16];
lean_object* v_lt_475_ = _args[17];
lean_object* v_zero_476_ = _args[18];
lean_object* v_one_477_ = _args[19];
lean_object* v_add_478_ = _args[20];
lean_object* v_mul_479_ = _args[21];
lean_object* v_nsmul_480_ = _args[22];
lean_object* v_npow_481_ = _args[23];
lean_object* v_natCast_482_ = _args[24];
lean_object* v_sup_483_ = _args[25];
lean_object* v_bot_484_ = _args[26];
_start:
{
lean_object* v_res_485_; 
v_res_485_ = lp_mathlib_Function_Injective_idemSemiring(v_00_u03b1_458_, v_00_u03b2_459_, v_inst_460_, v_inst_461_, v_inst_462_, v_inst_463_, v_inst_464_, v_inst_465_, v_inst_466_, v_inst_467_, v_inst_468_, v_inst_469_, v_inst_470_, v_inst_471_, v_f_472_, v_hf_473_, v_le_474_, v_lt_475_, v_zero_476_, v_one_477_, v_add_478_, v_mul_479_, v_nsmul_480_, v_npow_481_, v_natCast_482_, v_sup_483_, v_bot_484_);
lean_dec(v_f_472_);
lean_dec_ref(v_inst_460_);
return v_res_485_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_idemCommSemiring___redArg(lean_object* v_inst_486_, lean_object* v_inst_487_, lean_object* v_inst_488_, lean_object* v_inst_489_, lean_object* v_inst_490_, lean_object* v_inst_491_, lean_object* v_inst_492_, lean_object* v_inst_493_, lean_object* v_inst_494_, lean_object* v_inst_495_, lean_object* v_inst_496_){
_start:
{
lean_object* v___f_497_; lean_object* v___f_498_; lean_object* v___x_499_; lean_object* v___x_500_; lean_object* v___x_501_; lean_object* v___x_502_; lean_object* v___x_503_; lean_object* v___x_504_; lean_object* v___x_505_; 
v___f_497_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_idemSemiring___redArg___lam__0), 3, 1);
lean_closure_set(v___f_497_, 0, v_inst_492_);
v___f_498_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_idemSemiring___redArg___lam__1), 3, 1);
lean_closure_set(v___f_498_, 0, v_inst_495_);
v___x_499_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_499_, 0, lean_box(0));
lean_closure_set(v___x_499_, 1, v_inst_494_);
v___x_500_ = lp_mathlib_Function_Injective_addMonoid___redArg(v_inst_490_, v_inst_488_, v_inst_493_);
v___x_501_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_501_, 0, v_inst_489_);
lean_ctor_set(v___x_501_, 1, v_inst_491_);
lean_ctor_set(v___x_501_, 2, v___f_497_);
v___x_502_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_502_, 0, v___x_500_);
lean_ctor_set(v___x_502_, 1, v___x_501_);
lean_ctor_set(v___x_502_, 2, v___x_499_);
v___x_503_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_503_, 0, v_inst_486_);
lean_ctor_set(v___x_503_, 1, v_inst_487_);
v___x_504_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_504_, 0, v___x_503_);
lean_ctor_set(v___x_504_, 1, v___f_498_);
v___x_505_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_505_, 0, v___x_502_);
lean_ctor_set(v___x_505_, 1, v___x_504_);
lean_ctor_set(v___x_505_, 2, v_inst_496_);
return v___x_505_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_idemCommSemiring(lean_object* v_00_u03b1_506_, lean_object* v_00_u03b2_507_, lean_object* v_inst_508_, lean_object* v_inst_509_, lean_object* v_inst_510_, lean_object* v_inst_511_, lean_object* v_inst_512_, lean_object* v_inst_513_, lean_object* v_inst_514_, lean_object* v_inst_515_, lean_object* v_inst_516_, lean_object* v_inst_517_, lean_object* v_inst_518_, lean_object* v_inst_519_, lean_object* v_f_520_, lean_object* v_hf_521_, lean_object* v_le_522_, lean_object* v_lt_523_, lean_object* v_zero_524_, lean_object* v_one_525_, lean_object* v_add_526_, lean_object* v_mul_527_, lean_object* v_nsmul_528_, lean_object* v_npow_529_, lean_object* v_natCast_530_, lean_object* v_sup_531_, lean_object* v_bot_532_){
_start:
{
lean_object* v___f_533_; lean_object* v___f_534_; lean_object* v___x_535_; lean_object* v___x_536_; lean_object* v___x_537_; lean_object* v___x_538_; lean_object* v___x_539_; lean_object* v___x_540_; lean_object* v___x_541_; 
v___f_533_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_idemSemiring___redArg___lam__0), 3, 1);
lean_closure_set(v___f_533_, 0, v_inst_515_);
v___f_534_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_idemSemiring___redArg___lam__1), 3, 1);
lean_closure_set(v___f_534_, 0, v_inst_518_);
v___x_535_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_535_, 0, lean_box(0));
lean_closure_set(v___x_535_, 1, v_inst_517_);
v___x_536_ = lp_mathlib_Function_Injective_addMonoid___redArg(v_inst_513_, v_inst_511_, v_inst_516_);
v___x_537_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_537_, 0, v_inst_512_);
lean_ctor_set(v___x_537_, 1, v_inst_514_);
lean_ctor_set(v___x_537_, 2, v___f_533_);
v___x_538_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_538_, 0, v___x_536_);
lean_ctor_set(v___x_538_, 1, v___x_537_);
lean_ctor_set(v___x_538_, 2, v___x_535_);
v___x_539_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_539_, 0, v_inst_509_);
lean_ctor_set(v___x_539_, 1, v_inst_510_);
v___x_540_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_540_, 0, v___x_539_);
lean_ctor_set(v___x_540_, 1, v___f_534_);
v___x_541_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_541_, 0, v___x_538_);
lean_ctor_set(v___x_541_, 1, v___x_540_);
lean_ctor_set(v___x_541_, 2, v_inst_519_);
return v___x_541_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_idemCommSemiring___boxed(lean_object** _args){
lean_object* v_00_u03b1_542_ = _args[0];
lean_object* v_00_u03b2_543_ = _args[1];
lean_object* v_inst_544_ = _args[2];
lean_object* v_inst_545_ = _args[3];
lean_object* v_inst_546_ = _args[4];
lean_object* v_inst_547_ = _args[5];
lean_object* v_inst_548_ = _args[6];
lean_object* v_inst_549_ = _args[7];
lean_object* v_inst_550_ = _args[8];
lean_object* v_inst_551_ = _args[9];
lean_object* v_inst_552_ = _args[10];
lean_object* v_inst_553_ = _args[11];
lean_object* v_inst_554_ = _args[12];
lean_object* v_inst_555_ = _args[13];
lean_object* v_f_556_ = _args[14];
lean_object* v_hf_557_ = _args[15];
lean_object* v_le_558_ = _args[16];
lean_object* v_lt_559_ = _args[17];
lean_object* v_zero_560_ = _args[18];
lean_object* v_one_561_ = _args[19];
lean_object* v_add_562_ = _args[20];
lean_object* v_mul_563_ = _args[21];
lean_object* v_nsmul_564_ = _args[22];
lean_object* v_npow_565_ = _args[23];
lean_object* v_natCast_566_ = _args[24];
lean_object* v_sup_567_ = _args[25];
lean_object* v_bot_568_ = _args[26];
_start:
{
lean_object* v_res_569_; 
v_res_569_ = lp_mathlib_Function_Injective_idemCommSemiring(v_00_u03b1_542_, v_00_u03b2_543_, v_inst_544_, v_inst_545_, v_inst_546_, v_inst_547_, v_inst_548_, v_inst_549_, v_inst_550_, v_inst_551_, v_inst_552_, v_inst_553_, v_inst_554_, v_inst_555_, v_f_556_, v_hf_557_, v_le_558_, v_lt_559_, v_zero_560_, v_one_561_, v_add_562_, v_mul_563_, v_nsmul_564_, v_npow_565_, v_natCast_566_, v_sup_567_, v_bot_568_);
lean_dec(v_f_556_);
lean_dec_ref(v_inst_544_);
return v_res_569_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_kleeneAlgebra___redArg(lean_object* v_inst_570_, lean_object* v_inst_571_, lean_object* v_inst_572_, lean_object* v_inst_573_, lean_object* v_inst_574_, lean_object* v_inst_575_, lean_object* v_inst_576_, lean_object* v_inst_577_, lean_object* v_inst_578_, lean_object* v_inst_579_, lean_object* v_inst_580_, lean_object* v_inst_581_){
_start:
{
lean_object* v___f_582_; lean_object* v___f_583_; lean_object* v___x_584_; lean_object* v___x_585_; lean_object* v___x_586_; lean_object* v___x_587_; lean_object* v___x_588_; lean_object* v___x_589_; lean_object* v___x_590_; lean_object* v___x_591_; 
v___f_582_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_idemSemiring___redArg___lam__0), 3, 1);
lean_closure_set(v___f_582_, 0, v_inst_576_);
v___f_583_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_idemSemiring___redArg___lam__1), 3, 1);
lean_closure_set(v___f_583_, 0, v_inst_579_);
v___x_584_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_584_, 0, lean_box(0));
lean_closure_set(v___x_584_, 1, v_inst_578_);
v___x_585_ = lp_mathlib_Function_Injective_addMonoid___redArg(v_inst_574_, v_inst_572_, v_inst_577_);
v___x_586_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_586_, 0, v_inst_573_);
lean_ctor_set(v___x_586_, 1, v_inst_575_);
lean_ctor_set(v___x_586_, 2, v___f_582_);
v___x_587_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_587_, 0, v___x_585_);
lean_ctor_set(v___x_587_, 1, v___x_586_);
lean_ctor_set(v___x_587_, 2, v___x_584_);
v___x_588_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_588_, 0, v_inst_570_);
lean_ctor_set(v___x_588_, 1, v_inst_571_);
v___x_589_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_589_, 0, v___x_588_);
lean_ctor_set(v___x_589_, 1, v___f_583_);
v___x_590_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_590_, 0, v___x_587_);
lean_ctor_set(v___x_590_, 1, v___x_589_);
lean_ctor_set(v___x_590_, 2, v_inst_580_);
v___x_591_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_591_, 0, v___x_590_);
lean_ctor_set(v___x_591_, 1, v_inst_581_);
return v___x_591_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_kleeneAlgebra(lean_object* v_00_u03b1_592_, lean_object* v_00_u03b2_593_, lean_object* v_inst_594_, lean_object* v_inst_595_, lean_object* v_inst_596_, lean_object* v_inst_597_, lean_object* v_inst_598_, lean_object* v_inst_599_, lean_object* v_inst_600_, lean_object* v_inst_601_, lean_object* v_inst_602_, lean_object* v_inst_603_, lean_object* v_inst_604_, lean_object* v_inst_605_, lean_object* v_inst_606_, lean_object* v_f_607_, lean_object* v_hf_608_, lean_object* v_le_609_, lean_object* v_lt_610_, lean_object* v_zero_611_, lean_object* v_one_612_, lean_object* v_add_613_, lean_object* v_mul_614_, lean_object* v_nsmul_615_, lean_object* v_npow_616_, lean_object* v_natCast_617_, lean_object* v_sup_618_, lean_object* v_bot_619_, lean_object* v_kstar_620_){
_start:
{
lean_object* v___f_621_; lean_object* v___f_622_; lean_object* v___x_623_; lean_object* v___x_624_; lean_object* v___x_625_; lean_object* v___x_626_; lean_object* v___x_627_; lean_object* v___x_628_; lean_object* v___x_629_; lean_object* v___x_630_; 
v___f_621_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_idemSemiring___redArg___lam__0), 3, 1);
lean_closure_set(v___f_621_, 0, v_inst_601_);
v___f_622_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_idemSemiring___redArg___lam__1), 3, 1);
lean_closure_set(v___f_622_, 0, v_inst_604_);
v___x_623_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_623_, 0, lean_box(0));
lean_closure_set(v___x_623_, 1, v_inst_603_);
v___x_624_ = lp_mathlib_Function_Injective_addMonoid___redArg(v_inst_599_, v_inst_597_, v_inst_602_);
v___x_625_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_625_, 0, v_inst_598_);
lean_ctor_set(v___x_625_, 1, v_inst_600_);
lean_ctor_set(v___x_625_, 2, v___f_621_);
v___x_626_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_626_, 0, v___x_624_);
lean_ctor_set(v___x_626_, 1, v___x_625_);
lean_ctor_set(v___x_626_, 2, v___x_623_);
v___x_627_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_627_, 0, v_inst_595_);
lean_ctor_set(v___x_627_, 1, v_inst_596_);
v___x_628_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_628_, 0, v___x_627_);
lean_ctor_set(v___x_628_, 1, v___f_622_);
v___x_629_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_629_, 0, v___x_626_);
lean_ctor_set(v___x_629_, 1, v___x_628_);
lean_ctor_set(v___x_629_, 2, v_inst_605_);
v___x_630_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_630_, 0, v___x_629_);
lean_ctor_set(v___x_630_, 1, v_inst_606_);
return v___x_630_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_kleeneAlgebra___boxed(lean_object** _args){
lean_object* v_00_u03b1_631_ = _args[0];
lean_object* v_00_u03b2_632_ = _args[1];
lean_object* v_inst_633_ = _args[2];
lean_object* v_inst_634_ = _args[3];
lean_object* v_inst_635_ = _args[4];
lean_object* v_inst_636_ = _args[5];
lean_object* v_inst_637_ = _args[6];
lean_object* v_inst_638_ = _args[7];
lean_object* v_inst_639_ = _args[8];
lean_object* v_inst_640_ = _args[9];
lean_object* v_inst_641_ = _args[10];
lean_object* v_inst_642_ = _args[11];
lean_object* v_inst_643_ = _args[12];
lean_object* v_inst_644_ = _args[13];
lean_object* v_inst_645_ = _args[14];
lean_object* v_f_646_ = _args[15];
lean_object* v_hf_647_ = _args[16];
lean_object* v_le_648_ = _args[17];
lean_object* v_lt_649_ = _args[18];
lean_object* v_zero_650_ = _args[19];
lean_object* v_one_651_ = _args[20];
lean_object* v_add_652_ = _args[21];
lean_object* v_mul_653_ = _args[22];
lean_object* v_nsmul_654_ = _args[23];
lean_object* v_npow_655_ = _args[24];
lean_object* v_natCast_656_ = _args[25];
lean_object* v_sup_657_ = _args[26];
lean_object* v_bot_658_ = _args[27];
lean_object* v_kstar_659_ = _args[28];
_start:
{
lean_object* v_res_660_; 
v_res_660_ = lp_mathlib_Function_Injective_kleeneAlgebra(v_00_u03b1_631_, v_00_u03b2_632_, v_inst_633_, v_inst_634_, v_inst_635_, v_inst_636_, v_inst_637_, v_inst_638_, v_inst_639_, v_inst_640_, v_inst_641_, v_inst_642_, v_inst_643_, v_inst_644_, v_inst_645_, v_f_646_, v_hf_647_, v_le_648_, v_lt_649_, v_zero_650_, v_one_651_, v_add_652_, v_mul_653_, v_nsmul_654_, v_npow_655_, v_natCast_656_, v_sup_657_, v_bot_658_, v_kstar_659_);
lean_dec(v_f_646_);
lean_dec_ref(v_inst_633_);
return v_res_660_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Monoid_Canonical_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_InjSurj(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Pi(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Prod(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Monotonicity_Attr(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Kleene(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Monoid_Canonical_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_InjSurj(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Monotonicity_Attr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Order_Kleene(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_IdemSemiring_add__eq__sup___autoParam = _init_lp_mathlib_IdemSemiring_add__eq__sup___autoParam();
lean_mark_persistent(lp_mathlib_IdemSemiring_add__eq__sup___autoParam);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Monoid_Canonical_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_InjSurj(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Pi(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Monotonicity_Attr(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Order_Kleene(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Monoid_Canonical_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_InjSurj(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Monotonicity_Attr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Kleene(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Order_Kleene(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Order_Kleene(builtin);
}
#ifdef __cplusplus
}
#endif
