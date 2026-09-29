// Lean compiler output
// Module: Mathlib.Tactic.FieldSimp.Lemmas
// Imports: public import Init public meta import Init public import Mathlib.Algebra.BigOperators.Group.List.Basic public import Mathlib.Algebra.Field.Defs public import Mathlib.Algebra.Order.GroupWithZero.Basic public import Mathlib.Algebra.Ring.Int.Parity public meta import Mathlib.Util.Qq
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
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Level_succ___override(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Level_ofNat(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_int_neg(lean_object*);
lean_object* lean_nat_to_int(lean_object*);
uint8_t lean_int_dec_le(lean_object*, lean_object*);
lean_object* l_Int_toNat(lean_object*);
lean_object* l_Lean_instToExprInt_mkNat(lean_object*);
lean_object* l_Lean_mkApp3(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Int_decidableDvd(lean_object*, lean_object*);
lean_object* lp_mathlib_Qq_mkDecideProofQ(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkNatLit(lean_object*);
uint8_t l_Nat_decidable__dvd(lean_object*, lean_object*);
lean_object* lean_int_mul(lean_object*, lean_object*);
lean_object* l_List_mapTR_loop___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF_cons___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF_cons(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "FieldSimp"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "NF"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 9, .m_data = "term_::ᵣ_"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__5_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__5_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__5_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(187, 95, 53, 248, 72, 241, 7, 199)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__5_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__5_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__3_value),LEAN_SCALAR_PTR_LITERAL(183, 56, 233, 168, 186, 6, 216, 255)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__5_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(218, 121, 183, 193, 144, 179, 169, 189)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 5, .m_data = " ::ᵣ "};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__8_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__10_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__11_value),((lean_object*)(((size_t)(101) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__9_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__12_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__5_value),((lean_object*)(((size_t)(100) << 1) | 1)),((lean_object*)(((size_t)(100) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__13_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__14_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63__ = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "cons"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__5_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__6;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(164, 29, 181, 108, 63, 0, 109, 66)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__8_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(187, 95, 53, 248, 72, 241, 7, 199)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__8_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__8_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__3_value),LEAN_SCALAR_PTR_LITERAL(183, 56, 233, 168, 186, 6, 216, 255)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__8_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(248, 157, 233, 41, 159, 17, 134, 135)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "List"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(245, 188, 225, 225, 165, 5, 251, 132)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__11_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(98, 170, 59, 223, 79, 132, 139, 119)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__11_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__12_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__9_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__13_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__15_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__16_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______unexpand__Mathlib__Tactic__FieldSimp__NF__cons__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______unexpand__Mathlib__Tactic__FieldSimp__NF__cons__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______unexpand__Mathlib__Tactic__FieldSimp__NF__cons__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______unexpand__Mathlib__Tactic__FieldSimp__NF__cons__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______unexpand__Mathlib__Tactic__FieldSimp__NF__cons__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______unexpand__Mathlib__Tactic__FieldSimp__NF__cons__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______unexpand__Mathlib__Tactic__FieldSimp__NF__cons__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______unexpand__Mathlib__Tactic__FieldSimp__NF__cons__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______unexpand__Mathlib__Tactic__FieldSimp__NF__cons__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF_instInv___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF_instInv___lam__1(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_FieldSimp_NF_instInv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_FieldSimp_NF_instInv___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF_instInv___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF_instInv___closed__0_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_FieldSimp_NF_instInv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_FieldSimp_NF_instInv___lam__1, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF_instInv___closed__0_value)} };
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF_instInv___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF_instInv___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF_instInv(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF_instPowInt___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF_instPowInt___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF_instPowInt___lam__1(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_FieldSimp_NF_instPowInt___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_FieldSimp_NF_instPowInt___lam__1, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF_instPowInt___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF_instPowInt___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF_instPowInt(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF_instPowNat___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF_instPowNat___lam__1(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_FieldSimp_NF_instPowNat___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_FieldSimp_NF_instPowNat___lam__1, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF_instPowNat___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF_instPowNat___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF_instPowNat(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_ctorIdx___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_ctorIdx___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_ctorIdx(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_ctorIdx___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_plus_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_plus_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_plus_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_minus_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_minus_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_minus_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Neg"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "neg"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(94, 4, 109, 108, 64, 81, 153, 133)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(105, 26, 70, 221, 245, 238, 127, 238)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "NegZeroClass"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "toNeg"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__3_value),LEAN_SCALAR_PTR_LITERAL(156, 44, 233, 53, 1, 106, 24, 217)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__4_value),LEAN_SCALAR_PTR_LITERAL(124, 136, 108, 160, 134, 153, 101, 8)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "SubNegZeroMonoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "toNegZeroClass"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__6_value),LEAN_SCALAR_PTR_LITERAL(135, 233, 160, 34, 207, 245, 132, 138)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__7_value),LEAN_SCALAR_PTR_LITERAL(107, 179, 145, 12, 37, 42, 18, 108)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "SubtractionMonoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "toSubNegZeroMonoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__9_value),LEAN_SCALAR_PTR_LITERAL(203, 24, 17, 79, 61, 156, 198, 150)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__11_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__10_value),LEAN_SCALAR_PTR_LITERAL(94, 234, 159, 237, 9, 124, 201, 94)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "SubtractionCommMonoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "toSubtractionMonoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__12_value),LEAN_SCALAR_PTR_LITERAL(100, 8, 183, 201, 110, 57, 85, 213)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__14_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__13_value),LEAN_SCALAR_PTR_LITERAL(203, 26, 135, 240, 118, 74, 112, 111)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "AddCommGroup"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__15_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "toDivisionAddCommMonoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__17_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__15_value),LEAN_SCALAR_PTR_LITERAL(59, 221, 192, 169, 110, 67, 255, 76)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__17_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__16_value),LEAN_SCALAR_PTR_LITERAL(65, 138, 55, 164, 85, 246, 87, 209)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__17_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Ring"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__18_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "toAddCommGroup"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__20_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__18_value),LEAN_SCALAR_PTR_LITERAL(151, 9, 120, 97, 235, 184, 251, 227)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__20_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__19_value),LEAN_SCALAR_PTR_LITERAL(121, 151, 225, 139, 113, 68, 25, 156)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__20_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "DivisionRing"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__21_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "toRing"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__22_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__23_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__21_value),LEAN_SCALAR_PTR_LITERAL(34, 214, 17, 155, 7, 71, 232, 190)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__23_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__22_value),LEAN_SCALAR_PTR_LITERAL(196, 15, 37, 9, 106, 139, 236, 93)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__23_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Field"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__24_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "toDivisionRing"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__25_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__26_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__24_value),LEAN_SCALAR_PTR_LITERAL(22, 39, 49, 148, 16, 49, 114, 224)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__26_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__25_value),LEAN_SCALAR_PTR_LITERAL(60, 172, 238, 141, 54, 76, 141, 71)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__26_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "HMul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hMul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(254, 113, 255, 140, 142, 9, 169, 40)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(248, 227, 200, 215, 229, 255, 92, 22)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "instHMul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__3_value),LEAN_SCALAR_PTR_LITERAL(177, 107, 107, 59, 202, 230, 169, 251)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "MulZeroClass"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "toMul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(232, 169, 101, 213, 120, 247, 80, 71)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__7_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(14, 61, 8, 113, 227, 149, 226, 31)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "MulZeroOneClass"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "toMulZeroClass"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__8_value),LEAN_SCALAR_PTR_LITERAL(175, 32, 159, 62, 158, 163, 7, 102)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__10_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__9_value),LEAN_SCALAR_PTR_LITERAL(173, 172, 244, 29, 179, 60, 222, 203)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "MonoidWithZero"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "toMulZeroOneClass"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__13_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__11_value),LEAN_SCALAR_PTR_LITERAL(31, 173, 37, 182, 97, 174, 117, 138)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__13_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__12_value),LEAN_SCALAR_PTR_LITERAL(49, 123, 33, 227, 129, 160, 155, 144)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__13_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "GroupWithZero"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "toMonoidWithZero"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__14_value),LEAN_SCALAR_PTR_LITERAL(55, 132, 4, 209, 65, 207, 153, 53)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__16_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__15_value),LEAN_SCALAR_PTR_LITERAL(109, 12, 234, 152, 248, 161, 168, 7)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__16_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "CommGroupWithZero"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__17_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "toGroupWithZero"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__19_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__17_value),LEAN_SCALAR_PTR_LITERAL(197, 14, 145, 254, 35, 172, 249, 164)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__19_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__18_value),LEAN_SCALAR_PTR_LITERAL(227, 250, 20, 68, 202, 245, 131, 209)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__19_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "rfl"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__20_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__20_value),LEAN_SCALAR_PTR_LITERAL(77, 42, 253, 71, 61, 132, 173, 240)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__21_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Eq"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__22_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "symm"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__23_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__24_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__22_value),LEAN_SCALAR_PTR_LITERAL(143, 37, 101, 248, 9, 246, 191, 223)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__24_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__23_value),LEAN_SCALAR_PTR_LITERAL(220, 149, 144, 59, 77, 93, 25, 217)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__24_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "mul_neg"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__25_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__25_value),LEAN_SCALAR_PTR_LITERAL(3, 55, 76, 46, 6, 171, 223, 243)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__26_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Distrib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__27_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__28_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__27_value),LEAN_SCALAR_PTR_LITERAL(221, 154, 114, 180, 95, 113, 104, 161)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__28_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(159, 190, 95, 162, 187, 73, 156, 147)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__28 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__28_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "instDistribOfSemiring"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__29_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__29_value),LEAN_SCALAR_PTR_LITERAL(208, 10, 80, 43, 19, 152, 244, 119)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__30 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__30_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "CommSemiring"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__31 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__31_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "toSemiring"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__32 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__32_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__33_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__31_value),LEAN_SCALAR_PTR_LITERAL(22, 69, 197, 205, 197, 50, 81, 124)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__33_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__32_value),LEAN_SCALAR_PTR_LITERAL(1, 71, 172, 115, 76, 22, 6, 37)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__33 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__33_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Semifield"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__34 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__34_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "toCommSemiring"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__35 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__35_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__36_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__34_value),LEAN_SCALAR_PTR_LITERAL(205, 214, 159, 28, 31, 185, 116, 83)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__36_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__35_value),LEAN_SCALAR_PTR_LITERAL(134, 142, 86, 147, 34, 154, 178, 196)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__36 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__36_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "toSemifield"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__37 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__37_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__38_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__24_value),LEAN_SCALAR_PTR_LITERAL(22, 39, 49, 148, 16, 49, 114, 224)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__38_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__37_value),LEAN_SCALAR_PTR_LITERAL(104, 104, 95, 86, 153, 146, 86, 112)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__38 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__38_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "NonUnitalNonAssocRing"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__39 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__39_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "toHasDistribNeg"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__40 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__40_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__41_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__39_value),LEAN_SCALAR_PTR_LITERAL(228, 161, 95, 44, 207, 172, 118, 5)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__41_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__40_value),LEAN_SCALAR_PTR_LITERAL(1, 137, 128, 244, 245, 214, 238, 122)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__41 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__41_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "NonUnitalNonAssocCommRing"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__42 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__42_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "toNonUnitalNonAssocRing"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__43 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__43_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__44_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__42_value),LEAN_SCALAR_PTR_LITERAL(29, 250, 88, 168, 244, 166, 252, 128)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__44_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__44_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__43_value),LEAN_SCALAR_PTR_LITERAL(12, 184, 133, 121, 11, 96, 21, 164)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__44 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__44_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__45_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "NonUnitalCommRing"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__45 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__45_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 28, .m_capacity = 28, .m_length = 27, .m_data = "toNonUnitalNonAssocCommRing"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__46 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__46_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__47_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__45_value),LEAN_SCALAR_PTR_LITERAL(67, 19, 140, 213, 209, 177, 78, 98)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__47_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__47_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__46_value),LEAN_SCALAR_PTR_LITERAL(1, 226, 200, 0, 68, 127, 171, 18)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__47 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__47_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__48_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "CommRing"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__48 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__48_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__49_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "toNonUnitalCommRing"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__49 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__49_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__50_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__48_value),LEAN_SCALAR_PTR_LITERAL(78, 130, 181, 61, 179, 129, 164, 15)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__50_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__50_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__49_value),LEAN_SCALAR_PTR_LITERAL(197, 98, 241, 160, 219, 182, 36, 24)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__50 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__50_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__51_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "toCommRing"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__51 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__51_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__52_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__24_value),LEAN_SCALAR_PTR_LITERAL(22, 39, 49, 148, 16, 49, 114, 224)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__52_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__52_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__51_value),LEAN_SCALAR_PTR_LITERAL(80, 133, 194, 203, 22, 179, 103, 113)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__52 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__52_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mul___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "neg_mul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mul___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mul___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mul___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mul___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(92, 17, 94, 137, 199, 27, 147, 192)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mul___redArg___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mul___redArg___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mul___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "neg_mul_neg"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mul___redArg___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mul___redArg___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mul___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mul___redArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(239, 22, 245, 116, 176, 15, 227, 170)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mul___redArg___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mul___redArg___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mul___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mul___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Inv"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "inv"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(142, 68, 231, 210, 96, 163, 154, 19)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(63, 31, 248, 222, 13, 64, 40, 141)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "InvOneClass"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "toInv"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__3_value),LEAN_SCALAR_PTR_LITERAL(120, 190, 7, 179, 62, 236, 21, 116)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(28, 25, 248, 9, 15, 85, 72, 194)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "DivInvOneMonoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "toInvOneClass"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(162, 155, 123, 0, 237, 243, 28, 65)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__7_value),LEAN_SCALAR_PTR_LITERAL(181, 224, 200, 199, 184, 130, 54, 26)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "DivisionMonoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "toDivInvOneMonoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__9_value),LEAN_SCALAR_PTR_LITERAL(16, 242, 184, 157, 107, 26, 18, 78)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__11_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__10_value),LEAN_SCALAR_PTR_LITERAL(60, 63, 43, 77, 240, 6, 89, 70)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "DivisionCommMonoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "toDivisionMonoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__12_value),LEAN_SCALAR_PTR_LITERAL(24, 252, 206, 54, 37, 44, 48, 53)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__14_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__13_value),LEAN_SCALAR_PTR_LITERAL(149, 21, 120, 191, 172, 81, 156, 24)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "toDivisionCommMonoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__17_value),LEAN_SCALAR_PTR_LITERAL(197, 14, 145, 254, 35, 172, 249, 164)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__16_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__15_value),LEAN_SCALAR_PTR_LITERAL(94, 143, 233, 228, 60, 239, 1, 188)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__16_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "inv_neg"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__17_value),LEAN_SCALAR_PTR_LITERAL(20, 218, 189, 146, 37, 189, 74, 73)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__18_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "HDiv"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hDiv"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(74, 223, 78, 88, 255, 236, 144, 164)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(26, 183, 188, 240, 156, 118, 170, 84)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "instHDiv"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__3_value),LEAN_SCALAR_PTR_LITERAL(34, 70, 113, 198, 157, 211, 131, 18)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "DivInvMonoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "toDiv"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(231, 106, 236, 89, 112, 21, 122, 113)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__7_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(95, 209, 62, 72, 37, 30, 170, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "toDivInvMonoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__14_value),LEAN_SCALAR_PTR_LITERAL(55, 132, 4, 209, 65, 207, 153, 53)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__9_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__8_value),LEAN_SCALAR_PTR_LITERAL(172, 54, 25, 155, 165, 99, 150, 23)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "div_neg"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__10_value),LEAN_SCALAR_PTR_LITERAL(100, 196, 151, 147, 56, 93, 65, 148)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "neg_div"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__12_value),LEAN_SCALAR_PTR_LITERAL(157, 113, 5, 75, 97, 109, 23, 18)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__13_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "neg_div_neg_eq"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__14_value),LEAN_SCALAR_PTR_LITERAL(110, 100, 50, 183, 118, 66, 27, 131)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__15_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_neg___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "neg_neg"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_neg___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_neg___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_neg___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_neg___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(74, 57, 79, 148, 171, 238, 100, 82)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_neg___redArg___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_neg___redArg___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_neg___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "HasDistribNeg"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_neg___redArg___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_neg___redArg___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_neg___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "toInvolutiveNeg"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_neg___redArg___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_neg___redArg___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_neg___redArg___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_neg___redArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(9, 38, 157, 210, 21, 51, 121, 178)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_neg___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_neg___redArg___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_neg___redArg___closed__3_value),LEAN_SCALAR_PTR_LITERAL(23, 188, 204, 45, 134, 6, 169, 204)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_neg___redArg___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_neg___redArg___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_neg___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "NonUnitalNonAssocSemiring"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_neg___redArg___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_neg___redArg___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_neg___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "toDistrib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_neg___redArg___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_neg___redArg___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_neg___redArg___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_neg___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(193, 210, 66, 189, 16, 227, 173, 215)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_neg___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_neg___redArg___closed__7_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_neg___redArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(175, 228, 210, 241, 147, 121, 74, 183)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_neg___redArg___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_neg___redArg___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_neg___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 28, .m_capacity = 28, .m_length = 27, .m_data = "toNonUnitalNonAssocSemiring"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_neg___redArg___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_neg___redArg___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_neg___redArg___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__39_value),LEAN_SCALAR_PTR_LITERAL(228, 161, 95, 44, 207, 172, 118, 5)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_neg___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_neg___redArg___closed__9_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_neg___redArg___closed__8_value),LEAN_SCALAR_PTR_LITERAL(199, 79, 156, 187, 183, 171, 253, 86)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_neg___redArg___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_neg___redArg___closed__9_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_neg___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_neg___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_neg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_neg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "HPow"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hPow"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__0_value),LEAN_SCALAR_PTR_LITERAL(155, 188, 136, 200, 106, 253, 76, 178)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__1_value),LEAN_SCALAR_PTR_LITERAL(32, 63, 208, 57, 56, 184, 164, 144)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Nat"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__3_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__5;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "instHPow"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__6_value),LEAN_SCALAR_PTR_LITERAL(213, 197, 76, 235, 199, 0, 254, 199)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "NPow"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "toPow"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__9_value),LEAN_SCALAR_PTR_LITERAL(39, 79, 240, 225, 164, 207, 253, 237)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__11_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__10_value),LEAN_SCALAR_PTR_LITERAL(56, 108, 173, 227, 4, 14, 173, 115)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Monoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "toNPow"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__12_value),LEAN_SCALAR_PTR_LITERAL(162, 147, 2, 115, 233, 179, 113, 5)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__14_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__13_value),LEAN_SCALAR_PTR_LITERAL(224, 31, 132, 245, 47, 70, 119, 231)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "toMonoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__11_value),LEAN_SCALAR_PTR_LITERAL(31, 173, 37, 182, 97, 174, 117, 138)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__16_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__15_value),LEAN_SCALAR_PTR_LITERAL(148, 250, 186, 185, 105, 194, 144, 240)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__16_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Odd"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__17_value),LEAN_SCALAR_PTR_LITERAL(155, 77, 47, 151, 166, 191, 78, 53)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__18_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__19;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__20;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "instSemiring"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__21_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__22_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__3_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__22_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__21_value),LEAN_SCALAR_PTR_LITERAL(165, 46, 171, 214, 253, 53, 167, 185)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__22_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__23;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__24;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "neg_pow"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__25_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__26_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__17_value),LEAN_SCALAR_PTR_LITERAL(155, 77, 47, 151, 166, 191, 78, 53)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__26_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__25_value),LEAN_SCALAR_PTR_LITERAL(103, 227, 89, 164, 180, 21, 129, 5)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__26_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Semiring"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__27_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__28_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__27_value),LEAN_SCALAR_PTR_LITERAL(37, 127, 172, 14, 25, 240, 239, 179)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__28_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__15_value),LEAN_SCALAR_PTR_LITERAL(86, 172, 133, 187, 121, 84, 206, 170)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__28 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__28_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Even"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__29_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__29_value),LEAN_SCALAR_PTR_LITERAL(105, 47, 181, 124, 26, 189, 209, 164)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__30 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__30_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__31_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__31;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__32_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__32;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "instAddNat"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__33 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__33_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__33_value),LEAN_SCALAR_PTR_LITERAL(228, 164, 175, 25, 228, 165, 175, 183)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__34 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__34_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__35_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__35;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__36_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__36;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__37_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__29_value),LEAN_SCALAR_PTR_LITERAL(105, 47, 181, 124, 26, 189, 209, 164)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__37_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__25_value),LEAN_SCALAR_PTR_LITERAL(237, 1, 20, 142, 164, 203, 34, 50)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__37 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__37_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Int"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__0_value),LEAN_SCALAR_PTR_LITERAL(61, 25, 98, 154, 117, 127, 69, 97)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__2;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "ZPow"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__3_value),LEAN_SCALAR_PTR_LITERAL(73, 207, 245, 197, 62, 206, 208, 46)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__10_value),LEAN_SCALAR_PTR_LITERAL(110, 230, 127, 66, 154, 153, 82, 195)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "toZPow"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(231, 106, 236, 89, 112, 21, 122, 113)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__6_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__5_value),LEAN_SCALAR_PTR_LITERAL(128, 241, 50, 125, 164, 95, 147, 60)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__6_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__7;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__8;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__9;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__10;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "instNegInt"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__12_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__0_value),LEAN_SCALAR_PTR_LITERAL(61, 25, 98, 154, 117, 127, 69, 97)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__12_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__11_value),LEAN_SCALAR_PTR_LITERAL(217, 109, 233, 1, 211, 122, 77, 88)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__12_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__13;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__14;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__15;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__0_value),LEAN_SCALAR_PTR_LITERAL(61, 25, 98, 154, 117, 127, 69, 97)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__16_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__21_value),LEAN_SCALAR_PTR_LITERAL(67, 124, 140, 177, 178, 178, 136, 246)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__16_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__17;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__18;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "neg_zpow"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__20_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__17_value),LEAN_SCALAR_PTR_LITERAL(155, 77, 47, 151, 166, 191, 78, 53)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__20_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__19_value),LEAN_SCALAR_PTR_LITERAL(228, 189, 5, 85, 192, 50, 244, 89)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__20_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__21;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "instAdd"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__22_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__23_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__0_value),LEAN_SCALAR_PTR_LITERAL(61, 25, 98, 154, 117, 127, 69, 97)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__23_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__22_value),LEAN_SCALAR_PTR_LITERAL(142, 99, 69, 75, 84, 154, 200, 179)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__23_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__24;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__25_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__25;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__26_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__29_value),LEAN_SCALAR_PTR_LITERAL(105, 47, 181, 124, 26, 189, 209, 164)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__26_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__19_value),LEAN_SCALAR_PTR_LITERAL(158, 217, 127, 131, 254, 103, 209, 177)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__26_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_congr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "congr_arg"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_congr___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_congr___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_congr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_congr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(163, 213, 122, 158, 47, 83, 209, 122)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_congr___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_congr___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_congr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mkEqMul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mkEqMul___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "eq_mul_of_eq_eq_eq_mul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mkEqMul___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mkEqMul___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mkEqMul___redArg___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mkEqMul___redArg___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mkEqMul___redArg___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mkEqMul___redArg___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mkEqMul___redArg___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(187, 95, 53, 248, 72, 241, 7, 199)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mkEqMul___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mkEqMul___redArg___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mkEqMul___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(124, 228, 90, 182, 111, 113, 96, 198)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mkEqMul___redArg___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mkEqMul___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mkEqMul___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mkEqMul___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mkEqMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mkEqMul___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF_cons___redArg(lean_object* v_p_1_, lean_object* v_l_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3_, 0, v_p_1_);
lean_ctor_set(v___x_3_, 1, v_l_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF_cons(lean_object* v_M_4_, lean_object* v_p_5_, lean_object* v_l_6_){
_start:
{
lean_object* v___x_7_; 
v___x_7_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_7_, 0, v_p_5_);
lean_ctor_set(v___x_7_, 1, v_l_6_);
return v___x_7_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__6(void){
_start:
{
lean_object* v___x_50_; lean_object* v___x_51_; 
v___x_50_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__5));
v___x_51_ = l_String_toRawSubstring_x27(v___x_50_);
return v___x_51_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1(lean_object* v_x_78_, lean_object* v_a_79_, lean_object* v_a_80_){
_start:
{
lean_object* v___x_81_; uint8_t v___x_82_; 
v___x_81_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__5));
lean_inc(v_x_78_);
v___x_82_ = l_Lean_Syntax_isOfKind(v_x_78_, v___x_81_);
if (v___x_82_ == 0)
{
lean_object* v___x_83_; lean_object* v___x_84_; 
lean_dec(v_x_78_);
v___x_83_ = lean_box(1);
v___x_84_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_84_, 0, v___x_83_);
lean_ctor_set(v___x_84_, 1, v_a_80_);
return v___x_84_;
}
else
{
lean_object* v_quotContext_85_; lean_object* v_currMacroScope_86_; lean_object* v_ref_87_; lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; uint8_t v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; lean_object* v___x_98_; lean_object* v___x_99_; lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; lean_object* v___x_103_; 
v_quotContext_85_ = lean_ctor_get(v_a_79_, 1);
v_currMacroScope_86_ = lean_ctor_get(v_a_79_, 2);
v_ref_87_ = lean_ctor_get(v_a_79_, 5);
v___x_88_ = lean_unsigned_to_nat(0u);
v___x_89_ = l_Lean_Syntax_getArg(v_x_78_, v___x_88_);
v___x_90_ = lean_unsigned_to_nat(2u);
v___x_91_ = l_Lean_Syntax_getArg(v_x_78_, v___x_90_);
lean_dec(v_x_78_);
v___x_92_ = 0;
v___x_93_ = l_Lean_SourceInfo_fromRef(v_ref_87_, v___x_92_);
v___x_94_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__4));
v___x_95_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__6, &lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__6);
v___x_96_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__7));
lean_inc(v_currMacroScope_86_);
lean_inc(v_quotContext_85_);
v___x_97_ = l_Lean_addMacroScope(v_quotContext_85_, v___x_96_, v_currMacroScope_86_);
v___x_98_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__14));
lean_inc_n(v___x_93_, 2);
v___x_99_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_99_, 0, v___x_93_);
lean_ctor_set(v___x_99_, 1, v___x_95_);
lean_ctor_set(v___x_99_, 2, v___x_97_);
lean_ctor_set(v___x_99_, 3, v___x_98_);
v___x_100_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__16));
v___x_101_ = l_Lean_Syntax_node2(v___x_93_, v___x_100_, v___x_89_, v___x_91_);
v___x_102_ = l_Lean_Syntax_node2(v___x_93_, v___x_94_, v___x_99_, v___x_101_);
v___x_103_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_103_, 0, v___x_102_);
lean_ctor_set(v___x_103_, 1, v_a_80_);
return v___x_103_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___boxed(lean_object* v_x_104_, lean_object* v_a_105_, lean_object* v_a_106_){
_start:
{
lean_object* v_res_107_; 
v_res_107_ = lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1(v_x_104_, v_a_105_, v_a_106_);
lean_dec_ref(v_a_105_);
return v_res_107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______unexpand__Mathlib__Tactic__FieldSimp__NF__cons__1(lean_object* v_x_111_, lean_object* v_a_112_, lean_object* v_a_113_){
_start:
{
lean_object* v___x_114_; uint8_t v___x_115_; 
v___x_114_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______macroRules__Mathlib__Tactic__FieldSimp__NF__term___x3a_x3a_u1d63____1___closed__4));
lean_inc(v_x_111_);
v___x_115_ = l_Lean_Syntax_isOfKind(v_x_111_, v___x_114_);
if (v___x_115_ == 0)
{
lean_object* v___x_116_; lean_object* v___x_117_; 
lean_dec(v_x_111_);
v___x_116_ = lean_box(0);
v___x_117_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_117_, 0, v___x_116_);
lean_ctor_set(v___x_117_, 1, v_a_113_);
return v___x_117_;
}
else
{
lean_object* v___x_118_; lean_object* v___x_119_; lean_object* v___x_120_; uint8_t v___x_121_; 
v___x_118_ = lean_unsigned_to_nat(0u);
v___x_119_ = l_Lean_Syntax_getArg(v_x_111_, v___x_118_);
v___x_120_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______unexpand__Mathlib__Tactic__FieldSimp__NF__cons__1___closed__1));
lean_inc(v___x_119_);
v___x_121_ = l_Lean_Syntax_isOfKind(v___x_119_, v___x_120_);
if (v___x_121_ == 0)
{
lean_object* v___x_122_; lean_object* v___x_123_; 
lean_dec(v___x_119_);
lean_dec(v_x_111_);
v___x_122_ = lean_box(0);
v___x_123_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_123_, 0, v___x_122_);
lean_ctor_set(v___x_123_, 1, v_a_113_);
return v___x_123_;
}
else
{
lean_object* v___x_124_; lean_object* v___x_125_; lean_object* v___x_126_; uint8_t v___x_127_; 
v___x_124_ = lean_unsigned_to_nat(1u);
v___x_125_ = l_Lean_Syntax_getArg(v_x_111_, v___x_124_);
lean_dec(v_x_111_);
v___x_126_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_125_);
v___x_127_ = l_Lean_Syntax_matchesNull(v___x_125_, v___x_126_);
if (v___x_127_ == 0)
{
lean_object* v___x_128_; lean_object* v___x_129_; 
lean_dec(v___x_125_);
lean_dec(v___x_119_);
v___x_128_ = lean_box(0);
v___x_129_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_129_, 0, v___x_128_);
lean_ctor_set(v___x_129_, 1, v_a_113_);
return v___x_129_;
}
else
{
lean_object* v___x_130_; lean_object* v___x_131_; lean_object* v_ref_132_; uint8_t v___x_133_; lean_object* v___x_134_; lean_object* v___x_135_; lean_object* v___x_136_; lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; 
v___x_130_ = l_Lean_Syntax_getArg(v___x_125_, v___x_118_);
v___x_131_ = l_Lean_Syntax_getArg(v___x_125_, v___x_124_);
lean_dec(v___x_125_);
v_ref_132_ = l_Lean_replaceRef(v___x_119_, v_a_112_);
lean_dec(v___x_119_);
v___x_133_ = 0;
v___x_134_ = l_Lean_SourceInfo_fromRef(v_ref_132_, v___x_133_);
lean_dec(v_ref_132_);
v___x_135_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__5));
v___x_136_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_NF_term___x3a_x3a_u1d63___00__closed__8));
lean_inc(v___x_134_);
v___x_137_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_137_, 0, v___x_134_);
lean_ctor_set(v___x_137_, 1, v___x_136_);
v___x_138_ = l_Lean_Syntax_node3(v___x_134_, v___x_135_, v___x_130_, v___x_137_, v___x_131_);
v___x_139_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_139_, 0, v___x_138_);
lean_ctor_set(v___x_139_, 1, v_a_113_);
return v___x_139_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______unexpand__Mathlib__Tactic__FieldSimp__NF__cons__1___boxed(lean_object* v_x_140_, lean_object* v_a_141_, lean_object* v_a_142_){
_start:
{
lean_object* v_res_143_; 
v_res_143_ = lp_mathlib_Mathlib_Tactic_FieldSimp_NF___aux__Mathlib__Tactic__FieldSimp__Lemmas______unexpand__Mathlib__Tactic__FieldSimp__NF__cons__1(v_x_140_, v_a_141_, v_a_142_);
lean_dec(v_a_141_);
return v_res_143_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF_instInv___lam__0(lean_object* v_x_144_){
_start:
{
lean_object* v_fst_145_; lean_object* v_snd_146_; lean_object* v___x_148_; uint8_t v_isShared_149_; uint8_t v_isSharedCheck_154_; 
v_fst_145_ = lean_ctor_get(v_x_144_, 0);
v_snd_146_ = lean_ctor_get(v_x_144_, 1);
v_isSharedCheck_154_ = !lean_is_exclusive(v_x_144_);
if (v_isSharedCheck_154_ == 0)
{
v___x_148_ = v_x_144_;
v_isShared_149_ = v_isSharedCheck_154_;
goto v_resetjp_147_;
}
else
{
lean_inc(v_snd_146_);
lean_inc(v_fst_145_);
lean_dec(v_x_144_);
v___x_148_ = lean_box(0);
v_isShared_149_ = v_isSharedCheck_154_;
goto v_resetjp_147_;
}
v_resetjp_147_:
{
lean_object* v___x_150_; lean_object* v___x_152_; 
v___x_150_ = lean_int_neg(v_fst_145_);
lean_dec(v_fst_145_);
if (v_isShared_149_ == 0)
{
lean_ctor_set(v___x_148_, 0, v___x_150_);
v___x_152_ = v___x_148_;
goto v_reusejp_151_;
}
else
{
lean_object* v_reuseFailAlloc_153_; 
v_reuseFailAlloc_153_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_153_, 0, v___x_150_);
lean_ctor_set(v_reuseFailAlloc_153_, 1, v_snd_146_);
v___x_152_ = v_reuseFailAlloc_153_;
goto v_reusejp_151_;
}
v_reusejp_151_:
{
return v___x_152_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF_instInv___lam__1(lean_object* v___f_155_, lean_object* v_l_156_){
_start:
{
lean_object* v___x_157_; lean_object* v___x_158_; 
v___x_157_ = lean_box(0);
v___x_158_ = l_List_mapTR_loop___redArg(v___f_155_, v_l_156_, v___x_157_);
return v___x_158_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF_instInv(lean_object* v_M_162_){
_start:
{
lean_object* v___f_163_; 
v___f_163_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_NF_instInv___closed__1));
return v___f_163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF_instPowInt___lam__0(lean_object* v_r_164_, lean_object* v_x_165_){
_start:
{
lean_object* v_fst_166_; lean_object* v_snd_167_; lean_object* v___x_169_; uint8_t v_isShared_170_; uint8_t v_isSharedCheck_175_; 
v_fst_166_ = lean_ctor_get(v_x_165_, 0);
v_snd_167_ = lean_ctor_get(v_x_165_, 1);
v_isSharedCheck_175_ = !lean_is_exclusive(v_x_165_);
if (v_isSharedCheck_175_ == 0)
{
v___x_169_ = v_x_165_;
v_isShared_170_ = v_isSharedCheck_175_;
goto v_resetjp_168_;
}
else
{
lean_inc(v_snd_167_);
lean_inc(v_fst_166_);
lean_dec(v_x_165_);
v___x_169_ = lean_box(0);
v_isShared_170_ = v_isSharedCheck_175_;
goto v_resetjp_168_;
}
v_resetjp_168_:
{
lean_object* v___x_171_; lean_object* v___x_173_; 
v___x_171_ = lean_int_mul(v_r_164_, v_fst_166_);
lean_dec(v_fst_166_);
if (v_isShared_170_ == 0)
{
lean_ctor_set(v___x_169_, 0, v___x_171_);
v___x_173_ = v___x_169_;
goto v_reusejp_172_;
}
else
{
lean_object* v_reuseFailAlloc_174_; 
v_reuseFailAlloc_174_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_174_, 0, v___x_171_);
lean_ctor_set(v_reuseFailAlloc_174_, 1, v_snd_167_);
v___x_173_ = v_reuseFailAlloc_174_;
goto v_reusejp_172_;
}
v_reusejp_172_:
{
return v___x_173_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF_instPowInt___lam__0___boxed(lean_object* v_r_176_, lean_object* v_x_177_){
_start:
{
lean_object* v_res_178_; 
v_res_178_ = lp_mathlib_Mathlib_Tactic_FieldSimp_NF_instPowInt___lam__0(v_r_176_, v_x_177_);
lean_dec(v_r_176_);
return v_res_178_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF_instPowInt___lam__1(lean_object* v_l_179_, lean_object* v_r_180_){
_start:
{
lean_object* v___f_181_; lean_object* v___x_182_; lean_object* v___x_183_; 
v___f_181_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_FieldSimp_NF_instPowInt___lam__0___boxed), 2, 1);
lean_closure_set(v___f_181_, 0, v_r_180_);
v___x_182_ = lean_box(0);
v___x_183_ = l_List_mapTR_loop___redArg(v___f_181_, v_l_179_, v___x_182_);
return v___x_183_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF_instPowInt(lean_object* v_M_185_){
_start:
{
lean_object* v___f_186_; 
v___f_186_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_NF_instPowInt___closed__0));
return v___f_186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF_instPowNat___lam__0(lean_object* v_r_187_, lean_object* v_x_188_){
_start:
{
lean_object* v_fst_189_; lean_object* v_snd_190_; lean_object* v___x_192_; uint8_t v_isShared_193_; uint8_t v_isSharedCheck_199_; 
v_fst_189_ = lean_ctor_get(v_x_188_, 0);
v_snd_190_ = lean_ctor_get(v_x_188_, 1);
v_isSharedCheck_199_ = !lean_is_exclusive(v_x_188_);
if (v_isSharedCheck_199_ == 0)
{
v___x_192_ = v_x_188_;
v_isShared_193_ = v_isSharedCheck_199_;
goto v_resetjp_191_;
}
else
{
lean_inc(v_snd_190_);
lean_inc(v_fst_189_);
lean_dec(v_x_188_);
v___x_192_ = lean_box(0);
v_isShared_193_ = v_isSharedCheck_199_;
goto v_resetjp_191_;
}
v_resetjp_191_:
{
lean_object* v___x_194_; lean_object* v___x_195_; lean_object* v___x_197_; 
v___x_194_ = lean_nat_to_int(v_r_187_);
v___x_195_ = lean_int_mul(v___x_194_, v_fst_189_);
lean_dec(v_fst_189_);
lean_dec(v___x_194_);
if (v_isShared_193_ == 0)
{
lean_ctor_set(v___x_192_, 0, v___x_195_);
v___x_197_ = v___x_192_;
goto v_reusejp_196_;
}
else
{
lean_object* v_reuseFailAlloc_198_; 
v_reuseFailAlloc_198_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_198_, 0, v___x_195_);
lean_ctor_set(v_reuseFailAlloc_198_, 1, v_snd_190_);
v___x_197_ = v_reuseFailAlloc_198_;
goto v_reusejp_196_;
}
v_reusejp_196_:
{
return v___x_197_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF_instPowNat___lam__1(lean_object* v_l_200_, lean_object* v_r_201_){
_start:
{
lean_object* v___f_202_; lean_object* v___x_203_; lean_object* v___x_204_; 
v___f_202_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_FieldSimp_NF_instPowNat___lam__0), 2, 1);
lean_closure_set(v___f_202_, 0, v_r_201_);
v___x_203_ = lean_box(0);
v___x_204_ = l_List_mapTR_loop___redArg(v___f_202_, v_l_200_, v___x_203_);
return v___x_204_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_NF_instPowNat(lean_object* v_M_206_){
_start:
{
lean_object* v___f_207_; 
v___f_207_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_NF_instPowNat___closed__0));
return v___f_207_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_ctorIdx___redArg(lean_object* v_x_208_){
_start:
{
if (lean_obj_tag(v_x_208_) == 0)
{
lean_object* v___x_209_; 
v___x_209_ = lean_unsigned_to_nat(0u);
return v___x_209_;
}
else
{
lean_object* v___x_210_; 
v___x_210_ = lean_unsigned_to_nat(1u);
return v___x_210_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_ctorIdx___redArg___boxed(lean_object* v_x_211_){
_start:
{
lean_object* v_res_212_; 
v_res_212_ = lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_ctorIdx___redArg(v_x_211_);
lean_dec(v_x_211_);
return v_res_212_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_ctorIdx(lean_object* v_v_213_, lean_object* v_M_214_, lean_object* v_x_215_){
_start:
{
lean_object* v___x_216_; 
v___x_216_ = lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_ctorIdx___redArg(v_x_215_);
return v___x_216_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_ctorIdx___boxed(lean_object* v_v_217_, lean_object* v_M_218_, lean_object* v_x_219_){
_start:
{
lean_object* v_res_220_; 
v_res_220_ = lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_ctorIdx(v_v_217_, v_M_218_, v_x_219_);
lean_dec(v_x_219_);
lean_dec_ref(v_M_218_);
lean_dec(v_v_217_);
return v_res_220_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_ctorElim___redArg(lean_object* v_t_221_, lean_object* v_k_222_){
_start:
{
if (lean_obj_tag(v_t_221_) == 0)
{
return v_k_222_;
}
else
{
lean_object* v_iM_223_; lean_object* v___x_224_; 
v_iM_223_ = lean_ctor_get(v_t_221_, 0);
lean_inc_ref(v_iM_223_);
lean_dec_ref_known(v_t_221_, 1);
v___x_224_ = lean_apply_1(v_k_222_, v_iM_223_);
return v___x_224_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_ctorElim(lean_object* v_v_225_, lean_object* v_M_226_, lean_object* v_motive_227_, lean_object* v_ctorIdx_228_, lean_object* v_t_229_, lean_object* v_h_230_, lean_object* v_k_231_){
_start:
{
lean_object* v___x_232_; 
v___x_232_ = lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_ctorElim___redArg(v_t_229_, v_k_231_);
return v___x_232_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_ctorElim___boxed(lean_object* v_v_233_, lean_object* v_M_234_, lean_object* v_motive_235_, lean_object* v_ctorIdx_236_, lean_object* v_t_237_, lean_object* v_h_238_, lean_object* v_k_239_){
_start:
{
lean_object* v_res_240_; 
v_res_240_ = lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_ctorElim(v_v_233_, v_M_234_, v_motive_235_, v_ctorIdx_236_, v_t_237_, v_h_238_, v_k_239_);
lean_dec(v_ctorIdx_236_);
lean_dec_ref(v_M_234_);
lean_dec(v_v_233_);
return v_res_240_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_plus_elim___redArg(lean_object* v_t_241_, lean_object* v_plus_242_){
_start:
{
lean_object* v___x_243_; 
v___x_243_ = lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_ctorElim___redArg(v_t_241_, v_plus_242_);
return v___x_243_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_plus_elim(lean_object* v_v_244_, lean_object* v_M_245_, lean_object* v_motive_246_, lean_object* v_t_247_, lean_object* v_h_248_, lean_object* v_plus_249_){
_start:
{
lean_object* v___x_250_; 
v___x_250_ = lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_ctorElim___redArg(v_t_247_, v_plus_249_);
return v___x_250_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_plus_elim___boxed(lean_object* v_v_251_, lean_object* v_M_252_, lean_object* v_motive_253_, lean_object* v_t_254_, lean_object* v_h_255_, lean_object* v_plus_256_){
_start:
{
lean_object* v_res_257_; 
v_res_257_ = lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_plus_elim(v_v_251_, v_M_252_, v_motive_253_, v_t_254_, v_h_255_, v_plus_256_);
lean_dec_ref(v_M_252_);
lean_dec(v_v_251_);
return v_res_257_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_minus_elim___redArg(lean_object* v_t_258_, lean_object* v_minus_259_){
_start:
{
lean_object* v___x_260_; 
v___x_260_ = lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_ctorElim___redArg(v_t_258_, v_minus_259_);
return v___x_260_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_minus_elim(lean_object* v_v_261_, lean_object* v_M_262_, lean_object* v_motive_263_, lean_object* v_t_264_, lean_object* v_h_265_, lean_object* v_minus_266_){
_start:
{
lean_object* v___x_267_; 
v___x_267_ = lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_ctorElim___redArg(v_t_264_, v_minus_266_);
return v___x_267_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_minus_elim___boxed(lean_object* v_v_268_, lean_object* v_M_269_, lean_object* v_motive_270_, lean_object* v_t_271_, lean_object* v_h_272_, lean_object* v_minus_273_){
_start:
{
lean_object* v_res_274_; 
v_res_274_ = lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_minus_elim(v_v_268_, v_M_269_, v_motive_270_, v_t_271_, v_h_272_, v_minus_273_);
lean_dec_ref(v_M_269_);
lean_dec(v_v_268_);
return v_res_274_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr(lean_object* v_v_320_, lean_object* v_M_321_, lean_object* v_x_322_, lean_object* v_x_323_){
_start:
{
if (lean_obj_tag(v_x_322_) == 0)
{
lean_dec_ref(v_M_321_);
lean_dec(v_v_320_);
return v_x_323_;
}
else
{
lean_object* v_iM_324_; lean_object* v___x_325_; lean_object* v___x_326_; lean_object* v___x_327_; lean_object* v___x_328_; lean_object* v___x_329_; lean_object* v___x_330_; lean_object* v___x_331_; lean_object* v___x_332_; lean_object* v___x_333_; lean_object* v___x_334_; lean_object* v___x_335_; lean_object* v___x_336_; lean_object* v___x_337_; lean_object* v___x_338_; lean_object* v___x_339_; lean_object* v___x_340_; lean_object* v___x_341_; lean_object* v___x_342_; lean_object* v___x_343_; lean_object* v___x_344_; lean_object* v___x_345_; lean_object* v___x_346_; lean_object* v___x_347_; lean_object* v___x_348_; lean_object* v___x_349_; lean_object* v___x_350_; lean_object* v___x_351_; lean_object* v___x_352_; lean_object* v___x_353_; lean_object* v___x_354_; lean_object* v___x_355_; lean_object* v___x_356_; lean_object* v___x_357_; lean_object* v___x_358_; lean_object* v___x_359_; lean_object* v___x_360_; lean_object* v___x_361_; lean_object* v___x_362_; lean_object* v___x_363_; 
v_iM_324_ = lean_ctor_get(v_x_322_, 0);
lean_inc_ref(v_iM_324_);
lean_dec_ref_known(v_x_322_, 1);
v___x_325_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__2));
v___x_326_ = lean_box(0);
v___x_327_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_327_, 0, v_v_320_);
lean_ctor_set(v___x_327_, 1, v___x_326_);
lean_inc_ref_n(v___x_327_, 8);
v___x_328_ = l_Lean_Expr_const___override(v___x_325_, v___x_327_);
lean_inc_ref_n(v_M_321_, 8);
v___x_329_ = l_Lean_Expr_app___override(v___x_328_, v_M_321_);
v___x_330_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__5));
v___x_331_ = l_Lean_Expr_const___override(v___x_330_, v___x_327_);
v___x_332_ = l_Lean_Expr_app___override(v___x_331_, v_M_321_);
v___x_333_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__8));
v___x_334_ = l_Lean_Expr_const___override(v___x_333_, v___x_327_);
v___x_335_ = l_Lean_Expr_app___override(v___x_334_, v_M_321_);
v___x_336_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__11));
v___x_337_ = l_Lean_Expr_const___override(v___x_336_, v___x_327_);
v___x_338_ = l_Lean_Expr_app___override(v___x_337_, v_M_321_);
v___x_339_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__14));
v___x_340_ = l_Lean_Expr_const___override(v___x_339_, v___x_327_);
v___x_341_ = l_Lean_Expr_app___override(v___x_340_, v_M_321_);
v___x_342_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__17));
v___x_343_ = l_Lean_Expr_const___override(v___x_342_, v___x_327_);
v___x_344_ = l_Lean_Expr_app___override(v___x_343_, v_M_321_);
v___x_345_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__20));
v___x_346_ = l_Lean_Expr_const___override(v___x_345_, v___x_327_);
v___x_347_ = l_Lean_Expr_app___override(v___x_346_, v_M_321_);
v___x_348_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__23));
v___x_349_ = l_Lean_Expr_const___override(v___x_348_, v___x_327_);
v___x_350_ = l_Lean_Expr_app___override(v___x_349_, v_M_321_);
v___x_351_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__26));
v___x_352_ = l_Lean_Expr_const___override(v___x_351_, v___x_327_);
v___x_353_ = l_Lean_Expr_app___override(v___x_352_, v_M_321_);
v___x_354_ = l_Lean_Expr_app___override(v___x_353_, v_iM_324_);
v___x_355_ = l_Lean_Expr_app___override(v___x_350_, v___x_354_);
v___x_356_ = l_Lean_Expr_app___override(v___x_347_, v___x_355_);
v___x_357_ = l_Lean_Expr_app___override(v___x_344_, v___x_356_);
v___x_358_ = l_Lean_Expr_app___override(v___x_341_, v___x_357_);
v___x_359_ = l_Lean_Expr_app___override(v___x_338_, v___x_358_);
v___x_360_ = l_Lean_Expr_app___override(v___x_335_, v___x_359_);
v___x_361_ = l_Lean_Expr_app___override(v___x_332_, v___x_360_);
v___x_362_ = l_Lean_Expr_app___override(v___x_329_, v___x_361_);
v___x_363_ = l_Lean_Expr_app___override(v___x_362_, v_x_323_);
return v___x_363_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg(lean_object* v_v_453_, lean_object* v_M_454_, lean_object* v_iM_455_, lean_object* v_c_456_, lean_object* v_y_457_, lean_object* v_g_458_){
_start:
{
if (lean_obj_tag(v_g_458_) == 0)
{
lean_object* v___x_460_; lean_object* v___x_461_; lean_object* v___x_462_; lean_object* v___x_463_; lean_object* v___x_464_; lean_object* v___x_465_; lean_object* v___x_466_; lean_object* v___x_467_; lean_object* v___x_468_; lean_object* v___x_469_; lean_object* v___x_470_; lean_object* v___x_471_; lean_object* v___x_472_; lean_object* v___x_473_; lean_object* v___x_474_; lean_object* v___x_475_; lean_object* v___x_476_; lean_object* v___x_477_; lean_object* v___x_478_; lean_object* v___x_479_; lean_object* v___x_480_; lean_object* v___x_481_; lean_object* v___x_482_; lean_object* v___x_483_; lean_object* v___x_484_; lean_object* v___x_485_; lean_object* v___x_486_; lean_object* v___x_487_; lean_object* v___x_488_; lean_object* v___x_489_; lean_object* v___x_490_; lean_object* v___x_491_; lean_object* v___x_492_; lean_object* v___x_493_; lean_object* v___x_494_; lean_object* v___x_495_; lean_object* v___x_496_; lean_object* v___x_497_; lean_object* v___x_498_; lean_object* v___x_499_; lean_object* v___x_500_; lean_object* v___x_501_; lean_object* v___x_502_; 
lean_inc_n(v_v_453_, 3);
v___x_460_ = l_Lean_Level_succ___override(v_v_453_);
v___x_461_ = lean_box(0);
v___x_462_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_462_, 0, v___x_460_);
lean_ctor_set(v___x_462_, 1, v___x_461_);
v___x_463_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__2));
v___x_464_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_464_, 0, v_v_453_);
lean_ctor_set(v___x_464_, 1, v___x_461_);
lean_inc_ref_n(v___x_464_, 6);
v___x_465_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_465_, 0, v_v_453_);
lean_ctor_set(v___x_465_, 1, v___x_464_);
v___x_466_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_466_, 0, v_v_453_);
lean_ctor_set(v___x_466_, 1, v___x_465_);
v___x_467_ = l_Lean_Expr_const___override(v___x_463_, v___x_466_);
lean_inc_ref_n(v_M_454_, 9);
v___x_468_ = l_Lean_Expr_app___override(v___x_467_, v_M_454_);
v___x_469_ = l_Lean_Expr_app___override(v___x_468_, v_M_454_);
v___x_470_ = l_Lean_Expr_app___override(v___x_469_, v_M_454_);
v___x_471_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__4));
v___x_472_ = l_Lean_Expr_const___override(v___x_471_, v___x_464_);
v___x_473_ = l_Lean_Expr_app___override(v___x_472_, v_M_454_);
v___x_474_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__7));
v___x_475_ = l_Lean_Expr_const___override(v___x_474_, v___x_464_);
v___x_476_ = l_Lean_Expr_app___override(v___x_475_, v_M_454_);
v___x_477_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__10));
v___x_478_ = l_Lean_Expr_const___override(v___x_477_, v___x_464_);
v___x_479_ = l_Lean_Expr_app___override(v___x_478_, v_M_454_);
v___x_480_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__13));
v___x_481_ = l_Lean_Expr_const___override(v___x_480_, v___x_464_);
v___x_482_ = l_Lean_Expr_app___override(v___x_481_, v_M_454_);
v___x_483_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__16));
v___x_484_ = l_Lean_Expr_const___override(v___x_483_, v___x_464_);
v___x_485_ = l_Lean_Expr_app___override(v___x_484_, v_M_454_);
v___x_486_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__19));
v___x_487_ = l_Lean_Expr_const___override(v___x_486_, v___x_464_);
v___x_488_ = l_Lean_Expr_app___override(v___x_487_, v_M_454_);
v___x_489_ = l_Lean_Expr_app___override(v___x_488_, v_iM_455_);
v___x_490_ = l_Lean_Expr_app___override(v___x_485_, v___x_489_);
v___x_491_ = l_Lean_Expr_app___override(v___x_482_, v___x_490_);
v___x_492_ = l_Lean_Expr_app___override(v___x_479_, v___x_491_);
v___x_493_ = l_Lean_Expr_app___override(v___x_476_, v___x_492_);
v___x_494_ = l_Lean_Expr_app___override(v___x_473_, v___x_493_);
v___x_495_ = l_Lean_Expr_app___override(v___x_470_, v___x_494_);
v___x_496_ = l_Lean_Expr_app___override(v___x_495_, v_c_456_);
v___x_497_ = l_Lean_Expr_app___override(v___x_496_, v_y_457_);
v___x_498_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__21));
v___x_499_ = l_Lean_Expr_const___override(v___x_498_, v___x_462_);
v___x_500_ = l_Lean_Expr_app___override(v___x_499_, v_M_454_);
v___x_501_ = l_Lean_Expr_app___override(v___x_500_, v___x_497_);
v___x_502_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_502_, 0, v___x_501_);
return v___x_502_;
}
else
{
lean_object* v_iM_503_; lean_object* v___x_505_; uint8_t v_isShared_506_; uint8_t v_isSharedCheck_640_; 
v_iM_503_ = lean_ctor_get(v_g_458_, 0);
v_isSharedCheck_640_ = !lean_is_exclusive(v_g_458_);
if (v_isSharedCheck_640_ == 0)
{
v___x_505_ = v_g_458_;
v_isShared_506_ = v_isSharedCheck_640_;
goto v_resetjp_504_;
}
else
{
lean_inc(v_iM_503_);
lean_dec(v_g_458_);
v___x_505_ = lean_box(0);
v_isShared_506_ = v_isSharedCheck_640_;
goto v_resetjp_504_;
}
v_resetjp_504_:
{
lean_object* v___x_507_; lean_object* v___x_508_; lean_object* v___x_509_; lean_object* v___x_510_; lean_object* v___x_511_; lean_object* v___x_512_; lean_object* v___x_513_; lean_object* v___x_514_; lean_object* v___x_515_; lean_object* v___x_516_; lean_object* v___x_517_; lean_object* v___x_518_; lean_object* v___x_519_; lean_object* v___x_520_; lean_object* v___x_521_; lean_object* v___x_522_; lean_object* v___x_523_; lean_object* v___x_524_; lean_object* v___x_525_; lean_object* v___x_526_; lean_object* v___x_527_; lean_object* v___x_528_; lean_object* v___x_529_; lean_object* v___x_530_; lean_object* v___x_531_; lean_object* v___x_532_; lean_object* v___x_533_; lean_object* v___x_534_; lean_object* v___x_535_; lean_object* v___x_536_; lean_object* v___x_537_; lean_object* v___x_538_; lean_object* v___x_539_; lean_object* v___x_540_; lean_object* v___x_541_; lean_object* v___x_542_; lean_object* v___x_543_; lean_object* v___x_544_; lean_object* v___x_545_; lean_object* v___x_546_; lean_object* v___x_547_; lean_object* v___x_548_; lean_object* v___x_549_; lean_object* v___x_550_; lean_object* v___x_551_; lean_object* v___x_552_; lean_object* v___x_553_; lean_object* v___x_554_; lean_object* v___x_555_; lean_object* v___x_556_; lean_object* v___x_557_; lean_object* v___x_558_; lean_object* v___x_559_; lean_object* v___x_560_; lean_object* v___x_561_; lean_object* v___x_562_; lean_object* v___x_563_; lean_object* v___x_564_; lean_object* v___x_565_; lean_object* v___x_566_; lean_object* v___x_567_; lean_object* v___x_568_; lean_object* v___x_569_; lean_object* v___x_570_; lean_object* v___x_571_; lean_object* v___x_572_; lean_object* v___x_573_; lean_object* v___x_574_; lean_object* v___x_575_; lean_object* v___x_576_; lean_object* v___x_577_; lean_object* v___x_578_; lean_object* v___x_579_; lean_object* v___x_580_; lean_object* v___x_581_; lean_object* v___x_582_; lean_object* v___x_583_; lean_object* v___x_584_; lean_object* v___x_585_; lean_object* v___x_586_; lean_object* v___x_587_; lean_object* v___x_588_; lean_object* v___x_589_; lean_object* v___x_590_; lean_object* v___x_591_; lean_object* v___x_592_; lean_object* v___x_593_; lean_object* v___x_594_; lean_object* v___x_595_; lean_object* v___x_596_; lean_object* v___x_597_; lean_object* v___x_598_; lean_object* v___x_599_; lean_object* v___x_600_; lean_object* v___x_601_; lean_object* v___x_602_; lean_object* v___x_603_; lean_object* v___x_604_; lean_object* v___x_605_; lean_object* v___x_606_; lean_object* v___x_607_; lean_object* v___x_608_; lean_object* v___x_609_; lean_object* v___x_610_; lean_object* v___x_611_; lean_object* v___x_612_; lean_object* v___x_613_; lean_object* v___x_614_; lean_object* v___x_615_; lean_object* v___x_616_; lean_object* v___x_617_; lean_object* v___x_618_; lean_object* v___x_619_; lean_object* v___x_620_; lean_object* v___x_621_; lean_object* v___x_622_; lean_object* v___x_623_; lean_object* v___x_624_; lean_object* v___x_625_; lean_object* v___x_626_; lean_object* v___x_627_; lean_object* v___x_628_; lean_object* v___x_629_; lean_object* v___x_630_; lean_object* v___x_631_; lean_object* v___x_632_; lean_object* v___x_633_; lean_object* v___x_634_; lean_object* v___x_635_; lean_object* v___x_636_; lean_object* v___x_638_; 
lean_inc_n(v_v_453_, 3);
v___x_507_ = l_Lean_Level_succ___override(v_v_453_);
v___x_508_ = lean_box(0);
v___x_509_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_509_, 0, v___x_507_);
lean_ctor_set(v___x_509_, 1, v___x_508_);
v___x_510_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__2));
v___x_511_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_511_, 0, v_v_453_);
lean_ctor_set(v___x_511_, 1, v___x_508_);
lean_inc_ref_n(v___x_511_, 26);
v___x_512_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_512_, 0, v_v_453_);
lean_ctor_set(v___x_512_, 1, v___x_511_);
v___x_513_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_513_, 0, v_v_453_);
lean_ctor_set(v___x_513_, 1, v___x_512_);
v___x_514_ = l_Lean_Expr_const___override(v___x_510_, v___x_513_);
lean_inc_ref_n(v_M_454_, 29);
v___x_515_ = l_Lean_Expr_app___override(v___x_514_, v_M_454_);
v___x_516_ = l_Lean_Expr_app___override(v___x_515_, v_M_454_);
v___x_517_ = l_Lean_Expr_app___override(v___x_516_, v_M_454_);
v___x_518_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__4));
v___x_519_ = l_Lean_Expr_const___override(v___x_518_, v___x_511_);
v___x_520_ = l_Lean_Expr_app___override(v___x_519_, v_M_454_);
v___x_521_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__7));
v___x_522_ = l_Lean_Expr_const___override(v___x_521_, v___x_511_);
v___x_523_ = l_Lean_Expr_app___override(v___x_522_, v_M_454_);
v___x_524_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__10));
v___x_525_ = l_Lean_Expr_const___override(v___x_524_, v___x_511_);
v___x_526_ = l_Lean_Expr_app___override(v___x_525_, v_M_454_);
v___x_527_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__13));
v___x_528_ = l_Lean_Expr_const___override(v___x_527_, v___x_511_);
v___x_529_ = l_Lean_Expr_app___override(v___x_528_, v_M_454_);
v___x_530_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__16));
v___x_531_ = l_Lean_Expr_const___override(v___x_530_, v___x_511_);
v___x_532_ = l_Lean_Expr_app___override(v___x_531_, v_M_454_);
v___x_533_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__19));
v___x_534_ = l_Lean_Expr_const___override(v___x_533_, v___x_511_);
v___x_535_ = l_Lean_Expr_app___override(v___x_534_, v_M_454_);
v___x_536_ = l_Lean_Expr_app___override(v___x_535_, v_iM_455_);
v___x_537_ = l_Lean_Expr_app___override(v___x_532_, v___x_536_);
v___x_538_ = l_Lean_Expr_app___override(v___x_529_, v___x_537_);
v___x_539_ = l_Lean_Expr_app___override(v___x_526_, v___x_538_);
v___x_540_ = l_Lean_Expr_app___override(v___x_523_, v___x_539_);
v___x_541_ = l_Lean_Expr_app___override(v___x_520_, v___x_540_);
v___x_542_ = l_Lean_Expr_app___override(v___x_517_, v___x_541_);
lean_inc_ref(v_c_456_);
v___x_543_ = l_Lean_Expr_app___override(v___x_542_, v_c_456_);
lean_inc_ref_n(v_y_457_, 2);
lean_inc_ref(v___x_543_);
v___x_544_ = l_Lean_Expr_app___override(v___x_543_, v_y_457_);
v___x_545_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__24));
v___x_546_ = l_Lean_Expr_const___override(v___x_545_, v___x_509_);
v___x_547_ = l_Lean_Expr_app___override(v___x_546_, v_M_454_);
v___x_548_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__2));
v___x_549_ = l_Lean_Expr_const___override(v___x_548_, v___x_511_);
v___x_550_ = l_Lean_Expr_app___override(v___x_549_, v_M_454_);
v___x_551_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__5));
v___x_552_ = l_Lean_Expr_const___override(v___x_551_, v___x_511_);
v___x_553_ = l_Lean_Expr_app___override(v___x_552_, v_M_454_);
v___x_554_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__8));
v___x_555_ = l_Lean_Expr_const___override(v___x_554_, v___x_511_);
v___x_556_ = l_Lean_Expr_app___override(v___x_555_, v_M_454_);
v___x_557_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__11));
v___x_558_ = l_Lean_Expr_const___override(v___x_557_, v___x_511_);
v___x_559_ = l_Lean_Expr_app___override(v___x_558_, v_M_454_);
v___x_560_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__14));
v___x_561_ = l_Lean_Expr_const___override(v___x_560_, v___x_511_);
v___x_562_ = l_Lean_Expr_app___override(v___x_561_, v_M_454_);
v___x_563_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__17));
v___x_564_ = l_Lean_Expr_const___override(v___x_563_, v___x_511_);
v___x_565_ = l_Lean_Expr_app___override(v___x_564_, v_M_454_);
v___x_566_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__20));
v___x_567_ = l_Lean_Expr_const___override(v___x_566_, v___x_511_);
v___x_568_ = l_Lean_Expr_app___override(v___x_567_, v_M_454_);
v___x_569_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__23));
v___x_570_ = l_Lean_Expr_const___override(v___x_569_, v___x_511_);
v___x_571_ = l_Lean_Expr_app___override(v___x_570_, v_M_454_);
v___x_572_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__26));
v___x_573_ = l_Lean_Expr_const___override(v___x_572_, v___x_511_);
v___x_574_ = l_Lean_Expr_app___override(v___x_573_, v_M_454_);
lean_inc_ref_n(v_iM_503_, 2);
v___x_575_ = l_Lean_Expr_app___override(v___x_574_, v_iM_503_);
v___x_576_ = l_Lean_Expr_app___override(v___x_571_, v___x_575_);
v___x_577_ = l_Lean_Expr_app___override(v___x_568_, v___x_576_);
v___x_578_ = l_Lean_Expr_app___override(v___x_565_, v___x_577_);
v___x_579_ = l_Lean_Expr_app___override(v___x_562_, v___x_578_);
v___x_580_ = l_Lean_Expr_app___override(v___x_559_, v___x_579_);
v___x_581_ = l_Lean_Expr_app___override(v___x_556_, v___x_580_);
v___x_582_ = l_Lean_Expr_app___override(v___x_553_, v___x_581_);
v___x_583_ = l_Lean_Expr_app___override(v___x_550_, v___x_582_);
lean_inc_ref(v___x_583_);
v___x_584_ = l_Lean_Expr_app___override(v___x_583_, v_y_457_);
v___x_585_ = l_Lean_Expr_app___override(v___x_543_, v___x_584_);
v___x_586_ = l_Lean_Expr_app___override(v___x_547_, v___x_585_);
v___x_587_ = l_Lean_Expr_app___override(v___x_583_, v___x_544_);
v___x_588_ = l_Lean_Expr_app___override(v___x_586_, v___x_587_);
v___x_589_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__26));
v___x_590_ = l_Lean_Expr_const___override(v___x_589_, v___x_511_);
v___x_591_ = l_Lean_Expr_app___override(v___x_590_, v_M_454_);
v___x_592_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__28));
v___x_593_ = l_Lean_Expr_const___override(v___x_592_, v___x_511_);
v___x_594_ = l_Lean_Expr_app___override(v___x_593_, v_M_454_);
v___x_595_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__30));
v___x_596_ = l_Lean_Expr_const___override(v___x_595_, v___x_511_);
v___x_597_ = l_Lean_Expr_app___override(v___x_596_, v_M_454_);
v___x_598_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__33));
v___x_599_ = l_Lean_Expr_const___override(v___x_598_, v___x_511_);
v___x_600_ = l_Lean_Expr_app___override(v___x_599_, v_M_454_);
v___x_601_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__36));
v___x_602_ = l_Lean_Expr_const___override(v___x_601_, v___x_511_);
v___x_603_ = l_Lean_Expr_app___override(v___x_602_, v_M_454_);
v___x_604_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__38));
v___x_605_ = l_Lean_Expr_const___override(v___x_604_, v___x_511_);
v___x_606_ = l_Lean_Expr_app___override(v___x_605_, v_M_454_);
v___x_607_ = l_Lean_Expr_app___override(v___x_606_, v_iM_503_);
v___x_608_ = l_Lean_Expr_app___override(v___x_603_, v___x_607_);
v___x_609_ = l_Lean_Expr_app___override(v___x_600_, v___x_608_);
v___x_610_ = l_Lean_Expr_app___override(v___x_597_, v___x_609_);
v___x_611_ = l_Lean_Expr_app___override(v___x_594_, v___x_610_);
v___x_612_ = l_Lean_Expr_app___override(v___x_591_, v___x_611_);
v___x_613_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__41));
v___x_614_ = l_Lean_Expr_const___override(v___x_613_, v___x_511_);
v___x_615_ = l_Lean_Expr_app___override(v___x_614_, v_M_454_);
v___x_616_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__44));
v___x_617_ = l_Lean_Expr_const___override(v___x_616_, v___x_511_);
v___x_618_ = l_Lean_Expr_app___override(v___x_617_, v_M_454_);
v___x_619_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__47));
v___x_620_ = l_Lean_Expr_const___override(v___x_619_, v___x_511_);
v___x_621_ = l_Lean_Expr_app___override(v___x_620_, v_M_454_);
v___x_622_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__50));
v___x_623_ = l_Lean_Expr_const___override(v___x_622_, v___x_511_);
v___x_624_ = l_Lean_Expr_app___override(v___x_623_, v_M_454_);
v___x_625_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__52));
v___x_626_ = l_Lean_Expr_const___override(v___x_625_, v___x_511_);
v___x_627_ = l_Lean_Expr_app___override(v___x_626_, v_M_454_);
v___x_628_ = l_Lean_Expr_app___override(v___x_627_, v_iM_503_);
v___x_629_ = l_Lean_Expr_app___override(v___x_624_, v___x_628_);
v___x_630_ = l_Lean_Expr_app___override(v___x_621_, v___x_629_);
v___x_631_ = l_Lean_Expr_app___override(v___x_618_, v___x_630_);
v___x_632_ = l_Lean_Expr_app___override(v___x_615_, v___x_631_);
v___x_633_ = l_Lean_Expr_app___override(v___x_612_, v___x_632_);
v___x_634_ = l_Lean_Expr_app___override(v___x_633_, v_c_456_);
v___x_635_ = l_Lean_Expr_app___override(v___x_634_, v_y_457_);
v___x_636_ = l_Lean_Expr_app___override(v___x_588_, v___x_635_);
if (v_isShared_506_ == 0)
{
lean_ctor_set_tag(v___x_505_, 0);
lean_ctor_set(v___x_505_, 0, v___x_636_);
v___x_638_ = v___x_505_;
goto v_reusejp_637_;
}
else
{
lean_object* v_reuseFailAlloc_639_; 
v_reuseFailAlloc_639_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_639_, 0, v___x_636_);
v___x_638_ = v_reuseFailAlloc_639_;
goto v_reusejp_637_;
}
v_reusejp_637_:
{
return v___x_638_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___boxed(lean_object* v_v_641_, lean_object* v_M_642_, lean_object* v_iM_643_, lean_object* v_c_644_, lean_object* v_y_645_, lean_object* v_g_646_, lean_object* v_a_647_){
_start:
{
lean_object* v_res_648_; 
v_res_648_ = lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg(v_v_641_, v_M_642_, v_iM_643_, v_c_644_, v_y_645_, v_g_646_);
return v_res_648_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight(lean_object* v_v_649_, lean_object* v_M_650_, lean_object* v_iM_651_, lean_object* v_c_652_, lean_object* v_y_653_, lean_object* v_g_654_, lean_object* v_a_655_, lean_object* v_a_656_, lean_object* v_a_657_, lean_object* v_a_658_){
_start:
{
lean_object* v___x_660_; 
v___x_660_ = lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg(v_v_649_, v_M_650_, v_iM_651_, v_c_652_, v_y_653_, v_g_654_);
return v___x_660_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___boxed(lean_object* v_v_661_, lean_object* v_M_662_, lean_object* v_iM_663_, lean_object* v_c_664_, lean_object* v_y_665_, lean_object* v_g_666_, lean_object* v_a_667_, lean_object* v_a_668_, lean_object* v_a_669_, lean_object* v_a_670_, lean_object* v_a_671_){
_start:
{
lean_object* v_res_672_; 
v_res_672_ = lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight(v_v_661_, v_M_662_, v_iM_663_, v_c_664_, v_y_665_, v_g_666_, v_a_667_, v_a_668_, v_a_669_, v_a_670_);
lean_dec(v_a_670_);
lean_dec_ref(v_a_669_);
lean_dec(v_a_668_);
lean_dec_ref(v_a_667_);
return v_res_672_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mul___redArg(lean_object* v_v_679_, lean_object* v_M_680_, lean_object* v_iM_681_, lean_object* v_y_u2081_682_, lean_object* v_y_u2082_683_, lean_object* v_g_u2081_684_, lean_object* v_g_u2082_685_){
_start:
{
if (lean_obj_tag(v_g_u2081_684_) == 0)
{
if (lean_obj_tag(v_g_u2082_685_) == 0)
{
lean_object* v___x_687_; lean_object* v___x_688_; lean_object* v___x_689_; lean_object* v___x_690_; lean_object* v___x_691_; lean_object* v___x_692_; lean_object* v___x_693_; lean_object* v___x_694_; lean_object* v___x_695_; lean_object* v___x_696_; lean_object* v___x_697_; lean_object* v___x_698_; lean_object* v___x_699_; lean_object* v___x_700_; lean_object* v___x_701_; lean_object* v___x_702_; lean_object* v___x_703_; lean_object* v___x_704_; lean_object* v___x_705_; lean_object* v___x_706_; lean_object* v___x_707_; lean_object* v___x_708_; lean_object* v___x_709_; lean_object* v___x_710_; lean_object* v___x_711_; lean_object* v___x_712_; lean_object* v___x_713_; lean_object* v___x_714_; lean_object* v___x_715_; lean_object* v___x_716_; lean_object* v___x_717_; lean_object* v___x_718_; lean_object* v___x_719_; lean_object* v___x_720_; lean_object* v___x_721_; lean_object* v___x_722_; lean_object* v___x_723_; lean_object* v___x_724_; lean_object* v___x_725_; lean_object* v___x_726_; lean_object* v___x_727_; lean_object* v___x_728_; lean_object* v___x_729_; lean_object* v___x_730_; lean_object* v___x_731_; 
v___x_687_ = lean_box(0);
lean_inc_n(v_v_679_, 3);
v___x_688_ = l_Lean_Level_succ___override(v_v_679_);
v___x_689_ = lean_box(0);
v___x_690_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_690_, 0, v___x_688_);
lean_ctor_set(v___x_690_, 1, v___x_689_);
v___x_691_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__2));
v___x_692_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_692_, 0, v_v_679_);
lean_ctor_set(v___x_692_, 1, v___x_689_);
lean_inc_ref_n(v___x_692_, 6);
v___x_693_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_693_, 0, v_v_679_);
lean_ctor_set(v___x_693_, 1, v___x_692_);
v___x_694_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_694_, 0, v_v_679_);
lean_ctor_set(v___x_694_, 1, v___x_693_);
v___x_695_ = l_Lean_Expr_const___override(v___x_691_, v___x_694_);
lean_inc_ref_n(v_M_680_, 9);
v___x_696_ = l_Lean_Expr_app___override(v___x_695_, v_M_680_);
v___x_697_ = l_Lean_Expr_app___override(v___x_696_, v_M_680_);
v___x_698_ = l_Lean_Expr_app___override(v___x_697_, v_M_680_);
v___x_699_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__4));
v___x_700_ = l_Lean_Expr_const___override(v___x_699_, v___x_692_);
v___x_701_ = l_Lean_Expr_app___override(v___x_700_, v_M_680_);
v___x_702_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__7));
v___x_703_ = l_Lean_Expr_const___override(v___x_702_, v___x_692_);
v___x_704_ = l_Lean_Expr_app___override(v___x_703_, v_M_680_);
v___x_705_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__10));
v___x_706_ = l_Lean_Expr_const___override(v___x_705_, v___x_692_);
v___x_707_ = l_Lean_Expr_app___override(v___x_706_, v_M_680_);
v___x_708_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__13));
v___x_709_ = l_Lean_Expr_const___override(v___x_708_, v___x_692_);
v___x_710_ = l_Lean_Expr_app___override(v___x_709_, v_M_680_);
v___x_711_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__16));
v___x_712_ = l_Lean_Expr_const___override(v___x_711_, v___x_692_);
v___x_713_ = l_Lean_Expr_app___override(v___x_712_, v_M_680_);
v___x_714_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__19));
v___x_715_ = l_Lean_Expr_const___override(v___x_714_, v___x_692_);
v___x_716_ = l_Lean_Expr_app___override(v___x_715_, v_M_680_);
v___x_717_ = l_Lean_Expr_app___override(v___x_716_, v_iM_681_);
v___x_718_ = l_Lean_Expr_app___override(v___x_713_, v___x_717_);
v___x_719_ = l_Lean_Expr_app___override(v___x_710_, v___x_718_);
v___x_720_ = l_Lean_Expr_app___override(v___x_707_, v___x_719_);
v___x_721_ = l_Lean_Expr_app___override(v___x_704_, v___x_720_);
v___x_722_ = l_Lean_Expr_app___override(v___x_701_, v___x_721_);
v___x_723_ = l_Lean_Expr_app___override(v___x_698_, v___x_722_);
v___x_724_ = l_Lean_Expr_app___override(v___x_723_, v_y_u2081_682_);
v___x_725_ = l_Lean_Expr_app___override(v___x_724_, v_y_u2082_683_);
v___x_726_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__21));
v___x_727_ = l_Lean_Expr_const___override(v___x_726_, v___x_690_);
v___x_728_ = l_Lean_Expr_app___override(v___x_727_, v_M_680_);
v___x_729_ = l_Lean_Expr_app___override(v___x_728_, v___x_725_);
v___x_730_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_730_, 0, v___x_687_);
lean_ctor_set(v___x_730_, 1, v___x_729_);
v___x_731_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_731_, 0, v___x_730_);
return v___x_731_;
}
else
{
lean_object* v_iM_732_; lean_object* v___x_734_; uint8_t v_isShared_735_; uint8_t v_isSharedCheck_790_; 
lean_dec_ref(v_iM_681_);
v_iM_732_ = lean_ctor_get(v_g_u2082_685_, 0);
v_isSharedCheck_790_ = !lean_is_exclusive(v_g_u2082_685_);
if (v_isSharedCheck_790_ == 0)
{
v___x_734_ = v_g_u2082_685_;
v_isShared_735_ = v_isSharedCheck_790_;
goto v_resetjp_733_;
}
else
{
lean_inc(v_iM_732_);
lean_dec(v_g_u2082_685_);
v___x_734_ = lean_box(0);
v_isShared_735_ = v_isSharedCheck_790_;
goto v_resetjp_733_;
}
v_resetjp_733_:
{
lean_object* v___x_737_; 
lean_inc_ref(v_iM_732_);
if (v_isShared_735_ == 0)
{
v___x_737_ = v___x_734_;
goto v_reusejp_736_;
}
else
{
lean_object* v_reuseFailAlloc_789_; 
v_reuseFailAlloc_789_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_789_, 0, v_iM_732_);
v___x_737_ = v_reuseFailAlloc_789_;
goto v_reusejp_736_;
}
v_reusejp_736_:
{
lean_object* v___x_738_; lean_object* v___x_739_; lean_object* v___x_740_; lean_object* v___x_741_; lean_object* v___x_742_; lean_object* v___x_743_; lean_object* v___x_744_; lean_object* v___x_745_; lean_object* v___x_746_; lean_object* v___x_747_; lean_object* v___x_748_; lean_object* v___x_749_; lean_object* v___x_750_; lean_object* v___x_751_; lean_object* v___x_752_; lean_object* v___x_753_; lean_object* v___x_754_; lean_object* v___x_755_; lean_object* v___x_756_; lean_object* v___x_757_; lean_object* v___x_758_; lean_object* v___x_759_; lean_object* v___x_760_; lean_object* v___x_761_; lean_object* v___x_762_; lean_object* v___x_763_; lean_object* v___x_764_; lean_object* v___x_765_; lean_object* v___x_766_; lean_object* v___x_767_; lean_object* v___x_768_; lean_object* v___x_769_; lean_object* v___x_770_; lean_object* v___x_771_; lean_object* v___x_772_; lean_object* v___x_773_; lean_object* v___x_774_; lean_object* v___x_775_; lean_object* v___x_776_; lean_object* v___x_777_; lean_object* v___x_778_; lean_object* v___x_779_; lean_object* v___x_780_; lean_object* v___x_781_; lean_object* v___x_782_; lean_object* v___x_783_; lean_object* v___x_784_; lean_object* v___x_785_; lean_object* v___x_786_; lean_object* v___x_787_; lean_object* v___x_788_; 
v___x_738_ = lean_box(0);
v___x_739_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_739_, 0, v_v_679_);
lean_ctor_set(v___x_739_, 1, v___x_738_);
v___x_740_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__26));
lean_inc_ref_n(v___x_739_, 10);
v___x_741_ = l_Lean_Expr_const___override(v___x_740_, v___x_739_);
lean_inc_ref_n(v_M_680_, 10);
v___x_742_ = l_Lean_Expr_app___override(v___x_741_, v_M_680_);
v___x_743_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__28));
v___x_744_ = l_Lean_Expr_const___override(v___x_743_, v___x_739_);
v___x_745_ = l_Lean_Expr_app___override(v___x_744_, v_M_680_);
v___x_746_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__30));
v___x_747_ = l_Lean_Expr_const___override(v___x_746_, v___x_739_);
v___x_748_ = l_Lean_Expr_app___override(v___x_747_, v_M_680_);
v___x_749_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__33));
v___x_750_ = l_Lean_Expr_const___override(v___x_749_, v___x_739_);
v___x_751_ = l_Lean_Expr_app___override(v___x_750_, v_M_680_);
v___x_752_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__36));
v___x_753_ = l_Lean_Expr_const___override(v___x_752_, v___x_739_);
v___x_754_ = l_Lean_Expr_app___override(v___x_753_, v_M_680_);
v___x_755_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__38));
v___x_756_ = l_Lean_Expr_const___override(v___x_755_, v___x_739_);
v___x_757_ = l_Lean_Expr_app___override(v___x_756_, v_M_680_);
lean_inc_ref(v_iM_732_);
v___x_758_ = l_Lean_Expr_app___override(v___x_757_, v_iM_732_);
v___x_759_ = l_Lean_Expr_app___override(v___x_754_, v___x_758_);
v___x_760_ = l_Lean_Expr_app___override(v___x_751_, v___x_759_);
v___x_761_ = l_Lean_Expr_app___override(v___x_748_, v___x_760_);
v___x_762_ = l_Lean_Expr_app___override(v___x_745_, v___x_761_);
v___x_763_ = l_Lean_Expr_app___override(v___x_742_, v___x_762_);
v___x_764_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__41));
v___x_765_ = l_Lean_Expr_const___override(v___x_764_, v___x_739_);
v___x_766_ = l_Lean_Expr_app___override(v___x_765_, v_M_680_);
v___x_767_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__44));
v___x_768_ = l_Lean_Expr_const___override(v___x_767_, v___x_739_);
v___x_769_ = l_Lean_Expr_app___override(v___x_768_, v_M_680_);
v___x_770_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__47));
v___x_771_ = l_Lean_Expr_const___override(v___x_770_, v___x_739_);
v___x_772_ = l_Lean_Expr_app___override(v___x_771_, v_M_680_);
v___x_773_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__50));
v___x_774_ = l_Lean_Expr_const___override(v___x_773_, v___x_739_);
v___x_775_ = l_Lean_Expr_app___override(v___x_774_, v_M_680_);
v___x_776_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__52));
v___x_777_ = l_Lean_Expr_const___override(v___x_776_, v___x_739_);
v___x_778_ = l_Lean_Expr_app___override(v___x_777_, v_M_680_);
v___x_779_ = l_Lean_Expr_app___override(v___x_778_, v_iM_732_);
v___x_780_ = l_Lean_Expr_app___override(v___x_775_, v___x_779_);
v___x_781_ = l_Lean_Expr_app___override(v___x_772_, v___x_780_);
v___x_782_ = l_Lean_Expr_app___override(v___x_769_, v___x_781_);
v___x_783_ = l_Lean_Expr_app___override(v___x_766_, v___x_782_);
v___x_784_ = l_Lean_Expr_app___override(v___x_763_, v___x_783_);
v___x_785_ = l_Lean_Expr_app___override(v___x_784_, v_y_u2081_682_);
v___x_786_ = l_Lean_Expr_app___override(v___x_785_, v_y_u2082_683_);
v___x_787_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_787_, 0, v___x_737_);
lean_ctor_set(v___x_787_, 1, v___x_786_);
v___x_788_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_788_, 0, v___x_787_);
return v___x_788_;
}
}
}
}
else
{
lean_dec_ref(v_iM_681_);
if (lean_obj_tag(v_g_u2082_685_) == 0)
{
lean_object* v_iM_791_; lean_object* v___x_793_; uint8_t v_isShared_794_; uint8_t v_isSharedCheck_849_; 
v_iM_791_ = lean_ctor_get(v_g_u2081_684_, 0);
v_isSharedCheck_849_ = !lean_is_exclusive(v_g_u2081_684_);
if (v_isSharedCheck_849_ == 0)
{
v___x_793_ = v_g_u2081_684_;
v_isShared_794_ = v_isSharedCheck_849_;
goto v_resetjp_792_;
}
else
{
lean_inc(v_iM_791_);
lean_dec(v_g_u2081_684_);
v___x_793_ = lean_box(0);
v_isShared_794_ = v_isSharedCheck_849_;
goto v_resetjp_792_;
}
v_resetjp_792_:
{
lean_object* v___x_796_; 
lean_inc_ref(v_iM_791_);
if (v_isShared_794_ == 0)
{
v___x_796_ = v___x_793_;
goto v_reusejp_795_;
}
else
{
lean_object* v_reuseFailAlloc_848_; 
v_reuseFailAlloc_848_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_848_, 0, v_iM_791_);
v___x_796_ = v_reuseFailAlloc_848_;
goto v_reusejp_795_;
}
v_reusejp_795_:
{
lean_object* v___x_797_; lean_object* v___x_798_; lean_object* v___x_799_; lean_object* v___x_800_; lean_object* v___x_801_; lean_object* v___x_802_; lean_object* v___x_803_; lean_object* v___x_804_; lean_object* v___x_805_; lean_object* v___x_806_; lean_object* v___x_807_; lean_object* v___x_808_; lean_object* v___x_809_; lean_object* v___x_810_; lean_object* v___x_811_; lean_object* v___x_812_; lean_object* v___x_813_; lean_object* v___x_814_; lean_object* v___x_815_; lean_object* v___x_816_; lean_object* v___x_817_; lean_object* v___x_818_; lean_object* v___x_819_; lean_object* v___x_820_; lean_object* v___x_821_; lean_object* v___x_822_; lean_object* v___x_823_; lean_object* v___x_824_; lean_object* v___x_825_; lean_object* v___x_826_; lean_object* v___x_827_; lean_object* v___x_828_; lean_object* v___x_829_; lean_object* v___x_830_; lean_object* v___x_831_; lean_object* v___x_832_; lean_object* v___x_833_; lean_object* v___x_834_; lean_object* v___x_835_; lean_object* v___x_836_; lean_object* v___x_837_; lean_object* v___x_838_; lean_object* v___x_839_; lean_object* v___x_840_; lean_object* v___x_841_; lean_object* v___x_842_; lean_object* v___x_843_; lean_object* v___x_844_; lean_object* v___x_845_; lean_object* v___x_846_; lean_object* v___x_847_; 
v___x_797_ = lean_box(0);
v___x_798_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_798_, 0, v_v_679_);
lean_ctor_set(v___x_798_, 1, v___x_797_);
v___x_799_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mul___redArg___closed__1));
lean_inc_ref_n(v___x_798_, 10);
v___x_800_ = l_Lean_Expr_const___override(v___x_799_, v___x_798_);
lean_inc_ref_n(v_M_680_, 10);
v___x_801_ = l_Lean_Expr_app___override(v___x_800_, v_M_680_);
v___x_802_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__28));
v___x_803_ = l_Lean_Expr_const___override(v___x_802_, v___x_798_);
v___x_804_ = l_Lean_Expr_app___override(v___x_803_, v_M_680_);
v___x_805_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__30));
v___x_806_ = l_Lean_Expr_const___override(v___x_805_, v___x_798_);
v___x_807_ = l_Lean_Expr_app___override(v___x_806_, v_M_680_);
v___x_808_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__33));
v___x_809_ = l_Lean_Expr_const___override(v___x_808_, v___x_798_);
v___x_810_ = l_Lean_Expr_app___override(v___x_809_, v_M_680_);
v___x_811_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__36));
v___x_812_ = l_Lean_Expr_const___override(v___x_811_, v___x_798_);
v___x_813_ = l_Lean_Expr_app___override(v___x_812_, v_M_680_);
v___x_814_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__38));
v___x_815_ = l_Lean_Expr_const___override(v___x_814_, v___x_798_);
v___x_816_ = l_Lean_Expr_app___override(v___x_815_, v_M_680_);
lean_inc_ref(v_iM_791_);
v___x_817_ = l_Lean_Expr_app___override(v___x_816_, v_iM_791_);
v___x_818_ = l_Lean_Expr_app___override(v___x_813_, v___x_817_);
v___x_819_ = l_Lean_Expr_app___override(v___x_810_, v___x_818_);
v___x_820_ = l_Lean_Expr_app___override(v___x_807_, v___x_819_);
v___x_821_ = l_Lean_Expr_app___override(v___x_804_, v___x_820_);
v___x_822_ = l_Lean_Expr_app___override(v___x_801_, v___x_821_);
v___x_823_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__41));
v___x_824_ = l_Lean_Expr_const___override(v___x_823_, v___x_798_);
v___x_825_ = l_Lean_Expr_app___override(v___x_824_, v_M_680_);
v___x_826_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__44));
v___x_827_ = l_Lean_Expr_const___override(v___x_826_, v___x_798_);
v___x_828_ = l_Lean_Expr_app___override(v___x_827_, v_M_680_);
v___x_829_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__47));
v___x_830_ = l_Lean_Expr_const___override(v___x_829_, v___x_798_);
v___x_831_ = l_Lean_Expr_app___override(v___x_830_, v_M_680_);
v___x_832_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__50));
v___x_833_ = l_Lean_Expr_const___override(v___x_832_, v___x_798_);
v___x_834_ = l_Lean_Expr_app___override(v___x_833_, v_M_680_);
v___x_835_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__52));
v___x_836_ = l_Lean_Expr_const___override(v___x_835_, v___x_798_);
v___x_837_ = l_Lean_Expr_app___override(v___x_836_, v_M_680_);
v___x_838_ = l_Lean_Expr_app___override(v___x_837_, v_iM_791_);
v___x_839_ = l_Lean_Expr_app___override(v___x_834_, v___x_838_);
v___x_840_ = l_Lean_Expr_app___override(v___x_831_, v___x_839_);
v___x_841_ = l_Lean_Expr_app___override(v___x_828_, v___x_840_);
v___x_842_ = l_Lean_Expr_app___override(v___x_825_, v___x_841_);
v___x_843_ = l_Lean_Expr_app___override(v___x_822_, v___x_842_);
v___x_844_ = l_Lean_Expr_app___override(v___x_843_, v_y_u2081_682_);
v___x_845_ = l_Lean_Expr_app___override(v___x_844_, v_y_u2082_683_);
v___x_846_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_846_, 0, v___x_796_);
lean_ctor_set(v___x_846_, 1, v___x_845_);
v___x_847_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_847_, 0, v___x_846_);
return v___x_847_;
}
}
}
else
{
lean_object* v_iM_850_; lean_object* v___x_851_; lean_object* v___x_852_; lean_object* v___x_853_; lean_object* v___x_854_; lean_object* v___x_855_; lean_object* v___x_856_; lean_object* v___x_857_; lean_object* v___x_858_; lean_object* v___x_859_; lean_object* v___x_860_; lean_object* v___x_861_; lean_object* v___x_862_; lean_object* v___x_863_; lean_object* v___x_864_; lean_object* v___x_865_; lean_object* v___x_866_; lean_object* v___x_867_; lean_object* v___x_868_; lean_object* v___x_869_; lean_object* v___x_870_; lean_object* v___x_871_; lean_object* v___x_872_; lean_object* v___x_873_; lean_object* v___x_874_; lean_object* v___x_875_; lean_object* v___x_876_; lean_object* v___x_877_; lean_object* v___x_878_; lean_object* v___x_879_; lean_object* v___x_880_; lean_object* v___x_881_; lean_object* v___x_882_; lean_object* v___x_883_; lean_object* v___x_884_; lean_object* v___x_885_; lean_object* v___x_886_; lean_object* v___x_887_; lean_object* v___x_888_; lean_object* v___x_889_; lean_object* v___x_890_; lean_object* v___x_891_; lean_object* v___x_892_; lean_object* v___x_893_; lean_object* v___x_894_; lean_object* v___x_895_; lean_object* v___x_896_; lean_object* v___x_897_; lean_object* v___x_898_; lean_object* v___x_899_; lean_object* v___x_900_; lean_object* v___x_901_; lean_object* v___x_902_; 
lean_dec_ref_known(v_g_u2082_685_, 1);
v_iM_850_ = lean_ctor_get(v_g_u2081_684_, 0);
lean_inc_ref_n(v_iM_850_, 2);
lean_dec_ref_known(v_g_u2081_684_, 1);
v___x_851_ = lean_box(0);
v___x_852_ = lean_box(0);
v___x_853_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_853_, 0, v_v_679_);
lean_ctor_set(v___x_853_, 1, v___x_852_);
v___x_854_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mul___redArg___closed__3));
lean_inc_ref_n(v___x_853_, 10);
v___x_855_ = l_Lean_Expr_const___override(v___x_854_, v___x_853_);
lean_inc_ref_n(v_M_680_, 10);
v___x_856_ = l_Lean_Expr_app___override(v___x_855_, v_M_680_);
v___x_857_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__28));
v___x_858_ = l_Lean_Expr_const___override(v___x_857_, v___x_853_);
v___x_859_ = l_Lean_Expr_app___override(v___x_858_, v_M_680_);
v___x_860_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__30));
v___x_861_ = l_Lean_Expr_const___override(v___x_860_, v___x_853_);
v___x_862_ = l_Lean_Expr_app___override(v___x_861_, v_M_680_);
v___x_863_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__33));
v___x_864_ = l_Lean_Expr_const___override(v___x_863_, v___x_853_);
v___x_865_ = l_Lean_Expr_app___override(v___x_864_, v_M_680_);
v___x_866_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__36));
v___x_867_ = l_Lean_Expr_const___override(v___x_866_, v___x_853_);
v___x_868_ = l_Lean_Expr_app___override(v___x_867_, v_M_680_);
v___x_869_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__38));
v___x_870_ = l_Lean_Expr_const___override(v___x_869_, v___x_853_);
v___x_871_ = l_Lean_Expr_app___override(v___x_870_, v_M_680_);
v___x_872_ = l_Lean_Expr_app___override(v___x_871_, v_iM_850_);
v___x_873_ = l_Lean_Expr_app___override(v___x_868_, v___x_872_);
v___x_874_ = l_Lean_Expr_app___override(v___x_865_, v___x_873_);
v___x_875_ = l_Lean_Expr_app___override(v___x_862_, v___x_874_);
v___x_876_ = l_Lean_Expr_app___override(v___x_859_, v___x_875_);
v___x_877_ = l_Lean_Expr_app___override(v___x_856_, v___x_876_);
v___x_878_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__41));
v___x_879_ = l_Lean_Expr_const___override(v___x_878_, v___x_853_);
v___x_880_ = l_Lean_Expr_app___override(v___x_879_, v_M_680_);
v___x_881_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__44));
v___x_882_ = l_Lean_Expr_const___override(v___x_881_, v___x_853_);
v___x_883_ = l_Lean_Expr_app___override(v___x_882_, v_M_680_);
v___x_884_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__47));
v___x_885_ = l_Lean_Expr_const___override(v___x_884_, v___x_853_);
v___x_886_ = l_Lean_Expr_app___override(v___x_885_, v_M_680_);
v___x_887_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__50));
v___x_888_ = l_Lean_Expr_const___override(v___x_887_, v___x_853_);
v___x_889_ = l_Lean_Expr_app___override(v___x_888_, v_M_680_);
v___x_890_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__52));
v___x_891_ = l_Lean_Expr_const___override(v___x_890_, v___x_853_);
v___x_892_ = l_Lean_Expr_app___override(v___x_891_, v_M_680_);
v___x_893_ = l_Lean_Expr_app___override(v___x_892_, v_iM_850_);
v___x_894_ = l_Lean_Expr_app___override(v___x_889_, v___x_893_);
v___x_895_ = l_Lean_Expr_app___override(v___x_886_, v___x_894_);
v___x_896_ = l_Lean_Expr_app___override(v___x_883_, v___x_895_);
v___x_897_ = l_Lean_Expr_app___override(v___x_880_, v___x_896_);
v___x_898_ = l_Lean_Expr_app___override(v___x_877_, v___x_897_);
v___x_899_ = l_Lean_Expr_app___override(v___x_898_, v_y_u2081_682_);
v___x_900_ = l_Lean_Expr_app___override(v___x_899_, v_y_u2082_683_);
v___x_901_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_901_, 0, v___x_851_);
lean_ctor_set(v___x_901_, 1, v___x_900_);
v___x_902_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_902_, 0, v___x_901_);
return v___x_902_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mul___redArg___boxed(lean_object* v_v_903_, lean_object* v_M_904_, lean_object* v_iM_905_, lean_object* v_y_u2081_906_, lean_object* v_y_u2082_907_, lean_object* v_g_u2081_908_, lean_object* v_g_u2082_909_, lean_object* v_a_910_){
_start:
{
lean_object* v_res_911_; 
v_res_911_ = lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mul___redArg(v_v_903_, v_M_904_, v_iM_905_, v_y_u2081_906_, v_y_u2082_907_, v_g_u2081_908_, v_g_u2082_909_);
return v_res_911_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mul(lean_object* v_v_912_, lean_object* v_M_913_, lean_object* v_iM_914_, lean_object* v_y_u2081_915_, lean_object* v_y_u2082_916_, lean_object* v_g_u2081_917_, lean_object* v_g_u2082_918_, lean_object* v_a_919_, lean_object* v_a_920_, lean_object* v_a_921_, lean_object* v_a_922_){
_start:
{
lean_object* v___x_924_; 
v___x_924_ = lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mul___redArg(v_v_912_, v_M_913_, v_iM_914_, v_y_u2081_915_, v_y_u2082_916_, v_g_u2081_917_, v_g_u2082_918_);
return v___x_924_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mul___boxed(lean_object* v_v_925_, lean_object* v_M_926_, lean_object* v_iM_927_, lean_object* v_y_u2081_928_, lean_object* v_y_u2082_929_, lean_object* v_g_u2081_930_, lean_object* v_g_u2082_931_, lean_object* v_a_932_, lean_object* v_a_933_, lean_object* v_a_934_, lean_object* v_a_935_, lean_object* v_a_936_){
_start:
{
lean_object* v_res_937_; 
v_res_937_ = lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mul(v_v_925_, v_M_926_, v_iM_927_, v_y_u2081_928_, v_y_u2082_929_, v_g_u2081_930_, v_g_u2082_931_, v_a_932_, v_a_933_, v_a_934_, v_a_935_);
lean_dec(v_a_935_);
lean_dec_ref(v_a_934_);
lean_dec(v_a_933_);
lean_dec_ref(v_a_932_);
return v_res_937_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg(lean_object* v_v_970_, lean_object* v_M_971_, lean_object* v_iM_972_, lean_object* v_y_973_, lean_object* v_g_974_){
_start:
{
if (lean_obj_tag(v_g_974_) == 0)
{
lean_object* v___x_976_; lean_object* v___x_977_; lean_object* v___x_978_; lean_object* v___x_979_; lean_object* v___x_980_; lean_object* v___x_981_; lean_object* v___x_982_; lean_object* v___x_983_; lean_object* v___x_984_; lean_object* v___x_985_; lean_object* v___x_986_; lean_object* v___x_987_; lean_object* v___x_988_; lean_object* v___x_989_; lean_object* v___x_990_; lean_object* v___x_991_; lean_object* v___x_992_; lean_object* v___x_993_; lean_object* v___x_994_; lean_object* v___x_995_; lean_object* v___x_996_; lean_object* v___x_997_; lean_object* v___x_998_; lean_object* v___x_999_; lean_object* v___x_1000_; lean_object* v___x_1001_; lean_object* v___x_1002_; lean_object* v___x_1003_; lean_object* v___x_1004_; lean_object* v___x_1005_; lean_object* v___x_1006_; lean_object* v___x_1007_; lean_object* v___x_1008_; lean_object* v___x_1009_; 
lean_inc(v_v_970_);
v___x_976_ = l_Lean_Level_succ___override(v_v_970_);
v___x_977_ = lean_box(0);
v___x_978_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_978_, 0, v___x_976_);
lean_ctor_set(v___x_978_, 1, v___x_977_);
v___x_979_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__2));
v___x_980_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_980_, 0, v_v_970_);
lean_ctor_set(v___x_980_, 1, v___x_977_);
lean_inc_ref_n(v___x_980_, 5);
v___x_981_ = l_Lean_Expr_const___override(v___x_979_, v___x_980_);
lean_inc_ref_n(v_M_971_, 6);
v___x_982_ = l_Lean_Expr_app___override(v___x_981_, v_M_971_);
v___x_983_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__5));
v___x_984_ = l_Lean_Expr_const___override(v___x_983_, v___x_980_);
v___x_985_ = l_Lean_Expr_app___override(v___x_984_, v_M_971_);
v___x_986_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__8));
v___x_987_ = l_Lean_Expr_const___override(v___x_986_, v___x_980_);
v___x_988_ = l_Lean_Expr_app___override(v___x_987_, v_M_971_);
v___x_989_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__11));
v___x_990_ = l_Lean_Expr_const___override(v___x_989_, v___x_980_);
v___x_991_ = l_Lean_Expr_app___override(v___x_990_, v_M_971_);
v___x_992_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__14));
v___x_993_ = l_Lean_Expr_const___override(v___x_992_, v___x_980_);
v___x_994_ = l_Lean_Expr_app___override(v___x_993_, v_M_971_);
v___x_995_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__16));
v___x_996_ = l_Lean_Expr_const___override(v___x_995_, v___x_980_);
v___x_997_ = l_Lean_Expr_app___override(v___x_996_, v_M_971_);
v___x_998_ = l_Lean_Expr_app___override(v___x_997_, v_iM_972_);
v___x_999_ = l_Lean_Expr_app___override(v___x_994_, v___x_998_);
v___x_1000_ = l_Lean_Expr_app___override(v___x_991_, v___x_999_);
v___x_1001_ = l_Lean_Expr_app___override(v___x_988_, v___x_1000_);
v___x_1002_ = l_Lean_Expr_app___override(v___x_985_, v___x_1001_);
v___x_1003_ = l_Lean_Expr_app___override(v___x_982_, v___x_1002_);
v___x_1004_ = l_Lean_Expr_app___override(v___x_1003_, v_y_973_);
v___x_1005_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__21));
v___x_1006_ = l_Lean_Expr_const___override(v___x_1005_, v___x_978_);
v___x_1007_ = l_Lean_Expr_app___override(v___x_1006_, v_M_971_);
v___x_1008_ = l_Lean_Expr_app___override(v___x_1007_, v___x_1004_);
v___x_1009_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1009_, 0, v___x_1008_);
return v___x_1009_;
}
else
{
lean_object* v_iM_1010_; lean_object* v___x_1012_; uint8_t v_isShared_1013_; uint8_t v_isSharedCheck_1053_; 
v_iM_1010_ = lean_ctor_get(v_g_974_, 0);
v_isSharedCheck_1053_ = !lean_is_exclusive(v_g_974_);
if (v_isSharedCheck_1053_ == 0)
{
v___x_1012_ = v_g_974_;
v_isShared_1013_ = v_isSharedCheck_1053_;
goto v_resetjp_1011_;
}
else
{
lean_inc(v_iM_1010_);
lean_dec(v_g_974_);
v___x_1012_ = lean_box(0);
v_isShared_1013_ = v_isSharedCheck_1053_;
goto v_resetjp_1011_;
}
v_resetjp_1011_:
{
lean_object* v___x_1014_; lean_object* v___x_1015_; lean_object* v___x_1016_; lean_object* v___x_1017_; lean_object* v___x_1018_; lean_object* v___x_1019_; lean_object* v___x_1020_; lean_object* v___x_1021_; lean_object* v___x_1022_; lean_object* v___x_1023_; lean_object* v___x_1024_; lean_object* v___x_1025_; lean_object* v___x_1026_; lean_object* v___x_1027_; lean_object* v___x_1028_; lean_object* v___x_1029_; lean_object* v___x_1030_; lean_object* v___x_1031_; lean_object* v___x_1032_; lean_object* v___x_1033_; lean_object* v___x_1034_; lean_object* v___x_1035_; lean_object* v___x_1036_; lean_object* v___x_1037_; lean_object* v___x_1038_; lean_object* v___x_1039_; lean_object* v___x_1040_; lean_object* v___x_1041_; lean_object* v___x_1042_; lean_object* v___x_1043_; lean_object* v___x_1044_; lean_object* v___x_1045_; lean_object* v___x_1046_; lean_object* v___x_1047_; lean_object* v___x_1048_; lean_object* v___x_1049_; lean_object* v___x_1051_; 
v___x_1014_ = lean_box(0);
v___x_1015_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1015_, 0, v_v_970_);
lean_ctor_set(v___x_1015_, 1, v___x_1014_);
v___x_1016_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__14));
lean_inc_ref_n(v___x_1015_, 7);
v___x_1017_ = l_Lean_Expr_const___override(v___x_1016_, v___x_1015_);
lean_inc_ref_n(v_M_971_, 7);
v___x_1018_ = l_Lean_Expr_app___override(v___x_1017_, v_M_971_);
v___x_1019_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__16));
v___x_1020_ = l_Lean_Expr_const___override(v___x_1019_, v___x_1015_);
v___x_1021_ = l_Lean_Expr_app___override(v___x_1020_, v_M_971_);
v___x_1022_ = l_Lean_Expr_app___override(v___x_1021_, v_iM_972_);
v___x_1023_ = l_Lean_Expr_app___override(v___x_1018_, v___x_1022_);
v___x_1024_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__18));
v___x_1025_ = l_Lean_Expr_const___override(v___x_1024_, v___x_1015_);
v___x_1026_ = l_Lean_Expr_app___override(v___x_1025_, v_M_971_);
v___x_1027_ = l_Lean_Expr_app___override(v___x_1026_, v___x_1023_);
v___x_1028_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__41));
v___x_1029_ = l_Lean_Expr_const___override(v___x_1028_, v___x_1015_);
v___x_1030_ = l_Lean_Expr_app___override(v___x_1029_, v_M_971_);
v___x_1031_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__44));
v___x_1032_ = l_Lean_Expr_const___override(v___x_1031_, v___x_1015_);
v___x_1033_ = l_Lean_Expr_app___override(v___x_1032_, v_M_971_);
v___x_1034_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__47));
v___x_1035_ = l_Lean_Expr_const___override(v___x_1034_, v___x_1015_);
v___x_1036_ = l_Lean_Expr_app___override(v___x_1035_, v_M_971_);
v___x_1037_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__50));
v___x_1038_ = l_Lean_Expr_const___override(v___x_1037_, v___x_1015_);
v___x_1039_ = l_Lean_Expr_app___override(v___x_1038_, v_M_971_);
v___x_1040_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__52));
v___x_1041_ = l_Lean_Expr_const___override(v___x_1040_, v___x_1015_);
v___x_1042_ = l_Lean_Expr_app___override(v___x_1041_, v_M_971_);
v___x_1043_ = l_Lean_Expr_app___override(v___x_1042_, v_iM_1010_);
v___x_1044_ = l_Lean_Expr_app___override(v___x_1039_, v___x_1043_);
v___x_1045_ = l_Lean_Expr_app___override(v___x_1036_, v___x_1044_);
v___x_1046_ = l_Lean_Expr_app___override(v___x_1033_, v___x_1045_);
v___x_1047_ = l_Lean_Expr_app___override(v___x_1030_, v___x_1046_);
v___x_1048_ = l_Lean_Expr_app___override(v___x_1027_, v___x_1047_);
v___x_1049_ = l_Lean_Expr_app___override(v___x_1048_, v_y_973_);
if (v_isShared_1013_ == 0)
{
lean_ctor_set_tag(v___x_1012_, 0);
lean_ctor_set(v___x_1012_, 0, v___x_1049_);
v___x_1051_ = v___x_1012_;
goto v_reusejp_1050_;
}
else
{
lean_object* v_reuseFailAlloc_1052_; 
v_reuseFailAlloc_1052_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1052_, 0, v___x_1049_);
v___x_1051_ = v_reuseFailAlloc_1052_;
goto v_reusejp_1050_;
}
v_reusejp_1050_:
{
return v___x_1051_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___boxed(lean_object* v_v_1054_, lean_object* v_M_1055_, lean_object* v_iM_1056_, lean_object* v_y_1057_, lean_object* v_g_1058_, lean_object* v_a_1059_){
_start:
{
lean_object* v_res_1060_; 
v_res_1060_ = lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg(v_v_1054_, v_M_1055_, v_iM_1056_, v_y_1057_, v_g_1058_);
return v_res_1060_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv(lean_object* v_v_1061_, lean_object* v_M_1062_, lean_object* v_iM_1063_, lean_object* v_y_1064_, lean_object* v_g_1065_, lean_object* v_a_1066_, lean_object* v_a_1067_, lean_object* v_a_1068_, lean_object* v_a_1069_){
_start:
{
lean_object* v___x_1071_; 
v___x_1071_ = lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg(v_v_1061_, v_M_1062_, v_iM_1063_, v_y_1064_, v_g_1065_);
return v___x_1071_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___boxed(lean_object* v_v_1072_, lean_object* v_M_1073_, lean_object* v_iM_1074_, lean_object* v_y_1075_, lean_object* v_g_1076_, lean_object* v_a_1077_, lean_object* v_a_1078_, lean_object* v_a_1079_, lean_object* v_a_1080_, lean_object* v_a_1081_){
_start:
{
lean_object* v_res_1082_; 
v_res_1082_ = lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv(v_v_1072_, v_M_1073_, v_iM_1074_, v_y_1075_, v_g_1076_, v_a_1077_, v_a_1078_, v_a_1079_, v_a_1080_);
lean_dec(v_a_1080_);
lean_dec_ref(v_a_1079_);
lean_dec(v_a_1078_);
lean_dec_ref(v_a_1077_);
return v_res_1082_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg(lean_object* v_v_1109_, lean_object* v_M_1110_, lean_object* v_iM_1111_, lean_object* v_y_u2081_1112_, lean_object* v_y_u2082_1113_, lean_object* v_g_u2081_1114_, lean_object* v_g_u2082_1115_){
_start:
{
if (lean_obj_tag(v_g_u2081_1114_) == 0)
{
if (lean_obj_tag(v_g_u2082_1115_) == 0)
{
lean_object* v___x_1117_; lean_object* v___x_1118_; lean_object* v___x_1119_; lean_object* v___x_1120_; lean_object* v___x_1121_; lean_object* v___x_1122_; lean_object* v___x_1123_; lean_object* v___x_1124_; lean_object* v___x_1125_; lean_object* v___x_1126_; lean_object* v___x_1127_; lean_object* v___x_1128_; lean_object* v___x_1129_; lean_object* v___x_1130_; lean_object* v___x_1131_; lean_object* v___x_1132_; lean_object* v___x_1133_; lean_object* v___x_1134_; lean_object* v___x_1135_; lean_object* v___x_1136_; lean_object* v___x_1137_; lean_object* v___x_1138_; lean_object* v___x_1139_; lean_object* v___x_1140_; lean_object* v___x_1141_; lean_object* v___x_1142_; lean_object* v___x_1143_; lean_object* v___x_1144_; lean_object* v___x_1145_; lean_object* v___x_1146_; lean_object* v___x_1147_; lean_object* v___x_1148_; lean_object* v___x_1149_; lean_object* v___x_1150_; lean_object* v___x_1151_; lean_object* v___x_1152_; lean_object* v___x_1153_; 
v___x_1117_ = lean_box(0);
lean_inc_n(v_v_1109_, 3);
v___x_1118_ = l_Lean_Level_succ___override(v_v_1109_);
v___x_1119_ = lean_box(0);
v___x_1120_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1120_, 0, v___x_1118_);
lean_ctor_set(v___x_1120_, 1, v___x_1119_);
v___x_1121_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__2));
v___x_1122_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1122_, 0, v_v_1109_);
lean_ctor_set(v___x_1122_, 1, v___x_1119_);
lean_inc_ref_n(v___x_1122_, 4);
v___x_1123_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1123_, 0, v_v_1109_);
lean_ctor_set(v___x_1123_, 1, v___x_1122_);
v___x_1124_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1124_, 0, v_v_1109_);
lean_ctor_set(v___x_1124_, 1, v___x_1123_);
v___x_1125_ = l_Lean_Expr_const___override(v___x_1121_, v___x_1124_);
lean_inc_ref_n(v_M_1110_, 7);
v___x_1126_ = l_Lean_Expr_app___override(v___x_1125_, v_M_1110_);
v___x_1127_ = l_Lean_Expr_app___override(v___x_1126_, v_M_1110_);
v___x_1128_ = l_Lean_Expr_app___override(v___x_1127_, v_M_1110_);
v___x_1129_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__4));
v___x_1130_ = l_Lean_Expr_const___override(v___x_1129_, v___x_1122_);
v___x_1131_ = l_Lean_Expr_app___override(v___x_1130_, v_M_1110_);
v___x_1132_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__7));
v___x_1133_ = l_Lean_Expr_const___override(v___x_1132_, v___x_1122_);
v___x_1134_ = l_Lean_Expr_app___override(v___x_1133_, v_M_1110_);
v___x_1135_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__9));
v___x_1136_ = l_Lean_Expr_const___override(v___x_1135_, v___x_1122_);
v___x_1137_ = l_Lean_Expr_app___override(v___x_1136_, v_M_1110_);
v___x_1138_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__19));
v___x_1139_ = l_Lean_Expr_const___override(v___x_1138_, v___x_1122_);
v___x_1140_ = l_Lean_Expr_app___override(v___x_1139_, v_M_1110_);
v___x_1141_ = l_Lean_Expr_app___override(v___x_1140_, v_iM_1111_);
v___x_1142_ = l_Lean_Expr_app___override(v___x_1137_, v___x_1141_);
v___x_1143_ = l_Lean_Expr_app___override(v___x_1134_, v___x_1142_);
v___x_1144_ = l_Lean_Expr_app___override(v___x_1131_, v___x_1143_);
v___x_1145_ = l_Lean_Expr_app___override(v___x_1128_, v___x_1144_);
v___x_1146_ = l_Lean_Expr_app___override(v___x_1145_, v_y_u2081_1112_);
v___x_1147_ = l_Lean_Expr_app___override(v___x_1146_, v_y_u2082_1113_);
v___x_1148_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__21));
v___x_1149_ = l_Lean_Expr_const___override(v___x_1148_, v___x_1120_);
v___x_1150_ = l_Lean_Expr_app___override(v___x_1149_, v_M_1110_);
v___x_1151_ = l_Lean_Expr_app___override(v___x_1150_, v___x_1147_);
v___x_1152_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1152_, 0, v___x_1117_);
lean_ctor_set(v___x_1152_, 1, v___x_1151_);
v___x_1153_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1153_, 0, v___x_1152_);
return v___x_1153_;
}
else
{
lean_object* v_iM_1154_; lean_object* v___x_1156_; uint8_t v_isShared_1157_; uint8_t v_isSharedCheck_1200_; 
v_iM_1154_ = lean_ctor_get(v_g_u2082_1115_, 0);
v_isSharedCheck_1200_ = !lean_is_exclusive(v_g_u2082_1115_);
if (v_isSharedCheck_1200_ == 0)
{
v___x_1156_ = v_g_u2082_1115_;
v_isShared_1157_ = v_isSharedCheck_1200_;
goto v_resetjp_1155_;
}
else
{
lean_inc(v_iM_1154_);
lean_dec(v_g_u2082_1115_);
v___x_1156_ = lean_box(0);
v_isShared_1157_ = v_isSharedCheck_1200_;
goto v_resetjp_1155_;
}
v_resetjp_1155_:
{
lean_object* v___x_1159_; 
lean_inc_ref(v_iM_1154_);
if (v_isShared_1157_ == 0)
{
v___x_1159_ = v___x_1156_;
goto v_reusejp_1158_;
}
else
{
lean_object* v_reuseFailAlloc_1199_; 
v_reuseFailAlloc_1199_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1199_, 0, v_iM_1154_);
v___x_1159_ = v_reuseFailAlloc_1199_;
goto v_reusejp_1158_;
}
v_reusejp_1158_:
{
lean_object* v___x_1160_; lean_object* v___x_1161_; lean_object* v___x_1162_; lean_object* v___x_1163_; lean_object* v___x_1164_; lean_object* v___x_1165_; lean_object* v___x_1166_; lean_object* v___x_1167_; lean_object* v___x_1168_; lean_object* v___x_1169_; lean_object* v___x_1170_; lean_object* v___x_1171_; lean_object* v___x_1172_; lean_object* v___x_1173_; lean_object* v___x_1174_; lean_object* v___x_1175_; lean_object* v___x_1176_; lean_object* v___x_1177_; lean_object* v___x_1178_; lean_object* v___x_1179_; lean_object* v___x_1180_; lean_object* v___x_1181_; lean_object* v___x_1182_; lean_object* v___x_1183_; lean_object* v___x_1184_; lean_object* v___x_1185_; lean_object* v___x_1186_; lean_object* v___x_1187_; lean_object* v___x_1188_; lean_object* v___x_1189_; lean_object* v___x_1190_; lean_object* v___x_1191_; lean_object* v___x_1192_; lean_object* v___x_1193_; lean_object* v___x_1194_; lean_object* v___x_1195_; lean_object* v___x_1196_; lean_object* v___x_1197_; lean_object* v___x_1198_; 
v___x_1160_ = lean_box(0);
v___x_1161_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1161_, 0, v_v_1109_);
lean_ctor_set(v___x_1161_, 1, v___x_1160_);
v___x_1162_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__11));
lean_inc_ref_n(v___x_1161_, 7);
v___x_1163_ = l_Lean_Expr_const___override(v___x_1162_, v___x_1161_);
lean_inc_ref_n(v_M_1110_, 7);
v___x_1164_ = l_Lean_Expr_app___override(v___x_1163_, v_M_1110_);
v___x_1165_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__14));
v___x_1166_ = l_Lean_Expr_const___override(v___x_1165_, v___x_1161_);
v___x_1167_ = l_Lean_Expr_app___override(v___x_1166_, v_M_1110_);
v___x_1168_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__16));
v___x_1169_ = l_Lean_Expr_const___override(v___x_1168_, v___x_1161_);
v___x_1170_ = l_Lean_Expr_app___override(v___x_1169_, v_M_1110_);
v___x_1171_ = l_Lean_Expr_app___override(v___x_1170_, v_iM_1111_);
v___x_1172_ = l_Lean_Expr_app___override(v___x_1167_, v___x_1171_);
v___x_1173_ = l_Lean_Expr_app___override(v___x_1164_, v___x_1172_);
v___x_1174_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__41));
v___x_1175_ = l_Lean_Expr_const___override(v___x_1174_, v___x_1161_);
v___x_1176_ = l_Lean_Expr_app___override(v___x_1175_, v_M_1110_);
v___x_1177_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__44));
v___x_1178_ = l_Lean_Expr_const___override(v___x_1177_, v___x_1161_);
v___x_1179_ = l_Lean_Expr_app___override(v___x_1178_, v_M_1110_);
v___x_1180_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__47));
v___x_1181_ = l_Lean_Expr_const___override(v___x_1180_, v___x_1161_);
v___x_1182_ = l_Lean_Expr_app___override(v___x_1181_, v_M_1110_);
v___x_1183_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__50));
v___x_1184_ = l_Lean_Expr_const___override(v___x_1183_, v___x_1161_);
v___x_1185_ = l_Lean_Expr_app___override(v___x_1184_, v_M_1110_);
v___x_1186_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__52));
v___x_1187_ = l_Lean_Expr_const___override(v___x_1186_, v___x_1161_);
v___x_1188_ = l_Lean_Expr_app___override(v___x_1187_, v_M_1110_);
v___x_1189_ = l_Lean_Expr_app___override(v___x_1188_, v_iM_1154_);
v___x_1190_ = l_Lean_Expr_app___override(v___x_1185_, v___x_1189_);
v___x_1191_ = l_Lean_Expr_app___override(v___x_1182_, v___x_1190_);
v___x_1192_ = l_Lean_Expr_app___override(v___x_1179_, v___x_1191_);
v___x_1193_ = l_Lean_Expr_app___override(v___x_1176_, v___x_1192_);
v___x_1194_ = l_Lean_Expr_app___override(v___x_1173_, v___x_1193_);
v___x_1195_ = l_Lean_Expr_app___override(v___x_1194_, v_y_u2082_1113_);
v___x_1196_ = l_Lean_Expr_app___override(v___x_1195_, v_y_u2081_1112_);
v___x_1197_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1197_, 0, v___x_1159_);
lean_ctor_set(v___x_1197_, 1, v___x_1196_);
v___x_1198_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1198_, 0, v___x_1197_);
return v___x_1198_;
}
}
}
}
else
{
if (lean_obj_tag(v_g_u2082_1115_) == 0)
{
lean_object* v_iM_1201_; lean_object* v___x_1203_; uint8_t v_isShared_1204_; uint8_t v_isSharedCheck_1247_; 
v_iM_1201_ = lean_ctor_get(v_g_u2081_1114_, 0);
v_isSharedCheck_1247_ = !lean_is_exclusive(v_g_u2081_1114_);
if (v_isSharedCheck_1247_ == 0)
{
v___x_1203_ = v_g_u2081_1114_;
v_isShared_1204_ = v_isSharedCheck_1247_;
goto v_resetjp_1202_;
}
else
{
lean_inc(v_iM_1201_);
lean_dec(v_g_u2081_1114_);
v___x_1203_ = lean_box(0);
v_isShared_1204_ = v_isSharedCheck_1247_;
goto v_resetjp_1202_;
}
v_resetjp_1202_:
{
lean_object* v___x_1206_; 
lean_inc_ref(v_iM_1201_);
if (v_isShared_1204_ == 0)
{
v___x_1206_ = v___x_1203_;
goto v_reusejp_1205_;
}
else
{
lean_object* v_reuseFailAlloc_1246_; 
v_reuseFailAlloc_1246_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1246_, 0, v_iM_1201_);
v___x_1206_ = v_reuseFailAlloc_1246_;
goto v_reusejp_1205_;
}
v_reusejp_1205_:
{
lean_object* v___x_1207_; lean_object* v___x_1208_; lean_object* v___x_1209_; lean_object* v___x_1210_; lean_object* v___x_1211_; lean_object* v___x_1212_; lean_object* v___x_1213_; lean_object* v___x_1214_; lean_object* v___x_1215_; lean_object* v___x_1216_; lean_object* v___x_1217_; lean_object* v___x_1218_; lean_object* v___x_1219_; lean_object* v___x_1220_; lean_object* v___x_1221_; lean_object* v___x_1222_; lean_object* v___x_1223_; lean_object* v___x_1224_; lean_object* v___x_1225_; lean_object* v___x_1226_; lean_object* v___x_1227_; lean_object* v___x_1228_; lean_object* v___x_1229_; lean_object* v___x_1230_; lean_object* v___x_1231_; lean_object* v___x_1232_; lean_object* v___x_1233_; lean_object* v___x_1234_; lean_object* v___x_1235_; lean_object* v___x_1236_; lean_object* v___x_1237_; lean_object* v___x_1238_; lean_object* v___x_1239_; lean_object* v___x_1240_; lean_object* v___x_1241_; lean_object* v___x_1242_; lean_object* v___x_1243_; lean_object* v___x_1244_; lean_object* v___x_1245_; 
v___x_1207_ = lean_box(0);
v___x_1208_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1208_, 0, v_v_1109_);
lean_ctor_set(v___x_1208_, 1, v___x_1207_);
v___x_1209_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__13));
lean_inc_ref_n(v___x_1208_, 7);
v___x_1210_ = l_Lean_Expr_const___override(v___x_1209_, v___x_1208_);
lean_inc_ref_n(v_M_1110_, 7);
v___x_1211_ = l_Lean_Expr_app___override(v___x_1210_, v_M_1110_);
v___x_1212_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__14));
v___x_1213_ = l_Lean_Expr_const___override(v___x_1212_, v___x_1208_);
v___x_1214_ = l_Lean_Expr_app___override(v___x_1213_, v_M_1110_);
v___x_1215_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__16));
v___x_1216_ = l_Lean_Expr_const___override(v___x_1215_, v___x_1208_);
v___x_1217_ = l_Lean_Expr_app___override(v___x_1216_, v_M_1110_);
v___x_1218_ = l_Lean_Expr_app___override(v___x_1217_, v_iM_1111_);
v___x_1219_ = l_Lean_Expr_app___override(v___x_1214_, v___x_1218_);
v___x_1220_ = l_Lean_Expr_app___override(v___x_1211_, v___x_1219_);
v___x_1221_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__41));
v___x_1222_ = l_Lean_Expr_const___override(v___x_1221_, v___x_1208_);
v___x_1223_ = l_Lean_Expr_app___override(v___x_1222_, v_M_1110_);
v___x_1224_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__44));
v___x_1225_ = l_Lean_Expr_const___override(v___x_1224_, v___x_1208_);
v___x_1226_ = l_Lean_Expr_app___override(v___x_1225_, v_M_1110_);
v___x_1227_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__47));
v___x_1228_ = l_Lean_Expr_const___override(v___x_1227_, v___x_1208_);
v___x_1229_ = l_Lean_Expr_app___override(v___x_1228_, v_M_1110_);
v___x_1230_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__50));
v___x_1231_ = l_Lean_Expr_const___override(v___x_1230_, v___x_1208_);
v___x_1232_ = l_Lean_Expr_app___override(v___x_1231_, v_M_1110_);
v___x_1233_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__52));
v___x_1234_ = l_Lean_Expr_const___override(v___x_1233_, v___x_1208_);
v___x_1235_ = l_Lean_Expr_app___override(v___x_1234_, v_M_1110_);
v___x_1236_ = l_Lean_Expr_app___override(v___x_1235_, v_iM_1201_);
v___x_1237_ = l_Lean_Expr_app___override(v___x_1232_, v___x_1236_);
v___x_1238_ = l_Lean_Expr_app___override(v___x_1229_, v___x_1237_);
v___x_1239_ = l_Lean_Expr_app___override(v___x_1226_, v___x_1238_);
v___x_1240_ = l_Lean_Expr_app___override(v___x_1223_, v___x_1239_);
v___x_1241_ = l_Lean_Expr_app___override(v___x_1220_, v___x_1240_);
v___x_1242_ = l_Lean_Expr_app___override(v___x_1241_, v_y_u2082_1113_);
v___x_1243_ = l_Lean_Expr_app___override(v___x_1242_, v_y_u2081_1112_);
v___x_1244_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1244_, 0, v___x_1206_);
lean_ctor_set(v___x_1244_, 1, v___x_1243_);
v___x_1245_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1245_, 0, v___x_1244_);
return v___x_1245_;
}
}
}
else
{
lean_object* v_iM_1248_; lean_object* v___x_1249_; lean_object* v___x_1250_; lean_object* v___x_1251_; lean_object* v___x_1252_; lean_object* v___x_1253_; lean_object* v___x_1254_; lean_object* v___x_1255_; lean_object* v___x_1256_; lean_object* v___x_1257_; lean_object* v___x_1258_; lean_object* v___x_1259_; lean_object* v___x_1260_; lean_object* v___x_1261_; lean_object* v___x_1262_; lean_object* v___x_1263_; lean_object* v___x_1264_; lean_object* v___x_1265_; lean_object* v___x_1266_; lean_object* v___x_1267_; lean_object* v___x_1268_; lean_object* v___x_1269_; lean_object* v___x_1270_; lean_object* v___x_1271_; lean_object* v___x_1272_; lean_object* v___x_1273_; lean_object* v___x_1274_; lean_object* v___x_1275_; lean_object* v___x_1276_; lean_object* v___x_1277_; lean_object* v___x_1278_; lean_object* v___x_1279_; lean_object* v___x_1280_; lean_object* v___x_1281_; lean_object* v___x_1282_; lean_object* v___x_1283_; lean_object* v___x_1284_; lean_object* v___x_1285_; lean_object* v___x_1286_; lean_object* v___x_1287_; lean_object* v___x_1288_; 
lean_dec_ref_known(v_g_u2082_1115_, 1);
v_iM_1248_ = lean_ctor_get(v_g_u2081_1114_, 0);
lean_inc_ref(v_iM_1248_);
lean_dec_ref_known(v_g_u2081_1114_, 1);
v___x_1249_ = lean_box(0);
v___x_1250_ = lean_box(0);
v___x_1251_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1251_, 0, v_v_1109_);
lean_ctor_set(v___x_1251_, 1, v___x_1250_);
v___x_1252_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__15));
lean_inc_ref_n(v___x_1251_, 7);
v___x_1253_ = l_Lean_Expr_const___override(v___x_1252_, v___x_1251_);
lean_inc_ref_n(v_M_1110_, 7);
v___x_1254_ = l_Lean_Expr_app___override(v___x_1253_, v_M_1110_);
v___x_1255_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__14));
v___x_1256_ = l_Lean_Expr_const___override(v___x_1255_, v___x_1251_);
v___x_1257_ = l_Lean_Expr_app___override(v___x_1256_, v_M_1110_);
v___x_1258_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__16));
v___x_1259_ = l_Lean_Expr_const___override(v___x_1258_, v___x_1251_);
v___x_1260_ = l_Lean_Expr_app___override(v___x_1259_, v_M_1110_);
v___x_1261_ = l_Lean_Expr_app___override(v___x_1260_, v_iM_1111_);
v___x_1262_ = l_Lean_Expr_app___override(v___x_1257_, v___x_1261_);
v___x_1263_ = l_Lean_Expr_app___override(v___x_1254_, v___x_1262_);
v___x_1264_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__41));
v___x_1265_ = l_Lean_Expr_const___override(v___x_1264_, v___x_1251_);
v___x_1266_ = l_Lean_Expr_app___override(v___x_1265_, v_M_1110_);
v___x_1267_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__44));
v___x_1268_ = l_Lean_Expr_const___override(v___x_1267_, v___x_1251_);
v___x_1269_ = l_Lean_Expr_app___override(v___x_1268_, v_M_1110_);
v___x_1270_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__47));
v___x_1271_ = l_Lean_Expr_const___override(v___x_1270_, v___x_1251_);
v___x_1272_ = l_Lean_Expr_app___override(v___x_1271_, v_M_1110_);
v___x_1273_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__50));
v___x_1274_ = l_Lean_Expr_const___override(v___x_1273_, v___x_1251_);
v___x_1275_ = l_Lean_Expr_app___override(v___x_1274_, v_M_1110_);
v___x_1276_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__52));
v___x_1277_ = l_Lean_Expr_const___override(v___x_1276_, v___x_1251_);
v___x_1278_ = l_Lean_Expr_app___override(v___x_1277_, v_M_1110_);
v___x_1279_ = l_Lean_Expr_app___override(v___x_1278_, v_iM_1248_);
v___x_1280_ = l_Lean_Expr_app___override(v___x_1275_, v___x_1279_);
v___x_1281_ = l_Lean_Expr_app___override(v___x_1272_, v___x_1280_);
v___x_1282_ = l_Lean_Expr_app___override(v___x_1269_, v___x_1281_);
v___x_1283_ = l_Lean_Expr_app___override(v___x_1266_, v___x_1282_);
v___x_1284_ = l_Lean_Expr_app___override(v___x_1263_, v___x_1283_);
v___x_1285_ = l_Lean_Expr_app___override(v___x_1284_, v_y_u2081_1112_);
v___x_1286_ = l_Lean_Expr_app___override(v___x_1285_, v_y_u2082_1113_);
v___x_1287_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1287_, 0, v___x_1249_);
lean_ctor_set(v___x_1287_, 1, v___x_1286_);
v___x_1288_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1288_, 0, v___x_1287_);
return v___x_1288_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___boxed(lean_object* v_v_1289_, lean_object* v_M_1290_, lean_object* v_iM_1291_, lean_object* v_y_u2081_1292_, lean_object* v_y_u2082_1293_, lean_object* v_g_u2081_1294_, lean_object* v_g_u2082_1295_, lean_object* v_a_1296_){
_start:
{
lean_object* v_res_1297_; 
v_res_1297_ = lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg(v_v_1289_, v_M_1290_, v_iM_1291_, v_y_u2081_1292_, v_y_u2082_1293_, v_g_u2081_1294_, v_g_u2082_1295_);
return v_res_1297_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div(lean_object* v_v_1298_, lean_object* v_M_1299_, lean_object* v_iM_1300_, lean_object* v_y_u2081_1301_, lean_object* v_y_u2082_1302_, lean_object* v_g_u2081_1303_, lean_object* v_g_u2082_1304_, lean_object* v_a_1305_, lean_object* v_a_1306_, lean_object* v_a_1307_, lean_object* v_a_1308_){
_start:
{
lean_object* v___x_1310_; 
v___x_1310_ = lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg(v_v_1298_, v_M_1299_, v_iM_1300_, v_y_u2081_1301_, v_y_u2082_1302_, v_g_u2081_1303_, v_g_u2082_1304_);
return v___x_1310_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___boxed(lean_object* v_v_1311_, lean_object* v_M_1312_, lean_object* v_iM_1313_, lean_object* v_y_u2081_1314_, lean_object* v_y_u2082_1315_, lean_object* v_g_u2081_1316_, lean_object* v_g_u2082_1317_, lean_object* v_a_1318_, lean_object* v_a_1319_, lean_object* v_a_1320_, lean_object* v_a_1321_, lean_object* v_a_1322_){
_start:
{
lean_object* v_res_1323_; 
v_res_1323_ = lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div(v_v_1311_, v_M_1312_, v_iM_1313_, v_y_u2081_1314_, v_y_u2082_1315_, v_g_u2081_1316_, v_g_u2082_1317_, v_a_1318_, v_a_1319_, v_a_1320_, v_a_1321_);
lean_dec(v_a_1321_);
lean_dec_ref(v_a_1320_);
lean_dec(v_a_1319_);
lean_dec_ref(v_a_1318_);
return v_res_1323_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_neg___redArg(lean_object* v_v_1341_, lean_object* v_M_1342_, lean_object* v_iM_1343_, lean_object* v_y_1344_, lean_object* v_g_1345_){
_start:
{
if (lean_obj_tag(v_g_1345_) == 0)
{
lean_object* v___x_1347_; lean_object* v___x_1348_; lean_object* v___x_1349_; lean_object* v___x_1350_; lean_object* v___x_1351_; lean_object* v___x_1352_; lean_object* v___x_1353_; lean_object* v___x_1354_; lean_object* v___x_1355_; lean_object* v___x_1356_; lean_object* v___x_1357_; lean_object* v___x_1358_; lean_object* v___x_1359_; lean_object* v___x_1360_; lean_object* v___x_1361_; lean_object* v___x_1362_; lean_object* v___x_1363_; lean_object* v___x_1364_; lean_object* v___x_1365_; lean_object* v___x_1366_; lean_object* v___x_1367_; lean_object* v___x_1368_; lean_object* v___x_1369_; lean_object* v___x_1370_; lean_object* v___x_1371_; lean_object* v___x_1372_; lean_object* v___x_1373_; lean_object* v___x_1374_; lean_object* v___x_1375_; lean_object* v___x_1376_; lean_object* v___x_1377_; lean_object* v___x_1378_; lean_object* v___x_1379_; lean_object* v___x_1380_; lean_object* v___x_1381_; lean_object* v___x_1382_; lean_object* v___x_1383_; lean_object* v___x_1384_; lean_object* v___x_1385_; lean_object* v___x_1386_; lean_object* v___x_1387_; lean_object* v___x_1388_; lean_object* v___x_1389_; lean_object* v___x_1390_; lean_object* v___x_1391_; lean_object* v___x_1392_; lean_object* v___x_1393_; lean_object* v___x_1394_; 
lean_inc_ref(v_iM_1343_);
v___x_1347_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1347_, 0, v_iM_1343_);
lean_inc(v_v_1341_);
v___x_1348_ = l_Lean_Level_succ___override(v_v_1341_);
v___x_1349_ = lean_box(0);
v___x_1350_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1350_, 0, v___x_1348_);
lean_ctor_set(v___x_1350_, 1, v___x_1349_);
v___x_1351_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__2));
v___x_1352_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1352_, 0, v_v_1341_);
lean_ctor_set(v___x_1352_, 1, v___x_1349_);
lean_inc_ref_n(v___x_1352_, 8);
v___x_1353_ = l_Lean_Expr_const___override(v___x_1351_, v___x_1352_);
lean_inc_ref_n(v_M_1342_, 9);
v___x_1354_ = l_Lean_Expr_app___override(v___x_1353_, v_M_1342_);
v___x_1355_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__5));
v___x_1356_ = l_Lean_Expr_const___override(v___x_1355_, v___x_1352_);
v___x_1357_ = l_Lean_Expr_app___override(v___x_1356_, v_M_1342_);
v___x_1358_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__8));
v___x_1359_ = l_Lean_Expr_const___override(v___x_1358_, v___x_1352_);
v___x_1360_ = l_Lean_Expr_app___override(v___x_1359_, v_M_1342_);
v___x_1361_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__11));
v___x_1362_ = l_Lean_Expr_const___override(v___x_1361_, v___x_1352_);
v___x_1363_ = l_Lean_Expr_app___override(v___x_1362_, v_M_1342_);
v___x_1364_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__14));
v___x_1365_ = l_Lean_Expr_const___override(v___x_1364_, v___x_1352_);
v___x_1366_ = l_Lean_Expr_app___override(v___x_1365_, v_M_1342_);
v___x_1367_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__17));
v___x_1368_ = l_Lean_Expr_const___override(v___x_1367_, v___x_1352_);
v___x_1369_ = l_Lean_Expr_app___override(v___x_1368_, v_M_1342_);
v___x_1370_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__20));
v___x_1371_ = l_Lean_Expr_const___override(v___x_1370_, v___x_1352_);
v___x_1372_ = l_Lean_Expr_app___override(v___x_1371_, v_M_1342_);
v___x_1373_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__23));
v___x_1374_ = l_Lean_Expr_const___override(v___x_1373_, v___x_1352_);
v___x_1375_ = l_Lean_Expr_app___override(v___x_1374_, v_M_1342_);
v___x_1376_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__26));
v___x_1377_ = l_Lean_Expr_const___override(v___x_1376_, v___x_1352_);
v___x_1378_ = l_Lean_Expr_app___override(v___x_1377_, v_M_1342_);
v___x_1379_ = l_Lean_Expr_app___override(v___x_1378_, v_iM_1343_);
v___x_1380_ = l_Lean_Expr_app___override(v___x_1375_, v___x_1379_);
v___x_1381_ = l_Lean_Expr_app___override(v___x_1372_, v___x_1380_);
v___x_1382_ = l_Lean_Expr_app___override(v___x_1369_, v___x_1381_);
v___x_1383_ = l_Lean_Expr_app___override(v___x_1366_, v___x_1382_);
v___x_1384_ = l_Lean_Expr_app___override(v___x_1363_, v___x_1383_);
v___x_1385_ = l_Lean_Expr_app___override(v___x_1360_, v___x_1384_);
v___x_1386_ = l_Lean_Expr_app___override(v___x_1357_, v___x_1385_);
v___x_1387_ = l_Lean_Expr_app___override(v___x_1354_, v___x_1386_);
v___x_1388_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__21));
v___x_1389_ = l_Lean_Expr_const___override(v___x_1388_, v___x_1350_);
v___x_1390_ = l_Lean_Expr_app___override(v___x_1389_, v_M_1342_);
v___x_1391_ = l_Lean_Expr_app___override(v___x_1387_, v_y_1344_);
v___x_1392_ = l_Lean_Expr_app___override(v___x_1390_, v___x_1391_);
v___x_1393_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1393_, 0, v___x_1347_);
lean_ctor_set(v___x_1393_, 1, v___x_1392_);
v___x_1394_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1394_, 0, v___x_1393_);
return v___x_1394_;
}
else
{
lean_object* v___x_1396_; uint8_t v_isShared_1397_; uint8_t v_isSharedCheck_1447_; 
v_isSharedCheck_1447_ = !lean_is_exclusive(v_g_1345_);
if (v_isSharedCheck_1447_ == 0)
{
lean_object* v_unused_1448_; 
v_unused_1448_ = lean_ctor_get(v_g_1345_, 0);
lean_dec(v_unused_1448_);
v___x_1396_ = v_g_1345_;
v_isShared_1397_ = v_isSharedCheck_1447_;
goto v_resetjp_1395_;
}
else
{
lean_dec(v_g_1345_);
v___x_1396_ = lean_box(0);
v_isShared_1397_ = v_isSharedCheck_1447_;
goto v_resetjp_1395_;
}
v_resetjp_1395_:
{
lean_object* v___x_1398_; lean_object* v___x_1399_; lean_object* v___x_1400_; lean_object* v___x_1401_; lean_object* v___x_1402_; lean_object* v___x_1403_; lean_object* v___x_1404_; lean_object* v___x_1405_; lean_object* v___x_1406_; lean_object* v___x_1407_; lean_object* v___x_1408_; lean_object* v___x_1409_; lean_object* v___x_1410_; lean_object* v___x_1411_; lean_object* v___x_1412_; lean_object* v___x_1413_; lean_object* v___x_1414_; lean_object* v___x_1415_; lean_object* v___x_1416_; lean_object* v___x_1417_; lean_object* v___x_1418_; lean_object* v___x_1419_; lean_object* v___x_1420_; lean_object* v___x_1421_; lean_object* v___x_1422_; lean_object* v___x_1423_; lean_object* v___x_1424_; lean_object* v___x_1425_; lean_object* v___x_1426_; lean_object* v___x_1427_; lean_object* v___x_1428_; lean_object* v___x_1429_; lean_object* v___x_1430_; lean_object* v___x_1431_; lean_object* v___x_1432_; lean_object* v___x_1433_; lean_object* v___x_1434_; lean_object* v___x_1435_; lean_object* v___x_1436_; lean_object* v___x_1437_; lean_object* v___x_1438_; lean_object* v___x_1439_; lean_object* v___x_1440_; lean_object* v___x_1441_; lean_object* v___x_1442_; lean_object* v___x_1443_; lean_object* v___x_1445_; 
v___x_1398_ = lean_box(0);
v___x_1399_ = lean_box(0);
v___x_1400_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1400_, 0, v_v_1341_);
lean_ctor_set(v___x_1400_, 1, v___x_1399_);
v___x_1401_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_neg___redArg___closed__1));
lean_inc_ref_n(v___x_1400_, 9);
v___x_1402_ = l_Lean_Expr_const___override(v___x_1401_, v___x_1400_);
lean_inc_ref_n(v_M_1342_, 9);
v___x_1403_ = l_Lean_Expr_app___override(v___x_1402_, v_M_1342_);
v___x_1404_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_neg___redArg___closed__4));
v___x_1405_ = l_Lean_Expr_const___override(v___x_1404_, v___x_1400_);
v___x_1406_ = l_Lean_Expr_app___override(v___x_1405_, v_M_1342_);
v___x_1407_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__28));
v___x_1408_ = l_Lean_Expr_const___override(v___x_1407_, v___x_1400_);
v___x_1409_ = l_Lean_Expr_app___override(v___x_1408_, v_M_1342_);
v___x_1410_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_neg___redArg___closed__7));
v___x_1411_ = l_Lean_Expr_const___override(v___x_1410_, v___x_1400_);
v___x_1412_ = l_Lean_Expr_app___override(v___x_1411_, v_M_1342_);
v___x_1413_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_neg___redArg___closed__9));
v___x_1414_ = l_Lean_Expr_const___override(v___x_1413_, v___x_1400_);
v___x_1415_ = l_Lean_Expr_app___override(v___x_1414_, v_M_1342_);
v___x_1416_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__44));
v___x_1417_ = l_Lean_Expr_const___override(v___x_1416_, v___x_1400_);
v___x_1418_ = l_Lean_Expr_app___override(v___x_1417_, v_M_1342_);
v___x_1419_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__47));
v___x_1420_ = l_Lean_Expr_const___override(v___x_1419_, v___x_1400_);
v___x_1421_ = l_Lean_Expr_app___override(v___x_1420_, v_M_1342_);
v___x_1422_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__50));
v___x_1423_ = l_Lean_Expr_const___override(v___x_1422_, v___x_1400_);
v___x_1424_ = l_Lean_Expr_app___override(v___x_1423_, v_M_1342_);
v___x_1425_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__52));
v___x_1426_ = l_Lean_Expr_const___override(v___x_1425_, v___x_1400_);
v___x_1427_ = l_Lean_Expr_app___override(v___x_1426_, v_M_1342_);
v___x_1428_ = l_Lean_Expr_app___override(v___x_1427_, v_iM_1343_);
v___x_1429_ = l_Lean_Expr_app___override(v___x_1424_, v___x_1428_);
v___x_1430_ = l_Lean_Expr_app___override(v___x_1421_, v___x_1429_);
v___x_1431_ = l_Lean_Expr_app___override(v___x_1418_, v___x_1430_);
lean_inc_ref(v___x_1431_);
v___x_1432_ = l_Lean_Expr_app___override(v___x_1415_, v___x_1431_);
v___x_1433_ = l_Lean_Expr_app___override(v___x_1412_, v___x_1432_);
v___x_1434_ = l_Lean_Expr_app___override(v___x_1409_, v___x_1433_);
v___x_1435_ = l_Lean_Expr_app___override(v___x_1406_, v___x_1434_);
v___x_1436_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__41));
v___x_1437_ = l_Lean_Expr_const___override(v___x_1436_, v___x_1400_);
v___x_1438_ = l_Lean_Expr_app___override(v___x_1437_, v_M_1342_);
v___x_1439_ = l_Lean_Expr_app___override(v___x_1438_, v___x_1431_);
v___x_1440_ = l_Lean_Expr_app___override(v___x_1435_, v___x_1439_);
v___x_1441_ = l_Lean_Expr_app___override(v___x_1403_, v___x_1440_);
v___x_1442_ = l_Lean_Expr_app___override(v___x_1441_, v_y_1344_);
v___x_1443_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1443_, 0, v___x_1398_);
lean_ctor_set(v___x_1443_, 1, v___x_1442_);
if (v_isShared_1397_ == 0)
{
lean_ctor_set_tag(v___x_1396_, 0);
lean_ctor_set(v___x_1396_, 0, v___x_1443_);
v___x_1445_ = v___x_1396_;
goto v_reusejp_1444_;
}
else
{
lean_object* v_reuseFailAlloc_1446_; 
v_reuseFailAlloc_1446_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1446_, 0, v___x_1443_);
v___x_1445_ = v_reuseFailAlloc_1446_;
goto v_reusejp_1444_;
}
v_reusejp_1444_:
{
return v___x_1445_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_neg___redArg___boxed(lean_object* v_v_1449_, lean_object* v_M_1450_, lean_object* v_iM_1451_, lean_object* v_y_1452_, lean_object* v_g_1453_, lean_object* v_a_1454_){
_start:
{
lean_object* v_res_1455_; 
v_res_1455_ = lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_neg___redArg(v_v_1449_, v_M_1450_, v_iM_1451_, v_y_1452_, v_g_1453_);
return v_res_1455_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_neg(lean_object* v_v_1456_, lean_object* v_M_1457_, lean_object* v_iM_1458_, lean_object* v_y_1459_, lean_object* v_g_1460_, lean_object* v_a_1461_, lean_object* v_a_1462_, lean_object* v_a_1463_, lean_object* v_a_1464_){
_start:
{
lean_object* v___x_1466_; 
v___x_1466_ = lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_neg___redArg(v_v_1456_, v_M_1457_, v_iM_1458_, v_y_1459_, v_g_1460_);
return v___x_1466_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_neg___boxed(lean_object* v_v_1467_, lean_object* v_M_1468_, lean_object* v_iM_1469_, lean_object* v_y_1470_, lean_object* v_g_1471_, lean_object* v_a_1472_, lean_object* v_a_1473_, lean_object* v_a_1474_, lean_object* v_a_1475_, lean_object* v_a_1476_){
_start:
{
lean_object* v_res_1477_; 
v_res_1477_ = lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_neg(v_v_1467_, v_M_1468_, v_iM_1469_, v_y_1470_, v_g_1471_, v_a_1472_, v_a_1473_, v_a_1474_, v_a_1475_);
lean_dec(v_a_1475_);
lean_dec_ref(v_a_1474_);
lean_dec(v_a_1473_);
lean_dec_ref(v_a_1472_);
return v_res_1477_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__5(void){
_start:
{
lean_object* v___x_1486_; lean_object* v___x_1487_; lean_object* v___x_1488_; 
v___x_1486_ = lean_box(0);
v___x_1487_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__4));
v___x_1488_ = l_Lean_Expr_const___override(v___x_1487_, v___x_1486_);
return v___x_1488_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__19(void){
_start:
{
lean_object* v___x_1512_; lean_object* v___x_1513_; lean_object* v___x_1514_; 
v___x_1512_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__8));
v___x_1513_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__18));
v___x_1514_ = l_Lean_Expr_const___override(v___x_1513_, v___x_1512_);
return v___x_1514_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__20(void){
_start:
{
lean_object* v___x_1515_; lean_object* v___x_1516_; lean_object* v___x_1517_; 
v___x_1515_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__5, &lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__5);
v___x_1516_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__19, &lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__19_once, _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__19);
v___x_1517_ = l_Lean_Expr_app___override(v___x_1516_, v___x_1515_);
return v___x_1517_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__23(void){
_start:
{
lean_object* v___x_1522_; lean_object* v___x_1523_; lean_object* v___x_1524_; 
v___x_1522_ = lean_box(0);
v___x_1523_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__22));
v___x_1524_ = l_Lean_Expr_const___override(v___x_1523_, v___x_1522_);
return v___x_1524_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__24(void){
_start:
{
lean_object* v___x_1525_; lean_object* v___x_1526_; lean_object* v___x_1527_; 
v___x_1525_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__23, &lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__23_once, _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__23);
v___x_1526_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__20, &lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__20_once, _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__20);
v___x_1527_ = l_Lean_Expr_app___override(v___x_1526_, v___x_1525_);
return v___x_1527_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__31(void){
_start:
{
lean_object* v___x_1539_; lean_object* v___x_1540_; lean_object* v___x_1541_; 
v___x_1539_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__8));
v___x_1540_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__30));
v___x_1541_ = l_Lean_Expr_const___override(v___x_1540_, v___x_1539_);
return v___x_1541_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__32(void){
_start:
{
lean_object* v___x_1542_; lean_object* v___x_1543_; lean_object* v___x_1544_; 
v___x_1542_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__5, &lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__5);
v___x_1543_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__31, &lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__31_once, _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__31);
v___x_1544_ = l_Lean_Expr_app___override(v___x_1543_, v___x_1542_);
return v___x_1544_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__35(void){
_start:
{
lean_object* v___x_1548_; lean_object* v___x_1549_; lean_object* v___x_1550_; 
v___x_1548_ = lean_box(0);
v___x_1549_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__34));
v___x_1550_ = l_Lean_Expr_const___override(v___x_1549_, v___x_1548_);
return v___x_1550_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__36(void){
_start:
{
lean_object* v___x_1551_; lean_object* v___x_1552_; lean_object* v___x_1553_; 
v___x_1551_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__35, &lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__35_once, _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__35);
v___x_1552_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__32, &lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__32_once, _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__32);
v___x_1553_ = l_Lean_Expr_app___override(v___x_1552_, v___x_1551_);
return v___x_1553_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow(lean_object* v_v_1557_, lean_object* v_M_1558_, lean_object* v_iM_1559_, lean_object* v_y_1560_, lean_object* v_g_1561_, lean_object* v_s_1562_, lean_object* v_a_1563_, lean_object* v_a_1564_, lean_object* v_a_1565_, lean_object* v_a_1566_){
_start:
{
if (lean_obj_tag(v_g_1561_) == 0)
{
lean_object* v___x_1568_; lean_object* v___x_1569_; lean_object* v___x_1570_; lean_object* v___x_1571_; lean_object* v___x_1572_; lean_object* v___x_1573_; lean_object* v___x_1574_; lean_object* v___x_1575_; lean_object* v___x_1576_; lean_object* v___x_1577_; lean_object* v___x_1578_; lean_object* v___x_1579_; lean_object* v___x_1580_; lean_object* v___x_1581_; lean_object* v___x_1582_; lean_object* v___x_1583_; lean_object* v___x_1584_; lean_object* v___x_1585_; lean_object* v___x_1586_; lean_object* v___x_1587_; lean_object* v___x_1588_; lean_object* v___x_1589_; lean_object* v___x_1590_; lean_object* v___x_1591_; lean_object* v___x_1592_; lean_object* v___x_1593_; lean_object* v___x_1594_; lean_object* v___x_1595_; lean_object* v___x_1596_; lean_object* v___x_1597_; lean_object* v___x_1598_; lean_object* v___x_1599_; lean_object* v___x_1600_; lean_object* v___x_1601_; lean_object* v___x_1602_; lean_object* v___x_1603_; lean_object* v___x_1604_; lean_object* v___x_1605_; lean_object* v___x_1606_; lean_object* v___x_1607_; lean_object* v___x_1608_; lean_object* v___x_1609_; lean_object* v___x_1610_; lean_object* v___x_1611_; lean_object* v___x_1612_; lean_object* v___x_1613_; lean_object* v___x_1614_; lean_object* v___x_1615_; lean_object* v___x_1616_; lean_object* v___x_1617_; lean_object* v___x_1618_; 
v___x_1568_ = lean_box(0);
lean_inc_n(v_v_1557_, 3);
v___x_1569_ = l_Lean_Level_succ___override(v_v_1557_);
v___x_1570_ = lean_box(0);
v___x_1571_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1571_, 0, v___x_1569_);
lean_ctor_set(v___x_1571_, 1, v___x_1570_);
v___x_1572_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__2));
v___x_1573_ = lean_box(0);
v___x_1574_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1574_, 0, v_v_1557_);
lean_ctor_set(v___x_1574_, 1, v___x_1570_);
lean_inc_ref_n(v___x_1574_, 5);
v___x_1575_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1575_, 0, v___x_1573_);
lean_ctor_set(v___x_1575_, 1, v___x_1574_);
v___x_1576_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1576_, 0, v_v_1557_);
lean_ctor_set(v___x_1576_, 1, v___x_1575_);
v___x_1577_ = l_Lean_Expr_const___override(v___x_1572_, v___x_1576_);
lean_inc_ref_n(v_M_1558_, 8);
v___x_1578_ = l_Lean_Expr_app___override(v___x_1577_, v_M_1558_);
v___x_1579_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__5, &lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__5);
v___x_1580_ = l_Lean_Expr_app___override(v___x_1578_, v___x_1579_);
v___x_1581_ = l_Lean_Expr_app___override(v___x_1580_, v_M_1558_);
v___x_1582_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__7));
v___x_1583_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__8));
v___x_1584_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1584_, 0, v_v_1557_);
lean_ctor_set(v___x_1584_, 1, v___x_1583_);
v___x_1585_ = l_Lean_Expr_const___override(v___x_1582_, v___x_1584_);
v___x_1586_ = l_Lean_Expr_app___override(v___x_1585_, v_M_1558_);
v___x_1587_ = l_Lean_Expr_app___override(v___x_1586_, v___x_1579_);
v___x_1588_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__11));
v___x_1589_ = l_Lean_Expr_const___override(v___x_1588_, v___x_1574_);
v___x_1590_ = l_Lean_Expr_app___override(v___x_1589_, v_M_1558_);
v___x_1591_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__14));
v___x_1592_ = l_Lean_Expr_const___override(v___x_1591_, v___x_1574_);
v___x_1593_ = l_Lean_Expr_app___override(v___x_1592_, v_M_1558_);
v___x_1594_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__16));
v___x_1595_ = l_Lean_Expr_const___override(v___x_1594_, v___x_1574_);
v___x_1596_ = l_Lean_Expr_app___override(v___x_1595_, v_M_1558_);
v___x_1597_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__16));
v___x_1598_ = l_Lean_Expr_const___override(v___x_1597_, v___x_1574_);
v___x_1599_ = l_Lean_Expr_app___override(v___x_1598_, v_M_1558_);
v___x_1600_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__19));
v___x_1601_ = l_Lean_Expr_const___override(v___x_1600_, v___x_1574_);
v___x_1602_ = l_Lean_Expr_app___override(v___x_1601_, v_M_1558_);
v___x_1603_ = l_Lean_Expr_app___override(v___x_1602_, v_iM_1559_);
v___x_1604_ = l_Lean_Expr_app___override(v___x_1599_, v___x_1603_);
v___x_1605_ = l_Lean_Expr_app___override(v___x_1596_, v___x_1604_);
v___x_1606_ = l_Lean_Expr_app___override(v___x_1593_, v___x_1605_);
v___x_1607_ = l_Lean_Expr_app___override(v___x_1590_, v___x_1606_);
v___x_1608_ = l_Lean_Expr_app___override(v___x_1587_, v___x_1607_);
v___x_1609_ = l_Lean_Expr_app___override(v___x_1581_, v___x_1608_);
v___x_1610_ = l_Lean_mkNatLit(v_s_1562_);
v___x_1611_ = l_Lean_Expr_app___override(v___x_1609_, v_y_1560_);
v___x_1612_ = l_Lean_Expr_app___override(v___x_1611_, v___x_1610_);
v___x_1613_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__21));
v___x_1614_ = l_Lean_Expr_const___override(v___x_1613_, v___x_1571_);
v___x_1615_ = l_Lean_Expr_app___override(v___x_1614_, v_M_1558_);
v___x_1616_ = l_Lean_Expr_app___override(v___x_1615_, v___x_1612_);
v___x_1617_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1617_, 0, v___x_1568_);
lean_ctor_set(v___x_1617_, 1, v___x_1616_);
v___x_1618_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1618_, 0, v___x_1617_);
return v___x_1618_;
}
else
{
lean_object* v_iM_1619_; lean_object* v___x_1621_; uint8_t v_isShared_1622_; uint8_t v_isSharedCheck_1763_; 
lean_dec_ref(v_iM_1559_);
v_iM_1619_ = lean_ctor_get(v_g_1561_, 0);
v_isSharedCheck_1763_ = !lean_is_exclusive(v_g_1561_);
if (v_isSharedCheck_1763_ == 0)
{
v___x_1621_ = v_g_1561_;
v_isShared_1622_ = v_isSharedCheck_1763_;
goto v_resetjp_1620_;
}
else
{
lean_inc(v_iM_1619_);
lean_dec(v_g_1561_);
v___x_1621_ = lean_box(0);
v_isShared_1622_ = v_isSharedCheck_1763_;
goto v_resetjp_1620_;
}
v_resetjp_1620_:
{
lean_object* v___x_1623_; uint8_t v___x_1624_; 
v___x_1623_ = lean_unsigned_to_nat(2u);
v___x_1624_ = l_Nat_decidable__dvd(v___x_1623_, v_s_1562_);
if (v___x_1624_ == 0)
{
lean_object* v___x_1625_; lean_object* v___x_1626_; lean_object* v___x_1627_; lean_object* v___x_1628_; lean_object* v___x_1629_; 
v___x_1625_ = lean_box(0);
v___x_1626_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__24, &lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__24_once, _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__24);
v___x_1627_ = l_Lean_mkNatLit(v_s_1562_);
lean_inc_ref(v___x_1627_);
v___x_1628_ = l_Lean_Expr_app___override(v___x_1626_, v___x_1627_);
v___x_1629_ = lp_mathlib_Qq_mkDecideProofQ(v___x_1628_, v_a_1563_, v_a_1564_, v_a_1565_, v_a_1566_);
if (lean_obj_tag(v___x_1629_) == 0)
{
lean_object* v_a_1630_; lean_object* v___x_1632_; uint8_t v_isShared_1633_; uint8_t v_isSharedCheck_1686_; 
v_a_1630_ = lean_ctor_get(v___x_1629_, 0);
v_isSharedCheck_1686_ = !lean_is_exclusive(v___x_1629_);
if (v_isSharedCheck_1686_ == 0)
{
v___x_1632_ = v___x_1629_;
v_isShared_1633_ = v_isSharedCheck_1686_;
goto v_resetjp_1631_;
}
else
{
lean_inc(v_a_1630_);
lean_dec(v___x_1629_);
v___x_1632_ = lean_box(0);
v_isShared_1633_ = v_isSharedCheck_1686_;
goto v_resetjp_1631_;
}
v_resetjp_1631_:
{
lean_object* v___x_1635_; 
lean_inc_ref(v_iM_1619_);
if (v_isShared_1622_ == 0)
{
v___x_1635_ = v___x_1621_;
goto v_reusejp_1634_;
}
else
{
lean_object* v_reuseFailAlloc_1685_; 
v_reuseFailAlloc_1685_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1685_, 0, v_iM_1619_);
v___x_1635_ = v_reuseFailAlloc_1685_;
goto v_reusejp_1634_;
}
v_reusejp_1634_:
{
lean_object* v___x_1636_; lean_object* v___x_1637_; lean_object* v___x_1638_; lean_object* v___x_1639_; lean_object* v___x_1640_; lean_object* v___x_1641_; lean_object* v___x_1642_; lean_object* v___x_1643_; lean_object* v___x_1644_; lean_object* v___x_1645_; lean_object* v___x_1646_; lean_object* v___x_1647_; lean_object* v___x_1648_; lean_object* v___x_1649_; lean_object* v___x_1650_; lean_object* v___x_1651_; lean_object* v___x_1652_; lean_object* v___x_1653_; lean_object* v___x_1654_; lean_object* v___x_1655_; lean_object* v___x_1656_; lean_object* v___x_1657_; lean_object* v___x_1658_; lean_object* v___x_1659_; lean_object* v___x_1660_; lean_object* v___x_1661_; lean_object* v___x_1662_; lean_object* v___x_1663_; lean_object* v___x_1664_; lean_object* v___x_1665_; lean_object* v___x_1666_; lean_object* v___x_1667_; lean_object* v___x_1668_; lean_object* v___x_1669_; lean_object* v___x_1670_; lean_object* v___x_1671_; lean_object* v___x_1672_; lean_object* v___x_1673_; lean_object* v___x_1674_; lean_object* v___x_1675_; lean_object* v___x_1676_; lean_object* v___x_1677_; lean_object* v___x_1678_; lean_object* v___x_1679_; lean_object* v___x_1680_; lean_object* v___x_1681_; lean_object* v___x_1683_; 
v___x_1636_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1636_, 0, v_v_1557_);
lean_ctor_set(v___x_1636_, 1, v___x_1625_);
v___x_1637_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__26));
lean_inc_ref_n(v___x_1636_, 9);
v___x_1638_ = l_Lean_Expr_const___override(v___x_1637_, v___x_1636_);
lean_inc_ref_n(v_M_1558_, 9);
v___x_1639_ = l_Lean_Expr_app___override(v___x_1638_, v_M_1558_);
v___x_1640_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__28));
v___x_1641_ = l_Lean_Expr_const___override(v___x_1640_, v___x_1636_);
v___x_1642_ = l_Lean_Expr_app___override(v___x_1641_, v_M_1558_);
v___x_1643_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__33));
v___x_1644_ = l_Lean_Expr_const___override(v___x_1643_, v___x_1636_);
v___x_1645_ = l_Lean_Expr_app___override(v___x_1644_, v_M_1558_);
v___x_1646_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__36));
v___x_1647_ = l_Lean_Expr_const___override(v___x_1646_, v___x_1636_);
v___x_1648_ = l_Lean_Expr_app___override(v___x_1647_, v_M_1558_);
v___x_1649_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__38));
v___x_1650_ = l_Lean_Expr_const___override(v___x_1649_, v___x_1636_);
v___x_1651_ = l_Lean_Expr_app___override(v___x_1650_, v_M_1558_);
lean_inc_ref(v_iM_1619_);
v___x_1652_ = l_Lean_Expr_app___override(v___x_1651_, v_iM_1619_);
v___x_1653_ = l_Lean_Expr_app___override(v___x_1648_, v___x_1652_);
v___x_1654_ = l_Lean_Expr_app___override(v___x_1645_, v___x_1653_);
v___x_1655_ = l_Lean_Expr_app___override(v___x_1642_, v___x_1654_);
v___x_1656_ = l_Lean_Expr_app___override(v___x_1639_, v___x_1655_);
v___x_1657_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__41));
v___x_1658_ = l_Lean_Expr_const___override(v___x_1657_, v___x_1636_);
v___x_1659_ = l_Lean_Expr_app___override(v___x_1658_, v_M_1558_);
v___x_1660_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__44));
v___x_1661_ = l_Lean_Expr_const___override(v___x_1660_, v___x_1636_);
v___x_1662_ = l_Lean_Expr_app___override(v___x_1661_, v_M_1558_);
v___x_1663_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__47));
v___x_1664_ = l_Lean_Expr_const___override(v___x_1663_, v___x_1636_);
v___x_1665_ = l_Lean_Expr_app___override(v___x_1664_, v_M_1558_);
v___x_1666_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__50));
v___x_1667_ = l_Lean_Expr_const___override(v___x_1666_, v___x_1636_);
v___x_1668_ = l_Lean_Expr_app___override(v___x_1667_, v_M_1558_);
v___x_1669_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__52));
v___x_1670_ = l_Lean_Expr_const___override(v___x_1669_, v___x_1636_);
v___x_1671_ = l_Lean_Expr_app___override(v___x_1670_, v_M_1558_);
v___x_1672_ = l_Lean_Expr_app___override(v___x_1671_, v_iM_1619_);
v___x_1673_ = l_Lean_Expr_app___override(v___x_1668_, v___x_1672_);
v___x_1674_ = l_Lean_Expr_app___override(v___x_1665_, v___x_1673_);
v___x_1675_ = l_Lean_Expr_app___override(v___x_1662_, v___x_1674_);
v___x_1676_ = l_Lean_Expr_app___override(v___x_1659_, v___x_1675_);
v___x_1677_ = l_Lean_Expr_app___override(v___x_1656_, v___x_1676_);
v___x_1678_ = l_Lean_Expr_app___override(v___x_1677_, v___x_1627_);
v___x_1679_ = l_Lean_Expr_app___override(v___x_1678_, v_a_1630_);
v___x_1680_ = l_Lean_Expr_app___override(v___x_1679_, v_y_1560_);
v___x_1681_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1681_, 0, v___x_1635_);
lean_ctor_set(v___x_1681_, 1, v___x_1680_);
if (v_isShared_1633_ == 0)
{
lean_ctor_set(v___x_1632_, 0, v___x_1681_);
v___x_1683_ = v___x_1632_;
goto v_reusejp_1682_;
}
else
{
lean_object* v_reuseFailAlloc_1684_; 
v_reuseFailAlloc_1684_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1684_, 0, v___x_1681_);
v___x_1683_ = v_reuseFailAlloc_1684_;
goto v_reusejp_1682_;
}
v_reusejp_1682_:
{
return v___x_1683_;
}
}
}
}
else
{
lean_object* v_a_1687_; lean_object* v___x_1689_; uint8_t v_isShared_1690_; uint8_t v_isSharedCheck_1694_; 
lean_dec_ref(v___x_1627_);
lean_del_object(v___x_1621_);
lean_dec_ref(v_iM_1619_);
lean_dec_ref(v_y_1560_);
lean_dec_ref(v_M_1558_);
lean_dec(v_v_1557_);
v_a_1687_ = lean_ctor_get(v___x_1629_, 0);
v_isSharedCheck_1694_ = !lean_is_exclusive(v___x_1629_);
if (v_isSharedCheck_1694_ == 0)
{
v___x_1689_ = v___x_1629_;
v_isShared_1690_ = v_isSharedCheck_1694_;
goto v_resetjp_1688_;
}
else
{
lean_inc(v_a_1687_);
lean_dec(v___x_1629_);
v___x_1689_ = lean_box(0);
v_isShared_1690_ = v_isSharedCheck_1694_;
goto v_resetjp_1688_;
}
v_resetjp_1688_:
{
lean_object* v___x_1692_; 
if (v_isShared_1690_ == 0)
{
v___x_1692_ = v___x_1689_;
goto v_reusejp_1691_;
}
else
{
lean_object* v_reuseFailAlloc_1693_; 
v_reuseFailAlloc_1693_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1693_, 0, v_a_1687_);
v___x_1692_ = v_reuseFailAlloc_1693_;
goto v_reusejp_1691_;
}
v_reusejp_1691_:
{
return v___x_1692_;
}
}
}
}
else
{
lean_object* v___x_1695_; lean_object* v___x_1696_; lean_object* v___x_1697_; lean_object* v___x_1698_; lean_object* v___x_1699_; 
lean_del_object(v___x_1621_);
v___x_1695_ = lean_box(0);
v___x_1696_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__36, &lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__36_once, _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__36);
v___x_1697_ = l_Lean_mkNatLit(v_s_1562_);
lean_inc_ref(v___x_1697_);
v___x_1698_ = l_Lean_Expr_app___override(v___x_1696_, v___x_1697_);
v___x_1699_ = lp_mathlib_Qq_mkDecideProofQ(v___x_1698_, v_a_1563_, v_a_1564_, v_a_1565_, v_a_1566_);
if (lean_obj_tag(v___x_1699_) == 0)
{
lean_object* v_a_1700_; lean_object* v___x_1702_; uint8_t v_isShared_1703_; uint8_t v_isSharedCheck_1754_; 
v_a_1700_ = lean_ctor_get(v___x_1699_, 0);
v_isSharedCheck_1754_ = !lean_is_exclusive(v___x_1699_);
if (v_isSharedCheck_1754_ == 0)
{
v___x_1702_ = v___x_1699_;
v_isShared_1703_ = v_isSharedCheck_1754_;
goto v_resetjp_1701_;
}
else
{
lean_inc(v_a_1700_);
lean_dec(v___x_1699_);
v___x_1702_ = lean_box(0);
v_isShared_1703_ = v_isSharedCheck_1754_;
goto v_resetjp_1701_;
}
v_resetjp_1701_:
{
lean_object* v___x_1704_; lean_object* v___x_1705_; lean_object* v___x_1706_; lean_object* v___x_1707_; lean_object* v___x_1708_; lean_object* v___x_1709_; lean_object* v___x_1710_; lean_object* v___x_1711_; lean_object* v___x_1712_; lean_object* v___x_1713_; lean_object* v___x_1714_; lean_object* v___x_1715_; lean_object* v___x_1716_; lean_object* v___x_1717_; lean_object* v___x_1718_; lean_object* v___x_1719_; lean_object* v___x_1720_; lean_object* v___x_1721_; lean_object* v___x_1722_; lean_object* v___x_1723_; lean_object* v___x_1724_; lean_object* v___x_1725_; lean_object* v___x_1726_; lean_object* v___x_1727_; lean_object* v___x_1728_; lean_object* v___x_1729_; lean_object* v___x_1730_; lean_object* v___x_1731_; lean_object* v___x_1732_; lean_object* v___x_1733_; lean_object* v___x_1734_; lean_object* v___x_1735_; lean_object* v___x_1736_; lean_object* v___x_1737_; lean_object* v___x_1738_; lean_object* v___x_1739_; lean_object* v___x_1740_; lean_object* v___x_1741_; lean_object* v___x_1742_; lean_object* v___x_1743_; lean_object* v___x_1744_; lean_object* v___x_1745_; lean_object* v___x_1746_; lean_object* v___x_1747_; lean_object* v___x_1748_; lean_object* v___x_1749_; lean_object* v___x_1750_; lean_object* v___x_1752_; 
v___x_1704_ = lean_box(0);
v___x_1705_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1705_, 0, v_v_1557_);
lean_ctor_set(v___x_1705_, 1, v___x_1695_);
v___x_1706_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__37));
lean_inc_ref_n(v___x_1705_, 9);
v___x_1707_ = l_Lean_Expr_const___override(v___x_1706_, v___x_1705_);
lean_inc_ref_n(v_M_1558_, 9);
v___x_1708_ = l_Lean_Expr_app___override(v___x_1707_, v_M_1558_);
v___x_1709_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__28));
v___x_1710_ = l_Lean_Expr_const___override(v___x_1709_, v___x_1705_);
v___x_1711_ = l_Lean_Expr_app___override(v___x_1710_, v_M_1558_);
v___x_1712_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__33));
v___x_1713_ = l_Lean_Expr_const___override(v___x_1712_, v___x_1705_);
v___x_1714_ = l_Lean_Expr_app___override(v___x_1713_, v_M_1558_);
v___x_1715_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__36));
v___x_1716_ = l_Lean_Expr_const___override(v___x_1715_, v___x_1705_);
v___x_1717_ = l_Lean_Expr_app___override(v___x_1716_, v_M_1558_);
v___x_1718_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__38));
v___x_1719_ = l_Lean_Expr_const___override(v___x_1718_, v___x_1705_);
v___x_1720_ = l_Lean_Expr_app___override(v___x_1719_, v_M_1558_);
lean_inc_ref(v_iM_1619_);
v___x_1721_ = l_Lean_Expr_app___override(v___x_1720_, v_iM_1619_);
v___x_1722_ = l_Lean_Expr_app___override(v___x_1717_, v___x_1721_);
v___x_1723_ = l_Lean_Expr_app___override(v___x_1714_, v___x_1722_);
v___x_1724_ = l_Lean_Expr_app___override(v___x_1711_, v___x_1723_);
v___x_1725_ = l_Lean_Expr_app___override(v___x_1708_, v___x_1724_);
v___x_1726_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__41));
v___x_1727_ = l_Lean_Expr_const___override(v___x_1726_, v___x_1705_);
v___x_1728_ = l_Lean_Expr_app___override(v___x_1727_, v_M_1558_);
v___x_1729_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__44));
v___x_1730_ = l_Lean_Expr_const___override(v___x_1729_, v___x_1705_);
v___x_1731_ = l_Lean_Expr_app___override(v___x_1730_, v_M_1558_);
v___x_1732_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__47));
v___x_1733_ = l_Lean_Expr_const___override(v___x_1732_, v___x_1705_);
v___x_1734_ = l_Lean_Expr_app___override(v___x_1733_, v_M_1558_);
v___x_1735_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__50));
v___x_1736_ = l_Lean_Expr_const___override(v___x_1735_, v___x_1705_);
v___x_1737_ = l_Lean_Expr_app___override(v___x_1736_, v_M_1558_);
v___x_1738_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__52));
v___x_1739_ = l_Lean_Expr_const___override(v___x_1738_, v___x_1705_);
v___x_1740_ = l_Lean_Expr_app___override(v___x_1739_, v_M_1558_);
v___x_1741_ = l_Lean_Expr_app___override(v___x_1740_, v_iM_1619_);
v___x_1742_ = l_Lean_Expr_app___override(v___x_1737_, v___x_1741_);
v___x_1743_ = l_Lean_Expr_app___override(v___x_1734_, v___x_1742_);
v___x_1744_ = l_Lean_Expr_app___override(v___x_1731_, v___x_1743_);
v___x_1745_ = l_Lean_Expr_app___override(v___x_1728_, v___x_1744_);
v___x_1746_ = l_Lean_Expr_app___override(v___x_1725_, v___x_1745_);
v___x_1747_ = l_Lean_Expr_app___override(v___x_1746_, v___x_1697_);
v___x_1748_ = l_Lean_Expr_app___override(v___x_1747_, v_a_1700_);
v___x_1749_ = l_Lean_Expr_app___override(v___x_1748_, v_y_1560_);
v___x_1750_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1750_, 0, v___x_1704_);
lean_ctor_set(v___x_1750_, 1, v___x_1749_);
if (v_isShared_1703_ == 0)
{
lean_ctor_set(v___x_1702_, 0, v___x_1750_);
v___x_1752_ = v___x_1702_;
goto v_reusejp_1751_;
}
else
{
lean_object* v_reuseFailAlloc_1753_; 
v_reuseFailAlloc_1753_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1753_, 0, v___x_1750_);
v___x_1752_ = v_reuseFailAlloc_1753_;
goto v_reusejp_1751_;
}
v_reusejp_1751_:
{
return v___x_1752_;
}
}
}
else
{
lean_object* v_a_1755_; lean_object* v___x_1757_; uint8_t v_isShared_1758_; uint8_t v_isSharedCheck_1762_; 
lean_dec_ref(v___x_1697_);
lean_dec_ref(v_iM_1619_);
lean_dec_ref(v_y_1560_);
lean_dec_ref(v_M_1558_);
lean_dec(v_v_1557_);
v_a_1755_ = lean_ctor_get(v___x_1699_, 0);
v_isSharedCheck_1762_ = !lean_is_exclusive(v___x_1699_);
if (v_isSharedCheck_1762_ == 0)
{
v___x_1757_ = v___x_1699_;
v_isShared_1758_ = v_isSharedCheck_1762_;
goto v_resetjp_1756_;
}
else
{
lean_inc(v_a_1755_);
lean_dec(v___x_1699_);
v___x_1757_ = lean_box(0);
v_isShared_1758_ = v_isSharedCheck_1762_;
goto v_resetjp_1756_;
}
v_resetjp_1756_:
{
lean_object* v___x_1760_; 
if (v_isShared_1758_ == 0)
{
v___x_1760_ = v___x_1757_;
goto v_reusejp_1759_;
}
else
{
lean_object* v_reuseFailAlloc_1761_; 
v_reuseFailAlloc_1761_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1761_, 0, v_a_1755_);
v___x_1760_ = v_reuseFailAlloc_1761_;
goto v_reusejp_1759_;
}
v_reusejp_1759_:
{
return v___x_1760_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___boxed(lean_object* v_v_1764_, lean_object* v_M_1765_, lean_object* v_iM_1766_, lean_object* v_y_1767_, lean_object* v_g_1768_, lean_object* v_s_1769_, lean_object* v_a_1770_, lean_object* v_a_1771_, lean_object* v_a_1772_, lean_object* v_a_1773_, lean_object* v_a_1774_){
_start:
{
lean_object* v_res_1775_; 
v_res_1775_ = lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow(v_v_1764_, v_M_1765_, v_iM_1766_, v_y_1767_, v_g_1768_, v_s_1769_, v_a_1770_, v_a_1771_, v_a_1772_, v_a_1773_);
lean_dec(v_a_1773_);
lean_dec_ref(v_a_1772_);
lean_dec(v_a_1771_);
lean_dec_ref(v_a_1770_);
return v_res_1775_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__2(void){
_start:
{
lean_object* v___x_1779_; lean_object* v___x_1780_; lean_object* v___x_1781_; 
v___x_1779_ = lean_box(0);
v___x_1780_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__1));
v___x_1781_ = l_Lean_Expr_const___override(v___x_1780_, v___x_1779_);
return v___x_1781_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__7(void){
_start:
{
lean_object* v___x_1790_; lean_object* v___x_1791_; 
v___x_1790_ = lean_unsigned_to_nat(0u);
v___x_1791_ = lean_nat_to_int(v___x_1790_);
return v___x_1791_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__8(void){
_start:
{
lean_object* v___x_1792_; lean_object* v___x_1793_; 
v___x_1792_ = lean_unsigned_to_nat(0u);
v___x_1793_ = l_Lean_Level_ofNat(v___x_1792_);
return v___x_1793_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__9(void){
_start:
{
lean_object* v___x_1794_; lean_object* v___x_1795_; lean_object* v___x_1796_; 
v___x_1794_ = lean_box(0);
v___x_1795_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__8, &lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__8);
v___x_1796_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1796_, 0, v___x_1795_);
lean_ctor_set(v___x_1796_, 1, v___x_1794_);
return v___x_1796_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__10(void){
_start:
{
lean_object* v___x_1797_; lean_object* v___x_1798_; lean_object* v___x_1799_; 
v___x_1797_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__9, &lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__9_once, _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__9);
v___x_1798_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__2));
v___x_1799_ = l_Lean_Expr_const___override(v___x_1798_, v___x_1797_);
return v___x_1799_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__13(void){
_start:
{
lean_object* v___x_1804_; lean_object* v___x_1805_; lean_object* v___x_1806_; 
v___x_1804_ = lean_box(0);
v___x_1805_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__12));
v___x_1806_ = l_Lean_Expr_const___override(v___x_1805_, v___x_1804_);
return v___x_1806_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__14(void){
_start:
{
lean_object* v___x_1807_; lean_object* v___x_1808_; 
v___x_1807_ = lean_unsigned_to_nat(2u);
v___x_1808_ = lean_nat_to_int(v___x_1807_);
return v___x_1808_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__15(void){
_start:
{
lean_object* v___x_1809_; lean_object* v___x_1810_; lean_object* v___x_1811_; 
v___x_1809_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__2, &lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__2);
v___x_1810_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__19, &lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__19_once, _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__19);
v___x_1811_ = l_Lean_Expr_app___override(v___x_1810_, v___x_1809_);
return v___x_1811_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__17(void){
_start:
{
lean_object* v___x_1815_; lean_object* v___x_1816_; lean_object* v___x_1817_; 
v___x_1815_ = lean_box(0);
v___x_1816_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__16));
v___x_1817_ = l_Lean_Expr_const___override(v___x_1816_, v___x_1815_);
return v___x_1817_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__18(void){
_start:
{
lean_object* v___x_1818_; lean_object* v___x_1819_; lean_object* v___x_1820_; 
v___x_1818_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__17, &lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__17_once, _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__17);
v___x_1819_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__15, &lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__15_once, _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__15);
v___x_1820_ = l_Lean_Expr_app___override(v___x_1819_, v___x_1818_);
return v___x_1820_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__21(void){
_start:
{
lean_object* v___x_1825_; lean_object* v___x_1826_; lean_object* v___x_1827_; 
v___x_1825_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__2, &lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__2);
v___x_1826_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__31, &lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__31_once, _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__31);
v___x_1827_ = l_Lean_Expr_app___override(v___x_1826_, v___x_1825_);
return v___x_1827_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__24(void){
_start:
{
lean_object* v___x_1832_; lean_object* v___x_1833_; lean_object* v___x_1834_; 
v___x_1832_ = lean_box(0);
v___x_1833_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__23));
v___x_1834_ = l_Lean_Expr_const___override(v___x_1833_, v___x_1832_);
return v___x_1834_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__25(void){
_start:
{
lean_object* v___x_1835_; lean_object* v___x_1836_; lean_object* v___x_1837_; 
v___x_1835_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__24, &lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__24_once, _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__24);
v___x_1836_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__21, &lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__21_once, _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__21);
v___x_1837_ = l_Lean_Expr_app___override(v___x_1836_, v___x_1835_);
return v___x_1837_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow(lean_object* v_v_1841_, lean_object* v_M_1842_, lean_object* v_iM_1843_, lean_object* v_y_1844_, lean_object* v_g_1845_, lean_object* v_s_1846_, lean_object* v_a_1847_, lean_object* v_a_1848_, lean_object* v_a_1849_, lean_object* v_a_1850_){
_start:
{
if (lean_obj_tag(v_g_1845_) == 0)
{
lean_object* v___x_1852_; lean_object* v___x_1853_; lean_object* v___x_1854_; lean_object* v___x_1855_; lean_object* v___x_1856_; lean_object* v___x_1857_; lean_object* v___x_1858_; lean_object* v___x_1859_; lean_object* v___x_1860_; lean_object* v___x_1861_; lean_object* v___x_1862_; lean_object* v___x_1863_; lean_object* v___x_1864_; lean_object* v___x_1865_; lean_object* v___x_1866_; lean_object* v___x_1867_; lean_object* v___x_1868_; lean_object* v___x_1869_; lean_object* v___x_1870_; lean_object* v___x_1871_; lean_object* v___x_1872_; lean_object* v___x_1873_; lean_object* v___x_1874_; lean_object* v___x_1875_; lean_object* v___x_1876_; lean_object* v___x_1877_; lean_object* v___x_1878_; lean_object* v___x_1879_; lean_object* v___x_1880_; lean_object* v___x_1881_; lean_object* v___x_1882_; lean_object* v___x_1883_; lean_object* v___x_1884_; lean_object* v___x_1885_; lean_object* v___x_1886_; lean_object* v___x_1887_; lean_object* v___x_1888_; lean_object* v___x_1889_; lean_object* v___y_1891_; lean_object* v___x_1900_; uint8_t v___x_1901_; 
v___x_1852_ = lean_box(0);
lean_inc_n(v_v_1841_, 3);
v___x_1853_ = l_Lean_Level_succ___override(v_v_1841_);
v___x_1854_ = lean_box(0);
v___x_1855_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1855_, 0, v___x_1853_);
lean_ctor_set(v___x_1855_, 1, v___x_1854_);
v___x_1856_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__2));
v___x_1857_ = lean_box(0);
v___x_1858_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1858_, 0, v_v_1841_);
lean_ctor_set(v___x_1858_, 1, v___x_1854_);
lean_inc_ref_n(v___x_1858_, 4);
v___x_1859_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1859_, 0, v___x_1857_);
lean_ctor_set(v___x_1859_, 1, v___x_1858_);
v___x_1860_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1860_, 0, v_v_1841_);
lean_ctor_set(v___x_1860_, 1, v___x_1859_);
v___x_1861_ = l_Lean_Expr_const___override(v___x_1856_, v___x_1860_);
lean_inc_ref_n(v_M_1842_, 7);
v___x_1862_ = l_Lean_Expr_app___override(v___x_1861_, v_M_1842_);
v___x_1863_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__2, &lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__2);
v___x_1864_ = l_Lean_Expr_app___override(v___x_1862_, v___x_1863_);
v___x_1865_ = l_Lean_Expr_app___override(v___x_1864_, v_M_1842_);
v___x_1866_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__7));
v___x_1867_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_pow___closed__8));
v___x_1868_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1868_, 0, v_v_1841_);
lean_ctor_set(v___x_1868_, 1, v___x_1867_);
v___x_1869_ = l_Lean_Expr_const___override(v___x_1866_, v___x_1868_);
v___x_1870_ = l_Lean_Expr_app___override(v___x_1869_, v_M_1842_);
v___x_1871_ = l_Lean_Expr_app___override(v___x_1870_, v___x_1863_);
v___x_1872_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__4));
v___x_1873_ = l_Lean_Expr_const___override(v___x_1872_, v___x_1858_);
v___x_1874_ = l_Lean_Expr_app___override(v___x_1873_, v_M_1842_);
v___x_1875_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__6));
v___x_1876_ = l_Lean_Expr_const___override(v___x_1875_, v___x_1858_);
v___x_1877_ = l_Lean_Expr_app___override(v___x_1876_, v_M_1842_);
v___x_1878_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_div___redArg___closed__9));
v___x_1879_ = l_Lean_Expr_const___override(v___x_1878_, v___x_1858_);
v___x_1880_ = l_Lean_Expr_app___override(v___x_1879_, v_M_1842_);
v___x_1881_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__19));
v___x_1882_ = l_Lean_Expr_const___override(v___x_1881_, v___x_1858_);
v___x_1883_ = l_Lean_Expr_app___override(v___x_1882_, v_M_1842_);
v___x_1884_ = l_Lean_Expr_app___override(v___x_1883_, v_iM_1843_);
v___x_1885_ = l_Lean_Expr_app___override(v___x_1880_, v___x_1884_);
v___x_1886_ = l_Lean_Expr_app___override(v___x_1877_, v___x_1885_);
v___x_1887_ = l_Lean_Expr_app___override(v___x_1874_, v___x_1886_);
v___x_1888_ = l_Lean_Expr_app___override(v___x_1871_, v___x_1887_);
v___x_1889_ = l_Lean_Expr_app___override(v___x_1865_, v___x_1888_);
v___x_1900_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__7, &lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__7);
v___x_1901_ = lean_int_dec_le(v___x_1900_, v_s_1846_);
if (v___x_1901_ == 0)
{
lean_object* v___x_1902_; lean_object* v___x_1903_; lean_object* v___x_1904_; lean_object* v___x_1905_; lean_object* v___x_1906_; lean_object* v___x_1907_; 
v___x_1902_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__10, &lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__10_once, _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__10);
v___x_1903_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__13, &lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__13_once, _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__13);
v___x_1904_ = lean_int_neg(v_s_1846_);
v___x_1905_ = l_Int_toNat(v___x_1904_);
lean_dec(v___x_1904_);
v___x_1906_ = l_Lean_instToExprInt_mkNat(v___x_1905_);
v___x_1907_ = l_Lean_mkApp3(v___x_1902_, v___x_1863_, v___x_1903_, v___x_1906_);
v___y_1891_ = v___x_1907_;
goto v___jp_1890_;
}
else
{
lean_object* v___x_1908_; lean_object* v___x_1909_; 
v___x_1908_ = l_Int_toNat(v_s_1846_);
v___x_1909_ = l_Lean_instToExprInt_mkNat(v___x_1908_);
v___y_1891_ = v___x_1909_;
goto v___jp_1890_;
}
v___jp_1890_:
{
lean_object* v___x_1892_; lean_object* v___x_1893_; lean_object* v___x_1894_; lean_object* v___x_1895_; lean_object* v___x_1896_; lean_object* v___x_1897_; lean_object* v___x_1898_; lean_object* v___x_1899_; 
v___x_1892_ = l_Lean_Expr_app___override(v___x_1889_, v_y_1844_);
v___x_1893_ = l_Lean_Expr_app___override(v___x_1892_, v___y_1891_);
v___x_1894_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__21));
v___x_1895_ = l_Lean_Expr_const___override(v___x_1894_, v___x_1855_);
v___x_1896_ = l_Lean_Expr_app___override(v___x_1895_, v_M_1842_);
v___x_1897_ = l_Lean_Expr_app___override(v___x_1896_, v___x_1893_);
v___x_1898_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1898_, 0, v___x_1852_);
lean_ctor_set(v___x_1898_, 1, v___x_1897_);
v___x_1899_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1899_, 0, v___x_1898_);
return v___x_1899_;
}
}
else
{
lean_object* v_iM_1910_; lean_object* v___x_1912_; uint8_t v_isShared_1913_; uint8_t v_isSharedCheck_2062_; 
v_iM_1910_ = lean_ctor_get(v_g_1845_, 0);
v_isSharedCheck_2062_ = !lean_is_exclusive(v_g_1845_);
if (v_isSharedCheck_2062_ == 0)
{
v___x_1912_ = v_g_1845_;
v_isShared_1913_ = v_isSharedCheck_2062_;
goto v_resetjp_1911_;
}
else
{
lean_inc(v_iM_1910_);
lean_dec(v_g_1845_);
v___x_1912_ = lean_box(0);
v_isShared_1913_ = v_isSharedCheck_2062_;
goto v_resetjp_1911_;
}
v_resetjp_1911_:
{
lean_object* v___x_1914_; uint8_t v___x_1915_; 
v___x_1914_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__14, &lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__14_once, _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__14);
v___x_1915_ = l_Int_decidableDvd(v___x_1914_, v_s_1846_);
if (v___x_1915_ == 0)
{
lean_object* v___x_1916_; lean_object* v___x_1917_; lean_object* v___x_1918_; lean_object* v___y_1920_; lean_object* v___x_1980_; uint8_t v___x_1981_; 
v___x_1916_ = lean_box(0);
v___x_1917_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__2, &lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__2);
v___x_1918_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__18, &lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__18_once, _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__18);
v___x_1980_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__7, &lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__7);
v___x_1981_ = lean_int_dec_le(v___x_1980_, v_s_1846_);
if (v___x_1981_ == 0)
{
lean_object* v___x_1982_; lean_object* v___x_1983_; lean_object* v___x_1984_; lean_object* v___x_1985_; lean_object* v___x_1986_; lean_object* v___x_1987_; 
v___x_1982_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__10, &lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__10_once, _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__10);
v___x_1983_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__13, &lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__13_once, _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__13);
v___x_1984_ = lean_int_neg(v_s_1846_);
v___x_1985_ = l_Int_toNat(v___x_1984_);
lean_dec(v___x_1984_);
v___x_1986_ = l_Lean_instToExprInt_mkNat(v___x_1985_);
v___x_1987_ = l_Lean_mkApp3(v___x_1982_, v___x_1917_, v___x_1983_, v___x_1986_);
v___y_1920_ = v___x_1987_;
goto v___jp_1919_;
}
else
{
lean_object* v___x_1988_; lean_object* v___x_1989_; 
v___x_1988_ = l_Int_toNat(v_s_1846_);
v___x_1989_ = l_Lean_instToExprInt_mkNat(v___x_1988_);
v___y_1920_ = v___x_1989_;
goto v___jp_1919_;
}
v___jp_1919_:
{
lean_object* v___x_1921_; lean_object* v___x_1922_; 
lean_inc_ref(v___y_1920_);
v___x_1921_ = l_Lean_Expr_app___override(v___x_1918_, v___y_1920_);
v___x_1922_ = lp_mathlib_Qq_mkDecideProofQ(v___x_1921_, v_a_1847_, v_a_1848_, v_a_1849_, v_a_1850_);
if (lean_obj_tag(v___x_1922_) == 0)
{
lean_object* v_a_1923_; lean_object* v___x_1925_; uint8_t v_isShared_1926_; uint8_t v_isSharedCheck_1971_; 
v_a_1923_ = lean_ctor_get(v___x_1922_, 0);
v_isSharedCheck_1971_ = !lean_is_exclusive(v___x_1922_);
if (v_isSharedCheck_1971_ == 0)
{
v___x_1925_ = v___x_1922_;
v_isShared_1926_ = v_isSharedCheck_1971_;
goto v_resetjp_1924_;
}
else
{
lean_inc(v_a_1923_);
lean_dec(v___x_1922_);
v___x_1925_ = lean_box(0);
v_isShared_1926_ = v_isSharedCheck_1971_;
goto v_resetjp_1924_;
}
v_resetjp_1924_:
{
lean_object* v___x_1928_; 
lean_inc_ref(v_iM_1910_);
if (v_isShared_1913_ == 0)
{
v___x_1928_ = v___x_1912_;
goto v_reusejp_1927_;
}
else
{
lean_object* v_reuseFailAlloc_1970_; 
v_reuseFailAlloc_1970_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1970_, 0, v_iM_1910_);
v___x_1928_ = v_reuseFailAlloc_1970_;
goto v_reusejp_1927_;
}
v_reusejp_1927_:
{
lean_object* v___x_1929_; lean_object* v___x_1930_; lean_object* v___x_1931_; lean_object* v___x_1932_; lean_object* v___x_1933_; lean_object* v___x_1934_; lean_object* v___x_1935_; lean_object* v___x_1936_; lean_object* v___x_1937_; lean_object* v___x_1938_; lean_object* v___x_1939_; lean_object* v___x_1940_; lean_object* v___x_1941_; lean_object* v___x_1942_; lean_object* v___x_1943_; lean_object* v___x_1944_; lean_object* v___x_1945_; lean_object* v___x_1946_; lean_object* v___x_1947_; lean_object* v___x_1948_; lean_object* v___x_1949_; lean_object* v___x_1950_; lean_object* v___x_1951_; lean_object* v___x_1952_; lean_object* v___x_1953_; lean_object* v___x_1954_; lean_object* v___x_1955_; lean_object* v___x_1956_; lean_object* v___x_1957_; lean_object* v___x_1958_; lean_object* v___x_1959_; lean_object* v___x_1960_; lean_object* v___x_1961_; lean_object* v___x_1962_; lean_object* v___x_1963_; lean_object* v___x_1964_; lean_object* v___x_1965_; lean_object* v___x_1966_; lean_object* v___x_1968_; 
v___x_1929_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1929_, 0, v_v_1841_);
lean_ctor_set(v___x_1929_, 1, v___x_1916_);
v___x_1930_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__20));
lean_inc_ref_n(v___x_1929_, 7);
v___x_1931_ = l_Lean_Expr_const___override(v___x_1930_, v___x_1929_);
lean_inc_ref_n(v_M_1842_, 7);
v___x_1932_ = l_Lean_Expr_app___override(v___x_1931_, v_M_1842_);
v___x_1933_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__14));
v___x_1934_ = l_Lean_Expr_const___override(v___x_1933_, v___x_1929_);
v___x_1935_ = l_Lean_Expr_app___override(v___x_1934_, v_M_1842_);
v___x_1936_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__16));
v___x_1937_ = l_Lean_Expr_const___override(v___x_1936_, v___x_1929_);
v___x_1938_ = l_Lean_Expr_app___override(v___x_1937_, v_M_1842_);
v___x_1939_ = l_Lean_Expr_app___override(v___x_1938_, v_iM_1843_);
v___x_1940_ = l_Lean_Expr_app___override(v___x_1935_, v___x_1939_);
v___x_1941_ = l_Lean_Expr_app___override(v___x_1932_, v___x_1940_);
v___x_1942_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__41));
v___x_1943_ = l_Lean_Expr_const___override(v___x_1942_, v___x_1929_);
v___x_1944_ = l_Lean_Expr_app___override(v___x_1943_, v_M_1842_);
v___x_1945_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__44));
v___x_1946_ = l_Lean_Expr_const___override(v___x_1945_, v___x_1929_);
v___x_1947_ = l_Lean_Expr_app___override(v___x_1946_, v_M_1842_);
v___x_1948_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__47));
v___x_1949_ = l_Lean_Expr_const___override(v___x_1948_, v___x_1929_);
v___x_1950_ = l_Lean_Expr_app___override(v___x_1949_, v_M_1842_);
v___x_1951_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__50));
v___x_1952_ = l_Lean_Expr_const___override(v___x_1951_, v___x_1929_);
v___x_1953_ = l_Lean_Expr_app___override(v___x_1952_, v_M_1842_);
v___x_1954_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__52));
v___x_1955_ = l_Lean_Expr_const___override(v___x_1954_, v___x_1929_);
v___x_1956_ = l_Lean_Expr_app___override(v___x_1955_, v_M_1842_);
v___x_1957_ = l_Lean_Expr_app___override(v___x_1956_, v_iM_1910_);
v___x_1958_ = l_Lean_Expr_app___override(v___x_1953_, v___x_1957_);
v___x_1959_ = l_Lean_Expr_app___override(v___x_1950_, v___x_1958_);
v___x_1960_ = l_Lean_Expr_app___override(v___x_1947_, v___x_1959_);
v___x_1961_ = l_Lean_Expr_app___override(v___x_1944_, v___x_1960_);
v___x_1962_ = l_Lean_Expr_app___override(v___x_1941_, v___x_1961_);
v___x_1963_ = l_Lean_Expr_app___override(v___x_1962_, v___y_1920_);
v___x_1964_ = l_Lean_Expr_app___override(v___x_1963_, v_a_1923_);
v___x_1965_ = l_Lean_Expr_app___override(v___x_1964_, v_y_1844_);
v___x_1966_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1966_, 0, v___x_1928_);
lean_ctor_set(v___x_1966_, 1, v___x_1965_);
if (v_isShared_1926_ == 0)
{
lean_ctor_set(v___x_1925_, 0, v___x_1966_);
v___x_1968_ = v___x_1925_;
goto v_reusejp_1967_;
}
else
{
lean_object* v_reuseFailAlloc_1969_; 
v_reuseFailAlloc_1969_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1969_, 0, v___x_1966_);
v___x_1968_ = v_reuseFailAlloc_1969_;
goto v_reusejp_1967_;
}
v_reusejp_1967_:
{
return v___x_1968_;
}
}
}
}
else
{
lean_object* v_a_1972_; lean_object* v___x_1974_; uint8_t v_isShared_1975_; uint8_t v_isSharedCheck_1979_; 
lean_dec_ref(v___y_1920_);
lean_del_object(v___x_1912_);
lean_dec_ref(v_iM_1910_);
lean_dec_ref(v_y_1844_);
lean_dec_ref(v_iM_1843_);
lean_dec_ref(v_M_1842_);
lean_dec(v_v_1841_);
v_a_1972_ = lean_ctor_get(v___x_1922_, 0);
v_isSharedCheck_1979_ = !lean_is_exclusive(v___x_1922_);
if (v_isSharedCheck_1979_ == 0)
{
v___x_1974_ = v___x_1922_;
v_isShared_1975_ = v_isSharedCheck_1979_;
goto v_resetjp_1973_;
}
else
{
lean_inc(v_a_1972_);
lean_dec(v___x_1922_);
v___x_1974_ = lean_box(0);
v_isShared_1975_ = v_isSharedCheck_1979_;
goto v_resetjp_1973_;
}
v_resetjp_1973_:
{
lean_object* v___x_1977_; 
if (v_isShared_1975_ == 0)
{
v___x_1977_ = v___x_1974_;
goto v_reusejp_1976_;
}
else
{
lean_object* v_reuseFailAlloc_1978_; 
v_reuseFailAlloc_1978_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1978_, 0, v_a_1972_);
v___x_1977_ = v_reuseFailAlloc_1978_;
goto v_reusejp_1976_;
}
v_reusejp_1976_:
{
return v___x_1977_;
}
}
}
}
}
else
{
lean_object* v___x_1990_; lean_object* v___x_1991_; lean_object* v___x_1992_; lean_object* v___y_1994_; lean_object* v___x_2052_; uint8_t v___x_2053_; 
lean_del_object(v___x_1912_);
v___x_1990_ = lean_box(0);
v___x_1991_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__2, &lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__2);
v___x_1992_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__25, &lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__25_once, _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__25);
v___x_2052_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__7, &lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__7);
v___x_2053_ = lean_int_dec_le(v___x_2052_, v_s_1846_);
if (v___x_2053_ == 0)
{
lean_object* v___x_2054_; lean_object* v___x_2055_; lean_object* v___x_2056_; lean_object* v___x_2057_; lean_object* v___x_2058_; lean_object* v___x_2059_; 
v___x_2054_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__10, &lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__10_once, _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__10);
v___x_2055_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__13, &lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__13_once, _init_lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__13);
v___x_2056_ = lean_int_neg(v_s_1846_);
v___x_2057_ = l_Int_toNat(v___x_2056_);
lean_dec(v___x_2056_);
v___x_2058_ = l_Lean_instToExprInt_mkNat(v___x_2057_);
v___x_2059_ = l_Lean_mkApp3(v___x_2054_, v___x_1991_, v___x_2055_, v___x_2058_);
v___y_1994_ = v___x_2059_;
goto v___jp_1993_;
}
else
{
lean_object* v___x_2060_; lean_object* v___x_2061_; 
v___x_2060_ = l_Int_toNat(v_s_1846_);
v___x_2061_ = l_Lean_instToExprInt_mkNat(v___x_2060_);
v___y_1994_ = v___x_2061_;
goto v___jp_1993_;
}
v___jp_1993_:
{
lean_object* v___x_1995_; lean_object* v___x_1996_; 
lean_inc_ref(v___y_1994_);
v___x_1995_ = l_Lean_Expr_app___override(v___x_1992_, v___y_1994_);
v___x_1996_ = lp_mathlib_Qq_mkDecideProofQ(v___x_1995_, v_a_1847_, v_a_1848_, v_a_1849_, v_a_1850_);
if (lean_obj_tag(v___x_1996_) == 0)
{
lean_object* v_a_1997_; lean_object* v___x_1999_; uint8_t v_isShared_2000_; uint8_t v_isSharedCheck_2043_; 
v_a_1997_ = lean_ctor_get(v___x_1996_, 0);
v_isSharedCheck_2043_ = !lean_is_exclusive(v___x_1996_);
if (v_isSharedCheck_2043_ == 0)
{
v___x_1999_ = v___x_1996_;
v_isShared_2000_ = v_isSharedCheck_2043_;
goto v_resetjp_1998_;
}
else
{
lean_inc(v_a_1997_);
lean_dec(v___x_1996_);
v___x_1999_ = lean_box(0);
v_isShared_2000_ = v_isSharedCheck_2043_;
goto v_resetjp_1998_;
}
v_resetjp_1998_:
{
lean_object* v___x_2001_; lean_object* v___x_2002_; lean_object* v___x_2003_; lean_object* v___x_2004_; lean_object* v___x_2005_; lean_object* v___x_2006_; lean_object* v___x_2007_; lean_object* v___x_2008_; lean_object* v___x_2009_; lean_object* v___x_2010_; lean_object* v___x_2011_; lean_object* v___x_2012_; lean_object* v___x_2013_; lean_object* v___x_2014_; lean_object* v___x_2015_; lean_object* v___x_2016_; lean_object* v___x_2017_; lean_object* v___x_2018_; lean_object* v___x_2019_; lean_object* v___x_2020_; lean_object* v___x_2021_; lean_object* v___x_2022_; lean_object* v___x_2023_; lean_object* v___x_2024_; lean_object* v___x_2025_; lean_object* v___x_2026_; lean_object* v___x_2027_; lean_object* v___x_2028_; lean_object* v___x_2029_; lean_object* v___x_2030_; lean_object* v___x_2031_; lean_object* v___x_2032_; lean_object* v___x_2033_; lean_object* v___x_2034_; lean_object* v___x_2035_; lean_object* v___x_2036_; lean_object* v___x_2037_; lean_object* v___x_2038_; lean_object* v___x_2039_; lean_object* v___x_2041_; 
v___x_2001_ = lean_box(0);
v___x_2002_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2002_, 0, v_v_1841_);
lean_ctor_set(v___x_2002_, 1, v___x_1990_);
v___x_2003_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___closed__26));
lean_inc_ref_n(v___x_2002_, 7);
v___x_2004_ = l_Lean_Expr_const___override(v___x_2003_, v___x_2002_);
lean_inc_ref_n(v_M_1842_, 7);
v___x_2005_ = l_Lean_Expr_app___override(v___x_2004_, v_M_1842_);
v___x_2006_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__14));
v___x_2007_ = l_Lean_Expr_const___override(v___x_2006_, v___x_2002_);
v___x_2008_ = l_Lean_Expr_app___override(v___x_2007_, v_M_1842_);
v___x_2009_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_inv___redArg___closed__16));
v___x_2010_ = l_Lean_Expr_const___override(v___x_2009_, v___x_2002_);
v___x_2011_ = l_Lean_Expr_app___override(v___x_2010_, v_M_1842_);
v___x_2012_ = l_Lean_Expr_app___override(v___x_2011_, v_iM_1843_);
v___x_2013_ = l_Lean_Expr_app___override(v___x_2008_, v___x_2012_);
v___x_2014_ = l_Lean_Expr_app___override(v___x_2005_, v___x_2013_);
v___x_2015_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__41));
v___x_2016_ = l_Lean_Expr_const___override(v___x_2015_, v___x_2002_);
v___x_2017_ = l_Lean_Expr_app___override(v___x_2016_, v_M_1842_);
v___x_2018_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__44));
v___x_2019_ = l_Lean_Expr_const___override(v___x_2018_, v___x_2002_);
v___x_2020_ = l_Lean_Expr_app___override(v___x_2019_, v_M_1842_);
v___x_2021_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__47));
v___x_2022_ = l_Lean_Expr_const___override(v___x_2021_, v___x_2002_);
v___x_2023_ = l_Lean_Expr_app___override(v___x_2022_, v_M_1842_);
v___x_2024_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__50));
v___x_2025_ = l_Lean_Expr_const___override(v___x_2024_, v___x_2002_);
v___x_2026_ = l_Lean_Expr_app___override(v___x_2025_, v_M_1842_);
v___x_2027_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__52));
v___x_2028_ = l_Lean_Expr_const___override(v___x_2027_, v___x_2002_);
v___x_2029_ = l_Lean_Expr_app___override(v___x_2028_, v_M_1842_);
v___x_2030_ = l_Lean_Expr_app___override(v___x_2029_, v_iM_1910_);
v___x_2031_ = l_Lean_Expr_app___override(v___x_2026_, v___x_2030_);
v___x_2032_ = l_Lean_Expr_app___override(v___x_2023_, v___x_2031_);
v___x_2033_ = l_Lean_Expr_app___override(v___x_2020_, v___x_2032_);
v___x_2034_ = l_Lean_Expr_app___override(v___x_2017_, v___x_2033_);
v___x_2035_ = l_Lean_Expr_app___override(v___x_2014_, v___x_2034_);
v___x_2036_ = l_Lean_Expr_app___override(v___x_2035_, v___y_1994_);
v___x_2037_ = l_Lean_Expr_app___override(v___x_2036_, v_a_1997_);
v___x_2038_ = l_Lean_Expr_app___override(v___x_2037_, v_y_1844_);
v___x_2039_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2039_, 0, v___x_2001_);
lean_ctor_set(v___x_2039_, 1, v___x_2038_);
if (v_isShared_2000_ == 0)
{
lean_ctor_set(v___x_1999_, 0, v___x_2039_);
v___x_2041_ = v___x_1999_;
goto v_reusejp_2040_;
}
else
{
lean_object* v_reuseFailAlloc_2042_; 
v_reuseFailAlloc_2042_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2042_, 0, v___x_2039_);
v___x_2041_ = v_reuseFailAlloc_2042_;
goto v_reusejp_2040_;
}
v_reusejp_2040_:
{
return v___x_2041_;
}
}
}
else
{
lean_object* v_a_2044_; lean_object* v___x_2046_; uint8_t v_isShared_2047_; uint8_t v_isSharedCheck_2051_; 
lean_dec_ref(v___y_1994_);
lean_dec_ref(v_iM_1910_);
lean_dec_ref(v_y_1844_);
lean_dec_ref(v_iM_1843_);
lean_dec_ref(v_M_1842_);
lean_dec(v_v_1841_);
v_a_2044_ = lean_ctor_get(v___x_1996_, 0);
v_isSharedCheck_2051_ = !lean_is_exclusive(v___x_1996_);
if (v_isSharedCheck_2051_ == 0)
{
v___x_2046_ = v___x_1996_;
v_isShared_2047_ = v_isSharedCheck_2051_;
goto v_resetjp_2045_;
}
else
{
lean_inc(v_a_2044_);
lean_dec(v___x_1996_);
v___x_2046_ = lean_box(0);
v_isShared_2047_ = v_isSharedCheck_2051_;
goto v_resetjp_2045_;
}
v_resetjp_2045_:
{
lean_object* v___x_2049_; 
if (v_isShared_2047_ == 0)
{
v___x_2049_ = v___x_2046_;
goto v_reusejp_2048_;
}
else
{
lean_object* v_reuseFailAlloc_2050_; 
v_reuseFailAlloc_2050_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2050_, 0, v_a_2044_);
v___x_2049_ = v_reuseFailAlloc_2050_;
goto v_reusejp_2048_;
}
v_reusejp_2048_:
{
return v___x_2049_;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow___boxed(lean_object* v_v_2063_, lean_object* v_M_2064_, lean_object* v_iM_2065_, lean_object* v_y_2066_, lean_object* v_g_2067_, lean_object* v_s_2068_, lean_object* v_a_2069_, lean_object* v_a_2070_, lean_object* v_a_2071_, lean_object* v_a_2072_, lean_object* v_a_2073_){
_start:
{
lean_object* v_res_2074_; 
v_res_2074_ = lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_zpow(v_v_2063_, v_M_2064_, v_iM_2065_, v_y_2066_, v_g_2067_, v_s_2068_, v_a_2069_, v_a_2070_, v_a_2071_, v_a_2072_);
lean_dec(v_a_2072_);
lean_dec_ref(v_a_2071_);
lean_dec(v_a_2070_);
lean_dec_ref(v_a_2069_);
lean_dec(v_s_2068_);
return v_res_2074_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_congr(lean_object* v_v_2078_, lean_object* v_M_2079_, lean_object* v_y_2080_, lean_object* v_y_x27_2081_, lean_object* v_g_2082_, lean_object* v_pf_2083_){
_start:
{
if (lean_obj_tag(v_g_2082_) == 0)
{
lean_dec_ref(v_y_x27_2081_);
lean_dec_ref(v_y_2080_);
lean_dec_ref(v_M_2079_);
lean_dec(v_v_2078_);
return v_pf_2083_;
}
else
{
lean_object* v_iM_2084_; lean_object* v___x_2085_; lean_object* v___x_2086_; lean_object* v___x_2087_; lean_object* v___x_2088_; lean_object* v___x_2089_; lean_object* v___x_2090_; lean_object* v___x_2091_; lean_object* v___x_2092_; lean_object* v___x_2093_; lean_object* v___x_2094_; lean_object* v___x_2095_; lean_object* v___x_2096_; lean_object* v___x_2097_; lean_object* v___x_2098_; lean_object* v___x_2099_; lean_object* v___x_2100_; lean_object* v___x_2101_; lean_object* v___x_2102_; lean_object* v___x_2103_; lean_object* v___x_2104_; lean_object* v___x_2105_; lean_object* v___x_2106_; lean_object* v___x_2107_; lean_object* v___x_2108_; lean_object* v___x_2109_; lean_object* v___x_2110_; lean_object* v___x_2111_; lean_object* v___x_2112_; lean_object* v___x_2113_; lean_object* v___x_2114_; lean_object* v___x_2115_; lean_object* v___x_2116_; lean_object* v___x_2117_; lean_object* v___x_2118_; lean_object* v___x_2119_; lean_object* v___x_2120_; lean_object* v___x_2121_; lean_object* v___x_2122_; lean_object* v___x_2123_; lean_object* v___x_2124_; lean_object* v___x_2125_; lean_object* v___x_2126_; lean_object* v___x_2127_; lean_object* v___x_2128_; lean_object* v___x_2129_; lean_object* v___x_2130_; lean_object* v___x_2131_; lean_object* v___x_2132_; lean_object* v___x_2133_; 
v_iM_2084_ = lean_ctor_get(v_g_2082_, 0);
lean_inc_ref(v_iM_2084_);
lean_dec_ref_known(v_g_2082_, 1);
lean_inc(v_v_2078_);
v___x_2085_ = l_Lean_Level_succ___override(v_v_2078_);
v___x_2086_ = lean_box(0);
lean_inc(v___x_2085_);
v___x_2087_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2087_, 0, v___x_2085_);
lean_ctor_set(v___x_2087_, 1, v___x_2086_);
v___x_2088_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_congr___closed__1));
v___x_2089_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2089_, 0, v___x_2085_);
lean_ctor_set(v___x_2089_, 1, v___x_2087_);
v___x_2090_ = l_Lean_Expr_const___override(v___x_2088_, v___x_2089_);
lean_inc_ref_n(v_M_2079_, 10);
v___x_2091_ = l_Lean_Expr_app___override(v___x_2090_, v_M_2079_);
v___x_2092_ = l_Lean_Expr_app___override(v___x_2091_, v_M_2079_);
v___x_2093_ = l_Lean_Expr_app___override(v___x_2092_, v_y_2080_);
v___x_2094_ = l_Lean_Expr_app___override(v___x_2093_, v_y_x27_2081_);
v___x_2095_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__2));
v___x_2096_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2096_, 0, v_v_2078_);
lean_ctor_set(v___x_2096_, 1, v___x_2086_);
lean_inc_ref_n(v___x_2096_, 8);
v___x_2097_ = l_Lean_Expr_const___override(v___x_2095_, v___x_2096_);
v___x_2098_ = l_Lean_Expr_app___override(v___x_2097_, v_M_2079_);
v___x_2099_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__5));
v___x_2100_ = l_Lean_Expr_const___override(v___x_2099_, v___x_2096_);
v___x_2101_ = l_Lean_Expr_app___override(v___x_2100_, v_M_2079_);
v___x_2102_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__8));
v___x_2103_ = l_Lean_Expr_const___override(v___x_2102_, v___x_2096_);
v___x_2104_ = l_Lean_Expr_app___override(v___x_2103_, v_M_2079_);
v___x_2105_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__11));
v___x_2106_ = l_Lean_Expr_const___override(v___x_2105_, v___x_2096_);
v___x_2107_ = l_Lean_Expr_app___override(v___x_2106_, v_M_2079_);
v___x_2108_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__14));
v___x_2109_ = l_Lean_Expr_const___override(v___x_2108_, v___x_2096_);
v___x_2110_ = l_Lean_Expr_app___override(v___x_2109_, v_M_2079_);
v___x_2111_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__17));
v___x_2112_ = l_Lean_Expr_const___override(v___x_2111_, v___x_2096_);
v___x_2113_ = l_Lean_Expr_app___override(v___x_2112_, v_M_2079_);
v___x_2114_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__20));
v___x_2115_ = l_Lean_Expr_const___override(v___x_2114_, v___x_2096_);
v___x_2116_ = l_Lean_Expr_app___override(v___x_2115_, v_M_2079_);
v___x_2117_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__23));
v___x_2118_ = l_Lean_Expr_const___override(v___x_2117_, v___x_2096_);
v___x_2119_ = l_Lean_Expr_app___override(v___x_2118_, v_M_2079_);
v___x_2120_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__26));
v___x_2121_ = l_Lean_Expr_const___override(v___x_2120_, v___x_2096_);
v___x_2122_ = l_Lean_Expr_app___override(v___x_2121_, v_M_2079_);
v___x_2123_ = l_Lean_Expr_app___override(v___x_2122_, v_iM_2084_);
v___x_2124_ = l_Lean_Expr_app___override(v___x_2119_, v___x_2123_);
v___x_2125_ = l_Lean_Expr_app___override(v___x_2116_, v___x_2124_);
v___x_2126_ = l_Lean_Expr_app___override(v___x_2113_, v___x_2125_);
v___x_2127_ = l_Lean_Expr_app___override(v___x_2110_, v___x_2126_);
v___x_2128_ = l_Lean_Expr_app___override(v___x_2107_, v___x_2127_);
v___x_2129_ = l_Lean_Expr_app___override(v___x_2104_, v___x_2128_);
v___x_2130_ = l_Lean_Expr_app___override(v___x_2101_, v___x_2129_);
v___x_2131_ = l_Lean_Expr_app___override(v___x_2098_, v___x_2130_);
v___x_2132_ = l_Lean_Expr_app___override(v___x_2094_, v___x_2131_);
v___x_2133_ = l_Lean_Expr_app___override(v___x_2132_, v_pf_2083_);
return v___x_2133_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mkEqMul___redArg___lam__0(lean_object* v___x_2134_, lean_object* v_M_2135_, lean_object* v_iM_2136_, lean_object* v_a_2137_){
_start:
{
lean_object* v___x_2138_; lean_object* v___x_2139_; lean_object* v___x_2140_; lean_object* v___x_2141_; lean_object* v___x_2142_; lean_object* v___x_2143_; lean_object* v___x_2144_; lean_object* v___x_2145_; lean_object* v___x_2146_; lean_object* v___x_2147_; lean_object* v___x_2148_; lean_object* v___x_2149_; lean_object* v___x_2150_; lean_object* v___x_2151_; lean_object* v___x_2152_; lean_object* v___x_2153_; lean_object* v___x_2154_; lean_object* v___x_2155_; lean_object* v___x_2156_; lean_object* v___x_2157_; lean_object* v___x_2158_; lean_object* v___x_2159_; lean_object* v___x_2160_; lean_object* v___x_2161_; lean_object* v___x_2162_; lean_object* v___x_2163_; lean_object* v___x_2164_; lean_object* v___x_2165_; lean_object* v___x_2166_; lean_object* v___x_2167_; lean_object* v___x_2168_; lean_object* v___x_2169_; lean_object* v___x_2170_; lean_object* v___x_2171_; lean_object* v___x_2172_; lean_object* v___x_2173_; lean_object* v___x_2174_; 
v___x_2138_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__2));
lean_inc_n(v___x_2134_, 8);
v___x_2139_ = l_Lean_Expr_const___override(v___x_2138_, v___x_2134_);
lean_inc_ref_n(v_M_2135_, 8);
v___x_2140_ = l_Lean_Expr_app___override(v___x_2139_, v_M_2135_);
v___x_2141_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__5));
v___x_2142_ = l_Lean_Expr_const___override(v___x_2141_, v___x_2134_);
v___x_2143_ = l_Lean_Expr_app___override(v___x_2142_, v_M_2135_);
v___x_2144_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__8));
v___x_2145_ = l_Lean_Expr_const___override(v___x_2144_, v___x_2134_);
v___x_2146_ = l_Lean_Expr_app___override(v___x_2145_, v_M_2135_);
v___x_2147_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__11));
v___x_2148_ = l_Lean_Expr_const___override(v___x_2147_, v___x_2134_);
v___x_2149_ = l_Lean_Expr_app___override(v___x_2148_, v_M_2135_);
v___x_2150_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__14));
v___x_2151_ = l_Lean_Expr_const___override(v___x_2150_, v___x_2134_);
v___x_2152_ = l_Lean_Expr_app___override(v___x_2151_, v_M_2135_);
v___x_2153_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__17));
v___x_2154_ = l_Lean_Expr_const___override(v___x_2153_, v___x_2134_);
v___x_2155_ = l_Lean_Expr_app___override(v___x_2154_, v_M_2135_);
v___x_2156_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__20));
v___x_2157_ = l_Lean_Expr_const___override(v___x_2156_, v___x_2134_);
v___x_2158_ = l_Lean_Expr_app___override(v___x_2157_, v_M_2135_);
v___x_2159_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__23));
v___x_2160_ = l_Lean_Expr_const___override(v___x_2159_, v___x_2134_);
v___x_2161_ = l_Lean_Expr_app___override(v___x_2160_, v_M_2135_);
v___x_2162_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_expr___closed__26));
v___x_2163_ = l_Lean_Expr_const___override(v___x_2162_, v___x_2134_);
v___x_2164_ = l_Lean_Expr_app___override(v___x_2163_, v_M_2135_);
v___x_2165_ = l_Lean_Expr_app___override(v___x_2164_, v_iM_2136_);
v___x_2166_ = l_Lean_Expr_app___override(v___x_2161_, v___x_2165_);
v___x_2167_ = l_Lean_Expr_app___override(v___x_2158_, v___x_2166_);
v___x_2168_ = l_Lean_Expr_app___override(v___x_2155_, v___x_2167_);
v___x_2169_ = l_Lean_Expr_app___override(v___x_2152_, v___x_2168_);
v___x_2170_ = l_Lean_Expr_app___override(v___x_2149_, v___x_2169_);
v___x_2171_ = l_Lean_Expr_app___override(v___x_2146_, v___x_2170_);
v___x_2172_ = l_Lean_Expr_app___override(v___x_2143_, v___x_2171_);
v___x_2173_ = l_Lean_Expr_app___override(v___x_2140_, v___x_2172_);
v___x_2174_ = l_Lean_Expr_app___override(v___x_2173_, v_a_2137_);
return v___x_2174_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mkEqMul___redArg(lean_object* v_v_2181_, lean_object* v_M_2182_, lean_object* v_iM_2183_, lean_object* v_a_2184_, lean_object* v_b_2185_, lean_object* v_C_2186_, lean_object* v_d_2187_, lean_object* v_e_2188_, lean_object* v_g_2189_, lean_object* v_pf_u2081_2190_, lean_object* v_pf_u2082_2191_, lean_object* v_pf_u2083_2192_){
_start:
{
lean_object* v___x_2194_; lean_object* v___x_2195_; lean_object* v___x_2196_; lean_object* v___x_2197_; lean_object* v___x_2198_; lean_object* v___x_2199_; lean_object* v___x_2200_; lean_object* v___x_2201_; lean_object* v___x_2202_; lean_object* v___x_2203_; lean_object* v___x_2204_; lean_object* v___x_2205_; lean_object* v___x_2206_; lean_object* v___x_2207_; lean_object* v___x_2208_; lean_object* v___x_2209_; lean_object* v___x_2210_; lean_object* v___x_2211_; lean_object* v___x_2212_; lean_object* v___x_2213_; lean_object* v___x_2214_; lean_object* v___x_2215_; lean_object* v___x_2216_; lean_object* v___x_2217_; lean_object* v___x_2218_; lean_object* v___x_2219_; lean_object* v___x_2220_; lean_object* v___x_2221_; lean_object* v___x_2222_; lean_object* v___x_2223_; lean_object* v___x_2224_; lean_object* v___x_2225_; lean_object* v___x_2226_; lean_object* v___x_2227_; lean_object* v___x_2228_; lean_object* v___x_2229_; lean_object* v_pf_u2082_x27_2230_; lean_object* v___x_2231_; lean_object* v_a_2232_; lean_object* v___x_2234_; uint8_t v_isShared_2235_; uint8_t v_isSharedCheck_2273_; 
v___x_2194_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__2));
v___x_2195_ = lean_box(0);
lean_inc_n(v_v_2181_, 5);
v___x_2196_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2196_, 0, v_v_2181_);
lean_ctor_set(v___x_2196_, 1, v___x_2195_);
lean_inc_ref_n(v___x_2196_, 7);
v___x_2197_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2197_, 0, v_v_2181_);
lean_ctor_set(v___x_2197_, 1, v___x_2196_);
v___x_2198_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2198_, 0, v_v_2181_);
lean_ctor_set(v___x_2198_, 1, v___x_2197_);
v___x_2199_ = l_Lean_Expr_const___override(v___x_2194_, v___x_2198_);
lean_inc_ref_n(v_M_2182_, 11);
v___x_2200_ = l_Lean_Expr_app___override(v___x_2199_, v_M_2182_);
v___x_2201_ = l_Lean_Expr_app___override(v___x_2200_, v_M_2182_);
v___x_2202_ = l_Lean_Expr_app___override(v___x_2201_, v_M_2182_);
v___x_2203_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__4));
v___x_2204_ = l_Lean_Expr_const___override(v___x_2203_, v___x_2196_);
v___x_2205_ = l_Lean_Expr_app___override(v___x_2204_, v_M_2182_);
v___x_2206_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__7));
v___x_2207_ = l_Lean_Expr_const___override(v___x_2206_, v___x_2196_);
v___x_2208_ = l_Lean_Expr_app___override(v___x_2207_, v_M_2182_);
v___x_2209_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__10));
v___x_2210_ = l_Lean_Expr_const___override(v___x_2209_, v___x_2196_);
v___x_2211_ = l_Lean_Expr_app___override(v___x_2210_, v_M_2182_);
v___x_2212_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__13));
v___x_2213_ = l_Lean_Expr_const___override(v___x_2212_, v___x_2196_);
v___x_2214_ = l_Lean_Expr_app___override(v___x_2213_, v_M_2182_);
v___x_2215_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__16));
v___x_2216_ = l_Lean_Expr_const___override(v___x_2215_, v___x_2196_);
v___x_2217_ = l_Lean_Expr_app___override(v___x_2216_, v_M_2182_);
v___x_2218_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg___closed__19));
v___x_2219_ = l_Lean_Expr_const___override(v___x_2218_, v___x_2196_);
v___x_2220_ = l_Lean_Expr_app___override(v___x_2219_, v_M_2182_);
lean_inc_ref(v_iM_2183_);
v___x_2221_ = l_Lean_Expr_app___override(v___x_2220_, v_iM_2183_);
v___x_2222_ = l_Lean_Expr_app___override(v___x_2217_, v___x_2221_);
v___x_2223_ = l_Lean_Expr_app___override(v___x_2214_, v___x_2222_);
v___x_2224_ = l_Lean_Expr_app___override(v___x_2211_, v___x_2223_);
v___x_2225_ = l_Lean_Expr_app___override(v___x_2208_, v___x_2224_);
lean_inc_ref(v___x_2225_);
v___x_2226_ = l_Lean_Expr_app___override(v___x_2205_, v___x_2225_);
v___x_2227_ = l_Lean_Expr_app___override(v___x_2202_, v___x_2226_);
lean_inc_ref_n(v_C_2186_, 2);
v___x_2228_ = l_Lean_Expr_app___override(v___x_2227_, v_C_2186_);
lean_inc_ref_n(v_d_2187_, 2);
v___x_2229_ = l_Lean_Expr_app___override(v___x_2228_, v_d_2187_);
lean_inc_n(v_g_2189_, 2);
lean_inc_ref(v___x_2229_);
lean_inc_ref(v_b_2185_);
v_pf_u2082_x27_2230_ = lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_congr(v_v_2181_, v_M_2182_, v_b_2185_, v___x_2229_, v_g_2189_, v_pf_u2082_2191_);
v___x_2231_ = lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mulRight___redArg(v_v_2181_, v_M_2182_, v_iM_2183_, v_C_2186_, v_d_2187_, v_g_2189_);
v_a_2232_ = lean_ctor_get(v___x_2231_, 0);
v_isSharedCheck_2273_ = !lean_is_exclusive(v___x_2231_);
if (v_isSharedCheck_2273_ == 0)
{
v___x_2234_ = v___x_2231_;
v_isShared_2235_ = v_isSharedCheck_2273_;
goto v_resetjp_2233_;
}
else
{
lean_inc(v_a_2232_);
lean_dec(v___x_2231_);
v___x_2234_ = lean_box(0);
v_isShared_2235_ = v_isSharedCheck_2273_;
goto v_resetjp_2233_;
}
v_resetjp_2233_:
{
lean_object* v___x_2236_; lean_object* v___y_2238_; lean_object* v___y_2239_; lean_object* v___y_2249_; lean_object* v___y_2250_; lean_object* v___y_2255_; lean_object* v___y_2256_; lean_object* v___x_2261_; lean_object* v___x_2262_; lean_object* v___x_2263_; lean_object* v___x_2264_; lean_object* v___x_2265_; lean_object* v___y_2267_; 
lean_inc(v_g_2189_);
lean_inc_ref(v_e_2188_);
lean_inc_ref(v_d_2187_);
lean_inc_ref_n(v_M_2182_, 2);
v___x_2236_ = lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_congr(v_v_2181_, v_M_2182_, v_d_2187_, v_e_2188_, v_g_2189_, v_pf_u2083_2192_);
v___x_2261_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mkEqMul___redArg___closed__1));
lean_inc_ref(v___x_2196_);
v___x_2262_ = l_Lean_Expr_const___override(v___x_2261_, v___x_2196_);
v___x_2263_ = l_Lean_Expr_app___override(v___x_2262_, v_M_2182_);
v___x_2264_ = l_Lean_Expr_app___override(v___x_2263_, v___x_2225_);
v___x_2265_ = l_Lean_Expr_app___override(v___x_2264_, v_a_2184_);
if (lean_obj_tag(v_g_2189_) == 0)
{
v___y_2267_ = v_b_2185_;
goto v___jp_2266_;
}
else
{
lean_object* v_iM_2271_; lean_object* v___x_2272_; 
v_iM_2271_ = lean_ctor_get(v_g_2189_, 0);
lean_inc_ref(v_iM_2271_);
lean_inc_ref(v_M_2182_);
lean_inc_ref(v___x_2196_);
v___x_2272_ = lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mkEqMul___redArg___lam__0(v___x_2196_, v_M_2182_, v_iM_2271_, v_b_2185_);
v___y_2267_ = v___x_2272_;
goto v___jp_2266_;
}
v___jp_2237_:
{
lean_object* v___x_2240_; lean_object* v___x_2241_; lean_object* v___x_2242_; lean_object* v___x_2243_; lean_object* v___x_2244_; lean_object* v___x_2246_; 
v___x_2240_ = l_Lean_Expr_app___override(v___y_2238_, v___y_2239_);
v___x_2241_ = l_Lean_Expr_app___override(v___x_2240_, v_pf_u2081_2190_);
v___x_2242_ = l_Lean_Expr_app___override(v___x_2241_, v_pf_u2082_x27_2230_);
v___x_2243_ = l_Lean_Expr_app___override(v___x_2242_, v_a_2232_);
v___x_2244_ = l_Lean_Expr_app___override(v___x_2243_, v___x_2236_);
if (v_isShared_2235_ == 0)
{
lean_ctor_set(v___x_2234_, 0, v___x_2244_);
v___x_2246_ = v___x_2234_;
goto v_reusejp_2245_;
}
else
{
lean_object* v_reuseFailAlloc_2247_; 
v_reuseFailAlloc_2247_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2247_, 0, v___x_2244_);
v___x_2246_ = v_reuseFailAlloc_2247_;
goto v_reusejp_2245_;
}
v_reusejp_2245_:
{
return v___x_2246_;
}
}
v___jp_2248_:
{
lean_object* v___x_2251_; 
v___x_2251_ = l_Lean_Expr_app___override(v___y_2249_, v___y_2250_);
if (lean_obj_tag(v_g_2189_) == 0)
{
lean_dec_ref_known(v___x_2196_, 2);
lean_dec_ref(v_M_2182_);
v___y_2238_ = v___x_2251_;
v___y_2239_ = v_e_2188_;
goto v___jp_2237_;
}
else
{
lean_object* v_iM_2252_; lean_object* v___x_2253_; 
v_iM_2252_ = lean_ctor_get(v_g_2189_, 0);
lean_inc_ref(v_iM_2252_);
lean_dec_ref_known(v_g_2189_, 1);
v___x_2253_ = lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mkEqMul___redArg___lam__0(v___x_2196_, v_M_2182_, v_iM_2252_, v_e_2188_);
v___y_2238_ = v___x_2251_;
v___y_2239_ = v___x_2253_;
goto v___jp_2237_;
}
}
v___jp_2254_:
{
lean_object* v___x_2257_; lean_object* v___x_2258_; 
v___x_2257_ = l_Lean_Expr_app___override(v___y_2255_, v___y_2256_);
v___x_2258_ = l_Lean_Expr_app___override(v___x_2257_, v_C_2186_);
if (lean_obj_tag(v_g_2189_) == 0)
{
v___y_2249_ = v___x_2258_;
v___y_2250_ = v_d_2187_;
goto v___jp_2248_;
}
else
{
lean_object* v_iM_2259_; lean_object* v___x_2260_; 
v_iM_2259_ = lean_ctor_get(v_g_2189_, 0);
lean_inc_ref(v_iM_2259_);
lean_inc_ref(v_M_2182_);
lean_inc_ref(v___x_2196_);
v___x_2260_ = lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mkEqMul___redArg___lam__0(v___x_2196_, v_M_2182_, v_iM_2259_, v_d_2187_);
v___y_2249_ = v___x_2258_;
v___y_2250_ = v___x_2260_;
goto v___jp_2248_;
}
}
v___jp_2266_:
{
lean_object* v___x_2268_; 
v___x_2268_ = l_Lean_Expr_app___override(v___x_2265_, v___y_2267_);
if (lean_obj_tag(v_g_2189_) == 0)
{
v___y_2255_ = v___x_2268_;
v___y_2256_ = v___x_2229_;
goto v___jp_2254_;
}
else
{
lean_object* v_iM_2269_; lean_object* v___x_2270_; 
v_iM_2269_ = lean_ctor_get(v_g_2189_, 0);
lean_inc_ref(v_iM_2269_);
lean_inc_ref(v_M_2182_);
lean_inc_ref(v___x_2196_);
v___x_2270_ = lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mkEqMul___redArg___lam__0(v___x_2196_, v_M_2182_, v_iM_2269_, v___x_2229_);
v___y_2255_ = v___x_2268_;
v___y_2256_ = v___x_2270_;
goto v___jp_2254_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mkEqMul___redArg___boxed(lean_object* v_v_2274_, lean_object* v_M_2275_, lean_object* v_iM_2276_, lean_object* v_a_2277_, lean_object* v_b_2278_, lean_object* v_C_2279_, lean_object* v_d_2280_, lean_object* v_e_2281_, lean_object* v_g_2282_, lean_object* v_pf_u2081_2283_, lean_object* v_pf_u2082_2284_, lean_object* v_pf_u2083_2285_, lean_object* v_a_2286_){
_start:
{
lean_object* v_res_2287_; 
v_res_2287_ = lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mkEqMul___redArg(v_v_2274_, v_M_2275_, v_iM_2276_, v_a_2277_, v_b_2278_, v_C_2279_, v_d_2280_, v_e_2281_, v_g_2282_, v_pf_u2081_2283_, v_pf_u2082_2284_, v_pf_u2083_2285_);
return v_res_2287_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mkEqMul(lean_object* v_v_2288_, lean_object* v_M_2289_, lean_object* v_iM_2290_, lean_object* v_a_2291_, lean_object* v_b_2292_, lean_object* v_C_2293_, lean_object* v_d_2294_, lean_object* v_e_2295_, lean_object* v_g_2296_, lean_object* v_pf_u2081_2297_, lean_object* v_pf_u2082_2298_, lean_object* v_pf_u2083_2299_, lean_object* v_a_2300_, lean_object* v_a_2301_, lean_object* v_a_2302_, lean_object* v_a_2303_){
_start:
{
lean_object* v___x_2305_; 
v___x_2305_ = lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mkEqMul___redArg(v_v_2288_, v_M_2289_, v_iM_2290_, v_a_2291_, v_b_2292_, v_C_2293_, v_d_2294_, v_e_2295_, v_g_2296_, v_pf_u2081_2297_, v_pf_u2082_2298_, v_pf_u2083_2299_);
return v___x_2305_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mkEqMul___boxed(lean_object** _args){
lean_object* v_v_2306_ = _args[0];
lean_object* v_M_2307_ = _args[1];
lean_object* v_iM_2308_ = _args[2];
lean_object* v_a_2309_ = _args[3];
lean_object* v_b_2310_ = _args[4];
lean_object* v_C_2311_ = _args[5];
lean_object* v_d_2312_ = _args[6];
lean_object* v_e_2313_ = _args[7];
lean_object* v_g_2314_ = _args[8];
lean_object* v_pf_u2081_2315_ = _args[9];
lean_object* v_pf_u2082_2316_ = _args[10];
lean_object* v_pf_u2083_2317_ = _args[11];
lean_object* v_a_2318_ = _args[12];
lean_object* v_a_2319_ = _args[13];
lean_object* v_a_2320_ = _args[14];
lean_object* v_a_2321_ = _args[15];
lean_object* v_a_2322_ = _args[16];
_start:
{
lean_object* v_res_2323_; 
v_res_2323_ = lp_mathlib_Mathlib_Tactic_FieldSimp_Sign_mkEqMul(v_v_2306_, v_M_2307_, v_iM_2308_, v_a_2309_, v_b_2310_, v_C_2311_, v_d_2312_, v_e_2313_, v_g_2314_, v_pf_u2081_2315_, v_pf_u2082_2316_, v_pf_u2083_2317_, v_a_2318_, v_a_2319_, v_a_2320_, v_a_2321_);
lean_dec(v_a_2321_);
lean_dec_ref(v_a_2320_);
lean_dec(v_a_2319_);
lean_dec_ref(v_a_2318_);
return v_res_2323_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Group_List_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Field_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Int_Parity(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_FieldSimp_Lemmas(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Group_List_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Field_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Int_Parity(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Util_Qq(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_FieldSimp_Lemmas(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Util_Qq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Group_List_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Field_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Int_Parity(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Util_Qq(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_FieldSimp_Lemmas(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_Group_List_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Field_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Int_Parity(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Util_Qq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_FieldSimp_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_FieldSimp_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_FieldSimp_Lemmas(builtin);
}
#ifdef __cplusplus
}
#endif
