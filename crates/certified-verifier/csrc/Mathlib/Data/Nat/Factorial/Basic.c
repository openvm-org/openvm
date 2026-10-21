// Lean compiler output
// Module: Mathlib.Data.Nat.Factorial.Basic
// Imports: public import Init public meta import Init public import Mathlib.Data.Nat.Basic public import Mathlib.Tactic.Common public import Mathlib.Tactic.CrossRefAttribute public import Mathlib.Tactic.Monotonicity.Attr public import Mathlib.Tactic.Attr.Core
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
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_nat_shiftr(lean_object*, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_factorial(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_factorial___boxed(lean_object*);
static const lean_string_object lp_mathlib_Nat_term___x21___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Nat"};
static const lean_object* lp_mathlib_Nat_term___x21___closed__0 = (const lean_object*)&lp_mathlib_Nat_term___x21___closed__0_value;
static const lean_string_object lp_mathlib_Nat_term___x21___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "term_!"};
static const lean_object* lp_mathlib_Nat_term___x21___closed__1 = (const lean_object*)&lp_mathlib_Nat_term___x21___closed__1_value;
static const lean_ctor_object lp_mathlib_Nat_term___x21___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Nat_term___x21___closed__0_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_ctor_object lp_mathlib_Nat_term___x21___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Nat_term___x21___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Nat_term___x21___closed__1_value),LEAN_SCALAR_PTR_LITERAL(59, 131, 157, 247, 80, 116, 106, 22)}};
static const lean_object* lp_mathlib_Nat_term___x21___closed__2 = (const lean_object*)&lp_mathlib_Nat_term___x21___closed__2_value;
static const lean_string_object lp_mathlib_Nat_term___x21___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "!"};
static const lean_object* lp_mathlib_Nat_term___x21___closed__3 = (const lean_object*)&lp_mathlib_Nat_term___x21___closed__3_value;
static const lean_ctor_object lp_mathlib_Nat_term___x21___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Nat_term___x21___closed__3_value)}};
static const lean_object* lp_mathlib_Nat_term___x21___closed__4 = (const lean_object*)&lp_mathlib_Nat_term___x21___closed__4_value;
static const lean_ctor_object lp_mathlib_Nat_term___x21___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_Nat_term___x21___closed__2_value),((lean_object*)(((size_t)(10000) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Nat_term___x21___closed__4_value)}};
static const lean_object* lp_mathlib_Nat_term___x21___closed__5 = (const lean_object*)&lp_mathlib_Nat_term___x21___closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_Nat_term___x21 = (const lean_object*)&lp_mathlib_Nat_term___x21___closed__5_value;
static const lean_string_object lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__0 = (const lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__0_value;
static const lean_string_object lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__1 = (const lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__1_value;
static const lean_string_object lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__2 = (const lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__2_value;
static const lean_string_object lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__3 = (const lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__3_value;
static const lean_ctor_object lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__4 = (const lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__4_value;
static const lean_string_object lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "Nat.factorial"};
static const lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__5 = (const lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__5_value;
static lean_once_cell_t lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__6;
static const lean_string_object lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "factorial"};
static const lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__7 = (const lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__7_value;
static const lean_ctor_object lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Nat_term___x21___closed__0_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_ctor_object lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(144, 220, 252, 235, 45, 151, 248, 148)}};
static const lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__8 = (const lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__8_value;
static const lean_ctor_object lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__9 = (const lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__9_value;
static const lean_ctor_object lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__8_value)}};
static const lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__10 = (const lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__10_value;
static const lean_ctor_object lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__10_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__11 = (const lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__11_value;
static const lean_ctor_object lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__9_value),((lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__11_value)}};
static const lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__12 = (const lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__12_value;
static const lean_string_object lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__13 = (const lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__13_value;
static const lean_ctor_object lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__13_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__14 = (const lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__14_value;
LEAN_EXPORT lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______unexpand__Nat__factorial__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______unexpand__Nat__factorial__1___closed__0 = (const lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______unexpand__Nat__factorial__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______unexpand__Nat__factorial__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______unexpand__Nat__factorial__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______unexpand__Nat__factorial__1___closed__1 = (const lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______unexpand__Nat__factorial__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______unexpand__Nat__factorial__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______unexpand__Nat__factorial__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_ascFactorial(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_ascFactorial___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Factorial_Basic_0__Nat_factorial_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Factorial_Basic_0__Nat_factorial_match__1_splitter___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Factorial_Basic_0__Nat_factorial_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Factorial_Basic_0__Nat_factorial_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_descFactorial(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_descFactorial___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_ascFactorialBinary(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_ascFactorialBinary___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Factorial_Basic_0__Nat_ascFactorialBinary_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Factorial_Basic_0__Nat_ascFactorialBinary_match__1_splitter___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Factorial_Basic_0__Nat_ascFactorialBinary_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Factorial_Basic_0__Nat_ascFactorialBinary_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_factorialBinarySplitting(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_factorialBinarySplitting___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_descFactorialBinary(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_descFactorialBinary___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_factorial(lean_object* v_x_1_){
_start:
{
lean_object* v_zero_2_; uint8_t v_isZero_3_; 
v_zero_2_ = lean_unsigned_to_nat(0u);
v_isZero_3_ = lean_nat_dec_eq(v_x_1_, v_zero_2_);
if (v_isZero_3_ == 1)
{
lean_object* v___x_4_; 
v___x_4_ = lean_unsigned_to_nat(1u);
return v___x_4_;
}
else
{
lean_object* v_one_5_; lean_object* v_n_6_; lean_object* v___x_7_; lean_object* v___x_8_; 
v_one_5_ = lean_unsigned_to_nat(1u);
v_n_6_ = lean_nat_sub(v_x_1_, v_one_5_);
v___x_7_ = lp_mathlib_Nat_factorial(v_n_6_);
lean_dec(v_n_6_);
v___x_8_ = lean_nat_mul(v_x_1_, v___x_7_);
lean_dec(v___x_7_);
return v___x_8_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_factorial___boxed(lean_object* v_x_9_){
_start:
{
lean_object* v_res_10_; 
v_res_10_ = lp_mathlib_Nat_factorial(v_x_9_);
lean_dec(v_x_9_);
return v_res_10_;
}
}
static lean_object* _init_lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__6(void){
_start:
{
lean_object* v___x_35_; lean_object* v___x_36_; 
v___x_35_ = ((lean_object*)(lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__5));
v___x_36_ = l_String_toRawSubstring_x27(v___x_35_);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1(lean_object* v_x_55_, lean_object* v_a_56_, lean_object* v_a_57_){
_start:
{
lean_object* v___x_58_; uint8_t v___x_59_; 
v___x_58_ = ((lean_object*)(lp_mathlib_Nat_term___x21___closed__2));
lean_inc(v_x_55_);
v___x_59_ = l_Lean_Syntax_isOfKind(v_x_55_, v___x_58_);
if (v___x_59_ == 0)
{
lean_object* v___x_60_; lean_object* v___x_61_; 
lean_dec(v_x_55_);
v___x_60_ = lean_box(1);
v___x_61_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_61_, 0, v___x_60_);
lean_ctor_set(v___x_61_, 1, v_a_57_);
return v___x_61_;
}
else
{
lean_object* v_quotContext_62_; lean_object* v_currMacroScope_63_; lean_object* v_ref_64_; lean_object* v___x_65_; lean_object* v___x_66_; uint8_t v___x_67_; lean_object* v___x_68_; lean_object* v___x_69_; lean_object* v___x_70_; lean_object* v___x_71_; lean_object* v___x_72_; lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v___x_75_; lean_object* v___x_76_; lean_object* v___x_77_; lean_object* v___x_78_; 
v_quotContext_62_ = lean_ctor_get(v_a_56_, 1);
v_currMacroScope_63_ = lean_ctor_get(v_a_56_, 2);
v_ref_64_ = lean_ctor_get(v_a_56_, 5);
v___x_65_ = lean_unsigned_to_nat(0u);
v___x_66_ = l_Lean_Syntax_getArg(v_x_55_, v___x_65_);
lean_dec(v_x_55_);
v___x_67_ = 0;
v___x_68_ = l_Lean_SourceInfo_fromRef(v_ref_64_, v___x_67_);
v___x_69_ = ((lean_object*)(lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__4));
v___x_70_ = lean_obj_once(&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__6, &lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__6_once, _init_lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__6);
v___x_71_ = ((lean_object*)(lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__8));
lean_inc(v_currMacroScope_63_);
lean_inc(v_quotContext_62_);
v___x_72_ = l_Lean_addMacroScope(v_quotContext_62_, v___x_71_, v_currMacroScope_63_);
v___x_73_ = ((lean_object*)(lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__12));
lean_inc_n(v___x_68_, 2);
v___x_74_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_74_, 0, v___x_68_);
lean_ctor_set(v___x_74_, 1, v___x_70_);
lean_ctor_set(v___x_74_, 2, v___x_72_);
lean_ctor_set(v___x_74_, 3, v___x_73_);
v___x_75_ = ((lean_object*)(lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__14));
v___x_76_ = l_Lean_Syntax_node1(v___x_68_, v___x_75_, v___x_66_);
v___x_77_ = l_Lean_Syntax_node2(v___x_68_, v___x_69_, v___x_74_, v___x_76_);
v___x_78_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_78_, 0, v___x_77_);
lean_ctor_set(v___x_78_, 1, v_a_57_);
return v___x_78_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___boxed(lean_object* v_x_79_, lean_object* v_a_80_, lean_object* v_a_81_){
_start:
{
lean_object* v_res_82_; 
v_res_82_ = lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1(v_x_79_, v_a_80_, v_a_81_);
lean_dec_ref(v_a_80_);
return v_res_82_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______unexpand__Nat__factorial__1(lean_object* v_x_86_, lean_object* v_a_87_, lean_object* v_a_88_){
_start:
{
lean_object* v___x_89_; uint8_t v___x_90_; 
v___x_89_ = ((lean_object*)(lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______macroRules__Nat__term___x21__1___closed__4));
lean_inc(v_x_86_);
v___x_90_ = l_Lean_Syntax_isOfKind(v_x_86_, v___x_89_);
if (v___x_90_ == 0)
{
lean_object* v___x_91_; lean_object* v___x_92_; 
lean_dec(v_x_86_);
v___x_91_ = lean_box(0);
v___x_92_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_92_, 0, v___x_91_);
lean_ctor_set(v___x_92_, 1, v_a_88_);
return v___x_92_;
}
else
{
lean_object* v___x_93_; lean_object* v___x_94_; lean_object* v___x_95_; uint8_t v___x_96_; 
v___x_93_ = lean_unsigned_to_nat(0u);
v___x_94_ = l_Lean_Syntax_getArg(v_x_86_, v___x_93_);
v___x_95_ = ((lean_object*)(lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______unexpand__Nat__factorial__1___closed__1));
lean_inc(v___x_94_);
v___x_96_ = l_Lean_Syntax_isOfKind(v___x_94_, v___x_95_);
if (v___x_96_ == 0)
{
lean_object* v___x_97_; lean_object* v___x_98_; 
lean_dec(v___x_94_);
lean_dec(v_x_86_);
v___x_97_ = lean_box(0);
v___x_98_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_98_, 0, v___x_97_);
lean_ctor_set(v___x_98_, 1, v_a_88_);
return v___x_98_;
}
else
{
lean_object* v___x_99_; lean_object* v___x_100_; uint8_t v___x_101_; 
v___x_99_ = lean_unsigned_to_nat(1u);
v___x_100_ = l_Lean_Syntax_getArg(v_x_86_, v___x_99_);
lean_dec(v_x_86_);
lean_inc(v___x_100_);
v___x_101_ = l_Lean_Syntax_matchesNull(v___x_100_, v___x_99_);
if (v___x_101_ == 0)
{
lean_object* v___x_102_; lean_object* v___x_103_; 
lean_dec(v___x_100_);
lean_dec(v___x_94_);
v___x_102_ = lean_box(0);
v___x_103_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_103_, 0, v___x_102_);
lean_ctor_set(v___x_103_, 1, v_a_88_);
return v___x_103_;
}
else
{
lean_object* v___x_104_; lean_object* v_ref_105_; uint8_t v___x_106_; lean_object* v___x_107_; lean_object* v___x_108_; lean_object* v___x_109_; lean_object* v___x_110_; lean_object* v___x_111_; lean_object* v___x_112_; 
v___x_104_ = l_Lean_Syntax_getArg(v___x_100_, v___x_93_);
lean_dec(v___x_100_);
v_ref_105_ = l_Lean_replaceRef(v___x_94_, v_a_87_);
lean_dec(v___x_94_);
v___x_106_ = 0;
v___x_107_ = l_Lean_SourceInfo_fromRef(v_ref_105_, v___x_106_);
lean_dec(v_ref_105_);
v___x_108_ = ((lean_object*)(lp_mathlib_Nat_term___x21___closed__2));
v___x_109_ = ((lean_object*)(lp_mathlib_Nat_term___x21___closed__3));
lean_inc(v___x_107_);
v___x_110_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_110_, 0, v___x_107_);
lean_ctor_set(v___x_110_, 1, v___x_109_);
v___x_111_ = l_Lean_Syntax_node2(v___x_107_, v___x_108_, v___x_104_, v___x_110_);
v___x_112_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_112_, 0, v___x_111_);
lean_ctor_set(v___x_112_, 1, v_a_88_);
return v___x_112_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______unexpand__Nat__factorial__1___boxed(lean_object* v_x_113_, lean_object* v_a_114_, lean_object* v_a_115_){
_start:
{
lean_object* v_res_116_; 
v_res_116_ = lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorial__Basic______unexpand__Nat__factorial__1(v_x_113_, v_a_114_, v_a_115_);
lean_dec(v_a_114_);
return v_res_116_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_ascFactorial(lean_object* v_n_117_, lean_object* v_x_118_){
_start:
{
lean_object* v_zero_119_; uint8_t v_isZero_120_; 
v_zero_119_ = lean_unsigned_to_nat(0u);
v_isZero_120_ = lean_nat_dec_eq(v_x_118_, v_zero_119_);
if (v_isZero_120_ == 1)
{
lean_object* v___x_121_; 
v___x_121_ = lean_unsigned_to_nat(1u);
return v___x_121_;
}
else
{
lean_object* v_one_122_; lean_object* v_n_123_; lean_object* v___x_124_; lean_object* v___x_125_; lean_object* v___x_126_; 
v_one_122_ = lean_unsigned_to_nat(1u);
v_n_123_ = lean_nat_sub(v_x_118_, v_one_122_);
v___x_124_ = lean_nat_add(v_n_117_, v_n_123_);
v___x_125_ = lp_mathlib_Nat_ascFactorial(v_n_117_, v_n_123_);
lean_dec(v_n_123_);
v___x_126_ = lean_nat_mul(v___x_124_, v___x_125_);
lean_dec(v___x_125_);
lean_dec(v___x_124_);
return v___x_126_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_ascFactorial___boxed(lean_object* v_n_127_, lean_object* v_x_128_){
_start:
{
lean_object* v_res_129_; 
v_res_129_ = lp_mathlib_Nat_ascFactorial(v_n_127_, v_x_128_);
lean_dec(v_x_128_);
lean_dec(v_n_127_);
return v_res_129_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Factorial_Basic_0__Nat_factorial_match__1_splitter___redArg(lean_object* v_x_130_, lean_object* v_h__1_131_, lean_object* v_h__2_132_){
_start:
{
lean_object* v_zero_133_; uint8_t v_isZero_134_; 
v_zero_133_ = lean_unsigned_to_nat(0u);
v_isZero_134_ = lean_nat_dec_eq(v_x_130_, v_zero_133_);
if (v_isZero_134_ == 1)
{
lean_object* v___x_135_; lean_object* v___x_136_; 
lean_dec(v_h__2_132_);
v___x_135_ = lean_box(0);
v___x_136_ = lean_apply_1(v_h__1_131_, v___x_135_);
return v___x_136_;
}
else
{
lean_object* v_one_137_; lean_object* v_n_138_; lean_object* v___x_139_; 
lean_dec(v_h__1_131_);
v_one_137_ = lean_unsigned_to_nat(1u);
v_n_138_ = lean_nat_sub(v_x_130_, v_one_137_);
v___x_139_ = lean_apply_1(v_h__2_132_, v_n_138_);
return v___x_139_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Factorial_Basic_0__Nat_factorial_match__1_splitter___redArg___boxed(lean_object* v_x_140_, lean_object* v_h__1_141_, lean_object* v_h__2_142_){
_start:
{
lean_object* v_res_143_; 
v_res_143_ = lp_mathlib___private_Mathlib_Data_Nat_Factorial_Basic_0__Nat_factorial_match__1_splitter___redArg(v_x_140_, v_h__1_141_, v_h__2_142_);
lean_dec(v_x_140_);
return v_res_143_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Factorial_Basic_0__Nat_factorial_match__1_splitter(lean_object* v_motive_144_, lean_object* v_x_145_, lean_object* v_h__1_146_, lean_object* v_h__2_147_){
_start:
{
lean_object* v_zero_148_; uint8_t v_isZero_149_; 
v_zero_148_ = lean_unsigned_to_nat(0u);
v_isZero_149_ = lean_nat_dec_eq(v_x_145_, v_zero_148_);
if (v_isZero_149_ == 1)
{
lean_object* v___x_150_; lean_object* v___x_151_; 
lean_dec(v_h__2_147_);
v___x_150_ = lean_box(0);
v___x_151_ = lean_apply_1(v_h__1_146_, v___x_150_);
return v___x_151_;
}
else
{
lean_object* v_one_152_; lean_object* v_n_153_; lean_object* v___x_154_; 
lean_dec(v_h__1_146_);
v_one_152_ = lean_unsigned_to_nat(1u);
v_n_153_ = lean_nat_sub(v_x_145_, v_one_152_);
v___x_154_ = lean_apply_1(v_h__2_147_, v_n_153_);
return v___x_154_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Factorial_Basic_0__Nat_factorial_match__1_splitter___boxed(lean_object* v_motive_155_, lean_object* v_x_156_, lean_object* v_h__1_157_, lean_object* v_h__2_158_){
_start:
{
lean_object* v_res_159_; 
v_res_159_ = lp_mathlib___private_Mathlib_Data_Nat_Factorial_Basic_0__Nat_factorial_match__1_splitter(v_motive_155_, v_x_156_, v_h__1_157_, v_h__2_158_);
lean_dec(v_x_156_);
return v_res_159_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_descFactorial(lean_object* v_n_160_, lean_object* v_x_161_){
_start:
{
lean_object* v_zero_162_; uint8_t v_isZero_163_; 
v_zero_162_ = lean_unsigned_to_nat(0u);
v_isZero_163_ = lean_nat_dec_eq(v_x_161_, v_zero_162_);
if (v_isZero_163_ == 1)
{
lean_object* v___x_164_; 
v___x_164_ = lean_unsigned_to_nat(1u);
return v___x_164_;
}
else
{
lean_object* v_one_165_; lean_object* v_n_166_; lean_object* v___x_167_; lean_object* v___x_168_; lean_object* v___x_169_; 
v_one_165_ = lean_unsigned_to_nat(1u);
v_n_166_ = lean_nat_sub(v_x_161_, v_one_165_);
v___x_167_ = lean_nat_sub(v_n_160_, v_n_166_);
v___x_168_ = lp_mathlib_Nat_descFactorial(v_n_160_, v_n_166_);
lean_dec(v_n_166_);
v___x_169_ = lean_nat_mul(v___x_167_, v___x_168_);
lean_dec(v___x_168_);
lean_dec(v___x_167_);
return v___x_169_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_descFactorial___boxed(lean_object* v_n_170_, lean_object* v_x_171_){
_start:
{
lean_object* v_res_172_; 
v_res_172_ = lp_mathlib_Nat_descFactorial(v_n_170_, v_x_171_);
lean_dec(v_x_171_);
lean_dec(v_n_170_);
return v_res_172_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_ascFactorialBinary(lean_object* v_n_173_, lean_object* v_k_174_){
_start:
{
lean_object* v_zero_175_; uint8_t v_isZero_176_; 
v_zero_175_ = lean_unsigned_to_nat(0u);
v_isZero_176_ = lean_nat_dec_eq(v_k_174_, v_zero_175_);
if (v_isZero_176_ == 1)
{
lean_object* v___x_177_; 
v___x_177_ = lean_unsigned_to_nat(1u);
return v___x_177_;
}
else
{
lean_object* v_one_178_; lean_object* v_n_179_; uint8_t v_isZero_180_; 
v_one_178_ = lean_unsigned_to_nat(1u);
v_n_179_ = lean_nat_sub(v_k_174_, v_one_178_);
v_isZero_180_ = lean_nat_dec_eq(v_n_179_, v_zero_175_);
lean_dec(v_n_179_);
if (v_isZero_180_ == 1)
{
lean_inc(v_n_173_);
return v_n_173_;
}
else
{
lean_object* v___x_181_; lean_object* v___x_182_; lean_object* v___x_183_; lean_object* v___x_184_; lean_object* v___x_185_; lean_object* v___x_186_; lean_object* v___x_187_; 
v___x_181_ = lean_nat_shiftr(v_k_174_, v_one_178_);
v___x_182_ = lp_mathlib_Nat_ascFactorialBinary(v_n_173_, v___x_181_);
v___x_183_ = lean_nat_add(v_n_173_, v___x_181_);
lean_dec(v___x_181_);
v___x_184_ = lean_nat_add(v_k_174_, v_one_178_);
v___x_185_ = lean_nat_shiftr(v___x_184_, v_one_178_);
lean_dec(v___x_184_);
v___x_186_ = lp_mathlib_Nat_ascFactorialBinary(v___x_183_, v___x_185_);
lean_dec(v___x_185_);
lean_dec(v___x_183_);
v___x_187_ = lean_nat_mul(v___x_182_, v___x_186_);
lean_dec(v___x_186_);
lean_dec(v___x_182_);
return v___x_187_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_ascFactorialBinary___boxed(lean_object* v_n_188_, lean_object* v_k_189_){
_start:
{
lean_object* v_res_190_; 
v_res_190_ = lp_mathlib_Nat_ascFactorialBinary(v_n_188_, v_k_189_);
lean_dec(v_k_189_);
lean_dec(v_n_188_);
return v_res_190_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Factorial_Basic_0__Nat_ascFactorialBinary_match__1_splitter___redArg(lean_object* v_k_191_, lean_object* v_h__1_192_, lean_object* v_h__2_193_, lean_object* v_h__3_194_){
_start:
{
lean_object* v_zero_195_; uint8_t v_isZero_196_; 
v_zero_195_ = lean_unsigned_to_nat(0u);
v_isZero_196_ = lean_nat_dec_eq(v_k_191_, v_zero_195_);
if (v_isZero_196_ == 1)
{
lean_object* v___x_197_; lean_object* v___x_198_; 
lean_dec(v_h__3_194_);
lean_dec(v_h__2_193_);
v___x_197_ = lean_box(0);
v___x_198_ = lean_apply_1(v_h__1_192_, v___x_197_);
return v___x_198_;
}
else
{
lean_object* v_one_199_; lean_object* v_n_200_; uint8_t v_isZero_201_; 
lean_dec(v_h__1_192_);
v_one_199_ = lean_unsigned_to_nat(1u);
v_n_200_ = lean_nat_sub(v_k_191_, v_one_199_);
v_isZero_201_ = lean_nat_dec_eq(v_n_200_, v_zero_195_);
if (v_isZero_201_ == 1)
{
lean_object* v___x_202_; lean_object* v___x_203_; 
lean_dec(v_n_200_);
lean_dec(v_h__3_194_);
v___x_202_ = lean_box(0);
v___x_203_ = lean_apply_1(v_h__2_193_, v___x_202_);
return v___x_203_;
}
else
{
lean_object* v_n_204_; lean_object* v___x_205_; 
lean_dec(v_h__2_193_);
v_n_204_ = lean_nat_sub(v_n_200_, v_one_199_);
lean_dec(v_n_200_);
v___x_205_ = lean_apply_1(v_h__3_194_, v_n_204_);
return v___x_205_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Factorial_Basic_0__Nat_ascFactorialBinary_match__1_splitter___redArg___boxed(lean_object* v_k_206_, lean_object* v_h__1_207_, lean_object* v_h__2_208_, lean_object* v_h__3_209_){
_start:
{
lean_object* v_res_210_; 
v_res_210_ = lp_mathlib___private_Mathlib_Data_Nat_Factorial_Basic_0__Nat_ascFactorialBinary_match__1_splitter___redArg(v_k_206_, v_h__1_207_, v_h__2_208_, v_h__3_209_);
lean_dec(v_k_206_);
return v_res_210_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Factorial_Basic_0__Nat_ascFactorialBinary_match__1_splitter(lean_object* v_motive_211_, lean_object* v_k_212_, lean_object* v_h__1_213_, lean_object* v_h__2_214_, lean_object* v_h__3_215_){
_start:
{
lean_object* v_zero_216_; uint8_t v_isZero_217_; 
v_zero_216_ = lean_unsigned_to_nat(0u);
v_isZero_217_ = lean_nat_dec_eq(v_k_212_, v_zero_216_);
if (v_isZero_217_ == 1)
{
lean_object* v___x_218_; lean_object* v___x_219_; 
lean_dec(v_h__3_215_);
lean_dec(v_h__2_214_);
v___x_218_ = lean_box(0);
v___x_219_ = lean_apply_1(v_h__1_213_, v___x_218_);
return v___x_219_;
}
else
{
lean_object* v_one_220_; lean_object* v_n_221_; uint8_t v_isZero_222_; 
lean_dec(v_h__1_213_);
v_one_220_ = lean_unsigned_to_nat(1u);
v_n_221_ = lean_nat_sub(v_k_212_, v_one_220_);
v_isZero_222_ = lean_nat_dec_eq(v_n_221_, v_zero_216_);
if (v_isZero_222_ == 1)
{
lean_object* v___x_223_; lean_object* v___x_224_; 
lean_dec(v_n_221_);
lean_dec(v_h__3_215_);
v___x_223_ = lean_box(0);
v___x_224_ = lean_apply_1(v_h__2_214_, v___x_223_);
return v___x_224_;
}
else
{
lean_object* v_n_225_; lean_object* v___x_226_; 
lean_dec(v_h__2_214_);
v_n_225_ = lean_nat_sub(v_n_221_, v_one_220_);
lean_dec(v_n_221_);
v___x_226_ = lean_apply_1(v_h__3_215_, v_n_225_);
return v___x_226_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Factorial_Basic_0__Nat_ascFactorialBinary_match__1_splitter___boxed(lean_object* v_motive_227_, lean_object* v_k_228_, lean_object* v_h__1_229_, lean_object* v_h__2_230_, lean_object* v_h__3_231_){
_start:
{
lean_object* v_res_232_; 
v_res_232_ = lp_mathlib___private_Mathlib_Data_Nat_Factorial_Basic_0__Nat_ascFactorialBinary_match__1_splitter(v_motive_227_, v_k_228_, v_h__1_229_, v_h__2_230_, v_h__3_231_);
lean_dec(v_k_228_);
return v_res_232_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_factorialBinarySplitting(lean_object* v_n_233_){
_start:
{
lean_object* v___x_234_; lean_object* v___x_235_; 
v___x_234_ = lean_unsigned_to_nat(1u);
v___x_235_ = lp_mathlib_Nat_ascFactorialBinary(v___x_234_, v_n_233_);
return v___x_235_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_factorialBinarySplitting___boxed(lean_object* v_n_236_){
_start:
{
lean_object* v_res_237_; 
v_res_237_ = lp_mathlib_Nat_factorialBinarySplitting(v_n_236_);
lean_dec(v_n_236_);
return v_res_237_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_descFactorialBinary(lean_object* v_n_238_, lean_object* v_k_239_){
_start:
{
uint8_t v___x_240_; 
v___x_240_ = lean_nat_dec_lt(v_n_238_, v_k_239_);
if (v___x_240_ == 0)
{
lean_object* v___x_241_; lean_object* v___x_242_; lean_object* v___x_243_; lean_object* v___x_244_; 
v___x_241_ = lean_nat_sub(v_n_238_, v_k_239_);
v___x_242_ = lean_unsigned_to_nat(1u);
v___x_243_ = lean_nat_add(v___x_241_, v___x_242_);
lean_dec(v___x_241_);
v___x_244_ = lp_mathlib_Nat_ascFactorialBinary(v___x_243_, v_k_239_);
lean_dec(v___x_243_);
return v___x_244_;
}
else
{
lean_object* v___x_245_; 
v___x_245_ = lean_unsigned_to_nat(0u);
return v___x_245_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_descFactorialBinary___boxed(lean_object* v_n_246_, lean_object* v_k_247_){
_start:
{
lean_object* v_res_248_; 
v_res_248_ = lp_mathlib_Nat_descFactorialBinary(v_n_246_, v_k_247_);
lean_dec(v_k_247_);
lean_dec(v_n_246_);
return v_res_248_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Common(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_CrossRefAttribute(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Monotonicity_Attr(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Attr_Core(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Factorial_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Common(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_CrossRefAttribute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Monotonicity_Attr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Attr_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Nat_Factorial_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Nat_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Common(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_CrossRefAttribute(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Monotonicity_Attr(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Attr_Core(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Nat_Factorial_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Common(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_CrossRefAttribute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Monotonicity_Attr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Attr_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Factorial_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Nat_Factorial_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Nat_Factorial_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
