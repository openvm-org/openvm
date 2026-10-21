// Lean compiler output
// Module: Mathlib.Tactic.Lemma
// Imports: public import Init public meta import Init public import Mathlib.Tactic.Linter.Header
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
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_setKind(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkAtomFrom(lean_object*, lean_object*, uint8_t);
static const lean_string_object lp_mathlib_lemma___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "lemma"};
static const lean_object* lp_mathlib_lemma___closed__0 = (const lean_object*)&lp_mathlib_lemma___closed__0_value;
static const lean_ctor_object lp_mathlib_lemma___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_lemma___closed__0_value),LEAN_SCALAR_PTR_LITERAL(117, 34, 246, 137, 114, 183, 220, 217)}};
static const lean_object* lp_mathlib_lemma___closed__1 = (const lean_object*)&lp_mathlib_lemma___closed__1_value;
static const lean_string_object lp_mathlib_lemma___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_lemma___closed__2 = (const lean_object*)&lp_mathlib_lemma___closed__2_value;
static const lean_ctor_object lp_mathlib_lemma___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_lemma___closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_lemma___closed__3 = (const lean_object*)&lp_mathlib_lemma___closed__3_value;
static const lean_string_object lp_mathlib_lemma___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "declModifiers"};
static const lean_object* lp_mathlib_lemma___closed__4 = (const lean_object*)&lp_mathlib_lemma___closed__4_value;
static const lean_ctor_object lp_mathlib_lemma___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_lemma___closed__4_value),LEAN_SCALAR_PTR_LITERAL(113, 135, 0, 93, 130, 217, 220, 132)}};
static const lean_object* lp_mathlib_lemma___closed__5 = (const lean_object*)&lp_mathlib_lemma___closed__5_value;
static const lean_ctor_object lp_mathlib_lemma___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_lemma___closed__5_value)}};
static const lean_object* lp_mathlib_lemma___closed__6 = (const lean_object*)&lp_mathlib_lemma___closed__6_value;
static const lean_string_object lp_mathlib_lemma___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "group"};
static const lean_object* lp_mathlib_lemma___closed__7 = (const lean_object*)&lp_mathlib_lemma___closed__7_value;
static const lean_ctor_object lp_mathlib_lemma___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_lemma___closed__7_value),LEAN_SCALAR_PTR_LITERAL(206, 113, 20, 57, 188, 177, 187, 30)}};
static const lean_object* lp_mathlib_lemma___closed__8 = (const lean_object*)&lp_mathlib_lemma___closed__8_value;
static const lean_string_object lp_mathlib_lemma___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "lemma "};
static const lean_object* lp_mathlib_lemma___closed__9 = (const lean_object*)&lp_mathlib_lemma___closed__9_value;
static const lean_ctor_object lp_mathlib_lemma___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_lemma___closed__9_value)}};
static const lean_object* lp_mathlib_lemma___closed__10 = (const lean_object*)&lp_mathlib_lemma___closed__10_value;
static const lean_string_object lp_mathlib_lemma___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "declId"};
static const lean_object* lp_mathlib_lemma___closed__11 = (const lean_object*)&lp_mathlib_lemma___closed__11_value;
static const lean_ctor_object lp_mathlib_lemma___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_lemma___closed__11_value),LEAN_SCALAR_PTR_LITERAL(210, 155, 24, 168, 139, 44, 164, 47)}};
static const lean_object* lp_mathlib_lemma___closed__12 = (const lean_object*)&lp_mathlib_lemma___closed__12_value;
static const lean_ctor_object lp_mathlib_lemma___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_lemma___closed__12_value)}};
static const lean_object* lp_mathlib_lemma___closed__13 = (const lean_object*)&lp_mathlib_lemma___closed__13_value;
static const lean_ctor_object lp_mathlib_lemma___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_lemma___closed__3_value),((lean_object*)&lp_mathlib_lemma___closed__10_value),((lean_object*)&lp_mathlib_lemma___closed__13_value)}};
static const lean_object* lp_mathlib_lemma___closed__14 = (const lean_object*)&lp_mathlib_lemma___closed__14_value;
static const lean_string_object lp_mathlib_lemma___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "ppIndent"};
static const lean_object* lp_mathlib_lemma___closed__15 = (const lean_object*)&lp_mathlib_lemma___closed__15_value;
static const lean_ctor_object lp_mathlib_lemma___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_lemma___closed__15_value),LEAN_SCALAR_PTR_LITERAL(240, 142, 232, 190, 100, 212, 29, 41)}};
static const lean_object* lp_mathlib_lemma___closed__16 = (const lean_object*)&lp_mathlib_lemma___closed__16_value;
static const lean_string_object lp_mathlib_lemma___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "declSig"};
static const lean_object* lp_mathlib_lemma___closed__17 = (const lean_object*)&lp_mathlib_lemma___closed__17_value;
static const lean_ctor_object lp_mathlib_lemma___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_lemma___closed__17_value),LEAN_SCALAR_PTR_LITERAL(79, 160, 221, 255, 50, 155, 99, 177)}};
static const lean_object* lp_mathlib_lemma___closed__18 = (const lean_object*)&lp_mathlib_lemma___closed__18_value;
static const lean_ctor_object lp_mathlib_lemma___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_lemma___closed__18_value)}};
static const lean_object* lp_mathlib_lemma___closed__19 = (const lean_object*)&lp_mathlib_lemma___closed__19_value;
static const lean_ctor_object lp_mathlib_lemma___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_lemma___closed__16_value),((lean_object*)&lp_mathlib_lemma___closed__19_value)}};
static const lean_object* lp_mathlib_lemma___closed__20 = (const lean_object*)&lp_mathlib_lemma___closed__20_value;
static const lean_ctor_object lp_mathlib_lemma___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_lemma___closed__3_value),((lean_object*)&lp_mathlib_lemma___closed__14_value),((lean_object*)&lp_mathlib_lemma___closed__20_value)}};
static const lean_object* lp_mathlib_lemma___closed__21 = (const lean_object*)&lp_mathlib_lemma___closed__21_value;
static const lean_string_object lp_mathlib_lemma___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "declVal"};
static const lean_object* lp_mathlib_lemma___closed__22 = (const lean_object*)&lp_mathlib_lemma___closed__22_value;
static const lean_ctor_object lp_mathlib_lemma___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_lemma___closed__22_value),LEAN_SCALAR_PTR_LITERAL(19, 167, 222, 34, 119, 174, 4, 130)}};
static const lean_object* lp_mathlib_lemma___closed__23 = (const lean_object*)&lp_mathlib_lemma___closed__23_value;
static const lean_ctor_object lp_mathlib_lemma___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_lemma___closed__23_value)}};
static const lean_object* lp_mathlib_lemma___closed__24 = (const lean_object*)&lp_mathlib_lemma___closed__24_value;
static const lean_ctor_object lp_mathlib_lemma___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_lemma___closed__3_value),((lean_object*)&lp_mathlib_lemma___closed__21_value),((lean_object*)&lp_mathlib_lemma___closed__24_value)}};
static const lean_object* lp_mathlib_lemma___closed__25 = (const lean_object*)&lp_mathlib_lemma___closed__25_value;
static const lean_ctor_object lp_mathlib_lemma___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_lemma___closed__8_value),((lean_object*)&lp_mathlib_lemma___closed__25_value)}};
static const lean_object* lp_mathlib_lemma___closed__26 = (const lean_object*)&lp_mathlib_lemma___closed__26_value;
static const lean_ctor_object lp_mathlib_lemma___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_lemma___closed__3_value),((lean_object*)&lp_mathlib_lemma___closed__6_value),((lean_object*)&lp_mathlib_lemma___closed__26_value)}};
static const lean_object* lp_mathlib_lemma___closed__27 = (const lean_object*)&lp_mathlib_lemma___closed__27_value;
static const lean_ctor_object lp_mathlib_lemma___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_lemma___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_lemma___closed__27_value)}};
static const lean_object* lp_mathlib_lemma___closed__28 = (const lean_object*)&lp_mathlib_lemma___closed__28_value;
LEAN_EXPORT const lean_object* lp_mathlib_lemma = (const lean_object*)&lp_mathlib_lemma___closed__28_value;
static const lean_string_object lp_mathlib_expandLemma___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_expandLemma___redArg___closed__0 = (const lean_object*)&lp_mathlib_expandLemma___redArg___closed__0_value;
static const lean_string_object lp_mathlib_expandLemma___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_expandLemma___redArg___closed__1 = (const lean_object*)&lp_mathlib_expandLemma___redArg___closed__1_value;
static const lean_string_object lp_mathlib_expandLemma___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Command"};
static const lean_object* lp_mathlib_expandLemma___redArg___closed__2 = (const lean_object*)&lp_mathlib_expandLemma___redArg___closed__2_value;
static const lean_string_object lp_mathlib_expandLemma___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "declaration"};
static const lean_object* lp_mathlib_expandLemma___redArg___closed__3 = (const lean_object*)&lp_mathlib_expandLemma___redArg___closed__3_value;
static const lean_ctor_object lp_mathlib_expandLemma___redArg___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_expandLemma___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_expandLemma___redArg___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_expandLemma___redArg___closed__4_value_aux_0),((lean_object*)&lp_mathlib_expandLemma___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_expandLemma___redArg___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_expandLemma___redArg___closed__4_value_aux_1),((lean_object*)&lp_mathlib_expandLemma___redArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_expandLemma___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_expandLemma___redArg___closed__4_value_aux_2),((lean_object*)&lp_mathlib_expandLemma___redArg___closed__3_value),LEAN_SCALAR_PTR_LITERAL(157, 246, 223, 221, 242, 35, 238, 117)}};
static const lean_object* lp_mathlib_expandLemma___redArg___closed__4 = (const lean_object*)&lp_mathlib_expandLemma___redArg___closed__4_value;
static const lean_string_object lp_mathlib_expandLemma___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "theorem"};
static const lean_object* lp_mathlib_expandLemma___redArg___closed__5 = (const lean_object*)&lp_mathlib_expandLemma___redArg___closed__5_value;
static const lean_ctor_object lp_mathlib_expandLemma___redArg___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_expandLemma___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_expandLemma___redArg___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_expandLemma___redArg___closed__6_value_aux_0),((lean_object*)&lp_mathlib_expandLemma___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_expandLemma___redArg___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_expandLemma___redArg___closed__6_value_aux_1),((lean_object*)&lp_mathlib_expandLemma___redArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_expandLemma___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_expandLemma___redArg___closed__6_value_aux_2),((lean_object*)&lp_mathlib_expandLemma___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(238, 116, 137, 74, 194, 103, 58, 54)}};
static const lean_object* lp_mathlib_expandLemma___redArg___closed__6 = (const lean_object*)&lp_mathlib_expandLemma___redArg___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib_expandLemma___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_expandLemma(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_expandLemma___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_expandLemma___redArg(lean_object* v_stx_78_, lean_object* v_a_79_){
_start:
{
lean_object* v___y_81_; 
if (lean_obj_tag(v_stx_78_) == 1)
{
lean_object* v_info_85_; lean_object* v_kind_86_; lean_object* v_args_87_; lean_object* v___x_88_; lean_object* v___x_89_; uint8_t v___x_90_; 
v_info_85_ = lean_ctor_get(v_stx_78_, 0);
v_kind_86_ = lean_ctor_get(v_stx_78_, 1);
v_args_87_ = lean_ctor_get(v_stx_78_, 2);
v___x_88_ = lean_unsigned_to_nat(1u);
v___x_89_ = lean_array_get_size(v_args_87_);
v___x_90_ = lean_nat_dec_lt(v___x_88_, v___x_89_);
if (v___x_90_ == 0)
{
v___y_81_ = v_stx_78_;
goto v___jp_80_;
}
else
{
lean_object* v___x_92_; uint8_t v_isShared_93_; uint8_t v_isSharedCheck_126_; 
lean_inc_ref(v_args_87_);
lean_inc(v_kind_86_);
lean_inc(v_info_85_);
v_isSharedCheck_126_ = !lean_is_exclusive(v_stx_78_);
if (v_isSharedCheck_126_ == 0)
{
lean_object* v_unused_127_; lean_object* v_unused_128_; lean_object* v_unused_129_; 
v_unused_127_ = lean_ctor_get(v_stx_78_, 2);
lean_dec(v_unused_127_);
v_unused_128_ = lean_ctor_get(v_stx_78_, 1);
lean_dec(v_unused_128_);
v_unused_129_ = lean_ctor_get(v_stx_78_, 0);
lean_dec(v_unused_129_);
v___x_92_ = v_stx_78_;
v_isShared_93_ = v_isSharedCheck_126_;
goto v_resetjp_91_;
}
else
{
lean_dec(v_stx_78_);
v___x_92_ = lean_box(0);
v_isShared_93_ = v_isSharedCheck_126_;
goto v_resetjp_91_;
}
v_resetjp_91_:
{
lean_object* v_v_94_; lean_object* v___x_95_; lean_object* v_xs_x27_96_; lean_object* v___y_98_; 
v_v_94_ = lean_array_fget(v_args_87_, v___x_88_);
v___x_95_ = lean_box(0);
v_xs_x27_96_ = lean_array_fset(v_args_87_, v___x_88_, v___x_95_);
if (lean_obj_tag(v_v_94_) == 1)
{
lean_object* v_info_105_; lean_object* v_kind_106_; lean_object* v_args_107_; lean_object* v___x_108_; lean_object* v___x_109_; uint8_t v___x_110_; 
v_info_105_ = lean_ctor_get(v_v_94_, 0);
v_kind_106_ = lean_ctor_get(v_v_94_, 1);
v_args_107_ = lean_ctor_get(v_v_94_, 2);
v___x_108_ = lean_unsigned_to_nat(0u);
v___x_109_ = lean_array_get_size(v_args_107_);
v___x_110_ = lean_nat_dec_lt(v___x_108_, v___x_109_);
if (v___x_110_ == 0)
{
v___y_98_ = v_v_94_;
goto v___jp_97_;
}
else
{
lean_object* v___x_112_; uint8_t v_isShared_113_; uint8_t v_isSharedCheck_122_; 
lean_inc_ref(v_args_107_);
lean_inc(v_kind_106_);
lean_inc(v_info_105_);
v_isSharedCheck_122_ = !lean_is_exclusive(v_v_94_);
if (v_isSharedCheck_122_ == 0)
{
lean_object* v_unused_123_; lean_object* v_unused_124_; lean_object* v_unused_125_; 
v_unused_123_ = lean_ctor_get(v_v_94_, 2);
lean_dec(v_unused_123_);
v_unused_124_ = lean_ctor_get(v_v_94_, 1);
lean_dec(v_unused_124_);
v_unused_125_ = lean_ctor_get(v_v_94_, 0);
lean_dec(v_unused_125_);
v___x_112_ = v_v_94_;
v_isShared_113_ = v_isSharedCheck_122_;
goto v_resetjp_111_;
}
else
{
lean_dec(v_v_94_);
v___x_112_ = lean_box(0);
v_isShared_113_ = v_isSharedCheck_122_;
goto v_resetjp_111_;
}
v_resetjp_111_:
{
lean_object* v_v_114_; lean_object* v_xs_x27_115_; lean_object* v___x_116_; lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_120_; 
v_v_114_ = lean_array_fget(v_args_107_, v___x_108_);
v_xs_x27_115_ = lean_array_fset(v_args_107_, v___x_108_, v___x_95_);
v___x_116_ = ((lean_object*)(lp_mathlib_expandLemma___redArg___closed__5));
v___x_117_ = l_Lean_mkAtomFrom(v_v_114_, v___x_116_, v___x_110_);
lean_dec(v_v_114_);
v___x_118_ = lean_array_fset(v_xs_x27_115_, v___x_108_, v___x_117_);
if (v_isShared_113_ == 0)
{
lean_ctor_set(v___x_112_, 2, v___x_118_);
v___x_120_ = v___x_112_;
goto v_reusejp_119_;
}
else
{
lean_object* v_reuseFailAlloc_121_; 
v_reuseFailAlloc_121_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_121_, 0, v_info_105_);
lean_ctor_set(v_reuseFailAlloc_121_, 1, v_kind_106_);
lean_ctor_set(v_reuseFailAlloc_121_, 2, v___x_118_);
v___x_120_ = v_reuseFailAlloc_121_;
goto v_reusejp_119_;
}
v_reusejp_119_:
{
v___y_98_ = v___x_120_;
goto v___jp_97_;
}
}
}
}
else
{
v___y_98_ = v_v_94_;
goto v___jp_97_;
}
v___jp_97_:
{
lean_object* v___x_99_; lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_103_; 
v___x_99_ = ((lean_object*)(lp_mathlib_expandLemma___redArg___closed__6));
v___x_100_ = l_Lean_Syntax_setKind(v___y_98_, v___x_99_);
v___x_101_ = lean_array_fset(v_xs_x27_96_, v___x_88_, v___x_100_);
if (v_isShared_93_ == 0)
{
lean_ctor_set(v___x_92_, 2, v___x_101_);
v___x_103_ = v___x_92_;
goto v_reusejp_102_;
}
else
{
lean_object* v_reuseFailAlloc_104_; 
v_reuseFailAlloc_104_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_104_, 0, v_info_85_);
lean_ctor_set(v_reuseFailAlloc_104_, 1, v_kind_86_);
lean_ctor_set(v_reuseFailAlloc_104_, 2, v___x_101_);
v___x_103_ = v_reuseFailAlloc_104_;
goto v_reusejp_102_;
}
v_reusejp_102_:
{
v___y_81_ = v___x_103_;
goto v___jp_80_;
}
}
}
}
}
else
{
v___y_81_ = v_stx_78_;
goto v___jp_80_;
}
v___jp_80_:
{
lean_object* v___x_82_; lean_object* v___x_83_; lean_object* v___x_84_; 
v___x_82_ = ((lean_object*)(lp_mathlib_expandLemma___redArg___closed__4));
v___x_83_ = l_Lean_Syntax_setKind(v___y_81_, v___x_82_);
v___x_84_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_84_, 0, v___x_83_);
lean_ctor_set(v___x_84_, 1, v_a_79_);
return v___x_84_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_expandLemma(lean_object* v_stx_130_, lean_object* v_a_131_, lean_object* v_a_132_){
_start:
{
lean_object* v___x_133_; 
v___x_133_ = lp_mathlib_expandLemma___redArg(v_stx_130_, v_a_132_);
return v___x_133_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_expandLemma___boxed(lean_object* v_stx_134_, lean_object* v_a_135_, lean_object* v_a_136_){
_start:
{
lean_object* v_res_137_; 
v_res_137_ = lp_mathlib_expandLemma(v_stx_134_, v_a_135_, v_a_136_);
lean_dec_ref(v_a_135_);
return v_res_137_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_Header(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Lemma(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_Header(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Lemma(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_Header(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Lemma(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linter_Header(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Lemma(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Lemma(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Lemma(builtin);
}
#ifdef __cplusplus
}
#endif
