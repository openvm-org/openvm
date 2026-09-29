// Lean compiler output
// Module: Mathlib.Algebra.BigOperators.Group.Finset.Defs
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Equiv.Opposite public import Mathlib.Algebra.Group.TypeTags.Basic public import Mathlib.Algebra.BigOperators.Group.Multiset.Defs public import Mathlib.Data.Fintype.Sets public import Mathlib.Data.Multiset.Bind public meta import Mathlib.Tactic.ToDual
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
lean_object* l_Lean_Expr_appArg_x21(lean_object*);
lean_object* l_Lean_SubExpr_Pos_push(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Macro_throwUnsupported___redArg(lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesIdent(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_getPPNotation___boxed(lean_object*);
extern lean_object* l_Lean_binderIdent;
lean_object* l_Lean_Expr_bindingBody_x21(lean_object*);
lean_object* lean_expr_instantiate1(lean_object*, lean_object*);
uint8_t l_Lean_Expr_binderInfo(lean_object*);
lean_object* l_Lean_Expr_bindingDomain_x21(lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_isAppOfArity(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_delab(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_delab___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_getAppNumArgs(lean_object*);
extern lean_object* l_Lean_instInhabitedExpr;
lean_object* l_Lean_Expr_sort___override(lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_SubExpr_Pos_pushNaryArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_getPPFunBinderTypes___boxed(lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_getPPOption___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_sub(size_t, size_t);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_map___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_prod___redArg(lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_SepArray_ofElems(lean_object*, lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isNone(lean_object*);
lean_object* lp_mathlib_Multiset_sum___redArg(lean_object*, lean_object*);
lean_object* l_Array_mkArray1___redArg(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
extern lean_object* l_Lean_NameSet_empty;
lean_object* l_Lean_PrettyPrinter_Delaborator_withBindingBodyUnusedName___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_isLambda(lean_object*);
lean_object* l_Lean_TSyntax_getId(lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_failure___redArg();
lean_object* l_Lean_PrettyPrinter_Delaborator_withOverApp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_whenPPOption(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_prod___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_prod___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_prod(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_prod___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_sum___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_sum___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_sum(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_sum___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LibraryNote_operator__precedence__of__big__operators;
static const lean_string_object lp_mathlib_BigOperators_bigOpBinder___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "bigOpBinder"};
static const lean_object* lp_mathlib_BigOperators_bigOpBinder___closed__0 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__0_value;
static const lean_string_object lp_mathlib_BigOperators_bigOpBinder___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "BigOperators"};
static const lean_object* lp_mathlib_BigOperators_bigOpBinder___closed__1 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__1_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBinder___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__1_value),LEAN_SCALAR_PTR_LITERAL(107, 249, 74, 90, 68, 148, 111, 165)}};
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBinder___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__2_value_aux_0),((lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__0_value),LEAN_SCALAR_PTR_LITERAL(223, 37, 1, 6, 121, 107, 157, 234)}};
static const lean_object* lp_mathlib_BigOperators_bigOpBinder___closed__2 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__2_value;
static const lean_string_object lp_mathlib_BigOperators_bigOpBinder___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_BigOperators_bigOpBinder___closed__3 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__3_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBinder___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__3_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_BigOperators_bigOpBinder___closed__4 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__4_value;
static const lean_string_object lp_mathlib_BigOperators_bigOpBinder___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_BigOperators_bigOpBinder___closed__5 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__5_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBinder___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__5_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_BigOperators_bigOpBinder___closed__6 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__6_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBinder___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__6_value),((lean_object*)(((size_t)(1024) << 1) | 1))}};
static const lean_object* lp_mathlib_BigOperators_bigOpBinder___closed__7 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__7_value;
static const lean_string_object lp_mathlib_BigOperators_bigOpBinder___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_BigOperators_bigOpBinder___closed__8 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__8_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBinder___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__8_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_BigOperators_bigOpBinder___closed__9 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__9_value;
static const lean_string_object lp_mathlib_BigOperators_bigOpBinder___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "orelse"};
static const lean_object* lp_mathlib_BigOperators_bigOpBinder___closed__10 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__10_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBinder___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__10_value),LEAN_SCALAR_PTR_LITERAL(78, 76, 4, 51, 251, 212, 116, 5)}};
static const lean_object* lp_mathlib_BigOperators_bigOpBinder___closed__11 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__11_value;
static const lean_string_object lp_mathlib_BigOperators_bigOpBinder___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "group"};
static const lean_object* lp_mathlib_BigOperators_bigOpBinder___closed__12 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__12_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBinder___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__12_value),LEAN_SCALAR_PTR_LITERAL(206, 113, 20, 57, 188, 177, 187, 30)}};
static const lean_object* lp_mathlib_BigOperators_bigOpBinder___closed__13 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__13_value;
static const lean_string_object lp_mathlib_BigOperators_bigOpBinder___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = " : "};
static const lean_object* lp_mathlib_BigOperators_bigOpBinder___closed__14 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__14_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBinder___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__14_value)}};
static const lean_object* lp_mathlib_BigOperators_bigOpBinder___closed__15 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__15_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBinder___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__6_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_BigOperators_bigOpBinder___closed__16 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__16_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBinder___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__4_value),((lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__15_value),((lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__16_value)}};
static const lean_object* lp_mathlib_BigOperators_bigOpBinder___closed__17 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__17_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBinder___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__13_value),((lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__17_value)}};
static const lean_object* lp_mathlib_BigOperators_bigOpBinder___closed__18 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__18_value;
static const lean_string_object lp_mathlib_BigOperators_bigOpBinder___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "binderPred"};
static const lean_object* lp_mathlib_BigOperators_bigOpBinder___closed__19 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__19_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBinder___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__19_value),LEAN_SCALAR_PTR_LITERAL(218, 134, 142, 164, 134, 201, 62, 191)}};
static const lean_object* lp_mathlib_BigOperators_bigOpBinder___closed__20 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__20_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBinder___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__20_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_BigOperators_bigOpBinder___closed__21 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__21_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBinder___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__11_value),((lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__18_value),((lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__21_value)}};
static const lean_object* lp_mathlib_BigOperators_bigOpBinder___closed__22 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__22_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBinder___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__9_value),((lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__22_value)}};
static const lean_object* lp_mathlib_BigOperators_bigOpBinder___closed__23 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__23_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBinder___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__4_value),((lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__7_value),((lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__23_value)}};
static const lean_object* lp_mathlib_BigOperators_bigOpBinder___closed__24 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__24_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBinder___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 9}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__0_value),((lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__2_value),((lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__24_value)}};
static const lean_object* lp_mathlib_BigOperators_bigOpBinder___closed__25 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__25_value;
LEAN_EXPORT const lean_object* lp_mathlib_BigOperators_bigOpBinder = (const lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__25_value;
static const lean_string_object lp_mathlib_BigOperators_bigOpBinderParenthesized___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "bigOpBinderParenthesized"};
static const lean_object* lp_mathlib_BigOperators_bigOpBinderParenthesized___closed__0 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBinderParenthesized___closed__0_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBinderParenthesized___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__1_value),LEAN_SCALAR_PTR_LITERAL(107, 249, 74, 90, 68, 148, 111, 165)}};
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBinderParenthesized___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigOpBinderParenthesized___closed__1_value_aux_0),((lean_object*)&lp_mathlib_BigOperators_bigOpBinderParenthesized___closed__0_value),LEAN_SCALAR_PTR_LITERAL(179, 86, 156, 138, 200, 160, 93, 105)}};
static const lean_object* lp_mathlib_BigOperators_bigOpBinderParenthesized___closed__1 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBinderParenthesized___closed__1_value;
static const lean_string_object lp_mathlib_BigOperators_bigOpBinderParenthesized___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = " ("};
static const lean_object* lp_mathlib_BigOperators_bigOpBinderParenthesized___closed__2 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBinderParenthesized___closed__2_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBinderParenthesized___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigOpBinderParenthesized___closed__2_value)}};
static const lean_object* lp_mathlib_BigOperators_bigOpBinderParenthesized___closed__3 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBinderParenthesized___closed__3_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBinderParenthesized___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__4_value),((lean_object*)&lp_mathlib_BigOperators_bigOpBinderParenthesized___closed__3_value),((lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__25_value)}};
static const lean_object* lp_mathlib_BigOperators_bigOpBinderParenthesized___closed__4 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBinderParenthesized___closed__4_value;
static const lean_string_object lp_mathlib_BigOperators_bigOpBinderParenthesized___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib_BigOperators_bigOpBinderParenthesized___closed__5 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBinderParenthesized___closed__5_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBinderParenthesized___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigOpBinderParenthesized___closed__5_value)}};
static const lean_object* lp_mathlib_BigOperators_bigOpBinderParenthesized___closed__6 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBinderParenthesized___closed__6_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBinderParenthesized___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__4_value),((lean_object*)&lp_mathlib_BigOperators_bigOpBinderParenthesized___closed__4_value),((lean_object*)&lp_mathlib_BigOperators_bigOpBinderParenthesized___closed__6_value)}};
static const lean_object* lp_mathlib_BigOperators_bigOpBinderParenthesized___closed__7 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBinderParenthesized___closed__7_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBinderParenthesized___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 9}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigOpBinderParenthesized___closed__0_value),((lean_object*)&lp_mathlib_BigOperators_bigOpBinderParenthesized___closed__1_value),((lean_object*)&lp_mathlib_BigOperators_bigOpBinderParenthesized___closed__7_value)}};
static const lean_object* lp_mathlib_BigOperators_bigOpBinderParenthesized___closed__8 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBinderParenthesized___closed__8_value;
LEAN_EXPORT const lean_object* lp_mathlib_BigOperators_bigOpBinderParenthesized = (const lean_object*)&lp_mathlib_BigOperators_bigOpBinderParenthesized___closed__8_value;
static const lean_string_object lp_mathlib_BigOperators_bigOpBinderCollection___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "bigOpBinderCollection"};
static const lean_object* lp_mathlib_BigOperators_bigOpBinderCollection___closed__0 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBinderCollection___closed__0_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBinderCollection___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__1_value),LEAN_SCALAR_PTR_LITERAL(107, 249, 74, 90, 68, 148, 111, 165)}};
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBinderCollection___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigOpBinderCollection___closed__1_value_aux_0),((lean_object*)&lp_mathlib_BigOperators_bigOpBinderCollection___closed__0_value),LEAN_SCALAR_PTR_LITERAL(18, 55, 33, 255, 210, 50, 109, 169)}};
static const lean_object* lp_mathlib_BigOperators_bigOpBinderCollection___closed__1 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBinderCollection___closed__1_value;
static const lean_string_object lp_mathlib_BigOperators_bigOpBinderCollection___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "many1"};
static const lean_object* lp_mathlib_BigOperators_bigOpBinderCollection___closed__2 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBinderCollection___closed__2_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBinderCollection___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_bigOpBinderCollection___closed__2_value),LEAN_SCALAR_PTR_LITERAL(55, 136, 52, 6, 12, 19, 78, 239)}};
static const lean_object* lp_mathlib_BigOperators_bigOpBinderCollection___closed__3 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBinderCollection___closed__3_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBinderCollection___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigOpBinderCollection___closed__3_value),((lean_object*)&lp_mathlib_BigOperators_bigOpBinderParenthesized___closed__8_value)}};
static const lean_object* lp_mathlib_BigOperators_bigOpBinderCollection___closed__4 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBinderCollection___closed__4_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBinderCollection___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 9}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigOpBinderCollection___closed__0_value),((lean_object*)&lp_mathlib_BigOperators_bigOpBinderCollection___closed__1_value),((lean_object*)&lp_mathlib_BigOperators_bigOpBinderCollection___closed__4_value)}};
static const lean_object* lp_mathlib_BigOperators_bigOpBinderCollection___closed__5 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBinderCollection___closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_BigOperators_bigOpBinderCollection = (const lean_object*)&lp_mathlib_BigOperators_bigOpBinderCollection___closed__5_value;
static const lean_string_object lp_mathlib_BigOperators_bigOpBinders___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "bigOpBinders"};
static const lean_object* lp_mathlib_BigOperators_bigOpBinders___closed__0 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBinders___closed__0_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBinders___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__1_value),LEAN_SCALAR_PTR_LITERAL(107, 249, 74, 90, 68, 148, 111, 165)}};
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBinders___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigOpBinders___closed__1_value_aux_0),((lean_object*)&lp_mathlib_BigOperators_bigOpBinders___closed__0_value),LEAN_SCALAR_PTR_LITERAL(83, 255, 230, 20, 101, 2, 142, 38)}};
static const lean_object* lp_mathlib_BigOperators_bigOpBinders___closed__1 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBinders___closed__1_value;
static const lean_string_object lp_mathlib_BigOperators_bigOpBinders___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_mathlib_BigOperators_bigOpBinders___closed__2 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBinders___closed__2_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBinders___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_bigOpBinders___closed__2_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_mathlib_BigOperators_bigOpBinders___closed__3 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBinders___closed__3_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBinders___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigOpBinders___closed__3_value)}};
static const lean_object* lp_mathlib_BigOperators_bigOpBinders___closed__4 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBinders___closed__4_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBinders___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__4_value),((lean_object*)&lp_mathlib_BigOperators_bigOpBinders___closed__4_value),((lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__25_value)}};
static const lean_object* lp_mathlib_BigOperators_bigOpBinders___closed__5 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBinders___closed__5_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBinders___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__11_value),((lean_object*)&lp_mathlib_BigOperators_bigOpBinderCollection___closed__5_value),((lean_object*)&lp_mathlib_BigOperators_bigOpBinders___closed__5_value)}};
static const lean_object* lp_mathlib_BigOperators_bigOpBinders___closed__6 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBinders___closed__6_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBinders___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 9}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigOpBinders___closed__0_value),((lean_object*)&lp_mathlib_BigOperators_bigOpBinders___closed__1_value),((lean_object*)&lp_mathlib_BigOperators_bigOpBinders___closed__6_value)}};
static const lean_object* lp_mathlib_BigOperators_bigOpBinders___closed__7 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBinders___closed__7_value;
LEAN_EXPORT const lean_object* lp_mathlib_BigOperators_bigOpBinders = (const lean_object*)&lp_mathlib_BigOperators_bigOpBinders___closed__7_value;
static const lean_string_object lp_mathlib_BigOperators_processBigOpBinder___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__0 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__0_value;
static const lean_string_object lp_mathlib_BigOperators_processBigOpBinder___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 12, .m_data = "binderPred∈_"};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__1 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__1_value;
static const lean_ctor_object lp_mathlib_BigOperators_processBigOpBinder___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_BigOperators_processBigOpBinder___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__2_value_aux_0),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__1_value),LEAN_SCALAR_PTR_LITERAL(150, 164, 254, 63, 76, 57, 126, 92)}};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__2 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__2_value;
static const lean_string_object lp_mathlib_BigOperators_processBigOpBinder___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 12, .m_data = "binderPred∉_"};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__3 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__3_value;
static const lean_ctor_object lp_mathlib_BigOperators_processBigOpBinder___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_BigOperators_processBigOpBinder___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__4_value_aux_0),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__3_value),LEAN_SCALAR_PTR_LITERAL(147, 253, 164, 249, 200, 108, 121, 70)}};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__4 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__4_value;
static const lean_string_object lp_mathlib_BigOperators_processBigOpBinder___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 12, .m_data = "binderPred≠_"};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__5 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__5_value;
static const lean_ctor_object lp_mathlib_BigOperators_processBigOpBinder___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_BigOperators_processBigOpBinder___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__6_value_aux_0),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__5_value),LEAN_SCALAR_PTR_LITERAL(39, 40, 245, 52, 138, 78, 140, 19)}};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__6 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__6_value;
static const lean_string_object lp_mathlib_BigOperators_processBigOpBinder___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "binderPred<_"};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__7 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__7_value;
static const lean_ctor_object lp_mathlib_BigOperators_processBigOpBinder___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_BigOperators_processBigOpBinder___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__8_value_aux_0),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__7_value),LEAN_SCALAR_PTR_LITERAL(87, 122, 58, 108, 39, 32, 195, 29)}};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__8 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__8_value;
static const lean_string_object lp_mathlib_BigOperators_processBigOpBinder___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 12, .m_data = "binderPred≤_"};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__9 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__9_value;
static const lean_ctor_object lp_mathlib_BigOperators_processBigOpBinder___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_BigOperators_processBigOpBinder___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__10_value_aux_0),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__9_value),LEAN_SCALAR_PTR_LITERAL(118, 254, 52, 209, 56, 53, 218, 188)}};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__10 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__10_value;
static const lean_string_object lp_mathlib_BigOperators_processBigOpBinder___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "binderPred>_"};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__11 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__11_value;
static const lean_ctor_object lp_mathlib_BigOperators_processBigOpBinder___closed__12_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_BigOperators_processBigOpBinder___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__12_value_aux_0),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__11_value),LEAN_SCALAR_PTR_LITERAL(140, 244, 246, 184, 111, 78, 213, 47)}};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__12 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__12_value;
static const lean_string_object lp_mathlib_BigOperators_processBigOpBinder___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 12, .m_data = "binderPred≥_"};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__13 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__13_value;
static const lean_ctor_object lp_mathlib_BigOperators_processBigOpBinder___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_BigOperators_processBigOpBinder___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__14_value_aux_0),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__13_value),LEAN_SCALAR_PTR_LITERAL(212, 139, 67, 46, 49, 133, 157, 246)}};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__14 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__14_value;
static const lean_string_object lp_mathlib_BigOperators_processBigOpBinder___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__15 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__15_value;
static const lean_string_object lp_mathlib_BigOperators_processBigOpBinder___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__16 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__16_value;
static const lean_string_object lp_mathlib_BigOperators_processBigOpBinder___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__17 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__17_value;
static const lean_ctor_object lp_mathlib_BigOperators_processBigOpBinder___closed__18_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_BigOperators_processBigOpBinder___closed__18_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__18_value_aux_0),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__15_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_BigOperators_processBigOpBinder___closed__18_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__18_value_aux_1),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__16_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_BigOperators_processBigOpBinder___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__18_value_aux_2),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__17_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__18 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__18_value;
static const lean_string_object lp_mathlib_BigOperators_processBigOpBinder___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "Finset.Ici"};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__19 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__19_value;
static lean_once_cell_t lp_mathlib_BigOperators_processBigOpBinder___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__20;
static const lean_string_object lp_mathlib_BigOperators_processBigOpBinder___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Finset"};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__21 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__21_value;
static const lean_string_object lp_mathlib_BigOperators_processBigOpBinder___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Ici"};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__22 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__22_value;
static const lean_ctor_object lp_mathlib_BigOperators_processBigOpBinder___closed__23_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__21_value),LEAN_SCALAR_PTR_LITERAL(87, 75, 221, 45, 221, 79, 84, 42)}};
static const lean_ctor_object lp_mathlib_BigOperators_processBigOpBinder___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__23_value_aux_0),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__22_value),LEAN_SCALAR_PTR_LITERAL(230, 63, 145, 248, 144, 220, 203, 85)}};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__23 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__23_value;
static const lean_string_object lp_mathlib_BigOperators_processBigOpBinder___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__24 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__24_value;
static const lean_ctor_object lp_mathlib_BigOperators_processBigOpBinder___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__24_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__25 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__25_value;
static const lean_string_object lp_mathlib_BigOperators_processBigOpBinder___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "Finset.Ioi"};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__26 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__26_value;
static lean_once_cell_t lp_mathlib_BigOperators_processBigOpBinder___closed__27_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__27;
static const lean_string_object lp_mathlib_BigOperators_processBigOpBinder___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Ioi"};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__28 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__28_value;
static const lean_ctor_object lp_mathlib_BigOperators_processBigOpBinder___closed__29_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__21_value),LEAN_SCALAR_PTR_LITERAL(87, 75, 221, 45, 221, 79, 84, 42)}};
static const lean_ctor_object lp_mathlib_BigOperators_processBigOpBinder___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__29_value_aux_0),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__28_value),LEAN_SCALAR_PTR_LITERAL(81, 37, 134, 251, 22, 233, 98, 30)}};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__29 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__29_value;
static const lean_string_object lp_mathlib_BigOperators_processBigOpBinder___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "Finset.Iic"};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__30 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__30_value;
static lean_once_cell_t lp_mathlib_BigOperators_processBigOpBinder___closed__31_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__31;
static const lean_string_object lp_mathlib_BigOperators_processBigOpBinder___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Iic"};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__32 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__32_value;
static const lean_ctor_object lp_mathlib_BigOperators_processBigOpBinder___closed__33_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__21_value),LEAN_SCALAR_PTR_LITERAL(87, 75, 221, 45, 221, 79, 84, 42)}};
static const lean_ctor_object lp_mathlib_BigOperators_processBigOpBinder___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__33_value_aux_0),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__32_value),LEAN_SCALAR_PTR_LITERAL(47, 251, 203, 130, 143, 174, 27, 182)}};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__33 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__33_value;
static const lean_string_object lp_mathlib_BigOperators_processBigOpBinder___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "Finset.Iio"};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__34 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__34_value;
static lean_once_cell_t lp_mathlib_BigOperators_processBigOpBinder___closed__35_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__35;
static const lean_string_object lp_mathlib_BigOperators_processBigOpBinder___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Iio"};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__36 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__36_value;
static const lean_ctor_object lp_mathlib_BigOperators_processBigOpBinder___closed__37_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__21_value),LEAN_SCALAR_PTR_LITERAL(87, 75, 221, 45, 221, 79, 84, 42)}};
static const lean_ctor_object lp_mathlib_BigOperators_processBigOpBinder___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__37_value_aux_0),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__36_value),LEAN_SCALAR_PTR_LITERAL(143, 117, 206, 115, 76, 88, 226, 57)}};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__37 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__37_value;
static const lean_string_object lp_mathlib_BigOperators_processBigOpBinder___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "Finset.univ.erase"};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__38 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__38_value;
static lean_once_cell_t lp_mathlib_BigOperators_processBigOpBinder___closed__39_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__39;
static const lean_string_object lp_mathlib_BigOperators_processBigOpBinder___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "univ"};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__40 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__40_value;
static const lean_string_object lp_mathlib_BigOperators_processBigOpBinder___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "erase"};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__41 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__41_value;
static const lean_ctor_object lp_mathlib_BigOperators_processBigOpBinder___closed__42_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__21_value),LEAN_SCALAR_PTR_LITERAL(87, 75, 221, 45, 221, 79, 84, 42)}};
static const lean_ctor_object lp_mathlib_BigOperators_processBigOpBinder___closed__42_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__42_value_aux_0),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__40_value),LEAN_SCALAR_PTR_LITERAL(177, 108, 234, 69, 25, 31, 35, 26)}};
static const lean_ctor_object lp_mathlib_BigOperators_processBigOpBinder___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__42_value_aux_1),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__41_value),LEAN_SCALAR_PTR_LITERAL(161, 250, 186, 216, 249, 198, 71, 118)}};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__42 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__42_value;
static const lean_string_object lp_mathlib_BigOperators_processBigOpBinder___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "finsetStx"};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__43 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__43_value;
static const lean_ctor_object lp_mathlib_BigOperators_processBigOpBinder___closed__44_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__43_value),LEAN_SCALAR_PTR_LITERAL(244, 212, 197, 80, 197, 7, 34, 44)}};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__44 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__44_value;
static const lean_string_object lp_mathlib_BigOperators_processBigOpBinder___closed__45_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "finset%"};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__45 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__45_value;
static const lean_string_object lp_mathlib_BigOperators_processBigOpBinder___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 6, .m_data = "term_ᶜ"};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__46 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__46_value;
static const lean_ctor_object lp_mathlib_BigOperators_processBigOpBinder___closed__47_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__46_value),LEAN_SCALAR_PTR_LITERAL(128, 3, 137, 103, 191, 193, 176, 89)}};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__47 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__47_value;
static const lean_string_object lp_mathlib_BigOperators_processBigOpBinder___closed__48_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "ᶜ"};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__48 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__48_value;
static const lean_string_object lp_mathlib_BigOperators_processBigOpBinder___closed__49_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "typeAscription"};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__49 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__49_value;
static const lean_ctor_object lp_mathlib_BigOperators_processBigOpBinder___closed__50_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_BigOperators_processBigOpBinder___closed__50_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__50_value_aux_0),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__15_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_BigOperators_processBigOpBinder___closed__50_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__50_value_aux_1),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__16_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_BigOperators_processBigOpBinder___closed__50_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__50_value_aux_2),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__49_value),LEAN_SCALAR_PTR_LITERAL(247, 209, 88, 141, 5, 195, 49, 74)}};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__50 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__50_value;
static const lean_string_object lp_mathlib_BigOperators_processBigOpBinder___closed__51_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "hygienicLParen"};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__51 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__51_value;
static const lean_ctor_object lp_mathlib_BigOperators_processBigOpBinder___closed__52_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_BigOperators_processBigOpBinder___closed__52_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__52_value_aux_0),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__15_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_BigOperators_processBigOpBinder___closed__52_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__52_value_aux_1),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__16_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_BigOperators_processBigOpBinder___closed__52_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__52_value_aux_2),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__51_value),LEAN_SCALAR_PTR_LITERAL(41, 104, 206, 51, 21, 254, 100, 101)}};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__52 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__52_value;
static const lean_string_object lp_mathlib_BigOperators_processBigOpBinder___closed__53_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__53 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__53_value;
static const lean_string_object lp_mathlib_BigOperators_processBigOpBinder___closed__54_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "hygieneInfo"};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__54 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__54_value;
static const lean_ctor_object lp_mathlib_BigOperators_processBigOpBinder___closed__55_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__54_value),LEAN_SCALAR_PTR_LITERAL(27, 64, 36, 144, 170, 151, 255, 136)}};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__55 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__55_value;
static const lean_string_object lp_mathlib_BigOperators_processBigOpBinder___closed__56_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__56 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__56_value;
static lean_once_cell_t lp_mathlib_BigOperators_processBigOpBinder___closed__57_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__57;
static const lean_string_object lp_mathlib_BigOperators_processBigOpBinder___closed__58_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "Finset.univ"};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__58 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__58_value;
static lean_once_cell_t lp_mathlib_BigOperators_processBigOpBinder___closed__59_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__59;
static const lean_ctor_object lp_mathlib_BigOperators_processBigOpBinder___closed__60_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__21_value),LEAN_SCALAR_PTR_LITERAL(87, 75, 221, 45, 221, 79, 84, 42)}};
static const lean_ctor_object lp_mathlib_BigOperators_processBigOpBinder___closed__60_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__60_value_aux_0),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__40_value),LEAN_SCALAR_PTR_LITERAL(177, 108, 234, 69, 25, 31, 35, 26)}};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__60 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__60_value;
static const lean_string_object lp_mathlib_BigOperators_processBigOpBinder___closed__61_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ":"};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__61 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__61_value;
static lean_once_cell_t lp_mathlib_BigOperators_processBigOpBinder___closed__62_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__62;
static const lean_ctor_object lp_mathlib_BigOperators_processBigOpBinder___closed__63_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__21_value),LEAN_SCALAR_PTR_LITERAL(87, 75, 221, 45, 221, 79, 84, 42)}};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__63 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__63_value;
static const lean_string_object lp_mathlib_BigOperators_processBigOpBinder___closed__64_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "paren"};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__64 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__64_value;
static const lean_ctor_object lp_mathlib_BigOperators_processBigOpBinder___closed__65_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_BigOperators_processBigOpBinder___closed__65_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__65_value_aux_0),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__15_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_BigOperators_processBigOpBinder___closed__65_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__65_value_aux_1),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__16_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_BigOperators_processBigOpBinder___closed__65_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__65_value_aux_2),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__64_value),LEAN_SCALAR_PTR_LITERAL(124, 9, 161, 194, 227, 100, 20, 110)}};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__65 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__65_value;
static const lean_string_object lp_mathlib_BigOperators_processBigOpBinder___closed__66_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "term_=_"};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__66 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__66_value;
static const lean_ctor_object lp_mathlib_BigOperators_processBigOpBinder___closed__67_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__66_value),LEAN_SCALAR_PTR_LITERAL(167, 251, 107, 62, 223, 239, 203, 78)}};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__67 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__67_value;
static const lean_string_object lp_mathlib_BigOperators_processBigOpBinder___closed__68_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "term_+_"};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__68 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__68_value;
static const lean_ctor_object lp_mathlib_BigOperators_processBigOpBinder___closed__69_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__68_value),LEAN_SCALAR_PTR_LITERAL(57, 160, 89, 154, 247, 230, 95, 119)}};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__69 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__69_value;
static const lean_string_object lp_mathlib_BigOperators_processBigOpBinder___closed__70_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "anonymousCtor"};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__70 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__70_value;
static const lean_ctor_object lp_mathlib_BigOperators_processBigOpBinder___closed__71_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_BigOperators_processBigOpBinder___closed__71_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__71_value_aux_0),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__15_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_BigOperators_processBigOpBinder___closed__71_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__71_value_aux_1),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__16_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_BigOperators_processBigOpBinder___closed__71_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__71_value_aux_2),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__70_value),LEAN_SCALAR_PTR_LITERAL(56, 53, 154, 97, 179, 232, 94, 186)}};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__71 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__71_value;
static const lean_string_object lp_mathlib_BigOperators_processBigOpBinder___closed__72_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "⟨"};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__72 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__72_value;
static const lean_string_object lp_mathlib_BigOperators_processBigOpBinder___closed__73_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__73 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__73_value;
static const lean_string_object lp_mathlib_BigOperators_processBigOpBinder___closed__74_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "⟩"};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__74 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__74_value;
static const lean_string_object lp_mathlib_BigOperators_processBigOpBinder___closed__75_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "Finset.Nat.antidiagonal"};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__75 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__75_value;
static lean_once_cell_t lp_mathlib_BigOperators_processBigOpBinder___closed__76_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__76;
static const lean_string_object lp_mathlib_BigOperators_processBigOpBinder___closed__77_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Nat"};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__77 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__77_value;
static const lean_string_object lp_mathlib_BigOperators_processBigOpBinder___closed__78_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "antidiagonal"};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__78 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__78_value;
static const lean_ctor_object lp_mathlib_BigOperators_processBigOpBinder___closed__79_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__21_value),LEAN_SCALAR_PTR_LITERAL(87, 75, 221, 45, 221, 79, 84, 42)}};
static const lean_ctor_object lp_mathlib_BigOperators_processBigOpBinder___closed__79_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__79_value_aux_0),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__77_value),LEAN_SCALAR_PTR_LITERAL(63, 246, 152, 140, 193, 17, 220, 60)}};
static const lean_ctor_object lp_mathlib_BigOperators_processBigOpBinder___closed__79_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__79_value_aux_1),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__78_value),LEAN_SCALAR_PTR_LITERAL(72, 148, 18, 116, 186, 116, 83, 174)}};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinder___closed__79 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__79_value;
LEAN_EXPORT lean_object* lp_mathlib_BigOperators_processBigOpBinder(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BigOperators_processBigOpBinder___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00BigOperators_processBigOpBinders_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00BigOperators_processBigOpBinders_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00BigOperators_processBigOpBinders_spec__1(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00BigOperators_processBigOpBinders_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_BigOperators_processBigOpBinders___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_BigOperators_processBigOpBinders___closed__0 = (const lean_object*)&lp_mathlib_BigOperators_processBigOpBinders___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_BigOperators_processBigOpBinders(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BigOperators_processBigOpBinders___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00BigOperators_bigOpBindersPattern_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00BigOperators_bigOpBindersPattern_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_BigOperators_bigOpBindersPattern___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_BigOperators_bigOpBindersPattern___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_BigOperators_bigOpBindersPattern(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BigOperators_bigOpBindersPattern___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00BigOperators_bigOpBindersProd_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "SProd"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00BigOperators_bigOpBindersProd_spec__0___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00BigOperators_bigOpBindersProd_spec__0___closed__0_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00BigOperators_bigOpBindersProd_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "sprod"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00BigOperators_bigOpBindersProd_spec__0___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00BigOperators_bigOpBindersProd_spec__0___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00BigOperators_bigOpBindersProd_spec__0___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00BigOperators_bigOpBindersProd_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(146, 74, 35, 234, 66, 135, 46, 236)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00BigOperators_bigOpBindersProd_spec__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00BigOperators_bigOpBindersProd_spec__0___closed__2_value_aux_0),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00BigOperators_bigOpBindersProd_spec__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(86, 89, 253, 36, 188, 23, 156, 35)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00BigOperators_bigOpBindersProd_spec__0___closed__2 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00BigOperators_bigOpBindersProd_spec__0___closed__2_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00BigOperators_bigOpBindersProd_spec__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "SProd.sprod"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00BigOperators_bigOpBindersProd_spec__0___closed__3 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00BigOperators_bigOpBindersProd_spec__0___closed__3_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00BigOperators_bigOpBindersProd_spec__0___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00BigOperators_bigOpBindersProd_spec__0___closed__4;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00BigOperators_bigOpBindersProd_spec__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00BigOperators_bigOpBindersProd_spec__0___closed__2_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00BigOperators_bigOpBindersProd_spec__0___closed__5 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00BigOperators_bigOpBindersProd_spec__0___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00BigOperators_bigOpBindersProd_spec__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00BigOperators_bigOpBindersProd_spec__0___closed__5_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00BigOperators_bigOpBindersProd_spec__0___closed__6 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00BigOperators_bigOpBindersProd_spec__0___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00BigOperators_bigOpBindersProd_spec__0(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00BigOperators_bigOpBindersProd_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBindersProd___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__1_value),LEAN_SCALAR_PTR_LITERAL(107, 249, 74, 90, 68, 148, 111, 165)}};
static const lean_object* lp_mathlib_BigOperators_bigOpBindersProd___closed__0 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__0_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBindersProd___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__0_value)}};
static const lean_object* lp_mathlib_BigOperators_bigOpBindersProd___closed__1 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__1_value;
static const lean_string_object lp_mathlib_BigOperators_bigOpBindersProd___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Meta"};
static const lean_object* lp_mathlib_BigOperators_bigOpBindersProd___closed__2 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__2_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBindersProd___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBindersProd___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__3_value_aux_0),((lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__2_value),LEAN_SCALAR_PTR_LITERAL(194, 50, 106, 158, 41, 60, 103, 214)}};
static const lean_object* lp_mathlib_BigOperators_bigOpBindersProd___closed__3 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__3_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBindersProd___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__3_value)}};
static const lean_object* lp_mathlib_BigOperators_bigOpBindersProd___closed__4 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__4_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBindersProd___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_object* lp_mathlib_BigOperators_bigOpBindersProd___closed__5 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__5_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBindersProd___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__5_value)}};
static const lean_object* lp_mathlib_BigOperators_bigOpBindersProd___closed__6 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__6_value;
static const lean_string_object lp_mathlib_BigOperators_bigOpBindersProd___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Batteries"};
static const lean_object* lp_mathlib_BigOperators_bigOpBindersProd___closed__7 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__7_value;
static const lean_string_object lp_mathlib_BigOperators_bigOpBindersProd___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "ExtendedBinder"};
static const lean_object* lp_mathlib_BigOperators_bigOpBindersProd___closed__8 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__8_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBindersProd___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__7_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBindersProd___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__9_value_aux_0),((lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__8_value),LEAN_SCALAR_PTR_LITERAL(56, 78, 248, 154, 49, 0, 91, 17)}};
static const lean_object* lp_mathlib_BigOperators_bigOpBindersProd___closed__9 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__9_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBindersProd___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__9_value)}};
static const lean_object* lp_mathlib_BigOperators_bigOpBindersProd___closed__10 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__10_value;
static const lean_string_object lp_mathlib_BigOperators_bigOpBindersProd___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Function"};
static const lean_object* lp_mathlib_BigOperators_bigOpBindersProd___closed__11 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__11_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBindersProd___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__11_value),LEAN_SCALAR_PTR_LITERAL(225, 8, 186, 189, 152, 89, 197, 12)}};
static const lean_object* lp_mathlib_BigOperators_bigOpBindersProd___closed__12 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__12_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBindersProd___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__12_value)}};
static const lean_object* lp_mathlib_BigOperators_bigOpBindersProd___closed__13 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__13_value;
static const lean_string_object lp_mathlib_BigOperators_bigOpBindersProd___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Fin"};
static const lean_object* lp_mathlib_BigOperators_bigOpBindersProd___closed__14 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__14_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBindersProd___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__14_value),LEAN_SCALAR_PTR_LITERAL(62, 91, 162, 2, 110, 238, 123, 219)}};
static const lean_object* lp_mathlib_BigOperators_bigOpBindersProd___closed__15 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__15_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBindersProd___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__15_value)}};
static const lean_object* lp_mathlib_BigOperators_bigOpBindersProd___closed__16 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__16_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBindersProd___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__16_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_BigOperators_bigOpBindersProd___closed__17 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__17_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBindersProd___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__13_value),((lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__17_value)}};
static const lean_object* lp_mathlib_BigOperators_bigOpBindersProd___closed__18 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__18_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBindersProd___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__10_value),((lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__18_value)}};
static const lean_object* lp_mathlib_BigOperators_bigOpBindersProd___closed__19 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__19_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBindersProd___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__6_value),((lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__19_value)}};
static const lean_object* lp_mathlib_BigOperators_bigOpBindersProd___closed__20 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__20_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBindersProd___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__4_value),((lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__20_value)}};
static const lean_object* lp_mathlib_BigOperators_bigOpBindersProd___closed__21 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__21_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBindersProd___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__1_value),((lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__21_value)}};
static const lean_object* lp_mathlib_BigOperators_bigOpBindersProd___closed__22 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__22_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBindersProd___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__60_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_BigOperators_bigOpBindersProd___closed__23 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__23_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBindersProd___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__23_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_BigOperators_bigOpBindersProd___closed__24 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__24_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBindersProd___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__63_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_BigOperators_bigOpBindersProd___closed__25 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__25_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBindersProd___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__63_value)}};
static const lean_object* lp_mathlib_BigOperators_bigOpBindersProd___closed__26 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__26_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBindersProd___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__26_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_BigOperators_bigOpBindersProd___closed__27 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__27_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBindersProd___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__25_value),((lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__27_value)}};
static const lean_object* lp_mathlib_BigOperators_bigOpBindersProd___closed__28 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__28_value;
static const lean_string_object lp_mathlib_BigOperators_bigOpBindersProd___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Unit"};
static const lean_object* lp_mathlib_BigOperators_bigOpBindersProd___closed__29 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__29_value;
static lean_once_cell_t lp_mathlib_BigOperators_bigOpBindersProd___closed__30_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_BigOperators_bigOpBindersProd___closed__30;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBindersProd___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__29_value),LEAN_SCALAR_PTR_LITERAL(230, 84, 106, 234, 91, 210, 120, 136)}};
static const lean_object* lp_mathlib_BigOperators_bigOpBindersProd___closed__31 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__31_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBindersProd___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__31_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_BigOperators_bigOpBindersProd___closed__32 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__32_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBindersProd___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__31_value)}};
static const lean_object* lp_mathlib_BigOperators_bigOpBindersProd___closed__33 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__33_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBindersProd___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__33_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_BigOperators_bigOpBindersProd___closed__34 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__34_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigOpBindersProd___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__32_value),((lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__34_value)}};
static const lean_object* lp_mathlib_BigOperators_bigOpBindersProd___closed__35 = (const lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__35_value;
LEAN_EXPORT lean_object* lp_mathlib_BigOperators_bigOpBindersProd(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BigOperators_bigOpBindersProd___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_BigOperators_BigOpWith___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "BigOpWith"};
static const lean_object* lp_mathlib_BigOperators_BigOpWith___closed__0 = (const lean_object*)&lp_mathlib_BigOperators_BigOpWith___closed__0_value;
static const lean_ctor_object lp_mathlib_BigOperators_BigOpWith___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__1_value),LEAN_SCALAR_PTR_LITERAL(107, 249, 74, 90, 68, 148, 111, 165)}};
static const lean_ctor_object lp_mathlib_BigOperators_BigOpWith___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_BigOpWith___closed__1_value_aux_0),((lean_object*)&lp_mathlib_BigOperators_BigOpWith___closed__0_value),LEAN_SCALAR_PTR_LITERAL(67, 151, 24, 172, 252, 223, 95, 49)}};
static const lean_object* lp_mathlib_BigOperators_BigOpWith___closed__1 = (const lean_object*)&lp_mathlib_BigOperators_BigOpWith___closed__1_value;
static const lean_string_object lp_mathlib_BigOperators_BigOpWith___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = " with "};
static const lean_object* lp_mathlib_BigOperators_BigOpWith___closed__2 = (const lean_object*)&lp_mathlib_BigOperators_BigOpWith___closed__2_value;
static const lean_ctor_object lp_mathlib_BigOperators_BigOpWith___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_BigOpWith___closed__2_value)}};
static const lean_object* lp_mathlib_BigOperators_BigOpWith___closed__3 = (const lean_object*)&lp_mathlib_BigOperators_BigOpWith___closed__3_value;
static const lean_string_object lp_mathlib_BigOperators_BigOpWith___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "atomic"};
static const lean_object* lp_mathlib_BigOperators_BigOpWith___closed__4 = (const lean_object*)&lp_mathlib_BigOperators_BigOpWith___closed__4_value;
static const lean_ctor_object lp_mathlib_BigOperators_BigOpWith___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_BigOpWith___closed__4_value),LEAN_SCALAR_PTR_LITERAL(56, 145, 113, 208, 127, 167, 216, 55)}};
static const lean_object* lp_mathlib_BigOperators_BigOpWith___closed__5 = (const lean_object*)&lp_mathlib_BigOperators_BigOpWith___closed__5_value;
static lean_once_cell_t lp_mathlib_BigOperators_BigOpWith___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_BigOperators_BigOpWith___closed__6;
static lean_once_cell_t lp_mathlib_BigOperators_BigOpWith___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_BigOperators_BigOpWith___closed__7;
static lean_once_cell_t lp_mathlib_BigOperators_BigOpWith___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_BigOperators_BigOpWith___closed__8;
static lean_once_cell_t lp_mathlib_BigOperators_BigOpWith___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_BigOperators_BigOpWith___closed__9;
static lean_once_cell_t lp_mathlib_BigOperators_BigOpWith___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_BigOperators_BigOpWith___closed__10;
static lean_once_cell_t lp_mathlib_BigOperators_BigOpWith___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_BigOperators_BigOpWith___closed__11;
LEAN_EXPORT lean_object* lp_mathlib_BigOperators_BigOpWith;
static const lean_string_object lp_mathlib_BigOperators_bigsum___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "bigsum"};
static const lean_object* lp_mathlib_BigOperators_bigsum___closed__0 = (const lean_object*)&lp_mathlib_BigOperators_bigsum___closed__0_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigsum___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__1_value),LEAN_SCALAR_PTR_LITERAL(107, 249, 74, 90, 68, 148, 111, 165)}};
static const lean_ctor_object lp_mathlib_BigOperators_bigsum___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigsum___closed__1_value_aux_0),((lean_object*)&lp_mathlib_BigOperators_bigsum___closed__0_value),LEAN_SCALAR_PTR_LITERAL(184, 251, 181, 230, 144, 247, 14, 157)}};
static const lean_object* lp_mathlib_BigOperators_bigsum___closed__1 = (const lean_object*)&lp_mathlib_BigOperators_bigsum___closed__1_value;
static const lean_string_object lp_mathlib_BigOperators_bigsum___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 2, .m_data = "∑ "};
static const lean_object* lp_mathlib_BigOperators_bigsum___closed__2 = (const lean_object*)&lp_mathlib_BigOperators_bigsum___closed__2_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigsum___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigsum___closed__2_value)}};
static const lean_object* lp_mathlib_BigOperators_bigsum___closed__3 = (const lean_object*)&lp_mathlib_BigOperators_bigsum___closed__3_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigsum___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__4_value),((lean_object*)&lp_mathlib_BigOperators_bigsum___closed__3_value),((lean_object*)&lp_mathlib_BigOperators_bigOpBinders___closed__7_value)}};
static const lean_object* lp_mathlib_BigOperators_bigsum___closed__4 = (const lean_object*)&lp_mathlib_BigOperators_bigsum___closed__4_value;
static lean_once_cell_t lp_mathlib_BigOperators_bigsum___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_BigOperators_bigsum___closed__5;
static lean_once_cell_t lp_mathlib_BigOperators_bigsum___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_BigOperators_bigsum___closed__6;
static const lean_string_object lp_mathlib_BigOperators_bigsum___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ", "};
static const lean_object* lp_mathlib_BigOperators_bigsum___closed__7 = (const lean_object*)&lp_mathlib_BigOperators_bigsum___closed__7_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigsum___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigsum___closed__7_value)}};
static const lean_object* lp_mathlib_BigOperators_bigsum___closed__8 = (const lean_object*)&lp_mathlib_BigOperators_bigsum___closed__8_value;
static lean_once_cell_t lp_mathlib_BigOperators_bigsum___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_BigOperators_bigsum___closed__9;
static const lean_ctor_object lp_mathlib_BigOperators_bigsum___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__6_value),((lean_object*)(((size_t)(67) << 1) | 1))}};
static const lean_object* lp_mathlib_BigOperators_bigsum___closed__10 = (const lean_object*)&lp_mathlib_BigOperators_bigsum___closed__10_value;
static lean_once_cell_t lp_mathlib_BigOperators_bigsum___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_BigOperators_bigsum___closed__11;
static lean_once_cell_t lp_mathlib_BigOperators_bigsum___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_BigOperators_bigsum___closed__12;
LEAN_EXPORT lean_object* lp_mathlib_BigOperators_bigsum;
static const lean_string_object lp_mathlib_BigOperators_bigprod___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "bigprod"};
static const lean_object* lp_mathlib_BigOperators_bigprod___closed__0 = (const lean_object*)&lp_mathlib_BigOperators_bigprod___closed__0_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigprod___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__1_value),LEAN_SCALAR_PTR_LITERAL(107, 249, 74, 90, 68, 148, 111, 165)}};
static const lean_ctor_object lp_mathlib_BigOperators_bigprod___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigprod___closed__1_value_aux_0),((lean_object*)&lp_mathlib_BigOperators_bigprod___closed__0_value),LEAN_SCALAR_PTR_LITERAL(53, 2, 176, 97, 1, 47, 45, 254)}};
static const lean_object* lp_mathlib_BigOperators_bigprod___closed__1 = (const lean_object*)&lp_mathlib_BigOperators_bigprod___closed__1_value;
static const lean_string_object lp_mathlib_BigOperators_bigprod___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 2, .m_data = "∏ "};
static const lean_object* lp_mathlib_BigOperators_bigprod___closed__2 = (const lean_object*)&lp_mathlib_BigOperators_bigprod___closed__2_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigprod___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigprod___closed__2_value)}};
static const lean_object* lp_mathlib_BigOperators_bigprod___closed__3 = (const lean_object*)&lp_mathlib_BigOperators_bigprod___closed__3_value;
static const lean_ctor_object lp_mathlib_BigOperators_bigprod___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigOpBinder___closed__4_value),((lean_object*)&lp_mathlib_BigOperators_bigprod___closed__3_value),((lean_object*)&lp_mathlib_BigOperators_bigOpBinders___closed__7_value)}};
static const lean_object* lp_mathlib_BigOperators_bigprod___closed__4 = (const lean_object*)&lp_mathlib_BigOperators_bigprod___closed__4_value;
static lean_once_cell_t lp_mathlib_BigOperators_bigprod___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_BigOperators_bigprod___closed__5;
static lean_once_cell_t lp_mathlib_BigOperators_bigprod___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_BigOperators_bigprod___closed__6;
static lean_once_cell_t lp_mathlib_BigOperators_bigprod___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_BigOperators_bigprod___closed__7;
static lean_once_cell_t lp_mathlib_BigOperators_bigprod___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_BigOperators_bigprod___closed__8;
LEAN_EXPORT lean_object* lp_mathlib_BigOperators_bigprod;
static const lean_string_object lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "Finset.sum"};
static const lean_object* lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__0 = (const lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__0_value;
static lean_once_cell_t lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__1;
static const lean_string_object lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "sum"};
static const lean_object* lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__2 = (const lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__2_value;
static const lean_ctor_object lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__21_value),LEAN_SCALAR_PTR_LITERAL(87, 75, 221, 45, 221, 79, 84, 42)}};
static const lean_ctor_object lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__3_value_aux_0),((lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(102, 91, 243, 213, 138, 7, 6, 233)}};
static const lean_object* lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__3 = (const lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__3_value;
static const lean_ctor_object lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__4 = (const lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__4_value;
static const lean_ctor_object lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__5 = (const lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__5_value;
static const lean_ctor_object lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__16_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__6 = (const lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__6_value;
static const lean_ctor_object lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__13_value),((lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__6_value)}};
static const lean_object* lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__7 = (const lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__7_value;
static const lean_ctor_object lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__10_value),((lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__7_value)}};
static const lean_object* lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__8 = (const lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__8_value;
static const lean_ctor_object lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__6_value),((lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__8_value)}};
static const lean_object* lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__9 = (const lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__9_value;
static const lean_ctor_object lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__4_value),((lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__9_value)}};
static const lean_object* lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__10 = (const lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__10_value;
static const lean_ctor_object lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators_bigOpBindersProd___closed__1_value),((lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__10_value)}};
static const lean_object* lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__11 = (const lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__11_value;
static const lean_string_object lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "fun"};
static const lean_object* lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__12 = (const lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__12_value;
static const lean_ctor_object lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__13_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__13_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__13_value_aux_0),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__15_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__13_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__13_value_aux_1),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__16_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__13_value_aux_2),((lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(249, 155, 133, 242, 71, 132, 191, 97)}};
static const lean_object* lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__13 = (const lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__13_value;
static const lean_string_object lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "basicFun"};
static const lean_object* lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__14 = (const lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__14_value;
static const lean_ctor_object lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__15_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__15_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__15_value_aux_0),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__15_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__15_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__15_value_aux_1),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__16_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__15_value_aux_2),((lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(209, 134, 40, 160, 122, 195, 31, 223)}};
static const lean_object* lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__15 = (const lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__15_value;
static const lean_string_object lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "↦"};
static const lean_object* lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__16 = (const lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__16_value;
static const lean_string_object lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "Finset.filter"};
static const lean_object* lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__17 = (const lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__17_value;
static lean_once_cell_t lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__18;
static const lean_string_object lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "filter"};
static const lean_object* lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__19 = (const lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__19_value;
static const lean_ctor_object lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__20_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__21_value),LEAN_SCALAR_PTR_LITERAL(87, 75, 221, 45, 221, 79, 84, 42)}};
static const lean_ctor_object lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__20_value_aux_0),((lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__19_value),LEAN_SCALAR_PTR_LITERAL(88, 243, 224, 152, 142, 113, 169, 220)}};
static const lean_object* lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__20 = (const lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__20_value;
static const lean_ctor_object lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__20_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__21 = (const lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__21_value;
static const lean_ctor_object lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__21_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__22 = (const lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__22_value;
static const lean_string_object lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "termDepIfThenElse"};
static const lean_object* lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__23 = (const lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__23_value;
static const lean_ctor_object lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__23_value),LEAN_SCALAR_PTR_LITERAL(191, 94, 17, 11, 145, 108, 236, 173)}};
static const lean_object* lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__24 = (const lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__24_value;
static const lean_string_object lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "if"};
static const lean_object* lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__25 = (const lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__25_value;
static const lean_string_object lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "then"};
static const lean_object* lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__26 = (const lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__26_value;
static const lean_string_object lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "else"};
static const lean_object* lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__27 = (const lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__27_value;
static const lean_string_object lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "num"};
static const lean_object* lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__28 = (const lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__28_value;
static const lean_ctor_object lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__28_value),LEAN_SCALAR_PTR_LITERAL(227, 68, 22, 222, 47, 51, 204, 84)}};
static const lean_object* lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__29 = (const lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__29_value;
static const lean_string_object lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "0"};
static const lean_object* lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__30 = (const lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__30_value;
static const lean_string_object lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "binderIdent"};
static const lean_object* lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__31 = (const lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__31_value;
static const lean_ctor_object lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__32_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__32_value_aux_0),((lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__31_value),LEAN_SCALAR_PTR_LITERAL(37, 194, 68, 106, 254, 181, 31, 191)}};
static const lean_object* lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__32 = (const lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__32_value;
LEAN_EXPORT lean_object* lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigprod__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "Finset.prod"};
static const lean_object* lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigprod__1___closed__0 = (const lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigprod__1___closed__0_value;
static lean_once_cell_t lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigprod__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigprod__1___closed__1;
static const lean_string_object lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigprod__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "prod"};
static const lean_object* lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigprod__1___closed__2 = (const lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigprod__1___closed__2_value;
static const lean_ctor_object lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigprod__1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_processBigOpBinder___closed__21_value),LEAN_SCALAR_PTR_LITERAL(87, 75, 221, 45, 221, 79, 84, 42)}};
static const lean_ctor_object lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigprod__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigprod__1___closed__3_value_aux_0),((lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigprod__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(247, 66, 46, 56, 151, 61, 191, 120)}};
static const lean_object* lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigprod__1___closed__3 = (const lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigprod__1___closed__3_value;
static const lean_ctor_object lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigprod__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigprod__1___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigprod__1___closed__4 = (const lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigprod__1___closed__4_value;
static const lean_ctor_object lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigprod__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigprod__1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigprod__1___closed__5 = (const lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigprod__1___closed__5_value;
static const lean_string_object lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigprod__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "1"};
static const lean_object* lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigprod__1___closed__6 = (const lean_object*)&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigprod__1___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigprod__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigprod__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_FinsetResult_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_FinsetResult_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_FinsetResult_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_FinsetResult_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_FinsetResult_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_FinsetResult_finset_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_FinsetResult_finset_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_FinsetResult_univ_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_FinsetResult_univ_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_FinsetResult_Iio_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_FinsetResult_Iio_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_FinsetResult_Iic_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_FinsetResult_Iic_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_FinsetResult_Ioi_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_FinsetResult_Ioi_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_FinsetResult_Ici_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_FinsetResult_Ici_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__1_spec__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__1___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__1___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_delab___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0_spec__0_spec__2___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0_spec__0_spec__2___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0_spec__0_spec__2___redArg(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0_spec__0_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0_spec__0_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0_spec__0___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0___redArg___lam__1___boxed, .m_arity = 8, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0_spec__0_spec__2(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BigOperators_delabFinsetProd___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BigOperators_delabFinsetProd___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00BigOperators_delabFinsetProd_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00BigOperators_delabFinsetProd_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "∏"};
static const lean_object* lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__0 = (const lean_object*)&lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__0_value;
static const lean_array_object lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__1 = (const lean_object*)&lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__1_value;
static const lean_string_object lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "∈"};
static const lean_object* lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__2 = (const lean_object*)&lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__2_value;
static const lean_string_object lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "<"};
static const lean_object* lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__3 = (const lean_object*)&lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__3_value;
static const lean_string_object lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "≤"};
static const lean_object* lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__4 = (const lean_object*)&lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__4_value;
static const lean_string_object lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ">"};
static const lean_object* lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__5 = (const lean_object*)&lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__5_value;
static const lean_string_object lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "≥"};
static const lean_object* lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__6 = (const lean_object*)&lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__6_value;
static const lean_closure_object lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_getPPFunBinderTypes___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__7 = (const lean_object*)&lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__7_value;
static const lean_closure_object lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_getPPOption___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__7_value)} };
static const lean_object* lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__8 = (const lean_object*)&lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__8_value;
static const lean_string_object lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "with"};
static const lean_object* lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__9 = (const lean_object*)&lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__9_value;
LEAN_EXPORT lean_object* lp_mathlib_BigOperators_delabFinsetProd___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BigOperators_delabFinsetProd___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_BigOperators_delabFinsetProd___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_BigOperators_delabFinsetProd___lam__0___boxed, .m_arity = 8, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_BigOperators_delabFinsetProd___closed__0 = (const lean_object*)&lp_mathlib_BigOperators_delabFinsetProd___closed__0_value;
static const lean_closure_object lp_mathlib_BigOperators_delabFinsetProd___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_getPPNotation___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_BigOperators_delabFinsetProd___closed__1 = (const lean_object*)&lp_mathlib_BigOperators_delabFinsetProd___closed__1_value;
static const lean_closure_object lp_mathlib_BigOperators_delabFinsetProd___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_BigOperators_delabFinsetProd___lam__1___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(5) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_delabFinsetProd___closed__0_value)} };
static const lean_object* lp_mathlib_BigOperators_delabFinsetProd___closed__2 = (const lean_object*)&lp_mathlib_BigOperators_delabFinsetProd___closed__2_value;
static const lean_closure_object lp_mathlib_BigOperators_delabFinsetProd___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_withOverApp___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(5) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_delabFinsetProd___closed__2_value)} };
static const lean_object* lp_mathlib_BigOperators_delabFinsetProd___closed__3 = (const lean_object*)&lp_mathlib_BigOperators_delabFinsetProd___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_BigOperators_delabFinsetProd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BigOperators_delabFinsetProd___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00BigOperators_delabFinsetProd_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00BigOperators_delabFinsetProd_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_BigOperators_delabFinsetSum___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "∑"};
static const lean_object* lp_mathlib_BigOperators_delabFinsetSum___lam__1___closed__0 = (const lean_object*)&lp_mathlib_BigOperators_delabFinsetSum___lam__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_BigOperators_delabFinsetSum___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BigOperators_delabFinsetSum___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_BigOperators_delabFinsetSum___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_BigOperators_delabFinsetSum___lam__1___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(5) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_delabFinsetProd___closed__0_value)} };
static const lean_object* lp_mathlib_BigOperators_delabFinsetSum___closed__0 = (const lean_object*)&lp_mathlib_BigOperators_delabFinsetSum___closed__0_value;
static const lean_closure_object lp_mathlib_BigOperators_delabFinsetSum___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_withOverApp___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(5) << 1) | 1)),((lean_object*)&lp_mathlib_BigOperators_delabFinsetSum___closed__0_value)} };
static const lean_object* lp_mathlib_BigOperators_delabFinsetSum___closed__1 = (const lean_object*)&lp_mathlib_BigOperators_delabFinsetSum___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_BigOperators_delabFinsetSum(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BigOperators_delabFinsetSum___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_prod___redArg(lean_object* v_inst_1_, lean_object* v_s_2_, lean_object* v_f_3_){
_start:
{
lean_object* v___x_4_; lean_object* v___x_5_; 
v___x_4_ = lp_mathlib_Multiset_map___redArg(v_f_3_, v_s_2_);
v___x_5_ = lp_mathlib_Multiset_prod___redArg(v_inst_1_, v___x_4_);
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_prod___redArg___boxed(lean_object* v_inst_6_, lean_object* v_s_7_, lean_object* v_f_8_){
_start:
{
lean_object* v_res_9_; 
v_res_9_ = lp_mathlib_Finset_prod___redArg(v_inst_6_, v_s_7_, v_f_8_);
lean_dec_ref(v_inst_6_);
return v_res_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_prod(lean_object* v_00_u03b9_10_, lean_object* v_M_11_, lean_object* v_inst_12_, lean_object* v_s_13_, lean_object* v_f_14_){
_start:
{
lean_object* v___x_15_; 
v___x_15_ = lp_mathlib_Finset_prod___redArg(v_inst_12_, v_s_13_, v_f_14_);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_prod___boxed(lean_object* v_00_u03b9_16_, lean_object* v_M_17_, lean_object* v_inst_18_, lean_object* v_s_19_, lean_object* v_f_20_){
_start:
{
lean_object* v_res_21_; 
v_res_21_ = lp_mathlib_Finset_prod(v_00_u03b9_16_, v_M_17_, v_inst_18_, v_s_19_, v_f_20_);
lean_dec_ref(v_inst_18_);
return v_res_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_sum___redArg(lean_object* v_inst_22_, lean_object* v_s_23_, lean_object* v_f_24_){
_start:
{
lean_object* v___x_25_; lean_object* v___x_26_; 
v___x_25_ = lp_mathlib_Multiset_map___redArg(v_f_24_, v_s_23_);
v___x_26_ = lp_mathlib_Multiset_sum___redArg(v_inst_22_, v___x_25_);
return v___x_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_sum___redArg___boxed(lean_object* v_inst_27_, lean_object* v_s_28_, lean_object* v_f_29_){
_start:
{
lean_object* v_res_30_; 
v_res_30_ = lp_mathlib_Finset_sum___redArg(v_inst_27_, v_s_28_, v_f_29_);
lean_dec_ref(v_inst_27_);
return v_res_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_sum(lean_object* v_00_u03b9_31_, lean_object* v_M_32_, lean_object* v_inst_33_, lean_object* v_s_34_, lean_object* v_f_35_){
_start:
{
lean_object* v___x_36_; 
v___x_36_ = lp_mathlib_Finset_sum___redArg(v_inst_33_, v_s_34_, v_f_35_);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_sum___boxed(lean_object* v_00_u03b9_37_, lean_object* v_M_38_, lean_object* v_inst_39_, lean_object* v_s_40_, lean_object* v_f_41_){
_start:
{
lean_object* v_res_42_; 
v_res_42_ = lp_mathlib_Finset_sum(v_00_u03b9_37_, v_M_38_, v_inst_39_, v_s_40_, v_f_41_);
lean_dec_ref(v_inst_39_);
return v_res_42_;
}
}
static lean_object* _init_lp_mathlib_LibraryNote_operator__precedence__of__big__operators(void){
_start:
{
lean_object* v___x_43_; 
v___x_43_ = lean_box(0);
return v___x_43_;
}
}
static lean_object* _init_lp_mathlib_BigOperators_processBigOpBinder___closed__20(void){
_start:
{
lean_object* v___x_200_; lean_object* v___x_201_; 
v___x_200_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__19));
v___x_201_ = l_String_toRawSubstring_x27(v___x_200_);
return v___x_201_;
}
}
static lean_object* _init_lp_mathlib_BigOperators_processBigOpBinder___closed__27(void){
_start:
{
lean_object* v___x_211_; lean_object* v___x_212_; 
v___x_211_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__26));
v___x_212_ = l_String_toRawSubstring_x27(v___x_211_);
return v___x_212_;
}
}
static lean_object* _init_lp_mathlib_BigOperators_processBigOpBinder___closed__31(void){
_start:
{
lean_object* v___x_218_; lean_object* v___x_219_; 
v___x_218_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__30));
v___x_219_ = l_String_toRawSubstring_x27(v___x_218_);
return v___x_219_;
}
}
static lean_object* _init_lp_mathlib_BigOperators_processBigOpBinder___closed__35(void){
_start:
{
lean_object* v___x_225_; lean_object* v___x_226_; 
v___x_225_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__34));
v___x_226_ = l_String_toRawSubstring_x27(v___x_225_);
return v___x_226_;
}
}
static lean_object* _init_lp_mathlib_BigOperators_processBigOpBinder___closed__39(void){
_start:
{
lean_object* v___x_232_; lean_object* v___x_233_; 
v___x_232_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__38));
v___x_233_ = l_String_toRawSubstring_x27(v___x_232_);
return v___x_233_;
}
}
static lean_object* _init_lp_mathlib_BigOperators_processBigOpBinder___closed__57(void){
_start:
{
lean_object* v___x_265_; lean_object* v___x_266_; 
v___x_265_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__56));
v___x_266_ = l_String_toRawSubstring_x27(v___x_265_);
return v___x_266_;
}
}
static lean_object* _init_lp_mathlib_BigOperators_processBigOpBinder___closed__59(void){
_start:
{
lean_object* v___x_268_; lean_object* v___x_269_; 
v___x_268_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__58));
v___x_269_ = l_String_toRawSubstring_x27(v___x_268_);
return v___x_269_;
}
}
static lean_object* _init_lp_mathlib_BigOperators_processBigOpBinder___closed__62(void){
_start:
{
lean_object* v___x_274_; lean_object* v___x_275_; 
v___x_274_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__21));
v___x_275_ = l_String_toRawSubstring_x27(v___x_274_);
return v___x_275_;
}
}
static lean_object* _init_lp_mathlib_BigOperators_processBigOpBinder___closed__76(void){
_start:
{
lean_object* v___x_300_; lean_object* v___x_301_; 
v___x_300_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__75));
v___x_301_ = l_String_toRawSubstring_x27(v___x_300_);
return v___x_301_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BigOperators_processBigOpBinder(lean_object* v_processed_308_, lean_object* v_binder_309_, lean_object* v_a_310_, lean_object* v_a_311_){
_start:
{
lean_object* v_ref_312_; lean_object* v___x_313_; uint8_t v___x_314_; 
v_ref_312_ = lean_ctor_get(v_a_310_, 5);
v___x_313_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBinder___closed__2));
lean_inc(v_binder_309_);
v___x_314_ = l_Lean_Syntax_isOfKind(v_binder_309_, v___x_313_);
if (v___x_314_ == 0)
{
lean_object* v___x_315_; 
lean_dec(v_binder_309_);
lean_dec_ref(v_processed_308_);
v___x_315_ = l_Lean_Macro_throwUnsupported___redArg(v_a_311_);
return v___x_315_;
}
else
{
lean_object* v_ref_316_; lean_object* v___x_317_; lean_object* v_x_318_; lean_object* v___x_319_; lean_object* v___x_320_; uint8_t v___x_321_; 
v_ref_316_ = l_Lean_replaceRef(v_binder_309_, v_ref_312_);
v___x_317_ = lean_unsigned_to_nat(0u);
v_x_318_ = l_Lean_Syntax_getArg(v_binder_309_, v___x_317_);
v___x_319_ = lean_unsigned_to_nat(1u);
v___x_320_ = l_Lean_Syntax_getArg(v_binder_309_, v___x_319_);
lean_dec(v_binder_309_);
lean_inc(v___x_320_);
v___x_321_ = l_Lean_Syntax_matchesNull(v___x_320_, v___x_317_);
if (v___x_321_ == 0)
{
uint8_t v___x_322_; 
lean_inc(v___x_320_);
v___x_322_ = l_Lean_Syntax_matchesNull(v___x_320_, v___x_319_);
if (v___x_322_ == 0)
{
lean_object* v___x_323_; 
lean_dec(v___x_320_);
lean_dec(v_x_318_);
lean_dec(v_ref_316_);
lean_dec_ref(v_processed_308_);
v___x_323_ = l_Lean_Macro_throwUnsupported___redArg(v_a_311_);
return v___x_323_;
}
else
{
lean_object* v___x_324_; lean_object* v___x_325_; uint8_t v___x_326_; 
v___x_324_ = l_Lean_Syntax_getArg(v___x_320_, v___x_317_);
lean_dec(v___x_320_);
v___x_325_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBinder___closed__13));
lean_inc(v___x_324_);
v___x_326_ = l_Lean_Syntax_isOfKind(v___x_324_, v___x_325_);
if (v___x_326_ == 0)
{
lean_object* v___x_327_; uint8_t v___x_328_; 
v___x_327_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__2));
lean_inc(v___x_324_);
v___x_328_ = l_Lean_Syntax_isOfKind(v___x_324_, v___x_327_);
if (v___x_328_ == 0)
{
lean_object* v___x_329_; uint8_t v___x_330_; 
v___x_329_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__4));
lean_inc(v___x_324_);
v___x_330_ = l_Lean_Syntax_isOfKind(v___x_324_, v___x_329_);
if (v___x_330_ == 0)
{
lean_object* v___x_331_; uint8_t v___x_332_; 
v___x_331_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__6));
lean_inc(v___x_324_);
v___x_332_ = l_Lean_Syntax_isOfKind(v___x_324_, v___x_331_);
if (v___x_332_ == 0)
{
lean_object* v___x_333_; uint8_t v___x_334_; 
v___x_333_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__8));
lean_inc(v___x_324_);
v___x_334_ = l_Lean_Syntax_isOfKind(v___x_324_, v___x_333_);
if (v___x_334_ == 0)
{
lean_object* v___x_335_; uint8_t v___x_336_; 
v___x_335_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__10));
lean_inc(v___x_324_);
v___x_336_ = l_Lean_Syntax_isOfKind(v___x_324_, v___x_335_);
if (v___x_336_ == 0)
{
lean_object* v___x_337_; uint8_t v___x_338_; 
v___x_337_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__12));
lean_inc(v___x_324_);
v___x_338_ = l_Lean_Syntax_isOfKind(v___x_324_, v___x_337_);
if (v___x_338_ == 0)
{
lean_object* v___x_339_; uint8_t v___x_340_; 
v___x_339_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__14));
lean_inc(v___x_324_);
v___x_340_ = l_Lean_Syntax_isOfKind(v___x_324_, v___x_339_);
if (v___x_340_ == 0)
{
lean_object* v___x_341_; 
lean_dec(v___x_324_);
lean_dec(v_x_318_);
lean_dec(v_ref_316_);
lean_dec_ref(v_processed_308_);
v___x_341_ = l_Lean_Macro_throwUnsupported___redArg(v_a_311_);
return v___x_341_;
}
else
{
lean_object* v___x_342_; lean_object* v___x_343_; lean_object* v___x_344_; lean_object* v___x_345_; lean_object* v___x_346_; lean_object* v___x_347_; lean_object* v___x_348_; lean_object* v___x_349_; lean_object* v___x_350_; lean_object* v___x_351_; lean_object* v___x_352_; lean_object* v___x_353_; lean_object* v___x_354_; 
v___x_342_ = l_Lean_Syntax_getArg(v___x_324_, v___x_319_);
lean_dec(v___x_324_);
v___x_343_ = l_Lean_SourceInfo_fromRef(v_ref_316_, v___x_338_);
lean_dec(v_ref_316_);
v___x_344_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__18));
v___x_345_ = lean_obj_once(&lp_mathlib_BigOperators_processBigOpBinder___closed__20, &lp_mathlib_BigOperators_processBigOpBinder___closed__20_once, _init_lp_mathlib_BigOperators_processBigOpBinder___closed__20);
v___x_346_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__23));
v___x_347_ = lean_box(0);
lean_inc_n(v___x_343_, 2);
v___x_348_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_348_, 0, v___x_343_);
lean_ctor_set(v___x_348_, 1, v___x_345_);
lean_ctor_set(v___x_348_, 2, v___x_346_);
lean_ctor_set(v___x_348_, 3, v___x_347_);
v___x_349_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__25));
v___x_350_ = l_Lean_Syntax_node1(v___x_343_, v___x_349_, v___x_342_);
v___x_351_ = l_Lean_Syntax_node2(v___x_343_, v___x_344_, v___x_348_, v___x_350_);
v___x_352_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_352_, 0, v_x_318_);
lean_ctor_set(v___x_352_, 1, v___x_351_);
v___x_353_ = lean_array_push(v_processed_308_, v___x_352_);
v___x_354_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_354_, 0, v___x_353_);
lean_ctor_set(v___x_354_, 1, v_a_311_);
return v___x_354_;
}
}
else
{
lean_object* v___x_355_; lean_object* v___x_356_; lean_object* v___x_357_; lean_object* v___x_358_; lean_object* v___x_359_; lean_object* v___x_360_; lean_object* v___x_361_; lean_object* v___x_362_; lean_object* v___x_363_; lean_object* v___x_364_; lean_object* v___x_365_; lean_object* v___x_366_; lean_object* v___x_367_; 
v___x_355_ = l_Lean_Syntax_getArg(v___x_324_, v___x_319_);
lean_dec(v___x_324_);
v___x_356_ = l_Lean_SourceInfo_fromRef(v_ref_316_, v___x_336_);
lean_dec(v_ref_316_);
v___x_357_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__18));
v___x_358_ = lean_obj_once(&lp_mathlib_BigOperators_processBigOpBinder___closed__27, &lp_mathlib_BigOperators_processBigOpBinder___closed__27_once, _init_lp_mathlib_BigOperators_processBigOpBinder___closed__27);
v___x_359_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__29));
v___x_360_ = lean_box(0);
lean_inc_n(v___x_356_, 2);
v___x_361_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_361_, 0, v___x_356_);
lean_ctor_set(v___x_361_, 1, v___x_358_);
lean_ctor_set(v___x_361_, 2, v___x_359_);
lean_ctor_set(v___x_361_, 3, v___x_360_);
v___x_362_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__25));
v___x_363_ = l_Lean_Syntax_node1(v___x_356_, v___x_362_, v___x_355_);
v___x_364_ = l_Lean_Syntax_node2(v___x_356_, v___x_357_, v___x_361_, v___x_363_);
v___x_365_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_365_, 0, v_x_318_);
lean_ctor_set(v___x_365_, 1, v___x_364_);
v___x_366_ = lean_array_push(v_processed_308_, v___x_365_);
v___x_367_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_367_, 0, v___x_366_);
lean_ctor_set(v___x_367_, 1, v_a_311_);
return v___x_367_;
}
}
else
{
lean_object* v___x_368_; lean_object* v___x_369_; lean_object* v___x_370_; lean_object* v___x_371_; lean_object* v___x_372_; lean_object* v___x_373_; lean_object* v___x_374_; lean_object* v___x_375_; lean_object* v___x_376_; lean_object* v___x_377_; lean_object* v___x_378_; lean_object* v___x_379_; lean_object* v___x_380_; 
v___x_368_ = l_Lean_Syntax_getArg(v___x_324_, v___x_319_);
lean_dec(v___x_324_);
v___x_369_ = l_Lean_SourceInfo_fromRef(v_ref_316_, v___x_334_);
lean_dec(v_ref_316_);
v___x_370_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__18));
v___x_371_ = lean_obj_once(&lp_mathlib_BigOperators_processBigOpBinder___closed__31, &lp_mathlib_BigOperators_processBigOpBinder___closed__31_once, _init_lp_mathlib_BigOperators_processBigOpBinder___closed__31);
v___x_372_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__33));
v___x_373_ = lean_box(0);
lean_inc_n(v___x_369_, 2);
v___x_374_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_374_, 0, v___x_369_);
lean_ctor_set(v___x_374_, 1, v___x_371_);
lean_ctor_set(v___x_374_, 2, v___x_372_);
lean_ctor_set(v___x_374_, 3, v___x_373_);
v___x_375_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__25));
v___x_376_ = l_Lean_Syntax_node1(v___x_369_, v___x_375_, v___x_368_);
v___x_377_ = l_Lean_Syntax_node2(v___x_369_, v___x_370_, v___x_374_, v___x_376_);
v___x_378_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_378_, 0, v_x_318_);
lean_ctor_set(v___x_378_, 1, v___x_377_);
v___x_379_ = lean_array_push(v_processed_308_, v___x_378_);
v___x_380_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_380_, 0, v___x_379_);
lean_ctor_set(v___x_380_, 1, v_a_311_);
return v___x_380_;
}
}
else
{
lean_object* v___x_381_; lean_object* v___x_382_; lean_object* v___x_383_; lean_object* v___x_384_; lean_object* v___x_385_; lean_object* v___x_386_; lean_object* v___x_387_; lean_object* v___x_388_; lean_object* v___x_389_; lean_object* v___x_390_; lean_object* v___x_391_; lean_object* v___x_392_; lean_object* v___x_393_; 
v___x_381_ = l_Lean_Syntax_getArg(v___x_324_, v___x_319_);
lean_dec(v___x_324_);
v___x_382_ = l_Lean_SourceInfo_fromRef(v_ref_316_, v___x_332_);
lean_dec(v_ref_316_);
v___x_383_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__18));
v___x_384_ = lean_obj_once(&lp_mathlib_BigOperators_processBigOpBinder___closed__35, &lp_mathlib_BigOperators_processBigOpBinder___closed__35_once, _init_lp_mathlib_BigOperators_processBigOpBinder___closed__35);
v___x_385_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__37));
v___x_386_ = lean_box(0);
lean_inc_n(v___x_382_, 2);
v___x_387_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_387_, 0, v___x_382_);
lean_ctor_set(v___x_387_, 1, v___x_384_);
lean_ctor_set(v___x_387_, 2, v___x_385_);
lean_ctor_set(v___x_387_, 3, v___x_386_);
v___x_388_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__25));
v___x_389_ = l_Lean_Syntax_node1(v___x_382_, v___x_388_, v___x_381_);
v___x_390_ = l_Lean_Syntax_node2(v___x_382_, v___x_383_, v___x_387_, v___x_389_);
v___x_391_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_391_, 0, v_x_318_);
lean_ctor_set(v___x_391_, 1, v___x_390_);
v___x_392_ = lean_array_push(v_processed_308_, v___x_391_);
v___x_393_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_393_, 0, v___x_392_);
lean_ctor_set(v___x_393_, 1, v_a_311_);
return v___x_393_;
}
}
else
{
lean_object* v___x_394_; lean_object* v___x_395_; lean_object* v___x_396_; lean_object* v___x_397_; lean_object* v___x_398_; lean_object* v___x_399_; lean_object* v___x_400_; lean_object* v___x_401_; lean_object* v___x_402_; lean_object* v___x_403_; lean_object* v___x_404_; lean_object* v___x_405_; lean_object* v___x_406_; 
v___x_394_ = l_Lean_Syntax_getArg(v___x_324_, v___x_319_);
lean_dec(v___x_324_);
v___x_395_ = l_Lean_SourceInfo_fromRef(v_ref_316_, v___x_330_);
lean_dec(v_ref_316_);
v___x_396_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__18));
v___x_397_ = lean_obj_once(&lp_mathlib_BigOperators_processBigOpBinder___closed__39, &lp_mathlib_BigOperators_processBigOpBinder___closed__39_once, _init_lp_mathlib_BigOperators_processBigOpBinder___closed__39);
v___x_398_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__42));
v___x_399_ = lean_box(0);
lean_inc_n(v___x_395_, 2);
v___x_400_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_400_, 0, v___x_395_);
lean_ctor_set(v___x_400_, 1, v___x_397_);
lean_ctor_set(v___x_400_, 2, v___x_398_);
lean_ctor_set(v___x_400_, 3, v___x_399_);
v___x_401_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__25));
v___x_402_ = l_Lean_Syntax_node1(v___x_395_, v___x_401_, v___x_394_);
v___x_403_ = l_Lean_Syntax_node2(v___x_395_, v___x_396_, v___x_400_, v___x_402_);
v___x_404_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_404_, 0, v_x_318_);
lean_ctor_set(v___x_404_, 1, v___x_403_);
v___x_405_ = lean_array_push(v_processed_308_, v___x_404_);
v___x_406_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_406_, 0, v___x_405_);
lean_ctor_set(v___x_406_, 1, v_a_311_);
return v___x_406_;
}
}
else
{
lean_object* v___x_407_; lean_object* v___x_408_; lean_object* v___x_409_; lean_object* v___x_410_; lean_object* v___x_411_; lean_object* v___x_412_; lean_object* v___x_413_; lean_object* v___x_414_; lean_object* v___x_415_; lean_object* v___x_416_; lean_object* v___x_417_; lean_object* v___x_418_; lean_object* v___x_419_; 
v___x_407_ = l_Lean_Syntax_getArg(v___x_324_, v___x_319_);
lean_dec(v___x_324_);
v___x_408_ = l_Lean_SourceInfo_fromRef(v_ref_316_, v___x_328_);
lean_dec(v_ref_316_);
v___x_409_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__44));
v___x_410_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__45));
lean_inc_n(v___x_408_, 3);
v___x_411_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_411_, 0, v___x_408_);
lean_ctor_set(v___x_411_, 1, v___x_410_);
v___x_412_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__47));
v___x_413_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__48));
v___x_414_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_414_, 0, v___x_408_);
lean_ctor_set(v___x_414_, 1, v___x_413_);
v___x_415_ = l_Lean_Syntax_node2(v___x_408_, v___x_412_, v___x_407_, v___x_414_);
v___x_416_ = l_Lean_Syntax_node2(v___x_408_, v___x_409_, v___x_411_, v___x_415_);
v___x_417_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_417_, 0, v_x_318_);
lean_ctor_set(v___x_417_, 1, v___x_416_);
v___x_418_ = lean_array_push(v_processed_308_, v___x_417_);
v___x_419_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_419_, 0, v___x_418_);
lean_ctor_set(v___x_419_, 1, v_a_311_);
return v___x_419_;
}
}
else
{
lean_object* v___x_420_; lean_object* v___x_421_; lean_object* v___x_422_; lean_object* v___x_423_; lean_object* v___x_424_; lean_object* v___x_425_; lean_object* v___x_426_; lean_object* v___x_427_; lean_object* v___x_428_; 
v___x_420_ = l_Lean_Syntax_getArg(v___x_324_, v___x_319_);
lean_dec(v___x_324_);
v___x_421_ = l_Lean_SourceInfo_fromRef(v_ref_316_, v___x_326_);
lean_dec(v_ref_316_);
v___x_422_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__44));
v___x_423_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__45));
lean_inc(v___x_421_);
v___x_424_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_424_, 0, v___x_421_);
lean_ctor_set(v___x_424_, 1, v___x_423_);
v___x_425_ = l_Lean_Syntax_node2(v___x_421_, v___x_422_, v___x_424_, v___x_420_);
v___x_426_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_426_, 0, v_x_318_);
lean_ctor_set(v___x_426_, 1, v___x_425_);
v___x_427_ = lean_array_push(v_processed_308_, v___x_426_);
v___x_428_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_428_, 0, v___x_427_);
lean_ctor_set(v___x_428_, 1, v_a_311_);
return v___x_428_;
}
}
else
{
lean_object* v___x_429_; lean_object* v___x_430_; lean_object* v___x_431_; lean_object* v___x_432_; lean_object* v___x_433_; lean_object* v___x_434_; lean_object* v___x_435_; lean_object* v___x_436_; lean_object* v___x_437_; lean_object* v___x_438_; lean_object* v___x_439_; lean_object* v___x_440_; lean_object* v___x_441_; lean_object* v___x_442_; lean_object* v___x_443_; lean_object* v___x_444_; lean_object* v___x_445_; lean_object* v___x_446_; lean_object* v___x_447_; lean_object* v___x_448_; lean_object* v___x_449_; lean_object* v___x_450_; lean_object* v___x_451_; lean_object* v___x_452_; lean_object* v___x_453_; lean_object* v___x_454_; lean_object* v___x_455_; lean_object* v___x_456_; lean_object* v___x_457_; lean_object* v___x_458_; lean_object* v___x_459_; lean_object* v___x_460_; 
v___x_429_ = l_Lean_Syntax_getArg(v___x_324_, v___x_319_);
lean_dec(v___x_324_);
v___x_430_ = l_Lean_SourceInfo_fromRef(v_ref_316_, v___x_321_);
lean_dec(v_ref_316_);
v___x_431_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__50));
v___x_432_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__52));
v___x_433_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__53));
lean_inc_n(v___x_430_, 11);
v___x_434_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_434_, 0, v___x_430_);
lean_ctor_set(v___x_434_, 1, v___x_433_);
v___x_435_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__55));
v___x_436_ = lean_obj_once(&lp_mathlib_BigOperators_processBigOpBinder___closed__57, &lp_mathlib_BigOperators_processBigOpBinder___closed__57_once, _init_lp_mathlib_BigOperators_processBigOpBinder___closed__57);
v___x_437_ = lean_box(0);
v___x_438_ = lean_box(0);
v___x_439_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_439_, 0, v___x_430_);
lean_ctor_set(v___x_439_, 1, v___x_436_);
lean_ctor_set(v___x_439_, 2, v___x_437_);
lean_ctor_set(v___x_439_, 3, v___x_438_);
v___x_440_ = l_Lean_Syntax_node1(v___x_430_, v___x_435_, v___x_439_);
v___x_441_ = l_Lean_Syntax_node2(v___x_430_, v___x_432_, v___x_434_, v___x_440_);
v___x_442_ = lean_obj_once(&lp_mathlib_BigOperators_processBigOpBinder___closed__59, &lp_mathlib_BigOperators_processBigOpBinder___closed__59_once, _init_lp_mathlib_BigOperators_processBigOpBinder___closed__59);
v___x_443_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__60));
v___x_444_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_444_, 0, v___x_430_);
lean_ctor_set(v___x_444_, 1, v___x_442_);
lean_ctor_set(v___x_444_, 2, v___x_443_);
lean_ctor_set(v___x_444_, 3, v___x_438_);
v___x_445_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__61));
v___x_446_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_446_, 0, v___x_430_);
lean_ctor_set(v___x_446_, 1, v___x_445_);
v___x_447_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__25));
v___x_448_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__18));
v___x_449_ = lean_obj_once(&lp_mathlib_BigOperators_processBigOpBinder___closed__62, &lp_mathlib_BigOperators_processBigOpBinder___closed__62_once, _init_lp_mathlib_BigOperators_processBigOpBinder___closed__62);
v___x_450_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__63));
v___x_451_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_451_, 0, v___x_430_);
lean_ctor_set(v___x_451_, 1, v___x_449_);
lean_ctor_set(v___x_451_, 2, v___x_450_);
lean_ctor_set(v___x_451_, 3, v___x_438_);
v___x_452_ = l_Lean_Syntax_node1(v___x_430_, v___x_447_, v___x_429_);
v___x_453_ = l_Lean_Syntax_node2(v___x_430_, v___x_448_, v___x_451_, v___x_452_);
v___x_454_ = l_Lean_Syntax_node1(v___x_430_, v___x_447_, v___x_453_);
v___x_455_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBinderParenthesized___closed__5));
v___x_456_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_456_, 0, v___x_430_);
lean_ctor_set(v___x_456_, 1, v___x_455_);
v___x_457_ = l_Lean_Syntax_node5(v___x_430_, v___x_431_, v___x_441_, v___x_444_, v___x_446_, v___x_454_, v___x_456_);
v___x_458_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_458_, 0, v_x_318_);
lean_ctor_set(v___x_458_, 1, v___x_457_);
v___x_459_ = lean_array_push(v_processed_308_, v___x_458_);
v___x_460_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_460_, 0, v___x_459_);
lean_ctor_set(v___x_460_, 1, v_a_311_);
return v___x_460_;
}
}
}
else
{
lean_object* v___x_461_; uint8_t v___x_462_; 
lean_dec(v___x_320_);
v___x_461_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__65));
lean_inc(v_x_318_);
v___x_462_ = l_Lean_Syntax_isOfKind(v_x_318_, v___x_461_);
if (v___x_462_ == 0)
{
lean_object* v___x_463_; lean_object* v___x_464_; lean_object* v___x_465_; lean_object* v___x_466_; lean_object* v___x_467_; lean_object* v___x_468_; lean_object* v___x_469_; lean_object* v___x_470_; 
v___x_463_ = l_Lean_SourceInfo_fromRef(v_ref_316_, v___x_462_);
lean_dec(v_ref_316_);
v___x_464_ = lean_obj_once(&lp_mathlib_BigOperators_processBigOpBinder___closed__59, &lp_mathlib_BigOperators_processBigOpBinder___closed__59_once, _init_lp_mathlib_BigOperators_processBigOpBinder___closed__59);
v___x_465_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__60));
v___x_466_ = lean_box(0);
v___x_467_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_467_, 0, v___x_463_);
lean_ctor_set(v___x_467_, 1, v___x_464_);
lean_ctor_set(v___x_467_, 2, v___x_465_);
lean_ctor_set(v___x_467_, 3, v___x_466_);
v___x_468_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_468_, 0, v_x_318_);
lean_ctor_set(v___x_468_, 1, v___x_467_);
v___x_469_ = lean_array_push(v_processed_308_, v___x_468_);
v___x_470_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_470_, 0, v___x_469_);
lean_ctor_set(v___x_470_, 1, v_a_311_);
return v___x_470_;
}
else
{
lean_object* v___x_471_; lean_object* v___x_472_; uint8_t v___x_473_; 
v___x_471_ = l_Lean_Syntax_getArg(v_x_318_, v___x_317_);
v___x_472_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__52));
lean_inc(v___x_471_);
v___x_473_ = l_Lean_Syntax_isOfKind(v___x_471_, v___x_472_);
if (v___x_473_ == 0)
{
lean_object* v___x_474_; lean_object* v___x_475_; lean_object* v___x_476_; lean_object* v___x_477_; lean_object* v___x_478_; lean_object* v___x_479_; lean_object* v___x_480_; lean_object* v___x_481_; 
lean_dec(v___x_471_);
v___x_474_ = l_Lean_SourceInfo_fromRef(v_ref_316_, v___x_473_);
lean_dec(v_ref_316_);
v___x_475_ = lean_obj_once(&lp_mathlib_BigOperators_processBigOpBinder___closed__59, &lp_mathlib_BigOperators_processBigOpBinder___closed__59_once, _init_lp_mathlib_BigOperators_processBigOpBinder___closed__59);
v___x_476_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__60));
v___x_477_ = lean_box(0);
v___x_478_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_478_, 0, v___x_474_);
lean_ctor_set(v___x_478_, 1, v___x_475_);
lean_ctor_set(v___x_478_, 2, v___x_476_);
lean_ctor_set(v___x_478_, 3, v___x_477_);
v___x_479_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_479_, 0, v_x_318_);
lean_ctor_set(v___x_479_, 1, v___x_478_);
v___x_480_ = lean_array_push(v_processed_308_, v___x_479_);
v___x_481_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_481_, 0, v___x_480_);
lean_ctor_set(v___x_481_, 1, v_a_311_);
return v___x_481_;
}
else
{
lean_object* v___x_482_; lean_object* v___x_483_; uint8_t v___x_484_; 
v___x_482_ = l_Lean_Syntax_getArg(v___x_471_, v___x_319_);
lean_dec(v___x_471_);
v___x_483_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__55));
lean_inc(v___x_482_);
v___x_484_ = l_Lean_Syntax_isOfKind(v___x_482_, v___x_483_);
if (v___x_484_ == 0)
{
lean_object* v___x_485_; lean_object* v___x_486_; lean_object* v___x_487_; lean_object* v___x_488_; lean_object* v___x_489_; lean_object* v___x_490_; lean_object* v___x_491_; lean_object* v___x_492_; 
lean_dec(v___x_482_);
v___x_485_ = l_Lean_SourceInfo_fromRef(v_ref_316_, v___x_484_);
lean_dec(v_ref_316_);
v___x_486_ = lean_obj_once(&lp_mathlib_BigOperators_processBigOpBinder___closed__59, &lp_mathlib_BigOperators_processBigOpBinder___closed__59_once, _init_lp_mathlib_BigOperators_processBigOpBinder___closed__59);
v___x_487_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__60));
v___x_488_ = lean_box(0);
v___x_489_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_489_, 0, v___x_485_);
lean_ctor_set(v___x_489_, 1, v___x_486_);
lean_ctor_set(v___x_489_, 2, v___x_487_);
lean_ctor_set(v___x_489_, 3, v___x_488_);
v___x_490_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_490_, 0, v_x_318_);
lean_ctor_set(v___x_490_, 1, v___x_489_);
v___x_491_ = lean_array_push(v_processed_308_, v___x_490_);
v___x_492_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_492_, 0, v___x_491_);
lean_ctor_set(v___x_492_, 1, v_a_311_);
return v___x_492_;
}
else
{
lean_object* v___x_493_; lean_object* v___x_494_; uint8_t v___x_495_; 
v___x_493_ = l_Lean_Syntax_getArg(v___x_482_, v___x_317_);
lean_dec(v___x_482_);
v___x_494_ = lean_box(0);
v___x_495_ = l_Lean_Syntax_matchesIdent(v___x_493_, v___x_494_);
lean_dec(v___x_493_);
if (v___x_495_ == 0)
{
lean_object* v___x_496_; lean_object* v___x_497_; lean_object* v___x_498_; lean_object* v___x_499_; lean_object* v___x_500_; lean_object* v___x_501_; lean_object* v___x_502_; lean_object* v___x_503_; 
v___x_496_ = l_Lean_SourceInfo_fromRef(v_ref_316_, v___x_495_);
lean_dec(v_ref_316_);
v___x_497_ = lean_obj_once(&lp_mathlib_BigOperators_processBigOpBinder___closed__59, &lp_mathlib_BigOperators_processBigOpBinder___closed__59_once, _init_lp_mathlib_BigOperators_processBigOpBinder___closed__59);
v___x_498_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__60));
v___x_499_ = lean_box(0);
v___x_500_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_500_, 0, v___x_496_);
lean_ctor_set(v___x_500_, 1, v___x_497_);
lean_ctor_set(v___x_500_, 2, v___x_498_);
lean_ctor_set(v___x_500_, 3, v___x_499_);
v___x_501_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_501_, 0, v_x_318_);
lean_ctor_set(v___x_501_, 1, v___x_500_);
v___x_502_ = lean_array_push(v_processed_308_, v___x_501_);
v___x_503_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_503_, 0, v___x_502_);
lean_ctor_set(v___x_503_, 1, v_a_311_);
return v___x_503_;
}
else
{
lean_object* v___x_504_; lean_object* v___x_505_; uint8_t v___x_506_; 
v___x_504_ = l_Lean_Syntax_getArg(v_x_318_, v___x_319_);
v___x_505_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__67));
lean_inc(v___x_504_);
v___x_506_ = l_Lean_Syntax_isOfKind(v___x_504_, v___x_505_);
if (v___x_506_ == 0)
{
lean_object* v___x_507_; lean_object* v___x_508_; lean_object* v___x_509_; lean_object* v___x_510_; lean_object* v___x_511_; lean_object* v___x_512_; lean_object* v___x_513_; lean_object* v___x_514_; 
lean_dec(v___x_504_);
v___x_507_ = l_Lean_SourceInfo_fromRef(v_ref_316_, v___x_506_);
lean_dec(v_ref_316_);
v___x_508_ = lean_obj_once(&lp_mathlib_BigOperators_processBigOpBinder___closed__59, &lp_mathlib_BigOperators_processBigOpBinder___closed__59_once, _init_lp_mathlib_BigOperators_processBigOpBinder___closed__59);
v___x_509_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__60));
v___x_510_ = lean_box(0);
v___x_511_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_511_, 0, v___x_507_);
lean_ctor_set(v___x_511_, 1, v___x_508_);
lean_ctor_set(v___x_511_, 2, v___x_509_);
lean_ctor_set(v___x_511_, 3, v___x_510_);
v___x_512_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_512_, 0, v_x_318_);
lean_ctor_set(v___x_512_, 1, v___x_511_);
v___x_513_ = lean_array_push(v_processed_308_, v___x_512_);
v___x_514_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_514_, 0, v___x_513_);
lean_ctor_set(v___x_514_, 1, v_a_311_);
return v___x_514_;
}
else
{
lean_object* v___x_515_; lean_object* v___x_516_; uint8_t v___x_517_; 
v___x_515_ = l_Lean_Syntax_getArg(v___x_504_, v___x_317_);
v___x_516_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__69));
lean_inc(v___x_515_);
v___x_517_ = l_Lean_Syntax_isOfKind(v___x_515_, v___x_516_);
if (v___x_517_ == 0)
{
lean_object* v___x_518_; lean_object* v___x_519_; lean_object* v___x_520_; lean_object* v___x_521_; lean_object* v___x_522_; lean_object* v___x_523_; lean_object* v___x_524_; lean_object* v___x_525_; 
lean_dec(v___x_515_);
lean_dec(v___x_504_);
v___x_518_ = l_Lean_SourceInfo_fromRef(v_ref_316_, v___x_517_);
lean_dec(v_ref_316_);
v___x_519_ = lean_obj_once(&lp_mathlib_BigOperators_processBigOpBinder___closed__59, &lp_mathlib_BigOperators_processBigOpBinder___closed__59_once, _init_lp_mathlib_BigOperators_processBigOpBinder___closed__59);
v___x_520_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__60));
v___x_521_ = lean_box(0);
v___x_522_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_522_, 0, v___x_518_);
lean_ctor_set(v___x_522_, 1, v___x_519_);
lean_ctor_set(v___x_522_, 2, v___x_520_);
lean_ctor_set(v___x_522_, 3, v___x_521_);
v___x_523_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_523_, 0, v_x_318_);
lean_ctor_set(v___x_523_, 1, v___x_522_);
v___x_524_ = lean_array_push(v_processed_308_, v___x_523_);
v___x_525_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_525_, 0, v___x_524_);
lean_ctor_set(v___x_525_, 1, v_a_311_);
return v___x_525_;
}
else
{
lean_object* v___x_526_; lean_object* v___x_527_; lean_object* v___x_528_; lean_object* v___x_529_; uint8_t v___x_530_; lean_object* v___x_531_; lean_object* v___x_532_; lean_object* v___x_533_; lean_object* v___x_534_; lean_object* v___x_535_; lean_object* v___x_536_; lean_object* v___x_537_; lean_object* v___x_538_; lean_object* v___x_539_; lean_object* v___x_540_; lean_object* v___x_541_; lean_object* v___x_542_; lean_object* v___x_543_; lean_object* v___x_544_; lean_object* v___x_545_; lean_object* v___x_546_; lean_object* v___x_547_; lean_object* v___x_548_; lean_object* v___x_549_; lean_object* v___x_550_; lean_object* v___x_551_; 
lean_dec(v_x_318_);
v___x_526_ = l_Lean_Syntax_getArg(v___x_515_, v___x_317_);
v___x_527_ = lean_unsigned_to_nat(2u);
v___x_528_ = l_Lean_Syntax_getArg(v___x_515_, v___x_527_);
lean_dec(v___x_515_);
v___x_529_ = l_Lean_Syntax_getArg(v___x_504_, v___x_527_);
lean_dec(v___x_504_);
v___x_530_ = 0;
v___x_531_ = l_Lean_SourceInfo_fromRef(v_ref_316_, v___x_530_);
lean_dec(v_ref_316_);
v___x_532_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__71));
v___x_533_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__72));
lean_inc_n(v___x_531_, 7);
v___x_534_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_534_, 0, v___x_531_);
lean_ctor_set(v___x_534_, 1, v___x_533_);
v___x_535_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__25));
v___x_536_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__73));
v___x_537_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_537_, 0, v___x_531_);
lean_ctor_set(v___x_537_, 1, v___x_536_);
v___x_538_ = l_Lean_Syntax_node3(v___x_531_, v___x_535_, v___x_526_, v___x_537_, v___x_528_);
v___x_539_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__74));
v___x_540_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_540_, 0, v___x_531_);
lean_ctor_set(v___x_540_, 1, v___x_539_);
v___x_541_ = l_Lean_Syntax_node3(v___x_531_, v___x_532_, v___x_534_, v___x_538_, v___x_540_);
v___x_542_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__18));
v___x_543_ = lean_obj_once(&lp_mathlib_BigOperators_processBigOpBinder___closed__76, &lp_mathlib_BigOperators_processBigOpBinder___closed__76_once, _init_lp_mathlib_BigOperators_processBigOpBinder___closed__76);
v___x_544_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__79));
v___x_545_ = lean_box(0);
v___x_546_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_546_, 0, v___x_531_);
lean_ctor_set(v___x_546_, 1, v___x_543_);
lean_ctor_set(v___x_546_, 2, v___x_544_);
lean_ctor_set(v___x_546_, 3, v___x_545_);
v___x_547_ = l_Lean_Syntax_node1(v___x_531_, v___x_535_, v___x_529_);
v___x_548_ = l_Lean_Syntax_node2(v___x_531_, v___x_542_, v___x_546_, v___x_547_);
v___x_549_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_549_, 0, v___x_541_);
lean_ctor_set(v___x_549_, 1, v___x_548_);
v___x_550_ = lean_array_push(v_processed_308_, v___x_549_);
v___x_551_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_551_, 0, v___x_550_);
lean_ctor_set(v___x_551_, 1, v_a_311_);
return v___x_551_;
}
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_BigOperators_processBigOpBinder___boxed(lean_object* v_processed_552_, lean_object* v_binder_553_, lean_object* v_a_554_, lean_object* v_a_555_){
_start:
{
lean_object* v_res_556_; 
v_res_556_ = lp_mathlib_BigOperators_processBigOpBinder(v_processed_552_, v_binder_553_, v_a_554_, v_a_555_);
lean_dec_ref(v_a_554_);
return v_res_556_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00BigOperators_processBigOpBinders_spec__0(size_t v_sz_557_, size_t v_i_558_, lean_object* v_bs_559_){
_start:
{
uint8_t v___x_560_; 
v___x_560_ = lean_usize_dec_lt(v_i_558_, v_sz_557_);
if (v___x_560_ == 0)
{
lean_object* v___x_561_; 
v___x_561_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_561_, 0, v_bs_559_);
return v___x_561_;
}
else
{
lean_object* v_v_562_; lean_object* v___x_563_; uint8_t v___x_564_; 
v_v_562_ = lean_array_uget_borrowed(v_bs_559_, v_i_558_);
v___x_563_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBinderParenthesized___closed__1));
lean_inc(v_v_562_);
v___x_564_ = l_Lean_Syntax_isOfKind(v_v_562_, v___x_563_);
if (v___x_564_ == 0)
{
lean_object* v___x_565_; 
lean_dec_ref(v_bs_559_);
v___x_565_ = lean_box(0);
return v___x_565_;
}
else
{
lean_object* v___x_566_; lean_object* v_bs_567_; lean_object* v___x_568_; uint8_t v___x_569_; 
v___x_566_ = lean_unsigned_to_nat(1u);
v_bs_567_ = l_Lean_Syntax_getArg(v_v_562_, v___x_566_);
v___x_568_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBinder___closed__2));
lean_inc(v_bs_567_);
v___x_569_ = l_Lean_Syntax_isOfKind(v_bs_567_, v___x_568_);
if (v___x_569_ == 0)
{
lean_object* v___x_570_; 
lean_dec(v_bs_567_);
lean_dec_ref(v_bs_559_);
v___x_570_ = lean_box(0);
return v___x_570_;
}
else
{
lean_object* v___x_571_; lean_object* v_bs_x27_572_; size_t v___x_573_; size_t v___x_574_; lean_object* v___x_575_; 
v___x_571_ = lean_unsigned_to_nat(0u);
v_bs_x27_572_ = lean_array_uset(v_bs_559_, v_i_558_, v___x_571_);
v___x_573_ = ((size_t)1ULL);
v___x_574_ = lean_usize_add(v_i_558_, v___x_573_);
v___x_575_ = lean_array_uset(v_bs_x27_572_, v_i_558_, v_bs_567_);
v_i_558_ = v___x_574_;
v_bs_559_ = v___x_575_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00BigOperators_processBigOpBinders_spec__0___boxed(lean_object* v_sz_577_, lean_object* v_i_578_, lean_object* v_bs_579_){
_start:
{
size_t v_sz_boxed_580_; size_t v_i_boxed_581_; lean_object* v_res_582_; 
v_sz_boxed_580_ = lean_unbox_usize(v_sz_577_);
lean_dec(v_sz_577_);
v_i_boxed_581_ = lean_unbox_usize(v_i_578_);
lean_dec(v_i_578_);
v_res_582_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00BigOperators_processBigOpBinders_spec__0(v_sz_boxed_580_, v_i_boxed_581_, v_bs_579_);
return v_res_582_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00BigOperators_processBigOpBinders_spec__1(lean_object* v_as_583_, size_t v_i_584_, size_t v_stop_585_, lean_object* v_b_586_, lean_object* v___y_587_, lean_object* v___y_588_){
_start:
{
uint8_t v___x_589_; 
v___x_589_ = lean_usize_dec_eq(v_i_584_, v_stop_585_);
if (v___x_589_ == 0)
{
lean_object* v___x_590_; lean_object* v___x_591_; 
v___x_590_ = lean_array_uget_borrowed(v_as_583_, v_i_584_);
lean_inc(v___x_590_);
v___x_591_ = lp_mathlib_BigOperators_processBigOpBinder(v_b_586_, v___x_590_, v___y_587_, v___y_588_);
if (lean_obj_tag(v___x_591_) == 0)
{
lean_object* v_a_592_; lean_object* v_a_593_; size_t v___x_594_; size_t v___x_595_; 
v_a_592_ = lean_ctor_get(v___x_591_, 0);
lean_inc(v_a_592_);
v_a_593_ = lean_ctor_get(v___x_591_, 1);
lean_inc(v_a_593_);
lean_dec_ref_known(v___x_591_, 2);
v___x_594_ = ((size_t)1ULL);
v___x_595_ = lean_usize_add(v_i_584_, v___x_594_);
v_i_584_ = v___x_595_;
v_b_586_ = v_a_592_;
v___y_588_ = v_a_593_;
goto _start;
}
else
{
return v___x_591_;
}
}
else
{
lean_object* v___x_597_; 
v___x_597_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_597_, 0, v_b_586_);
lean_ctor_set(v___x_597_, 1, v___y_588_);
return v___x_597_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00BigOperators_processBigOpBinders_spec__1___boxed(lean_object* v_as_598_, lean_object* v_i_599_, lean_object* v_stop_600_, lean_object* v_b_601_, lean_object* v___y_602_, lean_object* v___y_603_){
_start:
{
size_t v_i_boxed_604_; size_t v_stop_boxed_605_; lean_object* v_res_606_; 
v_i_boxed_604_ = lean_unbox_usize(v_i_599_);
lean_dec(v_i_599_);
v_stop_boxed_605_ = lean_unbox_usize(v_stop_600_);
lean_dec(v_stop_600_);
v_res_606_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00BigOperators_processBigOpBinders_spec__1(v_as_598_, v_i_boxed_604_, v_stop_boxed_605_, v_b_601_, v___y_602_, v___y_603_);
lean_dec_ref(v___y_602_);
lean_dec_ref(v_as_598_);
return v_res_606_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BigOperators_processBigOpBinders(lean_object* v_binders_609_, lean_object* v_a_610_, lean_object* v_a_611_){
_start:
{
lean_object* v___x_612_; uint8_t v___x_613_; 
v___x_612_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBinders___closed__1));
lean_inc(v_binders_609_);
v___x_613_ = l_Lean_Syntax_isOfKind(v_binders_609_, v___x_612_);
if (v___x_613_ == 0)
{
lean_object* v___x_614_; 
lean_dec(v_binders_609_);
v___x_614_ = l_Lean_Macro_throwUnsupported___redArg(v_a_611_);
return v___x_614_;
}
else
{
lean_object* v___x_615_; lean_object* v_b_616_; lean_object* v___x_617_; uint8_t v___x_618_; 
v___x_615_ = lean_unsigned_to_nat(0u);
v_b_616_ = l_Lean_Syntax_getArg(v_binders_609_, v___x_615_);
lean_dec(v_binders_609_);
v___x_617_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBinder___closed__2));
lean_inc(v_b_616_);
v___x_618_ = l_Lean_Syntax_isOfKind(v_b_616_, v___x_617_);
if (v___x_618_ == 0)
{
lean_object* v___x_619_; uint8_t v___x_620_; 
v___x_619_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBinderCollection___closed__1));
lean_inc(v_b_616_);
v___x_620_ = l_Lean_Syntax_isOfKind(v_b_616_, v___x_619_);
if (v___x_620_ == 0)
{
lean_object* v___x_621_; 
lean_dec(v_b_616_);
v___x_621_ = l_Lean_Macro_throwUnsupported___redArg(v_a_611_);
return v___x_621_;
}
else
{
lean_object* v___x_622_; lean_object* v___x_623_; size_t v_sz_624_; size_t v___x_625_; lean_object* v___x_626_; 
v___x_622_ = l_Lean_Syntax_getArg(v_b_616_, v___x_615_);
lean_dec(v_b_616_);
v___x_623_ = l_Lean_Syntax_getArgs(v___x_622_);
lean_dec(v___x_622_);
v_sz_624_ = lean_array_size(v___x_623_);
v___x_625_ = ((size_t)0ULL);
v___x_626_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00BigOperators_processBigOpBinders_spec__0(v_sz_624_, v___x_625_, v___x_623_);
if (lean_obj_tag(v___x_626_) == 0)
{
lean_object* v___x_627_; 
v___x_627_ = l_Lean_Macro_throwUnsupported___redArg(v_a_611_);
return v___x_627_;
}
else
{
lean_object* v_val_628_; lean_object* v___x_629_; lean_object* v___x_630_; uint8_t v___x_631_; 
v_val_628_ = lean_ctor_get(v___x_626_, 0);
lean_inc(v_val_628_);
lean_dec_ref_known(v___x_626_, 1);
v___x_629_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinders___closed__0));
v___x_630_ = lean_array_get_size(v_val_628_);
v___x_631_ = lean_nat_dec_lt(v___x_615_, v___x_630_);
if (v___x_631_ == 0)
{
lean_object* v___x_632_; 
lean_dec(v_val_628_);
v___x_632_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_632_, 0, v___x_629_);
lean_ctor_set(v___x_632_, 1, v_a_611_);
return v___x_632_;
}
else
{
uint8_t v___x_633_; 
v___x_633_ = lean_nat_dec_le(v___x_630_, v___x_630_);
if (v___x_633_ == 0)
{
if (v___x_631_ == 0)
{
lean_object* v___x_634_; 
lean_dec(v_val_628_);
v___x_634_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_634_, 0, v___x_629_);
lean_ctor_set(v___x_634_, 1, v_a_611_);
return v___x_634_;
}
else
{
size_t v___x_635_; lean_object* v___x_636_; 
v___x_635_ = lean_usize_of_nat(v___x_630_);
v___x_636_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00BigOperators_processBigOpBinders_spec__1(v_val_628_, v___x_625_, v___x_635_, v___x_629_, v_a_610_, v_a_611_);
lean_dec(v_val_628_);
return v___x_636_;
}
}
else
{
size_t v___x_637_; lean_object* v___x_638_; 
v___x_637_ = lean_usize_of_nat(v___x_630_);
v___x_638_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00BigOperators_processBigOpBinders_spec__1(v_val_628_, v___x_625_, v___x_637_, v___x_629_, v_a_610_, v_a_611_);
lean_dec(v_val_628_);
return v___x_638_;
}
}
}
}
}
else
{
lean_object* v___x_639_; lean_object* v___x_640_; 
v___x_639_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinders___closed__0));
v___x_640_ = lp_mathlib_BigOperators_processBigOpBinder(v___x_639_, v_b_616_, v_a_610_, v_a_611_);
return v___x_640_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_BigOperators_processBigOpBinders___boxed(lean_object* v_binders_641_, lean_object* v_a_642_, lean_object* v_a_643_){
_start:
{
lean_object* v_res_644_; 
v_res_644_ = lp_mathlib_BigOperators_processBigOpBinders(v_binders_641_, v_a_642_, v_a_643_);
lean_dec_ref(v_a_642_);
return v_res_644_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00BigOperators_bigOpBindersPattern_spec__0(size_t v_sz_645_, size_t v_i_646_, lean_object* v_bs_647_){
_start:
{
uint8_t v___x_648_; 
v___x_648_ = lean_usize_dec_lt(v_i_646_, v_sz_645_);
if (v___x_648_ == 0)
{
return v_bs_647_;
}
else
{
lean_object* v_v_649_; lean_object* v_fst_650_; lean_object* v___x_651_; lean_object* v_bs_x27_652_; size_t v___x_653_; size_t v___x_654_; lean_object* v___x_655_; 
v_v_649_ = lean_array_uget_borrowed(v_bs_647_, v_i_646_);
v_fst_650_ = lean_ctor_get(v_v_649_, 0);
lean_inc(v_fst_650_);
v___x_651_ = lean_unsigned_to_nat(0u);
v_bs_x27_652_ = lean_array_uset(v_bs_647_, v_i_646_, v___x_651_);
v___x_653_ = ((size_t)1ULL);
v___x_654_ = lean_usize_add(v_i_646_, v___x_653_);
v___x_655_ = lean_array_uset(v_bs_x27_652_, v_i_646_, v_fst_650_);
v_i_646_ = v___x_654_;
v_bs_647_ = v___x_655_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00BigOperators_bigOpBindersPattern_spec__0___boxed(lean_object* v_sz_657_, lean_object* v_i_658_, lean_object* v_bs_659_){
_start:
{
size_t v_sz_boxed_660_; size_t v_i_boxed_661_; lean_object* v_res_662_; 
v_sz_boxed_660_ = lean_unbox_usize(v_sz_657_);
lean_dec(v_sz_657_);
v_i_boxed_661_ = lean_unbox_usize(v_i_658_);
lean_dec(v_i_658_);
v_res_662_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00BigOperators_bigOpBindersPattern_spec__0(v_sz_boxed_660_, v_i_boxed_661_, v_bs_659_);
return v_res_662_;
}
}
static lean_object* _init_lp_mathlib_BigOperators_bigOpBindersPattern___closed__0(void){
_start:
{
lean_object* v___x_663_; 
v___x_663_ = l_Array_mkArray0(lean_box(0));
return v___x_663_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BigOperators_bigOpBindersPattern(lean_object* v_processed_664_, lean_object* v_a_665_, lean_object* v_a_666_){
_start:
{
size_t v_sz_667_; size_t v___x_668_; lean_object* v_ts_669_; lean_object* v___x_670_; lean_object* v___x_671_; uint8_t v___x_672_; 
v_sz_667_ = lean_array_size(v_processed_664_);
v___x_668_ = ((size_t)0ULL);
v_ts_669_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00BigOperators_bigOpBindersPattern_spec__0(v_sz_667_, v___x_668_, v_processed_664_);
v___x_670_ = lean_array_get_size(v_ts_669_);
v___x_671_ = lean_unsigned_to_nat(1u);
v___x_672_ = lean_nat_dec_eq(v___x_670_, v___x_671_);
if (v___x_672_ == 0)
{
lean_object* v_ref_673_; lean_object* v___x_674_; lean_object* v___x_675_; lean_object* v___x_676_; lean_object* v___x_677_; lean_object* v___x_678_; lean_object* v___x_679_; lean_object* v___x_680_; lean_object* v___x_681_; lean_object* v___x_682_; lean_object* v___x_683_; lean_object* v___x_684_; lean_object* v___x_685_; lean_object* v___x_686_; lean_object* v___x_687_; 
v_ref_673_ = lean_ctor_get(v_a_665_, 5);
v___x_674_ = l_Lean_SourceInfo_fromRef(v_ref_673_, v___x_672_);
v___x_675_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__71));
v___x_676_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__72));
lean_inc_n(v___x_674_, 3);
v___x_677_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_677_, 0, v___x_674_);
lean_ctor_set(v___x_677_, 1, v___x_676_);
v___x_678_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__25));
v___x_679_ = lean_obj_once(&lp_mathlib_BigOperators_bigOpBindersPattern___closed__0, &lp_mathlib_BigOperators_bigOpBindersPattern___closed__0_once, _init_lp_mathlib_BigOperators_bigOpBindersPattern___closed__0);
v___x_680_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__73));
v___x_681_ = l_Lean_Syntax_SepArray_ofElems(v___x_680_, v_ts_669_);
lean_dec_ref(v_ts_669_);
v___x_682_ = l_Array_append___redArg(v___x_679_, v___x_681_);
lean_dec_ref(v___x_681_);
v___x_683_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_683_, 0, v___x_674_);
lean_ctor_set(v___x_683_, 1, v___x_678_);
lean_ctor_set(v___x_683_, 2, v___x_682_);
v___x_684_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__74));
v___x_685_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_685_, 0, v___x_674_);
lean_ctor_set(v___x_685_, 1, v___x_684_);
v___x_686_ = l_Lean_Syntax_node3(v___x_674_, v___x_675_, v___x_677_, v___x_683_, v___x_685_);
v___x_687_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_687_, 0, v___x_686_);
lean_ctor_set(v___x_687_, 1, v_a_666_);
return v___x_687_;
}
else
{
lean_object* v___x_688_; lean_object* v___x_689_; lean_object* v___x_690_; 
v___x_688_ = lean_unsigned_to_nat(0u);
v___x_689_ = lean_array_fget(v_ts_669_, v___x_688_);
lean_dec_ref(v_ts_669_);
v___x_690_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_690_, 0, v___x_689_);
lean_ctor_set(v___x_690_, 1, v_a_666_);
return v___x_690_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_BigOperators_bigOpBindersPattern___boxed(lean_object* v_processed_691_, lean_object* v_a_692_, lean_object* v_a_693_){
_start:
{
lean_object* v_res_694_; 
v_res_694_ = lp_mathlib_BigOperators_bigOpBindersPattern(v_processed_691_, v_a_692_, v_a_693_);
lean_dec_ref(v_a_692_);
return v_res_694_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00BigOperators_bigOpBindersProd_spec__0___closed__4(void){
_start:
{
lean_object* v___x_701_; lean_object* v___x_702_; 
v___x_701_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00BigOperators_bigOpBindersProd_spec__0___closed__3));
v___x_702_ = l_String_toRawSubstring_x27(v___x_701_);
return v___x_702_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00BigOperators_bigOpBindersProd_spec__0(lean_object* v_as_709_, size_t v_i_710_, size_t v_stop_711_, lean_object* v_b_712_, lean_object* v___y_713_, lean_object* v___y_714_){
_start:
{
uint8_t v___x_715_; 
v___x_715_ = lean_usize_dec_eq(v_i_710_, v_stop_711_);
if (v___x_715_ == 0)
{
lean_object* v_quotContext_716_; lean_object* v_currMacroScope_717_; lean_object* v_ref_718_; size_t v___x_719_; size_t v___x_720_; lean_object* v___x_721_; lean_object* v_snd_722_; lean_object* v___x_723_; lean_object* v___x_724_; lean_object* v___x_725_; lean_object* v___x_726_; lean_object* v___x_727_; lean_object* v___x_728_; lean_object* v___x_729_; lean_object* v___x_730_; lean_object* v___x_731_; lean_object* v___x_732_; 
v_quotContext_716_ = lean_ctor_get(v___y_713_, 1);
v_currMacroScope_717_ = lean_ctor_get(v___y_713_, 2);
v_ref_718_ = lean_ctor_get(v___y_713_, 5);
v___x_719_ = ((size_t)1ULL);
v___x_720_ = lean_usize_sub(v_i_710_, v___x_719_);
v___x_721_ = lean_array_uget_borrowed(v_as_709_, v___x_720_);
v_snd_722_ = lean_ctor_get(v___x_721_, 1);
v___x_723_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00BigOperators_bigOpBindersProd_spec__0___closed__2));
lean_inc(v_currMacroScope_717_);
lean_inc(v_quotContext_716_);
v___x_724_ = l_Lean_addMacroScope(v_quotContext_716_, v___x_723_, v_currMacroScope_717_);
v___x_725_ = l_Lean_SourceInfo_fromRef(v_ref_718_, v___x_715_);
v___x_726_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__18));
v___x_727_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00BigOperators_bigOpBindersProd_spec__0___closed__4, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00BigOperators_bigOpBindersProd_spec__0___closed__4_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00BigOperators_bigOpBindersProd_spec__0___closed__4);
v___x_728_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00BigOperators_bigOpBindersProd_spec__0___closed__6));
lean_inc_n(v___x_725_, 2);
v___x_729_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_729_, 0, v___x_725_);
lean_ctor_set(v___x_729_, 1, v___x_727_);
lean_ctor_set(v___x_729_, 2, v___x_724_);
lean_ctor_set(v___x_729_, 3, v___x_728_);
v___x_730_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__25));
lean_inc(v_snd_722_);
v___x_731_ = l_Lean_Syntax_node2(v___x_725_, v___x_730_, v_snd_722_, v_b_712_);
v___x_732_ = l_Lean_Syntax_node2(v___x_725_, v___x_726_, v___x_729_, v___x_731_);
v_i_710_ = v___x_720_;
v_b_712_ = v___x_732_;
goto _start;
}
else
{
lean_object* v___x_734_; 
v___x_734_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_734_, 0, v_b_712_);
lean_ctor_set(v___x_734_, 1, v___y_714_);
return v___x_734_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00BigOperators_bigOpBindersProd_spec__0___boxed(lean_object* v_as_735_, lean_object* v_i_736_, lean_object* v_stop_737_, lean_object* v_b_738_, lean_object* v___y_739_, lean_object* v___y_740_){
_start:
{
size_t v_i_boxed_741_; size_t v_stop_boxed_742_; lean_object* v_res_743_; 
v_i_boxed_741_ = lean_unbox_usize(v_i_736_);
lean_dec(v_i_736_);
v_stop_boxed_742_ = lean_unbox_usize(v_stop_737_);
lean_dec(v_stop_737_);
v_res_743_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00BigOperators_bigOpBindersProd_spec__0(v_as_735_, v_i_boxed_741_, v_stop_boxed_742_, v_b_738_, v___y_739_, v___y_740_);
lean_dec_ref(v___y_739_);
lean_dec_ref(v_as_735_);
return v_res_743_;
}
}
static lean_object* _init_lp_mathlib_BigOperators_bigOpBindersProd___closed__30(void){
_start:
{
lean_object* v___x_811_; lean_object* v___x_812_; 
v___x_811_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBindersProd___closed__29));
v___x_812_ = l_String_toRawSubstring_x27(v___x_811_);
return v___x_812_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BigOperators_bigOpBindersProd(lean_object* v_processed_826_, lean_object* v_a_827_, lean_object* v_a_828_){
_start:
{
lean_object* v___x_829_; lean_object* v___x_830_; uint8_t v___x_831_; 
v___x_829_ = lean_array_get_size(v_processed_826_);
v___x_830_ = lean_unsigned_to_nat(0u);
v___x_831_ = lean_nat_dec_eq(v___x_829_, v___x_830_);
if (v___x_831_ == 0)
{
lean_object* v___x_832_; uint8_t v___x_833_; 
v___x_832_ = lean_unsigned_to_nat(1u);
v___x_833_ = lean_nat_dec_eq(v___x_829_, v___x_832_);
if (v___x_833_ == 0)
{
lean_object* v___x_834_; lean_object* v___x_835_; lean_object* v_snd_836_; lean_object* v___x_838_; uint8_t v_isShared_839_; uint8_t v_isSharedCheck_855_; 
v___x_834_ = lean_nat_sub(v___x_829_, v___x_832_);
v___x_835_ = lean_array_fget(v_processed_826_, v___x_834_);
v_snd_836_ = lean_ctor_get(v___x_835_, 1);
v_isSharedCheck_855_ = !lean_is_exclusive(v___x_835_);
if (v_isSharedCheck_855_ == 0)
{
lean_object* v_unused_856_; 
v_unused_856_ = lean_ctor_get(v___x_835_, 0);
lean_dec(v_unused_856_);
v___x_838_ = v___x_835_;
v_isShared_839_ = v_isSharedCheck_855_;
goto v_resetjp_837_;
}
else
{
lean_inc(v_snd_836_);
lean_dec(v___x_835_);
v___x_838_ = lean_box(0);
v_isShared_839_ = v_isSharedCheck_855_;
goto v_resetjp_837_;
}
v_resetjp_837_:
{
uint8_t v___x_840_; 
v___x_840_ = lean_nat_dec_le(v___x_834_, v___x_829_);
if (v___x_840_ == 0)
{
uint8_t v___x_841_; 
lean_dec(v___x_834_);
v___x_841_ = lean_nat_dec_lt(v___x_830_, v___x_829_);
if (v___x_841_ == 0)
{
lean_object* v___x_843_; 
if (v_isShared_839_ == 0)
{
lean_ctor_set(v___x_838_, 1, v_a_828_);
lean_ctor_set(v___x_838_, 0, v_snd_836_);
v___x_843_ = v___x_838_;
goto v_reusejp_842_;
}
else
{
lean_object* v_reuseFailAlloc_844_; 
v_reuseFailAlloc_844_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_844_, 0, v_snd_836_);
lean_ctor_set(v_reuseFailAlloc_844_, 1, v_a_828_);
v___x_843_ = v_reuseFailAlloc_844_;
goto v_reusejp_842_;
}
v_reusejp_842_:
{
return v___x_843_;
}
}
else
{
size_t v___x_845_; size_t v___x_846_; lean_object* v___x_847_; 
lean_del_object(v___x_838_);
v___x_845_ = lean_usize_of_nat(v___x_829_);
v___x_846_ = ((size_t)0ULL);
v___x_847_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00BigOperators_bigOpBindersProd_spec__0(v_processed_826_, v___x_845_, v___x_846_, v_snd_836_, v_a_827_, v_a_828_);
return v___x_847_;
}
}
else
{
uint8_t v___x_848_; 
v___x_848_ = lean_nat_dec_lt(v___x_830_, v___x_834_);
if (v___x_848_ == 0)
{
lean_object* v___x_850_; 
lean_dec(v___x_834_);
if (v_isShared_839_ == 0)
{
lean_ctor_set(v___x_838_, 1, v_a_828_);
lean_ctor_set(v___x_838_, 0, v_snd_836_);
v___x_850_ = v___x_838_;
goto v_reusejp_849_;
}
else
{
lean_object* v_reuseFailAlloc_851_; 
v_reuseFailAlloc_851_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_851_, 0, v_snd_836_);
lean_ctor_set(v_reuseFailAlloc_851_, 1, v_a_828_);
v___x_850_ = v_reuseFailAlloc_851_;
goto v_reusejp_849_;
}
v_reusejp_849_:
{
return v___x_850_;
}
}
else
{
size_t v___x_852_; size_t v___x_853_; lean_object* v___x_854_; 
lean_del_object(v___x_838_);
v___x_852_ = lean_usize_of_nat(v___x_834_);
lean_dec(v___x_834_);
v___x_853_ = ((size_t)0ULL);
v___x_854_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00BigOperators_bigOpBindersProd_spec__0(v_processed_826_, v___x_852_, v___x_853_, v_snd_836_, v_a_827_, v_a_828_);
return v___x_854_;
}
}
}
}
else
{
lean_object* v___x_857_; lean_object* v_snd_858_; lean_object* v___x_860_; uint8_t v_isShared_861_; uint8_t v_isSharedCheck_865_; 
v___x_857_ = lean_array_fget(v_processed_826_, v___x_830_);
v_snd_858_ = lean_ctor_get(v___x_857_, 1);
v_isSharedCheck_865_ = !lean_is_exclusive(v___x_857_);
if (v_isSharedCheck_865_ == 0)
{
lean_object* v_unused_866_; 
v_unused_866_ = lean_ctor_get(v___x_857_, 0);
lean_dec(v_unused_866_);
v___x_860_ = v___x_857_;
v_isShared_861_ = v_isSharedCheck_865_;
goto v_resetjp_859_;
}
else
{
lean_inc(v_snd_858_);
lean_dec(v___x_857_);
v___x_860_ = lean_box(0);
v_isShared_861_ = v_isSharedCheck_865_;
goto v_resetjp_859_;
}
v_resetjp_859_:
{
lean_object* v___x_863_; 
if (v_isShared_861_ == 0)
{
lean_ctor_set(v___x_860_, 1, v_a_828_);
lean_ctor_set(v___x_860_, 0, v_snd_858_);
v___x_863_ = v___x_860_;
goto v_reusejp_862_;
}
else
{
lean_object* v_reuseFailAlloc_864_; 
v_reuseFailAlloc_864_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_864_, 0, v_snd_858_);
lean_ctor_set(v_reuseFailAlloc_864_, 1, v_a_828_);
v___x_863_ = v_reuseFailAlloc_864_;
goto v_reusejp_862_;
}
v_reusejp_862_:
{
return v___x_863_;
}
}
}
}
else
{
lean_object* v_quotContext_867_; lean_object* v_currMacroScope_868_; lean_object* v_ref_869_; uint8_t v___x_870_; lean_object* v___x_871_; lean_object* v___x_872_; lean_object* v___x_873_; lean_object* v___x_874_; lean_object* v___x_875_; lean_object* v___x_876_; lean_object* v___x_877_; lean_object* v___x_878_; lean_object* v___x_879_; lean_object* v___x_880_; lean_object* v___x_881_; lean_object* v___x_882_; lean_object* v___x_883_; lean_object* v___x_884_; lean_object* v___x_885_; lean_object* v___x_886_; lean_object* v___x_887_; lean_object* v___x_888_; lean_object* v___x_889_; lean_object* v___x_890_; lean_object* v___x_891_; lean_object* v___x_892_; lean_object* v___x_893_; lean_object* v___x_894_; lean_object* v___x_895_; lean_object* v___x_896_; lean_object* v___x_897_; lean_object* v___x_898_; lean_object* v___x_899_; lean_object* v___x_900_; lean_object* v___x_901_; lean_object* v___x_902_; lean_object* v___x_903_; lean_object* v___x_904_; lean_object* v___x_905_; lean_object* v___x_906_; lean_object* v___x_907_; lean_object* v___x_908_; lean_object* v___x_909_; 
v_quotContext_867_ = lean_ctor_get(v_a_827_, 1);
v_currMacroScope_868_ = lean_ctor_get(v_a_827_, 2);
v_ref_869_ = lean_ctor_get(v_a_827_, 5);
v___x_870_ = 0;
v___x_871_ = l_Lean_SourceInfo_fromRef(v_ref_869_, v___x_870_);
v___x_872_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__50));
v___x_873_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__52));
v___x_874_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__53));
lean_inc_n(v___x_871_, 12);
v___x_875_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_875_, 0, v___x_871_);
lean_ctor_set(v___x_875_, 1, v___x_874_);
v___x_876_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__55));
v___x_877_ = lean_obj_once(&lp_mathlib_BigOperators_processBigOpBinder___closed__57, &lp_mathlib_BigOperators_processBigOpBinder___closed__57_once, _init_lp_mathlib_BigOperators_processBigOpBinder___closed__57);
v___x_878_ = lean_box(0);
lean_inc_n(v_currMacroScope_868_, 4);
lean_inc_n(v_quotContext_867_, 4);
v___x_879_ = l_Lean_addMacroScope(v_quotContext_867_, v___x_878_, v_currMacroScope_868_);
v___x_880_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBindersProd___closed__22));
v___x_881_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_881_, 0, v___x_871_);
lean_ctor_set(v___x_881_, 1, v___x_877_);
lean_ctor_set(v___x_881_, 2, v___x_879_);
lean_ctor_set(v___x_881_, 3, v___x_880_);
v___x_882_ = l_Lean_Syntax_node1(v___x_871_, v___x_876_, v___x_881_);
v___x_883_ = l_Lean_Syntax_node2(v___x_871_, v___x_873_, v___x_875_, v___x_882_);
v___x_884_ = lean_obj_once(&lp_mathlib_BigOperators_processBigOpBinder___closed__59, &lp_mathlib_BigOperators_processBigOpBinder___closed__59_once, _init_lp_mathlib_BigOperators_processBigOpBinder___closed__59);
v___x_885_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__60));
v___x_886_ = l_Lean_addMacroScope(v_quotContext_867_, v___x_885_, v_currMacroScope_868_);
v___x_887_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBindersProd___closed__24));
v___x_888_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_888_, 0, v___x_871_);
lean_ctor_set(v___x_888_, 1, v___x_884_);
lean_ctor_set(v___x_888_, 2, v___x_886_);
lean_ctor_set(v___x_888_, 3, v___x_887_);
v___x_889_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__61));
v___x_890_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_890_, 0, v___x_871_);
lean_ctor_set(v___x_890_, 1, v___x_889_);
v___x_891_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__25));
v___x_892_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__18));
v___x_893_ = lean_obj_once(&lp_mathlib_BigOperators_processBigOpBinder___closed__62, &lp_mathlib_BigOperators_processBigOpBinder___closed__62_once, _init_lp_mathlib_BigOperators_processBigOpBinder___closed__62);
v___x_894_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__63));
v___x_895_ = l_Lean_addMacroScope(v_quotContext_867_, v___x_894_, v_currMacroScope_868_);
v___x_896_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBindersProd___closed__28));
v___x_897_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_897_, 0, v___x_871_);
lean_ctor_set(v___x_897_, 1, v___x_893_);
lean_ctor_set(v___x_897_, 2, v___x_895_);
lean_ctor_set(v___x_897_, 3, v___x_896_);
v___x_898_ = lean_obj_once(&lp_mathlib_BigOperators_bigOpBindersProd___closed__30, &lp_mathlib_BigOperators_bigOpBindersProd___closed__30_once, _init_lp_mathlib_BigOperators_bigOpBindersProd___closed__30);
v___x_899_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBindersProd___closed__31));
v___x_900_ = l_Lean_addMacroScope(v_quotContext_867_, v___x_899_, v_currMacroScope_868_);
v___x_901_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBindersProd___closed__35));
v___x_902_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_902_, 0, v___x_871_);
lean_ctor_set(v___x_902_, 1, v___x_898_);
lean_ctor_set(v___x_902_, 2, v___x_900_);
lean_ctor_set(v___x_902_, 3, v___x_901_);
v___x_903_ = l_Lean_Syntax_node1(v___x_871_, v___x_891_, v___x_902_);
v___x_904_ = l_Lean_Syntax_node2(v___x_871_, v___x_892_, v___x_897_, v___x_903_);
v___x_905_ = l_Lean_Syntax_node1(v___x_871_, v___x_891_, v___x_904_);
v___x_906_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBinderParenthesized___closed__5));
v___x_907_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_907_, 0, v___x_871_);
lean_ctor_set(v___x_907_, 1, v___x_906_);
v___x_908_ = l_Lean_Syntax_node5(v___x_871_, v___x_872_, v___x_883_, v___x_888_, v___x_890_, v___x_905_, v___x_907_);
v___x_909_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_909_, 0, v___x_908_);
lean_ctor_set(v___x_909_, 1, v_a_828_);
return v___x_909_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_BigOperators_bigOpBindersProd___boxed(lean_object* v_processed_910_, lean_object* v_a_911_, lean_object* v_a_912_){
_start:
{
lean_object* v_res_913_; 
v_res_913_ = lp_mathlib_BigOperators_bigOpBindersProd(v_processed_910_, v_a_911_, v_a_912_);
lean_dec_ref(v_a_911_);
lean_dec_ref(v_processed_910_);
return v_res_913_;
}
}
static lean_object* _init_lp_mathlib_BigOperators_BigOpWith___closed__6(void){
_start:
{
lean_object* v___x_924_; lean_object* v___x_925_; lean_object* v___x_926_; lean_object* v___x_927_; 
v___x_924_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBinder___closed__15));
v___x_925_ = l_Lean_binderIdent;
v___x_926_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBinder___closed__4));
v___x_927_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_927_, 0, v___x_926_);
lean_ctor_set(v___x_927_, 1, v___x_925_);
lean_ctor_set(v___x_927_, 2, v___x_924_);
return v___x_927_;
}
}
static lean_object* _init_lp_mathlib_BigOperators_BigOpWith___closed__7(void){
_start:
{
lean_object* v___x_928_; lean_object* v___x_929_; lean_object* v___x_930_; 
v___x_928_ = lean_obj_once(&lp_mathlib_BigOperators_BigOpWith___closed__6, &lp_mathlib_BigOperators_BigOpWith___closed__6_once, _init_lp_mathlib_BigOperators_BigOpWith___closed__6);
v___x_929_ = ((lean_object*)(lp_mathlib_BigOperators_BigOpWith___closed__5));
v___x_930_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_930_, 0, v___x_929_);
lean_ctor_set(v___x_930_, 1, v___x_928_);
return v___x_930_;
}
}
static lean_object* _init_lp_mathlib_BigOperators_BigOpWith___closed__8(void){
_start:
{
lean_object* v___x_931_; lean_object* v___x_932_; lean_object* v___x_933_; 
v___x_931_ = lean_obj_once(&lp_mathlib_BigOperators_BigOpWith___closed__7, &lp_mathlib_BigOperators_BigOpWith___closed__7_once, _init_lp_mathlib_BigOperators_BigOpWith___closed__7);
v___x_932_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBinder___closed__9));
v___x_933_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_933_, 0, v___x_932_);
lean_ctor_set(v___x_933_, 1, v___x_931_);
return v___x_933_;
}
}
static lean_object* _init_lp_mathlib_BigOperators_BigOpWith___closed__9(void){
_start:
{
lean_object* v___x_934_; lean_object* v___x_935_; lean_object* v___x_936_; lean_object* v___x_937_; 
v___x_934_ = lean_obj_once(&lp_mathlib_BigOperators_BigOpWith___closed__8, &lp_mathlib_BigOperators_BigOpWith___closed__8_once, _init_lp_mathlib_BigOperators_BigOpWith___closed__8);
v___x_935_ = ((lean_object*)(lp_mathlib_BigOperators_BigOpWith___closed__3));
v___x_936_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBinder___closed__4));
v___x_937_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_937_, 0, v___x_936_);
lean_ctor_set(v___x_937_, 1, v___x_935_);
lean_ctor_set(v___x_937_, 2, v___x_934_);
return v___x_937_;
}
}
static lean_object* _init_lp_mathlib_BigOperators_BigOpWith___closed__10(void){
_start:
{
lean_object* v___x_938_; lean_object* v___x_939_; lean_object* v___x_940_; lean_object* v___x_941_; 
v___x_938_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBinder___closed__16));
v___x_939_ = lean_obj_once(&lp_mathlib_BigOperators_BigOpWith___closed__9, &lp_mathlib_BigOperators_BigOpWith___closed__9_once, _init_lp_mathlib_BigOperators_BigOpWith___closed__9);
v___x_940_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBinder___closed__4));
v___x_941_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_941_, 0, v___x_940_);
lean_ctor_set(v___x_941_, 1, v___x_939_);
lean_ctor_set(v___x_941_, 2, v___x_938_);
return v___x_941_;
}
}
static lean_object* _init_lp_mathlib_BigOperators_BigOpWith___closed__11(void){
_start:
{
lean_object* v___x_942_; lean_object* v___x_943_; lean_object* v___x_944_; lean_object* v___x_945_; 
v___x_942_ = lean_obj_once(&lp_mathlib_BigOperators_BigOpWith___closed__10, &lp_mathlib_BigOperators_BigOpWith___closed__10_once, _init_lp_mathlib_BigOperators_BigOpWith___closed__10);
v___x_943_ = ((lean_object*)(lp_mathlib_BigOperators_BigOpWith___closed__1));
v___x_944_ = ((lean_object*)(lp_mathlib_BigOperators_BigOpWith___closed__0));
v___x_945_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_945_, 0, v___x_944_);
lean_ctor_set(v___x_945_, 1, v___x_943_);
lean_ctor_set(v___x_945_, 2, v___x_942_);
return v___x_945_;
}
}
static lean_object* _init_lp_mathlib_BigOperators_BigOpWith(void){
_start:
{
lean_object* v___x_946_; 
v___x_946_ = lean_obj_once(&lp_mathlib_BigOperators_BigOpWith___closed__11, &lp_mathlib_BigOperators_BigOpWith___closed__11_once, _init_lp_mathlib_BigOperators_BigOpWith___closed__11);
return v___x_946_;
}
}
static lean_object* _init_lp_mathlib_BigOperators_bigsum___closed__5(void){
_start:
{
lean_object* v___x_958_; lean_object* v___x_959_; lean_object* v___x_960_; 
v___x_958_ = lp_mathlib_BigOperators_BigOpWith;
v___x_959_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBinder___closed__9));
v___x_960_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_960_, 0, v___x_959_);
lean_ctor_set(v___x_960_, 1, v___x_958_);
return v___x_960_;
}
}
static lean_object* _init_lp_mathlib_BigOperators_bigsum___closed__6(void){
_start:
{
lean_object* v___x_961_; lean_object* v___x_962_; lean_object* v___x_963_; lean_object* v___x_964_; 
v___x_961_ = lean_obj_once(&lp_mathlib_BigOperators_bigsum___closed__5, &lp_mathlib_BigOperators_bigsum___closed__5_once, _init_lp_mathlib_BigOperators_bigsum___closed__5);
v___x_962_ = ((lean_object*)(lp_mathlib_BigOperators_bigsum___closed__4));
v___x_963_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBinder___closed__4));
v___x_964_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_964_, 0, v___x_963_);
lean_ctor_set(v___x_964_, 1, v___x_962_);
lean_ctor_set(v___x_964_, 2, v___x_961_);
return v___x_964_;
}
}
static lean_object* _init_lp_mathlib_BigOperators_bigsum___closed__9(void){
_start:
{
lean_object* v___x_968_; lean_object* v___x_969_; lean_object* v___x_970_; lean_object* v___x_971_; 
v___x_968_ = ((lean_object*)(lp_mathlib_BigOperators_bigsum___closed__8));
v___x_969_ = lean_obj_once(&lp_mathlib_BigOperators_bigsum___closed__6, &lp_mathlib_BigOperators_bigsum___closed__6_once, _init_lp_mathlib_BigOperators_bigsum___closed__6);
v___x_970_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBinder___closed__4));
v___x_971_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_971_, 0, v___x_970_);
lean_ctor_set(v___x_971_, 1, v___x_969_);
lean_ctor_set(v___x_971_, 2, v___x_968_);
return v___x_971_;
}
}
static lean_object* _init_lp_mathlib_BigOperators_bigsum___closed__11(void){
_start:
{
lean_object* v___x_975_; lean_object* v___x_976_; lean_object* v___x_977_; lean_object* v___x_978_; 
v___x_975_ = ((lean_object*)(lp_mathlib_BigOperators_bigsum___closed__10));
v___x_976_ = lean_obj_once(&lp_mathlib_BigOperators_bigsum___closed__9, &lp_mathlib_BigOperators_bigsum___closed__9_once, _init_lp_mathlib_BigOperators_bigsum___closed__9);
v___x_977_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBinder___closed__4));
v___x_978_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_978_, 0, v___x_977_);
lean_ctor_set(v___x_978_, 1, v___x_976_);
lean_ctor_set(v___x_978_, 2, v___x_975_);
return v___x_978_;
}
}
static lean_object* _init_lp_mathlib_BigOperators_bigsum___closed__12(void){
_start:
{
lean_object* v___x_979_; lean_object* v___x_980_; lean_object* v___x_981_; lean_object* v___x_982_; 
v___x_979_ = lean_obj_once(&lp_mathlib_BigOperators_bigsum___closed__11, &lp_mathlib_BigOperators_bigsum___closed__11_once, _init_lp_mathlib_BigOperators_bigsum___closed__11);
v___x_980_ = lean_unsigned_to_nat(1022u);
v___x_981_ = ((lean_object*)(lp_mathlib_BigOperators_bigsum___closed__1));
v___x_982_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_982_, 0, v___x_981_);
lean_ctor_set(v___x_982_, 1, v___x_980_);
lean_ctor_set(v___x_982_, 2, v___x_979_);
return v___x_982_;
}
}
static lean_object* _init_lp_mathlib_BigOperators_bigsum(void){
_start:
{
lean_object* v___x_983_; 
v___x_983_ = lean_obj_once(&lp_mathlib_BigOperators_bigsum___closed__12, &lp_mathlib_BigOperators_bigsum___closed__12_once, _init_lp_mathlib_BigOperators_bigsum___closed__12);
return v___x_983_;
}
}
static lean_object* _init_lp_mathlib_BigOperators_bigprod___closed__5(void){
_start:
{
lean_object* v___x_995_; lean_object* v___x_996_; lean_object* v___x_997_; lean_object* v___x_998_; 
v___x_995_ = lean_obj_once(&lp_mathlib_BigOperators_bigsum___closed__5, &lp_mathlib_BigOperators_bigsum___closed__5_once, _init_lp_mathlib_BigOperators_bigsum___closed__5);
v___x_996_ = ((lean_object*)(lp_mathlib_BigOperators_bigprod___closed__4));
v___x_997_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBinder___closed__4));
v___x_998_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_998_, 0, v___x_997_);
lean_ctor_set(v___x_998_, 1, v___x_996_);
lean_ctor_set(v___x_998_, 2, v___x_995_);
return v___x_998_;
}
}
static lean_object* _init_lp_mathlib_BigOperators_bigprod___closed__6(void){
_start:
{
lean_object* v___x_999_; lean_object* v___x_1000_; lean_object* v___x_1001_; lean_object* v___x_1002_; 
v___x_999_ = ((lean_object*)(lp_mathlib_BigOperators_bigsum___closed__8));
v___x_1000_ = lean_obj_once(&lp_mathlib_BigOperators_bigprod___closed__5, &lp_mathlib_BigOperators_bigprod___closed__5_once, _init_lp_mathlib_BigOperators_bigprod___closed__5);
v___x_1001_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBinder___closed__4));
v___x_1002_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1002_, 0, v___x_1001_);
lean_ctor_set(v___x_1002_, 1, v___x_1000_);
lean_ctor_set(v___x_1002_, 2, v___x_999_);
return v___x_1002_;
}
}
static lean_object* _init_lp_mathlib_BigOperators_bigprod___closed__7(void){
_start:
{
lean_object* v___x_1003_; lean_object* v___x_1004_; lean_object* v___x_1005_; lean_object* v___x_1006_; 
v___x_1003_ = ((lean_object*)(lp_mathlib_BigOperators_bigsum___closed__10));
v___x_1004_ = lean_obj_once(&lp_mathlib_BigOperators_bigprod___closed__6, &lp_mathlib_BigOperators_bigprod___closed__6_once, _init_lp_mathlib_BigOperators_bigprod___closed__6);
v___x_1005_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBinder___closed__4));
v___x_1006_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1006_, 0, v___x_1005_);
lean_ctor_set(v___x_1006_, 1, v___x_1004_);
lean_ctor_set(v___x_1006_, 2, v___x_1003_);
return v___x_1006_;
}
}
static lean_object* _init_lp_mathlib_BigOperators_bigprod___closed__8(void){
_start:
{
lean_object* v___x_1007_; lean_object* v___x_1008_; lean_object* v___x_1009_; lean_object* v___x_1010_; 
v___x_1007_ = lean_obj_once(&lp_mathlib_BigOperators_bigprod___closed__7, &lp_mathlib_BigOperators_bigprod___closed__7_once, _init_lp_mathlib_BigOperators_bigprod___closed__7);
v___x_1008_ = lean_unsigned_to_nat(1022u);
v___x_1009_ = ((lean_object*)(lp_mathlib_BigOperators_bigprod___closed__1));
v___x_1010_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_1010_, 0, v___x_1009_);
lean_ctor_set(v___x_1010_, 1, v___x_1008_);
lean_ctor_set(v___x_1010_, 2, v___x_1007_);
return v___x_1010_;
}
}
static lean_object* _init_lp_mathlib_BigOperators_bigprod(void){
_start:
{
lean_object* v___x_1011_; 
v___x_1011_ = lean_obj_once(&lp_mathlib_BigOperators_bigprod___closed__8, &lp_mathlib_BigOperators_bigprod___closed__8_once, _init_lp_mathlib_BigOperators_bigprod___closed__8);
return v___x_1011_;
}
}
static lean_object* _init_lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__1(void){
_start:
{
lean_object* v___x_1013_; lean_object* v___x_1014_; 
v___x_1013_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__0));
v___x_1014_ = l_String_toRawSubstring_x27(v___x_1013_);
return v___x_1014_;
}
}
static lean_object* _init_lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__18(void){
_start:
{
lean_object* v___x_1057_; lean_object* v___x_1058_; 
v___x_1057_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__17));
v___x_1058_ = l_String_toRawSubstring_x27(v___x_1057_);
return v___x_1058_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1(lean_object* v_x_1083_, lean_object* v_a_1084_, lean_object* v_a_1085_){
_start:
{
lean_object* v___y_1087_; lean_object* v___y_1088_; lean_object* v___y_1089_; lean_object* v___y_1090_; lean_object* v___y_1091_; lean_object* v___y_1134_; lean_object* v___y_1135_; lean_object* v___y_1136_; lean_object* v_p_1137_; lean_object* v___y_1138_; lean_object* v___y_1139_; lean_object* v___x_1192_; uint8_t v___x_1193_; 
v___x_1192_ = ((lean_object*)(lp_mathlib_BigOperators_bigsum___closed__1));
lean_inc(v_x_1083_);
v___x_1193_ = l_Lean_Syntax_isOfKind(v_x_1083_, v___x_1192_);
if (v___x_1193_ == 0)
{
lean_object* v___x_1194_; lean_object* v___x_1195_; 
lean_dec(v_x_1083_);
v___x_1194_ = lean_box(1);
v___x_1195_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1195_, 0, v___x_1194_);
lean_ctor_set(v___x_1195_, 1, v_a_1085_);
return v___x_1195_;
}
else
{
lean_object* v___x_1196_; lean_object* v_bs_1197_; lean_object* v_hx_x3f_x3f_1199_; lean_object* v_p_x3f_1200_; lean_object* v___y_1201_; lean_object* v___y_1202_; lean_object* v___x_1289_; uint8_t v___x_1290_; 
v___x_1196_ = lean_unsigned_to_nat(1u);
v_bs_1197_ = l_Lean_Syntax_getArg(v_x_1083_, v___x_1196_);
v___x_1289_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBinders___closed__1));
lean_inc(v_bs_1197_);
v___x_1290_ = l_Lean_Syntax_isOfKind(v_bs_1197_, v___x_1289_);
if (v___x_1290_ == 0)
{
lean_object* v___x_1291_; lean_object* v___x_1292_; 
lean_dec(v_bs_1197_);
lean_dec(v_x_1083_);
v___x_1291_ = lean_box(1);
v___x_1292_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1292_, 0, v___x_1291_);
lean_ctor_set(v___x_1292_, 1, v_a_1085_);
return v___x_1292_;
}
else
{
lean_object* v___x_1293_; lean_object* v___x_1294_; uint8_t v___x_1295_; 
v___x_1293_ = lean_unsigned_to_nat(2u);
v___x_1294_ = l_Lean_Syntax_getArg(v_x_1083_, v___x_1293_);
v___x_1295_ = l_Lean_Syntax_isNone(v___x_1294_);
if (v___x_1295_ == 0)
{
uint8_t v___x_1296_; 
lean_inc(v___x_1294_);
v___x_1296_ = l_Lean_Syntax_matchesNull(v___x_1294_, v___x_1196_);
if (v___x_1296_ == 0)
{
lean_object* v___x_1297_; lean_object* v___x_1298_; 
lean_dec(v___x_1294_);
lean_dec(v_bs_1197_);
lean_dec(v_x_1083_);
v___x_1297_ = lean_box(1);
v___x_1298_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1298_, 0, v___x_1297_);
lean_ctor_set(v___x_1298_, 1, v_a_1085_);
return v___x_1298_;
}
else
{
lean_object* v___x_1299_; lean_object* v___x_1300_; lean_object* v_hx_x3f_x3f_1302_; lean_object* v___y_1303_; lean_object* v___y_1304_; lean_object* v___x_1308_; uint8_t v___x_1309_; 
v___x_1299_ = lean_unsigned_to_nat(0u);
v___x_1300_ = l_Lean_Syntax_getArg(v___x_1294_, v___x_1299_);
lean_dec(v___x_1294_);
v___x_1308_ = ((lean_object*)(lp_mathlib_BigOperators_BigOpWith___closed__1));
lean_inc(v___x_1300_);
v___x_1309_ = l_Lean_Syntax_isOfKind(v___x_1300_, v___x_1308_);
if (v___x_1309_ == 0)
{
lean_object* v___x_1310_; lean_object* v___x_1311_; 
lean_dec(v___x_1300_);
lean_dec(v_bs_1197_);
lean_dec(v_x_1083_);
v___x_1310_ = lean_box(1);
v___x_1311_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1311_, 0, v___x_1310_);
lean_ctor_set(v___x_1311_, 1, v_a_1085_);
return v___x_1311_;
}
else
{
lean_object* v___x_1312_; uint8_t v___x_1313_; 
v___x_1312_ = l_Lean_Syntax_getArg(v___x_1300_, v___x_1196_);
v___x_1313_ = l_Lean_Syntax_isNone(v___x_1312_);
if (v___x_1313_ == 0)
{
uint8_t v___x_1314_; 
lean_inc(v___x_1312_);
v___x_1314_ = l_Lean_Syntax_matchesNull(v___x_1312_, v___x_1293_);
if (v___x_1314_ == 0)
{
lean_object* v___x_1315_; lean_object* v___x_1316_; 
lean_dec(v___x_1312_);
lean_dec(v___x_1300_);
lean_dec(v_bs_1197_);
lean_dec(v_x_1083_);
v___x_1315_ = lean_box(1);
v___x_1316_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1316_, 0, v___x_1315_);
lean_ctor_set(v___x_1316_, 1, v_a_1085_);
return v___x_1316_;
}
else
{
lean_object* v_hx_x3f_x3f_1317_; lean_object* v___x_1318_; uint8_t v___x_1319_; 
v_hx_x3f_x3f_1317_ = l_Lean_Syntax_getArg(v___x_1312_, v___x_1299_);
lean_dec(v___x_1312_);
v___x_1318_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__32));
lean_inc(v_hx_x3f_x3f_1317_);
v___x_1319_ = l_Lean_Syntax_isOfKind(v_hx_x3f_x3f_1317_, v___x_1318_);
if (v___x_1319_ == 0)
{
lean_object* v___x_1320_; lean_object* v___x_1321_; 
lean_dec(v_hx_x3f_x3f_1317_);
lean_dec(v___x_1300_);
lean_dec(v_bs_1197_);
lean_dec(v_x_1083_);
v___x_1320_ = lean_box(1);
v___x_1321_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1321_, 0, v___x_1320_);
lean_ctor_set(v___x_1321_, 1, v_a_1085_);
return v___x_1321_;
}
else
{
lean_object* v___x_1322_; 
v___x_1322_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1322_, 0, v_hx_x3f_x3f_1317_);
v_hx_x3f_x3f_1302_ = v___x_1322_;
v___y_1303_ = v_a_1084_;
v___y_1304_ = v_a_1085_;
goto v___jp_1301_;
}
}
}
else
{
lean_object* v___x_1323_; 
lean_dec(v___x_1312_);
v___x_1323_ = lean_box(0);
v_hx_x3f_x3f_1302_ = v___x_1323_;
v___y_1303_ = v_a_1084_;
v___y_1304_ = v_a_1085_;
goto v___jp_1301_;
}
}
v___jp_1301_:
{
lean_object* v_p_x3f_1305_; lean_object* v___x_1306_; lean_object* v___x_1307_; 
v_p_x3f_1305_ = l_Lean_Syntax_getArg(v___x_1300_, v___x_1293_);
lean_dec(v___x_1300_);
v___x_1306_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1306_, 0, v_hx_x3f_x3f_1302_);
v___x_1307_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1307_, 0, v_p_x3f_1305_);
v_hx_x3f_x3f_1199_ = v___x_1306_;
v_p_x3f_1200_ = v___x_1307_;
v___y_1201_ = v___y_1303_;
v___y_1202_ = v___y_1304_;
goto v___jp_1198_;
}
}
}
else
{
lean_object* v___x_1324_; 
lean_dec(v___x_1294_);
v___x_1324_ = lean_box(0);
v_hx_x3f_x3f_1199_ = v___x_1324_;
v_p_x3f_1200_ = v___x_1324_;
v___y_1201_ = v_a_1084_;
v___y_1202_ = v_a_1085_;
goto v___jp_1198_;
}
}
v___jp_1198_:
{
lean_object* v___x_1203_; 
v___x_1203_ = lp_mathlib_BigOperators_processBigOpBinders(v_bs_1197_, v___y_1201_, v___y_1202_);
if (lean_obj_tag(v___x_1203_) == 0)
{
lean_object* v_a_1204_; lean_object* v_a_1205_; lean_object* v___x_1206_; lean_object* v_a_1207_; lean_object* v_a_1208_; lean_object* v___x_1210_; uint8_t v_isShared_1211_; uint8_t v_isSharedCheck_1279_; 
v_a_1204_ = lean_ctor_get(v___x_1203_, 0);
lean_inc_n(v_a_1204_, 2);
v_a_1205_ = lean_ctor_get(v___x_1203_, 1);
lean_inc(v_a_1205_);
lean_dec_ref_known(v___x_1203_, 2);
v___x_1206_ = lp_mathlib_BigOperators_bigOpBindersPattern(v_a_1204_, v___y_1201_, v_a_1205_);
v_a_1207_ = lean_ctor_get(v___x_1206_, 0);
v_a_1208_ = lean_ctor_get(v___x_1206_, 1);
v_isSharedCheck_1279_ = !lean_is_exclusive(v___x_1206_);
if (v_isSharedCheck_1279_ == 0)
{
v___x_1210_ = v___x_1206_;
v_isShared_1211_ = v_isSharedCheck_1279_;
goto v_resetjp_1209_;
}
else
{
lean_inc(v_a_1208_);
lean_inc(v_a_1207_);
lean_dec(v___x_1206_);
v___x_1210_ = lean_box(0);
v_isShared_1211_ = v_isSharedCheck_1279_;
goto v_resetjp_1209_;
}
v_resetjp_1209_:
{
lean_object* v___x_1212_; 
v___x_1212_ = lp_mathlib_BigOperators_bigOpBindersProd(v_a_1204_, v___y_1201_, v_a_1208_);
lean_dec(v_a_1204_);
if (lean_obj_tag(v___x_1212_) == 0)
{
lean_object* v_a_1213_; lean_object* v_a_1214_; lean_object* v___x_1216_; uint8_t v_isShared_1217_; uint8_t v_isSharedCheck_1269_; 
v_a_1213_ = lean_ctor_get(v___x_1212_, 0);
v_a_1214_ = lean_ctor_get(v___x_1212_, 1);
v_isSharedCheck_1269_ = !lean_is_exclusive(v___x_1212_);
if (v_isSharedCheck_1269_ == 0)
{
v___x_1216_ = v___x_1212_;
v_isShared_1217_ = v_isSharedCheck_1269_;
goto v_resetjp_1215_;
}
else
{
lean_inc(v_a_1214_);
lean_inc(v_a_1213_);
lean_dec(v___x_1212_);
v___x_1216_ = lean_box(0);
v_isShared_1217_ = v_isSharedCheck_1269_;
goto v_resetjp_1215_;
}
v_resetjp_1215_:
{
lean_object* v___x_1218_; lean_object* v___x_1219_; 
v___x_1218_ = lean_unsigned_to_nat(4u);
v___x_1219_ = l_Lean_Syntax_getArg(v_x_1083_, v___x_1218_);
lean_dec(v_x_1083_);
if (lean_obj_tag(v_hx_x3f_x3f_1199_) == 1)
{
lean_object* v_val_1220_; 
v_val_1220_ = lean_ctor_get(v_hx_x3f_x3f_1199_, 0);
lean_inc(v_val_1220_);
lean_dec_ref_known(v_hx_x3f_x3f_1199_, 1);
if (lean_obj_tag(v_val_1220_) == 1)
{
if (lean_obj_tag(v_p_x3f_1200_) == 0)
{
lean_dec_ref_known(v_val_1220_, 1);
lean_del_object(v___x_1216_);
lean_del_object(v___x_1210_);
v___y_1087_ = v_a_1213_;
v___y_1088_ = v___x_1219_;
v___y_1089_ = v_a_1207_;
v___y_1090_ = v___y_1201_;
v___y_1091_ = v_a_1214_;
goto v___jp_1086_;
}
else
{
lean_object* v_val_1221_; lean_object* v_val_1222_; lean_object* v_quotContext_1223_; lean_object* v_currMacroScope_1224_; lean_object* v_ref_1225_; uint8_t v___x_1226_; lean_object* v___x_1227_; lean_object* v___x_1228_; lean_object* v___x_1229_; lean_object* v___x_1230_; lean_object* v___x_1231_; lean_object* v___x_1232_; lean_object* v___x_1233_; lean_object* v___x_1234_; lean_object* v___x_1235_; lean_object* v___x_1236_; lean_object* v___x_1238_; 
v_val_1221_ = lean_ctor_get(v_val_1220_, 0);
lean_inc(v_val_1221_);
lean_dec_ref_known(v_val_1220_, 1);
v_val_1222_ = lean_ctor_get(v_p_x3f_1200_, 0);
lean_inc(v_val_1222_);
lean_dec_ref_known(v_p_x3f_1200_, 1);
v_quotContext_1223_ = lean_ctor_get(v___y_1201_, 1);
v_currMacroScope_1224_ = lean_ctor_get(v___y_1201_, 2);
v_ref_1225_ = lean_ctor_get(v___y_1201_, 5);
v___x_1226_ = 0;
v___x_1227_ = l_Lean_SourceInfo_fromRef(v_ref_1225_, v___x_1226_);
v___x_1228_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__18));
v___x_1229_ = lean_obj_once(&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__1, &lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__1_once, _init_lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__1);
v___x_1230_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__3));
lean_inc(v_currMacroScope_1224_);
lean_inc(v_quotContext_1223_);
v___x_1231_ = l_Lean_addMacroScope(v_quotContext_1223_, v___x_1230_, v_currMacroScope_1224_);
v___x_1232_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__5));
lean_inc_n(v___x_1227_, 2);
v___x_1233_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1233_, 0, v___x_1227_);
lean_ctor_set(v___x_1233_, 1, v___x_1229_);
lean_ctor_set(v___x_1233_, 2, v___x_1231_);
lean_ctor_set(v___x_1233_, 3, v___x_1232_);
v___x_1234_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__25));
v___x_1235_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__12));
v___x_1236_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__13));
if (v_isShared_1211_ == 0)
{
lean_ctor_set_tag(v___x_1210_, 2);
lean_ctor_set(v___x_1210_, 1, v___x_1235_);
lean_ctor_set(v___x_1210_, 0, v___x_1227_);
v___x_1238_ = v___x_1210_;
goto v_reusejp_1237_;
}
else
{
lean_object* v_reuseFailAlloc_1266_; 
v_reuseFailAlloc_1266_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1266_, 0, v___x_1227_);
lean_ctor_set(v_reuseFailAlloc_1266_, 1, v___x_1235_);
v___x_1238_ = v_reuseFailAlloc_1266_;
goto v_reusejp_1237_;
}
v_reusejp_1237_:
{
lean_object* v___x_1239_; lean_object* v___x_1240_; lean_object* v___x_1241_; lean_object* v___x_1242_; lean_object* v___x_1243_; lean_object* v___x_1244_; lean_object* v___x_1245_; lean_object* v___x_1246_; lean_object* v___x_1247_; lean_object* v___x_1248_; lean_object* v___x_1249_; lean_object* v___x_1250_; lean_object* v___x_1251_; lean_object* v___x_1252_; lean_object* v___x_1253_; lean_object* v___x_1254_; lean_object* v___x_1255_; lean_object* v___x_1256_; lean_object* v___x_1257_; lean_object* v___x_1258_; lean_object* v___x_1259_; lean_object* v___x_1260_; lean_object* v___x_1261_; lean_object* v___x_1262_; lean_object* v___x_1264_; 
v___x_1239_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__15));
lean_inc_n(v___x_1227_, 13);
v___x_1240_ = l_Lean_Syntax_node1(v___x_1227_, v___x_1234_, v_a_1207_);
v___x_1241_ = lean_obj_once(&lp_mathlib_BigOperators_bigOpBindersPattern___closed__0, &lp_mathlib_BigOperators_bigOpBindersPattern___closed__0_once, _init_lp_mathlib_BigOperators_bigOpBindersPattern___closed__0);
v___x_1242_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1242_, 0, v___x_1227_);
lean_ctor_set(v___x_1242_, 1, v___x_1234_);
lean_ctor_set(v___x_1242_, 2, v___x_1241_);
v___x_1243_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__16));
v___x_1244_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1244_, 0, v___x_1227_);
lean_ctor_set(v___x_1244_, 1, v___x_1243_);
v___x_1245_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__24));
v___x_1246_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__25));
v___x_1247_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1247_, 0, v___x_1227_);
lean_ctor_set(v___x_1247_, 1, v___x_1246_);
v___x_1248_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__61));
v___x_1249_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1249_, 0, v___x_1227_);
lean_ctor_set(v___x_1249_, 1, v___x_1248_);
v___x_1250_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__26));
v___x_1251_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1251_, 0, v___x_1227_);
lean_ctor_set(v___x_1251_, 1, v___x_1250_);
v___x_1252_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__27));
v___x_1253_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1253_, 0, v___x_1227_);
lean_ctor_set(v___x_1253_, 1, v___x_1252_);
v___x_1254_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__29));
v___x_1255_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__30));
v___x_1256_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1256_, 0, v___x_1227_);
lean_ctor_set(v___x_1256_, 1, v___x_1255_);
v___x_1257_ = l_Lean_Syntax_node1(v___x_1227_, v___x_1254_, v___x_1256_);
v___x_1258_ = l_Lean_Syntax_node8(v___x_1227_, v___x_1245_, v___x_1247_, v_val_1221_, v___x_1249_, v_val_1222_, v___x_1251_, v___x_1219_, v___x_1253_, v___x_1257_);
v___x_1259_ = l_Lean_Syntax_node4(v___x_1227_, v___x_1239_, v___x_1240_, v___x_1242_, v___x_1244_, v___x_1258_);
v___x_1260_ = l_Lean_Syntax_node2(v___x_1227_, v___x_1236_, v___x_1238_, v___x_1259_);
v___x_1261_ = l_Lean_Syntax_node2(v___x_1227_, v___x_1234_, v_a_1213_, v___x_1260_);
v___x_1262_ = l_Lean_Syntax_node2(v___x_1227_, v___x_1228_, v___x_1233_, v___x_1261_);
if (v_isShared_1217_ == 0)
{
lean_ctor_set(v___x_1216_, 0, v___x_1262_);
v___x_1264_ = v___x_1216_;
goto v_reusejp_1263_;
}
else
{
lean_object* v_reuseFailAlloc_1265_; 
v_reuseFailAlloc_1265_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1265_, 0, v___x_1262_);
lean_ctor_set(v_reuseFailAlloc_1265_, 1, v_a_1214_);
v___x_1264_ = v_reuseFailAlloc_1265_;
goto v_reusejp_1263_;
}
v_reusejp_1263_:
{
return v___x_1264_;
}
}
}
}
else
{
lean_dec(v_val_1220_);
lean_del_object(v___x_1216_);
lean_del_object(v___x_1210_);
if (lean_obj_tag(v_p_x3f_1200_) == 0)
{
v___y_1087_ = v_a_1213_;
v___y_1088_ = v___x_1219_;
v___y_1089_ = v_a_1207_;
v___y_1090_ = v___y_1201_;
v___y_1091_ = v_a_1214_;
goto v___jp_1086_;
}
else
{
lean_object* v_val_1267_; 
v_val_1267_ = lean_ctor_get(v_p_x3f_1200_, 0);
lean_inc(v_val_1267_);
lean_dec_ref_known(v_p_x3f_1200_, 1);
v___y_1134_ = v_a_1213_;
v___y_1135_ = v___x_1219_;
v___y_1136_ = v_a_1207_;
v_p_1137_ = v_val_1267_;
v___y_1138_ = v___y_1201_;
v___y_1139_ = v_a_1214_;
goto v___jp_1133_;
}
}
}
else
{
lean_del_object(v___x_1216_);
lean_del_object(v___x_1210_);
lean_dec(v_hx_x3f_x3f_1199_);
if (lean_obj_tag(v_p_x3f_1200_) == 0)
{
v___y_1087_ = v_a_1213_;
v___y_1088_ = v___x_1219_;
v___y_1089_ = v_a_1207_;
v___y_1090_ = v___y_1201_;
v___y_1091_ = v_a_1214_;
goto v___jp_1086_;
}
else
{
lean_object* v_val_1268_; 
v_val_1268_ = lean_ctor_get(v_p_x3f_1200_, 0);
lean_inc(v_val_1268_);
lean_dec_ref_known(v_p_x3f_1200_, 1);
v___y_1134_ = v_a_1213_;
v___y_1135_ = v___x_1219_;
v___y_1136_ = v_a_1207_;
v_p_1137_ = v_val_1268_;
v___y_1138_ = v___y_1201_;
v___y_1139_ = v_a_1214_;
goto v___jp_1133_;
}
}
}
}
else
{
lean_object* v_a_1270_; lean_object* v_a_1271_; lean_object* v___x_1273_; uint8_t v_isShared_1274_; uint8_t v_isSharedCheck_1278_; 
lean_del_object(v___x_1210_);
lean_dec(v_a_1207_);
lean_dec(v_p_x3f_1200_);
lean_dec(v_hx_x3f_x3f_1199_);
lean_dec(v_x_1083_);
v_a_1270_ = lean_ctor_get(v___x_1212_, 0);
v_a_1271_ = lean_ctor_get(v___x_1212_, 1);
v_isSharedCheck_1278_ = !lean_is_exclusive(v___x_1212_);
if (v_isSharedCheck_1278_ == 0)
{
v___x_1273_ = v___x_1212_;
v_isShared_1274_ = v_isSharedCheck_1278_;
goto v_resetjp_1272_;
}
else
{
lean_inc(v_a_1271_);
lean_inc(v_a_1270_);
lean_dec(v___x_1212_);
v___x_1273_ = lean_box(0);
v_isShared_1274_ = v_isSharedCheck_1278_;
goto v_resetjp_1272_;
}
v_resetjp_1272_:
{
lean_object* v___x_1276_; 
if (v_isShared_1274_ == 0)
{
v___x_1276_ = v___x_1273_;
goto v_reusejp_1275_;
}
else
{
lean_object* v_reuseFailAlloc_1277_; 
v_reuseFailAlloc_1277_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1277_, 0, v_a_1270_);
lean_ctor_set(v_reuseFailAlloc_1277_, 1, v_a_1271_);
v___x_1276_ = v_reuseFailAlloc_1277_;
goto v_reusejp_1275_;
}
v_reusejp_1275_:
{
return v___x_1276_;
}
}
}
}
}
else
{
lean_object* v_a_1280_; lean_object* v_a_1281_; lean_object* v___x_1283_; uint8_t v_isShared_1284_; uint8_t v_isSharedCheck_1288_; 
lean_dec(v_p_x3f_1200_);
lean_dec(v_hx_x3f_x3f_1199_);
lean_dec(v_x_1083_);
v_a_1280_ = lean_ctor_get(v___x_1203_, 0);
v_a_1281_ = lean_ctor_get(v___x_1203_, 1);
v_isSharedCheck_1288_ = !lean_is_exclusive(v___x_1203_);
if (v_isSharedCheck_1288_ == 0)
{
v___x_1283_ = v___x_1203_;
v_isShared_1284_ = v_isSharedCheck_1288_;
goto v_resetjp_1282_;
}
else
{
lean_inc(v_a_1281_);
lean_inc(v_a_1280_);
lean_dec(v___x_1203_);
v___x_1283_ = lean_box(0);
v_isShared_1284_ = v_isSharedCheck_1288_;
goto v_resetjp_1282_;
}
v_resetjp_1282_:
{
lean_object* v___x_1286_; 
if (v_isShared_1284_ == 0)
{
v___x_1286_ = v___x_1283_;
goto v_reusejp_1285_;
}
else
{
lean_object* v_reuseFailAlloc_1287_; 
v_reuseFailAlloc_1287_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1287_, 0, v_a_1280_);
lean_ctor_set(v_reuseFailAlloc_1287_, 1, v_a_1281_);
v___x_1286_ = v_reuseFailAlloc_1287_;
goto v_reusejp_1285_;
}
v_reusejp_1285_:
{
return v___x_1286_;
}
}
}
}
}
v___jp_1086_:
{
lean_object* v_quotContext_1092_; lean_object* v_currMacroScope_1093_; lean_object* v_ref_1094_; uint8_t v___x_1095_; lean_object* v___x_1096_; lean_object* v___x_1097_; lean_object* v___x_1098_; lean_object* v___x_1099_; lean_object* v___x_1100_; lean_object* v___x_1101_; lean_object* v___x_1102_; lean_object* v___x_1103_; lean_object* v___x_1104_; lean_object* v___x_1105_; lean_object* v___x_1106_; lean_object* v___x_1107_; lean_object* v___x_1108_; lean_object* v___x_1109_; lean_object* v___x_1110_; lean_object* v___x_1111_; lean_object* v___x_1112_; lean_object* v___x_1113_; lean_object* v___x_1114_; lean_object* v___x_1115_; lean_object* v___x_1116_; lean_object* v___x_1117_; lean_object* v___x_1118_; lean_object* v___x_1119_; lean_object* v___x_1120_; lean_object* v___x_1121_; lean_object* v___x_1122_; lean_object* v___x_1123_; lean_object* v___x_1124_; lean_object* v___x_1125_; lean_object* v___x_1126_; lean_object* v___x_1127_; lean_object* v___x_1128_; lean_object* v___x_1129_; lean_object* v___x_1130_; lean_object* v___x_1131_; lean_object* v___x_1132_; 
v_quotContext_1092_ = lean_ctor_get(v___y_1090_, 1);
v_currMacroScope_1093_ = lean_ctor_get(v___y_1090_, 2);
v_ref_1094_ = lean_ctor_get(v___y_1090_, 5);
v___x_1095_ = 0;
v___x_1096_ = l_Lean_SourceInfo_fromRef(v_ref_1094_, v___x_1095_);
v___x_1097_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__18));
v___x_1098_ = lean_obj_once(&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__1, &lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__1_once, _init_lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__1);
v___x_1099_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__3));
lean_inc_n(v_currMacroScope_1093_, 2);
lean_inc_n(v_quotContext_1092_, 2);
v___x_1100_ = l_Lean_addMacroScope(v_quotContext_1092_, v___x_1099_, v_currMacroScope_1093_);
v___x_1101_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__5));
lean_inc_n(v___x_1096_, 14);
v___x_1102_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1102_, 0, v___x_1096_);
lean_ctor_set(v___x_1102_, 1, v___x_1098_);
lean_ctor_set(v___x_1102_, 2, v___x_1100_);
lean_ctor_set(v___x_1102_, 3, v___x_1101_);
v___x_1103_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__25));
v___x_1104_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__65));
v___x_1105_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__52));
v___x_1106_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__53));
v___x_1107_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1107_, 0, v___x_1096_);
lean_ctor_set(v___x_1107_, 1, v___x_1106_);
v___x_1108_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__55));
v___x_1109_ = lean_obj_once(&lp_mathlib_BigOperators_processBigOpBinder___closed__57, &lp_mathlib_BigOperators_processBigOpBinder___closed__57_once, _init_lp_mathlib_BigOperators_processBigOpBinder___closed__57);
v___x_1110_ = lean_box(0);
v___x_1111_ = l_Lean_addMacroScope(v_quotContext_1092_, v___x_1110_, v_currMacroScope_1093_);
v___x_1112_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__11));
v___x_1113_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1113_, 0, v___x_1096_);
lean_ctor_set(v___x_1113_, 1, v___x_1109_);
lean_ctor_set(v___x_1113_, 2, v___x_1111_);
lean_ctor_set(v___x_1113_, 3, v___x_1112_);
v___x_1114_ = l_Lean_Syntax_node1(v___x_1096_, v___x_1108_, v___x_1113_);
v___x_1115_ = l_Lean_Syntax_node2(v___x_1096_, v___x_1105_, v___x_1107_, v___x_1114_);
v___x_1116_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__12));
v___x_1117_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__13));
v___x_1118_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1118_, 0, v___x_1096_);
lean_ctor_set(v___x_1118_, 1, v___x_1116_);
v___x_1119_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__15));
v___x_1120_ = l_Lean_Syntax_node1(v___x_1096_, v___x_1103_, v___y_1089_);
v___x_1121_ = lean_obj_once(&lp_mathlib_BigOperators_bigOpBindersPattern___closed__0, &lp_mathlib_BigOperators_bigOpBindersPattern___closed__0_once, _init_lp_mathlib_BigOperators_bigOpBindersPattern___closed__0);
v___x_1122_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1122_, 0, v___x_1096_);
lean_ctor_set(v___x_1122_, 1, v___x_1103_);
lean_ctor_set(v___x_1122_, 2, v___x_1121_);
v___x_1123_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__16));
v___x_1124_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1124_, 0, v___x_1096_);
lean_ctor_set(v___x_1124_, 1, v___x_1123_);
v___x_1125_ = l_Lean_Syntax_node4(v___x_1096_, v___x_1119_, v___x_1120_, v___x_1122_, v___x_1124_, v___y_1088_);
v___x_1126_ = l_Lean_Syntax_node2(v___x_1096_, v___x_1117_, v___x_1118_, v___x_1125_);
v___x_1127_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBinderParenthesized___closed__5));
v___x_1128_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1128_, 0, v___x_1096_);
lean_ctor_set(v___x_1128_, 1, v___x_1127_);
v___x_1129_ = l_Lean_Syntax_node3(v___x_1096_, v___x_1104_, v___x_1115_, v___x_1126_, v___x_1128_);
v___x_1130_ = l_Lean_Syntax_node2(v___x_1096_, v___x_1103_, v___y_1087_, v___x_1129_);
v___x_1131_ = l_Lean_Syntax_node2(v___x_1096_, v___x_1097_, v___x_1102_, v___x_1130_);
v___x_1132_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1132_, 0, v___x_1131_);
lean_ctor_set(v___x_1132_, 1, v___y_1091_);
return v___x_1132_;
}
v___jp_1133_:
{
lean_object* v_quotContext_1140_; lean_object* v_currMacroScope_1141_; lean_object* v_ref_1142_; uint8_t v___x_1143_; lean_object* v___x_1144_; lean_object* v___x_1145_; lean_object* v___x_1146_; lean_object* v___x_1147_; lean_object* v___x_1148_; lean_object* v___x_1149_; lean_object* v___x_1150_; lean_object* v___x_1151_; lean_object* v___x_1152_; lean_object* v___x_1153_; lean_object* v___x_1154_; lean_object* v___x_1155_; lean_object* v___x_1156_; lean_object* v___x_1157_; lean_object* v___x_1158_; lean_object* v___x_1159_; lean_object* v___x_1160_; lean_object* v___x_1161_; lean_object* v___x_1162_; lean_object* v___x_1163_; lean_object* v___x_1164_; lean_object* v___x_1165_; lean_object* v___x_1166_; lean_object* v___x_1167_; lean_object* v___x_1168_; lean_object* v___x_1169_; lean_object* v___x_1170_; lean_object* v___x_1171_; lean_object* v___x_1172_; lean_object* v___x_1173_; lean_object* v___x_1174_; lean_object* v___x_1175_; lean_object* v___x_1176_; lean_object* v___x_1177_; lean_object* v___x_1178_; lean_object* v___x_1179_; lean_object* v___x_1180_; lean_object* v___x_1181_; lean_object* v___x_1182_; lean_object* v___x_1183_; lean_object* v___x_1184_; lean_object* v___x_1185_; lean_object* v___x_1186_; lean_object* v___x_1187_; lean_object* v___x_1188_; lean_object* v___x_1189_; lean_object* v___x_1190_; lean_object* v___x_1191_; 
v_quotContext_1140_ = lean_ctor_get(v___y_1138_, 1);
v_currMacroScope_1141_ = lean_ctor_get(v___y_1138_, 2);
v_ref_1142_ = lean_ctor_get(v___y_1138_, 5);
v___x_1143_ = 0;
v___x_1144_ = l_Lean_SourceInfo_fromRef(v_ref_1142_, v___x_1143_);
v___x_1145_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__18));
v___x_1146_ = lean_obj_once(&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__1, &lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__1_once, _init_lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__1);
v___x_1147_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__3));
lean_inc_n(v_currMacroScope_1141_, 3);
lean_inc_n(v_quotContext_1140_, 3);
v___x_1148_ = l_Lean_addMacroScope(v_quotContext_1140_, v___x_1147_, v_currMacroScope_1141_);
v___x_1149_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__5));
lean_inc_n(v___x_1144_, 21);
v___x_1150_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1150_, 0, v___x_1144_);
lean_ctor_set(v___x_1150_, 1, v___x_1146_);
lean_ctor_set(v___x_1150_, 2, v___x_1148_);
lean_ctor_set(v___x_1150_, 3, v___x_1149_);
v___x_1151_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__25));
v___x_1152_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__65));
v___x_1153_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__52));
v___x_1154_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__53));
v___x_1155_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1155_, 0, v___x_1144_);
lean_ctor_set(v___x_1155_, 1, v___x_1154_);
v___x_1156_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__55));
v___x_1157_ = lean_obj_once(&lp_mathlib_BigOperators_processBigOpBinder___closed__57, &lp_mathlib_BigOperators_processBigOpBinder___closed__57_once, _init_lp_mathlib_BigOperators_processBigOpBinder___closed__57);
v___x_1158_ = lean_box(0);
v___x_1159_ = l_Lean_addMacroScope(v_quotContext_1140_, v___x_1158_, v_currMacroScope_1141_);
v___x_1160_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__11));
v___x_1161_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1161_, 0, v___x_1144_);
lean_ctor_set(v___x_1161_, 1, v___x_1157_);
lean_ctor_set(v___x_1161_, 2, v___x_1159_);
lean_ctor_set(v___x_1161_, 3, v___x_1160_);
v___x_1162_ = l_Lean_Syntax_node1(v___x_1144_, v___x_1156_, v___x_1161_);
v___x_1163_ = l_Lean_Syntax_node2(v___x_1144_, v___x_1153_, v___x_1155_, v___x_1162_);
v___x_1164_ = lean_obj_once(&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__18, &lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__18_once, _init_lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__18);
v___x_1165_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__20));
v___x_1166_ = l_Lean_addMacroScope(v_quotContext_1140_, v___x_1165_, v_currMacroScope_1141_);
v___x_1167_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__22));
v___x_1168_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1168_, 0, v___x_1144_);
lean_ctor_set(v___x_1168_, 1, v___x_1164_);
lean_ctor_set(v___x_1168_, 2, v___x_1166_);
lean_ctor_set(v___x_1168_, 3, v___x_1167_);
v___x_1169_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__12));
v___x_1170_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__13));
v___x_1171_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1171_, 0, v___x_1144_);
lean_ctor_set(v___x_1171_, 1, v___x_1169_);
v___x_1172_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__15));
v___x_1173_ = l_Lean_Syntax_node1(v___x_1144_, v___x_1151_, v___y_1136_);
v___x_1174_ = lean_obj_once(&lp_mathlib_BigOperators_bigOpBindersPattern___closed__0, &lp_mathlib_BigOperators_bigOpBindersPattern___closed__0_once, _init_lp_mathlib_BigOperators_bigOpBindersPattern___closed__0);
v___x_1175_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1175_, 0, v___x_1144_);
lean_ctor_set(v___x_1175_, 1, v___x_1151_);
lean_ctor_set(v___x_1175_, 2, v___x_1174_);
v___x_1176_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__16));
v___x_1177_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1177_, 0, v___x_1144_);
lean_ctor_set(v___x_1177_, 1, v___x_1176_);
lean_inc_ref(v___x_1177_);
lean_inc_ref(v___x_1175_);
lean_inc(v___x_1173_);
v___x_1178_ = l_Lean_Syntax_node4(v___x_1144_, v___x_1172_, v___x_1173_, v___x_1175_, v___x_1177_, v_p_1137_);
lean_inc_ref(v___x_1171_);
v___x_1179_ = l_Lean_Syntax_node2(v___x_1144_, v___x_1170_, v___x_1171_, v___x_1178_);
v___x_1180_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBinderParenthesized___closed__5));
v___x_1181_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1181_, 0, v___x_1144_);
lean_ctor_set(v___x_1181_, 1, v___x_1180_);
lean_inc_ref_n(v___x_1181_, 2);
lean_inc_n(v___x_1163_, 2);
v___x_1182_ = l_Lean_Syntax_node3(v___x_1144_, v___x_1152_, v___x_1163_, v___x_1179_, v___x_1181_);
v___x_1183_ = l_Lean_Syntax_node2(v___x_1144_, v___x_1151_, v___x_1182_, v___y_1134_);
v___x_1184_ = l_Lean_Syntax_node2(v___x_1144_, v___x_1145_, v___x_1168_, v___x_1183_);
v___x_1185_ = l_Lean_Syntax_node3(v___x_1144_, v___x_1152_, v___x_1163_, v___x_1184_, v___x_1181_);
v___x_1186_ = l_Lean_Syntax_node4(v___x_1144_, v___x_1172_, v___x_1173_, v___x_1175_, v___x_1177_, v___y_1135_);
v___x_1187_ = l_Lean_Syntax_node2(v___x_1144_, v___x_1170_, v___x_1171_, v___x_1186_);
v___x_1188_ = l_Lean_Syntax_node3(v___x_1144_, v___x_1152_, v___x_1163_, v___x_1187_, v___x_1181_);
v___x_1189_ = l_Lean_Syntax_node2(v___x_1144_, v___x_1151_, v___x_1185_, v___x_1188_);
v___x_1190_ = l_Lean_Syntax_node2(v___x_1144_, v___x_1145_, v___x_1150_, v___x_1189_);
v___x_1191_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1191_, 0, v___x_1190_);
lean_ctor_set(v___x_1191_, 1, v___y_1139_);
return v___x_1191_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___boxed(lean_object* v_x_1325_, lean_object* v_a_1326_, lean_object* v_a_1327_){
_start:
{
lean_object* v_res_1328_; 
v_res_1328_ = lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1(v_x_1325_, v_a_1326_, v_a_1327_);
lean_dec_ref(v_a_1326_);
return v_res_1328_;
}
}
static lean_object* _init_lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigprod__1___closed__1(void){
_start:
{
lean_object* v___x_1330_; lean_object* v___x_1331_; 
v___x_1330_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigprod__1___closed__0));
v___x_1331_ = l_String_toRawSubstring_x27(v___x_1330_);
return v___x_1331_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigprod__1(lean_object* v_x_1343_, lean_object* v_a_1344_, lean_object* v_a_1345_){
_start:
{
lean_object* v___y_1347_; lean_object* v___y_1348_; lean_object* v___y_1349_; lean_object* v___y_1350_; lean_object* v___y_1351_; lean_object* v___y_1394_; lean_object* v___y_1395_; lean_object* v___y_1396_; lean_object* v_p_1397_; lean_object* v___y_1398_; lean_object* v___y_1399_; lean_object* v___x_1452_; uint8_t v___x_1453_; 
v___x_1452_ = ((lean_object*)(lp_mathlib_BigOperators_bigprod___closed__1));
lean_inc(v_x_1343_);
v___x_1453_ = l_Lean_Syntax_isOfKind(v_x_1343_, v___x_1452_);
if (v___x_1453_ == 0)
{
lean_object* v___x_1454_; lean_object* v___x_1455_; 
lean_dec(v_x_1343_);
v___x_1454_ = lean_box(1);
v___x_1455_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1455_, 0, v___x_1454_);
lean_ctor_set(v___x_1455_, 1, v_a_1345_);
return v___x_1455_;
}
else
{
lean_object* v___x_1456_; lean_object* v_bs_1457_; lean_object* v_hx_x3f_x3f_1459_; lean_object* v_p_x3f_1460_; lean_object* v___y_1461_; lean_object* v___y_1462_; lean_object* v___x_1549_; uint8_t v___x_1550_; 
v___x_1456_ = lean_unsigned_to_nat(1u);
v_bs_1457_ = l_Lean_Syntax_getArg(v_x_1343_, v___x_1456_);
v___x_1549_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBinders___closed__1));
lean_inc(v_bs_1457_);
v___x_1550_ = l_Lean_Syntax_isOfKind(v_bs_1457_, v___x_1549_);
if (v___x_1550_ == 0)
{
lean_object* v___x_1551_; lean_object* v___x_1552_; 
lean_dec(v_bs_1457_);
lean_dec(v_x_1343_);
v___x_1551_ = lean_box(1);
v___x_1552_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1552_, 0, v___x_1551_);
lean_ctor_set(v___x_1552_, 1, v_a_1345_);
return v___x_1552_;
}
else
{
lean_object* v___x_1553_; lean_object* v___x_1554_; uint8_t v___x_1555_; 
v___x_1553_ = lean_unsigned_to_nat(2u);
v___x_1554_ = l_Lean_Syntax_getArg(v_x_1343_, v___x_1553_);
v___x_1555_ = l_Lean_Syntax_isNone(v___x_1554_);
if (v___x_1555_ == 0)
{
uint8_t v___x_1556_; 
lean_inc(v___x_1554_);
v___x_1556_ = l_Lean_Syntax_matchesNull(v___x_1554_, v___x_1456_);
if (v___x_1556_ == 0)
{
lean_object* v___x_1557_; lean_object* v___x_1558_; 
lean_dec(v___x_1554_);
lean_dec(v_bs_1457_);
lean_dec(v_x_1343_);
v___x_1557_ = lean_box(1);
v___x_1558_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1558_, 0, v___x_1557_);
lean_ctor_set(v___x_1558_, 1, v_a_1345_);
return v___x_1558_;
}
else
{
lean_object* v___x_1559_; lean_object* v___x_1560_; lean_object* v_hx_x3f_x3f_1562_; lean_object* v___y_1563_; lean_object* v___y_1564_; lean_object* v___x_1568_; uint8_t v___x_1569_; 
v___x_1559_ = lean_unsigned_to_nat(0u);
v___x_1560_ = l_Lean_Syntax_getArg(v___x_1554_, v___x_1559_);
lean_dec(v___x_1554_);
v___x_1568_ = ((lean_object*)(lp_mathlib_BigOperators_BigOpWith___closed__1));
lean_inc(v___x_1560_);
v___x_1569_ = l_Lean_Syntax_isOfKind(v___x_1560_, v___x_1568_);
if (v___x_1569_ == 0)
{
lean_object* v___x_1570_; lean_object* v___x_1571_; 
lean_dec(v___x_1560_);
lean_dec(v_bs_1457_);
lean_dec(v_x_1343_);
v___x_1570_ = lean_box(1);
v___x_1571_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1571_, 0, v___x_1570_);
lean_ctor_set(v___x_1571_, 1, v_a_1345_);
return v___x_1571_;
}
else
{
lean_object* v___x_1572_; uint8_t v___x_1573_; 
v___x_1572_ = l_Lean_Syntax_getArg(v___x_1560_, v___x_1456_);
v___x_1573_ = l_Lean_Syntax_isNone(v___x_1572_);
if (v___x_1573_ == 0)
{
uint8_t v___x_1574_; 
lean_inc(v___x_1572_);
v___x_1574_ = l_Lean_Syntax_matchesNull(v___x_1572_, v___x_1553_);
if (v___x_1574_ == 0)
{
lean_object* v___x_1575_; lean_object* v___x_1576_; 
lean_dec(v___x_1572_);
lean_dec(v___x_1560_);
lean_dec(v_bs_1457_);
lean_dec(v_x_1343_);
v___x_1575_ = lean_box(1);
v___x_1576_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1576_, 0, v___x_1575_);
lean_ctor_set(v___x_1576_, 1, v_a_1345_);
return v___x_1576_;
}
else
{
lean_object* v_hx_x3f_x3f_1577_; lean_object* v___x_1578_; uint8_t v___x_1579_; 
v_hx_x3f_x3f_1577_ = l_Lean_Syntax_getArg(v___x_1572_, v___x_1559_);
lean_dec(v___x_1572_);
v___x_1578_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__32));
lean_inc(v_hx_x3f_x3f_1577_);
v___x_1579_ = l_Lean_Syntax_isOfKind(v_hx_x3f_x3f_1577_, v___x_1578_);
if (v___x_1579_ == 0)
{
lean_object* v___x_1580_; lean_object* v___x_1581_; 
lean_dec(v_hx_x3f_x3f_1577_);
lean_dec(v___x_1560_);
lean_dec(v_bs_1457_);
lean_dec(v_x_1343_);
v___x_1580_ = lean_box(1);
v___x_1581_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1581_, 0, v___x_1580_);
lean_ctor_set(v___x_1581_, 1, v_a_1345_);
return v___x_1581_;
}
else
{
lean_object* v___x_1582_; 
v___x_1582_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1582_, 0, v_hx_x3f_x3f_1577_);
v_hx_x3f_x3f_1562_ = v___x_1582_;
v___y_1563_ = v_a_1344_;
v___y_1564_ = v_a_1345_;
goto v___jp_1561_;
}
}
}
else
{
lean_object* v___x_1583_; 
lean_dec(v___x_1572_);
v___x_1583_ = lean_box(0);
v_hx_x3f_x3f_1562_ = v___x_1583_;
v___y_1563_ = v_a_1344_;
v___y_1564_ = v_a_1345_;
goto v___jp_1561_;
}
}
v___jp_1561_:
{
lean_object* v_p_x3f_1565_; lean_object* v___x_1566_; lean_object* v___x_1567_; 
v_p_x3f_1565_ = l_Lean_Syntax_getArg(v___x_1560_, v___x_1553_);
lean_dec(v___x_1560_);
v___x_1566_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1566_, 0, v_hx_x3f_x3f_1562_);
v___x_1567_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1567_, 0, v_p_x3f_1565_);
v_hx_x3f_x3f_1459_ = v___x_1566_;
v_p_x3f_1460_ = v___x_1567_;
v___y_1461_ = v___y_1563_;
v___y_1462_ = v___y_1564_;
goto v___jp_1458_;
}
}
}
else
{
lean_object* v___x_1584_; 
lean_dec(v___x_1554_);
v___x_1584_ = lean_box(0);
v_hx_x3f_x3f_1459_ = v___x_1584_;
v_p_x3f_1460_ = v___x_1584_;
v___y_1461_ = v_a_1344_;
v___y_1462_ = v_a_1345_;
goto v___jp_1458_;
}
}
v___jp_1458_:
{
lean_object* v___x_1463_; 
v___x_1463_ = lp_mathlib_BigOperators_processBigOpBinders(v_bs_1457_, v___y_1461_, v___y_1462_);
if (lean_obj_tag(v___x_1463_) == 0)
{
lean_object* v_a_1464_; lean_object* v_a_1465_; lean_object* v___x_1466_; lean_object* v_a_1467_; lean_object* v_a_1468_; lean_object* v___x_1470_; uint8_t v_isShared_1471_; uint8_t v_isSharedCheck_1539_; 
v_a_1464_ = lean_ctor_get(v___x_1463_, 0);
lean_inc_n(v_a_1464_, 2);
v_a_1465_ = lean_ctor_get(v___x_1463_, 1);
lean_inc(v_a_1465_);
lean_dec_ref_known(v___x_1463_, 2);
v___x_1466_ = lp_mathlib_BigOperators_bigOpBindersPattern(v_a_1464_, v___y_1461_, v_a_1465_);
v_a_1467_ = lean_ctor_get(v___x_1466_, 0);
v_a_1468_ = lean_ctor_get(v___x_1466_, 1);
v_isSharedCheck_1539_ = !lean_is_exclusive(v___x_1466_);
if (v_isSharedCheck_1539_ == 0)
{
v___x_1470_ = v___x_1466_;
v_isShared_1471_ = v_isSharedCheck_1539_;
goto v_resetjp_1469_;
}
else
{
lean_inc(v_a_1468_);
lean_inc(v_a_1467_);
lean_dec(v___x_1466_);
v___x_1470_ = lean_box(0);
v_isShared_1471_ = v_isSharedCheck_1539_;
goto v_resetjp_1469_;
}
v_resetjp_1469_:
{
lean_object* v___x_1472_; 
v___x_1472_ = lp_mathlib_BigOperators_bigOpBindersProd(v_a_1464_, v___y_1461_, v_a_1468_);
lean_dec(v_a_1464_);
if (lean_obj_tag(v___x_1472_) == 0)
{
lean_object* v_a_1473_; lean_object* v_a_1474_; lean_object* v___x_1476_; uint8_t v_isShared_1477_; uint8_t v_isSharedCheck_1529_; 
v_a_1473_ = lean_ctor_get(v___x_1472_, 0);
v_a_1474_ = lean_ctor_get(v___x_1472_, 1);
v_isSharedCheck_1529_ = !lean_is_exclusive(v___x_1472_);
if (v_isSharedCheck_1529_ == 0)
{
v___x_1476_ = v___x_1472_;
v_isShared_1477_ = v_isSharedCheck_1529_;
goto v_resetjp_1475_;
}
else
{
lean_inc(v_a_1474_);
lean_inc(v_a_1473_);
lean_dec(v___x_1472_);
v___x_1476_ = lean_box(0);
v_isShared_1477_ = v_isSharedCheck_1529_;
goto v_resetjp_1475_;
}
v_resetjp_1475_:
{
lean_object* v___x_1478_; lean_object* v___x_1479_; 
v___x_1478_ = lean_unsigned_to_nat(4u);
v___x_1479_ = l_Lean_Syntax_getArg(v_x_1343_, v___x_1478_);
lean_dec(v_x_1343_);
if (lean_obj_tag(v_hx_x3f_x3f_1459_) == 1)
{
lean_object* v_val_1480_; 
v_val_1480_ = lean_ctor_get(v_hx_x3f_x3f_1459_, 0);
lean_inc(v_val_1480_);
lean_dec_ref_known(v_hx_x3f_x3f_1459_, 1);
if (lean_obj_tag(v_val_1480_) == 1)
{
if (lean_obj_tag(v_p_x3f_1460_) == 0)
{
lean_dec_ref_known(v_val_1480_, 1);
lean_del_object(v___x_1476_);
lean_del_object(v___x_1470_);
v___y_1347_ = v_a_1473_;
v___y_1348_ = v_a_1467_;
v___y_1349_ = v___x_1479_;
v___y_1350_ = v___y_1461_;
v___y_1351_ = v_a_1474_;
goto v___jp_1346_;
}
else
{
lean_object* v_val_1481_; lean_object* v_val_1482_; lean_object* v_quotContext_1483_; lean_object* v_currMacroScope_1484_; lean_object* v_ref_1485_; uint8_t v___x_1486_; lean_object* v___x_1487_; lean_object* v___x_1488_; lean_object* v___x_1489_; lean_object* v___x_1490_; lean_object* v___x_1491_; lean_object* v___x_1492_; lean_object* v___x_1493_; lean_object* v___x_1494_; lean_object* v___x_1495_; lean_object* v___x_1496_; lean_object* v___x_1498_; 
v_val_1481_ = lean_ctor_get(v_val_1480_, 0);
lean_inc(v_val_1481_);
lean_dec_ref_known(v_val_1480_, 1);
v_val_1482_ = lean_ctor_get(v_p_x3f_1460_, 0);
lean_inc(v_val_1482_);
lean_dec_ref_known(v_p_x3f_1460_, 1);
v_quotContext_1483_ = lean_ctor_get(v___y_1461_, 1);
v_currMacroScope_1484_ = lean_ctor_get(v___y_1461_, 2);
v_ref_1485_ = lean_ctor_get(v___y_1461_, 5);
v___x_1486_ = 0;
v___x_1487_ = l_Lean_SourceInfo_fromRef(v_ref_1485_, v___x_1486_);
v___x_1488_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__18));
v___x_1489_ = lean_obj_once(&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigprod__1___closed__1, &lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigprod__1___closed__1_once, _init_lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigprod__1___closed__1);
v___x_1490_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigprod__1___closed__3));
lean_inc(v_currMacroScope_1484_);
lean_inc(v_quotContext_1483_);
v___x_1491_ = l_Lean_addMacroScope(v_quotContext_1483_, v___x_1490_, v_currMacroScope_1484_);
v___x_1492_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigprod__1___closed__5));
lean_inc_n(v___x_1487_, 2);
v___x_1493_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1493_, 0, v___x_1487_);
lean_ctor_set(v___x_1493_, 1, v___x_1489_);
lean_ctor_set(v___x_1493_, 2, v___x_1491_);
lean_ctor_set(v___x_1493_, 3, v___x_1492_);
v___x_1494_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__25));
v___x_1495_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__12));
v___x_1496_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__13));
if (v_isShared_1471_ == 0)
{
lean_ctor_set_tag(v___x_1470_, 2);
lean_ctor_set(v___x_1470_, 1, v___x_1495_);
lean_ctor_set(v___x_1470_, 0, v___x_1487_);
v___x_1498_ = v___x_1470_;
goto v_reusejp_1497_;
}
else
{
lean_object* v_reuseFailAlloc_1526_; 
v_reuseFailAlloc_1526_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1526_, 0, v___x_1487_);
lean_ctor_set(v_reuseFailAlloc_1526_, 1, v___x_1495_);
v___x_1498_ = v_reuseFailAlloc_1526_;
goto v_reusejp_1497_;
}
v_reusejp_1497_:
{
lean_object* v___x_1499_; lean_object* v___x_1500_; lean_object* v___x_1501_; lean_object* v___x_1502_; lean_object* v___x_1503_; lean_object* v___x_1504_; lean_object* v___x_1505_; lean_object* v___x_1506_; lean_object* v___x_1507_; lean_object* v___x_1508_; lean_object* v___x_1509_; lean_object* v___x_1510_; lean_object* v___x_1511_; lean_object* v___x_1512_; lean_object* v___x_1513_; lean_object* v___x_1514_; lean_object* v___x_1515_; lean_object* v___x_1516_; lean_object* v___x_1517_; lean_object* v___x_1518_; lean_object* v___x_1519_; lean_object* v___x_1520_; lean_object* v___x_1521_; lean_object* v___x_1522_; lean_object* v___x_1524_; 
v___x_1499_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__15));
lean_inc_n(v___x_1487_, 13);
v___x_1500_ = l_Lean_Syntax_node1(v___x_1487_, v___x_1494_, v_a_1467_);
v___x_1501_ = lean_obj_once(&lp_mathlib_BigOperators_bigOpBindersPattern___closed__0, &lp_mathlib_BigOperators_bigOpBindersPattern___closed__0_once, _init_lp_mathlib_BigOperators_bigOpBindersPattern___closed__0);
v___x_1502_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1502_, 0, v___x_1487_);
lean_ctor_set(v___x_1502_, 1, v___x_1494_);
lean_ctor_set(v___x_1502_, 2, v___x_1501_);
v___x_1503_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__16));
v___x_1504_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1504_, 0, v___x_1487_);
lean_ctor_set(v___x_1504_, 1, v___x_1503_);
v___x_1505_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__24));
v___x_1506_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__25));
v___x_1507_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1507_, 0, v___x_1487_);
lean_ctor_set(v___x_1507_, 1, v___x_1506_);
v___x_1508_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__61));
v___x_1509_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1509_, 0, v___x_1487_);
lean_ctor_set(v___x_1509_, 1, v___x_1508_);
v___x_1510_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__26));
v___x_1511_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1511_, 0, v___x_1487_);
lean_ctor_set(v___x_1511_, 1, v___x_1510_);
v___x_1512_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__27));
v___x_1513_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1513_, 0, v___x_1487_);
lean_ctor_set(v___x_1513_, 1, v___x_1512_);
v___x_1514_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__29));
v___x_1515_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigprod__1___closed__6));
v___x_1516_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1516_, 0, v___x_1487_);
lean_ctor_set(v___x_1516_, 1, v___x_1515_);
v___x_1517_ = l_Lean_Syntax_node1(v___x_1487_, v___x_1514_, v___x_1516_);
v___x_1518_ = l_Lean_Syntax_node8(v___x_1487_, v___x_1505_, v___x_1507_, v_val_1481_, v___x_1509_, v_val_1482_, v___x_1511_, v___x_1479_, v___x_1513_, v___x_1517_);
v___x_1519_ = l_Lean_Syntax_node4(v___x_1487_, v___x_1499_, v___x_1500_, v___x_1502_, v___x_1504_, v___x_1518_);
v___x_1520_ = l_Lean_Syntax_node2(v___x_1487_, v___x_1496_, v___x_1498_, v___x_1519_);
v___x_1521_ = l_Lean_Syntax_node2(v___x_1487_, v___x_1494_, v_a_1473_, v___x_1520_);
v___x_1522_ = l_Lean_Syntax_node2(v___x_1487_, v___x_1488_, v___x_1493_, v___x_1521_);
if (v_isShared_1477_ == 0)
{
lean_ctor_set(v___x_1476_, 0, v___x_1522_);
v___x_1524_ = v___x_1476_;
goto v_reusejp_1523_;
}
else
{
lean_object* v_reuseFailAlloc_1525_; 
v_reuseFailAlloc_1525_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1525_, 0, v___x_1522_);
lean_ctor_set(v_reuseFailAlloc_1525_, 1, v_a_1474_);
v___x_1524_ = v_reuseFailAlloc_1525_;
goto v_reusejp_1523_;
}
v_reusejp_1523_:
{
return v___x_1524_;
}
}
}
}
else
{
lean_dec(v_val_1480_);
lean_del_object(v___x_1476_);
lean_del_object(v___x_1470_);
if (lean_obj_tag(v_p_x3f_1460_) == 0)
{
v___y_1347_ = v_a_1473_;
v___y_1348_ = v_a_1467_;
v___y_1349_ = v___x_1479_;
v___y_1350_ = v___y_1461_;
v___y_1351_ = v_a_1474_;
goto v___jp_1346_;
}
else
{
lean_object* v_val_1527_; 
v_val_1527_ = lean_ctor_get(v_p_x3f_1460_, 0);
lean_inc(v_val_1527_);
lean_dec_ref_known(v_p_x3f_1460_, 1);
v___y_1394_ = v_a_1473_;
v___y_1395_ = v_a_1467_;
v___y_1396_ = v___x_1479_;
v_p_1397_ = v_val_1527_;
v___y_1398_ = v___y_1461_;
v___y_1399_ = v_a_1474_;
goto v___jp_1393_;
}
}
}
else
{
lean_del_object(v___x_1476_);
lean_del_object(v___x_1470_);
lean_dec(v_hx_x3f_x3f_1459_);
if (lean_obj_tag(v_p_x3f_1460_) == 0)
{
v___y_1347_ = v_a_1473_;
v___y_1348_ = v_a_1467_;
v___y_1349_ = v___x_1479_;
v___y_1350_ = v___y_1461_;
v___y_1351_ = v_a_1474_;
goto v___jp_1346_;
}
else
{
lean_object* v_val_1528_; 
v_val_1528_ = lean_ctor_get(v_p_x3f_1460_, 0);
lean_inc(v_val_1528_);
lean_dec_ref_known(v_p_x3f_1460_, 1);
v___y_1394_ = v_a_1473_;
v___y_1395_ = v_a_1467_;
v___y_1396_ = v___x_1479_;
v_p_1397_ = v_val_1528_;
v___y_1398_ = v___y_1461_;
v___y_1399_ = v_a_1474_;
goto v___jp_1393_;
}
}
}
}
else
{
lean_object* v_a_1530_; lean_object* v_a_1531_; lean_object* v___x_1533_; uint8_t v_isShared_1534_; uint8_t v_isSharedCheck_1538_; 
lean_del_object(v___x_1470_);
lean_dec(v_a_1467_);
lean_dec(v_p_x3f_1460_);
lean_dec(v_hx_x3f_x3f_1459_);
lean_dec(v_x_1343_);
v_a_1530_ = lean_ctor_get(v___x_1472_, 0);
v_a_1531_ = lean_ctor_get(v___x_1472_, 1);
v_isSharedCheck_1538_ = !lean_is_exclusive(v___x_1472_);
if (v_isSharedCheck_1538_ == 0)
{
v___x_1533_ = v___x_1472_;
v_isShared_1534_ = v_isSharedCheck_1538_;
goto v_resetjp_1532_;
}
else
{
lean_inc(v_a_1531_);
lean_inc(v_a_1530_);
lean_dec(v___x_1472_);
v___x_1533_ = lean_box(0);
v_isShared_1534_ = v_isSharedCheck_1538_;
goto v_resetjp_1532_;
}
v_resetjp_1532_:
{
lean_object* v___x_1536_; 
if (v_isShared_1534_ == 0)
{
v___x_1536_ = v___x_1533_;
goto v_reusejp_1535_;
}
else
{
lean_object* v_reuseFailAlloc_1537_; 
v_reuseFailAlloc_1537_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1537_, 0, v_a_1530_);
lean_ctor_set(v_reuseFailAlloc_1537_, 1, v_a_1531_);
v___x_1536_ = v_reuseFailAlloc_1537_;
goto v_reusejp_1535_;
}
v_reusejp_1535_:
{
return v___x_1536_;
}
}
}
}
}
else
{
lean_object* v_a_1540_; lean_object* v_a_1541_; lean_object* v___x_1543_; uint8_t v_isShared_1544_; uint8_t v_isSharedCheck_1548_; 
lean_dec(v_p_x3f_1460_);
lean_dec(v_hx_x3f_x3f_1459_);
lean_dec(v_x_1343_);
v_a_1540_ = lean_ctor_get(v___x_1463_, 0);
v_a_1541_ = lean_ctor_get(v___x_1463_, 1);
v_isSharedCheck_1548_ = !lean_is_exclusive(v___x_1463_);
if (v_isSharedCheck_1548_ == 0)
{
v___x_1543_ = v___x_1463_;
v_isShared_1544_ = v_isSharedCheck_1548_;
goto v_resetjp_1542_;
}
else
{
lean_inc(v_a_1541_);
lean_inc(v_a_1540_);
lean_dec(v___x_1463_);
v___x_1543_ = lean_box(0);
v_isShared_1544_ = v_isSharedCheck_1548_;
goto v_resetjp_1542_;
}
v_resetjp_1542_:
{
lean_object* v___x_1546_; 
if (v_isShared_1544_ == 0)
{
v___x_1546_ = v___x_1543_;
goto v_reusejp_1545_;
}
else
{
lean_object* v_reuseFailAlloc_1547_; 
v_reuseFailAlloc_1547_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1547_, 0, v_a_1540_);
lean_ctor_set(v_reuseFailAlloc_1547_, 1, v_a_1541_);
v___x_1546_ = v_reuseFailAlloc_1547_;
goto v_reusejp_1545_;
}
v_reusejp_1545_:
{
return v___x_1546_;
}
}
}
}
}
v___jp_1346_:
{
lean_object* v_quotContext_1352_; lean_object* v_currMacroScope_1353_; lean_object* v_ref_1354_; uint8_t v___x_1355_; lean_object* v___x_1356_; lean_object* v___x_1357_; lean_object* v___x_1358_; lean_object* v___x_1359_; lean_object* v___x_1360_; lean_object* v___x_1361_; lean_object* v___x_1362_; lean_object* v___x_1363_; lean_object* v___x_1364_; lean_object* v___x_1365_; lean_object* v___x_1366_; lean_object* v___x_1367_; lean_object* v___x_1368_; lean_object* v___x_1369_; lean_object* v___x_1370_; lean_object* v___x_1371_; lean_object* v___x_1372_; lean_object* v___x_1373_; lean_object* v___x_1374_; lean_object* v___x_1375_; lean_object* v___x_1376_; lean_object* v___x_1377_; lean_object* v___x_1378_; lean_object* v___x_1379_; lean_object* v___x_1380_; lean_object* v___x_1381_; lean_object* v___x_1382_; lean_object* v___x_1383_; lean_object* v___x_1384_; lean_object* v___x_1385_; lean_object* v___x_1386_; lean_object* v___x_1387_; lean_object* v___x_1388_; lean_object* v___x_1389_; lean_object* v___x_1390_; lean_object* v___x_1391_; lean_object* v___x_1392_; 
v_quotContext_1352_ = lean_ctor_get(v___y_1350_, 1);
v_currMacroScope_1353_ = lean_ctor_get(v___y_1350_, 2);
v_ref_1354_ = lean_ctor_get(v___y_1350_, 5);
v___x_1355_ = 0;
v___x_1356_ = l_Lean_SourceInfo_fromRef(v_ref_1354_, v___x_1355_);
v___x_1357_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__18));
v___x_1358_ = lean_obj_once(&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigprod__1___closed__1, &lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigprod__1___closed__1_once, _init_lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigprod__1___closed__1);
v___x_1359_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigprod__1___closed__3));
lean_inc_n(v_currMacroScope_1353_, 2);
lean_inc_n(v_quotContext_1352_, 2);
v___x_1360_ = l_Lean_addMacroScope(v_quotContext_1352_, v___x_1359_, v_currMacroScope_1353_);
v___x_1361_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigprod__1___closed__5));
lean_inc_n(v___x_1356_, 14);
v___x_1362_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1362_, 0, v___x_1356_);
lean_ctor_set(v___x_1362_, 1, v___x_1358_);
lean_ctor_set(v___x_1362_, 2, v___x_1360_);
lean_ctor_set(v___x_1362_, 3, v___x_1361_);
v___x_1363_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__25));
v___x_1364_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__65));
v___x_1365_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__52));
v___x_1366_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__53));
v___x_1367_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1367_, 0, v___x_1356_);
lean_ctor_set(v___x_1367_, 1, v___x_1366_);
v___x_1368_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__55));
v___x_1369_ = lean_obj_once(&lp_mathlib_BigOperators_processBigOpBinder___closed__57, &lp_mathlib_BigOperators_processBigOpBinder___closed__57_once, _init_lp_mathlib_BigOperators_processBigOpBinder___closed__57);
v___x_1370_ = lean_box(0);
v___x_1371_ = l_Lean_addMacroScope(v_quotContext_1352_, v___x_1370_, v_currMacroScope_1353_);
v___x_1372_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__11));
v___x_1373_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1373_, 0, v___x_1356_);
lean_ctor_set(v___x_1373_, 1, v___x_1369_);
lean_ctor_set(v___x_1373_, 2, v___x_1371_);
lean_ctor_set(v___x_1373_, 3, v___x_1372_);
v___x_1374_ = l_Lean_Syntax_node1(v___x_1356_, v___x_1368_, v___x_1373_);
v___x_1375_ = l_Lean_Syntax_node2(v___x_1356_, v___x_1365_, v___x_1367_, v___x_1374_);
v___x_1376_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__12));
v___x_1377_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__13));
v___x_1378_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1378_, 0, v___x_1356_);
lean_ctor_set(v___x_1378_, 1, v___x_1376_);
v___x_1379_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__15));
v___x_1380_ = l_Lean_Syntax_node1(v___x_1356_, v___x_1363_, v___y_1348_);
v___x_1381_ = lean_obj_once(&lp_mathlib_BigOperators_bigOpBindersPattern___closed__0, &lp_mathlib_BigOperators_bigOpBindersPattern___closed__0_once, _init_lp_mathlib_BigOperators_bigOpBindersPattern___closed__0);
v___x_1382_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1382_, 0, v___x_1356_);
lean_ctor_set(v___x_1382_, 1, v___x_1363_);
lean_ctor_set(v___x_1382_, 2, v___x_1381_);
v___x_1383_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__16));
v___x_1384_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1384_, 0, v___x_1356_);
lean_ctor_set(v___x_1384_, 1, v___x_1383_);
v___x_1385_ = l_Lean_Syntax_node4(v___x_1356_, v___x_1379_, v___x_1380_, v___x_1382_, v___x_1384_, v___y_1349_);
v___x_1386_ = l_Lean_Syntax_node2(v___x_1356_, v___x_1377_, v___x_1378_, v___x_1385_);
v___x_1387_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBinderParenthesized___closed__5));
v___x_1388_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1388_, 0, v___x_1356_);
lean_ctor_set(v___x_1388_, 1, v___x_1387_);
v___x_1389_ = l_Lean_Syntax_node3(v___x_1356_, v___x_1364_, v___x_1375_, v___x_1386_, v___x_1388_);
v___x_1390_ = l_Lean_Syntax_node2(v___x_1356_, v___x_1363_, v___y_1347_, v___x_1389_);
v___x_1391_ = l_Lean_Syntax_node2(v___x_1356_, v___x_1357_, v___x_1362_, v___x_1390_);
v___x_1392_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1392_, 0, v___x_1391_);
lean_ctor_set(v___x_1392_, 1, v___y_1351_);
return v___x_1392_;
}
v___jp_1393_:
{
lean_object* v_quotContext_1400_; lean_object* v_currMacroScope_1401_; lean_object* v_ref_1402_; uint8_t v___x_1403_; lean_object* v___x_1404_; lean_object* v___x_1405_; lean_object* v___x_1406_; lean_object* v___x_1407_; lean_object* v___x_1408_; lean_object* v___x_1409_; lean_object* v___x_1410_; lean_object* v___x_1411_; lean_object* v___x_1412_; lean_object* v___x_1413_; lean_object* v___x_1414_; lean_object* v___x_1415_; lean_object* v___x_1416_; lean_object* v___x_1417_; lean_object* v___x_1418_; lean_object* v___x_1419_; lean_object* v___x_1420_; lean_object* v___x_1421_; lean_object* v___x_1422_; lean_object* v___x_1423_; lean_object* v___x_1424_; lean_object* v___x_1425_; lean_object* v___x_1426_; lean_object* v___x_1427_; lean_object* v___x_1428_; lean_object* v___x_1429_; lean_object* v___x_1430_; lean_object* v___x_1431_; lean_object* v___x_1432_; lean_object* v___x_1433_; lean_object* v___x_1434_; lean_object* v___x_1435_; lean_object* v___x_1436_; lean_object* v___x_1437_; lean_object* v___x_1438_; lean_object* v___x_1439_; lean_object* v___x_1440_; lean_object* v___x_1441_; lean_object* v___x_1442_; lean_object* v___x_1443_; lean_object* v___x_1444_; lean_object* v___x_1445_; lean_object* v___x_1446_; lean_object* v___x_1447_; lean_object* v___x_1448_; lean_object* v___x_1449_; lean_object* v___x_1450_; lean_object* v___x_1451_; 
v_quotContext_1400_ = lean_ctor_get(v___y_1398_, 1);
v_currMacroScope_1401_ = lean_ctor_get(v___y_1398_, 2);
v_ref_1402_ = lean_ctor_get(v___y_1398_, 5);
v___x_1403_ = 0;
v___x_1404_ = l_Lean_SourceInfo_fromRef(v_ref_1402_, v___x_1403_);
v___x_1405_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__18));
v___x_1406_ = lean_obj_once(&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigprod__1___closed__1, &lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigprod__1___closed__1_once, _init_lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigprod__1___closed__1);
v___x_1407_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigprod__1___closed__3));
lean_inc_n(v_currMacroScope_1401_, 3);
lean_inc_n(v_quotContext_1400_, 3);
v___x_1408_ = l_Lean_addMacroScope(v_quotContext_1400_, v___x_1407_, v_currMacroScope_1401_);
v___x_1409_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigprod__1___closed__5));
lean_inc_n(v___x_1404_, 21);
v___x_1410_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1410_, 0, v___x_1404_);
lean_ctor_set(v___x_1410_, 1, v___x_1406_);
lean_ctor_set(v___x_1410_, 2, v___x_1408_);
lean_ctor_set(v___x_1410_, 3, v___x_1409_);
v___x_1411_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__25));
v___x_1412_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__65));
v___x_1413_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__52));
v___x_1414_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__53));
v___x_1415_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1415_, 0, v___x_1404_);
lean_ctor_set(v___x_1415_, 1, v___x_1414_);
v___x_1416_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__55));
v___x_1417_ = lean_obj_once(&lp_mathlib_BigOperators_processBigOpBinder___closed__57, &lp_mathlib_BigOperators_processBigOpBinder___closed__57_once, _init_lp_mathlib_BigOperators_processBigOpBinder___closed__57);
v___x_1418_ = lean_box(0);
v___x_1419_ = l_Lean_addMacroScope(v_quotContext_1400_, v___x_1418_, v_currMacroScope_1401_);
v___x_1420_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__11));
v___x_1421_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1421_, 0, v___x_1404_);
lean_ctor_set(v___x_1421_, 1, v___x_1417_);
lean_ctor_set(v___x_1421_, 2, v___x_1419_);
lean_ctor_set(v___x_1421_, 3, v___x_1420_);
v___x_1422_ = l_Lean_Syntax_node1(v___x_1404_, v___x_1416_, v___x_1421_);
v___x_1423_ = l_Lean_Syntax_node2(v___x_1404_, v___x_1413_, v___x_1415_, v___x_1422_);
v___x_1424_ = lean_obj_once(&lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__18, &lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__18_once, _init_lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__18);
v___x_1425_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__20));
v___x_1426_ = l_Lean_addMacroScope(v_quotContext_1400_, v___x_1425_, v_currMacroScope_1401_);
v___x_1427_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__22));
v___x_1428_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1428_, 0, v___x_1404_);
lean_ctor_set(v___x_1428_, 1, v___x_1424_);
lean_ctor_set(v___x_1428_, 2, v___x_1426_);
lean_ctor_set(v___x_1428_, 3, v___x_1427_);
v___x_1429_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__12));
v___x_1430_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__13));
v___x_1431_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1431_, 0, v___x_1404_);
lean_ctor_set(v___x_1431_, 1, v___x_1429_);
v___x_1432_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__15));
v___x_1433_ = l_Lean_Syntax_node1(v___x_1404_, v___x_1411_, v___y_1395_);
v___x_1434_ = lean_obj_once(&lp_mathlib_BigOperators_bigOpBindersPattern___closed__0, &lp_mathlib_BigOperators_bigOpBindersPattern___closed__0_once, _init_lp_mathlib_BigOperators_bigOpBindersPattern___closed__0);
v___x_1435_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1435_, 0, v___x_1404_);
lean_ctor_set(v___x_1435_, 1, v___x_1411_);
lean_ctor_set(v___x_1435_, 2, v___x_1434_);
v___x_1436_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__16));
v___x_1437_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1437_, 0, v___x_1404_);
lean_ctor_set(v___x_1437_, 1, v___x_1436_);
lean_inc_ref(v___x_1437_);
lean_inc_ref(v___x_1435_);
lean_inc(v___x_1433_);
v___x_1438_ = l_Lean_Syntax_node4(v___x_1404_, v___x_1432_, v___x_1433_, v___x_1435_, v___x_1437_, v_p_1397_);
lean_inc_ref(v___x_1431_);
v___x_1439_ = l_Lean_Syntax_node2(v___x_1404_, v___x_1430_, v___x_1431_, v___x_1438_);
v___x_1440_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBinderParenthesized___closed__5));
v___x_1441_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1441_, 0, v___x_1404_);
lean_ctor_set(v___x_1441_, 1, v___x_1440_);
lean_inc_ref_n(v___x_1441_, 2);
lean_inc_n(v___x_1423_, 2);
v___x_1442_ = l_Lean_Syntax_node3(v___x_1404_, v___x_1412_, v___x_1423_, v___x_1439_, v___x_1441_);
v___x_1443_ = l_Lean_Syntax_node2(v___x_1404_, v___x_1411_, v___x_1442_, v___y_1394_);
v___x_1444_ = l_Lean_Syntax_node2(v___x_1404_, v___x_1405_, v___x_1428_, v___x_1443_);
v___x_1445_ = l_Lean_Syntax_node3(v___x_1404_, v___x_1412_, v___x_1423_, v___x_1444_, v___x_1441_);
v___x_1446_ = l_Lean_Syntax_node4(v___x_1404_, v___x_1432_, v___x_1433_, v___x_1435_, v___x_1437_, v___y_1396_);
v___x_1447_ = l_Lean_Syntax_node2(v___x_1404_, v___x_1430_, v___x_1431_, v___x_1446_);
v___x_1448_ = l_Lean_Syntax_node3(v___x_1404_, v___x_1412_, v___x_1423_, v___x_1447_, v___x_1441_);
v___x_1449_ = l_Lean_Syntax_node2(v___x_1404_, v___x_1411_, v___x_1445_, v___x_1448_);
v___x_1450_ = l_Lean_Syntax_node2(v___x_1404_, v___x_1405_, v___x_1410_, v___x_1449_);
v___x_1451_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1451_, 0, v___x_1450_);
lean_ctor_set(v___x_1451_, 1, v___y_1399_);
return v___x_1451_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigprod__1___boxed(lean_object* v_x_1585_, lean_object* v_a_1586_, lean_object* v_a_1587_){
_start:
{
lean_object* v_res_1588_; 
v_res_1588_ = lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigprod__1(v_x_1585_, v_a_1586_, v_a_1587_);
lean_dec_ref(v_a_1586_);
return v_res_1588_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_FinsetResult_ctorIdx(lean_object* v_x_1589_){
_start:
{
switch(lean_obj_tag(v_x_1589_))
{
case 0:
{
lean_object* v___x_1590_; 
v___x_1590_ = lean_unsigned_to_nat(0u);
return v___x_1590_;
}
case 1:
{
lean_object* v___x_1591_; 
v___x_1591_ = lean_unsigned_to_nat(1u);
return v___x_1591_;
}
case 2:
{
lean_object* v___x_1592_; 
v___x_1592_ = lean_unsigned_to_nat(2u);
return v___x_1592_;
}
case 3:
{
lean_object* v___x_1593_; 
v___x_1593_ = lean_unsigned_to_nat(3u);
return v___x_1593_;
}
case 4:
{
lean_object* v___x_1594_; 
v___x_1594_ = lean_unsigned_to_nat(4u);
return v___x_1594_;
}
default: 
{
lean_object* v___x_1595_; 
v___x_1595_ = lean_unsigned_to_nat(5u);
return v___x_1595_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_FinsetResult_ctorIdx___boxed(lean_object* v_x_1596_){
_start:
{
lean_object* v_res_1597_; 
v_res_1597_ = lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_FinsetResult_ctorIdx(v_x_1596_);
lean_dec(v_x_1596_);
return v_res_1597_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_FinsetResult_ctorElim___redArg(lean_object* v_t_1598_, lean_object* v_k_1599_){
_start:
{
if (lean_obj_tag(v_t_1598_) == 1)
{
return v_k_1599_;
}
else
{
lean_object* v_s_1600_; lean_object* v___x_1601_; 
v_s_1600_ = lean_ctor_get(v_t_1598_, 0);
lean_inc(v_s_1600_);
lean_dec(v_t_1598_);
v___x_1601_ = lean_apply_1(v_k_1599_, v_s_1600_);
return v___x_1601_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_FinsetResult_ctorElim(lean_object* v_motive_1602_, lean_object* v_ctorIdx_1603_, lean_object* v_t_1604_, lean_object* v_h_1605_, lean_object* v_k_1606_){
_start:
{
lean_object* v___x_1607_; 
v___x_1607_ = lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_FinsetResult_ctorElim___redArg(v_t_1604_, v_k_1606_);
return v___x_1607_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_FinsetResult_ctorElim___boxed(lean_object* v_motive_1608_, lean_object* v_ctorIdx_1609_, lean_object* v_t_1610_, lean_object* v_h_1611_, lean_object* v_k_1612_){
_start:
{
lean_object* v_res_1613_; 
v_res_1613_ = lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_FinsetResult_ctorElim(v_motive_1608_, v_ctorIdx_1609_, v_t_1610_, v_h_1611_, v_k_1612_);
lean_dec(v_ctorIdx_1609_);
return v_res_1613_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_FinsetResult_finset_elim___redArg(lean_object* v_t_1614_, lean_object* v_finset_1615_){
_start:
{
lean_object* v___x_1616_; 
v___x_1616_ = lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_FinsetResult_ctorElim___redArg(v_t_1614_, v_finset_1615_);
return v___x_1616_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_FinsetResult_finset_elim(lean_object* v_motive_1617_, lean_object* v_t_1618_, lean_object* v_h_1619_, lean_object* v_finset_1620_){
_start:
{
lean_object* v___x_1621_; 
v___x_1621_ = lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_FinsetResult_ctorElim___redArg(v_t_1618_, v_finset_1620_);
return v___x_1621_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_FinsetResult_univ_elim___redArg(lean_object* v_t_1622_, lean_object* v_univ_1623_){
_start:
{
lean_object* v___x_1624_; 
v___x_1624_ = lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_FinsetResult_ctorElim___redArg(v_t_1622_, v_univ_1623_);
return v___x_1624_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_FinsetResult_univ_elim(lean_object* v_motive_1625_, lean_object* v_t_1626_, lean_object* v_h_1627_, lean_object* v_univ_1628_){
_start:
{
lean_object* v___x_1629_; 
v___x_1629_ = lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_FinsetResult_ctorElim___redArg(v_t_1626_, v_univ_1628_);
return v___x_1629_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_FinsetResult_Iio_elim___redArg(lean_object* v_t_1630_, lean_object* v_Iio_1631_){
_start:
{
lean_object* v___x_1632_; 
v___x_1632_ = lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_FinsetResult_ctorElim___redArg(v_t_1630_, v_Iio_1631_);
return v___x_1632_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_FinsetResult_Iio_elim(lean_object* v_motive_1633_, lean_object* v_t_1634_, lean_object* v_h_1635_, lean_object* v_Iio_1636_){
_start:
{
lean_object* v___x_1637_; 
v___x_1637_ = lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_FinsetResult_ctorElim___redArg(v_t_1634_, v_Iio_1636_);
return v___x_1637_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_FinsetResult_Iic_elim___redArg(lean_object* v_t_1638_, lean_object* v_Iic_1639_){
_start:
{
lean_object* v___x_1640_; 
v___x_1640_ = lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_FinsetResult_ctorElim___redArg(v_t_1638_, v_Iic_1639_);
return v___x_1640_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_FinsetResult_Iic_elim(lean_object* v_motive_1641_, lean_object* v_t_1642_, lean_object* v_h_1643_, lean_object* v_Iic_1644_){
_start:
{
lean_object* v___x_1645_; 
v___x_1645_ = lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_FinsetResult_ctorElim___redArg(v_t_1642_, v_Iic_1644_);
return v___x_1645_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_FinsetResult_Ioi_elim___redArg(lean_object* v_t_1646_, lean_object* v_Ioi_1647_){
_start:
{
lean_object* v___x_1648_; 
v___x_1648_ = lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_FinsetResult_ctorElim___redArg(v_t_1646_, v_Ioi_1647_);
return v___x_1648_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_FinsetResult_Ioi_elim(lean_object* v_motive_1649_, lean_object* v_t_1650_, lean_object* v_h_1651_, lean_object* v_Ioi_1652_){
_start:
{
lean_object* v___x_1653_; 
v___x_1653_ = lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_FinsetResult_ctorElim___redArg(v_t_1650_, v_Ioi_1652_);
return v___x_1653_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_FinsetResult_Ici_elim___redArg(lean_object* v_t_1654_, lean_object* v_Ici_1655_){
_start:
{
lean_object* v___x_1656_; 
v___x_1656_ = lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_FinsetResult_ctorElim___redArg(v_t_1654_, v_Ici_1655_);
return v___x_1656_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_FinsetResult_Ici_elim(lean_object* v_motive_1657_, lean_object* v_t_1658_, lean_object* v_h_1659_, lean_object* v_Ici_1660_){
_start:
{
lean_object* v___x_1661_; 
v___x_1661_ = lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_FinsetResult_ctorElim___redArg(v_t_1658_, v_Ici_1660_);
return v___x_1661_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__0___redArg(lean_object* v___y_1662_){
_start:
{
lean_object* v_subExpr_1664_; lean_object* v_expr_1665_; lean_object* v___x_1666_; 
v_subExpr_1664_ = lean_ctor_get(v___y_1662_, 3);
v_expr_1665_ = lean_ctor_get(v_subExpr_1664_, 0);
lean_inc_ref(v_expr_1665_);
v___x_1666_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1666_, 0, v_expr_1665_);
return v___x_1666_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__0___redArg___boxed(lean_object* v___y_1667_, lean_object* v___y_1668_){
_start:
{
lean_object* v_res_1669_; 
v_res_1669_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__0___redArg(v___y_1667_);
lean_dec_ref(v___y_1667_);
return v_res_1669_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__0(lean_object* v___y_1670_, lean_object* v___y_1671_, lean_object* v___y_1672_, lean_object* v___y_1673_, lean_object* v___y_1674_, lean_object* v___y_1675_){
_start:
{
lean_object* v___x_1677_; 
v___x_1677_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__0___redArg(v___y_1670_);
return v___x_1677_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__0___boxed(lean_object* v___y_1678_, lean_object* v___y_1679_, lean_object* v___y_1680_, lean_object* v___y_1681_, lean_object* v___y_1682_, lean_object* v___y_1683_, lean_object* v___y_1684_){
_start:
{
lean_object* v_res_1685_; 
v_res_1685_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__0(v___y_1678_, v___y_1679_, v___y_1680_, v___y_1681_, v___y_1682_, v___y_1683_);
lean_dec(v___y_1683_);
lean_dec_ref(v___y_1682_);
lean_dec(v___y_1681_);
lean_dec_ref(v___y_1680_);
lean_dec(v___y_1679_);
lean_dec_ref(v___y_1678_);
return v_res_1685_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__1_spec__1___redArg(lean_object* v___y_1686_){
_start:
{
lean_object* v_subExpr_1688_; lean_object* v_pos_1689_; lean_object* v___x_1690_; 
v_subExpr_1688_ = lean_ctor_get(v___y_1686_, 3);
v_pos_1689_ = lean_ctor_get(v_subExpr_1688_, 1);
lean_inc(v_pos_1689_);
v___x_1690_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1690_, 0, v_pos_1689_);
return v___x_1690_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__1_spec__1___redArg___boxed(lean_object* v___y_1691_, lean_object* v___y_1692_){
_start:
{
lean_object* v_res_1693_; 
v_res_1693_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__1_spec__1___redArg(v___y_1691_);
lean_dec_ref(v___y_1691_);
return v_res_1693_;
}
}
static lean_object* _init_lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__1___redArg___closed__0(void){
_start:
{
lean_object* v___x_1694_; lean_object* v_dummy_1695_; 
v___x_1694_ = lean_box(0);
v_dummy_1695_ = l_Lean_Expr_sort___override(v___x_1694_);
return v_dummy_1695_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__1___redArg(lean_object* v_argIdx_1696_, lean_object* v_x_1697_, lean_object* v___y_1698_, lean_object* v___y_1699_, lean_object* v___y_1700_, lean_object* v___y_1701_, lean_object* v___y_1702_, lean_object* v___y_1703_){
_start:
{
lean_object* v___x_1705_; lean_object* v_a_1706_; lean_object* v___x_1707_; lean_object* v_a_1708_; lean_object* v_optionsPerPos_1709_; lean_object* v_currNamespace_1710_; lean_object* v_openDecls_1711_; uint8_t v_inPattern_1712_; lean_object* v_depth_1713_; lean_object* v_lctxInitIndices_1714_; lean_object* v_nargs_1715_; lean_object* v___x_1716_; lean_object* v_dummy_1717_; lean_object* v___x_1718_; lean_object* v___x_1719_; lean_object* v___x_1720_; lean_object* v_args_1721_; lean_object* v___x_1722_; lean_object* v_newPos_1723_; lean_object* v___x_1724_; lean_object* v___x_1725_; lean_object* v___x_1726_; lean_object* v___x_1727_; 
v___x_1705_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__0___redArg(v___y_1698_);
v_a_1706_ = lean_ctor_get(v___x_1705_, 0);
lean_inc(v_a_1706_);
lean_dec_ref(v___x_1705_);
v___x_1707_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__1_spec__1___redArg(v___y_1698_);
v_a_1708_ = lean_ctor_get(v___x_1707_, 0);
lean_inc(v_a_1708_);
lean_dec_ref(v___x_1707_);
v_optionsPerPos_1709_ = lean_ctor_get(v___y_1698_, 0);
v_currNamespace_1710_ = lean_ctor_get(v___y_1698_, 1);
v_openDecls_1711_ = lean_ctor_get(v___y_1698_, 2);
v_inPattern_1712_ = lean_ctor_get_uint8(v___y_1698_, sizeof(void*)*6);
v_depth_1713_ = lean_ctor_get(v___y_1698_, 4);
v_lctxInitIndices_1714_ = lean_ctor_get(v___y_1698_, 5);
v_nargs_1715_ = l_Lean_Expr_getAppNumArgs(v_a_1706_);
v___x_1716_ = l_Lean_instInhabitedExpr;
v_dummy_1717_ = lean_obj_once(&lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__1___redArg___closed__0, &lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__1___redArg___closed__0_once, _init_lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__1___redArg___closed__0);
lean_inc(v_nargs_1715_);
v___x_1718_ = lean_mk_array(v_nargs_1715_, v_dummy_1717_);
v___x_1719_ = lean_unsigned_to_nat(1u);
v___x_1720_ = lean_nat_sub(v_nargs_1715_, v___x_1719_);
lean_dec(v_nargs_1715_);
v_args_1721_ = l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(v_a_1706_, v___x_1718_, v___x_1720_);
v___x_1722_ = lean_array_get_size(v_args_1721_);
v_newPos_1723_ = l_Lean_SubExpr_Pos_pushNaryArg(v___x_1722_, v_argIdx_1696_, v_a_1708_);
lean_dec(v_a_1708_);
v___x_1724_ = lean_array_get(v___x_1716_, v_args_1721_, v_argIdx_1696_);
lean_dec_ref(v_args_1721_);
v___x_1725_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1725_, 0, v___x_1724_);
lean_ctor_set(v___x_1725_, 1, v_newPos_1723_);
lean_inc(v_lctxInitIndices_1714_);
lean_inc(v_depth_1713_);
lean_inc(v_openDecls_1711_);
lean_inc(v_currNamespace_1710_);
lean_inc(v_optionsPerPos_1709_);
v___x_1726_ = lean_alloc_ctor(0, 6, 1);
lean_ctor_set(v___x_1726_, 0, v_optionsPerPos_1709_);
lean_ctor_set(v___x_1726_, 1, v_currNamespace_1710_);
lean_ctor_set(v___x_1726_, 2, v_openDecls_1711_);
lean_ctor_set(v___x_1726_, 3, v___x_1725_);
lean_ctor_set(v___x_1726_, 4, v_depth_1713_);
lean_ctor_set(v___x_1726_, 5, v_lctxInitIndices_1714_);
lean_ctor_set_uint8(v___x_1726_, sizeof(void*)*6, v_inPattern_1712_);
lean_inc(v___y_1703_);
lean_inc_ref(v___y_1702_);
lean_inc(v___y_1701_);
lean_inc_ref(v___y_1700_);
lean_inc(v___y_1699_);
v___x_1727_ = lean_apply_7(v_x_1697_, v___x_1726_, v___y_1699_, v___y_1700_, v___y_1701_, v___y_1702_, v___y_1703_, lean_box(0));
return v___x_1727_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__1___redArg___boxed(lean_object* v_argIdx_1728_, lean_object* v_x_1729_, lean_object* v___y_1730_, lean_object* v___y_1731_, lean_object* v___y_1732_, lean_object* v___y_1733_, lean_object* v___y_1734_, lean_object* v___y_1735_, lean_object* v___y_1736_){
_start:
{
lean_object* v_res_1737_; 
v_res_1737_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__1___redArg(v_argIdx_1728_, v_x_1729_, v___y_1730_, v___y_1731_, v___y_1732_, v___y_1733_, v___y_1734_, v___y_1735_);
lean_dec(v___y_1735_);
lean_dec_ref(v___y_1734_);
lean_dec(v___y_1733_);
lean_dec_ref(v___y_1732_);
lean_dec(v___y_1731_);
lean_dec_ref(v___y_1730_);
lean_dec(v_argIdx_1728_);
return v_res_1737_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult(lean_object* v_a_1739_, lean_object* v_a_1740_, lean_object* v_a_1741_, lean_object* v_a_1742_, lean_object* v_a_1743_, lean_object* v_a_1744_){
_start:
{
lean_object* v___x_1746_; lean_object* v_a_1747_; lean_object* v___x_1749_; uint8_t v_isShared_1750_; uint8_t v_isSharedCheck_1865_; 
v___x_1746_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__0___redArg(v_a_1739_);
v_a_1747_ = lean_ctor_get(v___x_1746_, 0);
v_isSharedCheck_1865_ = !lean_is_exclusive(v___x_1746_);
if (v_isSharedCheck_1865_ == 0)
{
v___x_1749_ = v___x_1746_;
v_isShared_1750_ = v_isSharedCheck_1865_;
goto v_resetjp_1748_;
}
else
{
lean_inc(v_a_1747_);
lean_dec(v___x_1746_);
v___x_1749_ = lean_box(0);
v_isShared_1750_ = v_isSharedCheck_1865_;
goto v_resetjp_1748_;
}
v_resetjp_1748_:
{
lean_object* v___x_1751_; lean_object* v___x_1752_; uint8_t v___x_1753_; 
v___x_1751_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__60));
v___x_1752_ = lean_unsigned_to_nat(2u);
v___x_1753_ = l_Lean_Expr_isAppOfArity(v_a_1747_, v___x_1751_, v___x_1752_);
if (v___x_1753_ == 0)
{
lean_object* v___x_1754_; lean_object* v___x_1755_; uint8_t v___x_1756_; 
lean_del_object(v___x_1749_);
v___x_1754_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__37));
v___x_1755_ = lean_unsigned_to_nat(4u);
v___x_1756_ = l_Lean_Expr_isAppOfArity(v_a_1747_, v___x_1754_, v___x_1755_);
if (v___x_1756_ == 0)
{
lean_object* v___x_1757_; uint8_t v___x_1758_; 
v___x_1757_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__33));
v___x_1758_ = l_Lean_Expr_isAppOfArity(v_a_1747_, v___x_1757_, v___x_1755_);
if (v___x_1758_ == 0)
{
lean_object* v___x_1759_; uint8_t v___x_1760_; 
v___x_1759_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__29));
v___x_1760_ = l_Lean_Expr_isAppOfArity(v_a_1747_, v___x_1759_, v___x_1755_);
if (v___x_1760_ == 0)
{
lean_object* v___x_1761_; uint8_t v___x_1762_; 
v___x_1761_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__23));
v___x_1762_ = l_Lean_Expr_isAppOfArity(v_a_1747_, v___x_1761_, v___x_1755_);
lean_dec(v_a_1747_);
if (v___x_1762_ == 0)
{
lean_object* v___x_1763_; 
v___x_1763_ = l_Lean_PrettyPrinter_Delaborator_delab(v_a_1739_, v_a_1740_, v_a_1741_, v_a_1742_, v_a_1743_, v_a_1744_);
if (lean_obj_tag(v___x_1763_) == 0)
{
lean_object* v_a_1764_; lean_object* v___x_1766_; uint8_t v_isShared_1767_; uint8_t v_isSharedCheck_1772_; 
v_a_1764_ = lean_ctor_get(v___x_1763_, 0);
v_isSharedCheck_1772_ = !lean_is_exclusive(v___x_1763_);
if (v_isSharedCheck_1772_ == 0)
{
v___x_1766_ = v___x_1763_;
v_isShared_1767_ = v_isSharedCheck_1772_;
goto v_resetjp_1765_;
}
else
{
lean_inc(v_a_1764_);
lean_dec(v___x_1763_);
v___x_1766_ = lean_box(0);
v_isShared_1767_ = v_isSharedCheck_1772_;
goto v_resetjp_1765_;
}
v_resetjp_1765_:
{
lean_object* v___x_1768_; lean_object* v___x_1770_; 
v___x_1768_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1768_, 0, v_a_1764_);
if (v_isShared_1767_ == 0)
{
lean_ctor_set(v___x_1766_, 0, v___x_1768_);
v___x_1770_ = v___x_1766_;
goto v_reusejp_1769_;
}
else
{
lean_object* v_reuseFailAlloc_1771_; 
v_reuseFailAlloc_1771_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1771_, 0, v___x_1768_);
v___x_1770_ = v_reuseFailAlloc_1771_;
goto v_reusejp_1769_;
}
v_reusejp_1769_:
{
return v___x_1770_;
}
}
}
else
{
lean_object* v_a_1773_; lean_object* v___x_1775_; uint8_t v_isShared_1776_; uint8_t v_isSharedCheck_1780_; 
v_a_1773_ = lean_ctor_get(v___x_1763_, 0);
v_isSharedCheck_1780_ = !lean_is_exclusive(v___x_1763_);
if (v_isSharedCheck_1780_ == 0)
{
v___x_1775_ = v___x_1763_;
v_isShared_1776_ = v_isSharedCheck_1780_;
goto v_resetjp_1774_;
}
else
{
lean_inc(v_a_1773_);
lean_dec(v___x_1763_);
v___x_1775_ = lean_box(0);
v_isShared_1776_ = v_isSharedCheck_1780_;
goto v_resetjp_1774_;
}
v_resetjp_1774_:
{
lean_object* v___x_1778_; 
if (v_isShared_1776_ == 0)
{
v___x_1778_ = v___x_1775_;
goto v_reusejp_1777_;
}
else
{
lean_object* v_reuseFailAlloc_1779_; 
v_reuseFailAlloc_1779_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1779_, 0, v_a_1773_);
v___x_1778_ = v_reuseFailAlloc_1779_;
goto v_reusejp_1777_;
}
v_reusejp_1777_:
{
return v___x_1778_;
}
}
}
}
else
{
lean_object* v___x_1781_; lean_object* v___x_1782_; lean_object* v___x_1783_; 
v___x_1781_ = lean_unsigned_to_nat(3u);
v___x_1782_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult___closed__0));
v___x_1783_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__1___redArg(v___x_1781_, v___x_1782_, v_a_1739_, v_a_1740_, v_a_1741_, v_a_1742_, v_a_1743_, v_a_1744_);
if (lean_obj_tag(v___x_1783_) == 0)
{
lean_object* v_a_1784_; lean_object* v___x_1786_; uint8_t v_isShared_1787_; uint8_t v_isSharedCheck_1792_; 
v_a_1784_ = lean_ctor_get(v___x_1783_, 0);
v_isSharedCheck_1792_ = !lean_is_exclusive(v___x_1783_);
if (v_isSharedCheck_1792_ == 0)
{
v___x_1786_ = v___x_1783_;
v_isShared_1787_ = v_isSharedCheck_1792_;
goto v_resetjp_1785_;
}
else
{
lean_inc(v_a_1784_);
lean_dec(v___x_1783_);
v___x_1786_ = lean_box(0);
v_isShared_1787_ = v_isSharedCheck_1792_;
goto v_resetjp_1785_;
}
v_resetjp_1785_:
{
lean_object* v___x_1788_; lean_object* v___x_1790_; 
v___x_1788_ = lean_alloc_ctor(5, 1, 0);
lean_ctor_set(v___x_1788_, 0, v_a_1784_);
if (v_isShared_1787_ == 0)
{
lean_ctor_set(v___x_1786_, 0, v___x_1788_);
v___x_1790_ = v___x_1786_;
goto v_reusejp_1789_;
}
else
{
lean_object* v_reuseFailAlloc_1791_; 
v_reuseFailAlloc_1791_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1791_, 0, v___x_1788_);
v___x_1790_ = v_reuseFailAlloc_1791_;
goto v_reusejp_1789_;
}
v_reusejp_1789_:
{
return v___x_1790_;
}
}
}
else
{
lean_object* v_a_1793_; lean_object* v___x_1795_; uint8_t v_isShared_1796_; uint8_t v_isSharedCheck_1800_; 
v_a_1793_ = lean_ctor_get(v___x_1783_, 0);
v_isSharedCheck_1800_ = !lean_is_exclusive(v___x_1783_);
if (v_isSharedCheck_1800_ == 0)
{
v___x_1795_ = v___x_1783_;
v_isShared_1796_ = v_isSharedCheck_1800_;
goto v_resetjp_1794_;
}
else
{
lean_inc(v_a_1793_);
lean_dec(v___x_1783_);
v___x_1795_ = lean_box(0);
v_isShared_1796_ = v_isSharedCheck_1800_;
goto v_resetjp_1794_;
}
v_resetjp_1794_:
{
lean_object* v___x_1798_; 
if (v_isShared_1796_ == 0)
{
v___x_1798_ = v___x_1795_;
goto v_reusejp_1797_;
}
else
{
lean_object* v_reuseFailAlloc_1799_; 
v_reuseFailAlloc_1799_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1799_, 0, v_a_1793_);
v___x_1798_ = v_reuseFailAlloc_1799_;
goto v_reusejp_1797_;
}
v_reusejp_1797_:
{
return v___x_1798_;
}
}
}
}
}
else
{
lean_object* v___x_1801_; lean_object* v___x_1802_; lean_object* v___x_1803_; 
lean_dec(v_a_1747_);
v___x_1801_ = lean_unsigned_to_nat(3u);
v___x_1802_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult___closed__0));
v___x_1803_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__1___redArg(v___x_1801_, v___x_1802_, v_a_1739_, v_a_1740_, v_a_1741_, v_a_1742_, v_a_1743_, v_a_1744_);
if (lean_obj_tag(v___x_1803_) == 0)
{
lean_object* v_a_1804_; lean_object* v___x_1806_; uint8_t v_isShared_1807_; uint8_t v_isSharedCheck_1812_; 
v_a_1804_ = lean_ctor_get(v___x_1803_, 0);
v_isSharedCheck_1812_ = !lean_is_exclusive(v___x_1803_);
if (v_isSharedCheck_1812_ == 0)
{
v___x_1806_ = v___x_1803_;
v_isShared_1807_ = v_isSharedCheck_1812_;
goto v_resetjp_1805_;
}
else
{
lean_inc(v_a_1804_);
lean_dec(v___x_1803_);
v___x_1806_ = lean_box(0);
v_isShared_1807_ = v_isSharedCheck_1812_;
goto v_resetjp_1805_;
}
v_resetjp_1805_:
{
lean_object* v___x_1808_; lean_object* v___x_1810_; 
v___x_1808_ = lean_alloc_ctor(4, 1, 0);
lean_ctor_set(v___x_1808_, 0, v_a_1804_);
if (v_isShared_1807_ == 0)
{
lean_ctor_set(v___x_1806_, 0, v___x_1808_);
v___x_1810_ = v___x_1806_;
goto v_reusejp_1809_;
}
else
{
lean_object* v_reuseFailAlloc_1811_; 
v_reuseFailAlloc_1811_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1811_, 0, v___x_1808_);
v___x_1810_ = v_reuseFailAlloc_1811_;
goto v_reusejp_1809_;
}
v_reusejp_1809_:
{
return v___x_1810_;
}
}
}
else
{
lean_object* v_a_1813_; lean_object* v___x_1815_; uint8_t v_isShared_1816_; uint8_t v_isSharedCheck_1820_; 
v_a_1813_ = lean_ctor_get(v___x_1803_, 0);
v_isSharedCheck_1820_ = !lean_is_exclusive(v___x_1803_);
if (v_isSharedCheck_1820_ == 0)
{
v___x_1815_ = v___x_1803_;
v_isShared_1816_ = v_isSharedCheck_1820_;
goto v_resetjp_1814_;
}
else
{
lean_inc(v_a_1813_);
lean_dec(v___x_1803_);
v___x_1815_ = lean_box(0);
v_isShared_1816_ = v_isSharedCheck_1820_;
goto v_resetjp_1814_;
}
v_resetjp_1814_:
{
lean_object* v___x_1818_; 
if (v_isShared_1816_ == 0)
{
v___x_1818_ = v___x_1815_;
goto v_reusejp_1817_;
}
else
{
lean_object* v_reuseFailAlloc_1819_; 
v_reuseFailAlloc_1819_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1819_, 0, v_a_1813_);
v___x_1818_ = v_reuseFailAlloc_1819_;
goto v_reusejp_1817_;
}
v_reusejp_1817_:
{
return v___x_1818_;
}
}
}
}
}
else
{
lean_object* v___x_1821_; lean_object* v___x_1822_; lean_object* v___x_1823_; 
lean_dec(v_a_1747_);
v___x_1821_ = lean_unsigned_to_nat(3u);
v___x_1822_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult___closed__0));
v___x_1823_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__1___redArg(v___x_1821_, v___x_1822_, v_a_1739_, v_a_1740_, v_a_1741_, v_a_1742_, v_a_1743_, v_a_1744_);
if (lean_obj_tag(v___x_1823_) == 0)
{
lean_object* v_a_1824_; lean_object* v___x_1826_; uint8_t v_isShared_1827_; uint8_t v_isSharedCheck_1832_; 
v_a_1824_ = lean_ctor_get(v___x_1823_, 0);
v_isSharedCheck_1832_ = !lean_is_exclusive(v___x_1823_);
if (v_isSharedCheck_1832_ == 0)
{
v___x_1826_ = v___x_1823_;
v_isShared_1827_ = v_isSharedCheck_1832_;
goto v_resetjp_1825_;
}
else
{
lean_inc(v_a_1824_);
lean_dec(v___x_1823_);
v___x_1826_ = lean_box(0);
v_isShared_1827_ = v_isSharedCheck_1832_;
goto v_resetjp_1825_;
}
v_resetjp_1825_:
{
lean_object* v___x_1828_; lean_object* v___x_1830_; 
v___x_1828_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1828_, 0, v_a_1824_);
if (v_isShared_1827_ == 0)
{
lean_ctor_set(v___x_1826_, 0, v___x_1828_);
v___x_1830_ = v___x_1826_;
goto v_reusejp_1829_;
}
else
{
lean_object* v_reuseFailAlloc_1831_; 
v_reuseFailAlloc_1831_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1831_, 0, v___x_1828_);
v___x_1830_ = v_reuseFailAlloc_1831_;
goto v_reusejp_1829_;
}
v_reusejp_1829_:
{
return v___x_1830_;
}
}
}
else
{
lean_object* v_a_1833_; lean_object* v___x_1835_; uint8_t v_isShared_1836_; uint8_t v_isSharedCheck_1840_; 
v_a_1833_ = lean_ctor_get(v___x_1823_, 0);
v_isSharedCheck_1840_ = !lean_is_exclusive(v___x_1823_);
if (v_isSharedCheck_1840_ == 0)
{
v___x_1835_ = v___x_1823_;
v_isShared_1836_ = v_isSharedCheck_1840_;
goto v_resetjp_1834_;
}
else
{
lean_inc(v_a_1833_);
lean_dec(v___x_1823_);
v___x_1835_ = lean_box(0);
v_isShared_1836_ = v_isSharedCheck_1840_;
goto v_resetjp_1834_;
}
v_resetjp_1834_:
{
lean_object* v___x_1838_; 
if (v_isShared_1836_ == 0)
{
v___x_1838_ = v___x_1835_;
goto v_reusejp_1837_;
}
else
{
lean_object* v_reuseFailAlloc_1839_; 
v_reuseFailAlloc_1839_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1839_, 0, v_a_1833_);
v___x_1838_ = v_reuseFailAlloc_1839_;
goto v_reusejp_1837_;
}
v_reusejp_1837_:
{
return v___x_1838_;
}
}
}
}
}
else
{
lean_object* v___x_1841_; lean_object* v___x_1842_; lean_object* v___x_1843_; 
lean_dec(v_a_1747_);
v___x_1841_ = lean_unsigned_to_nat(3u);
v___x_1842_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult___closed__0));
v___x_1843_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__1___redArg(v___x_1841_, v___x_1842_, v_a_1739_, v_a_1740_, v_a_1741_, v_a_1742_, v_a_1743_, v_a_1744_);
if (lean_obj_tag(v___x_1843_) == 0)
{
lean_object* v_a_1844_; lean_object* v___x_1846_; uint8_t v_isShared_1847_; uint8_t v_isSharedCheck_1852_; 
v_a_1844_ = lean_ctor_get(v___x_1843_, 0);
v_isSharedCheck_1852_ = !lean_is_exclusive(v___x_1843_);
if (v_isSharedCheck_1852_ == 0)
{
v___x_1846_ = v___x_1843_;
v_isShared_1847_ = v_isSharedCheck_1852_;
goto v_resetjp_1845_;
}
else
{
lean_inc(v_a_1844_);
lean_dec(v___x_1843_);
v___x_1846_ = lean_box(0);
v_isShared_1847_ = v_isSharedCheck_1852_;
goto v_resetjp_1845_;
}
v_resetjp_1845_:
{
lean_object* v___x_1848_; lean_object* v___x_1850_; 
v___x_1848_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_1848_, 0, v_a_1844_);
if (v_isShared_1847_ == 0)
{
lean_ctor_set(v___x_1846_, 0, v___x_1848_);
v___x_1850_ = v___x_1846_;
goto v_reusejp_1849_;
}
else
{
lean_object* v_reuseFailAlloc_1851_; 
v_reuseFailAlloc_1851_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1851_, 0, v___x_1848_);
v___x_1850_ = v_reuseFailAlloc_1851_;
goto v_reusejp_1849_;
}
v_reusejp_1849_:
{
return v___x_1850_;
}
}
}
else
{
lean_object* v_a_1853_; lean_object* v___x_1855_; uint8_t v_isShared_1856_; uint8_t v_isSharedCheck_1860_; 
v_a_1853_ = lean_ctor_get(v___x_1843_, 0);
v_isSharedCheck_1860_ = !lean_is_exclusive(v___x_1843_);
if (v_isSharedCheck_1860_ == 0)
{
v___x_1855_ = v___x_1843_;
v_isShared_1856_ = v_isSharedCheck_1860_;
goto v_resetjp_1854_;
}
else
{
lean_inc(v_a_1853_);
lean_dec(v___x_1843_);
v___x_1855_ = lean_box(0);
v_isShared_1856_ = v_isSharedCheck_1860_;
goto v_resetjp_1854_;
}
v_resetjp_1854_:
{
lean_object* v___x_1858_; 
if (v_isShared_1856_ == 0)
{
v___x_1858_ = v___x_1855_;
goto v_reusejp_1857_;
}
else
{
lean_object* v_reuseFailAlloc_1859_; 
v_reuseFailAlloc_1859_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1859_, 0, v_a_1853_);
v___x_1858_ = v_reuseFailAlloc_1859_;
goto v_reusejp_1857_;
}
v_reusejp_1857_:
{
return v___x_1858_;
}
}
}
}
}
else
{
lean_object* v___x_1861_; lean_object* v___x_1863_; 
lean_dec(v_a_1747_);
v___x_1861_ = lean_box(1);
if (v_isShared_1750_ == 0)
{
lean_ctor_set(v___x_1749_, 0, v___x_1861_);
v___x_1863_ = v___x_1749_;
goto v_reusejp_1862_;
}
else
{
lean_object* v_reuseFailAlloc_1864_; 
v_reuseFailAlloc_1864_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1864_, 0, v___x_1861_);
v___x_1863_ = v_reuseFailAlloc_1864_;
goto v_reusejp_1862_;
}
v_reusejp_1862_:
{
return v___x_1863_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult___boxed(lean_object* v_a_1866_, lean_object* v_a_1867_, lean_object* v_a_1868_, lean_object* v_a_1869_, lean_object* v_a_1870_, lean_object* v_a_1871_, lean_object* v_a_1872_){
_start:
{
lean_object* v_res_1873_; 
v_res_1873_ = lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult(v_a_1866_, v_a_1867_, v_a_1868_, v_a_1869_, v_a_1870_, v_a_1871_);
lean_dec(v_a_1871_);
lean_dec_ref(v_a_1870_);
lean_dec(v_a_1869_);
lean_dec_ref(v_a_1868_);
lean_dec(v_a_1867_);
lean_dec_ref(v_a_1866_);
return v_res_1873_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__1_spec__1(lean_object* v___y_1874_, lean_object* v___y_1875_, lean_object* v___y_1876_, lean_object* v___y_1877_, lean_object* v___y_1878_, lean_object* v___y_1879_){
_start:
{
lean_object* v___x_1881_; 
v___x_1881_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__1_spec__1___redArg(v___y_1874_);
return v___x_1881_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__1_spec__1___boxed(lean_object* v___y_1882_, lean_object* v___y_1883_, lean_object* v___y_1884_, lean_object* v___y_1885_, lean_object* v___y_1886_, lean_object* v___y_1887_, lean_object* v___y_1888_){
_start:
{
lean_object* v_res_1889_; 
v_res_1889_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__1_spec__1(v___y_1882_, v___y_1883_, v___y_1884_, v___y_1885_, v___y_1886_, v___y_1887_);
lean_dec(v___y_1887_);
lean_dec_ref(v___y_1886_);
lean_dec(v___y_1885_);
lean_dec_ref(v___y_1884_);
lean_dec(v___y_1883_);
lean_dec_ref(v___y_1882_);
return v_res_1889_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__1(lean_object* v_00_u03b1_1890_, lean_object* v_argIdx_1891_, lean_object* v_x_1892_, lean_object* v___y_1893_, lean_object* v___y_1894_, lean_object* v___y_1895_, lean_object* v___y_1896_, lean_object* v___y_1897_, lean_object* v___y_1898_){
_start:
{
lean_object* v___x_1900_; 
v___x_1900_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__1___redArg(v_argIdx_1891_, v_x_1892_, v___y_1893_, v___y_1894_, v___y_1895_, v___y_1896_, v___y_1897_, v___y_1898_);
return v___x_1900_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__1___boxed(lean_object* v_00_u03b1_1901_, lean_object* v_argIdx_1902_, lean_object* v_x_1903_, lean_object* v___y_1904_, lean_object* v___y_1905_, lean_object* v___y_1906_, lean_object* v___y_1907_, lean_object* v___y_1908_, lean_object* v___y_1909_, lean_object* v___y_1910_){
_start:
{
lean_object* v_res_1911_; 
v_res_1911_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__1(v_00_u03b1_1901_, v_argIdx_1902_, v_x_1903_, v___y_1904_, v___y_1905_, v___y_1906_, v___y_1907_, v___y_1908_, v___y_1909_);
lean_dec(v___y_1909_);
lean_dec_ref(v___y_1908_);
lean_dec(v___y_1907_);
lean_dec_ref(v___y_1906_);
lean_dec(v___y_1905_);
lean_dec_ref(v___y_1904_);
lean_dec(v_argIdx_1902_);
return v_res_1911_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0___redArg___lam__1(lean_object* v_x_1912_, lean_object* v___y_1913_, lean_object* v___y_1914_, lean_object* v___y_1915_, lean_object* v___y_1916_, lean_object* v___y_1917_, lean_object* v___y_1918_){
_start:
{
lean_object* v___x_1920_; lean_object* v___x_1921_; 
v___x_1920_ = lean_box(0);
v___x_1921_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1921_, 0, v___x_1920_);
return v___x_1921_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0___redArg___lam__1___boxed(lean_object* v_x_1922_, lean_object* v___y_1923_, lean_object* v___y_1924_, lean_object* v___y_1925_, lean_object* v___y_1926_, lean_object* v___y_1927_, lean_object* v___y_1928_, lean_object* v___y_1929_){
_start:
{
lean_object* v_res_1930_; 
v_res_1930_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0___redArg___lam__1(v_x_1922_, v___y_1923_, v___y_1924_, v___y_1925_, v___y_1926_, v___y_1927_, v___y_1928_);
lean_dec(v___y_1928_);
lean_dec_ref(v___y_1927_);
lean_dec(v___y_1926_);
lean_dec_ref(v___y_1925_);
lean_dec(v___y_1924_);
lean_dec_ref(v___y_1923_);
lean_dec_ref(v_x_1922_);
return v_res_1930_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0___redArg___lam__0(lean_object* v_x_1931_, lean_object* v_x_1932_, lean_object* v___y_1933_, lean_object* v___y_1934_, lean_object* v___y_1935_, lean_object* v___y_1936_, lean_object* v___y_1937_, lean_object* v___y_1938_){
_start:
{
lean_object* v___x_1940_; 
lean_inc(v___y_1938_);
lean_inc_ref(v___y_1937_);
lean_inc(v___y_1936_);
lean_inc_ref(v___y_1935_);
lean_inc(v___y_1934_);
lean_inc_ref(v___y_1933_);
v___x_1940_ = lean_apply_7(v_x_1931_, v___y_1933_, v___y_1934_, v___y_1935_, v___y_1936_, v___y_1937_, v___y_1938_, lean_box(0));
return v___x_1940_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0___redArg___lam__0___boxed(lean_object* v_x_1941_, lean_object* v_x_1942_, lean_object* v___y_1943_, lean_object* v___y_1944_, lean_object* v___y_1945_, lean_object* v___y_1946_, lean_object* v___y_1947_, lean_object* v___y_1948_, lean_object* v___y_1949_){
_start:
{
lean_object* v_res_1950_; 
v_res_1950_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0___redArg___lam__0(v_x_1941_, v_x_1942_, v___y_1943_, v___y_1944_, v___y_1945_, v___y_1946_, v___y_1947_, v___y_1948_);
lean_dec(v___y_1948_);
lean_dec_ref(v___y_1947_);
lean_dec(v___y_1946_);
lean_dec_ref(v___y_1945_);
lean_dec(v___y_1944_);
lean_dec_ref(v___y_1943_);
return v_res_1950_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0_spec__0_spec__2___redArg___lam__0(lean_object* v_k_1951_, lean_object* v___y_1952_, lean_object* v___y_1953_, lean_object* v_b_1954_, lean_object* v___y_1955_, lean_object* v___y_1956_, lean_object* v___y_1957_, lean_object* v___y_1958_){
_start:
{
lean_object* v___x_1960_; 
lean_inc(v___y_1958_);
lean_inc_ref(v___y_1957_);
lean_inc(v___y_1956_);
lean_inc_ref(v___y_1955_);
lean_inc(v___y_1953_);
lean_inc_ref(v___y_1952_);
v___x_1960_ = lean_apply_8(v_k_1951_, v_b_1954_, v___y_1952_, v___y_1953_, v___y_1955_, v___y_1956_, v___y_1957_, v___y_1958_, lean_box(0));
return v___x_1960_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0_spec__0_spec__2___redArg___lam__0___boxed(lean_object* v_k_1961_, lean_object* v___y_1962_, lean_object* v___y_1963_, lean_object* v_b_1964_, lean_object* v___y_1965_, lean_object* v___y_1966_, lean_object* v___y_1967_, lean_object* v___y_1968_, lean_object* v___y_1969_){
_start:
{
lean_object* v_res_1970_; 
v_res_1970_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0_spec__0_spec__2___redArg___lam__0(v_k_1961_, v___y_1962_, v___y_1963_, v_b_1964_, v___y_1965_, v___y_1966_, v___y_1967_, v___y_1968_);
lean_dec(v___y_1968_);
lean_dec_ref(v___y_1967_);
lean_dec(v___y_1966_);
lean_dec_ref(v___y_1965_);
lean_dec(v___y_1963_);
lean_dec_ref(v___y_1962_);
return v_res_1970_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0_spec__0_spec__2___redArg(lean_object* v_name_1971_, uint8_t v_bi_1972_, lean_object* v_type_1973_, lean_object* v_k_1974_, uint8_t v_kind_1975_, lean_object* v___y_1976_, lean_object* v___y_1977_, lean_object* v___y_1978_, lean_object* v___y_1979_, lean_object* v___y_1980_, lean_object* v___y_1981_){
_start:
{
lean_object* v___f_1983_; lean_object* v___x_1984_; 
lean_inc(v___y_1977_);
lean_inc_ref(v___y_1976_);
v___f_1983_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0_spec__0_spec__2___redArg___lam__0___boxed), 9, 3);
lean_closure_set(v___f_1983_, 0, v_k_1974_);
lean_closure_set(v___f_1983_, 1, v___y_1976_);
lean_closure_set(v___f_1983_, 2, v___y_1977_);
v___x_1984_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_box(0), v_name_1971_, v_bi_1972_, v_type_1973_, v___f_1983_, v_kind_1975_, v___y_1978_, v___y_1979_, v___y_1980_, v___y_1981_);
if (lean_obj_tag(v___x_1984_) == 0)
{
return v___x_1984_;
}
else
{
lean_object* v_a_1985_; lean_object* v___x_1987_; uint8_t v_isShared_1988_; uint8_t v_isSharedCheck_1992_; 
v_a_1985_ = lean_ctor_get(v___x_1984_, 0);
v_isSharedCheck_1992_ = !lean_is_exclusive(v___x_1984_);
if (v_isSharedCheck_1992_ == 0)
{
v___x_1987_ = v___x_1984_;
v_isShared_1988_ = v_isSharedCheck_1992_;
goto v_resetjp_1986_;
}
else
{
lean_inc(v_a_1985_);
lean_dec(v___x_1984_);
v___x_1987_ = lean_box(0);
v_isShared_1988_ = v_isSharedCheck_1992_;
goto v_resetjp_1986_;
}
v_resetjp_1986_:
{
lean_object* v___x_1990_; 
if (v_isShared_1988_ == 0)
{
v___x_1990_ = v___x_1987_;
goto v_reusejp_1989_;
}
else
{
lean_object* v_reuseFailAlloc_1991_; 
v_reuseFailAlloc_1991_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1991_, 0, v_a_1985_);
v___x_1990_ = v_reuseFailAlloc_1991_;
goto v_reusejp_1989_;
}
v_reusejp_1989_:
{
return v___x_1990_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0_spec__0_spec__2___redArg___boxed(lean_object* v_name_1993_, lean_object* v_bi_1994_, lean_object* v_type_1995_, lean_object* v_k_1996_, lean_object* v_kind_1997_, lean_object* v___y_1998_, lean_object* v___y_1999_, lean_object* v___y_2000_, lean_object* v___y_2001_, lean_object* v___y_2002_, lean_object* v___y_2003_, lean_object* v___y_2004_){
_start:
{
uint8_t v_bi_boxed_2005_; uint8_t v_kind_boxed_2006_; lean_object* v_res_2007_; 
v_bi_boxed_2005_ = lean_unbox(v_bi_1994_);
v_kind_boxed_2006_ = lean_unbox(v_kind_1997_);
v_res_2007_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0_spec__0_spec__2___redArg(v_name_1993_, v_bi_boxed_2005_, v_type_1995_, v_k_1996_, v_kind_boxed_2006_, v___y_1998_, v___y_1999_, v___y_2000_, v___y_2001_, v___y_2002_, v___y_2003_);
lean_dec(v___y_2003_);
lean_dec_ref(v___y_2002_);
lean_dec(v___y_2001_);
lean_dec_ref(v___y_2000_);
lean_dec(v___y_1999_);
lean_dec_ref(v___y_1998_);
return v_res_2007_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0_spec__0_spec__1___redArg(lean_object* v_child_2008_, lean_object* v_childIdx_2009_, lean_object* v_x_2010_, lean_object* v___y_2011_, lean_object* v___y_2012_, lean_object* v___y_2013_, lean_object* v___y_2014_, lean_object* v___y_2015_, lean_object* v___y_2016_){
_start:
{
lean_object* v_subExpr_2018_; lean_object* v_optionsPerPos_2019_; lean_object* v_currNamespace_2020_; lean_object* v_openDecls_2021_; uint8_t v_inPattern_2022_; lean_object* v_depth_2023_; lean_object* v_lctxInitIndices_2024_; lean_object* v_pos_2025_; lean_object* v___x_2026_; lean_object* v___x_2027_; lean_object* v___x_2028_; lean_object* v___x_2029_; 
v_subExpr_2018_ = lean_ctor_get(v___y_2011_, 3);
v_optionsPerPos_2019_ = lean_ctor_get(v___y_2011_, 0);
v_currNamespace_2020_ = lean_ctor_get(v___y_2011_, 1);
v_openDecls_2021_ = lean_ctor_get(v___y_2011_, 2);
v_inPattern_2022_ = lean_ctor_get_uint8(v___y_2011_, sizeof(void*)*6);
v_depth_2023_ = lean_ctor_get(v___y_2011_, 4);
v_lctxInitIndices_2024_ = lean_ctor_get(v___y_2011_, 5);
v_pos_2025_ = lean_ctor_get(v_subExpr_2018_, 1);
v___x_2026_ = l_Lean_SubExpr_Pos_push(v_pos_2025_, v_childIdx_2009_);
v___x_2027_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2027_, 0, v_child_2008_);
lean_ctor_set(v___x_2027_, 1, v___x_2026_);
lean_inc(v_lctxInitIndices_2024_);
lean_inc(v_depth_2023_);
lean_inc(v_openDecls_2021_);
lean_inc(v_currNamespace_2020_);
lean_inc(v_optionsPerPos_2019_);
v___x_2028_ = lean_alloc_ctor(0, 6, 1);
lean_ctor_set(v___x_2028_, 0, v_optionsPerPos_2019_);
lean_ctor_set(v___x_2028_, 1, v_currNamespace_2020_);
lean_ctor_set(v___x_2028_, 2, v_openDecls_2021_);
lean_ctor_set(v___x_2028_, 3, v___x_2027_);
lean_ctor_set(v___x_2028_, 4, v_depth_2023_);
lean_ctor_set(v___x_2028_, 5, v_lctxInitIndices_2024_);
lean_ctor_set_uint8(v___x_2028_, sizeof(void*)*6, v_inPattern_2022_);
lean_inc(v___y_2016_);
lean_inc_ref(v___y_2015_);
lean_inc(v___y_2014_);
lean_inc_ref(v___y_2013_);
lean_inc(v___y_2012_);
v___x_2029_ = lean_apply_7(v_x_2010_, v___x_2028_, v___y_2012_, v___y_2013_, v___y_2014_, v___y_2015_, v___y_2016_, lean_box(0));
return v___x_2029_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0_spec__0_spec__1___redArg___boxed(lean_object* v_child_2030_, lean_object* v_childIdx_2031_, lean_object* v_x_2032_, lean_object* v___y_2033_, lean_object* v___y_2034_, lean_object* v___y_2035_, lean_object* v___y_2036_, lean_object* v___y_2037_, lean_object* v___y_2038_, lean_object* v___y_2039_){
_start:
{
lean_object* v_res_2040_; 
v_res_2040_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0_spec__0_spec__1___redArg(v_child_2030_, v_childIdx_2031_, v_x_2032_, v___y_2033_, v___y_2034_, v___y_2035_, v___y_2036_, v___y_2037_, v___y_2038_);
lean_dec(v___y_2038_);
lean_dec_ref(v___y_2037_);
lean_dec(v___y_2036_);
lean_dec_ref(v___y_2035_);
lean_dec(v___y_2034_);
lean_dec_ref(v___y_2033_);
return v_res_2040_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0_spec__0___redArg___lam__0(lean_object* v_v_2041_, lean_object* v_a_2042_, lean_object* v_x_2043_, lean_object* v_fvar_2044_, lean_object* v___y_2045_, lean_object* v___y_2046_, lean_object* v___y_2047_, lean_object* v___y_2048_, lean_object* v___y_2049_, lean_object* v___y_2050_){
_start:
{
lean_object* v___x_2052_; 
lean_inc(v___y_2050_);
lean_inc_ref(v___y_2049_);
lean_inc(v___y_2048_);
lean_inc_ref(v___y_2047_);
lean_inc(v___y_2046_);
lean_inc_ref(v___y_2045_);
lean_inc_ref(v_fvar_2044_);
v___x_2052_ = lean_apply_8(v_v_2041_, v_fvar_2044_, v___y_2045_, v___y_2046_, v___y_2047_, v___y_2048_, v___y_2049_, v___y_2050_, lean_box(0));
if (lean_obj_tag(v___x_2052_) == 0)
{
lean_object* v_a_2053_; lean_object* v___x_2054_; lean_object* v___x_2055_; lean_object* v___x_2056_; lean_object* v___x_2057_; lean_object* v___x_2058_; 
v_a_2053_ = lean_ctor_get(v___x_2052_, 0);
lean_inc(v_a_2053_);
lean_dec_ref_known(v___x_2052_, 1);
v___x_2054_ = l_Lean_Expr_bindingBody_x21(v_a_2042_);
v___x_2055_ = lean_expr_instantiate1(v___x_2054_, v_fvar_2044_);
lean_dec_ref(v_fvar_2044_);
lean_dec_ref(v___x_2054_);
v___x_2056_ = lean_unsigned_to_nat(1u);
v___x_2057_ = lean_apply_1(v_x_2043_, v_a_2053_);
v___x_2058_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0_spec__0_spec__1___redArg(v___x_2055_, v___x_2056_, v___x_2057_, v___y_2045_, v___y_2046_, v___y_2047_, v___y_2048_, v___y_2049_, v___y_2050_);
return v___x_2058_;
}
else
{
lean_object* v_a_2059_; lean_object* v___x_2061_; uint8_t v_isShared_2062_; uint8_t v_isSharedCheck_2066_; 
lean_dec_ref(v_fvar_2044_);
lean_dec_ref(v_x_2043_);
v_a_2059_ = lean_ctor_get(v___x_2052_, 0);
v_isSharedCheck_2066_ = !lean_is_exclusive(v___x_2052_);
if (v_isSharedCheck_2066_ == 0)
{
v___x_2061_ = v___x_2052_;
v_isShared_2062_ = v_isSharedCheck_2066_;
goto v_resetjp_2060_;
}
else
{
lean_inc(v_a_2059_);
lean_dec(v___x_2052_);
v___x_2061_ = lean_box(0);
v_isShared_2062_ = v_isSharedCheck_2066_;
goto v_resetjp_2060_;
}
v_resetjp_2060_:
{
lean_object* v___x_2064_; 
if (v_isShared_2062_ == 0)
{
v___x_2064_ = v___x_2061_;
goto v_reusejp_2063_;
}
else
{
lean_object* v_reuseFailAlloc_2065_; 
v_reuseFailAlloc_2065_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2065_, 0, v_a_2059_);
v___x_2064_ = v_reuseFailAlloc_2065_;
goto v_reusejp_2063_;
}
v_reusejp_2063_:
{
return v___x_2064_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0_spec__0___redArg___lam__0___boxed(lean_object* v_v_2067_, lean_object* v_a_2068_, lean_object* v_x_2069_, lean_object* v_fvar_2070_, lean_object* v___y_2071_, lean_object* v___y_2072_, lean_object* v___y_2073_, lean_object* v___y_2074_, lean_object* v___y_2075_, lean_object* v___y_2076_, lean_object* v___y_2077_){
_start:
{
lean_object* v_res_2078_; 
v_res_2078_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0_spec__0___redArg___lam__0(v_v_2067_, v_a_2068_, v_x_2069_, v_fvar_2070_, v___y_2071_, v___y_2072_, v___y_2073_, v___y_2074_, v___y_2075_, v___y_2076_);
lean_dec(v___y_2076_);
lean_dec_ref(v___y_2075_);
lean_dec(v___y_2074_);
lean_dec_ref(v___y_2073_);
lean_dec(v___y_2072_);
lean_dec_ref(v___y_2071_);
lean_dec_ref(v_a_2068_);
return v_res_2078_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0_spec__0___redArg(lean_object* v_n_2079_, lean_object* v_v_2080_, lean_object* v_x_2081_, lean_object* v___y_2082_, lean_object* v___y_2083_, lean_object* v___y_2084_, lean_object* v___y_2085_, lean_object* v___y_2086_, lean_object* v___y_2087_){
_start:
{
lean_object* v___x_2089_; lean_object* v_a_2090_; lean_object* v___f_2091_; uint8_t v___x_2092_; lean_object* v___x_2093_; uint8_t v___x_2094_; lean_object* v___x_2095_; 
v___x_2089_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__0___redArg(v___y_2082_);
v_a_2090_ = lean_ctor_get(v___x_2089_, 0);
lean_inc_n(v_a_2090_, 2);
lean_dec_ref(v___x_2089_);
v___f_2091_ = lean_alloc_closure((void*)(lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0_spec__0___redArg___lam__0___boxed), 11, 3);
lean_closure_set(v___f_2091_, 0, v_v_2080_);
lean_closure_set(v___f_2091_, 1, v_a_2090_);
lean_closure_set(v___f_2091_, 2, v_x_2081_);
v___x_2092_ = l_Lean_Expr_binderInfo(v_a_2090_);
v___x_2093_ = l_Lean_Expr_bindingDomain_x21(v_a_2090_);
lean_dec(v_a_2090_);
v___x_2094_ = 0;
v___x_2095_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0_spec__0_spec__2___redArg(v_n_2079_, v___x_2092_, v___x_2093_, v___f_2091_, v___x_2094_, v___y_2082_, v___y_2083_, v___y_2084_, v___y_2085_, v___y_2086_, v___y_2087_);
return v___x_2095_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0_spec__0___redArg___boxed(lean_object* v_n_2096_, lean_object* v_v_2097_, lean_object* v_x_2098_, lean_object* v___y_2099_, lean_object* v___y_2100_, lean_object* v___y_2101_, lean_object* v___y_2102_, lean_object* v___y_2103_, lean_object* v___y_2104_, lean_object* v___y_2105_){
_start:
{
lean_object* v_res_2106_; 
v_res_2106_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0_spec__0___redArg(v_n_2096_, v_v_2097_, v_x_2098_, v___y_2099_, v___y_2100_, v___y_2101_, v___y_2102_, v___y_2103_, v___y_2104_);
lean_dec(v___y_2104_);
lean_dec_ref(v___y_2103_);
lean_dec(v___y_2102_);
lean_dec_ref(v___y_2101_);
lean_dec(v___y_2100_);
lean_dec_ref(v___y_2099_);
return v_res_2106_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0___redArg(lean_object* v_n_2108_, lean_object* v_x_2109_, lean_object* v___y_2110_, lean_object* v___y_2111_, lean_object* v___y_2112_, lean_object* v___y_2113_, lean_object* v___y_2114_, lean_object* v___y_2115_){
_start:
{
lean_object* v___f_2117_; lean_object* v___f_2118_; lean_object* v___x_2119_; 
v___f_2117_ = lean_alloc_closure((void*)(lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0___redArg___lam__0___boxed), 9, 1);
lean_closure_set(v___f_2117_, 0, v_x_2109_);
v___f_2118_ = ((lean_object*)(lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0___redArg___closed__0));
v___x_2119_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0_spec__0___redArg(v_n_2108_, v___f_2118_, v___f_2117_, v___y_2110_, v___y_2111_, v___y_2112_, v___y_2113_, v___y_2114_, v___y_2115_);
return v___x_2119_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0___redArg___boxed(lean_object* v_n_2120_, lean_object* v_x_2121_, lean_object* v___y_2122_, lean_object* v___y_2123_, lean_object* v___y_2124_, lean_object* v___y_2125_, lean_object* v___y_2126_, lean_object* v___y_2127_, lean_object* v___y_2128_){
_start:
{
lean_object* v_res_2129_; 
v_res_2129_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0___redArg(v_n_2120_, v_x_2121_, v___y_2122_, v___y_2123_, v___y_2124_, v___y_2125_, v___y_2126_, v___y_2127_);
lean_dec(v___y_2127_);
lean_dec_ref(v___y_2126_);
lean_dec(v___y_2125_);
lean_dec_ref(v___y_2124_);
lean_dec(v___y_2123_);
lean_dec_ref(v___y_2122_);
return v_res_2129_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg___lam__0(lean_object* v_i_2130_, lean_object* v___y_2131_, lean_object* v___y_2132_, lean_object* v___y_2133_, lean_object* v___y_2134_, lean_object* v___y_2135_, lean_object* v___y_2136_){
_start:
{
lean_object* v___x_2138_; lean_object* v_a_2139_; uint8_t v___x_2140_; 
v___x_2138_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__0___redArg(v___y_2131_);
v_a_2139_ = lean_ctor_get(v___x_2138_, 0);
lean_inc(v_a_2139_);
lean_dec_ref(v___x_2138_);
v___x_2140_ = l_Lean_Expr_isLambda(v_a_2139_);
lean_dec(v_a_2139_);
if (v___x_2140_ == 0)
{
lean_object* v___x_2141_; 
v___x_2141_ = l_Lean_PrettyPrinter_Delaborator_delab(v___y_2131_, v___y_2132_, v___y_2133_, v___y_2134_, v___y_2135_, v___y_2136_);
if (lean_obj_tag(v___x_2141_) == 0)
{
lean_object* v_a_2142_; lean_object* v___x_2144_; uint8_t v_isShared_2145_; uint8_t v_isSharedCheck_2155_; 
v_a_2142_ = lean_ctor_get(v___x_2141_, 0);
v_isSharedCheck_2155_ = !lean_is_exclusive(v___x_2141_);
if (v_isSharedCheck_2155_ == 0)
{
v___x_2144_ = v___x_2141_;
v_isShared_2145_ = v_isSharedCheck_2155_;
goto v_resetjp_2143_;
}
else
{
lean_inc(v_a_2142_);
lean_dec(v___x_2141_);
v___x_2144_ = lean_box(0);
v_isShared_2145_ = v_isSharedCheck_2155_;
goto v_resetjp_2143_;
}
v_resetjp_2143_:
{
lean_object* v_ref_2146_; lean_object* v___x_2147_; lean_object* v___x_2148_; lean_object* v___x_2149_; lean_object* v___x_2150_; lean_object* v___x_2151_; lean_object* v___x_2153_; 
v_ref_2146_ = lean_ctor_get(v___y_2135_, 5);
v___x_2147_ = l_Lean_SourceInfo_fromRef(v_ref_2146_, v___x_2140_);
v___x_2148_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__18));
v___x_2149_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__25));
lean_inc(v___x_2147_);
v___x_2150_ = l_Lean_Syntax_node1(v___x_2147_, v___x_2149_, v_i_2130_);
v___x_2151_ = l_Lean_Syntax_node2(v___x_2147_, v___x_2148_, v_a_2142_, v___x_2150_);
if (v_isShared_2145_ == 0)
{
lean_ctor_set(v___x_2144_, 0, v___x_2151_);
v___x_2153_ = v___x_2144_;
goto v_reusejp_2152_;
}
else
{
lean_object* v_reuseFailAlloc_2154_; 
v_reuseFailAlloc_2154_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2154_, 0, v___x_2151_);
v___x_2153_ = v_reuseFailAlloc_2154_;
goto v_reusejp_2152_;
}
v_reusejp_2152_:
{
return v___x_2153_;
}
}
}
else
{
lean_dec(v_i_2130_);
return v___x_2141_;
}
}
else
{
lean_object* v___x_2156_; lean_object* v___x_2157_; lean_object* v___x_2158_; 
v___x_2156_ = l_Lean_TSyntax_getId(v_i_2130_);
lean_dec(v_i_2130_);
v___x_2157_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult___closed__0));
v___x_2158_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0___redArg(v___x_2156_, v___x_2157_, v___y_2131_, v___y_2132_, v___y_2133_, v___y_2134_, v___y_2135_, v___y_2136_);
return v___x_2158_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg___lam__0___boxed(lean_object* v_i_2159_, lean_object* v___y_2160_, lean_object* v___y_2161_, lean_object* v___y_2162_, lean_object* v___y_2163_, lean_object* v___y_2164_, lean_object* v___y_2165_, lean_object* v___y_2166_){
_start:
{
lean_object* v_res_2167_; 
v_res_2167_ = lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg___lam__0(v_i_2159_, v___y_2160_, v___y_2161_, v___y_2162_, v___y_2163_, v___y_2164_, v___y_2165_);
lean_dec(v___y_2165_);
lean_dec_ref(v___y_2164_);
lean_dec(v___y_2163_);
lean_dec_ref(v___y_2162_);
lean_dec(v___y_2161_);
lean_dec_ref(v___y_2160_);
return v_res_2167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg(lean_object* v_i_2168_, lean_object* v_a_2169_, lean_object* v_a_2170_, lean_object* v_a_2171_, lean_object* v_a_2172_, lean_object* v_a_2173_, lean_object* v_a_2174_){
_start:
{
lean_object* v___x_2176_; lean_object* v_a_2177_; lean_object* v___x_2179_; uint8_t v_isShared_2180_; uint8_t v_isSharedCheck_2238_; 
v___x_2176_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__0___redArg(v_a_2169_);
v_a_2177_ = lean_ctor_get(v___x_2176_, 0);
v_isSharedCheck_2238_ = !lean_is_exclusive(v___x_2176_);
if (v_isSharedCheck_2238_ == 0)
{
v___x_2179_ = v___x_2176_;
v_isShared_2180_ = v_isSharedCheck_2238_;
goto v_resetjp_2178_;
}
else
{
lean_inc(v_a_2177_);
lean_dec(v___x_2176_);
v___x_2179_ = lean_box(0);
v_isShared_2180_ = v_isSharedCheck_2238_;
goto v_resetjp_2178_;
}
v_resetjp_2178_:
{
lean_object* v___x_2181_; lean_object* v___x_2182_; uint8_t v___x_2183_; 
v___x_2181_ = ((lean_object*)(lp_mathlib_BigOperators___aux__Mathlib__Algebra__BigOperators__Group__Finset__Defs______macroRules__BigOperators__bigsum__1___closed__20));
v___x_2182_ = lean_unsigned_to_nat(4u);
v___x_2183_ = l_Lean_Expr_isAppOfArity(v_a_2177_, v___x_2181_, v___x_2182_);
lean_dec(v_a_2177_);
if (v___x_2183_ == 0)
{
lean_object* v___x_2184_; 
lean_del_object(v___x_2179_);
lean_dec(v_i_2168_);
v___x_2184_ = lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult(v_a_2169_, v_a_2170_, v_a_2171_, v_a_2172_, v_a_2173_, v_a_2174_);
if (lean_obj_tag(v___x_2184_) == 0)
{
lean_object* v_a_2185_; lean_object* v___x_2187_; uint8_t v_isShared_2188_; uint8_t v_isSharedCheck_2194_; 
v_a_2185_ = lean_ctor_get(v___x_2184_, 0);
v_isSharedCheck_2194_ = !lean_is_exclusive(v___x_2184_);
if (v_isSharedCheck_2194_ == 0)
{
v___x_2187_ = v___x_2184_;
v_isShared_2188_ = v_isSharedCheck_2194_;
goto v_resetjp_2186_;
}
else
{
lean_inc(v_a_2185_);
lean_dec(v___x_2184_);
v___x_2187_ = lean_box(0);
v_isShared_2188_ = v_isSharedCheck_2194_;
goto v_resetjp_2186_;
}
v_resetjp_2186_:
{
lean_object* v___x_2189_; lean_object* v___x_2190_; lean_object* v___x_2192_; 
v___x_2189_ = lean_box(0);
v___x_2190_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2190_, 0, v_a_2185_);
lean_ctor_set(v___x_2190_, 1, v___x_2189_);
if (v_isShared_2188_ == 0)
{
lean_ctor_set(v___x_2187_, 0, v___x_2190_);
v___x_2192_ = v___x_2187_;
goto v_reusejp_2191_;
}
else
{
lean_object* v_reuseFailAlloc_2193_; 
v_reuseFailAlloc_2193_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2193_, 0, v___x_2190_);
v___x_2192_ = v_reuseFailAlloc_2193_;
goto v_reusejp_2191_;
}
v_reusejp_2191_:
{
return v___x_2192_;
}
}
}
else
{
lean_object* v_a_2195_; lean_object* v___x_2197_; uint8_t v_isShared_2198_; uint8_t v_isSharedCheck_2202_; 
v_a_2195_ = lean_ctor_get(v___x_2184_, 0);
v_isSharedCheck_2202_ = !lean_is_exclusive(v___x_2184_);
if (v_isSharedCheck_2202_ == 0)
{
v___x_2197_ = v___x_2184_;
v_isShared_2198_ = v_isSharedCheck_2202_;
goto v_resetjp_2196_;
}
else
{
lean_inc(v_a_2195_);
lean_dec(v___x_2184_);
v___x_2197_ = lean_box(0);
v_isShared_2198_ = v_isSharedCheck_2202_;
goto v_resetjp_2196_;
}
v_resetjp_2196_:
{
lean_object* v___x_2200_; 
if (v_isShared_2198_ == 0)
{
v___x_2200_ = v___x_2197_;
goto v_reusejp_2199_;
}
else
{
lean_object* v_reuseFailAlloc_2201_; 
v_reuseFailAlloc_2201_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2201_, 0, v_a_2195_);
v___x_2200_ = v_reuseFailAlloc_2201_;
goto v_reusejp_2199_;
}
v_reusejp_2199_:
{
return v___x_2200_;
}
}
}
}
else
{
lean_object* v___f_2203_; lean_object* v___x_2204_; lean_object* v___x_2205_; 
v___f_2203_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg___lam__0___boxed), 8, 1);
lean_closure_set(v___f_2203_, 0, v_i_2168_);
v___x_2204_ = lean_unsigned_to_nat(1u);
v___x_2205_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__1___redArg(v___x_2204_, v___f_2203_, v_a_2169_, v_a_2170_, v_a_2171_, v_a_2172_, v_a_2173_, v_a_2174_);
if (lean_obj_tag(v___x_2205_) == 0)
{
lean_object* v_a_2206_; lean_object* v___x_2207_; lean_object* v___x_2208_; lean_object* v___x_2209_; 
v_a_2206_ = lean_ctor_get(v___x_2205_, 0);
lean_inc(v_a_2206_);
lean_dec_ref_known(v___x_2205_, 1);
v___x_2207_ = lean_unsigned_to_nat(3u);
v___x_2208_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult___boxed), 7, 0);
v___x_2209_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__1___redArg(v___x_2207_, v___x_2208_, v_a_2169_, v_a_2170_, v_a_2171_, v_a_2172_, v_a_2173_, v_a_2174_);
if (lean_obj_tag(v___x_2209_) == 0)
{
lean_object* v_a_2210_; lean_object* v___x_2212_; uint8_t v_isShared_2213_; uint8_t v_isSharedCheck_2221_; 
v_a_2210_ = lean_ctor_get(v___x_2209_, 0);
v_isSharedCheck_2221_ = !lean_is_exclusive(v___x_2209_);
if (v_isSharedCheck_2221_ == 0)
{
v___x_2212_ = v___x_2209_;
v_isShared_2213_ = v_isSharedCheck_2221_;
goto v_resetjp_2211_;
}
else
{
lean_inc(v_a_2210_);
lean_dec(v___x_2209_);
v___x_2212_ = lean_box(0);
v_isShared_2213_ = v_isSharedCheck_2221_;
goto v_resetjp_2211_;
}
v_resetjp_2211_:
{
lean_object* v___x_2215_; 
if (v_isShared_2180_ == 0)
{
lean_ctor_set_tag(v___x_2179_, 1);
lean_ctor_set(v___x_2179_, 0, v_a_2206_);
v___x_2215_ = v___x_2179_;
goto v_reusejp_2214_;
}
else
{
lean_object* v_reuseFailAlloc_2220_; 
v_reuseFailAlloc_2220_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2220_, 0, v_a_2206_);
v___x_2215_ = v_reuseFailAlloc_2220_;
goto v_reusejp_2214_;
}
v_reusejp_2214_:
{
lean_object* v___x_2216_; lean_object* v___x_2218_; 
v___x_2216_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2216_, 0, v_a_2210_);
lean_ctor_set(v___x_2216_, 1, v___x_2215_);
if (v_isShared_2213_ == 0)
{
lean_ctor_set(v___x_2212_, 0, v___x_2216_);
v___x_2218_ = v___x_2212_;
goto v_reusejp_2217_;
}
else
{
lean_object* v_reuseFailAlloc_2219_; 
v_reuseFailAlloc_2219_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2219_, 0, v___x_2216_);
v___x_2218_ = v_reuseFailAlloc_2219_;
goto v_reusejp_2217_;
}
v_reusejp_2217_:
{
return v___x_2218_;
}
}
}
}
else
{
lean_object* v_a_2222_; lean_object* v___x_2224_; uint8_t v_isShared_2225_; uint8_t v_isSharedCheck_2229_; 
lean_dec(v_a_2206_);
lean_del_object(v___x_2179_);
v_a_2222_ = lean_ctor_get(v___x_2209_, 0);
v_isSharedCheck_2229_ = !lean_is_exclusive(v___x_2209_);
if (v_isSharedCheck_2229_ == 0)
{
v___x_2224_ = v___x_2209_;
v_isShared_2225_ = v_isSharedCheck_2229_;
goto v_resetjp_2223_;
}
else
{
lean_inc(v_a_2222_);
lean_dec(v___x_2209_);
v___x_2224_ = lean_box(0);
v_isShared_2225_ = v_isSharedCheck_2229_;
goto v_resetjp_2223_;
}
v_resetjp_2223_:
{
lean_object* v___x_2227_; 
if (v_isShared_2225_ == 0)
{
v___x_2227_ = v___x_2224_;
goto v_reusejp_2226_;
}
else
{
lean_object* v_reuseFailAlloc_2228_; 
v_reuseFailAlloc_2228_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2228_, 0, v_a_2222_);
v___x_2227_ = v_reuseFailAlloc_2228_;
goto v_reusejp_2226_;
}
v_reusejp_2226_:
{
return v___x_2227_;
}
}
}
}
else
{
lean_object* v_a_2230_; lean_object* v___x_2232_; uint8_t v_isShared_2233_; uint8_t v_isSharedCheck_2237_; 
lean_del_object(v___x_2179_);
v_a_2230_ = lean_ctor_get(v___x_2205_, 0);
v_isSharedCheck_2237_ = !lean_is_exclusive(v___x_2205_);
if (v_isSharedCheck_2237_ == 0)
{
v___x_2232_ = v___x_2205_;
v_isShared_2233_ = v_isSharedCheck_2237_;
goto v_resetjp_2231_;
}
else
{
lean_inc(v_a_2230_);
lean_dec(v___x_2205_);
v___x_2232_ = lean_box(0);
v_isShared_2233_ = v_isSharedCheck_2237_;
goto v_resetjp_2231_;
}
v_resetjp_2231_:
{
lean_object* v___x_2235_; 
if (v_isShared_2233_ == 0)
{
v___x_2235_ = v___x_2232_;
goto v_reusejp_2234_;
}
else
{
lean_object* v_reuseFailAlloc_2236_; 
v_reuseFailAlloc_2236_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2236_, 0, v_a_2230_);
v___x_2235_ = v_reuseFailAlloc_2236_;
goto v_reusejp_2234_;
}
v_reusejp_2234_:
{
return v___x_2235_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg___boxed(lean_object* v_i_2239_, lean_object* v_a_2240_, lean_object* v_a_2241_, lean_object* v_a_2242_, lean_object* v_a_2243_, lean_object* v_a_2244_, lean_object* v_a_2245_, lean_object* v_a_2246_){
_start:
{
lean_object* v_res_2247_; 
v_res_2247_ = lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg(v_i_2239_, v_a_2240_, v_a_2241_, v_a_2242_, v_a_2243_, v_a_2244_, v_a_2245_);
lean_dec(v_a_2245_);
lean_dec_ref(v_a_2244_);
lean_dec(v_a_2243_);
lean_dec_ref(v_a_2242_);
lean_dec(v_a_2241_);
lean_dec_ref(v_a_2240_);
return v_res_2247_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0(lean_object* v_00_u03b1_2248_, lean_object* v_n_2249_, lean_object* v_x_2250_, lean_object* v___y_2251_, lean_object* v___y_2252_, lean_object* v___y_2253_, lean_object* v___y_2254_, lean_object* v___y_2255_, lean_object* v___y_2256_){
_start:
{
lean_object* v___x_2258_; 
v___x_2258_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0___redArg(v_n_2249_, v_x_2250_, v___y_2251_, v___y_2252_, v___y_2253_, v___y_2254_, v___y_2255_, v___y_2256_);
return v___x_2258_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0___boxed(lean_object* v_00_u03b1_2259_, lean_object* v_n_2260_, lean_object* v_x_2261_, lean_object* v___y_2262_, lean_object* v___y_2263_, lean_object* v___y_2264_, lean_object* v___y_2265_, lean_object* v___y_2266_, lean_object* v___y_2267_, lean_object* v___y_2268_){
_start:
{
lean_object* v_res_2269_; 
v_res_2269_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0(v_00_u03b1_2259_, v_n_2260_, v_x_2261_, v___y_2262_, v___y_2263_, v___y_2264_, v___y_2265_, v___y_2266_, v___y_2267_);
lean_dec(v___y_2267_);
lean_dec_ref(v___y_2266_);
lean_dec(v___y_2265_);
lean_dec_ref(v___y_2264_);
lean_dec(v___y_2263_);
lean_dec_ref(v___y_2262_);
return v_res_2269_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0_spec__0_spec__1(lean_object* v_00_u03b1_2270_, lean_object* v_child_2271_, lean_object* v_childIdx_2272_, lean_object* v_x_2273_, lean_object* v___y_2274_, lean_object* v___y_2275_, lean_object* v___y_2276_, lean_object* v___y_2277_, lean_object* v___y_2278_, lean_object* v___y_2279_){
_start:
{
lean_object* v___x_2281_; 
v___x_2281_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0_spec__0_spec__1___redArg(v_child_2271_, v_childIdx_2272_, v_x_2273_, v___y_2274_, v___y_2275_, v___y_2276_, v___y_2277_, v___y_2278_, v___y_2279_);
return v___x_2281_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0_spec__0_spec__1___boxed(lean_object* v_00_u03b1_2282_, lean_object* v_child_2283_, lean_object* v_childIdx_2284_, lean_object* v_x_2285_, lean_object* v___y_2286_, lean_object* v___y_2287_, lean_object* v___y_2288_, lean_object* v___y_2289_, lean_object* v___y_2290_, lean_object* v___y_2291_, lean_object* v___y_2292_){
_start:
{
lean_object* v_res_2293_; 
v_res_2293_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0_spec__0_spec__1(v_00_u03b1_2282_, v_child_2283_, v_childIdx_2284_, v_x_2285_, v___y_2286_, v___y_2287_, v___y_2288_, v___y_2289_, v___y_2290_, v___y_2291_);
lean_dec(v___y_2291_);
lean_dec_ref(v___y_2290_);
lean_dec(v___y_2289_);
lean_dec_ref(v___y_2288_);
lean_dec(v___y_2287_);
lean_dec_ref(v___y_2286_);
return v_res_2293_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0_spec__0_spec__2(lean_object* v_00_u03b1_2294_, lean_object* v_name_2295_, uint8_t v_bi_2296_, lean_object* v_type_2297_, lean_object* v_k_2298_, uint8_t v_kind_2299_, lean_object* v___y_2300_, lean_object* v___y_2301_, lean_object* v___y_2302_, lean_object* v___y_2303_, lean_object* v___y_2304_, lean_object* v___y_2305_){
_start:
{
lean_object* v___x_2307_; 
v___x_2307_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0_spec__0_spec__2___redArg(v_name_2295_, v_bi_2296_, v_type_2297_, v_k_2298_, v_kind_2299_, v___y_2300_, v___y_2301_, v___y_2302_, v___y_2303_, v___y_2304_, v___y_2305_);
return v___x_2307_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0_spec__0_spec__2___boxed(lean_object* v_00_u03b1_2308_, lean_object* v_name_2309_, lean_object* v_bi_2310_, lean_object* v_type_2311_, lean_object* v_k_2312_, lean_object* v_kind_2313_, lean_object* v___y_2314_, lean_object* v___y_2315_, lean_object* v___y_2316_, lean_object* v___y_2317_, lean_object* v___y_2318_, lean_object* v___y_2319_, lean_object* v___y_2320_){
_start:
{
uint8_t v_bi_boxed_2321_; uint8_t v_kind_boxed_2322_; lean_object* v_res_2323_; 
v_bi_boxed_2321_ = lean_unbox(v_bi_2310_);
v_kind_boxed_2322_ = lean_unbox(v_kind_2313_);
v_res_2323_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0_spec__0_spec__2(v_00_u03b1_2308_, v_name_2309_, v_bi_boxed_2321_, v_type_2311_, v_k_2312_, v_kind_boxed_2322_, v___y_2314_, v___y_2315_, v___y_2316_, v___y_2317_, v___y_2318_, v___y_2319_);
lean_dec(v___y_2319_);
lean_dec_ref(v___y_2318_);
lean_dec(v___y_2317_);
lean_dec_ref(v___y_2316_);
lean_dec(v___y_2315_);
lean_dec_ref(v___y_2314_);
return v_res_2323_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0_spec__0(lean_object* v_00_u03b1_2324_, lean_object* v_00_u03b2_2325_, lean_object* v_n_2326_, lean_object* v_v_2327_, lean_object* v_x_2328_, lean_object* v___y_2329_, lean_object* v___y_2330_, lean_object* v___y_2331_, lean_object* v___y_2332_, lean_object* v___y_2333_, lean_object* v___y_2334_){
_start:
{
lean_object* v___x_2336_; 
v___x_2336_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0_spec__0___redArg(v_n_2326_, v_v_2327_, v_x_2328_, v___y_2329_, v___y_2330_, v___y_2331_, v___y_2332_, v___y_2333_, v___y_2334_);
return v___x_2336_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0_spec__0___boxed(lean_object* v_00_u03b1_2337_, lean_object* v_00_u03b2_2338_, lean_object* v_n_2339_, lean_object* v_v_2340_, lean_object* v_x_2341_, lean_object* v___y_2342_, lean_object* v___y_2343_, lean_object* v___y_2344_, lean_object* v___y_2345_, lean_object* v___y_2346_, lean_object* v___y_2347_, lean_object* v___y_2348_){
_start:
{
lean_object* v_res_2349_; 
v_res_2349_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0_spec__0(v_00_u03b1_2337_, v_00_u03b2_2338_, v_n_2339_, v_v_2340_, v_x_2341_, v___y_2342_, v___y_2343_, v___y_2344_, v___y_2345_, v___y_2346_, v___y_2347_);
lean_dec(v___y_2347_);
lean_dec_ref(v___y_2346_);
lean_dec(v___y_2345_);
lean_dec_ref(v___y_2344_);
lean_dec(v___y_2343_);
lean_dec_ref(v___y_2342_);
return v_res_2349_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BigOperators_delabFinsetProd___lam__0(lean_object* v_i_2350_, lean_object* v___y_2351_, lean_object* v___y_2352_, lean_object* v___y_2353_, lean_object* v___y_2354_, lean_object* v___y_2355_, lean_object* v___y_2356_){
_start:
{
lean_object* v___x_2358_; 
v___x_2358_ = l_Lean_PrettyPrinter_Delaborator_delab(v___y_2351_, v___y_2352_, v___y_2353_, v___y_2354_, v___y_2355_, v___y_2356_);
if (lean_obj_tag(v___x_2358_) == 0)
{
lean_object* v_a_2359_; lean_object* v___x_2361_; uint8_t v_isShared_2362_; uint8_t v_isSharedCheck_2367_; 
v_a_2359_ = lean_ctor_get(v___x_2358_, 0);
v_isSharedCheck_2367_ = !lean_is_exclusive(v___x_2358_);
if (v_isSharedCheck_2367_ == 0)
{
v___x_2361_ = v___x_2358_;
v_isShared_2362_ = v_isSharedCheck_2367_;
goto v_resetjp_2360_;
}
else
{
lean_inc(v_a_2359_);
lean_dec(v___x_2358_);
v___x_2361_ = lean_box(0);
v_isShared_2362_ = v_isSharedCheck_2367_;
goto v_resetjp_2360_;
}
v_resetjp_2360_:
{
lean_object* v___x_2363_; lean_object* v___x_2365_; 
v___x_2363_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2363_, 0, v_i_2350_);
lean_ctor_set(v___x_2363_, 1, v_a_2359_);
if (v_isShared_2362_ == 0)
{
lean_ctor_set(v___x_2361_, 0, v___x_2363_);
v___x_2365_ = v___x_2361_;
goto v_reusejp_2364_;
}
else
{
lean_object* v_reuseFailAlloc_2366_; 
v_reuseFailAlloc_2366_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2366_, 0, v___x_2363_);
v___x_2365_ = v_reuseFailAlloc_2366_;
goto v_reusejp_2364_;
}
v_reusejp_2364_:
{
return v___x_2365_;
}
}
}
else
{
lean_object* v_a_2368_; lean_object* v___x_2370_; uint8_t v_isShared_2371_; uint8_t v_isSharedCheck_2375_; 
lean_dec(v_i_2350_);
v_a_2368_ = lean_ctor_get(v___x_2358_, 0);
v_isSharedCheck_2375_ = !lean_is_exclusive(v___x_2358_);
if (v_isSharedCheck_2375_ == 0)
{
v___x_2370_ = v___x_2358_;
v_isShared_2371_ = v_isSharedCheck_2375_;
goto v_resetjp_2369_;
}
else
{
lean_inc(v_a_2368_);
lean_dec(v___x_2358_);
v___x_2370_ = lean_box(0);
v_isShared_2371_ = v_isSharedCheck_2375_;
goto v_resetjp_2369_;
}
v_resetjp_2369_:
{
lean_object* v___x_2373_; 
if (v_isShared_2371_ == 0)
{
v___x_2373_ = v___x_2370_;
goto v_reusejp_2372_;
}
else
{
lean_object* v_reuseFailAlloc_2374_; 
v_reuseFailAlloc_2374_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2374_, 0, v_a_2368_);
v___x_2373_ = v_reuseFailAlloc_2374_;
goto v_reusejp_2372_;
}
v_reusejp_2372_:
{
return v___x_2373_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_BigOperators_delabFinsetProd___lam__0___boxed(lean_object* v_i_2376_, lean_object* v___y_2377_, lean_object* v___y_2378_, lean_object* v___y_2379_, lean_object* v___y_2380_, lean_object* v___y_2381_, lean_object* v___y_2382_, lean_object* v___y_2383_){
_start:
{
lean_object* v_res_2384_; 
v_res_2384_ = lp_mathlib_BigOperators_delabFinsetProd___lam__0(v_i_2376_, v___y_2377_, v___y_2378_, v___y_2379_, v___y_2380_, v___y_2381_, v___y_2382_);
lean_dec(v___y_2382_);
lean_dec_ref(v___y_2381_);
lean_dec(v___y_2380_);
lean_dec_ref(v___y_2379_);
lean_dec(v___y_2378_);
lean_dec_ref(v___y_2377_);
return v_res_2384_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00BigOperators_delabFinsetProd_spec__0___redArg(lean_object* v_x_2385_, lean_object* v___y_2386_, lean_object* v___y_2387_, lean_object* v___y_2388_, lean_object* v___y_2389_, lean_object* v___y_2390_, lean_object* v___y_2391_){
_start:
{
lean_object* v___x_2393_; lean_object* v_a_2394_; lean_object* v___x_2395_; lean_object* v___x_2396_; lean_object* v___x_2397_; 
v___x_2393_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__0___redArg(v___y_2386_);
v_a_2394_ = lean_ctor_get(v___x_2393_, 0);
lean_inc(v_a_2394_);
lean_dec_ref(v___x_2393_);
v___x_2395_ = l_Lean_Expr_appArg_x21(v_a_2394_);
lean_dec(v_a_2394_);
v___x_2396_ = lean_unsigned_to_nat(1u);
v___x_2397_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg_spec__0_spec__0_spec__1___redArg(v___x_2395_, v___x_2396_, v_x_2385_, v___y_2386_, v___y_2387_, v___y_2388_, v___y_2389_, v___y_2390_, v___y_2391_);
return v___x_2397_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00BigOperators_delabFinsetProd_spec__0___redArg___boxed(lean_object* v_x_2398_, lean_object* v___y_2399_, lean_object* v___y_2400_, lean_object* v___y_2401_, lean_object* v___y_2402_, lean_object* v___y_2403_, lean_object* v___y_2404_, lean_object* v___y_2405_){
_start:
{
lean_object* v_res_2406_; 
v_res_2406_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00BigOperators_delabFinsetProd_spec__0___redArg(v_x_2398_, v___y_2399_, v___y_2400_, v___y_2401_, v___y_2402_, v___y_2403_, v___y_2404_);
lean_dec(v___y_2404_);
lean_dec_ref(v___y_2403_);
lean_dec(v___y_2402_);
lean_dec_ref(v___y_2401_);
lean_dec(v___y_2400_);
lean_dec_ref(v___y_2399_);
return v_res_2406_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BigOperators_delabFinsetProd___lam__1(lean_object* v___x_2419_, lean_object* v___f_2420_, lean_object* v___y_2421_, lean_object* v___y_2422_, lean_object* v___y_2423_, lean_object* v___y_2424_, lean_object* v___y_2425_, lean_object* v___y_2426_){
_start:
{
lean_object* v___y_2429_; lean_object* v___y_2430_; lean_object* v___y_2431_; lean_object* v___y_2432_; lean_object* v___y_2433_; lean_object* v___y_2434_; lean_object* v___y_2435_; lean_object* v___y_2436_; lean_object* v___y_2444_; lean_object* v___y_2445_; lean_object* v___y_2446_; lean_object* v___y_2447_; lean_object* v___y_2448_; lean_object* v___y_2449_; lean_object* v___y_2450_; lean_object* v___y_2451_; lean_object* v___y_2459_; lean_object* v___y_2460_; lean_object* v___y_2461_; lean_object* v___y_2462_; lean_object* v___y_2463_; lean_object* v___y_2464_; lean_object* v___y_2465_; lean_object* v___y_2466_; lean_object* v___y_2474_; lean_object* v___y_2475_; lean_object* v___y_2476_; lean_object* v___y_2477_; lean_object* v___y_2478_; lean_object* v___y_2479_; lean_object* v___y_2480_; lean_object* v___y_2481_; lean_object* v___y_2489_; lean_object* v___y_2490_; lean_object* v___y_2491_; lean_object* v___y_2492_; lean_object* v___y_2493_; lean_object* v___y_2494_; lean_object* v___y_2495_; lean_object* v___y_2496_; lean_object* v___y_2504_; lean_object* v___y_2505_; lean_object* v___y_2506_; lean_object* v___y_2507_; lean_object* v___y_2508_; lean_object* v___y_2509_; lean_object* v___y_2510_; lean_object* v___y_2511_; lean_object* v___y_2519_; uint8_t v___y_2520_; lean_object* v___y_2521_; lean_object* v_binder_2522_; lean_object* v_ref_2523_; lean_object* v___y_2536_; uint8_t v___y_2537_; lean_object* v___y_2538_; uint8_t v___y_2539_; lean_object* v___y_2540_; lean_object* v_a_2541_; lean_object* v___x_2732_; lean_object* v_a_2733_; lean_object* v_dummy_2734_; lean_object* v_nargs_2735_; lean_object* v___x_2736_; lean_object* v___x_2737_; lean_object* v___x_2738_; lean_object* v___x_2739_; lean_object* v___x_2740_; uint8_t v___x_2741_; 
v___x_2732_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__0___redArg(v___y_2421_);
v_a_2733_ = lean_ctor_get(v___x_2732_, 0);
lean_inc(v_a_2733_);
lean_dec_ref(v___x_2732_);
v_dummy_2734_ = lean_obj_once(&lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__1___redArg___closed__0, &lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__1___redArg___closed__0_once, _init_lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__1___redArg___closed__0);
v_nargs_2735_ = l_Lean_Expr_getAppNumArgs(v_a_2733_);
lean_inc(v_nargs_2735_);
v___x_2736_ = lean_mk_array(v_nargs_2735_, v_dummy_2734_);
v___x_2737_ = lean_unsigned_to_nat(1u);
v___x_2738_ = lean_nat_sub(v_nargs_2735_, v___x_2737_);
lean_dec(v_nargs_2735_);
v___x_2739_ = l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(v_a_2733_, v___x_2736_, v___x_2738_);
v___x_2740_ = lean_array_get_size(v___x_2739_);
v___x_2741_ = lean_nat_dec_eq(v___x_2740_, v___x_2419_);
if (v___x_2741_ == 0)
{
lean_object* v___x_2742_; 
lean_dec_ref(v___x_2739_);
lean_dec_ref(v___f_2420_);
v___x_2742_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_2742_;
}
else
{
lean_object* v___x_2743_; lean_object* v___x_2744_; uint8_t v___x_2745_; 
v___x_2743_ = lean_unsigned_to_nat(4u);
v___x_2744_ = lean_array_fget(v___x_2739_, v___x_2743_);
lean_dec_ref(v___x_2739_);
v___x_2745_ = l_Lean_Expr_isLambda(v___x_2744_);
lean_dec(v___x_2744_);
if (v___x_2745_ == 0)
{
lean_object* v___x_2746_; 
v___x_2746_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
if (lean_obj_tag(v___x_2746_) == 0)
{
lean_dec_ref_known(v___x_2746_, 1);
goto v___jp_2663_;
}
else
{
lean_object* v_a_2747_; lean_object* v___x_2749_; uint8_t v_isShared_2750_; uint8_t v_isSharedCheck_2754_; 
lean_dec_ref(v___f_2420_);
v_a_2747_ = lean_ctor_get(v___x_2746_, 0);
v_isSharedCheck_2754_ = !lean_is_exclusive(v___x_2746_);
if (v_isSharedCheck_2754_ == 0)
{
v___x_2749_ = v___x_2746_;
v_isShared_2750_ = v_isSharedCheck_2754_;
goto v_resetjp_2748_;
}
else
{
lean_inc(v_a_2747_);
lean_dec(v___x_2746_);
v___x_2749_ = lean_box(0);
v_isShared_2750_ = v_isSharedCheck_2754_;
goto v_resetjp_2748_;
}
v_resetjp_2748_:
{
lean_object* v___x_2752_; 
if (v_isShared_2750_ == 0)
{
v___x_2752_ = v___x_2749_;
goto v_reusejp_2751_;
}
else
{
lean_object* v_reuseFailAlloc_2753_; 
v_reuseFailAlloc_2753_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2753_, 0, v_a_2747_);
v___x_2752_ = v_reuseFailAlloc_2753_;
goto v_reusejp_2751_;
}
v_reusejp_2751_:
{
return v___x_2752_;
}
}
}
}
else
{
goto v___jp_2663_;
}
}
v___jp_2428_:
{
lean_object* v___x_2437_; lean_object* v___x_2438_; lean_object* v___x_2439_; lean_object* v___x_2440_; lean_object* v___x_2441_; lean_object* v___x_2442_; 
lean_inc_ref(v___y_2431_);
v___x_2437_ = l_Array_append___redArg(v___y_2431_, v___y_2436_);
lean_dec_ref(v___y_2436_);
lean_inc_n(v___y_2434_, 2);
v___x_2438_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2438_, 0, v___y_2434_);
lean_ctor_set(v___x_2438_, 1, v___y_2430_);
lean_ctor_set(v___x_2438_, 2, v___x_2437_);
v___x_2439_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__73));
v___x_2440_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2440_, 0, v___y_2434_);
lean_ctor_set(v___x_2440_, 1, v___x_2439_);
lean_inc(v___y_2435_);
v___x_2441_ = l_Lean_Syntax_node5(v___y_2434_, v___y_2435_, v___y_2433_, v___y_2432_, v___x_2438_, v___x_2440_, v___y_2429_);
v___x_2442_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2442_, 0, v___x_2441_);
return v___x_2442_;
}
v___jp_2443_:
{
lean_object* v___x_2452_; lean_object* v___x_2453_; lean_object* v___x_2454_; lean_object* v___x_2455_; lean_object* v___x_2456_; lean_object* v___x_2457_; 
lean_inc_ref(v___y_2449_);
v___x_2452_ = l_Array_append___redArg(v___y_2449_, v___y_2451_);
lean_dec_ref(v___y_2451_);
lean_inc_n(v___y_2448_, 2);
v___x_2453_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2453_, 0, v___y_2448_);
lean_ctor_set(v___x_2453_, 1, v___y_2444_);
lean_ctor_set(v___x_2453_, 2, v___x_2452_);
v___x_2454_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__73));
v___x_2455_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2455_, 0, v___y_2448_);
lean_ctor_set(v___x_2455_, 1, v___x_2454_);
lean_inc(v___y_2450_);
v___x_2456_ = l_Lean_Syntax_node5(v___y_2448_, v___y_2450_, v___y_2446_, v___y_2447_, v___x_2453_, v___x_2455_, v___y_2445_);
v___x_2457_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2457_, 0, v___x_2456_);
return v___x_2457_;
}
v___jp_2458_:
{
lean_object* v___x_2467_; lean_object* v___x_2468_; lean_object* v___x_2469_; lean_object* v___x_2470_; lean_object* v___x_2471_; lean_object* v___x_2472_; 
lean_inc_ref(v___y_2460_);
v___x_2467_ = l_Array_append___redArg(v___y_2460_, v___y_2466_);
lean_dec_ref(v___y_2466_);
lean_inc_n(v___y_2462_, 2);
v___x_2468_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2468_, 0, v___y_2462_);
lean_ctor_set(v___x_2468_, 1, v___y_2463_);
lean_ctor_set(v___x_2468_, 2, v___x_2467_);
v___x_2469_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__73));
v___x_2470_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2470_, 0, v___y_2462_);
lean_ctor_set(v___x_2470_, 1, v___x_2469_);
lean_inc(v___y_2459_);
v___x_2471_ = l_Lean_Syntax_node5(v___y_2462_, v___y_2459_, v___y_2465_, v___y_2464_, v___x_2468_, v___x_2470_, v___y_2461_);
v___x_2472_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2472_, 0, v___x_2471_);
return v___x_2472_;
}
v___jp_2473_:
{
lean_object* v___x_2482_; lean_object* v___x_2483_; lean_object* v___x_2484_; lean_object* v___x_2485_; lean_object* v___x_2486_; lean_object* v___x_2487_; 
lean_inc_ref(v___y_2475_);
v___x_2482_ = l_Array_append___redArg(v___y_2475_, v___y_2481_);
lean_dec_ref(v___y_2481_);
lean_inc_n(v___y_2478_, 2);
v___x_2483_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2483_, 0, v___y_2478_);
lean_ctor_set(v___x_2483_, 1, v___y_2479_);
lean_ctor_set(v___x_2483_, 2, v___x_2482_);
v___x_2484_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__73));
v___x_2485_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2485_, 0, v___y_2478_);
lean_ctor_set(v___x_2485_, 1, v___x_2484_);
lean_inc(v___y_2480_);
v___x_2486_ = l_Lean_Syntax_node5(v___y_2478_, v___y_2480_, v___y_2477_, v___y_2474_, v___x_2483_, v___x_2485_, v___y_2476_);
v___x_2487_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2487_, 0, v___x_2486_);
return v___x_2487_;
}
v___jp_2488_:
{
lean_object* v___x_2497_; lean_object* v___x_2498_; lean_object* v___x_2499_; lean_object* v___x_2500_; lean_object* v___x_2501_; lean_object* v___x_2502_; 
lean_inc_ref(v___y_2495_);
v___x_2497_ = l_Array_append___redArg(v___y_2495_, v___y_2496_);
lean_dec_ref(v___y_2496_);
lean_inc_n(v___y_2493_, 2);
v___x_2498_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2498_, 0, v___y_2493_);
lean_ctor_set(v___x_2498_, 1, v___y_2490_);
lean_ctor_set(v___x_2498_, 2, v___x_2497_);
v___x_2499_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__73));
v___x_2500_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2500_, 0, v___y_2493_);
lean_ctor_set(v___x_2500_, 1, v___x_2499_);
lean_inc(v___y_2494_);
v___x_2501_ = l_Lean_Syntax_node5(v___y_2493_, v___y_2494_, v___y_2492_, v___y_2491_, v___x_2498_, v___x_2500_, v___y_2489_);
v___x_2502_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2502_, 0, v___x_2501_);
return v___x_2502_;
}
v___jp_2503_:
{
lean_object* v___x_2512_; lean_object* v___x_2513_; lean_object* v___x_2514_; lean_object* v___x_2515_; lean_object* v___x_2516_; lean_object* v___x_2517_; 
lean_inc_ref(v___y_2504_);
v___x_2512_ = l_Array_append___redArg(v___y_2504_, v___y_2511_);
lean_dec_ref(v___y_2511_);
lean_inc(v___y_2505_);
lean_inc_n(v___y_2510_, 2);
v___x_2513_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2513_, 0, v___y_2510_);
lean_ctor_set(v___x_2513_, 1, v___y_2505_);
lean_ctor_set(v___x_2513_, 2, v___x_2512_);
v___x_2514_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__73));
v___x_2515_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2515_, 0, v___y_2510_);
lean_ctor_set(v___x_2515_, 1, v___x_2514_);
lean_inc(v___y_2509_);
v___x_2516_ = l_Lean_Syntax_node5(v___y_2510_, v___y_2509_, v___y_2508_, v___y_2507_, v___x_2513_, v___x_2515_, v___y_2506_);
v___x_2517_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2517_, 0, v___x_2516_);
return v___x_2517_;
}
v___jp_2518_:
{
lean_object* v___x_2524_; lean_object* v___x_2525_; lean_object* v___x_2526_; lean_object* v___x_2527_; lean_object* v___x_2528_; lean_object* v___x_2529_; lean_object* v___x_2530_; lean_object* v___x_2531_; 
v___x_2524_ = l_Lean_SourceInfo_fromRef(v_ref_2523_, v___y_2520_);
v___x_2525_ = ((lean_object*)(lp_mathlib_BigOperators_bigprod___closed__1));
v___x_2526_ = ((lean_object*)(lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__0));
lean_inc_n(v___x_2524_, 2);
v___x_2527_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2527_, 0, v___x_2524_);
lean_ctor_set(v___x_2527_, 1, v___x_2526_);
v___x_2528_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBinders___closed__1));
v___x_2529_ = l_Lean_Syntax_node1(v___x_2524_, v___x_2528_, v_binder_2522_);
v___x_2530_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__25));
v___x_2531_ = lean_obj_once(&lp_mathlib_BigOperators_bigOpBindersPattern___closed__0, &lp_mathlib_BigOperators_bigOpBindersPattern___closed__0_once, _init_lp_mathlib_BigOperators_bigOpBindersPattern___closed__0);
if (lean_obj_tag(v___y_2521_) == 1)
{
lean_object* v_val_2532_; lean_object* v___x_2533_; 
v_val_2532_ = lean_ctor_get(v___y_2521_, 0);
lean_inc(v_val_2532_);
lean_dec_ref_known(v___y_2521_, 1);
v___x_2533_ = l_Array_mkArray1___redArg(v_val_2532_);
v___y_2504_ = v___x_2531_;
v___y_2505_ = v___x_2530_;
v___y_2506_ = v___y_2519_;
v___y_2507_ = v___x_2529_;
v___y_2508_ = v___x_2527_;
v___y_2509_ = v___x_2525_;
v___y_2510_ = v___x_2524_;
v___y_2511_ = v___x_2533_;
goto v___jp_2503_;
}
else
{
lean_object* v___x_2534_; 
lean_dec(v___y_2521_);
v___x_2534_ = ((lean_object*)(lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__1));
v___y_2504_ = v___x_2531_;
v___y_2505_ = v___x_2530_;
v___y_2506_ = v___y_2519_;
v___y_2507_ = v___x_2529_;
v___y_2508_ = v___x_2527_;
v___y_2509_ = v___x_2525_;
v___y_2510_ = v___x_2524_;
v___y_2511_ = v___x_2534_;
goto v___jp_2503_;
}
}
v___jp_2535_:
{
switch(lean_obj_tag(v___y_2540_))
{
case 0:
{
lean_object* v_s_2542_; lean_object* v_ref_2543_; lean_object* v___x_2544_; lean_object* v___x_2545_; lean_object* v___x_2546_; lean_object* v___x_2547_; lean_object* v___x_2548_; lean_object* v___x_2549_; lean_object* v___x_2550_; lean_object* v___x_2551_; lean_object* v___x_2552_; lean_object* v___x_2553_; lean_object* v___x_2554_; lean_object* v___x_2555_; lean_object* v___x_2556_; lean_object* v___x_2557_; lean_object* v___x_2558_; 
v_s_2542_ = lean_ctor_get(v___y_2540_, 0);
lean_inc(v_s_2542_);
lean_dec_ref_known(v___y_2540_, 1);
v_ref_2543_ = lean_ctor_get(v___y_2425_, 5);
v___x_2544_ = l_Lean_SourceInfo_fromRef(v_ref_2543_, v___y_2539_);
v___x_2545_ = ((lean_object*)(lp_mathlib_BigOperators_bigprod___closed__1));
v___x_2546_ = ((lean_object*)(lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__0));
lean_inc_n(v___x_2544_, 6);
v___x_2547_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2547_, 0, v___x_2544_);
lean_ctor_set(v___x_2547_, 1, v___x_2546_);
v___x_2548_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBinders___closed__1));
v___x_2549_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBinder___closed__2));
v___x_2550_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__25));
v___x_2551_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__2));
v___x_2552_ = ((lean_object*)(lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__2));
v___x_2553_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2553_, 0, v___x_2544_);
lean_ctor_set(v___x_2553_, 1, v___x_2552_);
v___x_2554_ = l_Lean_Syntax_node2(v___x_2544_, v___x_2551_, v___x_2553_, v_s_2542_);
v___x_2555_ = l_Lean_Syntax_node1(v___x_2544_, v___x_2550_, v___x_2554_);
v___x_2556_ = l_Lean_Syntax_node2(v___x_2544_, v___x_2549_, v___y_2536_, v___x_2555_);
v___x_2557_ = l_Lean_Syntax_node1(v___x_2544_, v___x_2548_, v___x_2556_);
v___x_2558_ = lean_obj_once(&lp_mathlib_BigOperators_bigOpBindersPattern___closed__0, &lp_mathlib_BigOperators_bigOpBindersPattern___closed__0_once, _init_lp_mathlib_BigOperators_bigOpBindersPattern___closed__0);
if (lean_obj_tag(v_a_2541_) == 1)
{
lean_object* v_val_2559_; lean_object* v___x_2560_; 
v_val_2559_ = lean_ctor_get(v_a_2541_, 0);
lean_inc(v_val_2559_);
lean_dec_ref_known(v_a_2541_, 1);
v___x_2560_ = l_Array_mkArray1___redArg(v_val_2559_);
v___y_2489_ = v___y_2538_;
v___y_2490_ = v___x_2550_;
v___y_2491_ = v___x_2557_;
v___y_2492_ = v___x_2547_;
v___y_2493_ = v___x_2544_;
v___y_2494_ = v___x_2545_;
v___y_2495_ = v___x_2558_;
v___y_2496_ = v___x_2560_;
goto v___jp_2488_;
}
else
{
lean_object* v___x_2561_; 
lean_dec(v_a_2541_);
v___x_2561_ = ((lean_object*)(lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__1));
v___y_2489_ = v___y_2538_;
v___y_2490_ = v___x_2550_;
v___y_2491_ = v___x_2557_;
v___y_2492_ = v___x_2547_;
v___y_2493_ = v___x_2544_;
v___y_2494_ = v___x_2545_;
v___y_2495_ = v___x_2558_;
v___y_2496_ = v___x_2561_;
goto v___jp_2488_;
}
}
case 1:
{
if (v___y_2537_ == 0)
{
lean_object* v_ref_2562_; lean_object* v___x_2563_; lean_object* v___x_2564_; lean_object* v___x_2565_; lean_object* v___x_2566_; lean_object* v___x_2567_; lean_object* v___x_2568_; 
v_ref_2562_ = lean_ctor_get(v___y_2425_, 5);
v___x_2563_ = l_Lean_SourceInfo_fromRef(v_ref_2562_, v___y_2539_);
v___x_2564_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBinder___closed__2));
v___x_2565_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__25));
v___x_2566_ = lean_obj_once(&lp_mathlib_BigOperators_bigOpBindersPattern___closed__0, &lp_mathlib_BigOperators_bigOpBindersPattern___closed__0_once, _init_lp_mathlib_BigOperators_bigOpBindersPattern___closed__0);
lean_inc(v___x_2563_);
v___x_2567_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2567_, 0, v___x_2563_);
lean_ctor_set(v___x_2567_, 1, v___x_2565_);
lean_ctor_set(v___x_2567_, 2, v___x_2566_);
v___x_2568_ = l_Lean_Syntax_node2(v___x_2563_, v___x_2564_, v___y_2536_, v___x_2567_);
v___y_2519_ = v___y_2538_;
v___y_2520_ = v___y_2539_;
v___y_2521_ = v_a_2541_;
v_binder_2522_ = v___x_2568_;
v_ref_2523_ = v_ref_2562_;
goto v___jp_2518_;
}
else
{
lean_object* v___x_2569_; lean_object* v___x_2570_; lean_object* v___x_2571_; 
v___x_2569_ = lean_unsigned_to_nat(0u);
v___x_2570_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult___closed__0));
v___x_2571_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__1___redArg(v___x_2569_, v___x_2570_, v___y_2421_, v___y_2422_, v___y_2423_, v___y_2424_, v___y_2425_, v___y_2426_);
if (lean_obj_tag(v___x_2571_) == 0)
{
lean_object* v_a_2572_; lean_object* v_ref_2573_; lean_object* v___x_2574_; lean_object* v___x_2575_; lean_object* v___x_2576_; lean_object* v___x_2577_; lean_object* v___x_2578_; lean_object* v___x_2579_; lean_object* v___x_2580_; lean_object* v___x_2581_; lean_object* v___x_2582_; 
v_a_2572_ = lean_ctor_get(v___x_2571_, 0);
lean_inc(v_a_2572_);
lean_dec_ref_known(v___x_2571_, 1);
v_ref_2573_ = lean_ctor_get(v___y_2425_, 5);
v___x_2574_ = l_Lean_SourceInfo_fromRef(v_ref_2573_, v___y_2539_);
v___x_2575_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBinder___closed__2));
v___x_2576_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__25));
v___x_2577_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBinder___closed__13));
v___x_2578_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__61));
lean_inc_n(v___x_2574_, 3);
v___x_2579_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2579_, 0, v___x_2574_);
lean_ctor_set(v___x_2579_, 1, v___x_2578_);
v___x_2580_ = l_Lean_Syntax_node2(v___x_2574_, v___x_2577_, v___x_2579_, v_a_2572_);
v___x_2581_ = l_Lean_Syntax_node1(v___x_2574_, v___x_2576_, v___x_2580_);
v___x_2582_ = l_Lean_Syntax_node2(v___x_2574_, v___x_2575_, v___y_2536_, v___x_2581_);
v___y_2519_ = v___y_2538_;
v___y_2520_ = v___y_2539_;
v___y_2521_ = v_a_2541_;
v_binder_2522_ = v___x_2582_;
v_ref_2523_ = v_ref_2573_;
goto v___jp_2518_;
}
else
{
lean_dec(v_a_2541_);
lean_dec(v___y_2538_);
lean_dec(v___y_2536_);
return v___x_2571_;
}
}
}
case 2:
{
lean_object* v_n_2583_; lean_object* v_ref_2584_; lean_object* v___x_2585_; lean_object* v___x_2586_; lean_object* v___x_2587_; lean_object* v___x_2588_; lean_object* v___x_2589_; lean_object* v___x_2590_; lean_object* v___x_2591_; lean_object* v___x_2592_; lean_object* v___x_2593_; lean_object* v___x_2594_; lean_object* v___x_2595_; lean_object* v___x_2596_; lean_object* v___x_2597_; lean_object* v___x_2598_; lean_object* v___x_2599_; 
v_n_2583_ = lean_ctor_get(v___y_2540_, 0);
lean_inc(v_n_2583_);
lean_dec_ref_known(v___y_2540_, 1);
v_ref_2584_ = lean_ctor_get(v___y_2425_, 5);
v___x_2585_ = l_Lean_SourceInfo_fromRef(v_ref_2584_, v___y_2539_);
v___x_2586_ = ((lean_object*)(lp_mathlib_BigOperators_bigprod___closed__1));
v___x_2587_ = ((lean_object*)(lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__0));
lean_inc_n(v___x_2585_, 6);
v___x_2588_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2588_, 0, v___x_2585_);
lean_ctor_set(v___x_2588_, 1, v___x_2587_);
v___x_2589_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBinders___closed__1));
v___x_2590_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBinder___closed__2));
v___x_2591_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__25));
v___x_2592_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__8));
v___x_2593_ = ((lean_object*)(lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__3));
v___x_2594_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2594_, 0, v___x_2585_);
lean_ctor_set(v___x_2594_, 1, v___x_2593_);
v___x_2595_ = l_Lean_Syntax_node2(v___x_2585_, v___x_2592_, v___x_2594_, v_n_2583_);
v___x_2596_ = l_Lean_Syntax_node1(v___x_2585_, v___x_2591_, v___x_2595_);
v___x_2597_ = l_Lean_Syntax_node2(v___x_2585_, v___x_2590_, v___y_2536_, v___x_2596_);
v___x_2598_ = l_Lean_Syntax_node1(v___x_2585_, v___x_2589_, v___x_2597_);
v___x_2599_ = lean_obj_once(&lp_mathlib_BigOperators_bigOpBindersPattern___closed__0, &lp_mathlib_BigOperators_bigOpBindersPattern___closed__0_once, _init_lp_mathlib_BigOperators_bigOpBindersPattern___closed__0);
if (lean_obj_tag(v_a_2541_) == 1)
{
lean_object* v_val_2600_; lean_object* v___x_2601_; 
v_val_2600_ = lean_ctor_get(v_a_2541_, 0);
lean_inc(v_val_2600_);
lean_dec_ref_known(v_a_2541_, 1);
v___x_2601_ = l_Array_mkArray1___redArg(v_val_2600_);
v___y_2474_ = v___x_2598_;
v___y_2475_ = v___x_2599_;
v___y_2476_ = v___y_2538_;
v___y_2477_ = v___x_2588_;
v___y_2478_ = v___x_2585_;
v___y_2479_ = v___x_2591_;
v___y_2480_ = v___x_2586_;
v___y_2481_ = v___x_2601_;
goto v___jp_2473_;
}
else
{
lean_object* v___x_2602_; 
lean_dec(v_a_2541_);
v___x_2602_ = ((lean_object*)(lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__1));
v___y_2474_ = v___x_2598_;
v___y_2475_ = v___x_2599_;
v___y_2476_ = v___y_2538_;
v___y_2477_ = v___x_2588_;
v___y_2478_ = v___x_2585_;
v___y_2479_ = v___x_2591_;
v___y_2480_ = v___x_2586_;
v___y_2481_ = v___x_2602_;
goto v___jp_2473_;
}
}
case 3:
{
lean_object* v_n_2603_; lean_object* v_ref_2604_; lean_object* v___x_2605_; lean_object* v___x_2606_; lean_object* v___x_2607_; lean_object* v___x_2608_; lean_object* v___x_2609_; lean_object* v___x_2610_; lean_object* v___x_2611_; lean_object* v___x_2612_; lean_object* v___x_2613_; lean_object* v___x_2614_; lean_object* v___x_2615_; lean_object* v___x_2616_; lean_object* v___x_2617_; lean_object* v___x_2618_; lean_object* v___x_2619_; 
v_n_2603_ = lean_ctor_get(v___y_2540_, 0);
lean_inc(v_n_2603_);
lean_dec_ref_known(v___y_2540_, 1);
v_ref_2604_ = lean_ctor_get(v___y_2425_, 5);
v___x_2605_ = l_Lean_SourceInfo_fromRef(v_ref_2604_, v___y_2539_);
v___x_2606_ = ((lean_object*)(lp_mathlib_BigOperators_bigprod___closed__1));
v___x_2607_ = ((lean_object*)(lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__0));
lean_inc_n(v___x_2605_, 6);
v___x_2608_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2608_, 0, v___x_2605_);
lean_ctor_set(v___x_2608_, 1, v___x_2607_);
v___x_2609_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBinders___closed__1));
v___x_2610_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBinder___closed__2));
v___x_2611_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__25));
v___x_2612_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__10));
v___x_2613_ = ((lean_object*)(lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__4));
v___x_2614_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2614_, 0, v___x_2605_);
lean_ctor_set(v___x_2614_, 1, v___x_2613_);
v___x_2615_ = l_Lean_Syntax_node2(v___x_2605_, v___x_2612_, v___x_2614_, v_n_2603_);
v___x_2616_ = l_Lean_Syntax_node1(v___x_2605_, v___x_2611_, v___x_2615_);
v___x_2617_ = l_Lean_Syntax_node2(v___x_2605_, v___x_2610_, v___y_2536_, v___x_2616_);
v___x_2618_ = l_Lean_Syntax_node1(v___x_2605_, v___x_2609_, v___x_2617_);
v___x_2619_ = lean_obj_once(&lp_mathlib_BigOperators_bigOpBindersPattern___closed__0, &lp_mathlib_BigOperators_bigOpBindersPattern___closed__0_once, _init_lp_mathlib_BigOperators_bigOpBindersPattern___closed__0);
if (lean_obj_tag(v_a_2541_) == 1)
{
lean_object* v_val_2620_; lean_object* v___x_2621_; 
v_val_2620_ = lean_ctor_get(v_a_2541_, 0);
lean_inc(v_val_2620_);
lean_dec_ref_known(v_a_2541_, 1);
v___x_2621_ = l_Array_mkArray1___redArg(v_val_2620_);
v___y_2459_ = v___x_2606_;
v___y_2460_ = v___x_2619_;
v___y_2461_ = v___y_2538_;
v___y_2462_ = v___x_2605_;
v___y_2463_ = v___x_2611_;
v___y_2464_ = v___x_2618_;
v___y_2465_ = v___x_2608_;
v___y_2466_ = v___x_2621_;
goto v___jp_2458_;
}
else
{
lean_object* v___x_2622_; 
lean_dec(v_a_2541_);
v___x_2622_ = ((lean_object*)(lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__1));
v___y_2459_ = v___x_2606_;
v___y_2460_ = v___x_2619_;
v___y_2461_ = v___y_2538_;
v___y_2462_ = v___x_2605_;
v___y_2463_ = v___x_2611_;
v___y_2464_ = v___x_2618_;
v___y_2465_ = v___x_2608_;
v___y_2466_ = v___x_2622_;
goto v___jp_2458_;
}
}
case 4:
{
lean_object* v_n_2623_; lean_object* v_ref_2624_; lean_object* v___x_2625_; lean_object* v___x_2626_; lean_object* v___x_2627_; lean_object* v___x_2628_; lean_object* v___x_2629_; lean_object* v___x_2630_; lean_object* v___x_2631_; lean_object* v___x_2632_; lean_object* v___x_2633_; lean_object* v___x_2634_; lean_object* v___x_2635_; lean_object* v___x_2636_; lean_object* v___x_2637_; lean_object* v___x_2638_; lean_object* v___x_2639_; 
v_n_2623_ = lean_ctor_get(v___y_2540_, 0);
lean_inc(v_n_2623_);
lean_dec_ref_known(v___y_2540_, 1);
v_ref_2624_ = lean_ctor_get(v___y_2425_, 5);
v___x_2625_ = l_Lean_SourceInfo_fromRef(v_ref_2624_, v___y_2539_);
v___x_2626_ = ((lean_object*)(lp_mathlib_BigOperators_bigprod___closed__1));
v___x_2627_ = ((lean_object*)(lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__0));
lean_inc_n(v___x_2625_, 6);
v___x_2628_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2628_, 0, v___x_2625_);
lean_ctor_set(v___x_2628_, 1, v___x_2627_);
v___x_2629_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBinders___closed__1));
v___x_2630_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBinder___closed__2));
v___x_2631_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__25));
v___x_2632_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__12));
v___x_2633_ = ((lean_object*)(lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__5));
v___x_2634_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2634_, 0, v___x_2625_);
lean_ctor_set(v___x_2634_, 1, v___x_2633_);
v___x_2635_ = l_Lean_Syntax_node2(v___x_2625_, v___x_2632_, v___x_2634_, v_n_2623_);
v___x_2636_ = l_Lean_Syntax_node1(v___x_2625_, v___x_2631_, v___x_2635_);
v___x_2637_ = l_Lean_Syntax_node2(v___x_2625_, v___x_2630_, v___y_2536_, v___x_2636_);
v___x_2638_ = l_Lean_Syntax_node1(v___x_2625_, v___x_2629_, v___x_2637_);
v___x_2639_ = lean_obj_once(&lp_mathlib_BigOperators_bigOpBindersPattern___closed__0, &lp_mathlib_BigOperators_bigOpBindersPattern___closed__0_once, _init_lp_mathlib_BigOperators_bigOpBindersPattern___closed__0);
if (lean_obj_tag(v_a_2541_) == 1)
{
lean_object* v_val_2640_; lean_object* v___x_2641_; 
v_val_2640_ = lean_ctor_get(v_a_2541_, 0);
lean_inc(v_val_2640_);
lean_dec_ref_known(v_a_2541_, 1);
v___x_2641_ = l_Array_mkArray1___redArg(v_val_2640_);
v___y_2444_ = v___x_2631_;
v___y_2445_ = v___y_2538_;
v___y_2446_ = v___x_2628_;
v___y_2447_ = v___x_2638_;
v___y_2448_ = v___x_2625_;
v___y_2449_ = v___x_2639_;
v___y_2450_ = v___x_2626_;
v___y_2451_ = v___x_2641_;
goto v___jp_2443_;
}
else
{
lean_object* v___x_2642_; 
lean_dec(v_a_2541_);
v___x_2642_ = ((lean_object*)(lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__1));
v___y_2444_ = v___x_2631_;
v___y_2445_ = v___y_2538_;
v___y_2446_ = v___x_2628_;
v___y_2447_ = v___x_2638_;
v___y_2448_ = v___x_2625_;
v___y_2449_ = v___x_2639_;
v___y_2450_ = v___x_2626_;
v___y_2451_ = v___x_2642_;
goto v___jp_2443_;
}
}
default: 
{
lean_object* v_n_2643_; lean_object* v_ref_2644_; lean_object* v___x_2645_; lean_object* v___x_2646_; lean_object* v___x_2647_; lean_object* v___x_2648_; lean_object* v___x_2649_; lean_object* v___x_2650_; lean_object* v___x_2651_; lean_object* v___x_2652_; lean_object* v___x_2653_; lean_object* v___x_2654_; lean_object* v___x_2655_; lean_object* v___x_2656_; lean_object* v___x_2657_; lean_object* v___x_2658_; lean_object* v___x_2659_; 
v_n_2643_ = lean_ctor_get(v___y_2540_, 0);
lean_inc(v_n_2643_);
lean_dec_ref_known(v___y_2540_, 1);
v_ref_2644_ = lean_ctor_get(v___y_2425_, 5);
v___x_2645_ = l_Lean_SourceInfo_fromRef(v_ref_2644_, v___y_2539_);
v___x_2646_ = ((lean_object*)(lp_mathlib_BigOperators_bigprod___closed__1));
v___x_2647_ = ((lean_object*)(lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__0));
lean_inc_n(v___x_2645_, 6);
v___x_2648_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2648_, 0, v___x_2645_);
lean_ctor_set(v___x_2648_, 1, v___x_2647_);
v___x_2649_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBinders___closed__1));
v___x_2650_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBinder___closed__2));
v___x_2651_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__25));
v___x_2652_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__14));
v___x_2653_ = ((lean_object*)(lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__6));
v___x_2654_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2654_, 0, v___x_2645_);
lean_ctor_set(v___x_2654_, 1, v___x_2653_);
v___x_2655_ = l_Lean_Syntax_node2(v___x_2645_, v___x_2652_, v___x_2654_, v_n_2643_);
v___x_2656_ = l_Lean_Syntax_node1(v___x_2645_, v___x_2651_, v___x_2655_);
v___x_2657_ = l_Lean_Syntax_node2(v___x_2645_, v___x_2650_, v___y_2536_, v___x_2656_);
v___x_2658_ = l_Lean_Syntax_node1(v___x_2645_, v___x_2649_, v___x_2657_);
v___x_2659_ = lean_obj_once(&lp_mathlib_BigOperators_bigOpBindersPattern___closed__0, &lp_mathlib_BigOperators_bigOpBindersPattern___closed__0_once, _init_lp_mathlib_BigOperators_bigOpBindersPattern___closed__0);
if (lean_obj_tag(v_a_2541_) == 1)
{
lean_object* v_val_2660_; lean_object* v___x_2661_; 
v_val_2660_ = lean_ctor_get(v_a_2541_, 0);
lean_inc(v_val_2660_);
lean_dec_ref_known(v_a_2541_, 1);
v___x_2661_ = l_Array_mkArray1___redArg(v_val_2660_);
v___y_2429_ = v___y_2538_;
v___y_2430_ = v___x_2651_;
v___y_2431_ = v___x_2659_;
v___y_2432_ = v___x_2658_;
v___y_2433_ = v___x_2648_;
v___y_2434_ = v___x_2645_;
v___y_2435_ = v___x_2646_;
v___y_2436_ = v___x_2661_;
goto v___jp_2428_;
}
else
{
lean_object* v___x_2662_; 
lean_dec(v_a_2541_);
v___x_2662_ = ((lean_object*)(lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__1));
v___y_2429_ = v___y_2538_;
v___y_2430_ = v___x_2651_;
v___y_2431_ = v___x_2659_;
v___y_2432_ = v___x_2658_;
v___y_2433_ = v___x_2648_;
v___y_2434_ = v___x_2645_;
v___y_2435_ = v___x_2646_;
v___y_2436_ = v___x_2662_;
goto v___jp_2428_;
}
}
}
}
v___jp_2663_:
{
lean_object* v___x_2664_; lean_object* v___x_2665_; 
v___x_2664_ = ((lean_object*)(lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__8));
v___x_2665_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00BigOperators_delabFinsetProd_spec__0___redArg(v___x_2664_, v___y_2421_, v___y_2422_, v___y_2423_, v___y_2424_, v___y_2425_, v___y_2426_);
if (lean_obj_tag(v___x_2665_) == 0)
{
lean_object* v_a_2666_; uint8_t v___x_2667_; lean_object* v___x_2668_; lean_object* v___x_2669_; lean_object* v___x_2670_; lean_object* v___x_2671_; 
v_a_2666_ = lean_ctor_get(v___x_2665_, 0);
lean_inc(v_a_2666_);
lean_dec_ref_known(v___x_2665_, 1);
v___x_2667_ = 0;
v___x_2668_ = l_Lean_NameSet_empty;
v___x_2669_ = lean_box(v___x_2667_);
v___x_2670_ = lean_alloc_closure((void*)(l_Lean_PrettyPrinter_Delaborator_withBindingBodyUnusedName___boxed), 11, 4);
lean_closure_set(v___x_2670_, 0, lean_box(0));
lean_closure_set(v___x_2670_, 1, v___f_2420_);
lean_closure_set(v___x_2670_, 2, v___x_2669_);
lean_closure_set(v___x_2670_, 3, v___x_2668_);
v___x_2671_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00BigOperators_delabFinsetProd_spec__0___redArg(v___x_2670_, v___y_2421_, v___y_2422_, v___y_2423_, v___y_2424_, v___y_2425_, v___y_2426_);
if (lean_obj_tag(v___x_2671_) == 0)
{
lean_object* v_a_2672_; lean_object* v_fst_2673_; lean_object* v_snd_2674_; lean_object* v___x_2675_; lean_object* v___x_2676_; lean_object* v___x_2677_; 
v_a_2672_ = lean_ctor_get(v___x_2671_, 0);
lean_inc(v_a_2672_);
lean_dec_ref_known(v___x_2671_, 1);
v_fst_2673_ = lean_ctor_get(v_a_2672_, 0);
lean_inc_n(v_fst_2673_, 2);
v_snd_2674_ = lean_ctor_get(v_a_2672_, 1);
lean_inc(v_snd_2674_);
lean_dec(v_a_2672_);
v___x_2675_ = lean_unsigned_to_nat(3u);
v___x_2676_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg___boxed), 8, 1);
lean_closure_set(v___x_2676_, 0, v_fst_2673_);
v___x_2677_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__1___redArg(v___x_2675_, v___x_2676_, v___y_2421_, v___y_2422_, v___y_2423_, v___y_2424_, v___y_2425_, v___y_2426_);
if (lean_obj_tag(v___x_2677_) == 0)
{
lean_object* v_a_2678_; lean_object* v_filter_2679_; 
v_a_2678_ = lean_ctor_get(v___x_2677_, 0);
lean_inc(v_a_2678_);
lean_dec_ref_known(v___x_2677_, 1);
v_filter_2679_ = lean_ctor_get(v_a_2678_, 1);
lean_inc(v_filter_2679_);
if (lean_obj_tag(v_filter_2679_) == 0)
{
lean_object* v_finset_2680_; uint8_t v___x_2681_; 
v_finset_2680_ = lean_ctor_get(v_a_2678_, 0);
lean_inc(v_finset_2680_);
lean_dec(v_a_2678_);
v___x_2681_ = lean_unbox(v_a_2666_);
lean_dec(v_a_2666_);
v___y_2536_ = v_fst_2673_;
v___y_2537_ = v___x_2681_;
v___y_2538_ = v_snd_2674_;
v___y_2539_ = v___x_2667_;
v___y_2540_ = v_finset_2680_;
v_a_2541_ = v_filter_2679_;
goto v___jp_2535_;
}
else
{
lean_object* v_finset_2682_; lean_object* v___x_2684_; uint8_t v_isShared_2685_; uint8_t v_isSharedCheck_2706_; 
v_finset_2682_ = lean_ctor_get(v_a_2678_, 0);
v_isSharedCheck_2706_ = !lean_is_exclusive(v_a_2678_);
if (v_isSharedCheck_2706_ == 0)
{
lean_object* v_unused_2707_; 
v_unused_2707_ = lean_ctor_get(v_a_2678_, 1);
lean_dec(v_unused_2707_);
v___x_2684_ = v_a_2678_;
v_isShared_2685_ = v_isSharedCheck_2706_;
goto v_resetjp_2683_;
}
else
{
lean_inc(v_finset_2682_);
lean_dec(v_a_2678_);
v___x_2684_ = lean_box(0);
v_isShared_2685_ = v_isSharedCheck_2706_;
goto v_resetjp_2683_;
}
v_resetjp_2683_:
{
lean_object* v_val_2686_; lean_object* v___x_2688_; uint8_t v_isShared_2689_; uint8_t v_isSharedCheck_2705_; 
v_val_2686_ = lean_ctor_get(v_filter_2679_, 0);
v_isSharedCheck_2705_ = !lean_is_exclusive(v_filter_2679_);
if (v_isSharedCheck_2705_ == 0)
{
v___x_2688_ = v_filter_2679_;
v_isShared_2689_ = v_isSharedCheck_2705_;
goto v_resetjp_2687_;
}
else
{
lean_inc(v_val_2686_);
lean_dec(v_filter_2679_);
v___x_2688_ = lean_box(0);
v_isShared_2689_ = v_isSharedCheck_2705_;
goto v_resetjp_2687_;
}
v_resetjp_2687_:
{
lean_object* v_ref_2690_; lean_object* v___x_2691_; lean_object* v___x_2692_; lean_object* v___x_2693_; lean_object* v___x_2695_; 
v_ref_2690_ = lean_ctor_get(v___y_2425_, 5);
v___x_2691_ = l_Lean_SourceInfo_fromRef(v_ref_2690_, v___x_2667_);
v___x_2692_ = ((lean_object*)(lp_mathlib_BigOperators_BigOpWith___closed__1));
v___x_2693_ = ((lean_object*)(lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__9));
lean_inc(v___x_2691_);
if (v_isShared_2685_ == 0)
{
lean_ctor_set_tag(v___x_2684_, 2);
lean_ctor_set(v___x_2684_, 1, v___x_2693_);
lean_ctor_set(v___x_2684_, 0, v___x_2691_);
v___x_2695_ = v___x_2684_;
goto v_reusejp_2694_;
}
else
{
lean_object* v_reuseFailAlloc_2704_; 
v_reuseFailAlloc_2704_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2704_, 0, v___x_2691_);
lean_ctor_set(v_reuseFailAlloc_2704_, 1, v___x_2693_);
v___x_2695_ = v_reuseFailAlloc_2704_;
goto v_reusejp_2694_;
}
v_reusejp_2694_:
{
lean_object* v___x_2696_; lean_object* v___x_2697_; lean_object* v___x_2698_; lean_object* v___x_2699_; lean_object* v___x_2701_; 
v___x_2696_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__25));
v___x_2697_ = lean_obj_once(&lp_mathlib_BigOperators_bigOpBindersPattern___closed__0, &lp_mathlib_BigOperators_bigOpBindersPattern___closed__0_once, _init_lp_mathlib_BigOperators_bigOpBindersPattern___closed__0);
lean_inc(v___x_2691_);
v___x_2698_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2698_, 0, v___x_2691_);
lean_ctor_set(v___x_2698_, 1, v___x_2696_);
lean_ctor_set(v___x_2698_, 2, v___x_2697_);
v___x_2699_ = l_Lean_Syntax_node3(v___x_2691_, v___x_2692_, v___x_2695_, v___x_2698_, v_val_2686_);
if (v_isShared_2689_ == 0)
{
lean_ctor_set(v___x_2688_, 0, v___x_2699_);
v___x_2701_ = v___x_2688_;
goto v_reusejp_2700_;
}
else
{
lean_object* v_reuseFailAlloc_2703_; 
v_reuseFailAlloc_2703_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2703_, 0, v___x_2699_);
v___x_2701_ = v_reuseFailAlloc_2703_;
goto v_reusejp_2700_;
}
v_reusejp_2700_:
{
uint8_t v___x_2702_; 
v___x_2702_ = lean_unbox(v_a_2666_);
lean_dec(v_a_2666_);
v___y_2536_ = v_fst_2673_;
v___y_2537_ = v___x_2702_;
v___y_2538_ = v_snd_2674_;
v___y_2539_ = v___x_2667_;
v___y_2540_ = v_finset_2682_;
v_a_2541_ = v___x_2701_;
goto v___jp_2535_;
}
}
}
}
}
}
else
{
lean_object* v_a_2708_; lean_object* v___x_2710_; uint8_t v_isShared_2711_; uint8_t v_isSharedCheck_2715_; 
lean_dec(v_snd_2674_);
lean_dec(v_fst_2673_);
lean_dec(v_a_2666_);
v_a_2708_ = lean_ctor_get(v___x_2677_, 0);
v_isSharedCheck_2715_ = !lean_is_exclusive(v___x_2677_);
if (v_isSharedCheck_2715_ == 0)
{
v___x_2710_ = v___x_2677_;
v_isShared_2711_ = v_isSharedCheck_2715_;
goto v_resetjp_2709_;
}
else
{
lean_inc(v_a_2708_);
lean_dec(v___x_2677_);
v___x_2710_ = lean_box(0);
v_isShared_2711_ = v_isSharedCheck_2715_;
goto v_resetjp_2709_;
}
v_resetjp_2709_:
{
lean_object* v___x_2713_; 
if (v_isShared_2711_ == 0)
{
v___x_2713_ = v___x_2710_;
goto v_reusejp_2712_;
}
else
{
lean_object* v_reuseFailAlloc_2714_; 
v_reuseFailAlloc_2714_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2714_, 0, v_a_2708_);
v___x_2713_ = v_reuseFailAlloc_2714_;
goto v_reusejp_2712_;
}
v_reusejp_2712_:
{
return v___x_2713_;
}
}
}
}
else
{
lean_object* v_a_2716_; lean_object* v___x_2718_; uint8_t v_isShared_2719_; uint8_t v_isSharedCheck_2723_; 
lean_dec(v_a_2666_);
v_a_2716_ = lean_ctor_get(v___x_2671_, 0);
v_isSharedCheck_2723_ = !lean_is_exclusive(v___x_2671_);
if (v_isSharedCheck_2723_ == 0)
{
v___x_2718_ = v___x_2671_;
v_isShared_2719_ = v_isSharedCheck_2723_;
goto v_resetjp_2717_;
}
else
{
lean_inc(v_a_2716_);
lean_dec(v___x_2671_);
v___x_2718_ = lean_box(0);
v_isShared_2719_ = v_isSharedCheck_2723_;
goto v_resetjp_2717_;
}
v_resetjp_2717_:
{
lean_object* v___x_2721_; 
if (v_isShared_2719_ == 0)
{
v___x_2721_ = v___x_2718_;
goto v_reusejp_2720_;
}
else
{
lean_object* v_reuseFailAlloc_2722_; 
v_reuseFailAlloc_2722_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2722_, 0, v_a_2716_);
v___x_2721_ = v_reuseFailAlloc_2722_;
goto v_reusejp_2720_;
}
v_reusejp_2720_:
{
return v___x_2721_;
}
}
}
}
else
{
lean_object* v_a_2724_; lean_object* v___x_2726_; uint8_t v_isShared_2727_; uint8_t v_isSharedCheck_2731_; 
lean_dec_ref(v___f_2420_);
v_a_2724_ = lean_ctor_get(v___x_2665_, 0);
v_isSharedCheck_2731_ = !lean_is_exclusive(v___x_2665_);
if (v_isSharedCheck_2731_ == 0)
{
v___x_2726_ = v___x_2665_;
v_isShared_2727_ = v_isSharedCheck_2731_;
goto v_resetjp_2725_;
}
else
{
lean_inc(v_a_2724_);
lean_dec(v___x_2665_);
v___x_2726_ = lean_box(0);
v_isShared_2727_ = v_isSharedCheck_2731_;
goto v_resetjp_2725_;
}
v_resetjp_2725_:
{
lean_object* v___x_2729_; 
if (v_isShared_2727_ == 0)
{
v___x_2729_ = v___x_2726_;
goto v_reusejp_2728_;
}
else
{
lean_object* v_reuseFailAlloc_2730_; 
v_reuseFailAlloc_2730_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2730_, 0, v_a_2724_);
v___x_2729_ = v_reuseFailAlloc_2730_;
goto v_reusejp_2728_;
}
v_reusejp_2728_:
{
return v___x_2729_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_BigOperators_delabFinsetProd___lam__1___boxed(lean_object* v___x_2755_, lean_object* v___f_2756_, lean_object* v___y_2757_, lean_object* v___y_2758_, lean_object* v___y_2759_, lean_object* v___y_2760_, lean_object* v___y_2761_, lean_object* v___y_2762_, lean_object* v___y_2763_){
_start:
{
lean_object* v_res_2764_; 
v_res_2764_ = lp_mathlib_BigOperators_delabFinsetProd___lam__1(v___x_2755_, v___f_2756_, v___y_2757_, v___y_2758_, v___y_2759_, v___y_2760_, v___y_2761_, v___y_2762_);
lean_dec(v___y_2762_);
lean_dec_ref(v___y_2761_);
lean_dec(v___y_2760_);
lean_dec_ref(v___y_2759_);
lean_dec(v___y_2758_);
lean_dec_ref(v___y_2757_);
lean_dec(v___x_2755_);
return v_res_2764_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BigOperators_delabFinsetProd(lean_object* v_a_2773_, lean_object* v_a_2774_, lean_object* v_a_2775_, lean_object* v_a_2776_, lean_object* v_a_2777_, lean_object* v_a_2778_){
_start:
{
lean_object* v___x_2780_; lean_object* v___x_2781_; lean_object* v___x_2782_; 
v___x_2780_ = ((lean_object*)(lp_mathlib_BigOperators_delabFinsetProd___closed__1));
v___x_2781_ = ((lean_object*)(lp_mathlib_BigOperators_delabFinsetProd___closed__3));
v___x_2782_ = l_Lean_PrettyPrinter_Delaborator_whenPPOption(v___x_2780_, v___x_2781_, v_a_2773_, v_a_2774_, v_a_2775_, v_a_2776_, v_a_2777_, v_a_2778_);
return v___x_2782_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BigOperators_delabFinsetProd___boxed(lean_object* v_a_2783_, lean_object* v_a_2784_, lean_object* v_a_2785_, lean_object* v_a_2786_, lean_object* v_a_2787_, lean_object* v_a_2788_, lean_object* v_a_2789_){
_start:
{
lean_object* v_res_2790_; 
v_res_2790_ = lp_mathlib_BigOperators_delabFinsetProd(v_a_2783_, v_a_2784_, v_a_2785_, v_a_2786_, v_a_2787_, v_a_2788_);
lean_dec(v_a_2788_);
lean_dec_ref(v_a_2787_);
lean_dec(v_a_2786_);
lean_dec_ref(v_a_2785_);
lean_dec(v_a_2784_);
lean_dec_ref(v_a_2783_);
return v_res_2790_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00BigOperators_delabFinsetProd_spec__0(lean_object* v_00_u03b1_2791_, lean_object* v_x_2792_, lean_object* v___y_2793_, lean_object* v___y_2794_, lean_object* v___y_2795_, lean_object* v___y_2796_, lean_object* v___y_2797_, lean_object* v___y_2798_){
_start:
{
lean_object* v___x_2800_; 
v___x_2800_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00BigOperators_delabFinsetProd_spec__0___redArg(v_x_2792_, v___y_2793_, v___y_2794_, v___y_2795_, v___y_2796_, v___y_2797_, v___y_2798_);
return v___x_2800_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00BigOperators_delabFinsetProd_spec__0___boxed(lean_object* v_00_u03b1_2801_, lean_object* v_x_2802_, lean_object* v___y_2803_, lean_object* v___y_2804_, lean_object* v___y_2805_, lean_object* v___y_2806_, lean_object* v___y_2807_, lean_object* v___y_2808_, lean_object* v___y_2809_){
_start:
{
lean_object* v_res_2810_; 
v_res_2810_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00BigOperators_delabFinsetProd_spec__0(v_00_u03b1_2801_, v_x_2802_, v___y_2803_, v___y_2804_, v___y_2805_, v___y_2806_, v___y_2807_, v___y_2808_);
lean_dec(v___y_2808_);
lean_dec_ref(v___y_2807_);
lean_dec(v___y_2806_);
lean_dec_ref(v___y_2805_);
lean_dec(v___y_2804_);
lean_dec_ref(v___y_2803_);
return v_res_2810_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BigOperators_delabFinsetSum___lam__1(lean_object* v___x_2812_, lean_object* v___f_2813_, lean_object* v___y_2814_, lean_object* v___y_2815_, lean_object* v___y_2816_, lean_object* v___y_2817_, lean_object* v___y_2818_, lean_object* v___y_2819_){
_start:
{
lean_object* v___y_2822_; lean_object* v___y_2823_; lean_object* v___y_2824_; lean_object* v___y_2825_; lean_object* v___y_2826_; lean_object* v___y_2827_; lean_object* v___y_2828_; lean_object* v___y_2829_; lean_object* v___y_2837_; lean_object* v___y_2838_; lean_object* v___y_2839_; lean_object* v___y_2840_; lean_object* v___y_2841_; lean_object* v___y_2842_; lean_object* v___y_2843_; lean_object* v___y_2844_; lean_object* v___y_2852_; lean_object* v___y_2853_; lean_object* v___y_2854_; lean_object* v___y_2855_; lean_object* v___y_2856_; lean_object* v___y_2857_; lean_object* v___y_2858_; lean_object* v___y_2859_; lean_object* v___y_2867_; lean_object* v___y_2868_; lean_object* v___y_2869_; lean_object* v___y_2870_; lean_object* v___y_2871_; lean_object* v___y_2872_; lean_object* v___y_2873_; lean_object* v___y_2874_; lean_object* v___y_2882_; lean_object* v___y_2883_; lean_object* v___y_2884_; lean_object* v___y_2885_; lean_object* v___y_2886_; lean_object* v___y_2887_; lean_object* v___y_2888_; lean_object* v___y_2889_; lean_object* v___y_2897_; lean_object* v___y_2898_; lean_object* v___y_2899_; lean_object* v___y_2900_; lean_object* v___y_2901_; lean_object* v___y_2902_; lean_object* v___y_2903_; lean_object* v___y_2904_; lean_object* v___y_2912_; uint8_t v___y_2913_; lean_object* v___y_2914_; lean_object* v_binder_2915_; lean_object* v_ref_2916_; lean_object* v___y_2929_; uint8_t v___y_2930_; uint8_t v___y_2931_; lean_object* v___y_2932_; lean_object* v___y_2933_; lean_object* v_a_2934_; lean_object* v___x_3125_; lean_object* v_a_3126_; lean_object* v_dummy_3127_; lean_object* v_nargs_3128_; lean_object* v___x_3129_; lean_object* v___x_3130_; lean_object* v___x_3131_; lean_object* v___x_3132_; lean_object* v___x_3133_; uint8_t v___x_3134_; 
v___x_3125_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__0___redArg(v___y_2814_);
v_a_3126_ = lean_ctor_get(v___x_3125_, 0);
lean_inc(v_a_3126_);
lean_dec_ref(v___x_3125_);
v_dummy_3127_ = lean_obj_once(&lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__1___redArg___closed__0, &lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__1___redArg___closed__0_once, _init_lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__1___redArg___closed__0);
v_nargs_3128_ = l_Lean_Expr_getAppNumArgs(v_a_3126_);
lean_inc(v_nargs_3128_);
v___x_3129_ = lean_mk_array(v_nargs_3128_, v_dummy_3127_);
v___x_3130_ = lean_unsigned_to_nat(1u);
v___x_3131_ = lean_nat_sub(v_nargs_3128_, v___x_3130_);
lean_dec(v_nargs_3128_);
v___x_3132_ = l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(v_a_3126_, v___x_3129_, v___x_3131_);
v___x_3133_ = lean_array_get_size(v___x_3132_);
v___x_3134_ = lean_nat_dec_eq(v___x_3133_, v___x_2812_);
if (v___x_3134_ == 0)
{
lean_object* v___x_3135_; 
lean_dec_ref(v___x_3132_);
lean_dec_ref(v___f_2813_);
v___x_3135_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_3135_;
}
else
{
lean_object* v___x_3136_; lean_object* v___x_3137_; uint8_t v___x_3138_; 
v___x_3136_ = lean_unsigned_to_nat(4u);
v___x_3137_ = lean_array_fget(v___x_3132_, v___x_3136_);
lean_dec_ref(v___x_3132_);
v___x_3138_ = l_Lean_Expr_isLambda(v___x_3137_);
lean_dec(v___x_3137_);
if (v___x_3138_ == 0)
{
lean_object* v___x_3139_; 
v___x_3139_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
if (lean_obj_tag(v___x_3139_) == 0)
{
lean_dec_ref_known(v___x_3139_, 1);
goto v___jp_3056_;
}
else
{
lean_object* v_a_3140_; lean_object* v___x_3142_; uint8_t v_isShared_3143_; uint8_t v_isSharedCheck_3147_; 
lean_dec_ref(v___f_2813_);
v_a_3140_ = lean_ctor_get(v___x_3139_, 0);
v_isSharedCheck_3147_ = !lean_is_exclusive(v___x_3139_);
if (v_isSharedCheck_3147_ == 0)
{
v___x_3142_ = v___x_3139_;
v_isShared_3143_ = v_isSharedCheck_3147_;
goto v_resetjp_3141_;
}
else
{
lean_inc(v_a_3140_);
lean_dec(v___x_3139_);
v___x_3142_ = lean_box(0);
v_isShared_3143_ = v_isSharedCheck_3147_;
goto v_resetjp_3141_;
}
v_resetjp_3141_:
{
lean_object* v___x_3145_; 
if (v_isShared_3143_ == 0)
{
v___x_3145_ = v___x_3142_;
goto v_reusejp_3144_;
}
else
{
lean_object* v_reuseFailAlloc_3146_; 
v_reuseFailAlloc_3146_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3146_, 0, v_a_3140_);
v___x_3145_ = v_reuseFailAlloc_3146_;
goto v_reusejp_3144_;
}
v_reusejp_3144_:
{
return v___x_3145_;
}
}
}
}
else
{
goto v___jp_3056_;
}
}
v___jp_2821_:
{
lean_object* v___x_2830_; lean_object* v___x_2831_; lean_object* v___x_2832_; lean_object* v___x_2833_; lean_object* v___x_2834_; lean_object* v___x_2835_; 
lean_inc_ref(v___y_2826_);
v___x_2830_ = l_Array_append___redArg(v___y_2826_, v___y_2829_);
lean_dec_ref(v___y_2829_);
lean_inc_n(v___y_2824_, 2);
v___x_2831_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2831_, 0, v___y_2824_);
lean_ctor_set(v___x_2831_, 1, v___y_2823_);
lean_ctor_set(v___x_2831_, 2, v___x_2830_);
v___x_2832_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__73));
v___x_2833_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2833_, 0, v___y_2824_);
lean_ctor_set(v___x_2833_, 1, v___x_2832_);
lean_inc(v___y_2825_);
v___x_2834_ = l_Lean_Syntax_node5(v___y_2824_, v___y_2825_, v___y_2822_, v___y_2828_, v___x_2831_, v___x_2833_, v___y_2827_);
v___x_2835_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2835_, 0, v___x_2834_);
return v___x_2835_;
}
v___jp_2836_:
{
lean_object* v___x_2845_; lean_object* v___x_2846_; lean_object* v___x_2847_; lean_object* v___x_2848_; lean_object* v___x_2849_; lean_object* v___x_2850_; 
lean_inc_ref(v___y_2843_);
v___x_2845_ = l_Array_append___redArg(v___y_2843_, v___y_2844_);
lean_dec_ref(v___y_2844_);
lean_inc_n(v___y_2840_, 2);
v___x_2846_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2846_, 0, v___y_2840_);
lean_ctor_set(v___x_2846_, 1, v___y_2837_);
lean_ctor_set(v___x_2846_, 2, v___x_2845_);
v___x_2847_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__73));
v___x_2848_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2848_, 0, v___y_2840_);
lean_ctor_set(v___x_2848_, 1, v___x_2847_);
lean_inc(v___y_2841_);
v___x_2849_ = l_Lean_Syntax_node5(v___y_2840_, v___y_2841_, v___y_2839_, v___y_2838_, v___x_2846_, v___x_2848_, v___y_2842_);
v___x_2850_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2850_, 0, v___x_2849_);
return v___x_2850_;
}
v___jp_2851_:
{
lean_object* v___x_2860_; lean_object* v___x_2861_; lean_object* v___x_2862_; lean_object* v___x_2863_; lean_object* v___x_2864_; lean_object* v___x_2865_; 
lean_inc_ref(v___y_2854_);
v___x_2860_ = l_Array_append___redArg(v___y_2854_, v___y_2859_);
lean_dec_ref(v___y_2859_);
lean_inc_n(v___y_2855_, 2);
v___x_2861_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2861_, 0, v___y_2855_);
lean_ctor_set(v___x_2861_, 1, v___y_2852_);
lean_ctor_set(v___x_2861_, 2, v___x_2860_);
v___x_2862_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__73));
v___x_2863_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2863_, 0, v___y_2855_);
lean_ctor_set(v___x_2863_, 1, v___x_2862_);
lean_inc(v___y_2857_);
v___x_2864_ = l_Lean_Syntax_node5(v___y_2855_, v___y_2857_, v___y_2856_, v___y_2853_, v___x_2861_, v___x_2863_, v___y_2858_);
v___x_2865_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2865_, 0, v___x_2864_);
return v___x_2865_;
}
v___jp_2866_:
{
lean_object* v___x_2875_; lean_object* v___x_2876_; lean_object* v___x_2877_; lean_object* v___x_2878_; lean_object* v___x_2879_; lean_object* v___x_2880_; 
lean_inc_ref(v___y_2873_);
v___x_2875_ = l_Array_append___redArg(v___y_2873_, v___y_2874_);
lean_dec_ref(v___y_2874_);
lean_inc_n(v___y_2867_, 2);
v___x_2876_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2876_, 0, v___y_2867_);
lean_ctor_set(v___x_2876_, 1, v___y_2872_);
lean_ctor_set(v___x_2876_, 2, v___x_2875_);
v___x_2877_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__73));
v___x_2878_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2878_, 0, v___y_2867_);
lean_ctor_set(v___x_2878_, 1, v___x_2877_);
lean_inc(v___y_2868_);
v___x_2879_ = l_Lean_Syntax_node5(v___y_2867_, v___y_2868_, v___y_2870_, v___y_2869_, v___x_2876_, v___x_2878_, v___y_2871_);
v___x_2880_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2880_, 0, v___x_2879_);
return v___x_2880_;
}
v___jp_2881_:
{
lean_object* v___x_2890_; lean_object* v___x_2891_; lean_object* v___x_2892_; lean_object* v___x_2893_; lean_object* v___x_2894_; lean_object* v___x_2895_; 
lean_inc_ref(v___y_2885_);
v___x_2890_ = l_Array_append___redArg(v___y_2885_, v___y_2889_);
lean_dec_ref(v___y_2889_);
lean_inc_n(v___y_2883_, 2);
v___x_2891_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2891_, 0, v___y_2883_);
lean_ctor_set(v___x_2891_, 1, v___y_2886_);
lean_ctor_set(v___x_2891_, 2, v___x_2890_);
v___x_2892_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__73));
v___x_2893_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2893_, 0, v___y_2883_);
lean_ctor_set(v___x_2893_, 1, v___x_2892_);
lean_inc(v___y_2887_);
v___x_2894_ = l_Lean_Syntax_node5(v___y_2883_, v___y_2887_, v___y_2884_, v___y_2882_, v___x_2891_, v___x_2893_, v___y_2888_);
v___x_2895_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2895_, 0, v___x_2894_);
return v___x_2895_;
}
v___jp_2896_:
{
lean_object* v___x_2905_; lean_object* v___x_2906_; lean_object* v___x_2907_; lean_object* v___x_2908_; lean_object* v___x_2909_; lean_object* v___x_2910_; 
lean_inc_ref(v___y_2900_);
v___x_2905_ = l_Array_append___redArg(v___y_2900_, v___y_2904_);
lean_dec_ref(v___y_2904_);
lean_inc(v___y_2899_);
lean_inc_n(v___y_2898_, 2);
v___x_2906_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2906_, 0, v___y_2898_);
lean_ctor_set(v___x_2906_, 1, v___y_2899_);
lean_ctor_set(v___x_2906_, 2, v___x_2905_);
v___x_2907_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__73));
v___x_2908_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2908_, 0, v___y_2898_);
lean_ctor_set(v___x_2908_, 1, v___x_2907_);
lean_inc(v___y_2902_);
v___x_2909_ = l_Lean_Syntax_node5(v___y_2898_, v___y_2902_, v___y_2897_, v___y_2901_, v___x_2906_, v___x_2908_, v___y_2903_);
v___x_2910_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2910_, 0, v___x_2909_);
return v___x_2910_;
}
v___jp_2911_:
{
lean_object* v___x_2917_; lean_object* v___x_2918_; lean_object* v___x_2919_; lean_object* v___x_2920_; lean_object* v___x_2921_; lean_object* v___x_2922_; lean_object* v___x_2923_; lean_object* v___x_2924_; 
v___x_2917_ = l_Lean_SourceInfo_fromRef(v_ref_2916_, v___y_2913_);
v___x_2918_ = ((lean_object*)(lp_mathlib_BigOperators_bigsum___closed__1));
v___x_2919_ = ((lean_object*)(lp_mathlib_BigOperators_delabFinsetSum___lam__1___closed__0));
lean_inc_n(v___x_2917_, 2);
v___x_2920_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2920_, 0, v___x_2917_);
lean_ctor_set(v___x_2920_, 1, v___x_2919_);
v___x_2921_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBinders___closed__1));
v___x_2922_ = l_Lean_Syntax_node1(v___x_2917_, v___x_2921_, v_binder_2915_);
v___x_2923_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__25));
v___x_2924_ = lean_obj_once(&lp_mathlib_BigOperators_bigOpBindersPattern___closed__0, &lp_mathlib_BigOperators_bigOpBindersPattern___closed__0_once, _init_lp_mathlib_BigOperators_bigOpBindersPattern___closed__0);
if (lean_obj_tag(v___y_2912_) == 1)
{
lean_object* v_val_2925_; lean_object* v___x_2926_; 
v_val_2925_ = lean_ctor_get(v___y_2912_, 0);
lean_inc(v_val_2925_);
lean_dec_ref_known(v___y_2912_, 1);
v___x_2926_ = l_Array_mkArray1___redArg(v_val_2925_);
v___y_2897_ = v___x_2920_;
v___y_2898_ = v___x_2917_;
v___y_2899_ = v___x_2923_;
v___y_2900_ = v___x_2924_;
v___y_2901_ = v___x_2922_;
v___y_2902_ = v___x_2918_;
v___y_2903_ = v___y_2914_;
v___y_2904_ = v___x_2926_;
goto v___jp_2896_;
}
else
{
lean_object* v___x_2927_; 
lean_dec(v___y_2912_);
v___x_2927_ = ((lean_object*)(lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__1));
v___y_2897_ = v___x_2920_;
v___y_2898_ = v___x_2917_;
v___y_2899_ = v___x_2923_;
v___y_2900_ = v___x_2924_;
v___y_2901_ = v___x_2922_;
v___y_2902_ = v___x_2918_;
v___y_2903_ = v___y_2914_;
v___y_2904_ = v___x_2927_;
goto v___jp_2896_;
}
}
v___jp_2928_:
{
switch(lean_obj_tag(v___y_2929_))
{
case 0:
{
lean_object* v_s_2935_; lean_object* v_ref_2936_; lean_object* v___x_2937_; lean_object* v___x_2938_; lean_object* v___x_2939_; lean_object* v___x_2940_; lean_object* v___x_2941_; lean_object* v___x_2942_; lean_object* v___x_2943_; lean_object* v___x_2944_; lean_object* v___x_2945_; lean_object* v___x_2946_; lean_object* v___x_2947_; lean_object* v___x_2948_; lean_object* v___x_2949_; lean_object* v___x_2950_; lean_object* v___x_2951_; 
v_s_2935_ = lean_ctor_get(v___y_2929_, 0);
lean_inc(v_s_2935_);
lean_dec_ref_known(v___y_2929_, 1);
v_ref_2936_ = lean_ctor_get(v___y_2818_, 5);
v___x_2937_ = l_Lean_SourceInfo_fromRef(v_ref_2936_, v___y_2931_);
v___x_2938_ = ((lean_object*)(lp_mathlib_BigOperators_bigsum___closed__1));
v___x_2939_ = ((lean_object*)(lp_mathlib_BigOperators_delabFinsetSum___lam__1___closed__0));
lean_inc_n(v___x_2937_, 6);
v___x_2940_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2940_, 0, v___x_2937_);
lean_ctor_set(v___x_2940_, 1, v___x_2939_);
v___x_2941_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBinders___closed__1));
v___x_2942_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBinder___closed__2));
v___x_2943_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__25));
v___x_2944_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__2));
v___x_2945_ = ((lean_object*)(lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__2));
v___x_2946_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2946_, 0, v___x_2937_);
lean_ctor_set(v___x_2946_, 1, v___x_2945_);
v___x_2947_ = l_Lean_Syntax_node2(v___x_2937_, v___x_2944_, v___x_2946_, v_s_2935_);
v___x_2948_ = l_Lean_Syntax_node1(v___x_2937_, v___x_2943_, v___x_2947_);
v___x_2949_ = l_Lean_Syntax_node2(v___x_2937_, v___x_2942_, v___y_2932_, v___x_2948_);
v___x_2950_ = l_Lean_Syntax_node1(v___x_2937_, v___x_2941_, v___x_2949_);
v___x_2951_ = lean_obj_once(&lp_mathlib_BigOperators_bigOpBindersPattern___closed__0, &lp_mathlib_BigOperators_bigOpBindersPattern___closed__0_once, _init_lp_mathlib_BigOperators_bigOpBindersPattern___closed__0);
if (lean_obj_tag(v_a_2934_) == 1)
{
lean_object* v_val_2952_; lean_object* v___x_2953_; 
v_val_2952_ = lean_ctor_get(v_a_2934_, 0);
lean_inc(v_val_2952_);
lean_dec_ref_known(v_a_2934_, 1);
v___x_2953_ = l_Array_mkArray1___redArg(v_val_2952_);
v___y_2882_ = v___x_2950_;
v___y_2883_ = v___x_2937_;
v___y_2884_ = v___x_2940_;
v___y_2885_ = v___x_2951_;
v___y_2886_ = v___x_2943_;
v___y_2887_ = v___x_2938_;
v___y_2888_ = v___y_2933_;
v___y_2889_ = v___x_2953_;
goto v___jp_2881_;
}
else
{
lean_object* v___x_2954_; 
lean_dec(v_a_2934_);
v___x_2954_ = ((lean_object*)(lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__1));
v___y_2882_ = v___x_2950_;
v___y_2883_ = v___x_2937_;
v___y_2884_ = v___x_2940_;
v___y_2885_ = v___x_2951_;
v___y_2886_ = v___x_2943_;
v___y_2887_ = v___x_2938_;
v___y_2888_ = v___y_2933_;
v___y_2889_ = v___x_2954_;
goto v___jp_2881_;
}
}
case 1:
{
if (v___y_2930_ == 0)
{
lean_object* v_ref_2955_; lean_object* v___x_2956_; lean_object* v___x_2957_; lean_object* v___x_2958_; lean_object* v___x_2959_; lean_object* v___x_2960_; lean_object* v___x_2961_; 
v_ref_2955_ = lean_ctor_get(v___y_2818_, 5);
v___x_2956_ = l_Lean_SourceInfo_fromRef(v_ref_2955_, v___y_2931_);
v___x_2957_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBinder___closed__2));
v___x_2958_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__25));
v___x_2959_ = lean_obj_once(&lp_mathlib_BigOperators_bigOpBindersPattern___closed__0, &lp_mathlib_BigOperators_bigOpBindersPattern___closed__0_once, _init_lp_mathlib_BigOperators_bigOpBindersPattern___closed__0);
lean_inc(v___x_2956_);
v___x_2960_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2960_, 0, v___x_2956_);
lean_ctor_set(v___x_2960_, 1, v___x_2958_);
lean_ctor_set(v___x_2960_, 2, v___x_2959_);
v___x_2961_ = l_Lean_Syntax_node2(v___x_2956_, v___x_2957_, v___y_2932_, v___x_2960_);
v___y_2912_ = v_a_2934_;
v___y_2913_ = v___y_2931_;
v___y_2914_ = v___y_2933_;
v_binder_2915_ = v___x_2961_;
v_ref_2916_ = v_ref_2955_;
goto v___jp_2911_;
}
else
{
lean_object* v___x_2962_; lean_object* v___x_2963_; lean_object* v___x_2964_; 
v___x_2962_ = lean_unsigned_to_nat(0u);
v___x_2963_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult___closed__0));
v___x_2964_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__1___redArg(v___x_2962_, v___x_2963_, v___y_2814_, v___y_2815_, v___y_2816_, v___y_2817_, v___y_2818_, v___y_2819_);
if (lean_obj_tag(v___x_2964_) == 0)
{
lean_object* v_a_2965_; lean_object* v_ref_2966_; lean_object* v___x_2967_; lean_object* v___x_2968_; lean_object* v___x_2969_; lean_object* v___x_2970_; lean_object* v___x_2971_; lean_object* v___x_2972_; lean_object* v___x_2973_; lean_object* v___x_2974_; lean_object* v___x_2975_; 
v_a_2965_ = lean_ctor_get(v___x_2964_, 0);
lean_inc(v_a_2965_);
lean_dec_ref_known(v___x_2964_, 1);
v_ref_2966_ = lean_ctor_get(v___y_2818_, 5);
v___x_2967_ = l_Lean_SourceInfo_fromRef(v_ref_2966_, v___y_2931_);
v___x_2968_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBinder___closed__2));
v___x_2969_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__25));
v___x_2970_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBinder___closed__13));
v___x_2971_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__61));
lean_inc_n(v___x_2967_, 3);
v___x_2972_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2972_, 0, v___x_2967_);
lean_ctor_set(v___x_2972_, 1, v___x_2971_);
v___x_2973_ = l_Lean_Syntax_node2(v___x_2967_, v___x_2970_, v___x_2972_, v_a_2965_);
v___x_2974_ = l_Lean_Syntax_node1(v___x_2967_, v___x_2969_, v___x_2973_);
v___x_2975_ = l_Lean_Syntax_node2(v___x_2967_, v___x_2968_, v___y_2932_, v___x_2974_);
v___y_2912_ = v_a_2934_;
v___y_2913_ = v___y_2931_;
v___y_2914_ = v___y_2933_;
v_binder_2915_ = v___x_2975_;
v_ref_2916_ = v_ref_2966_;
goto v___jp_2911_;
}
else
{
lean_dec(v_a_2934_);
lean_dec(v___y_2933_);
lean_dec(v___y_2932_);
return v___x_2964_;
}
}
}
case 2:
{
lean_object* v_n_2976_; lean_object* v_ref_2977_; lean_object* v___x_2978_; lean_object* v___x_2979_; lean_object* v___x_2980_; lean_object* v___x_2981_; lean_object* v___x_2982_; lean_object* v___x_2983_; lean_object* v___x_2984_; lean_object* v___x_2985_; lean_object* v___x_2986_; lean_object* v___x_2987_; lean_object* v___x_2988_; lean_object* v___x_2989_; lean_object* v___x_2990_; lean_object* v___x_2991_; lean_object* v___x_2992_; 
v_n_2976_ = lean_ctor_get(v___y_2929_, 0);
lean_inc(v_n_2976_);
lean_dec_ref_known(v___y_2929_, 1);
v_ref_2977_ = lean_ctor_get(v___y_2818_, 5);
v___x_2978_ = l_Lean_SourceInfo_fromRef(v_ref_2977_, v___y_2931_);
v___x_2979_ = ((lean_object*)(lp_mathlib_BigOperators_bigsum___closed__1));
v___x_2980_ = ((lean_object*)(lp_mathlib_BigOperators_delabFinsetSum___lam__1___closed__0));
lean_inc_n(v___x_2978_, 6);
v___x_2981_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2981_, 0, v___x_2978_);
lean_ctor_set(v___x_2981_, 1, v___x_2980_);
v___x_2982_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBinders___closed__1));
v___x_2983_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBinder___closed__2));
v___x_2984_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__25));
v___x_2985_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__8));
v___x_2986_ = ((lean_object*)(lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__3));
v___x_2987_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2987_, 0, v___x_2978_);
lean_ctor_set(v___x_2987_, 1, v___x_2986_);
v___x_2988_ = l_Lean_Syntax_node2(v___x_2978_, v___x_2985_, v___x_2987_, v_n_2976_);
v___x_2989_ = l_Lean_Syntax_node1(v___x_2978_, v___x_2984_, v___x_2988_);
v___x_2990_ = l_Lean_Syntax_node2(v___x_2978_, v___x_2983_, v___y_2932_, v___x_2989_);
v___x_2991_ = l_Lean_Syntax_node1(v___x_2978_, v___x_2982_, v___x_2990_);
v___x_2992_ = lean_obj_once(&lp_mathlib_BigOperators_bigOpBindersPattern___closed__0, &lp_mathlib_BigOperators_bigOpBindersPattern___closed__0_once, _init_lp_mathlib_BigOperators_bigOpBindersPattern___closed__0);
if (lean_obj_tag(v_a_2934_) == 1)
{
lean_object* v_val_2993_; lean_object* v___x_2994_; 
v_val_2993_ = lean_ctor_get(v_a_2934_, 0);
lean_inc(v_val_2993_);
lean_dec_ref_known(v_a_2934_, 1);
v___x_2994_ = l_Array_mkArray1___redArg(v_val_2993_);
v___y_2867_ = v___x_2978_;
v___y_2868_ = v___x_2979_;
v___y_2869_ = v___x_2991_;
v___y_2870_ = v___x_2981_;
v___y_2871_ = v___y_2933_;
v___y_2872_ = v___x_2984_;
v___y_2873_ = v___x_2992_;
v___y_2874_ = v___x_2994_;
goto v___jp_2866_;
}
else
{
lean_object* v___x_2995_; 
lean_dec(v_a_2934_);
v___x_2995_ = ((lean_object*)(lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__1));
v___y_2867_ = v___x_2978_;
v___y_2868_ = v___x_2979_;
v___y_2869_ = v___x_2991_;
v___y_2870_ = v___x_2981_;
v___y_2871_ = v___y_2933_;
v___y_2872_ = v___x_2984_;
v___y_2873_ = v___x_2992_;
v___y_2874_ = v___x_2995_;
goto v___jp_2866_;
}
}
case 3:
{
lean_object* v_n_2996_; lean_object* v_ref_2997_; lean_object* v___x_2998_; lean_object* v___x_2999_; lean_object* v___x_3000_; lean_object* v___x_3001_; lean_object* v___x_3002_; lean_object* v___x_3003_; lean_object* v___x_3004_; lean_object* v___x_3005_; lean_object* v___x_3006_; lean_object* v___x_3007_; lean_object* v___x_3008_; lean_object* v___x_3009_; lean_object* v___x_3010_; lean_object* v___x_3011_; lean_object* v___x_3012_; 
v_n_2996_ = lean_ctor_get(v___y_2929_, 0);
lean_inc(v_n_2996_);
lean_dec_ref_known(v___y_2929_, 1);
v_ref_2997_ = lean_ctor_get(v___y_2818_, 5);
v___x_2998_ = l_Lean_SourceInfo_fromRef(v_ref_2997_, v___y_2931_);
v___x_2999_ = ((lean_object*)(lp_mathlib_BigOperators_bigsum___closed__1));
v___x_3000_ = ((lean_object*)(lp_mathlib_BigOperators_delabFinsetSum___lam__1___closed__0));
lean_inc_n(v___x_2998_, 6);
v___x_3001_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3001_, 0, v___x_2998_);
lean_ctor_set(v___x_3001_, 1, v___x_3000_);
v___x_3002_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBinders___closed__1));
v___x_3003_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBinder___closed__2));
v___x_3004_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__25));
v___x_3005_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__10));
v___x_3006_ = ((lean_object*)(lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__4));
v___x_3007_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3007_, 0, v___x_2998_);
lean_ctor_set(v___x_3007_, 1, v___x_3006_);
v___x_3008_ = l_Lean_Syntax_node2(v___x_2998_, v___x_3005_, v___x_3007_, v_n_2996_);
v___x_3009_ = l_Lean_Syntax_node1(v___x_2998_, v___x_3004_, v___x_3008_);
v___x_3010_ = l_Lean_Syntax_node2(v___x_2998_, v___x_3003_, v___y_2932_, v___x_3009_);
v___x_3011_ = l_Lean_Syntax_node1(v___x_2998_, v___x_3002_, v___x_3010_);
v___x_3012_ = lean_obj_once(&lp_mathlib_BigOperators_bigOpBindersPattern___closed__0, &lp_mathlib_BigOperators_bigOpBindersPattern___closed__0_once, _init_lp_mathlib_BigOperators_bigOpBindersPattern___closed__0);
if (lean_obj_tag(v_a_2934_) == 1)
{
lean_object* v_val_3013_; lean_object* v___x_3014_; 
v_val_3013_ = lean_ctor_get(v_a_2934_, 0);
lean_inc(v_val_3013_);
lean_dec_ref_known(v_a_2934_, 1);
v___x_3014_ = l_Array_mkArray1___redArg(v_val_3013_);
v___y_2852_ = v___x_3004_;
v___y_2853_ = v___x_3011_;
v___y_2854_ = v___x_3012_;
v___y_2855_ = v___x_2998_;
v___y_2856_ = v___x_3001_;
v___y_2857_ = v___x_2999_;
v___y_2858_ = v___y_2933_;
v___y_2859_ = v___x_3014_;
goto v___jp_2851_;
}
else
{
lean_object* v___x_3015_; 
lean_dec(v_a_2934_);
v___x_3015_ = ((lean_object*)(lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__1));
v___y_2852_ = v___x_3004_;
v___y_2853_ = v___x_3011_;
v___y_2854_ = v___x_3012_;
v___y_2855_ = v___x_2998_;
v___y_2856_ = v___x_3001_;
v___y_2857_ = v___x_2999_;
v___y_2858_ = v___y_2933_;
v___y_2859_ = v___x_3015_;
goto v___jp_2851_;
}
}
case 4:
{
lean_object* v_n_3016_; lean_object* v_ref_3017_; lean_object* v___x_3018_; lean_object* v___x_3019_; lean_object* v___x_3020_; lean_object* v___x_3021_; lean_object* v___x_3022_; lean_object* v___x_3023_; lean_object* v___x_3024_; lean_object* v___x_3025_; lean_object* v___x_3026_; lean_object* v___x_3027_; lean_object* v___x_3028_; lean_object* v___x_3029_; lean_object* v___x_3030_; lean_object* v___x_3031_; lean_object* v___x_3032_; 
v_n_3016_ = lean_ctor_get(v___y_2929_, 0);
lean_inc(v_n_3016_);
lean_dec_ref_known(v___y_2929_, 1);
v_ref_3017_ = lean_ctor_get(v___y_2818_, 5);
v___x_3018_ = l_Lean_SourceInfo_fromRef(v_ref_3017_, v___y_2931_);
v___x_3019_ = ((lean_object*)(lp_mathlib_BigOperators_bigsum___closed__1));
v___x_3020_ = ((lean_object*)(lp_mathlib_BigOperators_delabFinsetSum___lam__1___closed__0));
lean_inc_n(v___x_3018_, 6);
v___x_3021_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3021_, 0, v___x_3018_);
lean_ctor_set(v___x_3021_, 1, v___x_3020_);
v___x_3022_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBinders___closed__1));
v___x_3023_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBinder___closed__2));
v___x_3024_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__25));
v___x_3025_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__12));
v___x_3026_ = ((lean_object*)(lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__5));
v___x_3027_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3027_, 0, v___x_3018_);
lean_ctor_set(v___x_3027_, 1, v___x_3026_);
v___x_3028_ = l_Lean_Syntax_node2(v___x_3018_, v___x_3025_, v___x_3027_, v_n_3016_);
v___x_3029_ = l_Lean_Syntax_node1(v___x_3018_, v___x_3024_, v___x_3028_);
v___x_3030_ = l_Lean_Syntax_node2(v___x_3018_, v___x_3023_, v___y_2932_, v___x_3029_);
v___x_3031_ = l_Lean_Syntax_node1(v___x_3018_, v___x_3022_, v___x_3030_);
v___x_3032_ = lean_obj_once(&lp_mathlib_BigOperators_bigOpBindersPattern___closed__0, &lp_mathlib_BigOperators_bigOpBindersPattern___closed__0_once, _init_lp_mathlib_BigOperators_bigOpBindersPattern___closed__0);
if (lean_obj_tag(v_a_2934_) == 1)
{
lean_object* v_val_3033_; lean_object* v___x_3034_; 
v_val_3033_ = lean_ctor_get(v_a_2934_, 0);
lean_inc(v_val_3033_);
lean_dec_ref_known(v_a_2934_, 1);
v___x_3034_ = l_Array_mkArray1___redArg(v_val_3033_);
v___y_2837_ = v___x_3024_;
v___y_2838_ = v___x_3031_;
v___y_2839_ = v___x_3021_;
v___y_2840_ = v___x_3018_;
v___y_2841_ = v___x_3019_;
v___y_2842_ = v___y_2933_;
v___y_2843_ = v___x_3032_;
v___y_2844_ = v___x_3034_;
goto v___jp_2836_;
}
else
{
lean_object* v___x_3035_; 
lean_dec(v_a_2934_);
v___x_3035_ = ((lean_object*)(lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__1));
v___y_2837_ = v___x_3024_;
v___y_2838_ = v___x_3031_;
v___y_2839_ = v___x_3021_;
v___y_2840_ = v___x_3018_;
v___y_2841_ = v___x_3019_;
v___y_2842_ = v___y_2933_;
v___y_2843_ = v___x_3032_;
v___y_2844_ = v___x_3035_;
goto v___jp_2836_;
}
}
default: 
{
lean_object* v_n_3036_; lean_object* v_ref_3037_; lean_object* v___x_3038_; lean_object* v___x_3039_; lean_object* v___x_3040_; lean_object* v___x_3041_; lean_object* v___x_3042_; lean_object* v___x_3043_; lean_object* v___x_3044_; lean_object* v___x_3045_; lean_object* v___x_3046_; lean_object* v___x_3047_; lean_object* v___x_3048_; lean_object* v___x_3049_; lean_object* v___x_3050_; lean_object* v___x_3051_; lean_object* v___x_3052_; 
v_n_3036_ = lean_ctor_get(v___y_2929_, 0);
lean_inc(v_n_3036_);
lean_dec_ref_known(v___y_2929_, 1);
v_ref_3037_ = lean_ctor_get(v___y_2818_, 5);
v___x_3038_ = l_Lean_SourceInfo_fromRef(v_ref_3037_, v___y_2931_);
v___x_3039_ = ((lean_object*)(lp_mathlib_BigOperators_bigsum___closed__1));
v___x_3040_ = ((lean_object*)(lp_mathlib_BigOperators_delabFinsetSum___lam__1___closed__0));
lean_inc_n(v___x_3038_, 6);
v___x_3041_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3041_, 0, v___x_3038_);
lean_ctor_set(v___x_3041_, 1, v___x_3040_);
v___x_3042_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBinders___closed__1));
v___x_3043_ = ((lean_object*)(lp_mathlib_BigOperators_bigOpBinder___closed__2));
v___x_3044_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__25));
v___x_3045_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__14));
v___x_3046_ = ((lean_object*)(lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__6));
v___x_3047_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3047_, 0, v___x_3038_);
lean_ctor_set(v___x_3047_, 1, v___x_3046_);
v___x_3048_ = l_Lean_Syntax_node2(v___x_3038_, v___x_3045_, v___x_3047_, v_n_3036_);
v___x_3049_ = l_Lean_Syntax_node1(v___x_3038_, v___x_3044_, v___x_3048_);
v___x_3050_ = l_Lean_Syntax_node2(v___x_3038_, v___x_3043_, v___y_2932_, v___x_3049_);
v___x_3051_ = l_Lean_Syntax_node1(v___x_3038_, v___x_3042_, v___x_3050_);
v___x_3052_ = lean_obj_once(&lp_mathlib_BigOperators_bigOpBindersPattern___closed__0, &lp_mathlib_BigOperators_bigOpBindersPattern___closed__0_once, _init_lp_mathlib_BigOperators_bigOpBindersPattern___closed__0);
if (lean_obj_tag(v_a_2934_) == 1)
{
lean_object* v_val_3053_; lean_object* v___x_3054_; 
v_val_3053_ = lean_ctor_get(v_a_2934_, 0);
lean_inc(v_val_3053_);
lean_dec_ref_known(v_a_2934_, 1);
v___x_3054_ = l_Array_mkArray1___redArg(v_val_3053_);
v___y_2822_ = v___x_3041_;
v___y_2823_ = v___x_3044_;
v___y_2824_ = v___x_3038_;
v___y_2825_ = v___x_3039_;
v___y_2826_ = v___x_3052_;
v___y_2827_ = v___y_2933_;
v___y_2828_ = v___x_3051_;
v___y_2829_ = v___x_3054_;
goto v___jp_2821_;
}
else
{
lean_object* v___x_3055_; 
lean_dec(v_a_2934_);
v___x_3055_ = ((lean_object*)(lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__1));
v___y_2822_ = v___x_3041_;
v___y_2823_ = v___x_3044_;
v___y_2824_ = v___x_3038_;
v___y_2825_ = v___x_3039_;
v___y_2826_ = v___x_3052_;
v___y_2827_ = v___y_2933_;
v___y_2828_ = v___x_3051_;
v___y_2829_ = v___x_3055_;
goto v___jp_2821_;
}
}
}
}
v___jp_3056_:
{
lean_object* v___x_3057_; lean_object* v___x_3058_; 
v___x_3057_ = ((lean_object*)(lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__8));
v___x_3058_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00BigOperators_delabFinsetProd_spec__0___redArg(v___x_3057_, v___y_2814_, v___y_2815_, v___y_2816_, v___y_2817_, v___y_2818_, v___y_2819_);
if (lean_obj_tag(v___x_3058_) == 0)
{
lean_object* v_a_3059_; uint8_t v___x_3060_; lean_object* v___x_3061_; lean_object* v___x_3062_; lean_object* v___x_3063_; lean_object* v___x_3064_; 
v_a_3059_ = lean_ctor_get(v___x_3058_, 0);
lean_inc(v_a_3059_);
lean_dec_ref_known(v___x_3058_, 1);
v___x_3060_ = 0;
v___x_3061_ = l_Lean_NameSet_empty;
v___x_3062_ = lean_box(v___x_3060_);
v___x_3063_ = lean_alloc_closure((void*)(l_Lean_PrettyPrinter_Delaborator_withBindingBodyUnusedName___boxed), 11, 4);
lean_closure_set(v___x_3063_, 0, lean_box(0));
lean_closure_set(v___x_3063_, 1, v___f_2813_);
lean_closure_set(v___x_3063_, 2, v___x_3062_);
lean_closure_set(v___x_3063_, 3, v___x_3061_);
v___x_3064_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00BigOperators_delabFinsetProd_spec__0___redArg(v___x_3063_, v___y_2814_, v___y_2815_, v___y_2816_, v___y_2817_, v___y_2818_, v___y_2819_);
if (lean_obj_tag(v___x_3064_) == 0)
{
lean_object* v_a_3065_; lean_object* v_fst_3066_; lean_object* v_snd_3067_; lean_object* v___x_3068_; lean_object* v___x_3069_; lean_object* v___x_3070_; 
v_a_3065_ = lean_ctor_get(v___x_3064_, 0);
lean_inc(v_a_3065_);
lean_dec_ref_known(v___x_3064_, 1);
v_fst_3066_ = lean_ctor_get(v_a_3065_, 0);
lean_inc_n(v_fst_3066_, 2);
v_snd_3067_ = lean_ctor_get(v_a_3065_, 1);
lean_inc(v_snd_3067_);
lean_dec(v_a_3065_);
v___x_3068_ = lean_unsigned_to_nat(3u);
v___x_3069_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetArg___boxed), 8, 1);
lean_closure_set(v___x_3069_, 0, v_fst_3066_);
v___x_3070_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_BigOperators_Group_Finset_Defs_0__BigOperators_delabFinsetResult_spec__1___redArg(v___x_3068_, v___x_3069_, v___y_2814_, v___y_2815_, v___y_2816_, v___y_2817_, v___y_2818_, v___y_2819_);
if (lean_obj_tag(v___x_3070_) == 0)
{
lean_object* v_a_3071_; lean_object* v_filter_3072_; 
v_a_3071_ = lean_ctor_get(v___x_3070_, 0);
lean_inc(v_a_3071_);
lean_dec_ref_known(v___x_3070_, 1);
v_filter_3072_ = lean_ctor_get(v_a_3071_, 1);
lean_inc(v_filter_3072_);
if (lean_obj_tag(v_filter_3072_) == 0)
{
lean_object* v_finset_3073_; uint8_t v___x_3074_; 
v_finset_3073_ = lean_ctor_get(v_a_3071_, 0);
lean_inc(v_finset_3073_);
lean_dec(v_a_3071_);
v___x_3074_ = lean_unbox(v_a_3059_);
lean_dec(v_a_3059_);
v___y_2929_ = v_finset_3073_;
v___y_2930_ = v___x_3074_;
v___y_2931_ = v___x_3060_;
v___y_2932_ = v_fst_3066_;
v___y_2933_ = v_snd_3067_;
v_a_2934_ = v_filter_3072_;
goto v___jp_2928_;
}
else
{
lean_object* v_finset_3075_; lean_object* v___x_3077_; uint8_t v_isShared_3078_; uint8_t v_isSharedCheck_3099_; 
v_finset_3075_ = lean_ctor_get(v_a_3071_, 0);
v_isSharedCheck_3099_ = !lean_is_exclusive(v_a_3071_);
if (v_isSharedCheck_3099_ == 0)
{
lean_object* v_unused_3100_; 
v_unused_3100_ = lean_ctor_get(v_a_3071_, 1);
lean_dec(v_unused_3100_);
v___x_3077_ = v_a_3071_;
v_isShared_3078_ = v_isSharedCheck_3099_;
goto v_resetjp_3076_;
}
else
{
lean_inc(v_finset_3075_);
lean_dec(v_a_3071_);
v___x_3077_ = lean_box(0);
v_isShared_3078_ = v_isSharedCheck_3099_;
goto v_resetjp_3076_;
}
v_resetjp_3076_:
{
lean_object* v_val_3079_; lean_object* v___x_3081_; uint8_t v_isShared_3082_; uint8_t v_isSharedCheck_3098_; 
v_val_3079_ = lean_ctor_get(v_filter_3072_, 0);
v_isSharedCheck_3098_ = !lean_is_exclusive(v_filter_3072_);
if (v_isSharedCheck_3098_ == 0)
{
v___x_3081_ = v_filter_3072_;
v_isShared_3082_ = v_isSharedCheck_3098_;
goto v_resetjp_3080_;
}
else
{
lean_inc(v_val_3079_);
lean_dec(v_filter_3072_);
v___x_3081_ = lean_box(0);
v_isShared_3082_ = v_isSharedCheck_3098_;
goto v_resetjp_3080_;
}
v_resetjp_3080_:
{
lean_object* v_ref_3083_; lean_object* v___x_3084_; lean_object* v___x_3085_; lean_object* v___x_3086_; lean_object* v___x_3088_; 
v_ref_3083_ = lean_ctor_get(v___y_2818_, 5);
v___x_3084_ = l_Lean_SourceInfo_fromRef(v_ref_3083_, v___x_3060_);
v___x_3085_ = ((lean_object*)(lp_mathlib_BigOperators_BigOpWith___closed__1));
v___x_3086_ = ((lean_object*)(lp_mathlib_BigOperators_delabFinsetProd___lam__1___closed__9));
lean_inc(v___x_3084_);
if (v_isShared_3078_ == 0)
{
lean_ctor_set_tag(v___x_3077_, 2);
lean_ctor_set(v___x_3077_, 1, v___x_3086_);
lean_ctor_set(v___x_3077_, 0, v___x_3084_);
v___x_3088_ = v___x_3077_;
goto v_reusejp_3087_;
}
else
{
lean_object* v_reuseFailAlloc_3097_; 
v_reuseFailAlloc_3097_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3097_, 0, v___x_3084_);
lean_ctor_set(v_reuseFailAlloc_3097_, 1, v___x_3086_);
v___x_3088_ = v_reuseFailAlloc_3097_;
goto v_reusejp_3087_;
}
v_reusejp_3087_:
{
lean_object* v___x_3089_; lean_object* v___x_3090_; lean_object* v___x_3091_; lean_object* v___x_3092_; lean_object* v___x_3094_; 
v___x_3089_ = ((lean_object*)(lp_mathlib_BigOperators_processBigOpBinder___closed__25));
v___x_3090_ = lean_obj_once(&lp_mathlib_BigOperators_bigOpBindersPattern___closed__0, &lp_mathlib_BigOperators_bigOpBindersPattern___closed__0_once, _init_lp_mathlib_BigOperators_bigOpBindersPattern___closed__0);
lean_inc(v___x_3084_);
v___x_3091_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_3091_, 0, v___x_3084_);
lean_ctor_set(v___x_3091_, 1, v___x_3089_);
lean_ctor_set(v___x_3091_, 2, v___x_3090_);
v___x_3092_ = l_Lean_Syntax_node3(v___x_3084_, v___x_3085_, v___x_3088_, v___x_3091_, v_val_3079_);
if (v_isShared_3082_ == 0)
{
lean_ctor_set(v___x_3081_, 0, v___x_3092_);
v___x_3094_ = v___x_3081_;
goto v_reusejp_3093_;
}
else
{
lean_object* v_reuseFailAlloc_3096_; 
v_reuseFailAlloc_3096_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3096_, 0, v___x_3092_);
v___x_3094_ = v_reuseFailAlloc_3096_;
goto v_reusejp_3093_;
}
v_reusejp_3093_:
{
uint8_t v___x_3095_; 
v___x_3095_ = lean_unbox(v_a_3059_);
lean_dec(v_a_3059_);
v___y_2929_ = v_finset_3075_;
v___y_2930_ = v___x_3095_;
v___y_2931_ = v___x_3060_;
v___y_2932_ = v_fst_3066_;
v___y_2933_ = v_snd_3067_;
v_a_2934_ = v___x_3094_;
goto v___jp_2928_;
}
}
}
}
}
}
else
{
lean_object* v_a_3101_; lean_object* v___x_3103_; uint8_t v_isShared_3104_; uint8_t v_isSharedCheck_3108_; 
lean_dec(v_snd_3067_);
lean_dec(v_fst_3066_);
lean_dec(v_a_3059_);
v_a_3101_ = lean_ctor_get(v___x_3070_, 0);
v_isSharedCheck_3108_ = !lean_is_exclusive(v___x_3070_);
if (v_isSharedCheck_3108_ == 0)
{
v___x_3103_ = v___x_3070_;
v_isShared_3104_ = v_isSharedCheck_3108_;
goto v_resetjp_3102_;
}
else
{
lean_inc(v_a_3101_);
lean_dec(v___x_3070_);
v___x_3103_ = lean_box(0);
v_isShared_3104_ = v_isSharedCheck_3108_;
goto v_resetjp_3102_;
}
v_resetjp_3102_:
{
lean_object* v___x_3106_; 
if (v_isShared_3104_ == 0)
{
v___x_3106_ = v___x_3103_;
goto v_reusejp_3105_;
}
else
{
lean_object* v_reuseFailAlloc_3107_; 
v_reuseFailAlloc_3107_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3107_, 0, v_a_3101_);
v___x_3106_ = v_reuseFailAlloc_3107_;
goto v_reusejp_3105_;
}
v_reusejp_3105_:
{
return v___x_3106_;
}
}
}
}
else
{
lean_object* v_a_3109_; lean_object* v___x_3111_; uint8_t v_isShared_3112_; uint8_t v_isSharedCheck_3116_; 
lean_dec(v_a_3059_);
v_a_3109_ = lean_ctor_get(v___x_3064_, 0);
v_isSharedCheck_3116_ = !lean_is_exclusive(v___x_3064_);
if (v_isSharedCheck_3116_ == 0)
{
v___x_3111_ = v___x_3064_;
v_isShared_3112_ = v_isSharedCheck_3116_;
goto v_resetjp_3110_;
}
else
{
lean_inc(v_a_3109_);
lean_dec(v___x_3064_);
v___x_3111_ = lean_box(0);
v_isShared_3112_ = v_isSharedCheck_3116_;
goto v_resetjp_3110_;
}
v_resetjp_3110_:
{
lean_object* v___x_3114_; 
if (v_isShared_3112_ == 0)
{
v___x_3114_ = v___x_3111_;
goto v_reusejp_3113_;
}
else
{
lean_object* v_reuseFailAlloc_3115_; 
v_reuseFailAlloc_3115_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3115_, 0, v_a_3109_);
v___x_3114_ = v_reuseFailAlloc_3115_;
goto v_reusejp_3113_;
}
v_reusejp_3113_:
{
return v___x_3114_;
}
}
}
}
else
{
lean_object* v_a_3117_; lean_object* v___x_3119_; uint8_t v_isShared_3120_; uint8_t v_isSharedCheck_3124_; 
lean_dec_ref(v___f_2813_);
v_a_3117_ = lean_ctor_get(v___x_3058_, 0);
v_isSharedCheck_3124_ = !lean_is_exclusive(v___x_3058_);
if (v_isSharedCheck_3124_ == 0)
{
v___x_3119_ = v___x_3058_;
v_isShared_3120_ = v_isSharedCheck_3124_;
goto v_resetjp_3118_;
}
else
{
lean_inc(v_a_3117_);
lean_dec(v___x_3058_);
v___x_3119_ = lean_box(0);
v_isShared_3120_ = v_isSharedCheck_3124_;
goto v_resetjp_3118_;
}
v_resetjp_3118_:
{
lean_object* v___x_3122_; 
if (v_isShared_3120_ == 0)
{
v___x_3122_ = v___x_3119_;
goto v_reusejp_3121_;
}
else
{
lean_object* v_reuseFailAlloc_3123_; 
v_reuseFailAlloc_3123_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3123_, 0, v_a_3117_);
v___x_3122_ = v_reuseFailAlloc_3123_;
goto v_reusejp_3121_;
}
v_reusejp_3121_:
{
return v___x_3122_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_BigOperators_delabFinsetSum___lam__1___boxed(lean_object* v___x_3148_, lean_object* v___f_3149_, lean_object* v___y_3150_, lean_object* v___y_3151_, lean_object* v___y_3152_, lean_object* v___y_3153_, lean_object* v___y_3154_, lean_object* v___y_3155_, lean_object* v___y_3156_){
_start:
{
lean_object* v_res_3157_; 
v_res_3157_ = lp_mathlib_BigOperators_delabFinsetSum___lam__1(v___x_3148_, v___f_3149_, v___y_3150_, v___y_3151_, v___y_3152_, v___y_3153_, v___y_3154_, v___y_3155_);
lean_dec(v___y_3155_);
lean_dec_ref(v___y_3154_);
lean_dec(v___y_3153_);
lean_dec_ref(v___y_3152_);
lean_dec(v___y_3151_);
lean_dec_ref(v___y_3150_);
lean_dec(v___x_3148_);
return v_res_3157_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BigOperators_delabFinsetSum(lean_object* v_a_3164_, lean_object* v_a_3165_, lean_object* v_a_3166_, lean_object* v_a_3167_, lean_object* v_a_3168_, lean_object* v_a_3169_){
_start:
{
lean_object* v___x_3171_; lean_object* v___x_3172_; lean_object* v___x_3173_; 
v___x_3171_ = ((lean_object*)(lp_mathlib_BigOperators_delabFinsetProd___closed__1));
v___x_3172_ = ((lean_object*)(lp_mathlib_BigOperators_delabFinsetSum___closed__1));
v___x_3173_ = l_Lean_PrettyPrinter_Delaborator_whenPPOption(v___x_3171_, v___x_3172_, v_a_3164_, v_a_3165_, v_a_3166_, v_a_3167_, v_a_3168_, v_a_3169_);
return v___x_3173_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BigOperators_delabFinsetSum___boxed(lean_object* v_a_3174_, lean_object* v_a_3175_, lean_object* v_a_3176_, lean_object* v_a_3177_, lean_object* v_a_3178_, lean_object* v_a_3179_, lean_object* v_a_3180_){
_start:
{
lean_object* v_res_3181_; 
v_res_3181_ = lp_mathlib_BigOperators_delabFinsetSum(v_a_3174_, v_a_3175_, v_a_3176_, v_a_3177_, v_a_3178_, v_a_3179_);
lean_dec(v_a_3179_);
lean_dec_ref(v_a_3178_);
lean_dec(v_a_3177_);
lean_dec_ref(v_a_3176_);
lean_dec(v_a_3175_);
lean_dec_ref(v_a_3174_);
return v_res_3181_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Equiv_Opposite(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Multiset_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_Sets(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Multiset_Bind(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Equiv_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Multiset_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_Sets(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Multiset_Bind(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ToDual(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Defs(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ToDual(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_LibraryNote_operator__precedence__of__big__operators = _init_lp_mathlib_LibraryNote_operator__precedence__of__big__operators();
lean_mark_persistent(lp_mathlib_LibraryNote_operator__precedence__of__big__operators);
lp_mathlib_BigOperators_BigOpWith = _init_lp_mathlib_BigOperators_BigOpWith();
lean_mark_persistent(lp_mathlib_BigOperators_BigOpWith);
lp_mathlib_BigOperators_bigsum = _init_lp_mathlib_BigOperators_bigsum();
lean_mark_persistent(lp_mathlib_BigOperators_bigsum);
lp_mathlib_BigOperators_bigprod = _init_lp_mathlib_BigOperators_bigprod();
lean_mark_persistent(lp_mathlib_BigOperators_bigprod);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Equiv_Opposite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Multiset_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Sets(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Multiset_Bind(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ToDual(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Equiv_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Multiset_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Sets(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Multiset_Bind(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ToDual(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
