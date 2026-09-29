// Lean compiler output
// Module: Mathlib.Tactic.Says
// Imports: public import Init public meta import Init public import Mathlib.Init public meta import Lean.Meta.Tactic.TryThis public meta import Mathlib.Lean.Elab.InfoTree public import Batteries.Linter.UnreachableTactic public import Mathlib.Tactic.Basic public meta import Qq.MatchImpl
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
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_batteries_Batteries_Linter_UnreachableTactic_addIgnoreTacticKind(lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Name_mkStr5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_register_option(lean_object*, lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_GuardExceptions_parseAsTacticSeq(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_evalTactic___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_withResetServerInfo___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Lean_Elab_collectTryThisSuggestions(lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_toString(lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_MessageData_nil;
lean_object* l_Lean_Meta_Tactic_TryThis_addSuggestion(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_evalTactic(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_Qq_Lean_Syntax_stripPos(lean_object*);
lean_object* l_Lean_PrettyPrinter_ppTactic(lean_object*, lean_object*, lean_object*);
extern lean_object* l_Std_Format_defWidth;
lean_object* l_Std_Format_pretty(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_io_getenv(lean_object*);
uint8_t l_Lean_Syntax_isNone(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4__spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__0_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "says"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__0_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__0_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__1_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "verify"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__1_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__1_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__2_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__0_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(141, 94, 166, 94, 136, 9, 136, 223)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__2_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__2_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__1_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(227, 252, 119, 140, 56, 211, 130, 200)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__2_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__2_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__3_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "Verify the output"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__3_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__3_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__4_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__3_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__4_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__4_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__5_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__5_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__5_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__6_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__6_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__6_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__7_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Says"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__7_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__7_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__8_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__5_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__8_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__8_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__6_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__8_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__8_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__7_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(49, 164, 223, 43, 155, 124, 248, 66)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__8_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4__value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__8_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__0_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(143, 165, 67, 238, 10, 199, 6, 65)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__8_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__8_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4__value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__1_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(185, 124, 16, 235, 145, 108, 51, 245)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__8_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__8_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Says_says_verify;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__0_00___x40_Mathlib_Tactic_Says_1791366168____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "no_verify_in_CI"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__0_00___x40_Mathlib_Tactic_Says_1791366168____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__0_00___x40_Mathlib_Tactic_Says_1791366168____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__1_00___x40_Mathlib_Tactic_Says_1791366168____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__0_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(141, 94, 166, 94, 136, 9, 136, 223)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__1_00___x40_Mathlib_Tactic_Says_1791366168____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__1_00___x40_Mathlib_Tactic_Says_1791366168____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__0_00___x40_Mathlib_Tactic_Says_1791366168____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(136, 77, 175, 186, 85, 217, 252, 211)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__1_00___x40_Mathlib_Tactic_Says_1791366168____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__1_00___x40_Mathlib_Tactic_Says_1791366168____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__2_00___x40_Mathlib_Tactic_Says_1791366168____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 70, .m_capacity = 70, .m_length = 69, .m_data = "Disable reverification, even if the `CI` environment variable is set."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__2_00___x40_Mathlib_Tactic_Says_1791366168____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__2_00___x40_Mathlib_Tactic_Says_1791366168____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__3_00___x40_Mathlib_Tactic_Says_1791366168____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__2_00___x40_Mathlib_Tactic_Says_1791366168____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__3_00___x40_Mathlib_Tactic_Says_1791366168____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__3_00___x40_Mathlib_Tactic_Says_1791366168____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__4_00___x40_Mathlib_Tactic_Says_1791366168____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__5_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__4_00___x40_Mathlib_Tactic_Says_1791366168____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__4_00___x40_Mathlib_Tactic_Says_1791366168____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__6_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__4_00___x40_Mathlib_Tactic_Says_1791366168____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__4_00___x40_Mathlib_Tactic_Says_1791366168____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__7_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(49, 164, 223, 43, 155, 124, 248, 66)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__4_00___x40_Mathlib_Tactic_Says_1791366168____hygCtx___hyg_4__value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__4_00___x40_Mathlib_Tactic_Says_1791366168____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__0_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(143, 165, 67, 238, 10, 199, 6, 65)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__4_00___x40_Mathlib_Tactic_Says_1791366168____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__4_00___x40_Mathlib_Tactic_Says_1791366168____hygCtx___hyg_4__value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__0_00___x40_Mathlib_Tactic_Says_1791366168____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(178, 13, 17, 0, 231, 133, 9, 116)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__4_00___x40_Mathlib_Tactic_Says_1791366168____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__4_00___x40_Mathlib_Tactic_Says_1791366168____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn_00___x40_Mathlib_Tactic_Says_1791366168____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn_00___x40_Mathlib_Tactic_Says_1791366168____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Says_says_no__verify__in__CI;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Says_evalTacticCapturingTryThis_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Says_evalTacticCapturingTryThis_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Says_evalTacticCapturingTryThis_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Says_evalTacticCapturingTryThis_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "<input>"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 41, .m_capacity = 41, .m_length = 40, .m_data = "Failed to parse 'Try this:' suggestion: "};
static const lean_object* lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__2;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "\n"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__4;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Tactic `"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__5_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__6;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 44, .m_capacity = 44, .m_length = 43, .m_data = "` did not produce a 'Try this:' suggestion."};
static const lean_object* lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__7_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__8;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 63, .m_capacity = 63, .m_length = 62, .m_data = "` produced a 'Try this:' suggestion with a non-tactic syntax: "};
static const lean_object* lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__9_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__10;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__13_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__15_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__11_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__15_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__15_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__12_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__15_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__15_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__6_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__15_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__13_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__15_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__17_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__11_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__17_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__17_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__12_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__17_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__17_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__6_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__17_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__16_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__17_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__18_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__19_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Says_evalTacticCapturingTryThis_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Says_evalTacticCapturingTryThis_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Says_says___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__5_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Says_says___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Says_says___closed__0_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__6_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Says_says___closed__0_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Says_says___closed__0_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__7_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(49, 164, 223, 43, 155, 124, 248, 66)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Says_says___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Says_says___closed__0_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__0_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(143, 165, 67, 238, 10, 199, 6, 65)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Says_says___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Says_says___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Says_says___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Says_says___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Says_says___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Says_says___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Says_says___closed__1_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Says_says___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Says_says___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Says_says___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = " says"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Says_says___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Says_says___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Says_says___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Says_says___closed__3_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Says_says___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Says_says___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Says_says___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Says_says___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Says_says___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Says_says___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Says_says___closed__5_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Says_says___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Says_says___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Says_says___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "colGt"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Says_says___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Says_says___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Says_says___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Says_says___closed__7_value),LEAN_SCALAR_PTR_LITERAL(185, 236, 32, 153, 169, 213, 53, 244)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Says_says___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Says_says___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Says_says___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Says_says___closed__8_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Says_says___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Says_says___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Says_says___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__13_value),LEAN_SCALAR_PTR_LITERAL(13, 106, 54, 236, 164, 218, 24, 154)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Says_says___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Says_says___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Says_says___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Says_says___closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Says_says___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Says_says___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Says_says___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Says_says___closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Says_says___closed__9_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Says_says___closed__11_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Says_says___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Says_says___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Says_says___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Says_says___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Says_says___closed__12_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Says_says___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Says_says___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Says_says___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Says_says___closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Says_says___closed__4_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Says_says___closed__13_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Says_says___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Says_says___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Says_says___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Says_says___closed__0_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Says_says___closed__14_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Says_says___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Says_says___closed__15_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_Says_says = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Says_says___closed__15_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1_spec__1___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__0;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Try this:"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "` produced `"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__3;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 37, .m_capacity = 37, .m_length = 36, .m_data = "`,\nbut was expecting it to produce `"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__5;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "`!"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__6_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__7;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 76, .m_capacity = 76, .m_length = 75, .m_data = "\n\nYou can reproduce this error locally using `set_option says.verify true`."};
static const lean_object* lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__8_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__9;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "CI"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__14_value),LEAN_SCALAR_PTR_LITERAL(99, 76, 33, 121, 85, 143, 17, 224)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__11_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn_00___x40_Mathlib_Tactic_Says_31067479____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn_00___x40_Mathlib_Tactic_Says_31067479____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4__spec__0(lean_object* v_name_1_, lean_object* v_decl_2_, lean_object* v_ref_3_){
_start:
{
lean_object* v_defValue_5_; lean_object* v_descr_6_; lean_object* v_deprecation_x3f_7_; lean_object* v___x_8_; uint8_t v___x_9_; lean_object* v___x_10_; lean_object* v___x_11_; 
v_defValue_5_ = lean_ctor_get(v_decl_2_, 0);
v_descr_6_ = lean_ctor_get(v_decl_2_, 1);
v_deprecation_x3f_7_ = lean_ctor_get(v_decl_2_, 2);
v___x_8_ = lean_alloc_ctor(1, 0, 1);
v___x_9_ = lean_unbox(v_defValue_5_);
lean_ctor_set_uint8(v___x_8_, 0, v___x_9_);
lean_inc(v_deprecation_x3f_7_);
lean_inc_ref(v_descr_6_);
lean_inc_n(v_name_1_, 2);
v___x_10_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_10_, 0, v_name_1_);
lean_ctor_set(v___x_10_, 1, v_ref_3_);
lean_ctor_set(v___x_10_, 2, v___x_8_);
lean_ctor_set(v___x_10_, 3, v_descr_6_);
lean_ctor_set(v___x_10_, 4, v_deprecation_x3f_7_);
v___x_11_ = lean_register_option(v_name_1_, v___x_10_);
if (lean_obj_tag(v___x_11_) == 0)
{
lean_object* v___x_13_; uint8_t v_isShared_14_; uint8_t v_isSharedCheck_19_; 
v_isSharedCheck_19_ = !lean_is_exclusive(v___x_11_);
if (v_isSharedCheck_19_ == 0)
{
lean_object* v_unused_20_; 
v_unused_20_ = lean_ctor_get(v___x_11_, 0);
lean_dec(v_unused_20_);
v___x_13_ = v___x_11_;
v_isShared_14_ = v_isSharedCheck_19_;
goto v_resetjp_12_;
}
else
{
lean_dec(v___x_11_);
v___x_13_ = lean_box(0);
v_isShared_14_ = v_isSharedCheck_19_;
goto v_resetjp_12_;
}
v_resetjp_12_:
{
lean_object* v___x_15_; lean_object* v___x_17_; 
lean_inc(v_defValue_5_);
v___x_15_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_15_, 0, v_name_1_);
lean_ctor_set(v___x_15_, 1, v_defValue_5_);
if (v_isShared_14_ == 0)
{
lean_ctor_set(v___x_13_, 0, v___x_15_);
v___x_17_ = v___x_13_;
goto v_reusejp_16_;
}
else
{
lean_object* v_reuseFailAlloc_18_; 
v_reuseFailAlloc_18_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_18_, 0, v___x_15_);
v___x_17_ = v_reuseFailAlloc_18_;
goto v_reusejp_16_;
}
v_reusejp_16_:
{
return v___x_17_;
}
}
}
else
{
lean_object* v_a_21_; lean_object* v___x_23_; uint8_t v_isShared_24_; uint8_t v_isSharedCheck_28_; 
lean_dec(v_name_1_);
v_a_21_ = lean_ctor_get(v___x_11_, 0);
v_isSharedCheck_28_ = !lean_is_exclusive(v___x_11_);
if (v_isSharedCheck_28_ == 0)
{
v___x_23_ = v___x_11_;
v_isShared_24_ = v_isSharedCheck_28_;
goto v_resetjp_22_;
}
else
{
lean_inc(v_a_21_);
lean_dec(v___x_11_);
v___x_23_ = lean_box(0);
v_isShared_24_ = v_isSharedCheck_28_;
goto v_resetjp_22_;
}
v_resetjp_22_:
{
lean_object* v___x_26_; 
if (v_isShared_24_ == 0)
{
v___x_26_ = v___x_23_;
goto v_reusejp_25_;
}
else
{
lean_object* v_reuseFailAlloc_27_; 
v_reuseFailAlloc_27_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_27_, 0, v_a_21_);
v___x_26_ = v_reuseFailAlloc_27_;
goto v_reusejp_25_;
}
v_reusejp_25_:
{
return v___x_26_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4__spec__0___boxed(lean_object* v_name_29_, lean_object* v_decl_30_, lean_object* v_ref_31_, lean_object* v_a_32_){
_start:
{
lean_object* v_res_33_; 
v_res_33_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4__spec__0(v_name_29_, v_decl_30_, v_ref_31_);
lean_dec_ref(v_decl_30_);
return v_res_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_55_; lean_object* v___x_56_; lean_object* v___x_57_; lean_object* v___x_58_; 
v___x_55_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__2_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4_));
v___x_56_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__4_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4_));
v___x_57_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__8_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4_));
v___x_58_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4__spec__0(v___x_55_, v___x_56_, v___x_57_);
return v___x_58_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4____boxed(lean_object* v_a_59_){
_start:
{
lean_object* v_res_60_; 
v_res_60_ = lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4_();
return v_res_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn_00___x40_Mathlib_Tactic_Says_1791366168____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_78_; lean_object* v___x_79_; lean_object* v___x_80_; lean_object* v___x_81_; 
v___x_78_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__1_00___x40_Mathlib_Tactic_Says_1791366168____hygCtx___hyg_4_));
v___x_79_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__3_00___x40_Mathlib_Tactic_Says_1791366168____hygCtx___hyg_4_));
v___x_80_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__4_00___x40_Mathlib_Tactic_Says_1791366168____hygCtx___hyg_4_));
v___x_81_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4__spec__0(v___x_78_, v___x_79_, v___x_80_);
return v___x_81_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn_00___x40_Mathlib_Tactic_Says_1791366168____hygCtx___hyg_4____boxed(lean_object* v_a_82_){
_start:
{
lean_object* v_res_83_; 
v_res_83_ = lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn_00___x40_Mathlib_Tactic_Says_1791366168____hygCtx___hyg_4_();
return v_res_83_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Says_evalTacticCapturingTryThis_spec__0_spec__0(lean_object* v_msgData_84_, lean_object* v___y_85_, lean_object* v___y_86_, lean_object* v___y_87_, lean_object* v___y_88_){
_start:
{
lean_object* v___x_90_; lean_object* v_env_91_; lean_object* v___x_92_; lean_object* v_mctx_93_; lean_object* v_lctx_94_; lean_object* v_options_95_; lean_object* v___x_96_; lean_object* v___x_97_; lean_object* v___x_98_; 
v___x_90_ = lean_st_ref_get(v___y_88_);
v_env_91_ = lean_ctor_get(v___x_90_, 0);
lean_inc_ref(v_env_91_);
lean_dec(v___x_90_);
v___x_92_ = lean_st_ref_get(v___y_86_);
v_mctx_93_ = lean_ctor_get(v___x_92_, 0);
lean_inc_ref(v_mctx_93_);
lean_dec(v___x_92_);
v_lctx_94_ = lean_ctor_get(v___y_85_, 2);
v_options_95_ = lean_ctor_get(v___y_87_, 2);
lean_inc_ref(v_options_95_);
lean_inc_ref(v_lctx_94_);
v___x_96_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_96_, 0, v_env_91_);
lean_ctor_set(v___x_96_, 1, v_mctx_93_);
lean_ctor_set(v___x_96_, 2, v_lctx_94_);
lean_ctor_set(v___x_96_, 3, v_options_95_);
v___x_97_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_97_, 0, v___x_96_);
lean_ctor_set(v___x_97_, 1, v_msgData_84_);
v___x_98_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_98_, 0, v___x_97_);
return v___x_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Says_evalTacticCapturingTryThis_spec__0_spec__0___boxed(lean_object* v_msgData_99_, lean_object* v___y_100_, lean_object* v___y_101_, lean_object* v___y_102_, lean_object* v___y_103_, lean_object* v___y_104_){
_start:
{
lean_object* v_res_105_; 
v_res_105_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Says_evalTacticCapturingTryThis_spec__0_spec__0(v_msgData_99_, v___y_100_, v___y_101_, v___y_102_, v___y_103_);
lean_dec(v___y_103_);
lean_dec_ref(v___y_102_);
lean_dec(v___y_101_);
lean_dec_ref(v___y_100_);
return v_res_105_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Says_evalTacticCapturingTryThis_spec__0___redArg(lean_object* v_msg_106_, lean_object* v___y_107_, lean_object* v___y_108_, lean_object* v___y_109_, lean_object* v___y_110_){
_start:
{
lean_object* v_ref_112_; lean_object* v___x_113_; lean_object* v_a_114_; lean_object* v___x_116_; uint8_t v_isShared_117_; uint8_t v_isSharedCheck_122_; 
v_ref_112_ = lean_ctor_get(v___y_109_, 5);
v___x_113_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Says_evalTacticCapturingTryThis_spec__0_spec__0(v_msg_106_, v___y_107_, v___y_108_, v___y_109_, v___y_110_);
v_a_114_ = lean_ctor_get(v___x_113_, 0);
v_isSharedCheck_122_ = !lean_is_exclusive(v___x_113_);
if (v_isSharedCheck_122_ == 0)
{
v___x_116_ = v___x_113_;
v_isShared_117_ = v_isSharedCheck_122_;
goto v_resetjp_115_;
}
else
{
lean_inc(v_a_114_);
lean_dec(v___x_113_);
v___x_116_ = lean_box(0);
v_isShared_117_ = v_isSharedCheck_122_;
goto v_resetjp_115_;
}
v_resetjp_115_:
{
lean_object* v___x_118_; lean_object* v___x_120_; 
lean_inc(v_ref_112_);
v___x_118_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_118_, 0, v_ref_112_);
lean_ctor_set(v___x_118_, 1, v_a_114_);
if (v_isShared_117_ == 0)
{
lean_ctor_set_tag(v___x_116_, 1);
lean_ctor_set(v___x_116_, 0, v___x_118_);
v___x_120_ = v___x_116_;
goto v_reusejp_119_;
}
else
{
lean_object* v_reuseFailAlloc_121_; 
v_reuseFailAlloc_121_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_121_, 0, v___x_118_);
v___x_120_ = v_reuseFailAlloc_121_;
goto v_reusejp_119_;
}
v_reusejp_119_:
{
return v___x_120_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Says_evalTacticCapturingTryThis_spec__0___redArg___boxed(lean_object* v_msg_123_, lean_object* v___y_124_, lean_object* v___y_125_, lean_object* v___y_126_, lean_object* v___y_127_, lean_object* v___y_128_){
_start:
{
lean_object* v_res_129_; 
v_res_129_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Says_evalTacticCapturingTryThis_spec__0___redArg(v_msg_123_, v___y_124_, v___y_125_, v___y_126_, v___y_127_);
lean_dec(v___y_127_);
lean_dec_ref(v___y_126_);
lean_dec(v___y_125_);
lean_dec_ref(v___y_124_);
return v_res_129_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__2(void){
_start:
{
lean_object* v___x_132_; lean_object* v___x_133_; 
v___x_132_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__1));
v___x_133_ = l_Lean_stringToMessageData(v___x_132_);
return v___x_133_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__4(void){
_start:
{
lean_object* v___x_135_; lean_object* v___x_136_; 
v___x_135_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__3));
v___x_136_ = l_Lean_stringToMessageData(v___x_135_);
return v___x_136_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__6(void){
_start:
{
lean_object* v___x_138_; lean_object* v___x_139_; 
v___x_138_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__5));
v___x_139_ = l_Lean_stringToMessageData(v___x_138_);
return v___x_139_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__8(void){
_start:
{
lean_object* v___x_141_; lean_object* v___x_142_; 
v___x_141_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__7));
v___x_142_ = l_Lean_stringToMessageData(v___x_141_);
return v___x_142_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__10(void){
_start:
{
lean_object* v___x_144_; lean_object* v___x_145_; 
v___x_144_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__9));
v___x_145_ = l_Lean_stringToMessageData(v___x_144_);
return v___x_145_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis(lean_object* v_tac_164_, lean_object* v_a_165_, lean_object* v_a_166_, lean_object* v_a_167_, lean_object* v_a_168_, lean_object* v_a_169_, lean_object* v_a_170_, lean_object* v_a_171_, lean_object* v_a_172_){
_start:
{
lean_object* v_a_175_; lean_object* v___y_176_; lean_object* v___y_177_; lean_object* v___y_178_; lean_object* v___y_179_; lean_object* v___y_180_; lean_object* v___y_181_; lean_object* v___y_182_; lean_object* v___y_183_; lean_object* v___x_205_; lean_object* v___x_206_; 
lean_inc(v_tac_164_);
v___x_205_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_evalTactic___boxed), 10, 1);
lean_closure_set(v___x_205_, 0, v_tac_164_);
v___x_206_ = lp_mathlib_Mathlib_Tactic_withResetServerInfo___redArg(v___x_205_, v_a_165_, v_a_166_, v_a_167_, v_a_168_, v_a_169_, v_a_170_, v_a_171_, v_a_172_);
if (lean_obj_tag(v___x_206_) == 0)
{
lean_object* v_a_207_; lean_object* v___x_209_; uint8_t v_isShared_210_; uint8_t v_isSharedCheck_287_; 
v_a_207_ = lean_ctor_get(v___x_206_, 0);
v_isSharedCheck_287_ = !lean_is_exclusive(v___x_206_);
if (v_isSharedCheck_287_ == 0)
{
v___x_209_ = v___x_206_;
v_isShared_210_ = v_isSharedCheck_287_;
goto v_resetjp_208_;
}
else
{
lean_inc(v_a_207_);
lean_dec(v___x_206_);
v___x_209_ = lean_box(0);
v_isShared_210_ = v_isSharedCheck_287_;
goto v_resetjp_208_;
}
v_resetjp_208_:
{
lean_object* v_trees_211_; lean_object* v___x_212_; lean_object* v___x_213_; lean_object* v___x_214_; uint8_t v___x_215_; 
v_trees_211_ = lean_ctor_get(v_a_207_, 2);
lean_inc_ref(v_trees_211_);
lean_dec(v_a_207_);
v___x_212_ = lp_mathlib_Lean_Elab_collectTryThisSuggestions(v_trees_211_);
lean_dec_ref(v_trees_211_);
v___x_213_ = lean_unsigned_to_nat(0u);
v___x_214_ = lean_array_get_size(v___x_212_);
v___x_215_ = lean_nat_dec_lt(v___x_213_, v___x_214_);
if (v___x_215_ == 0)
{
lean_object* v___x_216_; lean_object* v___x_217_; lean_object* v___x_218_; lean_object* v___x_219_; lean_object* v___x_220_; lean_object* v___x_221_; 
lean_dec_ref(v___x_212_);
lean_del_object(v___x_209_);
v___x_216_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__6, &lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__6);
v___x_217_ = l_Lean_MessageData_ofSyntax(v_tac_164_);
v___x_218_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_218_, 0, v___x_216_);
lean_ctor_set(v___x_218_, 1, v___x_217_);
v___x_219_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__8, &lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__8);
v___x_220_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_220_, 0, v___x_218_);
lean_ctor_set(v___x_220_, 1, v___x_219_);
v___x_221_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Says_evalTacticCapturingTryThis_spec__0___redArg(v___x_220_, v_a_169_, v_a_170_, v_a_171_, v_a_172_);
return v___x_221_;
}
else
{
lean_object* v___x_222_; lean_object* v_messageData_x3f_223_; 
v___x_222_ = lean_array_fget(v___x_212_, v___x_213_);
lean_dec_ref(v___x_212_);
v_messageData_x3f_223_ = lean_ctor_get(v___x_222_, 4);
if (lean_obj_tag(v_messageData_x3f_223_) == 1)
{
lean_object* v_val_224_; lean_object* v___x_225_; 
lean_inc_ref(v_messageData_x3f_223_);
lean_dec(v___x_222_);
lean_del_object(v___x_209_);
lean_dec(v_tac_164_);
v_val_224_ = lean_ctor_get(v_messageData_x3f_223_, 0);
lean_inc(v_val_224_);
lean_dec_ref_known(v_messageData_x3f_223_, 1);
v___x_225_ = l_Lean_MessageData_toString(v_val_224_);
v_a_175_ = v___x_225_;
v___y_176_ = v_a_165_;
v___y_177_ = v_a_166_;
v___y_178_ = v_a_167_;
v___y_179_ = v_a_168_;
v___y_180_ = v_a_169_;
v___y_181_ = v_a_170_;
v___y_182_ = v_a_171_;
v___y_183_ = v_a_172_;
goto v___jp_174_;
}
else
{
lean_object* v_suggestion_226_; 
v_suggestion_226_ = lean_ctor_get(v___x_222_, 0);
lean_inc_ref(v_suggestion_226_);
lean_dec(v___x_222_);
if (lean_obj_tag(v_suggestion_226_) == 0)
{
lean_object* v_kind_227_; lean_object* v_a_228_; lean_object* v___x_230_; uint8_t v_isShared_231_; uint8_t v_isSharedCheck_285_; 
v_kind_227_ = lean_ctor_get(v_suggestion_226_, 0);
v_a_228_ = lean_ctor_get(v_suggestion_226_, 1);
v_isSharedCheck_285_ = !lean_is_exclusive(v_suggestion_226_);
if (v_isSharedCheck_285_ == 0)
{
v___x_230_ = v_suggestion_226_;
v_isShared_231_ = v_isSharedCheck_285_;
goto v_resetjp_229_;
}
else
{
lean_inc(v_a_228_);
lean_inc(v_kind_227_);
lean_dec(v_suggestion_226_);
v___x_230_ = lean_box(0);
v_isShared_231_ = v_isSharedCheck_285_;
goto v_resetjp_229_;
}
v_resetjp_229_:
{
lean_object* v___y_233_; lean_object* v___y_234_; lean_object* v___y_235_; lean_object* v___y_236_; lean_object* v___y_237_; lean_object* v___y_238_; lean_object* v___y_239_; lean_object* v___y_240_; 
if (lean_obj_tag(v_kind_227_) == 1)
{
lean_object* v_pre_251_; 
v_pre_251_ = lean_ctor_get(v_kind_227_, 0);
switch(lean_obj_tag(v_pre_251_))
{
case 1:
{
lean_object* v_pre_252_; 
lean_inc_ref(v_pre_251_);
v_pre_252_ = lean_ctor_get(v_pre_251_, 0);
lean_inc(v_pre_252_);
if (lean_obj_tag(v_pre_252_) == 1)
{
lean_object* v_pre_253_; 
v_pre_253_ = lean_ctor_get(v_pre_252_, 0);
lean_inc(v_pre_253_);
if (lean_obj_tag(v_pre_253_) == 1)
{
lean_object* v_pre_254_; 
v_pre_254_ = lean_ctor_get(v_pre_253_, 0);
if (lean_obj_tag(v_pre_254_) == 0)
{
lean_object* v_str_255_; lean_object* v_str_256_; lean_object* v_str_257_; lean_object* v_str_258_; lean_object* v___x_259_; uint8_t v___x_260_; 
v_str_255_ = lean_ctor_get(v_kind_227_, 1);
lean_inc_ref(v_str_255_);
lean_dec_ref_known(v_kind_227_, 2);
v_str_256_ = lean_ctor_get(v_pre_251_, 1);
lean_inc_ref(v_str_256_);
lean_dec_ref_known(v_pre_251_, 2);
v_str_257_ = lean_ctor_get(v_pre_252_, 1);
lean_inc_ref(v_str_257_);
lean_dec_ref_known(v_pre_252_, 2);
v_str_258_ = lean_ctor_get(v_pre_253_, 1);
lean_inc_ref(v_str_258_);
lean_dec_ref_known(v_pre_253_, 2);
v___x_259_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__11));
v___x_260_ = lean_string_dec_eq(v_str_258_, v___x_259_);
lean_dec_ref(v_str_258_);
if (v___x_260_ == 0)
{
lean_dec_ref(v_str_257_);
lean_dec_ref(v_str_256_);
lean_dec_ref(v_str_255_);
lean_del_object(v___x_209_);
v___y_233_ = v_a_165_;
v___y_234_ = v_a_166_;
v___y_235_ = v_a_167_;
v___y_236_ = v_a_168_;
v___y_237_ = v_a_169_;
v___y_238_ = v_a_170_;
v___y_239_ = v_a_171_;
v___y_240_ = v_a_172_;
goto v___jp_232_;
}
else
{
lean_object* v___x_261_; uint8_t v___x_262_; 
v___x_261_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__12));
v___x_262_ = lean_string_dec_eq(v_str_257_, v___x_261_);
lean_dec_ref(v_str_257_);
if (v___x_262_ == 0)
{
lean_dec_ref(v_str_256_);
lean_dec_ref(v_str_255_);
lean_del_object(v___x_209_);
v___y_233_ = v_a_165_;
v___y_234_ = v_a_166_;
v___y_235_ = v_a_167_;
v___y_236_ = v_a_168_;
v___y_237_ = v_a_169_;
v___y_238_ = v_a_170_;
v___y_239_ = v_a_171_;
v___y_240_ = v_a_172_;
goto v___jp_232_;
}
else
{
lean_object* v___x_263_; uint8_t v___x_264_; 
v___x_263_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__6_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4_));
v___x_264_ = lean_string_dec_eq(v_str_256_, v___x_263_);
lean_dec_ref(v_str_256_);
if (v___x_264_ == 0)
{
lean_dec_ref(v_str_255_);
lean_del_object(v___x_209_);
v___y_233_ = v_a_165_;
v___y_234_ = v_a_166_;
v___y_235_ = v_a_167_;
v___y_236_ = v_a_168_;
v___y_237_ = v_a_169_;
v___y_238_ = v_a_170_;
v___y_239_ = v_a_171_;
v___y_240_ = v_a_172_;
goto v___jp_232_;
}
else
{
lean_object* v___x_265_; uint8_t v___x_266_; 
v___x_265_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__13));
v___x_266_ = lean_string_dec_eq(v_str_255_, v___x_265_);
lean_dec_ref(v_str_255_);
if (v___x_266_ == 0)
{
lean_del_object(v___x_209_);
v___y_233_ = v_a_165_;
v___y_234_ = v_a_166_;
v___y_235_ = v_a_167_;
v___y_236_ = v_a_168_;
v___y_237_ = v_a_169_;
v___y_238_ = v_a_170_;
v___y_239_ = v_a_171_;
v___y_240_ = v_a_172_;
goto v___jp_232_;
}
else
{
lean_object* v___x_268_; 
lean_del_object(v___x_230_);
lean_dec(v_tac_164_);
if (v_isShared_210_ == 0)
{
lean_ctor_set(v___x_209_, 0, v_a_228_);
v___x_268_ = v___x_209_;
goto v_reusejp_267_;
}
else
{
lean_object* v_reuseFailAlloc_269_; 
v_reuseFailAlloc_269_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_269_, 0, v_a_228_);
v___x_268_ = v_reuseFailAlloc_269_;
goto v_reusejp_267_;
}
v_reusejp_267_:
{
return v___x_268_;
}
}
}
}
}
}
else
{
lean_dec_ref_known(v_pre_253_, 2);
lean_dec_ref_known(v_pre_252_, 2);
lean_dec_ref_known(v_pre_251_, 2);
lean_dec_ref_known(v_kind_227_, 2);
lean_del_object(v___x_209_);
v___y_233_ = v_a_165_;
v___y_234_ = v_a_166_;
v___y_235_ = v_a_167_;
v___y_236_ = v_a_168_;
v___y_237_ = v_a_169_;
v___y_238_ = v_a_170_;
v___y_239_ = v_a_171_;
v___y_240_ = v_a_172_;
goto v___jp_232_;
}
}
else
{
lean_dec(v_pre_253_);
lean_dec_ref_known(v_pre_252_, 2);
lean_dec_ref_known(v_pre_251_, 2);
lean_dec_ref_known(v_kind_227_, 2);
lean_del_object(v___x_209_);
v___y_233_ = v_a_165_;
v___y_234_ = v_a_166_;
v___y_235_ = v_a_167_;
v___y_236_ = v_a_168_;
v___y_237_ = v_a_169_;
v___y_238_ = v_a_170_;
v___y_239_ = v_a_171_;
v___y_240_ = v_a_172_;
goto v___jp_232_;
}
}
else
{
lean_dec(v_pre_252_);
lean_dec_ref_known(v_pre_251_, 2);
lean_dec_ref_known(v_kind_227_, 2);
lean_del_object(v___x_209_);
v___y_233_ = v_a_165_;
v___y_234_ = v_a_166_;
v___y_235_ = v_a_167_;
v___y_236_ = v_a_168_;
v___y_237_ = v_a_169_;
v___y_238_ = v_a_170_;
v___y_239_ = v_a_171_;
v___y_240_ = v_a_172_;
goto v___jp_232_;
}
}
case 0:
{
lean_object* v_str_270_; lean_object* v___x_271_; uint8_t v___x_272_; 
v_str_270_ = lean_ctor_get(v_kind_227_, 1);
lean_inc_ref(v_str_270_);
lean_dec_ref_known(v_kind_227_, 2);
v___x_271_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__14));
v___x_272_ = lean_string_dec_eq(v_str_270_, v___x_271_);
lean_dec_ref(v_str_270_);
if (v___x_272_ == 0)
{
lean_del_object(v___x_209_);
v___y_233_ = v_a_165_;
v___y_234_ = v_a_166_;
v___y_235_ = v_a_167_;
v___y_236_ = v_a_168_;
v___y_237_ = v_a_169_;
v___y_238_ = v_a_170_;
v___y_239_ = v_a_171_;
v___y_240_ = v_a_172_;
goto v___jp_232_;
}
else
{
lean_object* v_ref_273_; uint8_t v___x_274_; lean_object* v___x_275_; lean_object* v___x_276_; lean_object* v___x_277_; lean_object* v___x_278_; lean_object* v___x_279_; lean_object* v___x_280_; lean_object* v___x_281_; lean_object* v___x_283_; 
lean_del_object(v___x_230_);
lean_dec(v_tac_164_);
v_ref_273_ = lean_ctor_get(v_a_171_, 5);
v___x_274_ = 0;
v___x_275_ = l_Lean_SourceInfo_fromRef(v_ref_273_, v___x_274_);
v___x_276_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__15));
v___x_277_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__17));
v___x_278_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__19));
lean_inc_n(v___x_275_, 2);
v___x_279_ = l_Lean_Syntax_node1(v___x_275_, v___x_278_, v_a_228_);
v___x_280_ = l_Lean_Syntax_node1(v___x_275_, v___x_277_, v___x_279_);
v___x_281_ = l_Lean_Syntax_node1(v___x_275_, v___x_276_, v___x_280_);
if (v_isShared_210_ == 0)
{
lean_ctor_set(v___x_209_, 0, v___x_281_);
v___x_283_ = v___x_209_;
goto v_reusejp_282_;
}
else
{
lean_object* v_reuseFailAlloc_284_; 
v_reuseFailAlloc_284_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_284_, 0, v___x_281_);
v___x_283_ = v_reuseFailAlloc_284_;
goto v_reusejp_282_;
}
v_reusejp_282_:
{
return v___x_283_;
}
}
}
default: 
{
lean_dec_ref_known(v_kind_227_, 2);
lean_del_object(v___x_209_);
v___y_233_ = v_a_165_;
v___y_234_ = v_a_166_;
v___y_235_ = v_a_167_;
v___y_236_ = v_a_168_;
v___y_237_ = v_a_169_;
v___y_238_ = v_a_170_;
v___y_239_ = v_a_171_;
v___y_240_ = v_a_172_;
goto v___jp_232_;
}
}
}
else
{
lean_dec(v_kind_227_);
lean_del_object(v___x_209_);
v___y_233_ = v_a_165_;
v___y_234_ = v_a_166_;
v___y_235_ = v_a_167_;
v___y_236_ = v_a_168_;
v___y_237_ = v_a_169_;
v___y_238_ = v_a_170_;
v___y_239_ = v_a_171_;
v___y_240_ = v_a_172_;
goto v___jp_232_;
}
v___jp_232_:
{
lean_object* v___x_241_; lean_object* v___x_242_; lean_object* v___x_244_; 
v___x_241_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__6, &lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__6);
v___x_242_ = l_Lean_MessageData_ofSyntax(v_tac_164_);
if (v_isShared_231_ == 0)
{
lean_ctor_set_tag(v___x_230_, 7);
lean_ctor_set(v___x_230_, 1, v___x_242_);
lean_ctor_set(v___x_230_, 0, v___x_241_);
v___x_244_ = v___x_230_;
goto v_reusejp_243_;
}
else
{
lean_object* v_reuseFailAlloc_250_; 
v_reuseFailAlloc_250_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_250_, 0, v___x_241_);
lean_ctor_set(v_reuseFailAlloc_250_, 1, v___x_242_);
v___x_244_ = v_reuseFailAlloc_250_;
goto v_reusejp_243_;
}
v_reusejp_243_:
{
lean_object* v___x_245_; lean_object* v___x_246_; lean_object* v___x_247_; lean_object* v___x_248_; lean_object* v___x_249_; 
v___x_245_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__10, &lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__10_once, _init_lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__10);
v___x_246_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_246_, 0, v___x_244_);
lean_ctor_set(v___x_246_, 1, v___x_245_);
v___x_247_ = l_Lean_MessageData_ofSyntax(v_a_228_);
v___x_248_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_248_, 0, v___x_246_);
lean_ctor_set(v___x_248_, 1, v___x_247_);
v___x_249_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Says_evalTacticCapturingTryThis_spec__0___redArg(v___x_248_, v___y_237_, v___y_238_, v___y_239_, v___y_240_);
return v___x_249_;
}
}
}
}
else
{
lean_object* v_a_286_; 
lean_del_object(v___x_209_);
lean_dec(v_tac_164_);
v_a_286_ = lean_ctor_get(v_suggestion_226_, 0);
lean_inc_ref(v_a_286_);
lean_dec_ref_known(v_suggestion_226_, 1);
v_a_175_ = v_a_286_;
v___y_176_ = v_a_165_;
v___y_177_ = v_a_166_;
v___y_178_ = v_a_167_;
v___y_179_ = v_a_168_;
v___y_180_ = v_a_169_;
v___y_181_ = v_a_170_;
v___y_182_ = v_a_171_;
v___y_183_ = v_a_172_;
goto v___jp_174_;
}
}
}
}
}
else
{
lean_object* v_a_288_; lean_object* v___x_290_; uint8_t v_isShared_291_; uint8_t v_isSharedCheck_295_; 
lean_dec(v_tac_164_);
v_a_288_ = lean_ctor_get(v___x_206_, 0);
v_isSharedCheck_295_ = !lean_is_exclusive(v___x_206_);
if (v_isSharedCheck_295_ == 0)
{
v___x_290_ = v___x_206_;
v_isShared_291_ = v_isSharedCheck_295_;
goto v_resetjp_289_;
}
else
{
lean_inc(v_a_288_);
lean_dec(v___x_206_);
v___x_290_ = lean_box(0);
v_isShared_291_ = v_isSharedCheck_295_;
goto v_resetjp_289_;
}
v_resetjp_289_:
{
lean_object* v___x_293_; 
if (v_isShared_291_ == 0)
{
v___x_293_ = v___x_290_;
goto v_reusejp_292_;
}
else
{
lean_object* v_reuseFailAlloc_294_; 
v_reuseFailAlloc_294_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_294_, 0, v_a_288_);
v___x_293_ = v_reuseFailAlloc_294_;
goto v_reusejp_292_;
}
v_reusejp_292_:
{
return v___x_293_;
}
}
}
v___jp_174_:
{
lean_object* v___x_184_; lean_object* v_env_185_; lean_object* v___x_186_; lean_object* v___x_187_; 
v___x_184_ = lean_st_ref_get(v___y_183_);
v_env_185_ = lean_ctor_get(v___x_184_, 0);
lean_inc_ref(v_env_185_);
lean_dec(v___x_184_);
v___x_186_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__0));
lean_inc_ref(v_a_175_);
v___x_187_ = lp_mathlib_Mathlib_GuardExceptions_parseAsTacticSeq(v_env_185_, v_a_175_, v___x_186_);
if (lean_obj_tag(v___x_187_) == 0)
{
lean_object* v_a_188_; lean_object* v___x_189_; lean_object* v___x_190_; lean_object* v___x_191_; lean_object* v___x_192_; lean_object* v___x_193_; lean_object* v___x_194_; lean_object* v___x_195_; lean_object* v___x_196_; 
v_a_188_ = lean_ctor_get(v___x_187_, 0);
lean_inc(v_a_188_);
lean_dec_ref_known(v___x_187_, 1);
v___x_189_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__2, &lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__2);
v___x_190_ = l_Lean_stringToMessageData(v_a_175_);
v___x_191_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_191_, 0, v___x_189_);
lean_ctor_set(v___x_191_, 1, v___x_190_);
v___x_192_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__4, &lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__4);
v___x_193_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_193_, 0, v___x_191_);
lean_ctor_set(v___x_193_, 1, v___x_192_);
v___x_194_ = l_Lean_stringToMessageData(v_a_188_);
v___x_195_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_195_, 0, v___x_193_);
lean_ctor_set(v___x_195_, 1, v___x_194_);
v___x_196_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Says_evalTacticCapturingTryThis_spec__0___redArg(v___x_195_, v___y_180_, v___y_181_, v___y_182_, v___y_183_);
return v___x_196_;
}
else
{
lean_object* v_a_197_; lean_object* v___x_199_; uint8_t v_isShared_200_; uint8_t v_isSharedCheck_204_; 
lean_dec_ref(v_a_175_);
v_a_197_ = lean_ctor_get(v___x_187_, 0);
v_isSharedCheck_204_ = !lean_is_exclusive(v___x_187_);
if (v_isSharedCheck_204_ == 0)
{
v___x_199_ = v___x_187_;
v_isShared_200_ = v_isSharedCheck_204_;
goto v_resetjp_198_;
}
else
{
lean_inc(v_a_197_);
lean_dec(v___x_187_);
v___x_199_ = lean_box(0);
v_isShared_200_ = v_isSharedCheck_204_;
goto v_resetjp_198_;
}
v_resetjp_198_:
{
lean_object* v___x_202_; 
if (v_isShared_200_ == 0)
{
lean_ctor_set_tag(v___x_199_, 0);
v___x_202_ = v___x_199_;
goto v_reusejp_201_;
}
else
{
lean_object* v_reuseFailAlloc_203_; 
v_reuseFailAlloc_203_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_203_, 0, v_a_197_);
v___x_202_ = v_reuseFailAlloc_203_;
goto v_reusejp_201_;
}
v_reusejp_201_:
{
return v___x_202_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___boxed(lean_object* v_tac_296_, lean_object* v_a_297_, lean_object* v_a_298_, lean_object* v_a_299_, lean_object* v_a_300_, lean_object* v_a_301_, lean_object* v_a_302_, lean_object* v_a_303_, lean_object* v_a_304_, lean_object* v_a_305_){
_start:
{
lean_object* v_res_306_; 
v_res_306_ = lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis(v_tac_296_, v_a_297_, v_a_298_, v_a_299_, v_a_300_, v_a_301_, v_a_302_, v_a_303_, v_a_304_);
lean_dec(v_a_304_);
lean_dec_ref(v_a_303_);
lean_dec(v_a_302_);
lean_dec_ref(v_a_301_);
lean_dec(v_a_300_);
lean_dec_ref(v_a_299_);
lean_dec(v_a_298_);
lean_dec_ref(v_a_297_);
return v_res_306_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Says_evalTacticCapturingTryThis_spec__0(lean_object* v_00_u03b1_307_, lean_object* v_msg_308_, lean_object* v___y_309_, lean_object* v___y_310_, lean_object* v___y_311_, lean_object* v___y_312_, lean_object* v___y_313_, lean_object* v___y_314_, lean_object* v___y_315_, lean_object* v___y_316_){
_start:
{
lean_object* v___x_318_; 
v___x_318_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Says_evalTacticCapturingTryThis_spec__0___redArg(v_msg_308_, v___y_313_, v___y_314_, v___y_315_, v___y_316_);
return v___x_318_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Says_evalTacticCapturingTryThis_spec__0___boxed(lean_object* v_00_u03b1_319_, lean_object* v_msg_320_, lean_object* v___y_321_, lean_object* v___y_322_, lean_object* v___y_323_, lean_object* v___y_324_, lean_object* v___y_325_, lean_object* v___y_326_, lean_object* v___y_327_, lean_object* v___y_328_, lean_object* v___y_329_){
_start:
{
lean_object* v_res_330_; 
v_res_330_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Says_evalTacticCapturingTryThis_spec__0(v_00_u03b1_319_, v_msg_320_, v___y_321_, v___y_322_, v___y_323_, v___y_324_, v___y_325_, v___y_326_, v___y_327_, v___y_328_);
lean_dec(v___y_328_);
lean_dec_ref(v___y_327_);
lean_dec(v___y_326_);
lean_dec_ref(v___y_325_);
lean_dec(v___y_324_);
lean_dec_ref(v___y_323_);
lean_dec(v___y_322_);
lean_dec_ref(v___y_321_);
return v_res_330_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_371_; lean_object* v___x_372_; lean_object* v___x_373_; 
v___x_371_ = lean_box(0);
v___x_372_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_373_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_373_, 0, v___x_372_);
lean_ctor_set(v___x_373_, 1, v___x_371_);
return v___x_373_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1_spec__0___redArg(){
_start:
{
lean_object* v___x_375_; lean_object* v___x_376_; 
v___x_375_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1_spec__0___redArg___closed__0);
v___x_376_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_376_, 0, v___x_375_);
return v___x_376_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1_spec__0___redArg___boxed(lean_object* v___y_377_){
_start:
{
lean_object* v_res_378_; 
v_res_378_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1_spec__0___redArg();
return v_res_378_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1_spec__0(lean_object* v_00_u03b1_379_, lean_object* v___y_380_, lean_object* v___y_381_, lean_object* v___y_382_, lean_object* v___y_383_, lean_object* v___y_384_, lean_object* v___y_385_, lean_object* v___y_386_, lean_object* v___y_387_){
_start:
{
lean_object* v___x_389_; 
v___x_389_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1_spec__0___redArg();
return v___x_389_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1_spec__0___boxed(lean_object* v_00_u03b1_390_, lean_object* v___y_391_, lean_object* v___y_392_, lean_object* v___y_393_, lean_object* v___y_394_, lean_object* v___y_395_, lean_object* v___y_396_, lean_object* v___y_397_, lean_object* v___y_398_, lean_object* v___y_399_){
_start:
{
lean_object* v_res_400_; 
v_res_400_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1_spec__0(v_00_u03b1_390_, v___y_391_, v___y_392_, v___y_393_, v___y_394_, v___y_395_, v___y_396_, v___y_397_, v___y_398_);
lean_dec(v___y_398_);
lean_dec_ref(v___y_397_);
lean_dec(v___y_396_);
lean_dec_ref(v___y_395_);
lean_dec(v___y_394_);
lean_dec_ref(v___y_393_);
lean_dec(v___y_392_);
lean_dec_ref(v___y_391_);
return v_res_400_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1_spec__1(lean_object* v_opts_401_, lean_object* v_opt_402_){
_start:
{
lean_object* v_name_403_; lean_object* v_defValue_404_; lean_object* v_map_405_; lean_object* v___x_406_; 
v_name_403_ = lean_ctor_get(v_opt_402_, 0);
v_defValue_404_ = lean_ctor_get(v_opt_402_, 1);
v_map_405_ = lean_ctor_get(v_opts_401_, 0);
v___x_406_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_405_, v_name_403_);
if (lean_obj_tag(v___x_406_) == 0)
{
uint8_t v___x_407_; 
v___x_407_ = lean_unbox(v_defValue_404_);
return v___x_407_;
}
else
{
lean_object* v_val_408_; 
v_val_408_ = lean_ctor_get(v___x_406_, 0);
lean_inc(v_val_408_);
lean_dec_ref_known(v___x_406_, 1);
if (lean_obj_tag(v_val_408_) == 1)
{
uint8_t v_v_409_; 
v_v_409_ = lean_ctor_get_uint8(v_val_408_, 0);
lean_dec_ref_known(v_val_408_, 0);
return v_v_409_;
}
else
{
uint8_t v___x_410_; 
lean_dec(v_val_408_);
v___x_410_ = lean_unbox(v_defValue_404_);
return v___x_410_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1_spec__1___boxed(lean_object* v_opts_411_, lean_object* v_opt_412_){
_start:
{
uint8_t v_res_413_; lean_object* v_r_414_; 
v_res_413_ = lp_mathlib_Lean_Option_get___at___00Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1_spec__1(v_opts_411_, v_opt_412_);
lean_dec_ref(v_opt_412_);
lean_dec_ref(v_opts_411_);
v_r_414_ = lean_box(v_res_413_);
return v_r_414_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__0(void){
_start:
{
lean_object* v___x_415_; 
v___x_415_ = l_Array_mkArray0(lean_box(0));
return v___x_415_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__3(void){
_start:
{
lean_object* v___x_418_; lean_object* v___x_419_; 
v___x_418_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__2));
v___x_419_ = l_Lean_stringToMessageData(v___x_418_);
return v___x_419_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__5(void){
_start:
{
lean_object* v___x_421_; lean_object* v___x_422_; 
v___x_421_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__4));
v___x_422_ = l_Lean_stringToMessageData(v___x_421_);
return v___x_422_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__7(void){
_start:
{
lean_object* v___x_424_; lean_object* v___x_425_; 
v___x_424_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__6));
v___x_425_ = l_Lean_stringToMessageData(v___x_424_);
return v___x_425_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__9(void){
_start:
{
lean_object* v___x_427_; lean_object* v___x_428_; 
v___x_427_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__8));
v___x_428_ = l_Lean_stringToMessageData(v___x_427_);
return v___x_428_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1(lean_object* v_x_432_, lean_object* v_a_433_, lean_object* v_a_434_, lean_object* v_a_435_, lean_object* v_a_436_, lean_object* v_a_437_, lean_object* v_a_438_, lean_object* v_a_439_, lean_object* v_a_440_){
_start:
{
lean_object* v___x_442_; lean_object* v___x_443_; uint8_t v___x_444_; 
v___x_442_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn___closed__0_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4_));
v___x_443_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Says_says___closed__0));
lean_inc(v_x_432_);
v___x_444_ = l_Lean_Syntax_isOfKind(v_x_432_, v___x_443_);
if (v___x_444_ == 0)
{
lean_object* v___x_445_; 
lean_dec(v_x_432_);
v___x_445_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1_spec__0___redArg();
return v___x_445_;
}
else
{
lean_object* v___x_446_; lean_object* v_tac_447_; lean_object* v___x_448_; lean_object* v_tk_449_; lean_object* v___y_451_; lean_object* v___y_452_; lean_object* v___y_453_; lean_object* v___y_454_; lean_object* v___y_455_; lean_object* v___y_456_; lean_object* v___y_457_; lean_object* v___y_458_; lean_object* v___y_459_; lean_object* v___y_460_; uint8_t v___y_461_; lean_object* v___y_554_; lean_object* v___y_555_; lean_object* v___y_556_; lean_object* v___y_557_; lean_object* v___y_558_; lean_object* v___y_559_; lean_object* v___y_560_; lean_object* v___y_561_; uint8_t v___y_562_; lean_object* v___y_563_; lean_object* v___y_564_; lean_object* v___y_565_; uint8_t v___y_566_; lean_object* v_result_568_; lean_object* v___y_569_; lean_object* v___y_570_; lean_object* v___y_571_; lean_object* v___y_572_; lean_object* v___y_573_; lean_object* v___y_574_; lean_object* v___y_575_; lean_object* v___y_576_; lean_object* v___x_585_; lean_object* v___x_586_; uint8_t v___x_587_; 
v___x_446_ = lean_unsigned_to_nat(0u);
v_tac_447_ = l_Lean_Syntax_getArg(v_x_432_, v___x_446_);
v___x_448_ = lean_unsigned_to_nat(1u);
v_tk_449_ = l_Lean_Syntax_getArg(v_x_432_, v___x_448_);
v___x_585_ = lean_unsigned_to_nat(2u);
v___x_586_ = l_Lean_Syntax_getArg(v_x_432_, v___x_585_);
lean_dec(v_x_432_);
v___x_587_ = l_Lean_Syntax_isNone(v___x_586_);
if (v___x_587_ == 0)
{
uint8_t v___x_588_; 
lean_inc(v___x_586_);
v___x_588_ = l_Lean_Syntax_matchesNull(v___x_586_, v___x_448_);
if (v___x_588_ == 0)
{
lean_object* v___x_589_; 
lean_dec(v___x_586_);
lean_dec(v_tk_449_);
lean_dec(v_tac_447_);
v___x_589_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1_spec__0___redArg();
return v___x_589_;
}
else
{
lean_object* v_result_590_; lean_object* v___x_591_; uint8_t v___x_592_; 
v_result_590_ = l_Lean_Syntax_getArg(v___x_586_, v___x_446_);
lean_dec(v___x_586_);
v___x_591_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__15));
lean_inc(v_result_590_);
v___x_592_ = l_Lean_Syntax_isOfKind(v_result_590_, v___x_591_);
if (v___x_592_ == 0)
{
lean_object* v___x_593_; 
lean_dec(v_result_590_);
lean_dec(v_tk_449_);
lean_dec(v_tac_447_);
v___x_593_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1_spec__0___redArg();
return v___x_593_;
}
else
{
lean_object* v___x_594_; 
v___x_594_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_594_, 0, v_result_590_);
v_result_568_ = v___x_594_;
v___y_569_ = v_a_433_;
v___y_570_ = v_a_434_;
v___y_571_ = v_a_435_;
v___y_572_ = v_a_436_;
v___y_573_ = v_a_437_;
v___y_574_ = v_a_438_;
v___y_575_ = v_a_439_;
v___y_576_ = v_a_440_;
goto v___jp_567_;
}
}
}
else
{
lean_object* v___x_595_; 
lean_dec(v___x_586_);
v___x_595_ = lean_box(0);
v_result_568_ = v___x_595_;
v___y_569_ = v_a_433_;
v___y_570_ = v_a_434_;
v___y_571_ = v_a_435_;
v___y_572_ = v_a_436_;
v___y_573_ = v_a_437_;
v___y_574_ = v_a_438_;
v___y_575_ = v_a_439_;
v___y_576_ = v_a_440_;
goto v___jp_567_;
}
v___jp_450_:
{
if (lean_obj_tag(v___y_456_) == 0)
{
lean_object* v___x_462_; 
lean_inc(v_tac_447_);
v___x_462_ = lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis(v_tac_447_, v___y_455_, v___y_459_, v___y_457_, v___y_453_, v___y_452_, v___y_454_, v___y_458_, v___y_460_);
if (lean_obj_tag(v___x_462_) == 0)
{
lean_object* v_a_463_; lean_object* v_ref_464_; uint8_t v___x_465_; lean_object* v___x_466_; lean_object* v___x_467_; lean_object* v___x_468_; lean_object* v___x_469_; lean_object* v___x_470_; lean_object* v___x_471_; lean_object* v___x_472_; lean_object* v___x_473_; lean_object* v___x_474_; lean_object* v___x_475_; lean_object* v___x_476_; lean_object* v___x_477_; lean_object* v___x_478_; uint8_t v___x_479_; lean_object* v___x_480_; lean_object* v___x_481_; 
v_a_463_ = lean_ctor_get(v___x_462_, 0);
lean_inc(v_a_463_);
lean_dec_ref_known(v___x_462_, 1);
v_ref_464_ = lean_ctor_get(v___y_458_, 5);
v___x_465_ = 0;
v___x_466_ = l_Lean_SourceInfo_fromRef(v_ref_464_, v___x_465_);
lean_inc_n(v___x_466_, 4);
v___x_467_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_467_, 0, v___x_466_);
lean_ctor_set(v___x_467_, 1, v___x_442_);
v___x_468_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__19));
v___x_469_ = l_Lean_Syntax_node1(v___x_466_, v___x_468_, v_a_463_);
lean_inc_ref(v___x_467_);
lean_inc(v_tac_447_);
v___x_470_ = l_Lean_Syntax_node3(v___x_466_, v___x_443_, v_tac_447_, v___x_467_, v___x_469_);
v___x_471_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__0, &lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__0);
v___x_472_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_472_, 0, v___x_466_);
lean_ctor_set(v___x_472_, 1, v___x_468_);
lean_ctor_set(v___x_472_, 2, v___x_471_);
v___x_473_ = l_Lean_Syntax_node3(v___x_466_, v___x_443_, v_tac_447_, v___x_467_, v___x_472_);
lean_inc(v___y_451_);
v___x_474_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_474_, 0, v___y_451_);
lean_ctor_set(v___x_474_, 1, v___x_470_);
v___x_475_ = lean_box(0);
v___x_476_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_476_, 0, v___x_474_);
lean_ctor_set(v___x_476_, 1, v___x_475_);
lean_ctor_set(v___x_476_, 2, v___x_475_);
lean_ctor_set(v___x_476_, 3, v___x_475_);
lean_ctor_set(v___x_476_, 4, v___x_475_);
lean_ctor_set(v___x_476_, 5, v___x_475_);
v___x_477_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_477_, 0, v___x_473_);
v___x_478_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__1));
v___x_479_ = 4;
v___x_480_ = l_Lean_MessageData_nil;
v___x_481_ = l_Lean_Meta_Tactic_TryThis_addSuggestion(v_tk_449_, v___x_476_, v___x_477_, v___x_478_, v___x_475_, v___x_479_, v___x_480_, v___y_458_, v___y_460_);
return v___x_481_;
}
else
{
lean_object* v_a_482_; lean_object* v___x_484_; uint8_t v_isShared_485_; uint8_t v_isSharedCheck_489_; 
lean_dec(v_tk_449_);
lean_dec(v_tac_447_);
v_a_482_ = lean_ctor_get(v___x_462_, 0);
v_isSharedCheck_489_ = !lean_is_exclusive(v___x_462_);
if (v_isSharedCheck_489_ == 0)
{
v___x_484_ = v___x_462_;
v_isShared_485_ = v_isSharedCheck_489_;
goto v_resetjp_483_;
}
else
{
lean_inc(v_a_482_);
lean_dec(v___x_462_);
v___x_484_ = lean_box(0);
v_isShared_485_ = v_isSharedCheck_489_;
goto v_resetjp_483_;
}
v_resetjp_483_:
{
lean_object* v___x_487_; 
if (v_isShared_485_ == 0)
{
v___x_487_ = v___x_484_;
goto v_reusejp_486_;
}
else
{
lean_object* v_reuseFailAlloc_488_; 
v_reuseFailAlloc_488_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_488_, 0, v_a_482_);
v___x_487_ = v_reuseFailAlloc_488_;
goto v_reusejp_486_;
}
v_reusejp_486_:
{
return v___x_487_;
}
}
}
}
else
{
lean_dec(v_tk_449_);
if (v___y_461_ == 0)
{
lean_object* v_val_490_; lean_object* v___x_491_; 
lean_dec(v_tac_447_);
v_val_490_ = lean_ctor_get(v___y_456_, 0);
lean_inc(v_val_490_);
lean_dec_ref_known(v___y_456_, 1);
v___x_491_ = l_Lean_Elab_Tactic_evalTactic(v_val_490_, v___y_455_, v___y_459_, v___y_457_, v___y_453_, v___y_452_, v___y_454_, v___y_458_, v___y_460_);
return v___x_491_;
}
else
{
lean_object* v_val_492_; lean_object* v___x_493_; 
v_val_492_ = lean_ctor_get(v___y_456_, 0);
lean_inc(v_val_492_);
lean_dec_ref_known(v___y_456_, 1);
lean_inc(v_tac_447_);
v___x_493_ = lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis(v_tac_447_, v___y_455_, v___y_459_, v___y_457_, v___y_453_, v___y_452_, v___y_454_, v___y_458_, v___y_460_);
if (lean_obj_tag(v___x_493_) == 0)
{
lean_object* v_a_494_; lean_object* v___x_495_; lean_object* v___x_496_; 
v_a_494_ = lean_ctor_get(v___x_493_, 0);
lean_inc(v_a_494_);
lean_dec_ref_known(v___x_493_, 1);
v___x_495_ = lp_Qq_Lean_Syntax_stripPos(v_a_494_);
v___x_496_ = l_Lean_PrettyPrinter_ppTactic(v___x_495_, v___y_458_, v___y_460_);
if (lean_obj_tag(v___x_496_) == 0)
{
lean_object* v_a_497_; lean_object* v___x_498_; lean_object* v___x_499_; lean_object* v___x_500_; lean_object* v___x_501_; 
v_a_497_ = lean_ctor_get(v___x_496_, 0);
lean_inc(v_a_497_);
lean_dec_ref_known(v___x_496_, 1);
v___x_498_ = l_Std_Format_defWidth;
v___x_499_ = l_Std_Format_pretty(v_a_497_, v___x_498_, v___x_446_, v___x_446_);
v___x_500_ = lp_Qq_Lean_Syntax_stripPos(v_val_492_);
v___x_501_ = l_Lean_PrettyPrinter_ppTactic(v___x_500_, v___y_458_, v___y_460_);
if (lean_obj_tag(v___x_501_) == 0)
{
lean_object* v_a_502_; lean_object* v___x_504_; uint8_t v_isShared_505_; uint8_t v_isSharedCheck_528_; 
v_a_502_ = lean_ctor_get(v___x_501_, 0);
v_isSharedCheck_528_ = !lean_is_exclusive(v___x_501_);
if (v_isSharedCheck_528_ == 0)
{
v___x_504_ = v___x_501_;
v_isShared_505_ = v_isSharedCheck_528_;
goto v_resetjp_503_;
}
else
{
lean_inc(v_a_502_);
lean_dec(v___x_501_);
v___x_504_ = lean_box(0);
v_isShared_505_ = v_isSharedCheck_528_;
goto v_resetjp_503_;
}
v_resetjp_503_:
{
lean_object* v___x_506_; uint8_t v___x_507_; 
v___x_506_ = l_Std_Format_pretty(v_a_502_, v___x_498_, v___x_446_, v___x_446_);
v___x_507_ = lean_string_dec_eq(v___x_499_, v___x_506_);
if (v___x_507_ == 0)
{
lean_object* v___x_508_; lean_object* v___x_509_; lean_object* v___x_510_; lean_object* v___x_511_; lean_object* v___x_512_; lean_object* v___x_513_; lean_object* v___x_514_; lean_object* v___x_515_; lean_object* v___x_516_; lean_object* v___x_517_; lean_object* v___x_518_; lean_object* v___x_519_; lean_object* v___x_520_; lean_object* v___x_521_; lean_object* v___x_522_; lean_object* v___x_523_; 
lean_del_object(v___x_504_);
v___x_508_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__6, &lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_Says_evalTacticCapturingTryThis___closed__6);
v___x_509_ = l_Lean_MessageData_ofSyntax(v_tac_447_);
v___x_510_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_510_, 0, v___x_508_);
lean_ctor_set(v___x_510_, 1, v___x_509_);
v___x_511_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__3, &lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__3);
v___x_512_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_512_, 0, v___x_510_);
lean_ctor_set(v___x_512_, 1, v___x_511_);
v___x_513_ = l_Lean_stringToMessageData(v___x_499_);
v___x_514_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_514_, 0, v___x_512_);
lean_ctor_set(v___x_514_, 1, v___x_513_);
v___x_515_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__5, &lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__5);
v___x_516_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_516_, 0, v___x_514_);
lean_ctor_set(v___x_516_, 1, v___x_515_);
v___x_517_ = l_Lean_stringToMessageData(v___x_506_);
v___x_518_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_518_, 0, v___x_516_);
lean_ctor_set(v___x_518_, 1, v___x_517_);
v___x_519_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__7, &lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__7);
v___x_520_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_520_, 0, v___x_518_);
lean_ctor_set(v___x_520_, 1, v___x_519_);
v___x_521_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__9, &lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__9_once, _init_lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__9);
v___x_522_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_522_, 0, v___x_520_);
lean_ctor_set(v___x_522_, 1, v___x_521_);
v___x_523_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Says_evalTacticCapturingTryThis_spec__0___redArg(v___x_522_, v___y_452_, v___y_454_, v___y_458_, v___y_460_);
return v___x_523_;
}
else
{
lean_object* v___x_524_; lean_object* v___x_526_; 
lean_dec_ref(v___x_506_);
lean_dec_ref(v___x_499_);
lean_dec(v_tac_447_);
v___x_524_ = lean_box(0);
if (v_isShared_505_ == 0)
{
lean_ctor_set(v___x_504_, 0, v___x_524_);
v___x_526_ = v___x_504_;
goto v_reusejp_525_;
}
else
{
lean_object* v_reuseFailAlloc_527_; 
v_reuseFailAlloc_527_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_527_, 0, v___x_524_);
v___x_526_ = v_reuseFailAlloc_527_;
goto v_reusejp_525_;
}
v_reusejp_525_:
{
return v___x_526_;
}
}
}
}
else
{
lean_object* v_a_529_; lean_object* v___x_531_; uint8_t v_isShared_532_; uint8_t v_isSharedCheck_536_; 
lean_dec_ref(v___x_499_);
lean_dec(v_tac_447_);
v_a_529_ = lean_ctor_get(v___x_501_, 0);
v_isSharedCheck_536_ = !lean_is_exclusive(v___x_501_);
if (v_isSharedCheck_536_ == 0)
{
v___x_531_ = v___x_501_;
v_isShared_532_ = v_isSharedCheck_536_;
goto v_resetjp_530_;
}
else
{
lean_inc(v_a_529_);
lean_dec(v___x_501_);
v___x_531_ = lean_box(0);
v_isShared_532_ = v_isSharedCheck_536_;
goto v_resetjp_530_;
}
v_resetjp_530_:
{
lean_object* v___x_534_; 
if (v_isShared_532_ == 0)
{
v___x_534_ = v___x_531_;
goto v_reusejp_533_;
}
else
{
lean_object* v_reuseFailAlloc_535_; 
v_reuseFailAlloc_535_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_535_, 0, v_a_529_);
v___x_534_ = v_reuseFailAlloc_535_;
goto v_reusejp_533_;
}
v_reusejp_533_:
{
return v___x_534_;
}
}
}
}
else
{
lean_object* v_a_537_; lean_object* v___x_539_; uint8_t v_isShared_540_; uint8_t v_isSharedCheck_544_; 
lean_dec(v_val_492_);
lean_dec(v_tac_447_);
v_a_537_ = lean_ctor_get(v___x_496_, 0);
v_isSharedCheck_544_ = !lean_is_exclusive(v___x_496_);
if (v_isSharedCheck_544_ == 0)
{
v___x_539_ = v___x_496_;
v_isShared_540_ = v_isSharedCheck_544_;
goto v_resetjp_538_;
}
else
{
lean_inc(v_a_537_);
lean_dec(v___x_496_);
v___x_539_ = lean_box(0);
v_isShared_540_ = v_isSharedCheck_544_;
goto v_resetjp_538_;
}
v_resetjp_538_:
{
lean_object* v___x_542_; 
if (v_isShared_540_ == 0)
{
v___x_542_ = v___x_539_;
goto v_reusejp_541_;
}
else
{
lean_object* v_reuseFailAlloc_543_; 
v_reuseFailAlloc_543_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_543_, 0, v_a_537_);
v___x_542_ = v_reuseFailAlloc_543_;
goto v_reusejp_541_;
}
v_reusejp_541_:
{
return v___x_542_;
}
}
}
}
else
{
lean_object* v_a_545_; lean_object* v___x_547_; uint8_t v_isShared_548_; uint8_t v_isSharedCheck_552_; 
lean_dec(v_val_492_);
lean_dec(v_tac_447_);
v_a_545_ = lean_ctor_get(v___x_493_, 0);
v_isSharedCheck_552_ = !lean_is_exclusive(v___x_493_);
if (v_isSharedCheck_552_ == 0)
{
v___x_547_ = v___x_493_;
v_isShared_548_ = v_isSharedCheck_552_;
goto v_resetjp_546_;
}
else
{
lean_inc(v_a_545_);
lean_dec(v___x_493_);
v___x_547_ = lean_box(0);
v_isShared_548_ = v_isSharedCheck_552_;
goto v_resetjp_546_;
}
v_resetjp_546_:
{
lean_object* v___x_550_; 
if (v_isShared_548_ == 0)
{
v___x_550_ = v___x_547_;
goto v_reusejp_549_;
}
else
{
lean_object* v_reuseFailAlloc_551_; 
v_reuseFailAlloc_551_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_551_, 0, v_a_545_);
v___x_550_ = v_reuseFailAlloc_551_;
goto v_reusejp_549_;
}
v_reusejp_549_:
{
return v___x_550_;
}
}
}
}
}
}
v___jp_553_:
{
if (lean_obj_tag(v___y_560_) == 0)
{
v___y_451_ = v___y_554_;
v___y_452_ = v___y_559_;
v___y_453_ = v___y_555_;
v___y_454_ = v___y_556_;
v___y_455_ = v___y_561_;
v___y_456_ = v___y_557_;
v___y_457_ = v___y_563_;
v___y_458_ = v___y_558_;
v___y_459_ = v___y_564_;
v___y_460_ = v___y_565_;
v___y_461_ = v___y_562_;
goto v___jp_450_;
}
else
{
lean_dec_ref_known(v___y_560_, 1);
v___y_451_ = v___y_554_;
v___y_452_ = v___y_559_;
v___y_453_ = v___y_555_;
v___y_454_ = v___y_556_;
v___y_455_ = v___y_561_;
v___y_456_ = v___y_557_;
v___y_457_ = v___y_563_;
v___y_458_ = v___y_558_;
v___y_459_ = v___y_564_;
v___y_460_ = v___y_565_;
v___y_461_ = v___y_566_;
goto v___jp_450_;
}
}
v___jp_567_:
{
lean_object* v___x_577_; lean_object* v___x_578_; lean_object* v_options_579_; lean_object* v___x_580_; lean_object* v___x_581_; uint8_t v___x_582_; 
v___x_577_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__10));
v___x_578_ = lean_io_getenv(v___x_577_);
v_options_579_ = lean_ctor_get(v___y_575_, 2);
v___x_580_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___closed__11));
v___x_581_ = lp_mathlib_Mathlib_Tactic_Says_says_verify;
v___x_582_ = lp_mathlib_Lean_Option_get___at___00Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1_spec__1(v_options_579_, v___x_581_);
if (v___x_582_ == 0)
{
lean_object* v___x_583_; uint8_t v___x_584_; 
v___x_583_ = lp_mathlib_Mathlib_Tactic_Says_says_no__verify__in__CI;
v___x_584_ = lp_mathlib_Lean_Option_get___at___00Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1_spec__1(v_options_579_, v___x_583_);
if (v___x_584_ == 0)
{
v___y_554_ = v___x_580_;
v___y_555_ = v___y_572_;
v___y_556_ = v___y_574_;
v___y_557_ = v_result_568_;
v___y_558_ = v___y_575_;
v___y_559_ = v___y_573_;
v___y_560_ = v___x_578_;
v___y_561_ = v___y_569_;
v___y_562_ = v___x_582_;
v___y_563_ = v___y_571_;
v___y_564_ = v___y_570_;
v___y_565_ = v___y_576_;
v___y_566_ = v___x_444_;
goto v___jp_553_;
}
else
{
if (v___x_582_ == 0)
{
lean_dec(v___x_578_);
v___y_451_ = v___x_580_;
v___y_452_ = v___y_573_;
v___y_453_ = v___y_572_;
v___y_454_ = v___y_574_;
v___y_455_ = v___y_569_;
v___y_456_ = v_result_568_;
v___y_457_ = v___y_571_;
v___y_458_ = v___y_575_;
v___y_459_ = v___y_570_;
v___y_460_ = v___y_576_;
v___y_461_ = v___x_582_;
goto v___jp_450_;
}
else
{
v___y_554_ = v___x_580_;
v___y_555_ = v___y_572_;
v___y_556_ = v___y_574_;
v___y_557_ = v_result_568_;
v___y_558_ = v___y_575_;
v___y_559_ = v___y_573_;
v___y_560_ = v___x_578_;
v___y_561_ = v___y_569_;
v___y_562_ = v___x_582_;
v___y_563_ = v___y_571_;
v___y_564_ = v___y_570_;
v___y_565_ = v___y_576_;
v___y_566_ = v___x_582_;
goto v___jp_553_;
}
}
}
else
{
lean_dec(v___x_578_);
v___y_451_ = v___x_580_;
v___y_452_ = v___y_573_;
v___y_453_ = v___y_572_;
v___y_454_ = v___y_574_;
v___y_455_ = v___y_569_;
v___y_456_ = v_result_568_;
v___y_457_ = v___y_571_;
v___y_458_ = v___y_575_;
v___y_459_ = v___y_570_;
v___y_460_ = v___y_576_;
v___y_461_ = v___x_444_;
goto v___jp_450_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1___boxed(lean_object* v_x_596_, lean_object* v_a_597_, lean_object* v_a_598_, lean_object* v_a_599_, lean_object* v_a_600_, lean_object* v_a_601_, lean_object* v_a_602_, lean_object* v_a_603_, lean_object* v_a_604_, lean_object* v_a_605_){
_start:
{
lean_object* v_res_606_; 
v_res_606_ = lp_mathlib_Mathlib_Tactic_Says___aux__Mathlib__Tactic__Says______elabRules__Mathlib__Tactic__Says__says__1(v_x_596_, v_a_597_, v_a_598_, v_a_599_, v_a_600_, v_a_601_, v_a_602_, v_a_603_, v_a_604_);
lean_dec(v_a_604_);
lean_dec_ref(v_a_603_);
lean_dec(v_a_602_);
lean_dec_ref(v_a_601_);
lean_dec(v_a_600_);
lean_dec_ref(v_a_599_);
lean_dec(v_a_598_);
lean_dec_ref(v_a_597_);
return v_res_606_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn_00___x40_Mathlib_Tactic_Says_31067479____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_608_; lean_object* v___x_609_; 
v___x_608_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Says_says___closed__0));
v___x_609_ = lp_batteries_Batteries_Linter_UnreachableTactic_addIgnoreTacticKind(v___x_608_);
return v___x_609_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn_00___x40_Mathlib_Tactic_Says_31067479____hygCtx___hyg_2____boxed(lean_object* v_a_610_){
_start:
{
lean_object* v_res_611_; 
v_res_611_ = lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn_00___x40_Mathlib_Tactic_Says_31067479____hygCtx___hyg_2_();
return v_res_611_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Linter_UnreachableTactic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Says(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Linter_UnreachableTactic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_TryThis(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Lean_Elab_InfoTree(uint8_t builtin);
lean_object* runtime_initialize_Qq_Qq_MatchImpl(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Says(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_TryThis(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Lean_Elab_InfoTree(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Qq_Qq_MatchImpl(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn_00___x40_Mathlib_Tactic_Says_1890619178____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Tactic_Says_says_verify = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Says_says_verify);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn_00___x40_Mathlib_Tactic_Says_1791366168____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Tactic_Says_says_no__verify__in__CI = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Says_says_no__verify__in__CI);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Says_0__Mathlib_Tactic_Says_initFn_00___x40_Mathlib_Tactic_Says_31067479____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_TryThis(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Lean_Elab_InfoTree(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Linter_UnreachableTactic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Basic(uint8_t builtin);
lean_object* initialize_Qq_Qq_MatchImpl(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Says(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_TryThis(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Lean_Elab_InfoTree(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Linter_UnreachableTactic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Qq_Qq_MatchImpl(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Says(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Says(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Says(builtin);
}
#ifdef __cplusplus
}
#endif
