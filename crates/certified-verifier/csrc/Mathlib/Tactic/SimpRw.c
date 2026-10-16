// Lean compiler output
// Module: Mathlib.Tactic.SimpRw
// Imports: public import Init public meta import Init public import Mathlib.Init
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
extern lean_object* l_Lean_Parser_Tactic_location;
lean_object* l_Lean_Name_mkStr1(lean_object*);
extern lean_object* l_Lean_Parser_Tactic_rwRuleSeq;
extern lean_object* l_Lean_Parser_Tactic_optConfig;
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_Parser_Tactic_getConfigItems(lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_evalTactic(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_mkInitialTacticInfo(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
lean_object* lean_array_get_size(lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_nat_shiftr(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isNone(lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_focus___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getOptional_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_withSimpRWRulesSeq___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_withSimpRWRulesSeq___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_withSimpRWRulesSeq___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_withSimpRWRulesSeq___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Elab_getResetInfoTrees___at___00Lean_Elab_withInfoTreeContext___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__0_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_getResetInfoTrees___at___00Lean_Elab_withInfoTreeContext___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__0_spec__0___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Lean_Elab_getResetInfoTrees___at___00Lean_Elab_withInfoTreeContext___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__0_spec__0___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_getResetInfoTrees___at___00Lean_Elab_withInfoTreeContext___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__0_spec__0___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getResetInfoTrees___at___00Lean_Elab_withInfoTreeContext___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__0_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getResetInfoTrees___at___00Lean_Elab_withInfoTreeContext___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withInfoTreeContext___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withInfoTreeContext___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__0___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withInfoTreeContext___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withInfoTreeContext___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__1___redArg___lam__1(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__1___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__1___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__1___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__1___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__1___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__1___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__1___redArg___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__1___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_withSimpRWRulesSeq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_withSimpRWRulesSeq___lam__1___boxed, .m_arity = 10, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Mathlib_Tactic_withSimpRWRulesSeq___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_withSimpRWRulesSeq___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_withSimpRWRulesSeq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_withSimpRWRulesSeq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getResetInfoTrees___at___00Lean_Elab_withInfoTreeContext___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getResetInfoTrees___at___00Lean_Elab_withInfoTreeContext___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withInfoTreeContext___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withInfoTreeContext___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "tacticSimp_rw___"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(96, 239, 0, 73, 66, 75, 163, 56)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "simp_rw "};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__7_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__8;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__9;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__10_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__11_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__12;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__13;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__14;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_tacticSimp__rw______;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "simp"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__1___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__1___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "only"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__1___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__1___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "simpLemma"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__1___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__1___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "←"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__1___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__1___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1_spec__1(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1_spec__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "configItem"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "valConfigItem"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "failIfUnchanged"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__4;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__3_value),LEAN_SCALAR_PTR_LITERAL(6, 104, 167, 161, 191, 186, 8, 81)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ":="};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "false"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__7_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__8;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__7_value),LEAN_SCALAR_PTR_LITERAL(160, 214, 196, 140, 104, 187, 164, 111)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Bool"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__10_value),LEAN_SCALAR_PTR_LITERAL(250, 44, 198, 216, 184, 195, 199, 178)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__11_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__7_value),LEAN_SCALAR_PTR_LITERAL(117, 151, 161, 190, 111, 237, 188, 218)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__11_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__12_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__13_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__14_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___boxed(lean_object**);
static const lean_closure_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__0___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "optConfig"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(137, 208, 10, 74, 108, 50, 106, 48)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_withSimpRWRulesSeq___lam__0(lean_object* v_a_1_, lean_object* v_trees_2_, lean_object* v___y_3_, lean_object* v___y_4_, lean_object* v___y_5_, lean_object* v___y_6_, lean_object* v___y_7_, lean_object* v___y_8_, lean_object* v___y_9_, lean_object* v___y_10_){
_start:
{
lean_object* v___x_12_; 
lean_inc(v___y_10_);
lean_inc_ref(v___y_9_);
lean_inc(v___y_8_);
lean_inc_ref(v___y_7_);
lean_inc(v___y_6_);
lean_inc_ref(v___y_5_);
lean_inc(v___y_4_);
lean_inc_ref(v___y_3_);
v___x_12_ = lean_apply_9(v_a_1_, v___y_3_, v___y_4_, v___y_5_, v___y_6_, v___y_7_, v___y_8_, v___y_9_, v___y_10_, lean_box(0));
if (lean_obj_tag(v___x_12_) == 0)
{
lean_object* v_a_13_; lean_object* v___x_15_; uint8_t v_isShared_16_; uint8_t v_isSharedCheck_21_; 
v_a_13_ = lean_ctor_get(v___x_12_, 0);
v_isSharedCheck_21_ = !lean_is_exclusive(v___x_12_);
if (v_isSharedCheck_21_ == 0)
{
v___x_15_ = v___x_12_;
v_isShared_16_ = v_isSharedCheck_21_;
goto v_resetjp_14_;
}
else
{
lean_inc(v_a_13_);
lean_dec(v___x_12_);
v___x_15_ = lean_box(0);
v_isShared_16_ = v_isSharedCheck_21_;
goto v_resetjp_14_;
}
v_resetjp_14_:
{
lean_object* v___x_17_; lean_object* v___x_19_; 
v___x_17_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_17_, 0, v_a_13_);
lean_ctor_set(v___x_17_, 1, v_trees_2_);
if (v_isShared_16_ == 0)
{
lean_ctor_set(v___x_15_, 0, v___x_17_);
v___x_19_ = v___x_15_;
goto v_reusejp_18_;
}
else
{
lean_object* v_reuseFailAlloc_20_; 
v_reuseFailAlloc_20_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_20_, 0, v___x_17_);
v___x_19_ = v_reuseFailAlloc_20_;
goto v_reusejp_18_;
}
v_reusejp_18_:
{
return v___x_19_;
}
}
}
else
{
lean_object* v_a_22_; lean_object* v___x_24_; uint8_t v_isShared_25_; uint8_t v_isSharedCheck_29_; 
lean_dec_ref(v_trees_2_);
v_a_22_ = lean_ctor_get(v___x_12_, 0);
v_isSharedCheck_29_ = !lean_is_exclusive(v___x_12_);
if (v_isSharedCheck_29_ == 0)
{
v___x_24_ = v___x_12_;
v_isShared_25_ = v_isSharedCheck_29_;
goto v_resetjp_23_;
}
else
{
lean_inc(v_a_22_);
lean_dec(v___x_12_);
v___x_24_ = lean_box(0);
v_isShared_25_ = v_isSharedCheck_29_;
goto v_resetjp_23_;
}
v_resetjp_23_:
{
lean_object* v___x_27_; 
if (v_isShared_25_ == 0)
{
v___x_27_ = v___x_24_;
goto v_reusejp_26_;
}
else
{
lean_object* v_reuseFailAlloc_28_; 
v_reuseFailAlloc_28_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_28_, 0, v_a_22_);
v___x_27_ = v_reuseFailAlloc_28_;
goto v_reusejp_26_;
}
v_reusejp_26_:
{
return v___x_27_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_withSimpRWRulesSeq___lam__0___boxed(lean_object* v_a_30_, lean_object* v_trees_31_, lean_object* v___y_32_, lean_object* v___y_33_, lean_object* v___y_34_, lean_object* v___y_35_, lean_object* v___y_36_, lean_object* v___y_37_, lean_object* v___y_38_, lean_object* v___y_39_, lean_object* v___y_40_){
_start:
{
lean_object* v_res_41_; 
v_res_41_ = lp_mathlib_Mathlib_Tactic_withSimpRWRulesSeq___lam__0(v_a_30_, v_trees_31_, v___y_32_, v___y_33_, v___y_34_, v___y_35_, v___y_36_, v___y_37_, v___y_38_, v___y_39_);
lean_dec(v___y_39_);
lean_dec_ref(v___y_38_);
lean_dec(v___y_37_);
lean_dec_ref(v___y_36_);
lean_dec(v___y_35_);
lean_dec_ref(v___y_34_);
lean_dec(v___y_33_);
lean_dec_ref(v___y_32_);
return v_res_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_withSimpRWRulesSeq___lam__1(lean_object* v___x_42_, lean_object* v___y_43_, lean_object* v___y_44_, lean_object* v___y_45_, lean_object* v___y_46_, lean_object* v___y_47_, lean_object* v___y_48_, lean_object* v___y_49_, lean_object* v___y_50_){
_start:
{
lean_object* v___x_52_; 
v___x_52_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_52_, 0, v___x_42_);
return v___x_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_withSimpRWRulesSeq___lam__1___boxed(lean_object* v___x_53_, lean_object* v___y_54_, lean_object* v___y_55_, lean_object* v___y_56_, lean_object* v___y_57_, lean_object* v___y_58_, lean_object* v___y_59_, lean_object* v___y_60_, lean_object* v___y_61_, lean_object* v___y_62_){
_start:
{
lean_object* v_res_63_; 
v_res_63_ = lp_mathlib_Mathlib_Tactic_withSimpRWRulesSeq___lam__1(v___x_53_, v___y_54_, v___y_55_, v___y_56_, v___y_57_, v___y_58_, v___y_59_, v___y_60_, v___y_61_);
lean_dec(v___y_61_);
lean_dec_ref(v___y_60_);
lean_dec(v___y_59_);
lean_dec_ref(v___y_58_);
lean_dec(v___y_57_);
lean_dec_ref(v___y_56_);
lean_dec(v___y_55_);
lean_dec_ref(v___y_54_);
return v_res_63_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_getResetInfoTrees___at___00Lean_Elab_withInfoTreeContext___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__0_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_64_; lean_object* v___x_65_; lean_object* v___x_66_; 
v___x_64_ = lean_unsigned_to_nat(32u);
v___x_65_ = lean_mk_empty_array_with_capacity(v___x_64_);
v___x_66_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_66_, 0, v___x_65_);
return v___x_66_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_getResetInfoTrees___at___00Lean_Elab_withInfoTreeContext___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__0_spec__0___redArg___closed__1(void){
_start:
{
size_t v___x_67_; lean_object* v___x_68_; lean_object* v___x_69_; lean_object* v___x_70_; lean_object* v___x_71_; lean_object* v___x_72_; 
v___x_67_ = ((size_t)5ULL);
v___x_68_ = lean_unsigned_to_nat(0u);
v___x_69_ = lean_unsigned_to_nat(32u);
v___x_70_ = lean_mk_empty_array_with_capacity(v___x_69_);
v___x_71_ = lean_obj_once(&lp_mathlib_Lean_Elab_getResetInfoTrees___at___00Lean_Elab_withInfoTreeContext___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__0_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_getResetInfoTrees___at___00Lean_Elab_withInfoTreeContext___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__0_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_getResetInfoTrees___at___00Lean_Elab_withInfoTreeContext___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__0_spec__0___redArg___closed__0);
v___x_72_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_72_, 0, v___x_71_);
lean_ctor_set(v___x_72_, 1, v___x_70_);
lean_ctor_set(v___x_72_, 2, v___x_68_);
lean_ctor_set(v___x_72_, 3, v___x_68_);
lean_ctor_set_usize(v___x_72_, 4, v___x_67_);
return v___x_72_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getResetInfoTrees___at___00Lean_Elab_withInfoTreeContext___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__0_spec__0___redArg(lean_object* v___y_73_){
_start:
{
lean_object* v___x_75_; lean_object* v_infoState_76_; lean_object* v_trees_77_; lean_object* v___x_78_; lean_object* v_infoState_79_; lean_object* v_env_80_; lean_object* v_nextMacroScope_81_; lean_object* v_ngen_82_; lean_object* v_auxDeclNGen_83_; lean_object* v_traceState_84_; lean_object* v_cache_85_; lean_object* v_messages_86_; lean_object* v_snapshotTasks_87_; lean_object* v___x_89_; uint8_t v_isShared_90_; uint8_t v_isSharedCheck_108_; 
v___x_75_ = lean_st_ref_get(v___y_73_);
v_infoState_76_ = lean_ctor_get(v___x_75_, 7);
lean_inc_ref(v_infoState_76_);
lean_dec(v___x_75_);
v_trees_77_ = lean_ctor_get(v_infoState_76_, 2);
lean_inc_ref(v_trees_77_);
lean_dec_ref(v_infoState_76_);
v___x_78_ = lean_st_ref_take(v___y_73_);
v_infoState_79_ = lean_ctor_get(v___x_78_, 7);
v_env_80_ = lean_ctor_get(v___x_78_, 0);
v_nextMacroScope_81_ = lean_ctor_get(v___x_78_, 1);
v_ngen_82_ = lean_ctor_get(v___x_78_, 2);
v_auxDeclNGen_83_ = lean_ctor_get(v___x_78_, 3);
v_traceState_84_ = lean_ctor_get(v___x_78_, 4);
v_cache_85_ = lean_ctor_get(v___x_78_, 5);
v_messages_86_ = lean_ctor_get(v___x_78_, 6);
v_snapshotTasks_87_ = lean_ctor_get(v___x_78_, 8);
v_isSharedCheck_108_ = !lean_is_exclusive(v___x_78_);
if (v_isSharedCheck_108_ == 0)
{
v___x_89_ = v___x_78_;
v_isShared_90_ = v_isSharedCheck_108_;
goto v_resetjp_88_;
}
else
{
lean_inc(v_snapshotTasks_87_);
lean_inc(v_infoState_79_);
lean_inc(v_messages_86_);
lean_inc(v_cache_85_);
lean_inc(v_traceState_84_);
lean_inc(v_auxDeclNGen_83_);
lean_inc(v_ngen_82_);
lean_inc(v_nextMacroScope_81_);
lean_inc(v_env_80_);
lean_dec(v___x_78_);
v___x_89_ = lean_box(0);
v_isShared_90_ = v_isSharedCheck_108_;
goto v_resetjp_88_;
}
v_resetjp_88_:
{
uint8_t v_enabled_91_; lean_object* v_assignment_92_; lean_object* v_lazyAssignment_93_; lean_object* v___x_95_; uint8_t v_isShared_96_; uint8_t v_isSharedCheck_106_; 
v_enabled_91_ = lean_ctor_get_uint8(v_infoState_79_, sizeof(void*)*3);
v_assignment_92_ = lean_ctor_get(v_infoState_79_, 0);
v_lazyAssignment_93_ = lean_ctor_get(v_infoState_79_, 1);
v_isSharedCheck_106_ = !lean_is_exclusive(v_infoState_79_);
if (v_isSharedCheck_106_ == 0)
{
lean_object* v_unused_107_; 
v_unused_107_ = lean_ctor_get(v_infoState_79_, 2);
lean_dec(v_unused_107_);
v___x_95_ = v_infoState_79_;
v_isShared_96_ = v_isSharedCheck_106_;
goto v_resetjp_94_;
}
else
{
lean_inc(v_lazyAssignment_93_);
lean_inc(v_assignment_92_);
lean_dec(v_infoState_79_);
v___x_95_ = lean_box(0);
v_isShared_96_ = v_isSharedCheck_106_;
goto v_resetjp_94_;
}
v_resetjp_94_:
{
lean_object* v___x_97_; lean_object* v___x_99_; 
v___x_97_ = lean_obj_once(&lp_mathlib_Lean_Elab_getResetInfoTrees___at___00Lean_Elab_withInfoTreeContext___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__0_spec__0___redArg___closed__1, &lp_mathlib_Lean_Elab_getResetInfoTrees___at___00Lean_Elab_withInfoTreeContext___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__0_spec__0___redArg___closed__1_once, _init_lp_mathlib_Lean_Elab_getResetInfoTrees___at___00Lean_Elab_withInfoTreeContext___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__0_spec__0___redArg___closed__1);
if (v_isShared_96_ == 0)
{
lean_ctor_set(v___x_95_, 2, v___x_97_);
v___x_99_ = v___x_95_;
goto v_reusejp_98_;
}
else
{
lean_object* v_reuseFailAlloc_105_; 
v_reuseFailAlloc_105_ = lean_alloc_ctor(0, 3, 1);
lean_ctor_set(v_reuseFailAlloc_105_, 0, v_assignment_92_);
lean_ctor_set(v_reuseFailAlloc_105_, 1, v_lazyAssignment_93_);
lean_ctor_set(v_reuseFailAlloc_105_, 2, v___x_97_);
lean_ctor_set_uint8(v_reuseFailAlloc_105_, sizeof(void*)*3, v_enabled_91_);
v___x_99_ = v_reuseFailAlloc_105_;
goto v_reusejp_98_;
}
v_reusejp_98_:
{
lean_object* v___x_101_; 
if (v_isShared_90_ == 0)
{
lean_ctor_set(v___x_89_, 7, v___x_99_);
v___x_101_ = v___x_89_;
goto v_reusejp_100_;
}
else
{
lean_object* v_reuseFailAlloc_104_; 
v_reuseFailAlloc_104_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_104_, 0, v_env_80_);
lean_ctor_set(v_reuseFailAlloc_104_, 1, v_nextMacroScope_81_);
lean_ctor_set(v_reuseFailAlloc_104_, 2, v_ngen_82_);
lean_ctor_set(v_reuseFailAlloc_104_, 3, v_auxDeclNGen_83_);
lean_ctor_set(v_reuseFailAlloc_104_, 4, v_traceState_84_);
lean_ctor_set(v_reuseFailAlloc_104_, 5, v_cache_85_);
lean_ctor_set(v_reuseFailAlloc_104_, 6, v_messages_86_);
lean_ctor_set(v_reuseFailAlloc_104_, 7, v___x_99_);
lean_ctor_set(v_reuseFailAlloc_104_, 8, v_snapshotTasks_87_);
v___x_101_ = v_reuseFailAlloc_104_;
goto v_reusejp_100_;
}
v_reusejp_100_:
{
lean_object* v___x_102_; lean_object* v___x_103_; 
v___x_102_ = lean_st_ref_set(v___y_73_, v___x_101_);
v___x_103_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_103_, 0, v_trees_77_);
return v___x_103_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getResetInfoTrees___at___00Lean_Elab_withInfoTreeContext___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__0_spec__0___redArg___boxed(lean_object* v___y_109_, lean_object* v___y_110_){
_start:
{
lean_object* v_res_111_; 
v_res_111_ = lp_mathlib_Lean_Elab_getResetInfoTrees___at___00Lean_Elab_withInfoTreeContext___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__0_spec__0___redArg(v___y_109_);
lean_dec(v___y_109_);
return v_res_111_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withInfoTreeContext___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__0___redArg___lam__0(lean_object* v___y_112_, lean_object* v_mkInfoTree_113_, lean_object* v___y_114_, lean_object* v___y_115_, lean_object* v___y_116_, lean_object* v___y_117_, lean_object* v___y_118_, lean_object* v___y_119_, lean_object* v___y_120_, lean_object* v_a_121_, lean_object* v_a_x3f_122_){
_start:
{
lean_object* v___x_124_; lean_object* v_infoState_125_; lean_object* v_trees_126_; lean_object* v___x_127_; 
v___x_124_ = lean_st_ref_get(v___y_112_);
v_infoState_125_ = lean_ctor_get(v___x_124_, 7);
lean_inc_ref(v_infoState_125_);
lean_dec(v___x_124_);
v_trees_126_ = lean_ctor_get(v_infoState_125_, 2);
lean_inc_ref(v_trees_126_);
lean_dec_ref(v_infoState_125_);
lean_inc(v___y_112_);
lean_inc_ref(v___y_120_);
lean_inc(v___y_119_);
lean_inc_ref(v___y_118_);
lean_inc(v___y_117_);
lean_inc_ref(v___y_116_);
lean_inc(v___y_115_);
lean_inc_ref(v___y_114_);
v___x_127_ = lean_apply_10(v_mkInfoTree_113_, v_trees_126_, v___y_114_, v___y_115_, v___y_116_, v___y_117_, v___y_118_, v___y_119_, v___y_120_, v___y_112_, lean_box(0));
if (lean_obj_tag(v___x_127_) == 0)
{
lean_object* v_a_128_; lean_object* v___x_130_; uint8_t v_isShared_131_; uint8_t v_isSharedCheck_166_; 
v_a_128_ = lean_ctor_get(v___x_127_, 0);
v_isSharedCheck_166_ = !lean_is_exclusive(v___x_127_);
if (v_isSharedCheck_166_ == 0)
{
v___x_130_ = v___x_127_;
v_isShared_131_ = v_isSharedCheck_166_;
goto v_resetjp_129_;
}
else
{
lean_inc(v_a_128_);
lean_dec(v___x_127_);
v___x_130_ = lean_box(0);
v_isShared_131_ = v_isSharedCheck_166_;
goto v_resetjp_129_;
}
v_resetjp_129_:
{
lean_object* v___x_132_; lean_object* v_infoState_133_; lean_object* v_env_134_; lean_object* v_nextMacroScope_135_; lean_object* v_ngen_136_; lean_object* v_auxDeclNGen_137_; lean_object* v_traceState_138_; lean_object* v_cache_139_; lean_object* v_messages_140_; lean_object* v_snapshotTasks_141_; lean_object* v___x_143_; uint8_t v_isShared_144_; uint8_t v_isSharedCheck_165_; 
v___x_132_ = lean_st_ref_take(v___y_112_);
v_infoState_133_ = lean_ctor_get(v___x_132_, 7);
v_env_134_ = lean_ctor_get(v___x_132_, 0);
v_nextMacroScope_135_ = lean_ctor_get(v___x_132_, 1);
v_ngen_136_ = lean_ctor_get(v___x_132_, 2);
v_auxDeclNGen_137_ = lean_ctor_get(v___x_132_, 3);
v_traceState_138_ = lean_ctor_get(v___x_132_, 4);
v_cache_139_ = lean_ctor_get(v___x_132_, 5);
v_messages_140_ = lean_ctor_get(v___x_132_, 6);
v_snapshotTasks_141_ = lean_ctor_get(v___x_132_, 8);
v_isSharedCheck_165_ = !lean_is_exclusive(v___x_132_);
if (v_isSharedCheck_165_ == 0)
{
v___x_143_ = v___x_132_;
v_isShared_144_ = v_isSharedCheck_165_;
goto v_resetjp_142_;
}
else
{
lean_inc(v_snapshotTasks_141_);
lean_inc(v_infoState_133_);
lean_inc(v_messages_140_);
lean_inc(v_cache_139_);
lean_inc(v_traceState_138_);
lean_inc(v_auxDeclNGen_137_);
lean_inc(v_ngen_136_);
lean_inc(v_nextMacroScope_135_);
lean_inc(v_env_134_);
lean_dec(v___x_132_);
v___x_143_ = lean_box(0);
v_isShared_144_ = v_isSharedCheck_165_;
goto v_resetjp_142_;
}
v_resetjp_142_:
{
uint8_t v_enabled_145_; lean_object* v_assignment_146_; lean_object* v_lazyAssignment_147_; lean_object* v___x_149_; uint8_t v_isShared_150_; uint8_t v_isSharedCheck_163_; 
v_enabled_145_ = lean_ctor_get_uint8(v_infoState_133_, sizeof(void*)*3);
v_assignment_146_ = lean_ctor_get(v_infoState_133_, 0);
v_lazyAssignment_147_ = lean_ctor_get(v_infoState_133_, 1);
v_isSharedCheck_163_ = !lean_is_exclusive(v_infoState_133_);
if (v_isSharedCheck_163_ == 0)
{
lean_object* v_unused_164_; 
v_unused_164_ = lean_ctor_get(v_infoState_133_, 2);
lean_dec(v_unused_164_);
v___x_149_ = v_infoState_133_;
v_isShared_150_ = v_isSharedCheck_163_;
goto v_resetjp_148_;
}
else
{
lean_inc(v_lazyAssignment_147_);
lean_inc(v_assignment_146_);
lean_dec(v_infoState_133_);
v___x_149_ = lean_box(0);
v_isShared_150_ = v_isSharedCheck_163_;
goto v_resetjp_148_;
}
v_resetjp_148_:
{
lean_object* v___x_151_; lean_object* v___x_153_; 
v___x_151_ = l_Lean_PersistentArray_push___redArg(v_a_121_, v_a_128_);
if (v_isShared_150_ == 0)
{
lean_ctor_set(v___x_149_, 2, v___x_151_);
v___x_153_ = v___x_149_;
goto v_reusejp_152_;
}
else
{
lean_object* v_reuseFailAlloc_162_; 
v_reuseFailAlloc_162_ = lean_alloc_ctor(0, 3, 1);
lean_ctor_set(v_reuseFailAlloc_162_, 0, v_assignment_146_);
lean_ctor_set(v_reuseFailAlloc_162_, 1, v_lazyAssignment_147_);
lean_ctor_set(v_reuseFailAlloc_162_, 2, v___x_151_);
lean_ctor_set_uint8(v_reuseFailAlloc_162_, sizeof(void*)*3, v_enabled_145_);
v___x_153_ = v_reuseFailAlloc_162_;
goto v_reusejp_152_;
}
v_reusejp_152_:
{
lean_object* v___x_155_; 
if (v_isShared_144_ == 0)
{
lean_ctor_set(v___x_143_, 7, v___x_153_);
v___x_155_ = v___x_143_;
goto v_reusejp_154_;
}
else
{
lean_object* v_reuseFailAlloc_161_; 
v_reuseFailAlloc_161_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_161_, 0, v_env_134_);
lean_ctor_set(v_reuseFailAlloc_161_, 1, v_nextMacroScope_135_);
lean_ctor_set(v_reuseFailAlloc_161_, 2, v_ngen_136_);
lean_ctor_set(v_reuseFailAlloc_161_, 3, v_auxDeclNGen_137_);
lean_ctor_set(v_reuseFailAlloc_161_, 4, v_traceState_138_);
lean_ctor_set(v_reuseFailAlloc_161_, 5, v_cache_139_);
lean_ctor_set(v_reuseFailAlloc_161_, 6, v_messages_140_);
lean_ctor_set(v_reuseFailAlloc_161_, 7, v___x_153_);
lean_ctor_set(v_reuseFailAlloc_161_, 8, v_snapshotTasks_141_);
v___x_155_ = v_reuseFailAlloc_161_;
goto v_reusejp_154_;
}
v_reusejp_154_:
{
lean_object* v___x_156_; lean_object* v___x_157_; lean_object* v___x_159_; 
v___x_156_ = lean_st_ref_set(v___y_112_, v___x_155_);
v___x_157_ = lean_box(0);
if (v_isShared_131_ == 0)
{
lean_ctor_set(v___x_130_, 0, v___x_157_);
v___x_159_ = v___x_130_;
goto v_reusejp_158_;
}
else
{
lean_object* v_reuseFailAlloc_160_; 
v_reuseFailAlloc_160_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_160_, 0, v___x_157_);
v___x_159_ = v_reuseFailAlloc_160_;
goto v_reusejp_158_;
}
v_reusejp_158_:
{
return v___x_159_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_167_; lean_object* v___x_169_; uint8_t v_isShared_170_; uint8_t v_isSharedCheck_174_; 
lean_dec_ref(v_a_121_);
v_a_167_ = lean_ctor_get(v___x_127_, 0);
v_isSharedCheck_174_ = !lean_is_exclusive(v___x_127_);
if (v_isSharedCheck_174_ == 0)
{
v___x_169_ = v___x_127_;
v_isShared_170_ = v_isSharedCheck_174_;
goto v_resetjp_168_;
}
else
{
lean_inc(v_a_167_);
lean_dec(v___x_127_);
v___x_169_ = lean_box(0);
v_isShared_170_ = v_isSharedCheck_174_;
goto v_resetjp_168_;
}
v_resetjp_168_:
{
lean_object* v___x_172_; 
if (v_isShared_170_ == 0)
{
v___x_172_ = v___x_169_;
goto v_reusejp_171_;
}
else
{
lean_object* v_reuseFailAlloc_173_; 
v_reuseFailAlloc_173_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_173_, 0, v_a_167_);
v___x_172_ = v_reuseFailAlloc_173_;
goto v_reusejp_171_;
}
v_reusejp_171_:
{
return v___x_172_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withInfoTreeContext___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__0___redArg___lam__0___boxed(lean_object* v___y_175_, lean_object* v_mkInfoTree_176_, lean_object* v___y_177_, lean_object* v___y_178_, lean_object* v___y_179_, lean_object* v___y_180_, lean_object* v___y_181_, lean_object* v___y_182_, lean_object* v___y_183_, lean_object* v_a_184_, lean_object* v_a_x3f_185_, lean_object* v___y_186_){
_start:
{
lean_object* v_res_187_; 
v_res_187_ = lp_mathlib_Lean_Elab_withInfoTreeContext___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__0___redArg___lam__0(v___y_175_, v_mkInfoTree_176_, v___y_177_, v___y_178_, v___y_179_, v___y_180_, v___y_181_, v___y_182_, v___y_183_, v_a_184_, v_a_x3f_185_);
lean_dec(v_a_x3f_185_);
lean_dec_ref(v___y_183_);
lean_dec(v___y_182_);
lean_dec_ref(v___y_181_);
lean_dec(v___y_180_);
lean_dec_ref(v___y_179_);
lean_dec(v___y_178_);
lean_dec_ref(v___y_177_);
lean_dec(v___y_175_);
return v_res_187_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withInfoTreeContext___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__0___redArg(lean_object* v_x_188_, lean_object* v_mkInfoTree_189_, lean_object* v___y_190_, lean_object* v___y_191_, lean_object* v___y_192_, lean_object* v___y_193_, lean_object* v___y_194_, lean_object* v___y_195_, lean_object* v___y_196_, lean_object* v___y_197_){
_start:
{
lean_object* v___x_199_; lean_object* v_infoState_200_; uint8_t v_enabled_201_; 
v___x_199_ = lean_st_ref_get(v___y_197_);
v_infoState_200_ = lean_ctor_get(v___x_199_, 7);
lean_inc_ref(v_infoState_200_);
lean_dec(v___x_199_);
v_enabled_201_ = lean_ctor_get_uint8(v_infoState_200_, sizeof(void*)*3);
lean_dec_ref(v_infoState_200_);
if (v_enabled_201_ == 0)
{
lean_object* v___x_202_; 
lean_dec_ref(v_mkInfoTree_189_);
lean_inc(v___y_197_);
lean_inc_ref(v___y_196_);
lean_inc(v___y_195_);
lean_inc_ref(v___y_194_);
lean_inc(v___y_193_);
lean_inc_ref(v___y_192_);
lean_inc(v___y_191_);
lean_inc_ref(v___y_190_);
v___x_202_ = lean_apply_9(v_x_188_, v___y_190_, v___y_191_, v___y_192_, v___y_193_, v___y_194_, v___y_195_, v___y_196_, v___y_197_, lean_box(0));
return v___x_202_;
}
else
{
lean_object* v___x_203_; lean_object* v_a_204_; lean_object* v_r_205_; 
v___x_203_ = lp_mathlib_Lean_Elab_getResetInfoTrees___at___00Lean_Elab_withInfoTreeContext___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__0_spec__0___redArg(v___y_197_);
v_a_204_ = lean_ctor_get(v___x_203_, 0);
lean_inc(v_a_204_);
lean_dec_ref(v___x_203_);
lean_inc(v___y_197_);
lean_inc_ref(v___y_196_);
lean_inc(v___y_195_);
lean_inc_ref(v___y_194_);
lean_inc(v___y_193_);
lean_inc_ref(v___y_192_);
lean_inc(v___y_191_);
lean_inc_ref(v___y_190_);
v_r_205_ = lean_apply_9(v_x_188_, v___y_190_, v___y_191_, v___y_192_, v___y_193_, v___y_194_, v___y_195_, v___y_196_, v___y_197_, lean_box(0));
if (lean_obj_tag(v_r_205_) == 0)
{
lean_object* v_a_206_; lean_object* v___x_208_; uint8_t v_isShared_209_; uint8_t v_isSharedCheck_230_; 
v_a_206_ = lean_ctor_get(v_r_205_, 0);
v_isSharedCheck_230_ = !lean_is_exclusive(v_r_205_);
if (v_isSharedCheck_230_ == 0)
{
v___x_208_ = v_r_205_;
v_isShared_209_ = v_isSharedCheck_230_;
goto v_resetjp_207_;
}
else
{
lean_inc(v_a_206_);
lean_dec(v_r_205_);
v___x_208_ = lean_box(0);
v_isShared_209_ = v_isSharedCheck_230_;
goto v_resetjp_207_;
}
v_resetjp_207_:
{
lean_object* v___x_211_; 
lean_inc(v_a_206_);
if (v_isShared_209_ == 0)
{
lean_ctor_set_tag(v___x_208_, 1);
v___x_211_ = v___x_208_;
goto v_reusejp_210_;
}
else
{
lean_object* v_reuseFailAlloc_229_; 
v_reuseFailAlloc_229_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_229_, 0, v_a_206_);
v___x_211_ = v_reuseFailAlloc_229_;
goto v_reusejp_210_;
}
v_reusejp_210_:
{
lean_object* v___x_212_; 
v___x_212_ = lp_mathlib_Lean_Elab_withInfoTreeContext___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__0___redArg___lam__0(v___y_197_, v_mkInfoTree_189_, v___y_190_, v___y_191_, v___y_192_, v___y_193_, v___y_194_, v___y_195_, v___y_196_, v_a_204_, v___x_211_);
lean_dec_ref(v___x_211_);
if (lean_obj_tag(v___x_212_) == 0)
{
lean_object* v___x_214_; uint8_t v_isShared_215_; uint8_t v_isSharedCheck_219_; 
v_isSharedCheck_219_ = !lean_is_exclusive(v___x_212_);
if (v_isSharedCheck_219_ == 0)
{
lean_object* v_unused_220_; 
v_unused_220_ = lean_ctor_get(v___x_212_, 0);
lean_dec(v_unused_220_);
v___x_214_ = v___x_212_;
v_isShared_215_ = v_isSharedCheck_219_;
goto v_resetjp_213_;
}
else
{
lean_dec(v___x_212_);
v___x_214_ = lean_box(0);
v_isShared_215_ = v_isSharedCheck_219_;
goto v_resetjp_213_;
}
v_resetjp_213_:
{
lean_object* v___x_217_; 
if (v_isShared_215_ == 0)
{
lean_ctor_set(v___x_214_, 0, v_a_206_);
v___x_217_ = v___x_214_;
goto v_reusejp_216_;
}
else
{
lean_object* v_reuseFailAlloc_218_; 
v_reuseFailAlloc_218_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_218_, 0, v_a_206_);
v___x_217_ = v_reuseFailAlloc_218_;
goto v_reusejp_216_;
}
v_reusejp_216_:
{
return v___x_217_;
}
}
}
else
{
lean_object* v_a_221_; lean_object* v___x_223_; uint8_t v_isShared_224_; uint8_t v_isSharedCheck_228_; 
lean_dec(v_a_206_);
v_a_221_ = lean_ctor_get(v___x_212_, 0);
v_isSharedCheck_228_ = !lean_is_exclusive(v___x_212_);
if (v_isSharedCheck_228_ == 0)
{
v___x_223_ = v___x_212_;
v_isShared_224_ = v_isSharedCheck_228_;
goto v_resetjp_222_;
}
else
{
lean_inc(v_a_221_);
lean_dec(v___x_212_);
v___x_223_ = lean_box(0);
v_isShared_224_ = v_isSharedCheck_228_;
goto v_resetjp_222_;
}
v_resetjp_222_:
{
lean_object* v___x_226_; 
if (v_isShared_224_ == 0)
{
v___x_226_ = v___x_223_;
goto v_reusejp_225_;
}
else
{
lean_object* v_reuseFailAlloc_227_; 
v_reuseFailAlloc_227_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_227_, 0, v_a_221_);
v___x_226_ = v_reuseFailAlloc_227_;
goto v_reusejp_225_;
}
v_reusejp_225_:
{
return v___x_226_;
}
}
}
}
}
}
else
{
lean_object* v_a_231_; lean_object* v___x_232_; lean_object* v___x_233_; 
v_a_231_ = lean_ctor_get(v_r_205_, 0);
lean_inc(v_a_231_);
lean_dec_ref_known(v_r_205_, 1);
v___x_232_ = lean_box(0);
v___x_233_ = lp_mathlib_Lean_Elab_withInfoTreeContext___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__0___redArg___lam__0(v___y_197_, v_mkInfoTree_189_, v___y_190_, v___y_191_, v___y_192_, v___y_193_, v___y_194_, v___y_195_, v___y_196_, v_a_204_, v___x_232_);
if (lean_obj_tag(v___x_233_) == 0)
{
lean_object* v___x_235_; uint8_t v_isShared_236_; uint8_t v_isSharedCheck_240_; 
v_isSharedCheck_240_ = !lean_is_exclusive(v___x_233_);
if (v_isSharedCheck_240_ == 0)
{
lean_object* v_unused_241_; 
v_unused_241_ = lean_ctor_get(v___x_233_, 0);
lean_dec(v_unused_241_);
v___x_235_ = v___x_233_;
v_isShared_236_ = v_isSharedCheck_240_;
goto v_resetjp_234_;
}
else
{
lean_dec(v___x_233_);
v___x_235_ = lean_box(0);
v_isShared_236_ = v_isSharedCheck_240_;
goto v_resetjp_234_;
}
v_resetjp_234_:
{
lean_object* v___x_238_; 
if (v_isShared_236_ == 0)
{
lean_ctor_set_tag(v___x_235_, 1);
lean_ctor_set(v___x_235_, 0, v_a_231_);
v___x_238_ = v___x_235_;
goto v_reusejp_237_;
}
else
{
lean_object* v_reuseFailAlloc_239_; 
v_reuseFailAlloc_239_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_239_, 0, v_a_231_);
v___x_238_ = v_reuseFailAlloc_239_;
goto v_reusejp_237_;
}
v_reusejp_237_:
{
return v___x_238_;
}
}
}
else
{
lean_object* v_a_242_; lean_object* v___x_244_; uint8_t v_isShared_245_; uint8_t v_isSharedCheck_249_; 
lean_dec(v_a_231_);
v_a_242_ = lean_ctor_get(v___x_233_, 0);
v_isSharedCheck_249_ = !lean_is_exclusive(v___x_233_);
if (v_isSharedCheck_249_ == 0)
{
v___x_244_ = v___x_233_;
v_isShared_245_ = v_isSharedCheck_249_;
goto v_resetjp_243_;
}
else
{
lean_inc(v_a_242_);
lean_dec(v___x_233_);
v___x_244_ = lean_box(0);
v_isShared_245_ = v_isSharedCheck_249_;
goto v_resetjp_243_;
}
v_resetjp_243_:
{
lean_object* v___x_247_; 
if (v_isShared_245_ == 0)
{
v___x_247_ = v___x_244_;
goto v_reusejp_246_;
}
else
{
lean_object* v_reuseFailAlloc_248_; 
v_reuseFailAlloc_248_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_248_, 0, v_a_242_);
v___x_247_ = v_reuseFailAlloc_248_;
goto v_reusejp_246_;
}
v_reusejp_246_:
{
return v___x_247_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withInfoTreeContext___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__0___redArg___boxed(lean_object* v_x_250_, lean_object* v_mkInfoTree_251_, lean_object* v___y_252_, lean_object* v___y_253_, lean_object* v___y_254_, lean_object* v___y_255_, lean_object* v___y_256_, lean_object* v___y_257_, lean_object* v___y_258_, lean_object* v___y_259_, lean_object* v___y_260_){
_start:
{
lean_object* v_res_261_; 
v_res_261_ = lp_mathlib_Lean_Elab_withInfoTreeContext___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__0___redArg(v_x_250_, v_mkInfoTree_251_, v___y_252_, v___y_253_, v___y_254_, v___y_255_, v___y_256_, v___y_257_, v___y_258_, v___y_259_);
lean_dec(v___y_259_);
lean_dec_ref(v___y_258_);
lean_dec(v___y_257_);
lean_dec_ref(v___y_256_);
lean_dec(v___y_255_);
lean_dec_ref(v___y_254_);
lean_dec(v___y_253_);
lean_dec_ref(v___y_252_);
return v_res_261_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__1___redArg___lam__1(lean_object* v___x_262_, lean_object* v_x_263_, uint8_t v___y_264_, lean_object* v___x_265_, lean_object* v___y_266_, lean_object* v___y_267_, lean_object* v___y_268_, lean_object* v___y_269_, lean_object* v___y_270_, lean_object* v___y_271_, lean_object* v___y_272_, lean_object* v___y_273_){
_start:
{
lean_object* v_fileName_275_; lean_object* v_fileMap_276_; lean_object* v_options_277_; lean_object* v_currRecDepth_278_; lean_object* v_maxRecDepth_279_; lean_object* v_ref_280_; lean_object* v_currNamespace_281_; lean_object* v_openDecls_282_; lean_object* v_initHeartbeats_283_; lean_object* v_maxHeartbeats_284_; lean_object* v_quotContext_285_; lean_object* v_currMacroScope_286_; uint8_t v_diag_287_; lean_object* v_cancelTk_x3f_288_; uint8_t v_suppressElabErrors_289_; lean_object* v_inheritedTraceOptions_290_; lean_object* v___x_292_; uint8_t v_isShared_293_; uint8_t v_isSharedCheck_300_; 
v_fileName_275_ = lean_ctor_get(v___y_272_, 0);
v_fileMap_276_ = lean_ctor_get(v___y_272_, 1);
v_options_277_ = lean_ctor_get(v___y_272_, 2);
v_currRecDepth_278_ = lean_ctor_get(v___y_272_, 3);
v_maxRecDepth_279_ = lean_ctor_get(v___y_272_, 4);
v_ref_280_ = lean_ctor_get(v___y_272_, 5);
v_currNamespace_281_ = lean_ctor_get(v___y_272_, 6);
v_openDecls_282_ = lean_ctor_get(v___y_272_, 7);
v_initHeartbeats_283_ = lean_ctor_get(v___y_272_, 8);
v_maxHeartbeats_284_ = lean_ctor_get(v___y_272_, 9);
v_quotContext_285_ = lean_ctor_get(v___y_272_, 10);
v_currMacroScope_286_ = lean_ctor_get(v___y_272_, 11);
v_diag_287_ = lean_ctor_get_uint8(v___y_272_, sizeof(void*)*14);
v_cancelTk_x3f_288_ = lean_ctor_get(v___y_272_, 12);
v_suppressElabErrors_289_ = lean_ctor_get_uint8(v___y_272_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_290_ = lean_ctor_get(v___y_272_, 13);
v_isSharedCheck_300_ = !lean_is_exclusive(v___y_272_);
if (v_isSharedCheck_300_ == 0)
{
v___x_292_ = v___y_272_;
v_isShared_293_ = v_isSharedCheck_300_;
goto v_resetjp_291_;
}
else
{
lean_inc(v_inheritedTraceOptions_290_);
lean_inc(v_cancelTk_x3f_288_);
lean_inc(v_currMacroScope_286_);
lean_inc(v_quotContext_285_);
lean_inc(v_maxHeartbeats_284_);
lean_inc(v_initHeartbeats_283_);
lean_inc(v_openDecls_282_);
lean_inc(v_currNamespace_281_);
lean_inc(v_ref_280_);
lean_inc(v_maxRecDepth_279_);
lean_inc(v_currRecDepth_278_);
lean_inc(v_options_277_);
lean_inc(v_fileMap_276_);
lean_inc(v_fileName_275_);
lean_dec(v___y_272_);
v___x_292_ = lean_box(0);
v_isShared_293_ = v_isSharedCheck_300_;
goto v_resetjp_291_;
}
v_resetjp_291_:
{
lean_object* v_ref_294_; lean_object* v___x_296_; 
v_ref_294_ = l_Lean_replaceRef(v___x_262_, v_ref_280_);
lean_dec(v_ref_280_);
if (v_isShared_293_ == 0)
{
lean_ctor_set(v___x_292_, 5, v_ref_294_);
v___x_296_ = v___x_292_;
goto v_reusejp_295_;
}
else
{
lean_object* v_reuseFailAlloc_299_; 
v_reuseFailAlloc_299_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v_reuseFailAlloc_299_, 0, v_fileName_275_);
lean_ctor_set(v_reuseFailAlloc_299_, 1, v_fileMap_276_);
lean_ctor_set(v_reuseFailAlloc_299_, 2, v_options_277_);
lean_ctor_set(v_reuseFailAlloc_299_, 3, v_currRecDepth_278_);
lean_ctor_set(v_reuseFailAlloc_299_, 4, v_maxRecDepth_279_);
lean_ctor_set(v_reuseFailAlloc_299_, 5, v_ref_294_);
lean_ctor_set(v_reuseFailAlloc_299_, 6, v_currNamespace_281_);
lean_ctor_set(v_reuseFailAlloc_299_, 7, v_openDecls_282_);
lean_ctor_set(v_reuseFailAlloc_299_, 8, v_initHeartbeats_283_);
lean_ctor_set(v_reuseFailAlloc_299_, 9, v_maxHeartbeats_284_);
lean_ctor_set(v_reuseFailAlloc_299_, 10, v_quotContext_285_);
lean_ctor_set(v_reuseFailAlloc_299_, 11, v_currMacroScope_286_);
lean_ctor_set(v_reuseFailAlloc_299_, 12, v_cancelTk_x3f_288_);
lean_ctor_set(v_reuseFailAlloc_299_, 13, v_inheritedTraceOptions_290_);
lean_ctor_set_uint8(v_reuseFailAlloc_299_, sizeof(void*)*14, v_diag_287_);
lean_ctor_set_uint8(v_reuseFailAlloc_299_, sizeof(void*)*14 + 1, v_suppressElabErrors_289_);
v___x_296_ = v_reuseFailAlloc_299_;
goto v_reusejp_295_;
}
v_reusejp_295_:
{
lean_object* v___x_297_; lean_object* v___x_298_; 
v___x_297_ = lean_box(v___y_264_);
v___x_298_ = lean_apply_11(v_x_263_, v___x_297_, v___x_265_, v___y_266_, v___y_267_, v___y_268_, v___y_269_, v___y_270_, v___y_271_, v___x_296_, v___y_273_, lean_box(0));
return v___x_298_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__1___redArg___lam__1___boxed(lean_object* v___x_301_, lean_object* v_x_302_, lean_object* v___y_303_, lean_object* v___x_304_, lean_object* v___y_305_, lean_object* v___y_306_, lean_object* v___y_307_, lean_object* v___y_308_, lean_object* v___y_309_, lean_object* v___y_310_, lean_object* v___y_311_, lean_object* v___y_312_, lean_object* v___y_313_){
_start:
{
uint8_t v___y_9068__boxed_314_; lean_object* v_res_315_; 
v___y_9068__boxed_314_ = lean_unbox(v___y_303_);
v_res_315_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__1___redArg___lam__1(v___x_301_, v_x_302_, v___y_9068__boxed_314_, v___x_304_, v___y_305_, v___y_306_, v___y_307_, v___y_308_, v___y_309_, v___y_310_, v___y_311_, v___y_312_);
lean_dec(v___x_301_);
return v_res_315_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__1___redArg(lean_object* v_rules_319_, lean_object* v_x_320_, lean_object* v_range_321_, lean_object* v_b_322_, lean_object* v_i_323_, lean_object* v___y_324_, lean_object* v___y_325_, lean_object* v___y_326_, lean_object* v___y_327_, lean_object* v___y_328_, lean_object* v___y_329_, lean_object* v___y_330_, lean_object* v___y_331_){
_start:
{
lean_object* v_stop_333_; lean_object* v_step_334_; uint8_t v___x_335_; 
v_stop_333_ = lean_ctor_get(v_range_321_, 1);
v_step_334_ = lean_ctor_get(v_range_321_, 2);
v___x_335_ = lean_nat_dec_lt(v_i_323_, v_stop_333_);
if (v___x_335_ == 0)
{
lean_object* v___x_336_; 
lean_dec(v_i_323_);
lean_dec_ref(v_x_320_);
v___x_336_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_336_, 0, v_b_322_);
return v___x_336_;
}
else
{
lean_object* v___x_337_; lean_object* v___x_338_; lean_object* v___x_339_; lean_object* v___x_340_; lean_object* v___x_341_; lean_object* v___x_342_; lean_object* v___x_343_; lean_object* v___y_345_; uint8_t v___y_346_; lean_object* v___y_365_; lean_object* v___x_375_; lean_object* v___x_376_; uint8_t v___x_377_; 
v___x_337_ = lean_unsigned_to_nat(2u);
v___x_338_ = lean_box(0);
v___x_339_ = lean_unsigned_to_nat(1u);
v___x_340_ = lean_box(0);
v___x_341_ = lean_unsigned_to_nat(0u);
v___x_342_ = lean_nat_mul(v_i_323_, v___x_337_);
v___x_343_ = lean_array_get_borrowed(v___x_338_, v_rules_319_, v___x_342_);
v___x_375_ = lean_nat_add(v___x_342_, v___x_339_);
lean_dec(v___x_342_);
v___x_376_ = lean_array_get_size(v_rules_319_);
v___x_377_ = lean_nat_dec_lt(v___x_375_, v___x_376_);
if (v___x_377_ == 0)
{
lean_dec(v___x_375_);
v___y_365_ = v___x_338_;
goto v___jp_364_;
}
else
{
lean_object* v___x_378_; 
v___x_378_ = lean_array_fget_borrowed(v_rules_319_, v___x_375_);
lean_dec(v___x_375_);
lean_inc(v___x_378_);
v___y_365_ = v___x_378_;
goto v___jp_364_;
}
v___jp_344_:
{
lean_object* v___x_347_; 
v___x_347_ = l_Lean_Elab_Tactic_mkInitialTacticInfo(v___y_345_, v___y_324_, v___y_325_, v___y_326_, v___y_327_, v___y_328_, v___y_329_, v___y_330_, v___y_331_);
if (lean_obj_tag(v___x_347_) == 0)
{
lean_object* v_a_348_; lean_object* v___f_349_; lean_object* v___x_350_; lean_object* v___x_351_; lean_object* v___f_352_; lean_object* v___x_353_; 
v_a_348_ = lean_ctor_get(v___x_347_, 0);
lean_inc(v_a_348_);
lean_dec_ref_known(v___x_347_, 1);
v___f_349_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_withSimpRWRulesSeq___lam__0___boxed), 11, 1);
lean_closure_set(v___f_349_, 0, v_a_348_);
v___x_350_ = l_Lean_Syntax_getArg(v___x_343_, v___x_339_);
v___x_351_ = lean_box(v___y_346_);
lean_inc_ref(v_x_320_);
lean_inc(v___x_343_);
v___f_352_ = lean_alloc_closure((void*)(lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__1___redArg___lam__1___boxed), 13, 4);
lean_closure_set(v___f_352_, 0, v___x_343_);
lean_closure_set(v___f_352_, 1, v_x_320_);
lean_closure_set(v___f_352_, 2, v___x_351_);
lean_closure_set(v___f_352_, 3, v___x_350_);
v___x_353_ = lp_mathlib_Lean_Elab_withInfoTreeContext___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__0___redArg(v___f_352_, v___f_349_, v___y_324_, v___y_325_, v___y_326_, v___y_327_, v___y_328_, v___y_329_, v___y_330_, v___y_331_);
if (lean_obj_tag(v___x_353_) == 0)
{
lean_object* v___x_354_; 
lean_dec_ref_known(v___x_353_, 1);
v___x_354_ = lean_nat_add(v_i_323_, v_step_334_);
lean_dec(v_i_323_);
v_b_322_ = v___x_340_;
v_i_323_ = v___x_354_;
goto _start;
}
else
{
lean_dec(v_i_323_);
lean_dec_ref(v_x_320_);
return v___x_353_;
}
}
else
{
lean_object* v_a_356_; lean_object* v___x_358_; uint8_t v_isShared_359_; uint8_t v_isSharedCheck_363_; 
lean_dec(v_i_323_);
lean_dec_ref(v_x_320_);
v_a_356_ = lean_ctor_get(v___x_347_, 0);
v_isSharedCheck_363_ = !lean_is_exclusive(v___x_347_);
if (v_isSharedCheck_363_ == 0)
{
v___x_358_ = v___x_347_;
v_isShared_359_ = v_isSharedCheck_363_;
goto v_resetjp_357_;
}
else
{
lean_inc(v_a_356_);
lean_dec(v___x_347_);
v___x_358_ = lean_box(0);
v_isShared_359_ = v_isSharedCheck_363_;
goto v_resetjp_357_;
}
v_resetjp_357_:
{
lean_object* v___x_361_; 
if (v_isShared_359_ == 0)
{
v___x_361_ = v___x_358_;
goto v_reusejp_360_;
}
else
{
lean_object* v_reuseFailAlloc_362_; 
v_reuseFailAlloc_362_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_362_, 0, v_a_356_);
v___x_361_ = v_reuseFailAlloc_362_;
goto v_reusejp_360_;
}
v_reusejp_360_:
{
return v___x_361_;
}
}
}
}
v___jp_364_:
{
lean_object* v___x_366_; lean_object* v___x_367_; lean_object* v___x_368_; lean_object* v___x_369_; lean_object* v___x_370_; lean_object* v___x_371_; lean_object* v___x_372_; uint8_t v___x_373_; 
v___x_366_ = lean_mk_empty_array_with_capacity(v___x_337_);
lean_inc(v___x_343_);
v___x_367_ = lean_array_push(v___x_366_, v___x_343_);
v___x_368_ = lean_array_push(v___x_367_, v___y_365_);
v___x_369_ = ((lean_object*)(lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__1___redArg___closed__1));
v___x_370_ = lean_box(2);
v___x_371_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_371_, 0, v___x_370_);
lean_ctor_set(v___x_371_, 1, v___x_369_);
lean_ctor_set(v___x_371_, 2, v___x_368_);
v___x_372_ = l_Lean_Syntax_getArg(v___x_343_, v___x_341_);
v___x_373_ = l_Lean_Syntax_isNone(v___x_372_);
lean_dec(v___x_372_);
if (v___x_373_ == 0)
{
v___y_345_ = v___x_371_;
v___y_346_ = v___x_335_;
goto v___jp_344_;
}
else
{
uint8_t v___x_374_; 
v___x_374_ = 0;
v___y_345_ = v___x_371_;
v___y_346_ = v___x_374_;
goto v___jp_344_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__1___redArg___boxed(lean_object* v_rules_379_, lean_object* v_x_380_, lean_object* v_range_381_, lean_object* v_b_382_, lean_object* v_i_383_, lean_object* v___y_384_, lean_object* v___y_385_, lean_object* v___y_386_, lean_object* v___y_387_, lean_object* v___y_388_, lean_object* v___y_389_, lean_object* v___y_390_, lean_object* v___y_391_, lean_object* v___y_392_){
_start:
{
lean_object* v_res_393_; 
v_res_393_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__1___redArg(v_rules_379_, v_x_380_, v_range_381_, v_b_382_, v_i_383_, v___y_384_, v___y_385_, v___y_386_, v___y_387_, v___y_388_, v___y_389_, v___y_390_, v___y_391_);
lean_dec(v___y_391_);
lean_dec_ref(v___y_390_);
lean_dec(v___y_389_);
lean_dec_ref(v___y_388_);
lean_dec(v___y_387_);
lean_dec_ref(v___y_386_);
lean_dec(v___y_385_);
lean_dec_ref(v___y_384_);
lean_dec_ref(v_range_381_);
lean_dec_ref(v_rules_379_);
return v_res_393_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_withSimpRWRulesSeq(lean_object* v_rwRulesSeqStx_396_, lean_object* v_x_397_, lean_object* v_a_398_, lean_object* v_a_399_, lean_object* v_a_400_, lean_object* v_a_401_, lean_object* v_a_402_, lean_object* v_a_403_, lean_object* v_a_404_, lean_object* v_a_405_){
_start:
{
lean_object* v___x_407_; lean_object* v_lbrak_408_; lean_object* v___x_409_; 
v___x_407_ = lean_unsigned_to_nat(0u);
v_lbrak_408_ = l_Lean_Syntax_getArg(v_rwRulesSeqStx_396_, v___x_407_);
v___x_409_ = l_Lean_Elab_Tactic_mkInitialTacticInfo(v_lbrak_408_, v_a_398_, v_a_399_, v_a_400_, v_a_401_, v_a_402_, v_a_403_, v_a_404_, v_a_405_);
if (lean_obj_tag(v___x_409_) == 0)
{
lean_object* v_a_410_; lean_object* v___f_411_; lean_object* v___x_412_; lean_object* v___f_413_; lean_object* v___x_414_; 
v_a_410_ = lean_ctor_get(v___x_409_, 0);
lean_inc(v_a_410_);
lean_dec_ref_known(v___x_409_, 1);
v___f_411_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_withSimpRWRulesSeq___lam__0___boxed), 11, 1);
lean_closure_set(v___f_411_, 0, v_a_410_);
v___x_412_ = lean_box(0);
v___f_413_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_withSimpRWRulesSeq___closed__0));
v___x_414_ = lp_mathlib_Lean_Elab_withInfoTreeContext___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__0___redArg(v___f_413_, v___f_411_, v_a_398_, v_a_399_, v_a_400_, v_a_401_, v_a_402_, v_a_403_, v_a_404_, v_a_405_);
if (lean_obj_tag(v___x_414_) == 0)
{
lean_object* v___x_415_; lean_object* v___x_416_; lean_object* v_rules_417_; lean_object* v___x_418_; lean_object* v___x_419_; lean_object* v___x_420_; lean_object* v___x_421_; lean_object* v___x_422_; 
lean_dec_ref_known(v___x_414_, 1);
v___x_415_ = lean_unsigned_to_nat(1u);
v___x_416_ = l_Lean_Syntax_getArg(v_rwRulesSeqStx_396_, v___x_415_);
v_rules_417_ = l_Lean_Syntax_getArgs(v___x_416_);
lean_dec(v___x_416_);
v___x_418_ = lean_array_get_size(v_rules_417_);
v___x_419_ = lean_nat_add(v___x_418_, v___x_415_);
v___x_420_ = lean_nat_shiftr(v___x_419_, v___x_415_);
lean_dec(v___x_419_);
v___x_421_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_421_, 0, v___x_407_);
lean_ctor_set(v___x_421_, 1, v___x_420_);
lean_ctor_set(v___x_421_, 2, v___x_415_);
v___x_422_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__1___redArg(v_rules_417_, v_x_397_, v___x_421_, v___x_412_, v___x_407_, v_a_398_, v_a_399_, v_a_400_, v_a_401_, v_a_402_, v_a_403_, v_a_404_, v_a_405_);
lean_dec_ref_known(v___x_421_, 3);
lean_dec_ref(v_rules_417_);
if (lean_obj_tag(v___x_422_) == 0)
{
lean_object* v___x_424_; uint8_t v_isShared_425_; uint8_t v_isSharedCheck_429_; 
v_isSharedCheck_429_ = !lean_is_exclusive(v___x_422_);
if (v_isSharedCheck_429_ == 0)
{
lean_object* v_unused_430_; 
v_unused_430_ = lean_ctor_get(v___x_422_, 0);
lean_dec(v_unused_430_);
v___x_424_ = v___x_422_;
v_isShared_425_ = v_isSharedCheck_429_;
goto v_resetjp_423_;
}
else
{
lean_dec(v___x_422_);
v___x_424_ = lean_box(0);
v_isShared_425_ = v_isSharedCheck_429_;
goto v_resetjp_423_;
}
v_resetjp_423_:
{
lean_object* v___x_427_; 
if (v_isShared_425_ == 0)
{
lean_ctor_set(v___x_424_, 0, v___x_412_);
v___x_427_ = v___x_424_;
goto v_reusejp_426_;
}
else
{
lean_object* v_reuseFailAlloc_428_; 
v_reuseFailAlloc_428_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_428_, 0, v___x_412_);
v___x_427_ = v_reuseFailAlloc_428_;
goto v_reusejp_426_;
}
v_reusejp_426_:
{
return v___x_427_;
}
}
}
else
{
return v___x_422_;
}
}
else
{
lean_dec_ref(v_x_397_);
return v___x_414_;
}
}
else
{
lean_object* v_a_431_; lean_object* v___x_433_; uint8_t v_isShared_434_; uint8_t v_isSharedCheck_438_; 
lean_dec_ref(v_x_397_);
v_a_431_ = lean_ctor_get(v___x_409_, 0);
v_isSharedCheck_438_ = !lean_is_exclusive(v___x_409_);
if (v_isSharedCheck_438_ == 0)
{
v___x_433_ = v___x_409_;
v_isShared_434_ = v_isSharedCheck_438_;
goto v_resetjp_432_;
}
else
{
lean_inc(v_a_431_);
lean_dec(v___x_409_);
v___x_433_ = lean_box(0);
v_isShared_434_ = v_isSharedCheck_438_;
goto v_resetjp_432_;
}
v_resetjp_432_:
{
lean_object* v___x_436_; 
if (v_isShared_434_ == 0)
{
v___x_436_ = v___x_433_;
goto v_reusejp_435_;
}
else
{
lean_object* v_reuseFailAlloc_437_; 
v_reuseFailAlloc_437_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_437_, 0, v_a_431_);
v___x_436_ = v_reuseFailAlloc_437_;
goto v_reusejp_435_;
}
v_reusejp_435_:
{
return v___x_436_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_withSimpRWRulesSeq___boxed(lean_object* v_rwRulesSeqStx_439_, lean_object* v_x_440_, lean_object* v_a_441_, lean_object* v_a_442_, lean_object* v_a_443_, lean_object* v_a_444_, lean_object* v_a_445_, lean_object* v_a_446_, lean_object* v_a_447_, lean_object* v_a_448_, lean_object* v_a_449_){
_start:
{
lean_object* v_res_450_; 
v_res_450_ = lp_mathlib_Mathlib_Tactic_withSimpRWRulesSeq(v_rwRulesSeqStx_439_, v_x_440_, v_a_441_, v_a_442_, v_a_443_, v_a_444_, v_a_445_, v_a_446_, v_a_447_, v_a_448_);
lean_dec(v_a_448_);
lean_dec_ref(v_a_447_);
lean_dec(v_a_446_);
lean_dec_ref(v_a_445_);
lean_dec(v_a_444_);
lean_dec_ref(v_a_443_);
lean_dec(v_a_442_);
lean_dec_ref(v_a_441_);
lean_dec(v_rwRulesSeqStx_439_);
return v_res_450_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getResetInfoTrees___at___00Lean_Elab_withInfoTreeContext___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__0_spec__0(lean_object* v___y_451_, lean_object* v___y_452_, lean_object* v___y_453_, lean_object* v___y_454_, lean_object* v___y_455_, lean_object* v___y_456_, lean_object* v___y_457_, lean_object* v___y_458_){
_start:
{
lean_object* v___x_460_; 
v___x_460_ = lp_mathlib_Lean_Elab_getResetInfoTrees___at___00Lean_Elab_withInfoTreeContext___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__0_spec__0___redArg(v___y_458_);
return v___x_460_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getResetInfoTrees___at___00Lean_Elab_withInfoTreeContext___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__0_spec__0___boxed(lean_object* v___y_461_, lean_object* v___y_462_, lean_object* v___y_463_, lean_object* v___y_464_, lean_object* v___y_465_, lean_object* v___y_466_, lean_object* v___y_467_, lean_object* v___y_468_, lean_object* v___y_469_){
_start:
{
lean_object* v_res_470_; 
v_res_470_ = lp_mathlib_Lean_Elab_getResetInfoTrees___at___00Lean_Elab_withInfoTreeContext___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__0_spec__0(v___y_461_, v___y_462_, v___y_463_, v___y_464_, v___y_465_, v___y_466_, v___y_467_, v___y_468_);
lean_dec(v___y_468_);
lean_dec_ref(v___y_467_);
lean_dec(v___y_466_);
lean_dec_ref(v___y_465_);
lean_dec(v___y_464_);
lean_dec_ref(v___y_463_);
lean_dec(v___y_462_);
lean_dec_ref(v___y_461_);
return v_res_470_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withInfoTreeContext___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__0(lean_object* v_00_u03b1_471_, lean_object* v_x_472_, lean_object* v_mkInfoTree_473_, lean_object* v___y_474_, lean_object* v___y_475_, lean_object* v___y_476_, lean_object* v___y_477_, lean_object* v___y_478_, lean_object* v___y_479_, lean_object* v___y_480_, lean_object* v___y_481_){
_start:
{
lean_object* v___x_483_; 
v___x_483_ = lp_mathlib_Lean_Elab_withInfoTreeContext___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__0___redArg(v_x_472_, v_mkInfoTree_473_, v___y_474_, v___y_475_, v___y_476_, v___y_477_, v___y_478_, v___y_479_, v___y_480_, v___y_481_);
return v___x_483_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withInfoTreeContext___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__0___boxed(lean_object* v_00_u03b1_484_, lean_object* v_x_485_, lean_object* v_mkInfoTree_486_, lean_object* v___y_487_, lean_object* v___y_488_, lean_object* v___y_489_, lean_object* v___y_490_, lean_object* v___y_491_, lean_object* v___y_492_, lean_object* v___y_493_, lean_object* v___y_494_, lean_object* v___y_495_){
_start:
{
lean_object* v_res_496_; 
v_res_496_ = lp_mathlib_Lean_Elab_withInfoTreeContext___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__0(v_00_u03b1_484_, v_x_485_, v_mkInfoTree_486_, v___y_487_, v___y_488_, v___y_489_, v___y_490_, v___y_491_, v___y_492_, v___y_493_, v___y_494_);
lean_dec(v___y_494_);
lean_dec_ref(v___y_493_);
lean_dec(v___y_492_);
lean_dec_ref(v___y_491_);
lean_dec(v___y_490_);
lean_dec_ref(v___y_489_);
lean_dec(v___y_488_);
lean_dec_ref(v___y_487_);
return v_res_496_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__1(lean_object* v_rules_497_, lean_object* v_x_498_, lean_object* v_range_499_, lean_object* v_b_500_, lean_object* v_i_501_, lean_object* v_hs_502_, lean_object* v_hl_503_, lean_object* v___y_504_, lean_object* v___y_505_, lean_object* v___y_506_, lean_object* v___y_507_, lean_object* v___y_508_, lean_object* v___y_509_, lean_object* v___y_510_, lean_object* v___y_511_){
_start:
{
lean_object* v___x_513_; 
v___x_513_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__1___redArg(v_rules_497_, v_x_498_, v_range_499_, v_b_500_, v_i_501_, v___y_504_, v___y_505_, v___y_506_, v___y_507_, v___y_508_, v___y_509_, v___y_510_, v___y_511_);
return v___x_513_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__1___boxed(lean_object* v_rules_514_, lean_object* v_x_515_, lean_object* v_range_516_, lean_object* v_b_517_, lean_object* v_i_518_, lean_object* v_hs_519_, lean_object* v_hl_520_, lean_object* v___y_521_, lean_object* v___y_522_, lean_object* v___y_523_, lean_object* v___y_524_, lean_object* v___y_525_, lean_object* v___y_526_, lean_object* v___y_527_, lean_object* v___y_528_, lean_object* v___y_529_){
_start:
{
lean_object* v_res_530_; 
v_res_530_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__1(v_rules_514_, v_x_515_, v_range_516_, v_b_517_, v_i_518_, v_hs_519_, v_hl_520_, v___y_521_, v___y_522_, v___y_523_, v___y_524_, v___y_525_, v___y_526_, v___y_527_, v___y_528_);
lean_dec(v___y_528_);
lean_dec_ref(v___y_527_);
lean_dec(v___y_526_);
lean_dec_ref(v___y_525_);
lean_dec(v___y_524_);
lean_dec_ref(v___y_523_);
lean_dec(v___y_522_);
lean_dec_ref(v___y_521_);
lean_dec_ref(v_range_516_);
lean_dec_ref(v_rules_514_);
return v_res_530_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__8(void){
_start:
{
lean_object* v___x_545_; lean_object* v___x_546_; lean_object* v___x_547_; lean_object* v___x_548_; 
v___x_545_ = l_Lean_Parser_Tactic_optConfig;
v___x_546_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__7));
v___x_547_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__5));
v___x_548_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_548_, 0, v___x_547_);
lean_ctor_set(v___x_548_, 1, v___x_546_);
lean_ctor_set(v___x_548_, 2, v___x_545_);
return v___x_548_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__9(void){
_start:
{
lean_object* v___x_549_; lean_object* v___x_550_; lean_object* v___x_551_; lean_object* v___x_552_; 
v___x_549_ = l_Lean_Parser_Tactic_rwRuleSeq;
v___x_550_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__8, &lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__8_once, _init_lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__8);
v___x_551_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__5));
v___x_552_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_552_, 0, v___x_551_);
lean_ctor_set(v___x_552_, 1, v___x_550_);
lean_ctor_set(v___x_552_, 2, v___x_549_);
return v___x_552_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__12(void){
_start:
{
lean_object* v___x_556_; lean_object* v___x_557_; lean_object* v___x_558_; 
v___x_556_ = l_Lean_Parser_Tactic_location;
v___x_557_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__11));
v___x_558_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_558_, 0, v___x_557_);
lean_ctor_set(v___x_558_, 1, v___x_556_);
return v___x_558_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__13(void){
_start:
{
lean_object* v___x_559_; lean_object* v___x_560_; lean_object* v___x_561_; lean_object* v___x_562_; 
v___x_559_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__12, &lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__12_once, _init_lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__12);
v___x_560_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__9, &lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__9_once, _init_lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__9);
v___x_561_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__5));
v___x_562_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_562_, 0, v___x_561_);
lean_ctor_set(v___x_562_, 1, v___x_560_);
lean_ctor_set(v___x_562_, 2, v___x_559_);
return v___x_562_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__14(void){
_start:
{
lean_object* v___x_563_; lean_object* v___x_564_; lean_object* v___x_565_; lean_object* v___x_566_; 
v___x_563_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__13, &lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__13_once, _init_lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__13);
v___x_564_ = lean_unsigned_to_nat(1022u);
v___x_565_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__3));
v___x_566_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_566_, 0, v___x_565_);
lean_ctor_set(v___x_566_, 1, v___x_564_);
lean_ctor_set(v___x_566_, 2, v___x_563_);
return v___x_566_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticSimp__rw______(void){
_start:
{
lean_object* v___x_567_; 
v___x_567_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__14, &lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__14_once, _init_lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__14);
return v___x_567_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_568_; lean_object* v___x_569_; lean_object* v___x_570_; 
v___x_568_ = lean_box(0);
v___x_569_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_570_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_570_, 0, v___x_569_);
lean_ctor_set(v___x_570_, 1, v___x_568_);
return v___x_570_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1_spec__0___redArg(){
_start:
{
lean_object* v___x_572_; lean_object* v___x_573_; 
v___x_572_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1_spec__0___redArg___closed__0);
v___x_573_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_573_, 0, v___x_572_);
return v___x_573_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1_spec__0___redArg___boxed(lean_object* v___y_574_){
_start:
{
lean_object* v_res_575_; 
v_res_575_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1_spec__0___redArg();
return v_res_575_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1_spec__0(lean_object* v_00_u03b1_576_, lean_object* v___y_577_, lean_object* v___y_578_, lean_object* v___y_579_, lean_object* v___y_580_, lean_object* v___y_581_, lean_object* v___y_582_, lean_object* v___y_583_, lean_object* v___y_584_){
_start:
{
lean_object* v___x_586_; 
v___x_586_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1_spec__0___redArg();
return v___x_586_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1_spec__0___boxed(lean_object* v_00_u03b1_587_, lean_object* v___y_588_, lean_object* v___y_589_, lean_object* v___y_590_, lean_object* v___y_591_, lean_object* v___y_592_, lean_object* v___y_593_, lean_object* v___y_594_, lean_object* v___y_595_, lean_object* v___y_596_){
_start:
{
lean_object* v_res_597_; 
v_res_597_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1_spec__0(v_00_u03b1_587_, v___y_588_, v___y_589_, v___y_590_, v___y_591_, v___y_592_, v___y_593_, v___y_594_, v___y_595_);
lean_dec(v___y_595_);
lean_dec_ref(v___y_594_);
lean_dec(v___y_593_);
lean_dec_ref(v___y_592_);
lean_dec(v___y_591_);
lean_dec_ref(v___y_590_);
lean_dec(v___y_589_);
lean_dec_ref(v___y_588_);
return v_res_597_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__0(lean_object* v___y_598_, lean_object* v___y_599_, lean_object* v___y_600_, lean_object* v___y_601_, lean_object* v___y_602_, lean_object* v___y_603_, lean_object* v___y_604_, lean_object* v___y_605_){
_start:
{
lean_object* v_ref_607_; uint8_t v___x_608_; lean_object* v___x_609_; lean_object* v___x_610_; 
v_ref_607_ = lean_ctor_get(v___y_604_, 5);
v___x_608_ = 0;
v___x_609_ = l_Lean_SourceInfo_fromRef(v_ref_607_, v___x_608_);
v___x_610_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_610_, 0, v___x_609_);
return v___x_610_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__0___boxed(lean_object* v___y_611_, lean_object* v___y_612_, lean_object* v___y_613_, lean_object* v___y_614_, lean_object* v___y_615_, lean_object* v___y_616_, lean_object* v___y_617_, lean_object* v___y_618_, lean_object* v___y_619_){
_start:
{
lean_object* v_res_620_; 
v_res_620_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__0(v___y_611_, v___y_612_, v___y_613_, v___y_614_, v___y_615_, v___y_616_, v___y_617_, v___y_618_);
lean_dec(v___y_618_);
lean_dec_ref(v___y_617_);
lean_dec(v___y_616_);
lean_dec_ref(v___y_615_);
lean_dec(v___y_614_);
lean_dec_ref(v___y_613_);
lean_dec(v___y_612_);
lean_dec_ref(v___y_611_);
return v_res_620_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__1___closed__1(void){
_start:
{
lean_object* v___x_622_; 
v___x_622_ = l_Array_mkArray0(lean_box(0));
return v___x_622_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__1(lean_object* v___f_628_, lean_object* v___x_629_, lean_object* v___x_630_, lean_object* v___x_631_, uint8_t v___x_632_, lean_object* v___x_633_, lean_object* v___y_634_, lean_object* v___x_635_, uint8_t v_symm_636_, lean_object* v_term_637_, lean_object* v___y_638_, lean_object* v___y_639_, lean_object* v___y_640_, lean_object* v___y_641_, lean_object* v___y_642_, lean_object* v___y_643_, lean_object* v___y_644_, lean_object* v___y_645_){
_start:
{
if (v_symm_636_ == 0)
{
lean_object* v___x_647_; 
lean_inc(v___y_645_);
lean_inc_ref(v___y_644_);
lean_inc(v___y_643_);
lean_inc_ref(v___y_642_);
lean_inc(v___y_641_);
lean_inc_ref(v___y_640_);
lean_inc(v___y_639_);
lean_inc_ref(v___y_638_);
v___x_647_ = lean_apply_9(v___f_628_, v___y_638_, v___y_639_, v___y_640_, v___y_641_, v___y_642_, v___y_643_, v___y_644_, v___y_645_, lean_box(0));
if (lean_obj_tag(v___x_647_) == 0)
{
lean_object* v_a_648_; lean_object* v___x_649_; lean_object* v___x_650_; lean_object* v___x_651_; lean_object* v___x_652_; lean_object* v___x_653_; lean_object* v___x_654_; lean_object* v___x_655_; lean_object* v___x_656_; lean_object* v___x_657_; lean_object* v___x_658_; lean_object* v___x_659_; lean_object* v___x_660_; lean_object* v___x_661_; lean_object* v___x_662_; lean_object* v___x_663_; lean_object* v___x_664_; lean_object* v___x_665_; lean_object* v___x_666_; lean_object* v___x_667_; lean_object* v___y_669_; 
v_a_648_ = lean_ctor_get(v___x_647_, 0);
lean_inc_n(v_a_648_, 9);
lean_dec_ref_known(v___x_647_, 1);
v___x_649_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__1___closed__0));
lean_inc_ref(v___x_631_);
lean_inc_ref(v___x_630_);
lean_inc_ref(v___x_629_);
v___x_650_ = l_Lean_Name_mkStr4(v___x_629_, v___x_630_, v___x_631_, v___x_649_);
v___x_651_ = l_Lean_SourceInfo_fromRef(v_term_637_, v___x_632_);
v___x_652_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_652_, 0, v___x_651_);
lean_ctor_set(v___x_652_, 1, v___x_649_);
v___x_653_ = ((lean_object*)(lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__1___redArg___closed__1));
v___x_654_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__1___closed__1, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__1___closed__1_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__1___closed__1);
v___x_655_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_655_, 0, v_a_648_);
lean_ctor_set(v___x_655_, 1, v___x_653_);
lean_ctor_set(v___x_655_, 2, v___x_654_);
v___x_656_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__1___closed__2));
v___x_657_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_657_, 0, v_a_648_);
lean_ctor_set(v___x_657_, 1, v___x_656_);
v___x_658_ = l_Lean_Syntax_node1(v_a_648_, v___x_653_, v___x_657_);
v___x_659_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__1___closed__3));
v___x_660_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_660_, 0, v_a_648_);
lean_ctor_set(v___x_660_, 1, v___x_659_);
v___x_661_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__1___closed__4));
v___x_662_ = l_Lean_Name_mkStr4(v___x_629_, v___x_630_, v___x_631_, v___x_661_);
lean_inc_ref_n(v___x_655_, 2);
v___x_663_ = l_Lean_Syntax_node3(v_a_648_, v___x_662_, v___x_655_, v___x_655_, v_term_637_);
v___x_664_ = l_Lean_Syntax_node1(v_a_648_, v___x_653_, v___x_663_);
v___x_665_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__1___closed__5));
v___x_666_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_666_, 0, v_a_648_);
lean_ctor_set(v___x_666_, 1, v___x_665_);
v___x_667_ = l_Lean_Syntax_node3(v_a_648_, v___x_653_, v___x_660_, v___x_664_, v___x_666_);
if (lean_obj_tag(v___y_634_) == 0)
{
lean_object* v___x_674_; 
v___x_674_ = lean_mk_empty_array_with_capacity(v___x_635_);
v___y_669_ = v___x_674_;
goto v___jp_668_;
}
else
{
lean_object* v_val_675_; lean_object* v___x_676_; lean_object* v___x_677_; 
v_val_675_ = lean_ctor_get(v___y_634_, 0);
lean_inc(v_val_675_);
lean_dec_ref_known(v___y_634_, 1);
v___x_676_ = lean_mk_empty_array_with_capacity(v___x_635_);
v___x_677_ = lean_array_push(v___x_676_, v_val_675_);
v___y_669_ = v___x_677_;
goto v___jp_668_;
}
v___jp_668_:
{
lean_object* v___x_670_; lean_object* v___x_671_; lean_object* v___x_672_; lean_object* v___x_673_; 
v___x_670_ = l_Array_append___redArg(v___x_654_, v___y_669_);
lean_dec_ref(v___y_669_);
lean_inc(v_a_648_);
v___x_671_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_671_, 0, v_a_648_);
lean_ctor_set(v___x_671_, 1, v___x_653_);
lean_ctor_set(v___x_671_, 2, v___x_670_);
v___x_672_ = l_Lean_Syntax_node6(v_a_648_, v___x_650_, v___x_652_, v___x_633_, v___x_655_, v___x_658_, v___x_667_, v___x_671_);
v___x_673_ = l_Lean_Elab_Tactic_evalTactic(v___x_672_, v___y_638_, v___y_639_, v___y_640_, v___y_641_, v___y_642_, v___y_643_, v___y_644_, v___y_645_);
return v___x_673_;
}
}
else
{
lean_object* v_a_678_; lean_object* v___x_680_; uint8_t v_isShared_681_; uint8_t v_isSharedCheck_685_; 
lean_dec(v_term_637_);
lean_dec(v___y_634_);
lean_dec(v___x_633_);
lean_dec_ref(v___x_631_);
lean_dec_ref(v___x_630_);
lean_dec_ref(v___x_629_);
v_a_678_ = lean_ctor_get(v___x_647_, 0);
v_isSharedCheck_685_ = !lean_is_exclusive(v___x_647_);
if (v_isSharedCheck_685_ == 0)
{
v___x_680_ = v___x_647_;
v_isShared_681_ = v_isSharedCheck_685_;
goto v_resetjp_679_;
}
else
{
lean_inc(v_a_678_);
lean_dec(v___x_647_);
v___x_680_ = lean_box(0);
v_isShared_681_ = v_isSharedCheck_685_;
goto v_resetjp_679_;
}
v_resetjp_679_:
{
lean_object* v___x_683_; 
if (v_isShared_681_ == 0)
{
v___x_683_ = v___x_680_;
goto v_reusejp_682_;
}
else
{
lean_object* v_reuseFailAlloc_684_; 
v_reuseFailAlloc_684_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_684_, 0, v_a_678_);
v___x_683_ = v_reuseFailAlloc_684_;
goto v_reusejp_682_;
}
v_reusejp_682_:
{
return v___x_683_;
}
}
}
}
else
{
lean_object* v___x_686_; 
lean_inc(v___y_645_);
lean_inc_ref(v___y_644_);
lean_inc(v___y_643_);
lean_inc_ref(v___y_642_);
lean_inc(v___y_641_);
lean_inc_ref(v___y_640_);
lean_inc(v___y_639_);
lean_inc_ref(v___y_638_);
v___x_686_ = lean_apply_9(v___f_628_, v___y_638_, v___y_639_, v___y_640_, v___y_641_, v___y_642_, v___y_643_, v___y_644_, v___y_645_, lean_box(0));
if (lean_obj_tag(v___x_686_) == 0)
{
lean_object* v_a_687_; lean_object* v___x_688_; lean_object* v___x_689_; lean_object* v___x_690_; lean_object* v___x_691_; lean_object* v___x_692_; lean_object* v___x_693_; lean_object* v___x_694_; lean_object* v___x_695_; lean_object* v___x_696_; lean_object* v___x_697_; lean_object* v___x_698_; lean_object* v___x_699_; lean_object* v___x_700_; lean_object* v___x_701_; lean_object* v___x_702_; lean_object* v___x_703_; lean_object* v___x_704_; lean_object* v___x_705_; lean_object* v___x_706_; lean_object* v___x_707_; lean_object* v___x_708_; lean_object* v___x_709_; lean_object* v___y_711_; 
v_a_687_ = lean_ctor_get(v___x_686_, 0);
lean_inc_n(v_a_687_, 11);
lean_dec_ref_known(v___x_686_, 1);
v___x_688_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__1___closed__0));
lean_inc_ref(v___x_631_);
lean_inc_ref(v___x_630_);
lean_inc_ref(v___x_629_);
v___x_689_ = l_Lean_Name_mkStr4(v___x_629_, v___x_630_, v___x_631_, v___x_688_);
v___x_690_ = l_Lean_SourceInfo_fromRef(v_term_637_, v___x_632_);
v___x_691_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_691_, 0, v___x_690_);
lean_ctor_set(v___x_691_, 1, v___x_688_);
v___x_692_ = ((lean_object*)(lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__1___redArg___closed__1));
v___x_693_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__1___closed__1, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__1___closed__1_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__1___closed__1);
v___x_694_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_694_, 0, v_a_687_);
lean_ctor_set(v___x_694_, 1, v___x_692_);
lean_ctor_set(v___x_694_, 2, v___x_693_);
v___x_695_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__1___closed__2));
v___x_696_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_696_, 0, v_a_687_);
lean_ctor_set(v___x_696_, 1, v___x_695_);
v___x_697_ = l_Lean_Syntax_node1(v_a_687_, v___x_692_, v___x_696_);
v___x_698_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__1___closed__3));
v___x_699_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_699_, 0, v_a_687_);
lean_ctor_set(v___x_699_, 1, v___x_698_);
v___x_700_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__1___closed__4));
v___x_701_ = l_Lean_Name_mkStr4(v___x_629_, v___x_630_, v___x_631_, v___x_700_);
v___x_702_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__1___closed__6));
v___x_703_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_703_, 0, v_a_687_);
lean_ctor_set(v___x_703_, 1, v___x_702_);
v___x_704_ = l_Lean_Syntax_node1(v_a_687_, v___x_692_, v___x_703_);
lean_inc_ref(v___x_694_);
v___x_705_ = l_Lean_Syntax_node3(v_a_687_, v___x_701_, v___x_694_, v___x_704_, v_term_637_);
v___x_706_ = l_Lean_Syntax_node1(v_a_687_, v___x_692_, v___x_705_);
v___x_707_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__1___closed__5));
v___x_708_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_708_, 0, v_a_687_);
lean_ctor_set(v___x_708_, 1, v___x_707_);
v___x_709_ = l_Lean_Syntax_node3(v_a_687_, v___x_692_, v___x_699_, v___x_706_, v___x_708_);
if (lean_obj_tag(v___y_634_) == 0)
{
lean_object* v___x_716_; 
v___x_716_ = lean_mk_empty_array_with_capacity(v___x_635_);
v___y_711_ = v___x_716_;
goto v___jp_710_;
}
else
{
lean_object* v_val_717_; lean_object* v___x_718_; lean_object* v___x_719_; 
v_val_717_ = lean_ctor_get(v___y_634_, 0);
lean_inc(v_val_717_);
lean_dec_ref_known(v___y_634_, 1);
v___x_718_ = lean_mk_empty_array_with_capacity(v___x_635_);
v___x_719_ = lean_array_push(v___x_718_, v_val_717_);
v___y_711_ = v___x_719_;
goto v___jp_710_;
}
v___jp_710_:
{
lean_object* v___x_712_; lean_object* v___x_713_; lean_object* v___x_714_; lean_object* v___x_715_; 
v___x_712_ = l_Array_append___redArg(v___x_693_, v___y_711_);
lean_dec_ref(v___y_711_);
lean_inc(v_a_687_);
v___x_713_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_713_, 0, v_a_687_);
lean_ctor_set(v___x_713_, 1, v___x_692_);
lean_ctor_set(v___x_713_, 2, v___x_712_);
v___x_714_ = l_Lean_Syntax_node6(v_a_687_, v___x_689_, v___x_691_, v___x_633_, v___x_694_, v___x_697_, v___x_709_, v___x_713_);
v___x_715_ = l_Lean_Elab_Tactic_evalTactic(v___x_714_, v___y_638_, v___y_639_, v___y_640_, v___y_641_, v___y_642_, v___y_643_, v___y_644_, v___y_645_);
return v___x_715_;
}
}
else
{
lean_object* v_a_720_; lean_object* v___x_722_; uint8_t v_isShared_723_; uint8_t v_isSharedCheck_727_; 
lean_dec(v_term_637_);
lean_dec(v___y_634_);
lean_dec(v___x_633_);
lean_dec_ref(v___x_631_);
lean_dec_ref(v___x_630_);
lean_dec_ref(v___x_629_);
v_a_720_ = lean_ctor_get(v___x_686_, 0);
v_isSharedCheck_727_ = !lean_is_exclusive(v___x_686_);
if (v_isSharedCheck_727_ == 0)
{
v___x_722_ = v___x_686_;
v_isShared_723_ = v_isSharedCheck_727_;
goto v_resetjp_721_;
}
else
{
lean_inc(v_a_720_);
lean_dec(v___x_686_);
v___x_722_ = lean_box(0);
v_isShared_723_ = v_isSharedCheck_727_;
goto v_resetjp_721_;
}
v_resetjp_721_:
{
lean_object* v___x_725_; 
if (v_isShared_723_ == 0)
{
v___x_725_ = v___x_722_;
goto v_reusejp_724_;
}
else
{
lean_object* v_reuseFailAlloc_726_; 
v_reuseFailAlloc_726_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_726_, 0, v_a_720_);
v___x_725_ = v_reuseFailAlloc_726_;
goto v_reusejp_724_;
}
v_reusejp_724_:
{
return v___x_725_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__1___boxed(lean_object** _args){
lean_object* v___f_728_ = _args[0];
lean_object* v___x_729_ = _args[1];
lean_object* v___x_730_ = _args[2];
lean_object* v___x_731_ = _args[3];
lean_object* v___x_732_ = _args[4];
lean_object* v___x_733_ = _args[5];
lean_object* v___y_734_ = _args[6];
lean_object* v___x_735_ = _args[7];
lean_object* v_symm_736_ = _args[8];
lean_object* v_term_737_ = _args[9];
lean_object* v___y_738_ = _args[10];
lean_object* v___y_739_ = _args[11];
lean_object* v___y_740_ = _args[12];
lean_object* v___y_741_ = _args[13];
lean_object* v___y_742_ = _args[14];
lean_object* v___y_743_ = _args[15];
lean_object* v___y_744_ = _args[16];
lean_object* v___y_745_ = _args[17];
lean_object* v___y_746_ = _args[18];
_start:
{
uint8_t v___x_7968__boxed_747_; uint8_t v_symm_boxed_748_; lean_object* v_res_749_; 
v___x_7968__boxed_747_ = lean_unbox(v___x_732_);
v_symm_boxed_748_ = lean_unbox(v_symm_736_);
v_res_749_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__1(v___f_728_, v___x_729_, v___x_730_, v___x_731_, v___x_7968__boxed_747_, v___x_733_, v___y_734_, v___x_735_, v_symm_boxed_748_, v_term_737_, v___y_738_, v___y_739_, v___y_740_, v___y_741_, v___y_742_, v___y_743_, v___y_744_, v___y_745_);
lean_dec(v___y_745_);
lean_dec_ref(v___y_744_);
lean_dec(v___y_743_);
lean_dec_ref(v___y_742_);
lean_dec(v___y_741_);
lean_dec_ref(v___y_740_);
lean_dec(v___y_739_);
lean_dec_ref(v___y_738_);
lean_dec(v___x_735_);
return v_res_749_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1_spec__1(size_t v_sz_750_, size_t v_i_751_, lean_object* v_bs_752_){
_start:
{
uint8_t v___x_753_; 
v___x_753_ = lean_usize_dec_lt(v_i_751_, v_sz_750_);
if (v___x_753_ == 0)
{
return v_bs_752_;
}
else
{
lean_object* v_v_754_; lean_object* v___x_755_; lean_object* v_bs_x27_756_; size_t v___x_757_; size_t v___x_758_; lean_object* v___x_759_; 
v_v_754_ = lean_array_uget(v_bs_752_, v_i_751_);
v___x_755_ = lean_unsigned_to_nat(0u);
v_bs_x27_756_ = lean_array_uset(v_bs_752_, v_i_751_, v___x_755_);
v___x_757_ = ((size_t)1ULL);
v___x_758_ = lean_usize_add(v_i_751_, v___x_757_);
v___x_759_ = lean_array_uset(v_bs_x27_756_, v_i_751_, v_v_754_);
v_i_751_ = v___x_758_;
v_bs_752_ = v___x_759_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1_spec__1___boxed(lean_object* v_sz_761_, lean_object* v_i_762_, lean_object* v_bs_763_){
_start:
{
size_t v_sz_boxed_764_; size_t v_i_boxed_765_; lean_object* v_res_766_; 
v_sz_boxed_764_ = lean_unbox_usize(v_sz_761_);
lean_dec(v_sz_761_);
v_i_boxed_765_ = lean_unbox_usize(v_i_762_);
lean_dec(v_i_762_);
v_res_766_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1_spec__1(v_sz_boxed_764_, v_i_boxed_765_, v_bs_763_);
return v_res_766_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__4(void){
_start:
{
lean_object* v___x_771_; lean_object* v___x_772_; 
v___x_771_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__3));
v___x_772_ = l_String_toRawSubstring_x27(v___x_771_);
return v___x_772_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__8(void){
_start:
{
lean_object* v___x_777_; lean_object* v___x_778_; 
v___x_777_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__7));
v___x_778_ = l_String_toRawSubstring_x27(v___x_777_);
return v___x_778_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2(lean_object* v___x_792_, lean_object* v___x_793_, lean_object* v___x_794_, lean_object* v_s_795_, uint8_t v___x_796_, lean_object* v___x_797_, lean_object* v___x_798_, lean_object* v___x_799_, lean_object* v___f_800_, lean_object* v___y_801_, lean_object* v___x_802_, lean_object* v___y_803_, lean_object* v___y_804_, lean_object* v___y_805_, lean_object* v___y_806_, lean_object* v___y_807_, lean_object* v___y_808_, lean_object* v___y_809_, lean_object* v___y_810_){
_start:
{
lean_object* v_ref_812_; lean_object* v_quotContext_813_; lean_object* v_currMacroScope_814_; uint8_t v___x_815_; lean_object* v___x_816_; lean_object* v___x_817_; lean_object* v___x_818_; lean_object* v___x_819_; lean_object* v___x_820_; lean_object* v___x_821_; lean_object* v___x_822_; lean_object* v___x_823_; size_t v_sz_824_; size_t v___x_825_; lean_object* v___x_826_; lean_object* v___x_827_; lean_object* v___x_828_; lean_object* v___x_829_; lean_object* v___x_830_; lean_object* v___x_831_; lean_object* v___x_832_; lean_object* v___x_833_; lean_object* v___x_834_; lean_object* v___x_835_; lean_object* v___x_836_; lean_object* v___x_837_; lean_object* v___x_838_; lean_object* v___x_839_; lean_object* v___x_840_; lean_object* v___x_841_; lean_object* v___x_842_; lean_object* v___x_843_; lean_object* v___x_844_; lean_object* v___x_845_; lean_object* v___x_846_; lean_object* v___x_847_; lean_object* v___x_848_; lean_object* v___x_849_; lean_object* v___x_850_; lean_object* v___x_851_; lean_object* v___x_852_; lean_object* v___x_853_; lean_object* v___x_854_; lean_object* v___x_855_; lean_object* v___x_856_; lean_object* v___y_858_; 
v_ref_812_ = lean_ctor_get(v___y_809_, 5);
v_quotContext_813_ = lean_ctor_get(v___y_809_, 10);
v_currMacroScope_814_ = lean_ctor_get(v___y_809_, 11);
v___x_815_ = 0;
v___x_816_ = l_Lean_SourceInfo_fromRef(v_ref_812_, v___x_815_);
v___x_817_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__1___closed__0));
lean_inc_ref_n(v___x_794_, 2);
lean_inc_ref_n(v___x_793_, 2);
lean_inc_ref_n(v___x_792_, 2);
v___x_818_ = l_Lean_Name_mkStr4(v___x_792_, v___x_793_, v___x_794_, v___x_817_);
v___x_819_ = l_Lean_SourceInfo_fromRef(v_s_795_, v___x_796_);
v___x_820_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_820_, 0, v___x_819_);
lean_ctor_set(v___x_820_, 1, v___x_817_);
v___x_821_ = ((lean_object*)(lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_withSimpRWRulesSeq_spec__1___redArg___closed__1));
v___x_822_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__1___closed__1, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__1___closed__1_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__1___closed__1);
v___x_823_ = l_Lean_Parser_Tactic_getConfigItems(v___x_797_);
v_sz_824_ = lean_array_size(v___x_823_);
v___x_825_ = ((size_t)0ULL);
v___x_826_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1_spec__1(v_sz_824_, v___x_825_, v___x_823_);
v___x_827_ = l_Array_append___redArg(v___x_822_, v___x_826_);
lean_dec_ref(v___x_826_);
v___x_828_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__0));
v___x_829_ = l_Lean_Name_mkStr4(v___x_792_, v___x_793_, v___x_794_, v___x_828_);
v___x_830_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__1));
v___x_831_ = l_Lean_Name_mkStr4(v___x_792_, v___x_793_, v___x_794_, v___x_830_);
v___x_832_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__2));
lean_inc_n(v___x_816_, 12);
v___x_833_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_833_, 0, v___x_816_);
lean_ctor_set(v___x_833_, 1, v___x_832_);
v___x_834_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__4, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__4_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__4);
v___x_835_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__5));
lean_inc_n(v_currMacroScope_814_, 2);
lean_inc_n(v_quotContext_813_, 2);
v___x_836_ = l_Lean_addMacroScope(v_quotContext_813_, v___x_835_, v_currMacroScope_814_);
v___x_837_ = lean_box(0);
v___x_838_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_838_, 0, v___x_816_);
lean_ctor_set(v___x_838_, 1, v___x_834_);
lean_ctor_set(v___x_838_, 2, v___x_836_);
lean_ctor_set(v___x_838_, 3, v___x_837_);
v___x_839_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__6));
v___x_840_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_840_, 0, v___x_816_);
lean_ctor_set(v___x_840_, 1, v___x_839_);
v___x_841_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__8, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__8_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__8);
v___x_842_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__9));
v___x_843_ = l_Lean_addMacroScope(v_quotContext_813_, v___x_842_, v_currMacroScope_814_);
v___x_844_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__13));
v___x_845_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_845_, 0, v___x_816_);
lean_ctor_set(v___x_845_, 1, v___x_841_);
lean_ctor_set(v___x_845_, 2, v___x_843_);
lean_ctor_set(v___x_845_, 3, v___x_844_);
v___x_846_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___closed__14));
v___x_847_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_847_, 0, v___x_816_);
lean_ctor_set(v___x_847_, 1, v___x_846_);
v___x_848_ = l_Lean_Syntax_node5(v___x_816_, v___x_831_, v___x_833_, v___x_838_, v___x_840_, v___x_845_, v___x_847_);
v___x_849_ = l_Lean_Syntax_node1(v___x_816_, v___x_829_, v___x_848_);
v___x_850_ = lean_array_push(v___x_827_, v___x_849_);
v___x_851_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_851_, 0, v___x_816_);
lean_ctor_set(v___x_851_, 1, v___x_821_);
lean_ctor_set(v___x_851_, 2, v___x_850_);
v___x_852_ = l_Lean_Syntax_node1(v___x_816_, v___x_798_, v___x_851_);
v___x_853_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_853_, 0, v___x_816_);
lean_ctor_set(v___x_853_, 1, v___x_821_);
lean_ctor_set(v___x_853_, 2, v___x_822_);
v___x_854_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__1___closed__2));
v___x_855_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_855_, 0, v___x_816_);
lean_ctor_set(v___x_855_, 1, v___x_854_);
v___x_856_ = l_Lean_Syntax_node1(v___x_816_, v___x_821_, v___x_855_);
if (lean_obj_tag(v___y_801_) == 0)
{
lean_object* v___x_864_; 
v___x_864_ = lean_mk_empty_array_with_capacity(v___x_802_);
v___y_858_ = v___x_864_;
goto v___jp_857_;
}
else
{
lean_object* v_val_865_; lean_object* v___x_866_; lean_object* v___x_867_; 
v_val_865_ = lean_ctor_get(v___y_801_, 0);
lean_inc(v_val_865_);
lean_dec_ref_known(v___y_801_, 1);
v___x_866_ = lean_mk_empty_array_with_capacity(v___x_802_);
v___x_867_ = lean_array_push(v___x_866_, v_val_865_);
v___y_858_ = v___x_867_;
goto v___jp_857_;
}
v___jp_857_:
{
lean_object* v___x_859_; lean_object* v___x_860_; lean_object* v___x_861_; lean_object* v___x_862_; 
v___x_859_ = l_Array_append___redArg(v___x_822_, v___y_858_);
lean_dec_ref(v___y_858_);
lean_inc(v___x_816_);
v___x_860_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_860_, 0, v___x_816_);
lean_ctor_set(v___x_860_, 1, v___x_821_);
lean_ctor_set(v___x_860_, 2, v___x_859_);
lean_inc_ref(v___x_853_);
v___x_861_ = l_Lean_Syntax_node6(v___x_816_, v___x_818_, v___x_820_, v___x_852_, v___x_853_, v___x_856_, v___x_853_, v___x_860_);
v___x_862_ = l_Lean_Elab_Tactic_evalTactic(v___x_861_, v___y_803_, v___y_804_, v___y_805_, v___y_806_, v___y_807_, v___y_808_, v___y_809_, v___y_810_);
if (lean_obj_tag(v___x_862_) == 0)
{
lean_object* v___x_863_; 
lean_dec_ref_known(v___x_862_, 1);
v___x_863_ = lp_mathlib_Mathlib_Tactic_withSimpRWRulesSeq(v___x_799_, v___f_800_, v___y_803_, v___y_804_, v___y_805_, v___y_806_, v___y_807_, v___y_808_, v___y_809_, v___y_810_);
lean_dec_ref(v___y_809_);
return v___x_863_;
}
else
{
lean_dec_ref(v___y_809_);
lean_dec_ref(v___f_800_);
return v___x_862_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___boxed(lean_object** _args){
lean_object* v___x_868_ = _args[0];
lean_object* v___x_869_ = _args[1];
lean_object* v___x_870_ = _args[2];
lean_object* v_s_871_ = _args[3];
lean_object* v___x_872_ = _args[4];
lean_object* v___x_873_ = _args[5];
lean_object* v___x_874_ = _args[6];
lean_object* v___x_875_ = _args[7];
lean_object* v___f_876_ = _args[8];
lean_object* v___y_877_ = _args[9];
lean_object* v___x_878_ = _args[10];
lean_object* v___y_879_ = _args[11];
lean_object* v___y_880_ = _args[12];
lean_object* v___y_881_ = _args[13];
lean_object* v___y_882_ = _args[14];
lean_object* v___y_883_ = _args[15];
lean_object* v___y_884_ = _args[16];
lean_object* v___y_885_ = _args[17];
lean_object* v___y_886_ = _args[18];
lean_object* v___y_887_ = _args[19];
_start:
{
uint8_t v___x_8255__boxed_888_; lean_object* v_res_889_; 
v___x_8255__boxed_888_ = lean_unbox(v___x_872_);
v_res_889_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2(v___x_868_, v___x_869_, v___x_870_, v_s_871_, v___x_8255__boxed_888_, v___x_873_, v___x_874_, v___x_875_, v___f_876_, v___y_877_, v___x_878_, v___y_879_, v___y_880_, v___y_881_, v___y_882_, v___y_883_, v___y_884_, v___y_885_, v___y_886_);
lean_dec(v___y_886_);
lean_dec(v___y_884_);
lean_dec_ref(v___y_883_);
lean_dec(v___y_882_);
lean_dec_ref(v___y_881_);
lean_dec(v___y_880_);
lean_dec_ref(v___y_879_);
lean_dec(v___x_878_);
lean_dec(v___x_875_);
lean_dec(v_s_871_);
return v_res_889_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1(lean_object* v_x_899_, lean_object* v_a_900_, lean_object* v_a_901_, lean_object* v_a_902_, lean_object* v_a_903_, lean_object* v_a_904_, lean_object* v_a_905_, lean_object* v_a_906_, lean_object* v_a_907_){
_start:
{
lean_object* v___x_909_; lean_object* v___x_910_; uint8_t v___x_911_; 
v___x_909_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__1));
v___x_910_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticSimp__rw_______00__closed__3));
lean_inc(v_x_899_);
v___x_911_ = l_Lean_Syntax_isOfKind(v_x_899_, v___x_910_);
if (v___x_911_ == 0)
{
lean_object* v___x_912_; 
lean_dec(v_x_899_);
v___x_912_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1_spec__0___redArg();
return v___x_912_;
}
else
{
lean_object* v___f_913_; lean_object* v___x_914_; lean_object* v_s_915_; lean_object* v___x_916_; lean_object* v___x_917_; lean_object* v___x_918_; lean_object* v___x_919_; lean_object* v___y_921_; lean_object* v___x_930_; lean_object* v___x_931_; lean_object* v___x_932_; 
v___f_913_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___closed__0));
v___x_914_ = lean_unsigned_to_nat(0u);
v_s_915_ = l_Lean_Syntax_getArg(v_x_899_, v___x_914_);
v___x_916_ = lean_unsigned_to_nat(1u);
v___x_917_ = l_Lean_Syntax_getArg(v_x_899_, v___x_916_);
v___x_918_ = lean_unsigned_to_nat(2u);
v___x_919_ = l_Lean_Syntax_getArg(v_x_899_, v___x_918_);
v___x_930_ = lean_unsigned_to_nat(3u);
v___x_931_ = l_Lean_Syntax_getArg(v_x_899_, v___x_930_);
lean_dec(v_x_899_);
v___x_932_ = l_Lean_Syntax_getOptional_x3f(v___x_931_);
lean_dec(v___x_931_);
if (lean_obj_tag(v___x_932_) == 0)
{
lean_object* v___x_933_; 
v___x_933_ = lean_box(0);
v___y_921_ = v___x_933_;
goto v___jp_920_;
}
else
{
lean_object* v_val_934_; lean_object* v___x_936_; uint8_t v_isShared_937_; uint8_t v_isSharedCheck_941_; 
v_val_934_ = lean_ctor_get(v___x_932_, 0);
v_isSharedCheck_941_ = !lean_is_exclusive(v___x_932_);
if (v_isSharedCheck_941_ == 0)
{
v___x_936_ = v___x_932_;
v_isShared_937_ = v_isSharedCheck_941_;
goto v_resetjp_935_;
}
else
{
lean_inc(v_val_934_);
lean_dec(v___x_932_);
v___x_936_ = lean_box(0);
v_isShared_937_ = v_isSharedCheck_941_;
goto v_resetjp_935_;
}
v_resetjp_935_:
{
lean_object* v___x_939_; 
if (v_isShared_937_ == 0)
{
v___x_939_ = v___x_936_;
goto v_reusejp_938_;
}
else
{
lean_object* v_reuseFailAlloc_940_; 
v_reuseFailAlloc_940_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_940_, 0, v_val_934_);
v___x_939_ = v_reuseFailAlloc_940_;
goto v_reusejp_938_;
}
v_reusejp_938_:
{
v___y_921_ = v___x_939_;
goto v___jp_920_;
}
}
}
v___jp_920_:
{
lean_object* v___x_922_; lean_object* v___x_923_; lean_object* v___x_924_; lean_object* v___f_925_; lean_object* v___x_926_; lean_object* v___x_927_; lean_object* v___f_928_; lean_object* v___x_929_; 
v___x_922_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___closed__1));
v___x_923_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___closed__2));
v___x_924_ = lean_box(v___x_911_);
lean_inc(v___y_921_);
lean_inc(v___x_917_);
v___f_925_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__1___boxed), 19, 8);
lean_closure_set(v___f_925_, 0, v___f_913_);
lean_closure_set(v___f_925_, 1, v___x_922_);
lean_closure_set(v___f_925_, 2, v___x_923_);
lean_closure_set(v___f_925_, 3, v___x_909_);
lean_closure_set(v___f_925_, 4, v___x_924_);
lean_closure_set(v___f_925_, 5, v___x_917_);
lean_closure_set(v___f_925_, 6, v___y_921_);
lean_closure_set(v___f_925_, 7, v___x_914_);
v___x_926_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___closed__4));
v___x_927_ = lean_box(v___x_911_);
v___f_928_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___lam__2___boxed), 20, 11);
lean_closure_set(v___f_928_, 0, v___x_922_);
lean_closure_set(v___f_928_, 1, v___x_923_);
lean_closure_set(v___f_928_, 2, v___x_909_);
lean_closure_set(v___f_928_, 3, v_s_915_);
lean_closure_set(v___f_928_, 4, v___x_927_);
lean_closure_set(v___f_928_, 5, v___x_917_);
lean_closure_set(v___f_928_, 6, v___x_926_);
lean_closure_set(v___f_928_, 7, v___x_919_);
lean_closure_set(v___f_928_, 8, v___f_925_);
lean_closure_set(v___f_928_, 9, v___y_921_);
lean_closure_set(v___f_928_, 10, v___x_914_);
v___x_929_ = l_Lean_Elab_Tactic_focus___redArg(v___f_928_, v_a_900_, v_a_901_, v_a_902_, v_a_903_, v_a_904_, v_a_905_, v_a_906_, v_a_907_);
return v___x_929_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1___boxed(lean_object* v_x_942_, lean_object* v_a_943_, lean_object* v_a_944_, lean_object* v_a_945_, lean_object* v_a_946_, lean_object* v_a_947_, lean_object* v_a_948_, lean_object* v_a_949_, lean_object* v_a_950_, lean_object* v_a_951_){
_start:
{
lean_object* v_res_952_; 
v_res_952_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpRw______elabRules__Mathlib__Tactic__tacticSimp__rw________1(v_x_942_, v_a_943_, v_a_944_, v_a_945_, v_a_946_, v_a_947_, v_a_948_, v_a_949_, v_a_950_);
lean_dec(v_a_950_);
lean_dec_ref(v_a_949_);
lean_dec(v_a_948_);
lean_dec_ref(v_a_947_);
lean_dec(v_a_946_);
lean_dec_ref(v_a_945_);
lean_dec(v_a_944_);
lean_dec_ref(v_a_943_);
return v_res_952_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_SimpRw(uint8_t builtin) {
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
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_SimpRw(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Tactic_tacticSimp__rw______ = _init_lp_mathlib_Mathlib_Tactic_tacticSimp__rw______();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_tacticSimp__rw______);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_SimpRw(uint8_t builtin) {
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
res = runtime_initialize_mathlib_Mathlib_Tactic_SimpRw(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_SimpRw(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_SimpRw(builtin);
}
#ifdef __cplusplus
}
#endif
