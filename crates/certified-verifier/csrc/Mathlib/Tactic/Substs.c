// Lean compiler output
// Module: Mathlib.Tactic.Substs
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
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Lean_MessageLog_add(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_FileMap_toPosition(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasTag(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getTailPos_x3f(lean_object*, uint8_t);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getPos_x3f(lean_object*, uint8_t);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
uint8_t l_Lean_instBEqMessageSeverity_beq(uint8_t, uint8_t);
extern lean_object* l_Lean_warningAsError;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasSyntheticSorry(lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_MessageData_nil;
lean_object* l_Lean_Meta_Tactic_TryThis_addSuggestion(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_evalTactic(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
lean_object* l_Lean_Elab_Tactic_withMainContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Substs_substs___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Substs_substs___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Substs_substs___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Substs_substs___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Substs_substs___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Substs_substs___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Substs_substs___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Substs"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Substs_substs___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Substs_substs___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Substs_substs___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "substs"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Substs_substs___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Substs_substs___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Substs_substs___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Substs_substs___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Substs_substs___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Substs_substs___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Substs_substs___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Substs_substs___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Substs_substs___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Substs_substs___closed__2_value),LEAN_SCALAR_PTR_LITERAL(18, 189, 240, 248, 60, 134, 88, 155)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Substs_substs___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Substs_substs___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Substs_substs___closed__3_value),LEAN_SCALAR_PTR_LITERAL(216, 249, 246, 177, 255, 3, 116, 61)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Substs_substs___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Substs_substs___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Substs_substs___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Substs_substs___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Substs_substs___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Substs_substs___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Substs_substs___closed__5_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Substs_substs___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Substs_substs___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Substs_substs___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Substs_substs___closed__3_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Substs_substs___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Substs_substs___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Substs_substs___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "many"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Substs_substs___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Substs_substs___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Substs_substs___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Substs_substs___closed__8_value),LEAN_SCALAR_PTR_LITERAL(41, 35, 40, 86, 189, 97, 244, 31)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Substs_substs___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Substs_substs___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Substs_substs___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "colGt"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Substs_substs___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Substs_substs___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Substs_substs___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Substs_substs___closed__10_value),LEAN_SCALAR_PTR_LITERAL(185, 236, 32, 153, 169, 213, 53, 244)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Substs_substs___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Substs_substs___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Substs_substs___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Substs_substs___closed__11_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Substs_substs___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Substs_substs___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Substs_substs___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Substs_substs___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Substs_substs___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Substs_substs___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Substs_substs___closed__13_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Substs_substs___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Substs_substs___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Substs_substs___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Substs_substs___closed__14_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Substs_substs___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Substs_substs___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Substs_substs___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Substs_substs___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Substs_substs___closed__12_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Substs_substs___closed__15_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Substs_substs___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Substs_substs___closed__16_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Substs_substs___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Substs_substs___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Substs_substs___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Substs_substs___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Substs_substs___closed__17_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Substs_substs___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Substs_substs___closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Substs_substs___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Substs_substs___closed__18_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Substs_substs___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Substs_substs___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Substs_substs___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Substs_substs___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Substs_substs___closed__16_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Substs_substs___closed__19_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Substs_substs___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Substs_substs___closed__20_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Substs_substs___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Substs_substs___closed__9_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Substs_substs___closed__20_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Substs_substs___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Substs_substs___closed__21_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Substs_substs___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Substs_substs___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Substs_substs___closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Substs_substs___closed__21_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Substs_substs___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Substs_substs___closed__22_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Substs_substs___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Substs_substs___closed__4_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Substs_substs___closed__22_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Substs_substs___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Substs_substs___closed__23_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_Substs_substs = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Substs_substs___closed__23_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__1(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3___redArg___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3___redArg___lam__0___closed__0_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3___redArg___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "unsolvedGoals"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3___redArg___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3___redArg___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3___redArg___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "synthPlaceholder"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3___redArg___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3___redArg___lam__0___closed__2_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3___redArg___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "lean"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3___redArg___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3___redArg___lam__0___closed__3_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3___redArg___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "inductionWithNoAlts"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3___redArg___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3___redArg___lam__0___closed__4_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3___redArg___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "_namedError"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3___redArg___lam__0___closed__5 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3___redArg___lam__0___closed__5_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3___redArg___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3___redArg___lam__0___closed__6 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3___redArg___lam__0___closed__6_value;
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3___redArg___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3_spec__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3_spec__5___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3___redArg(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__2(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__2___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1___lam__0___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "subst"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1___lam__0___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 60, .m_capacity = 60, .m_length = 59, .m_data = "Deprecation warning: `substs` can be replaced with `subst`."};
static const lean_object* lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1___lam__0___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1___lam__0___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1___lam__0___closed__4;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1___lam__0___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1___lam__0___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1___lam__0___closed__5_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1___lam__0___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1___lam__0___closed__6_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1___lam__0___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1___lam__0___closed__7;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1___lam__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1___lam__0___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1___lam__0___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1___lam__0___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1___lam__0___closed__8_value),LEAN_SCALAR_PTR_LITERAL(99, 76, 33, 121, 85, 143, 17, 224)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1___lam__0___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1___lam__0___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1___lam__0___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Try this:"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1___lam__0___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1___lam__0___closed__10_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_54_; lean_object* v___x_55_; lean_object* v___x_56_; 
v___x_54_ = lean_box(0);
v___x_55_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_56_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_56_, 0, v___x_55_);
lean_ctor_set(v___x_56_, 1, v___x_54_);
return v___x_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__0___redArg(){
_start:
{
lean_object* v___x_58_; lean_object* v___x_59_; 
v___x_58_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__0___redArg___closed__0);
v___x_59_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_59_, 0, v___x_58_);
return v___x_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__0___redArg___boxed(lean_object* v___y_60_){
_start:
{
lean_object* v_res_61_; 
v_res_61_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__0___redArg();
return v_res_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__0(lean_object* v_00_u03b1_62_, lean_object* v___y_63_, lean_object* v___y_64_, lean_object* v___y_65_, lean_object* v___y_66_, lean_object* v___y_67_, lean_object* v___y_68_, lean_object* v___y_69_, lean_object* v___y_70_){
_start:
{
lean_object* v___x_72_; 
v___x_72_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__0___redArg();
return v___x_72_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__0___boxed(lean_object* v_00_u03b1_73_, lean_object* v___y_74_, lean_object* v___y_75_, lean_object* v___y_76_, lean_object* v___y_77_, lean_object* v___y_78_, lean_object* v___y_79_, lean_object* v___y_80_, lean_object* v___y_81_, lean_object* v___y_82_){
_start:
{
lean_object* v_res_83_; 
v_res_83_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__0(v_00_u03b1_73_, v___y_74_, v___y_75_, v___y_76_, v___y_77_, v___y_78_, v___y_79_, v___y_80_, v___y_81_);
lean_dec(v___y_81_);
lean_dec_ref(v___y_80_);
lean_dec(v___y_79_);
lean_dec_ref(v___y_78_);
lean_dec(v___y_77_);
lean_dec_ref(v___y_76_);
lean_dec(v___y_75_);
lean_dec_ref(v___y_74_);
return v_res_83_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__1(size_t v_sz_84_, size_t v_i_85_, lean_object* v_bs_86_){
_start:
{
uint8_t v___x_87_; 
v___x_87_ = lean_usize_dec_lt(v_i_85_, v_sz_84_);
if (v___x_87_ == 0)
{
return v_bs_86_;
}
else
{
lean_object* v_v_88_; lean_object* v___x_89_; lean_object* v_bs_x27_90_; size_t v___x_91_; size_t v___x_92_; lean_object* v___x_93_; 
v_v_88_ = lean_array_uget(v_bs_86_, v_i_85_);
v___x_89_ = lean_unsigned_to_nat(0u);
v_bs_x27_90_ = lean_array_uset(v_bs_86_, v_i_85_, v___x_89_);
v___x_91_ = ((size_t)1ULL);
v___x_92_ = lean_usize_add(v_i_85_, v___x_91_);
v___x_93_ = lean_array_uset(v_bs_x27_90_, v_i_85_, v_v_88_);
v_i_85_ = v___x_92_;
v_bs_86_ = v___x_93_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__1___boxed(lean_object* v_sz_95_, lean_object* v_i_96_, lean_object* v_bs_97_){
_start:
{
size_t v_sz_boxed_98_; size_t v_i_boxed_99_; lean_object* v_res_100_; 
v_sz_boxed_98_ = lean_unbox_usize(v_sz_95_);
lean_dec(v_sz_95_);
v_i_boxed_99_ = lean_unbox_usize(v_i_96_);
lean_dec(v_i_96_);
v_res_100_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__1(v_sz_boxed_98_, v_i_boxed_99_, v_bs_97_);
return v_res_100_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3_spec__4(lean_object* v_msgData_101_, lean_object* v___y_102_, lean_object* v___y_103_, lean_object* v___y_104_, lean_object* v___y_105_){
_start:
{
lean_object* v___x_107_; lean_object* v_env_108_; lean_object* v___x_109_; lean_object* v_mctx_110_; lean_object* v_lctx_111_; lean_object* v_options_112_; lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; 
v___x_107_ = lean_st_ref_get(v___y_105_);
v_env_108_ = lean_ctor_get(v___x_107_, 0);
lean_inc_ref(v_env_108_);
lean_dec(v___x_107_);
v___x_109_ = lean_st_ref_get(v___y_103_);
v_mctx_110_ = lean_ctor_get(v___x_109_, 0);
lean_inc_ref(v_mctx_110_);
lean_dec(v___x_109_);
v_lctx_111_ = lean_ctor_get(v___y_102_, 2);
v_options_112_ = lean_ctor_get(v___y_104_, 2);
lean_inc_ref(v_options_112_);
lean_inc_ref(v_lctx_111_);
v___x_113_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_113_, 0, v_env_108_);
lean_ctor_set(v___x_113_, 1, v_mctx_110_);
lean_ctor_set(v___x_113_, 2, v_lctx_111_);
lean_ctor_set(v___x_113_, 3, v_options_112_);
v___x_114_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_114_, 0, v___x_113_);
lean_ctor_set(v___x_114_, 1, v_msgData_101_);
v___x_115_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_115_, 0, v___x_114_);
return v___x_115_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3_spec__4___boxed(lean_object* v_msgData_116_, lean_object* v___y_117_, lean_object* v___y_118_, lean_object* v___y_119_, lean_object* v___y_120_, lean_object* v___y_121_){
_start:
{
lean_object* v_res_122_; 
v_res_122_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3_spec__4(v_msgData_116_, v___y_117_, v___y_118_, v___y_119_, v___y_120_);
lean_dec(v___y_120_);
lean_dec_ref(v___y_119_);
lean_dec(v___y_118_);
lean_dec_ref(v___y_117_);
return v_res_122_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3___redArg___lam__0(uint8_t v___y_130_, uint8_t v_suppressElabErrors_131_, lean_object* v_x_132_){
_start:
{
if (lean_obj_tag(v_x_132_) == 1)
{
lean_object* v_pre_133_; 
v_pre_133_ = lean_ctor_get(v_x_132_, 0);
switch(lean_obj_tag(v_pre_133_))
{
case 1:
{
lean_object* v_pre_134_; 
v_pre_134_ = lean_ctor_get(v_pre_133_, 0);
switch(lean_obj_tag(v_pre_134_))
{
case 0:
{
lean_object* v_str_135_; lean_object* v_str_136_; lean_object* v___x_137_; uint8_t v___x_138_; 
v_str_135_ = lean_ctor_get(v_x_132_, 1);
v_str_136_ = lean_ctor_get(v_pre_133_, 1);
v___x_137_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3___redArg___lam__0___closed__0));
v___x_138_ = lean_string_dec_eq(v_str_136_, v___x_137_);
if (v___x_138_ == 0)
{
lean_object* v___x_139_; uint8_t v___x_140_; 
v___x_139_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Substs_substs___closed__1));
v___x_140_ = lean_string_dec_eq(v_str_136_, v___x_139_);
if (v___x_140_ == 0)
{
return v___y_130_;
}
else
{
lean_object* v___x_141_; uint8_t v___x_142_; 
v___x_141_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3___redArg___lam__0___closed__1));
v___x_142_ = lean_string_dec_eq(v_str_135_, v___x_141_);
if (v___x_142_ == 0)
{
return v___y_130_;
}
else
{
return v_suppressElabErrors_131_;
}
}
}
else
{
lean_object* v___x_143_; uint8_t v___x_144_; 
v___x_143_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3___redArg___lam__0___closed__2));
v___x_144_ = lean_string_dec_eq(v_str_135_, v___x_143_);
if (v___x_144_ == 0)
{
return v___y_130_;
}
else
{
return v_suppressElabErrors_131_;
}
}
}
case 1:
{
lean_object* v_pre_145_; 
v_pre_145_ = lean_ctor_get(v_pre_134_, 0);
if (lean_obj_tag(v_pre_145_) == 0)
{
lean_object* v_str_146_; lean_object* v_str_147_; lean_object* v_str_148_; lean_object* v___x_149_; uint8_t v___x_150_; 
v_str_146_ = lean_ctor_get(v_x_132_, 1);
v_str_147_ = lean_ctor_get(v_pre_133_, 1);
v_str_148_ = lean_ctor_get(v_pre_134_, 1);
v___x_149_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3___redArg___lam__0___closed__3));
v___x_150_ = lean_string_dec_eq(v_str_148_, v___x_149_);
if (v___x_150_ == 0)
{
return v___y_130_;
}
else
{
lean_object* v___x_151_; uint8_t v___x_152_; 
v___x_151_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3___redArg___lam__0___closed__4));
v___x_152_ = lean_string_dec_eq(v_str_147_, v___x_151_);
if (v___x_152_ == 0)
{
return v___y_130_;
}
else
{
lean_object* v___x_153_; uint8_t v___x_154_; 
v___x_153_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3___redArg___lam__0___closed__5));
v___x_154_ = lean_string_dec_eq(v_str_146_, v___x_153_);
if (v___x_154_ == 0)
{
return v___y_130_;
}
else
{
return v_suppressElabErrors_131_;
}
}
}
}
else
{
return v___y_130_;
}
}
default: 
{
return v___y_130_;
}
}
}
case 0:
{
lean_object* v_str_155_; lean_object* v___x_156_; uint8_t v___x_157_; 
v_str_155_ = lean_ctor_get(v_x_132_, 1);
v___x_156_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3___redArg___lam__0___closed__6));
v___x_157_ = lean_string_dec_eq(v_str_155_, v___x_156_);
if (v___x_157_ == 0)
{
return v___y_130_;
}
else
{
return v_suppressElabErrors_131_;
}
}
default: 
{
return v___y_130_;
}
}
}
else
{
return v___y_130_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3___redArg___lam__0___boxed(lean_object* v___y_158_, lean_object* v_suppressElabErrors_159_, lean_object* v_x_160_){
_start:
{
uint8_t v___y_7877__boxed_161_; uint8_t v_suppressElabErrors_boxed_162_; uint8_t v_res_163_; lean_object* v_r_164_; 
v___y_7877__boxed_161_ = lean_unbox(v___y_158_);
v_suppressElabErrors_boxed_162_ = lean_unbox(v_suppressElabErrors_159_);
v_res_163_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3___redArg___lam__0(v___y_7877__boxed_161_, v_suppressElabErrors_boxed_162_, v_x_160_);
lean_dec(v_x_160_);
v_r_164_ = lean_box(v_res_163_);
return v_r_164_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3_spec__5(lean_object* v_opts_165_, lean_object* v_opt_166_){
_start:
{
lean_object* v_name_167_; lean_object* v_defValue_168_; lean_object* v_map_169_; lean_object* v___x_170_; 
v_name_167_ = lean_ctor_get(v_opt_166_, 0);
v_defValue_168_ = lean_ctor_get(v_opt_166_, 1);
v_map_169_ = lean_ctor_get(v_opts_165_, 0);
v___x_170_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_169_, v_name_167_);
if (lean_obj_tag(v___x_170_) == 0)
{
uint8_t v___x_171_; 
v___x_171_ = lean_unbox(v_defValue_168_);
return v___x_171_;
}
else
{
lean_object* v_val_172_; 
v_val_172_ = lean_ctor_get(v___x_170_, 0);
lean_inc(v_val_172_);
lean_dec_ref_known(v___x_170_, 1);
if (lean_obj_tag(v_val_172_) == 1)
{
uint8_t v_v_173_; 
v_v_173_ = lean_ctor_get_uint8(v_val_172_, 0);
lean_dec_ref_known(v_val_172_, 0);
return v_v_173_;
}
else
{
uint8_t v___x_174_; 
lean_dec(v_val_172_);
v___x_174_ = lean_unbox(v_defValue_168_);
return v___x_174_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3_spec__5___boxed(lean_object* v_opts_175_, lean_object* v_opt_176_){
_start:
{
uint8_t v_res_177_; lean_object* v_r_178_; 
v_res_177_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3_spec__5(v_opts_175_, v_opt_176_);
lean_dec_ref(v_opt_176_);
lean_dec_ref(v_opts_175_);
v_r_178_ = lean_box(v_res_177_);
return v_r_178_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3___redArg(lean_object* v_ref_180_, lean_object* v_msgData_181_, uint8_t v_severity_182_, uint8_t v_isSilent_183_, lean_object* v___y_184_, lean_object* v___y_185_, lean_object* v___y_186_, lean_object* v___y_187_){
_start:
{
lean_object* v___y_190_; lean_object* v___y_191_; uint8_t v___y_192_; lean_object* v___y_193_; uint8_t v___y_194_; lean_object* v___y_195_; lean_object* v___y_196_; lean_object* v___y_197_; lean_object* v___y_198_; lean_object* v___y_226_; uint8_t v___y_227_; lean_object* v___y_228_; lean_object* v___y_229_; uint8_t v___y_230_; uint8_t v___y_231_; lean_object* v___y_232_; lean_object* v___y_233_; lean_object* v___y_251_; uint8_t v___y_252_; lean_object* v___y_253_; lean_object* v___y_254_; uint8_t v___y_255_; uint8_t v___y_256_; lean_object* v___y_257_; lean_object* v___y_258_; lean_object* v___y_262_; uint8_t v___y_263_; lean_object* v___y_264_; lean_object* v___y_265_; lean_object* v___y_266_; uint8_t v___y_267_; uint8_t v___y_268_; uint8_t v___x_273_; uint8_t v___y_275_; lean_object* v___y_276_; lean_object* v___y_277_; lean_object* v___y_278_; lean_object* v___y_279_; uint8_t v___y_280_; uint8_t v___y_281_; uint8_t v___y_283_; uint8_t v___x_298_; 
v___x_273_ = 2;
v___x_298_ = l_Lean_instBEqMessageSeverity_beq(v_severity_182_, v___x_273_);
if (v___x_298_ == 0)
{
v___y_283_ = v___x_298_;
goto v___jp_282_;
}
else
{
uint8_t v___x_299_; 
lean_inc_ref(v_msgData_181_);
v___x_299_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_181_);
v___y_283_ = v___x_299_;
goto v___jp_282_;
}
v___jp_189_:
{
lean_object* v___x_199_; lean_object* v_currNamespace_200_; lean_object* v_openDecls_201_; lean_object* v_env_202_; lean_object* v_nextMacroScope_203_; lean_object* v_ngen_204_; lean_object* v_auxDeclNGen_205_; lean_object* v_traceState_206_; lean_object* v_cache_207_; lean_object* v_messages_208_; lean_object* v_infoState_209_; lean_object* v_snapshotTasks_210_; lean_object* v___x_212_; uint8_t v_isShared_213_; uint8_t v_isSharedCheck_224_; 
v___x_199_ = lean_st_ref_take(v___y_198_);
v_currNamespace_200_ = lean_ctor_get(v___y_197_, 6);
v_openDecls_201_ = lean_ctor_get(v___y_197_, 7);
v_env_202_ = lean_ctor_get(v___x_199_, 0);
v_nextMacroScope_203_ = lean_ctor_get(v___x_199_, 1);
v_ngen_204_ = lean_ctor_get(v___x_199_, 2);
v_auxDeclNGen_205_ = lean_ctor_get(v___x_199_, 3);
v_traceState_206_ = lean_ctor_get(v___x_199_, 4);
v_cache_207_ = lean_ctor_get(v___x_199_, 5);
v_messages_208_ = lean_ctor_get(v___x_199_, 6);
v_infoState_209_ = lean_ctor_get(v___x_199_, 7);
v_snapshotTasks_210_ = lean_ctor_get(v___x_199_, 8);
v_isSharedCheck_224_ = !lean_is_exclusive(v___x_199_);
if (v_isSharedCheck_224_ == 0)
{
v___x_212_ = v___x_199_;
v_isShared_213_ = v_isSharedCheck_224_;
goto v_resetjp_211_;
}
else
{
lean_inc(v_snapshotTasks_210_);
lean_inc(v_infoState_209_);
lean_inc(v_messages_208_);
lean_inc(v_cache_207_);
lean_inc(v_traceState_206_);
lean_inc(v_auxDeclNGen_205_);
lean_inc(v_ngen_204_);
lean_inc(v_nextMacroScope_203_);
lean_inc(v_env_202_);
lean_dec(v___x_199_);
v___x_212_ = lean_box(0);
v_isShared_213_ = v_isSharedCheck_224_;
goto v_resetjp_211_;
}
v_resetjp_211_:
{
lean_object* v___x_214_; lean_object* v___x_215_; lean_object* v___x_216_; lean_object* v___x_217_; lean_object* v___x_219_; 
lean_inc(v_openDecls_201_);
lean_inc(v_currNamespace_200_);
v___x_214_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_214_, 0, v_currNamespace_200_);
lean_ctor_set(v___x_214_, 1, v_openDecls_201_);
v___x_215_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_215_, 0, v___x_214_);
lean_ctor_set(v___x_215_, 1, v___y_195_);
lean_inc_ref(v___y_196_);
lean_inc_ref(v___y_191_);
v___x_216_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_216_, 0, v___y_191_);
lean_ctor_set(v___x_216_, 1, v___y_190_);
lean_ctor_set(v___x_216_, 2, v___y_193_);
lean_ctor_set(v___x_216_, 3, v___y_196_);
lean_ctor_set(v___x_216_, 4, v___x_215_);
lean_ctor_set_uint8(v___x_216_, sizeof(void*)*5, v___y_194_);
lean_ctor_set_uint8(v___x_216_, sizeof(void*)*5 + 1, v___y_192_);
lean_ctor_set_uint8(v___x_216_, sizeof(void*)*5 + 2, v_isSilent_183_);
v___x_217_ = l_Lean_MessageLog_add(v___x_216_, v_messages_208_);
if (v_isShared_213_ == 0)
{
lean_ctor_set(v___x_212_, 6, v___x_217_);
v___x_219_ = v___x_212_;
goto v_reusejp_218_;
}
else
{
lean_object* v_reuseFailAlloc_223_; 
v_reuseFailAlloc_223_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_223_, 0, v_env_202_);
lean_ctor_set(v_reuseFailAlloc_223_, 1, v_nextMacroScope_203_);
lean_ctor_set(v_reuseFailAlloc_223_, 2, v_ngen_204_);
lean_ctor_set(v_reuseFailAlloc_223_, 3, v_auxDeclNGen_205_);
lean_ctor_set(v_reuseFailAlloc_223_, 4, v_traceState_206_);
lean_ctor_set(v_reuseFailAlloc_223_, 5, v_cache_207_);
lean_ctor_set(v_reuseFailAlloc_223_, 6, v___x_217_);
lean_ctor_set(v_reuseFailAlloc_223_, 7, v_infoState_209_);
lean_ctor_set(v_reuseFailAlloc_223_, 8, v_snapshotTasks_210_);
v___x_219_ = v_reuseFailAlloc_223_;
goto v_reusejp_218_;
}
v_reusejp_218_:
{
lean_object* v___x_220_; lean_object* v___x_221_; lean_object* v___x_222_; 
v___x_220_ = lean_st_ref_set(v___y_198_, v___x_219_);
v___x_221_ = lean_box(0);
v___x_222_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_222_, 0, v___x_221_);
return v___x_222_;
}
}
}
v___jp_225_:
{
lean_object* v___x_234_; lean_object* v___x_235_; lean_object* v_a_236_; lean_object* v___x_238_; uint8_t v_isShared_239_; uint8_t v_isSharedCheck_249_; 
v___x_234_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_181_);
v___x_235_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3_spec__4(v___x_234_, v___y_184_, v___y_185_, v___y_186_, v___y_187_);
v_a_236_ = lean_ctor_get(v___x_235_, 0);
v_isSharedCheck_249_ = !lean_is_exclusive(v___x_235_);
if (v_isSharedCheck_249_ == 0)
{
v___x_238_ = v___x_235_;
v_isShared_239_ = v_isSharedCheck_249_;
goto v_resetjp_237_;
}
else
{
lean_inc(v_a_236_);
lean_dec(v___x_235_);
v___x_238_ = lean_box(0);
v_isShared_239_ = v_isSharedCheck_249_;
goto v_resetjp_237_;
}
v_resetjp_237_:
{
lean_object* v___x_240_; lean_object* v___x_241_; lean_object* v___x_242_; lean_object* v___x_243_; 
lean_inc_ref_n(v___y_229_, 2);
v___x_240_ = l_Lean_FileMap_toPosition(v___y_229_, v___y_232_);
lean_dec(v___y_232_);
v___x_241_ = l_Lean_FileMap_toPosition(v___y_229_, v___y_233_);
lean_dec(v___y_233_);
v___x_242_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_242_, 0, v___x_241_);
v___x_243_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3___redArg___closed__0));
if (v___y_227_ == 0)
{
lean_del_object(v___x_238_);
lean_dec_ref(v___y_226_);
v___y_190_ = v___x_240_;
v___y_191_ = v___y_228_;
v___y_192_ = v___y_230_;
v___y_193_ = v___x_242_;
v___y_194_ = v___y_231_;
v___y_195_ = v_a_236_;
v___y_196_ = v___x_243_;
v___y_197_ = v___y_186_;
v___y_198_ = v___y_187_;
goto v___jp_189_;
}
else
{
uint8_t v___x_244_; 
lean_inc(v_a_236_);
v___x_244_ = l_Lean_MessageData_hasTag(v___y_226_, v_a_236_);
if (v___x_244_ == 0)
{
lean_object* v___x_245_; lean_object* v___x_247_; 
lean_dec_ref_known(v___x_242_, 1);
lean_dec_ref(v___x_240_);
lean_dec(v_a_236_);
v___x_245_ = lean_box(0);
if (v_isShared_239_ == 0)
{
lean_ctor_set(v___x_238_, 0, v___x_245_);
v___x_247_ = v___x_238_;
goto v_reusejp_246_;
}
else
{
lean_object* v_reuseFailAlloc_248_; 
v_reuseFailAlloc_248_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_248_, 0, v___x_245_);
v___x_247_ = v_reuseFailAlloc_248_;
goto v_reusejp_246_;
}
v_reusejp_246_:
{
return v___x_247_;
}
}
else
{
lean_del_object(v___x_238_);
v___y_190_ = v___x_240_;
v___y_191_ = v___y_228_;
v___y_192_ = v___y_230_;
v___y_193_ = v___x_242_;
v___y_194_ = v___y_231_;
v___y_195_ = v_a_236_;
v___y_196_ = v___x_243_;
v___y_197_ = v___y_186_;
v___y_198_ = v___y_187_;
goto v___jp_189_;
}
}
}
}
v___jp_250_:
{
lean_object* v___x_259_; 
v___x_259_ = l_Lean_Syntax_getTailPos_x3f(v___y_257_, v___y_256_);
lean_dec(v___y_257_);
if (lean_obj_tag(v___x_259_) == 0)
{
lean_inc(v___y_258_);
v___y_226_ = v___y_251_;
v___y_227_ = v___y_252_;
v___y_228_ = v___y_253_;
v___y_229_ = v___y_254_;
v___y_230_ = v___y_255_;
v___y_231_ = v___y_256_;
v___y_232_ = v___y_258_;
v___y_233_ = v___y_258_;
goto v___jp_225_;
}
else
{
lean_object* v_val_260_; 
v_val_260_ = lean_ctor_get(v___x_259_, 0);
lean_inc(v_val_260_);
lean_dec_ref_known(v___x_259_, 1);
v___y_226_ = v___y_251_;
v___y_227_ = v___y_252_;
v___y_228_ = v___y_253_;
v___y_229_ = v___y_254_;
v___y_230_ = v___y_255_;
v___y_231_ = v___y_256_;
v___y_232_ = v___y_258_;
v___y_233_ = v_val_260_;
goto v___jp_225_;
}
}
v___jp_261_:
{
lean_object* v_ref_269_; lean_object* v___x_270_; 
v_ref_269_ = l_Lean_replaceRef(v_ref_180_, v___y_265_);
v___x_270_ = l_Lean_Syntax_getPos_x3f(v_ref_269_, v___y_267_);
if (lean_obj_tag(v___x_270_) == 0)
{
lean_object* v___x_271_; 
v___x_271_ = lean_unsigned_to_nat(0u);
v___y_251_ = v___y_262_;
v___y_252_ = v___y_263_;
v___y_253_ = v___y_264_;
v___y_254_ = v___y_266_;
v___y_255_ = v___y_268_;
v___y_256_ = v___y_267_;
v___y_257_ = v_ref_269_;
v___y_258_ = v___x_271_;
goto v___jp_250_;
}
else
{
lean_object* v_val_272_; 
v_val_272_ = lean_ctor_get(v___x_270_, 0);
lean_inc(v_val_272_);
lean_dec_ref_known(v___x_270_, 1);
v___y_251_ = v___y_262_;
v___y_252_ = v___y_263_;
v___y_253_ = v___y_264_;
v___y_254_ = v___y_266_;
v___y_255_ = v___y_268_;
v___y_256_ = v___y_267_;
v___y_257_ = v_ref_269_;
v___y_258_ = v_val_272_;
goto v___jp_250_;
}
}
v___jp_274_:
{
if (v___y_281_ == 0)
{
v___y_262_ = v___y_278_;
v___y_263_ = v___y_275_;
v___y_264_ = v___y_276_;
v___y_265_ = v___y_277_;
v___y_266_ = v___y_279_;
v___y_267_ = v___y_280_;
v___y_268_ = v_severity_182_;
goto v___jp_261_;
}
else
{
v___y_262_ = v___y_278_;
v___y_263_ = v___y_275_;
v___y_264_ = v___y_276_;
v___y_265_ = v___y_277_;
v___y_266_ = v___y_279_;
v___y_267_ = v___y_280_;
v___y_268_ = v___x_273_;
goto v___jp_261_;
}
}
v___jp_282_:
{
if (v___y_283_ == 0)
{
lean_object* v_fileName_284_; lean_object* v_fileMap_285_; lean_object* v_options_286_; lean_object* v_ref_287_; uint8_t v_suppressElabErrors_288_; lean_object* v___x_289_; lean_object* v___x_290_; lean_object* v___f_291_; uint8_t v___x_292_; uint8_t v___x_293_; 
v_fileName_284_ = lean_ctor_get(v___y_186_, 0);
v_fileMap_285_ = lean_ctor_get(v___y_186_, 1);
v_options_286_ = lean_ctor_get(v___y_186_, 2);
v_ref_287_ = lean_ctor_get(v___y_186_, 5);
v_suppressElabErrors_288_ = lean_ctor_get_uint8(v___y_186_, sizeof(void*)*14 + 1);
v___x_289_ = lean_box(v___y_283_);
v___x_290_ = lean_box(v_suppressElabErrors_288_);
v___f_291_ = lean_alloc_closure((void*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_291_, 0, v___x_289_);
lean_closure_set(v___f_291_, 1, v___x_290_);
v___x_292_ = 1;
v___x_293_ = l_Lean_instBEqMessageSeverity_beq(v_severity_182_, v___x_292_);
if (v___x_293_ == 0)
{
v___y_275_ = v_suppressElabErrors_288_;
v___y_276_ = v_fileName_284_;
v___y_277_ = v_ref_287_;
v___y_278_ = v___f_291_;
v___y_279_ = v_fileMap_285_;
v___y_280_ = v___y_283_;
v___y_281_ = v___x_293_;
goto v___jp_274_;
}
else
{
lean_object* v___x_294_; uint8_t v___x_295_; 
v___x_294_ = l_Lean_warningAsError;
v___x_295_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3_spec__5(v_options_286_, v___x_294_);
v___y_275_ = v_suppressElabErrors_288_;
v___y_276_ = v_fileName_284_;
v___y_277_ = v_ref_287_;
v___y_278_ = v___f_291_;
v___y_279_ = v_fileMap_285_;
v___y_280_ = v___y_283_;
v___y_281_ = v___x_295_;
goto v___jp_274_;
}
}
else
{
lean_object* v___x_296_; lean_object* v___x_297_; 
lean_dec_ref(v_msgData_181_);
v___x_296_ = lean_box(0);
v___x_297_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_297_, 0, v___x_296_);
return v___x_297_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3___redArg___boxed(lean_object* v_ref_300_, lean_object* v_msgData_301_, lean_object* v_severity_302_, lean_object* v_isSilent_303_, lean_object* v___y_304_, lean_object* v___y_305_, lean_object* v___y_306_, lean_object* v___y_307_, lean_object* v___y_308_){
_start:
{
uint8_t v_severity_boxed_309_; uint8_t v_isSilent_boxed_310_; lean_object* v_res_311_; 
v_severity_boxed_309_ = lean_unbox(v_severity_302_);
v_isSilent_boxed_310_ = lean_unbox(v_isSilent_303_);
v_res_311_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3___redArg(v_ref_300_, v_msgData_301_, v_severity_boxed_309_, v_isSilent_boxed_310_, v___y_304_, v___y_305_, v___y_306_, v___y_307_);
lean_dec(v___y_307_);
lean_dec_ref(v___y_306_);
lean_dec(v___y_305_);
lean_dec_ref(v___y_304_);
lean_dec(v_ref_300_);
return v_res_311_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3(lean_object* v_ref_312_, lean_object* v_msgData_313_, lean_object* v___y_314_, lean_object* v___y_315_, lean_object* v___y_316_, lean_object* v___y_317_, lean_object* v___y_318_, lean_object* v___y_319_, lean_object* v___y_320_, lean_object* v___y_321_){
_start:
{
uint8_t v___x_323_; uint8_t v___x_324_; lean_object* v___x_325_; 
v___x_323_ = 1;
v___x_324_ = 0;
v___x_325_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3___redArg(v_ref_312_, v_msgData_313_, v___x_323_, v___x_324_, v___y_318_, v___y_319_, v___y_320_, v___y_321_);
return v___x_325_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3___boxed(lean_object* v_ref_326_, lean_object* v_msgData_327_, lean_object* v___y_328_, lean_object* v___y_329_, lean_object* v___y_330_, lean_object* v___y_331_, lean_object* v___y_332_, lean_object* v___y_333_, lean_object* v___y_334_, lean_object* v___y_335_, lean_object* v___y_336_){
_start:
{
lean_object* v_res_337_; 
v_res_337_ = lp_mathlib_Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3(v_ref_326_, v_msgData_327_, v___y_328_, v___y_329_, v___y_330_, v___y_331_, v___y_332_, v___y_333_, v___y_334_, v___y_335_);
lean_dec(v___y_335_);
lean_dec_ref(v___y_334_);
lean_dec(v___y_333_);
lean_dec_ref(v___y_332_);
lean_dec(v___y_331_);
lean_dec_ref(v___y_330_);
lean_dec(v___y_329_);
lean_dec_ref(v___y_328_);
lean_dec(v_ref_326_);
return v_res_337_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__2(size_t v_sz_338_, size_t v_i_339_, lean_object* v_bs_340_){
_start:
{
uint8_t v___x_341_; 
v___x_341_ = lean_usize_dec_lt(v_i_339_, v_sz_338_);
if (v___x_341_ == 0)
{
return v_bs_340_;
}
else
{
lean_object* v_v_342_; lean_object* v___x_343_; lean_object* v_bs_x27_344_; size_t v___x_345_; size_t v___x_346_; lean_object* v___x_347_; 
v_v_342_ = lean_array_uget(v_bs_340_, v_i_339_);
v___x_343_ = lean_unsigned_to_nat(0u);
v_bs_x27_344_ = lean_array_uset(v_bs_340_, v_i_339_, v___x_343_);
v___x_345_ = ((size_t)1ULL);
v___x_346_ = lean_usize_add(v_i_339_, v___x_345_);
v___x_347_ = lean_array_uset(v_bs_x27_344_, v_i_339_, v_v_342_);
v_i_339_ = v___x_346_;
v_bs_340_ = v___x_347_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__2___boxed(lean_object* v_sz_349_, lean_object* v_i_350_, lean_object* v_bs_351_){
_start:
{
size_t v_sz_boxed_352_; size_t v_i_boxed_353_; lean_object* v_res_354_; 
v_sz_boxed_352_ = lean_unbox_usize(v_sz_349_);
lean_dec(v_sz_349_);
v_i_boxed_353_ = lean_unbox_usize(v_i_350_);
lean_dec(v_i_350_);
v_res_354_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__2(v_sz_boxed_352_, v_i_boxed_353_, v_bs_351_);
return v_res_354_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1___lam__0___closed__4(void){
_start:
{
lean_object* v___x_359_; lean_object* v___x_360_; 
v___x_359_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1___lam__0___closed__3));
v___x_360_ = l_Lean_stringToMessageData(v___x_359_);
return v___x_360_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1___lam__0___closed__7(void){
_start:
{
lean_object* v___x_364_; 
v___x_364_ = l_Array_mkArray0(lean_box(0));
return v___x_364_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1___lam__0(lean_object* v___x_369_, lean_object* v_xs_370_, lean_object* v_tk_371_, lean_object* v___y_372_, lean_object* v___y_373_, lean_object* v___y_374_, lean_object* v___y_375_, lean_object* v___y_376_, lean_object* v___y_377_, lean_object* v___y_378_, lean_object* v___y_379_){
_start:
{
lean_object* v_ref_381_; lean_object* v___x_382_; lean_object* v___x_383_; lean_object* v___x_384_; lean_object* v___x_385_; size_t v_sz_386_; lean_object* v___x_387_; lean_object* v___x_388_; 
v_ref_381_ = lean_ctor_get(v___y_378_, 5);
v___x_382_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1___lam__0___closed__0));
v___x_383_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1___lam__0___closed__1));
v___x_384_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1___lam__0___closed__2));
v___x_385_ = l_Lean_Name_mkStr4(v___x_382_, v___x_383_, v___x_369_, v___x_384_);
v_sz_386_ = lean_array_size(v_xs_370_);
v___x_387_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1___lam__0___closed__4, &lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1___lam__0___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1___lam__0___closed__4);
v___x_388_ = lp_mathlib_Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3(v_tk_371_, v___x_387_, v___y_372_, v___y_373_, v___y_374_, v___y_375_, v___y_376_, v___y_377_, v___y_378_, v___y_379_);
if (lean_obj_tag(v___x_388_) == 0)
{
lean_object* v___x_390_; uint8_t v_isShared_391_; uint8_t v_isSharedCheck_416_; 
v_isSharedCheck_416_ = !lean_is_exclusive(v___x_388_);
if (v_isSharedCheck_416_ == 0)
{
lean_object* v_unused_417_; 
v_unused_417_ = lean_ctor_get(v___x_388_, 0);
lean_dec(v_unused_417_);
v___x_390_ = v___x_388_;
v_isShared_391_ = v_isSharedCheck_416_;
goto v_resetjp_389_;
}
else
{
lean_dec(v___x_388_);
v___x_390_ = lean_box(0);
v_isShared_391_ = v_isSharedCheck_416_;
goto v_resetjp_389_;
}
v_resetjp_389_:
{
uint8_t v___x_392_; lean_object* v___x_393_; lean_object* v___x_394_; size_t v___x_395_; lean_object* v___x_396_; size_t v_sz_397_; lean_object* v___x_398_; lean_object* v___x_399_; lean_object* v___x_400_; lean_object* v___x_401_; lean_object* v___x_402_; lean_object* v___x_403_; lean_object* v___x_404_; lean_object* v___x_405_; lean_object* v___x_406_; lean_object* v___x_407_; lean_object* v___x_409_; 
v___x_392_ = 0;
v___x_393_ = l_Lean_SourceInfo_fromRef(v_ref_381_, v___x_392_);
lean_inc_n(v___x_393_, 2);
v___x_394_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_394_, 0, v___x_393_);
lean_ctor_set(v___x_394_, 1, v___x_384_);
v___x_395_ = ((size_t)0ULL);
v___x_396_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__1(v_sz_386_, v___x_395_, v_xs_370_);
v_sz_397_ = lean_array_size(v___x_396_);
v___x_398_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1___lam__0___closed__6));
v___x_399_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1___lam__0___closed__7, &lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1___lam__0___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1___lam__0___closed__7);
v___x_400_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__2(v_sz_397_, v___x_395_, v___x_396_);
v___x_401_ = l_Array_append___redArg(v___x_399_, v___x_400_);
lean_dec_ref(v___x_400_);
v___x_402_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_402_, 0, v___x_393_);
lean_ctor_set(v___x_402_, 1, v___x_398_);
lean_ctor_set(v___x_402_, 2, v___x_401_);
v___x_403_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1___lam__0___closed__9));
v___x_404_ = l_Lean_Syntax_node2(v___x_393_, v___x_385_, v___x_394_, v___x_402_);
lean_inc(v___x_404_);
v___x_405_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_405_, 0, v___x_403_);
lean_ctor_set(v___x_405_, 1, v___x_404_);
v___x_406_ = lean_box(0);
v___x_407_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_407_, 0, v___x_405_);
lean_ctor_set(v___x_407_, 1, v___x_406_);
lean_ctor_set(v___x_407_, 2, v___x_406_);
lean_ctor_set(v___x_407_, 3, v___x_406_);
lean_ctor_set(v___x_407_, 4, v___x_406_);
lean_ctor_set(v___x_407_, 5, v___x_406_);
lean_inc(v_ref_381_);
if (v_isShared_391_ == 0)
{
lean_ctor_set_tag(v___x_390_, 1);
lean_ctor_set(v___x_390_, 0, v_ref_381_);
v___x_409_ = v___x_390_;
goto v_reusejp_408_;
}
else
{
lean_object* v_reuseFailAlloc_415_; 
v_reuseFailAlloc_415_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_415_, 0, v_ref_381_);
v___x_409_ = v_reuseFailAlloc_415_;
goto v_reusejp_408_;
}
v_reusejp_408_:
{
lean_object* v___x_410_; uint8_t v___x_411_; lean_object* v___x_412_; lean_object* v___x_413_; 
v___x_410_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1___lam__0___closed__10));
v___x_411_ = 4;
v___x_412_ = l_Lean_MessageData_nil;
v___x_413_ = l_Lean_Meta_Tactic_TryThis_addSuggestion(v_tk_371_, v___x_407_, v___x_409_, v___x_410_, v___x_406_, v___x_411_, v___x_412_, v___y_378_, v___y_379_);
if (lean_obj_tag(v___x_413_) == 0)
{
lean_object* v___x_414_; 
lean_dec_ref_known(v___x_413_, 1);
v___x_414_ = l_Lean_Elab_Tactic_evalTactic(v___x_404_, v___y_372_, v___y_373_, v___y_374_, v___y_375_, v___y_376_, v___y_377_, v___y_378_, v___y_379_);
return v___x_414_;
}
else
{
lean_dec(v___x_404_);
return v___x_413_;
}
}
}
}
else
{
lean_dec(v___x_385_);
lean_dec(v_tk_371_);
lean_dec_ref(v_xs_370_);
return v___x_388_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1___lam__0___boxed(lean_object* v___x_418_, lean_object* v_xs_419_, lean_object* v_tk_420_, lean_object* v___y_421_, lean_object* v___y_422_, lean_object* v___y_423_, lean_object* v___y_424_, lean_object* v___y_425_, lean_object* v___y_426_, lean_object* v___y_427_, lean_object* v___y_428_, lean_object* v___y_429_){
_start:
{
lean_object* v_res_430_; 
v_res_430_ = lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1___lam__0(v___x_418_, v_xs_419_, v_tk_420_, v___y_421_, v___y_422_, v___y_423_, v___y_424_, v___y_425_, v___y_426_, v___y_427_, v___y_428_);
lean_dec(v___y_428_);
lean_dec_ref(v___y_427_);
lean_dec(v___y_426_);
lean_dec_ref(v___y_425_);
lean_dec(v___y_424_);
lean_dec_ref(v___y_423_);
lean_dec(v___y_422_);
lean_dec_ref(v___y_421_);
return v_res_430_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1(lean_object* v_x_431_, lean_object* v_a_432_, lean_object* v_a_433_, lean_object* v_a_434_, lean_object* v_a_435_, lean_object* v_a_436_, lean_object* v_a_437_, lean_object* v_a_438_, lean_object* v_a_439_){
_start:
{
lean_object* v___x_441_; lean_object* v___x_442_; uint8_t v___x_443_; 
v___x_441_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Substs_substs___closed__1));
v___x_442_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Substs_substs___closed__4));
lean_inc(v_x_431_);
v___x_443_ = l_Lean_Syntax_isOfKind(v_x_431_, v___x_442_);
if (v___x_443_ == 0)
{
lean_object* v___x_444_; 
lean_dec(v_x_431_);
v___x_444_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__0___redArg();
return v___x_444_;
}
else
{
lean_object* v___x_445_; lean_object* v_tk_446_; lean_object* v___x_447_; lean_object* v___x_448_; lean_object* v_xs_449_; lean_object* v___f_450_; lean_object* v___x_451_; 
v___x_445_ = lean_unsigned_to_nat(0u);
v_tk_446_ = l_Lean_Syntax_getArg(v_x_431_, v___x_445_);
v___x_447_ = lean_unsigned_to_nat(1u);
v___x_448_ = l_Lean_Syntax_getArg(v_x_431_, v___x_447_);
lean_dec(v_x_431_);
v_xs_449_ = l_Lean_Syntax_getArgs(v___x_448_);
lean_dec(v___x_448_);
v___f_450_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1___lam__0___boxed), 12, 3);
lean_closure_set(v___f_450_, 0, v___x_441_);
lean_closure_set(v___f_450_, 1, v_xs_449_);
lean_closure_set(v___f_450_, 2, v_tk_446_);
v___x_451_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_450_, v_a_432_, v_a_433_, v_a_434_, v_a_435_, v_a_436_, v_a_437_, v_a_438_, v_a_439_);
return v___x_451_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1___boxed(lean_object* v_x_452_, lean_object* v_a_453_, lean_object* v_a_454_, lean_object* v_a_455_, lean_object* v_a_456_, lean_object* v_a_457_, lean_object* v_a_458_, lean_object* v_a_459_, lean_object* v_a_460_, lean_object* v_a_461_){
_start:
{
lean_object* v_res_462_; 
v_res_462_ = lp_mathlib_Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1(v_x_452_, v_a_453_, v_a_454_, v_a_455_, v_a_456_, v_a_457_, v_a_458_, v_a_459_, v_a_460_);
lean_dec(v_a_460_);
lean_dec_ref(v_a_459_);
lean_dec(v_a_458_);
lean_dec_ref(v_a_457_);
lean_dec(v_a_456_);
lean_dec_ref(v_a_455_);
lean_dec(v_a_454_);
lean_dec_ref(v_a_453_);
return v_res_462_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3(lean_object* v_ref_463_, lean_object* v_msgData_464_, uint8_t v_severity_465_, uint8_t v_isSilent_466_, lean_object* v___y_467_, lean_object* v___y_468_, lean_object* v___y_469_, lean_object* v___y_470_, lean_object* v___y_471_, lean_object* v___y_472_, lean_object* v___y_473_, lean_object* v___y_474_){
_start:
{
lean_object* v___x_476_; 
v___x_476_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3___redArg(v_ref_463_, v_msgData_464_, v_severity_465_, v_isSilent_466_, v___y_471_, v___y_472_, v___y_473_, v___y_474_);
return v___x_476_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3___boxed(lean_object* v_ref_477_, lean_object* v_msgData_478_, lean_object* v_severity_479_, lean_object* v_isSilent_480_, lean_object* v___y_481_, lean_object* v___y_482_, lean_object* v___y_483_, lean_object* v___y_484_, lean_object* v___y_485_, lean_object* v___y_486_, lean_object* v___y_487_, lean_object* v___y_488_, lean_object* v___y_489_){
_start:
{
uint8_t v_severity_boxed_490_; uint8_t v_isSilent_boxed_491_; lean_object* v_res_492_; 
v_severity_boxed_490_ = lean_unbox(v_severity_479_);
v_isSilent_boxed_491_ = lean_unbox(v_isSilent_480_);
v_res_492_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_Substs___aux__Mathlib__Tactic__Substs______elabRules__Mathlib__Tactic__Substs__substs__1_spec__3_spec__3(v_ref_477_, v_msgData_478_, v_severity_boxed_490_, v_isSilent_boxed_491_, v___y_481_, v___y_482_, v___y_483_, v___y_484_, v___y_485_, v___y_486_, v___y_487_, v___y_488_);
lean_dec(v___y_488_);
lean_dec_ref(v___y_487_);
lean_dec(v___y_486_);
lean_dec_ref(v___y_485_);
lean_dec(v___y_484_);
lean_dec_ref(v___y_483_);
lean_dec(v___y_482_);
lean_dec_ref(v___y_481_);
lean_dec(v_ref_477_);
return v_res_492_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Substs(uint8_t builtin) {
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
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Substs(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Substs(uint8_t builtin) {
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
res = runtime_initialize_mathlib_Mathlib_Tactic_Substs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Substs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Substs(builtin);
}
#ifdef __cplusplus
}
#endif
