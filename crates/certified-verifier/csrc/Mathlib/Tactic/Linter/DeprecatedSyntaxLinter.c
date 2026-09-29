// Lean compiler output
// Module: Mathlib.Tactic.Linter.DeprecatedSyntaxLinter
// Imports: public import Init public meta import Init public meta import Lean.Elab.Command public meta import Mathlib.Tactic.Linter.Header public import Lean.Parser.Command
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
lean_object* l_Lean_Name_mkStr6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t l_Lean_Syntax_matchesIdent(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
lean_object* l_Lean_Syntax_TSepArray_getElems___redArg(lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
extern lean_object* l_Lean_Elab_Command_instInhabitedScope_default;
lean_object* l_List_head_x21___redArg(lean_object*, lean_object*);
extern lean_object* l_Lean_Linter_linterSetsExt;
extern lean_object* l_Lean_Linter_instInhabitedLinterSetsState_default;
lean_object* l_Lean_PersistentEnvExtension_getState___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Linter_getLinterValue(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* l_Lean_MessageData_note(lean_object*);
extern lean_object* l_Lean_Linter_linterMessageTag;
lean_object* l_Lean_Elab_Command_getScope___redArg(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Lean_MessageLog_add(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_FileMap_toPosition(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasTag(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getTailPos_x3f(lean_object*, uint8_t);
lean_object* l_Lean_Elab_Command_getRef___redArg(lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getPos_x3f(lean_object*, uint8_t);
uint8_t l_Lean_instBEqMessageSeverity_beq(uint8_t, uint8_t);
extern lean_object* l_Lean_warningAsError;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasSyntheticSorry(lean_object*);
uint8_t l_Lean_MessageLog_hasErrors(lean_object*);
lean_object* lean_string_utf8_extract(lean_object*, lean_object*, lean_object*);
lean_object* lean_string_utf8_byte_size(lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_string_utf8_next_fast(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint32_t lean_string_utf8_get_fast(lean_object*, lean_object*);
uint8_t lean_uint32_dec_eq(uint32_t, uint32_t);
lean_object* l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(lean_object*, uint8_t);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getAtomVal(lean_object*);
lean_object* l_Lean_TSyntax_getId(lean_object*);
lean_object* l_Lean_Syntax_find_x3f(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getTrailing_x3f(lean_object*);
lean_object* l_Lean_TSyntax_getNat(lean_object*);
lean_object* l_Lean_Name_components(lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofConstName(lean_object*, uint8_t);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* lean_register_option(lean_object*, lean_object*);
lean_object* l_Lean_withSetOptionIn___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_addLinter(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "style"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "refine"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(186, 218, 113, 226, 101, 176, 32, 79)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(105, 62, 218, 153, 100, 142, 29, 251)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(138, 84, 217, 209, 64, 124, 233, 42)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "enable the refine linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Style"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__9_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__9_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__9_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(120, 131, 127, 204, 79, 169, 80, 92)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__9_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__9_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(80, 121, 70, 102, 253, 204, 74, 141)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__9_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__9_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(173, 174, 0, 222, 115, 76, 186, 163)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__9_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value_aux_4 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__9_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(250, 214, 43, 113, 105, 153, 208, 218)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__9_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__9_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value_aux_4),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(197, 129, 224, 127, 3, 30, 216, 7)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__9_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__9_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_Style_linter_style_refine;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_278512193____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "cases"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_278512193____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_278512193____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_278512193____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(186, 218, 113, 226, 101, 176, 32, 79)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_278512193____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_278512193____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(105, 62, 218, 153, 100, 142, 29, 251)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_278512193____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_278512193____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_278512193____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(150, 199, 174, 209, 200, 84, 159, 62)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_278512193____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_278512193____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_278512193____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "enable the cases linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_278512193____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_278512193____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_278512193____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_278512193____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_278512193____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_278512193____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_278512193____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_278512193____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_278512193____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(120, 131, 127, 204, 79, 169, 80, 92)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_278512193____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_278512193____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(80, 121, 70, 102, 253, 204, 74, 141)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_278512193____hygCtx___hyg_4__value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_278512193____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(173, 174, 0, 222, 115, 76, 186, 163)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_278512193____hygCtx___hyg_4__value_aux_4 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_278512193____hygCtx___hyg_4__value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(250, 214, 43, 113, 105, 153, 208, 218)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_278512193____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_278512193____hygCtx___hyg_4__value_aux_4),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_278512193____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(49, 114, 63, 196, 1, 15, 162, 89)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_278512193____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_278512193____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_278512193____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_278512193____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_Style_linter_style_cases;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_3986038175____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "induction"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_3986038175____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_3986038175____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_3986038175____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(186, 218, 113, 226, 101, 176, 32, 79)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_3986038175____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_3986038175____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(105, 62, 218, 153, 100, 142, 29, 251)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_3986038175____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_3986038175____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_3986038175____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(148, 168, 220, 227, 91, 156, 218, 163)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_3986038175____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_3986038175____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_3986038175____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 28, .m_capacity = 28, .m_length = 27, .m_data = "enable the induction linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_3986038175____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_3986038175____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_3986038175____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_3986038175____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_3986038175____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_3986038175____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_3986038175____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_3986038175____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_3986038175____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(120, 131, 127, 204, 79, 169, 80, 92)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_3986038175____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_3986038175____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(80, 121, 70, 102, 253, 204, 74, 141)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_3986038175____hygCtx___hyg_4__value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_3986038175____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(173, 174, 0, 222, 115, 76, 186, 163)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_3986038175____hygCtx___hyg_4__value_aux_4 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_3986038175____hygCtx___hyg_4__value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(250, 214, 43, 113, 105, 153, 208, 218)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_3986038175____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_3986038175____hygCtx___hyg_4__value_aux_4),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_3986038175____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(35, 193, 155, 199, 187, 188, 252, 246)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_3986038175____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_3986038175____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_3986038175____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_3986038175____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_Style_linter_style_induction;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_947237925____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "admit"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_947237925____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_947237925____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_947237925____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(186, 218, 113, 226, 101, 176, 32, 79)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_947237925____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_947237925____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(105, 62, 218, 153, 100, 142, 29, 251)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_947237925____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_947237925____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_947237925____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(96, 181, 209, 75, 172, 135, 136, 58)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_947237925____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_947237925____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_947237925____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "enable the admit linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_947237925____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_947237925____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_947237925____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_947237925____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_947237925____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_947237925____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_947237925____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_947237925____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_947237925____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(120, 131, 127, 204, 79, 169, 80, 92)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_947237925____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_947237925____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(80, 121, 70, 102, 253, 204, 74, 141)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_947237925____hygCtx___hyg_4__value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_947237925____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(173, 174, 0, 222, 115, 76, 186, 163)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_947237925____hygCtx___hyg_4__value_aux_4 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_947237925____hygCtx___hyg_4__value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(250, 214, 43, 113, 105, 153, 208, 218)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_947237925____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_947237925____hygCtx___hyg_4__value_aux_4),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_947237925____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(239, 200, 48, 90, 129, 47, 8, 112)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_947237925____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_947237925____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_947237925____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_947237925____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_Style_linter_style_admit;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2792787194____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "native"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2792787194____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2792787194____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2792787194____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(186, 218, 113, 226, 101, 176, 32, 79)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2792787194____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2792787194____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(105, 62, 218, 153, 100, 142, 29, 251)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2792787194____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2792787194____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2792787194____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(202, 207, 201, 58, 7, 230, 201, 108)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2792787194____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2792787194____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2792787194____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 36, .m_capacity = 36, .m_length = 35, .m_data = "enable the native-evaluation linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2792787194____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2792787194____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2792787194____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2792787194____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2792787194____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2792787194____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2792787194____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2792787194____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2792787194____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(120, 131, 127, 204, 79, 169, 80, 92)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2792787194____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2792787194____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(80, 121, 70, 102, 253, 204, 74, 141)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2792787194____hygCtx___hyg_4__value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2792787194____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(173, 174, 0, 222, 115, 76, 186, 163)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2792787194____hygCtx___hyg_4__value_aux_4 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2792787194____hygCtx___hyg_4__value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(250, 214, 43, 113, 105, 153, 208, 218)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2792787194____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2792787194____hygCtx___hyg_4__value_aux_4),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2792787194____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(5, 18, 208, 72, 45, 182, 229, 146)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2792787194____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2792787194____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2792787194____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2792787194____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_Style_linter_style_native;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2098674011____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "nativeDecide"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2098674011____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2098674011____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2098674011____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(186, 218, 113, 226, 101, 176, 32, 79)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2098674011____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2098674011____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(105, 62, 218, 153, 100, 142, 29, 251)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2098674011____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2098674011____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2098674011____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(69, 228, 244, 229, 71, 77, 228, 32)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2098674011____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2098674011____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2098674011____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 57, .m_capacity = 57, .m_length = 56, .m_data = "deprecated: use the `linter.style.native` option instead"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2098674011____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2098674011____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2098674011____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "2026-08-28"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2098674011____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2098674011____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2098674011____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2098674011____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2098674011____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(120, 131, 127, 204, 79, 169, 80, 92)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2098674011____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2098674011____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(80, 121, 70, 102, 253, 204, 74, 141)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2098674011____hygCtx___hyg_4__value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2098674011____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(173, 174, 0, 222, 115, 76, 186, 163)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2098674011____hygCtx___hyg_4__value_aux_4 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2098674011____hygCtx___hyg_4__value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(250, 214, 43, 113, 105, 153, 208, 218)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2098674011____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2098674011____hygCtx___hyg_4__value_aux_4),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2098674011____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(10, 242, 188, 237, 47, 245, 53, 53)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2098674011____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2098674011____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2098674011____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2098674011____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_Style_linter_style_nativeDecide;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_303104340____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "maxHeartbeats"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_303104340____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_303104340____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_303104340____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(186, 218, 113, 226, 101, 176, 32, 79)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_303104340____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_303104340____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(105, 62, 218, 153, 100, 142, 29, 251)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_303104340____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_303104340____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_303104340____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(105, 91, 59, 62, 139, 161, 68, 208)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_303104340____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_303104340____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_303104340____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 32, .m_capacity = 32, .m_length = 31, .m_data = "enable the maxHeartbeats linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_303104340____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_303104340____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_303104340____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_303104340____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_303104340____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_303104340____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_303104340____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_303104340____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_303104340____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(120, 131, 127, 204, 79, 169, 80, 92)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_303104340____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_303104340____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(80, 121, 70, 102, 253, 204, 74, 141)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_303104340____hygCtx___hyg_4__value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_303104340____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(173, 174, 0, 222, 115, 76, 186, 163)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_303104340____hygCtx___hyg_4__value_aux_4 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_303104340____hygCtx___hyg_4__value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(250, 214, 43, 113, 105, 153, 208, 218)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_303104340____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_303104340____hygCtx___hyg_4__value_aux_4),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_303104340____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(206, 139, 119, 23, 50, 6, 244, 80)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_303104340____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_303104340____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_303104340____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_303104340____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_Style_linter_style_maxHeartbeats;
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_elem___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_elem___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment_spec__0___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Command"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "in"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__4_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__2_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__4_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__3_value),LEAN_SCALAR_PTR_LITERAL(65, 79, 35, 19, 21, 38, 89, 10)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "set_option"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__6_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__6_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__2_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__6_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__5_value),LEAN_SCALAR_PTR_LITERAL(216, 223, 149, 245, 150, 86, 134, 198)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__6_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "num"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__7_value),LEAN_SCALAR_PTR_LITERAL(227, 68, 22, 222, 47, 51, 204, 84)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__8_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___lam__0___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__3_value)} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__9_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__10_value),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__11_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__12_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__12_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__13_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__13_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__14_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_303104340____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(163, 202, 216, 251, 148, 187, 135, 206)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__15_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment(lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__0_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "structInstField"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__2_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__2_value_aux_1),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__2_value_aux_2),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(50, 77, 20, 88, 28, 210, 230, 84)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__2 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__2_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "structInstLVal"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__3 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__4_value_aux_1),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__4_value_aux_2),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__3_value),LEAN_SCALAR_PTR_LITERAL(185, 133, 6, 147, 6, 183, 100, 198)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__4 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2792787194____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(176, 29, 85, 252, 232, 140, 96, 238)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__5 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__5_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "structInstFieldDef"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__6 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__6_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__7_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__7_value_aux_1),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__7_value_aux_2),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__6_value),LEAN_SCALAR_PTR_LITERAL(81, 102, 39, 227, 176, 252, 65, 103)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__7 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__7_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "true"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__8 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__8_value),LEAN_SCALAR_PTR_LITERAL(235, 97, 249, 134, 197, 220, 12, 91)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__9 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__9_value;
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0(lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "configItem"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__1_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__0_value),LEAN_SCALAR_PTR_LITERAL(196, 225, 216, 244, 112, 3, 142, 188)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__3_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__3_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__3_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__0_value),LEAN_SCALAR_PTR_LITERAL(205, 9, 236, 192, 59, 252, 178, 140)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "posConfigItem"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__5_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__5_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__5_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__5_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__4_value),LEAN_SCALAR_PTR_LITERAL(232, 137, 50, 117, 152, 182, 155, 132)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "valConfigItem"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__7_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__7_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__7_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__6_value),LEAN_SCALAR_PTR_LITERAL(135, 67, 19, 169, 17, 95, 109, 188)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__7_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "config"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__8_value),LEAN_SCALAR_PTR_LITERAL(207, 146, 87, 28, 198, 178, 209, 199)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__9_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "structInst"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__11_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__11_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__11_value_aux_1),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__11_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__10_value),LEAN_SCALAR_PTR_LITERAL(50, 43, 73, 62, 118, 124, 31, 28)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__11_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "structInstFields"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__12_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__13_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__13_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__13_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__13_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__13_value_aux_1),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__13_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__12_value),LEAN_SCALAR_PTR_LITERAL(0, 82, 141, 43, 62, 171, 163, 69)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__13_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "optEllipsis"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__14_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__15_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__15_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__15_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__15_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__15_value_aux_1),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__15_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__14_value),LEAN_SCALAR_PTR_LITERAL(13, 1, 242, 203, 207, 188, 181, 160)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__15_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__16_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__16_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__16_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__16_value_aux_1),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__16_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__4_value),LEAN_SCALAR_PTR_LITERAL(9, 149, 144, 155, 11, 94, 233, 86)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__16_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__17_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__17_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__17_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__17_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__17_value_aux_1),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__17_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__6_value),LEAN_SCALAR_PTR_LITERAL(190, 235, 186, 157, 85, 171, 75, 46)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__17_value;
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___boxed(lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "MaxHeartbeats"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax_spec__1___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax_spec__1___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax_spec__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax_spec__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(74, 15, 185, 205, 141, 171, 77, 191)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax_spec__1___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax_spec__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax_spec__1(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_Pos_skipWhile___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_Pos_skipWhile___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax_spec__0___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 98, .m_capacity = 98, .m_length = 97, .m_data = "Please, add a comment explaining the need for modifying the maxHeartbeat limit, as in\nset_option "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = " "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = " in\n-- reason for change\n..."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__2_value;
static const lean_array_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 157, .m_capacity = 157, .m_length = 156, .m_data = "Using `+native` is not allowed in mathlib: because it trusts the entire Lean compiler (not just the Lean kernel), it could quite possibly be used to prove `"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__4_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__5;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "False"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__6_value),LEAN_SCALAR_PTR_LITERAL(227, 122, 176, 177, 50, 175, 152, 12)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__7_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "`."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__8_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__9;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "refine'"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__10_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "tacticAdmit"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__11_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 163, .m_capacity = 163, .m_length = 162, .m_data = "Using `native_decide` is not allowed in mathlib: because it trusts the entire Lean compiler (not just the Lean kernel), it could quite possibly be used to prove `"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__12_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__13;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 98, .m_capacity = 98, .m_length = 97, .m_data = "The `admit` tactic is discouraged: please strongly consider using the synonymous `sorry` instead."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__14_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__14_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__15_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__16;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 97, .m_capacity = 97, .m_length = 96, .m_data = "The `refine'` tactic is discouraged: please strongly consider using `refine` or `apply` instead."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__17_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__17_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__18 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__18_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__19;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "cases'"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__20 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__20_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "induction'"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__21 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__21_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 92, .m_capacity = 92, .m_length = 91, .m_data = "The `induction'` tactic is discouraged: please strongly consider using `induction` instead."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__22 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__22_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__22_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__23 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__23_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__24;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 106, .m_capacity = 106, .m_length = 105, .m_data = "The `cases'` tactic is discouraged: please strongly consider using `obtain`, `rcases` or `cases` instead."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__25 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__25_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__25_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__26 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__26_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__27_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__27;
static const lean_array_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__28 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__28_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax_spec__2(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__2;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__3;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__4;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3_spec__6___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3_spec__7(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3_spec__7___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3___lam__0___closed__0_value;
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 46, .m_capacity = 46, .m_length = 45, .m_data = "This linter can be disabled with `set_option "};
static const lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1___closed__0 = (const lean_object*)&lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1___closed__1;
static const lean_string_object lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = " false`"};
static const lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1___closed__2 = (const lean_object*)&lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__3(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___lam__1___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__2_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__2_value),LEAN_SCALAR_PTR_LITERAL(191, 70, 156, 159, 11, 54, 216, 94)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__4_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(37, 204, 154, 235, 250, 222, 148, 114)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "DeprecatedSyntaxLinter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__5_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__6_value),LEAN_SCALAR_PTR_LITERAL(153, 84, 20, 206, 10, 9, 187, 209)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(252, 243, 242, 191, 59, 91, 137, 209)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__8_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(205, 251, 104, 16, 130, 107, 81, 40)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__9_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(95, 140, 146, 13, 208, 198, 19, 175)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__10_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(3, 202, 97, 207, 110, 96, 80, 134)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__11_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "deprecatedSyntaxLinter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__12_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__11_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__12_value),LEAN_SCALAR_PTR_LITERAL(76, 28, 153, 173, 1, 127, 62, 152)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__13_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__0_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__13_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__14_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___closed__14_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3_spec__6(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2374602524____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2374602524____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__spec__0(lean_object* v_name_1_, lean_object* v_decl_2_, lean_object* v_ref_3_){
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__spec__0___boxed(lean_object* v_name_29_, lean_object* v_decl_30_, lean_object* v_ref_31_, lean_object* v_a_32_){
_start:
{
lean_object* v_res_33_; 
v_res_33_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__spec__0(v_name_29_, v_decl_30_, v_ref_31_);
lean_dec_ref(v_decl_30_);
return v_res_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_58_; lean_object* v___x_59_; lean_object* v___x_60_; lean_object* v___x_61_; 
v___x_58_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4_));
v___x_59_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4_));
v___x_60_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__9_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4_));
v___x_61_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__spec__0(v___x_58_, v___x_59_, v___x_60_);
return v___x_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4____boxed(lean_object* v_a_62_){
_start:
{
lean_object* v_res_63_; 
v_res_63_ = lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4_();
return v_res_63_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_278512193____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_83_; lean_object* v___x_84_; lean_object* v___x_85_; lean_object* v___x_86_; 
v___x_83_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_278512193____hygCtx___hyg_4_));
v___x_84_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_278512193____hygCtx___hyg_4_));
v___x_85_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_278512193____hygCtx___hyg_4_));
v___x_86_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__spec__0(v___x_83_, v___x_84_, v___x_85_);
return v___x_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_278512193____hygCtx___hyg_4____boxed(lean_object* v_a_87_){
_start:
{
lean_object* v_res_88_; 
v_res_88_ = lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_278512193____hygCtx___hyg_4_();
return v_res_88_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_3986038175____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_108_; lean_object* v___x_109_; lean_object* v___x_110_; lean_object* v___x_111_; 
v___x_108_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_3986038175____hygCtx___hyg_4_));
v___x_109_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_3986038175____hygCtx___hyg_4_));
v___x_110_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_3986038175____hygCtx___hyg_4_));
v___x_111_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__spec__0(v___x_108_, v___x_109_, v___x_110_);
return v___x_111_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_3986038175____hygCtx___hyg_4____boxed(lean_object* v_a_112_){
_start:
{
lean_object* v_res_113_; 
v_res_113_ = lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_3986038175____hygCtx___hyg_4_();
return v_res_113_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_947237925____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_133_; lean_object* v___x_134_; lean_object* v___x_135_; lean_object* v___x_136_; 
v___x_133_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_947237925____hygCtx___hyg_4_));
v___x_134_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_947237925____hygCtx___hyg_4_));
v___x_135_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_947237925____hygCtx___hyg_4_));
v___x_136_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__spec__0(v___x_133_, v___x_134_, v___x_135_);
return v___x_136_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_947237925____hygCtx___hyg_4____boxed(lean_object* v_a_137_){
_start:
{
lean_object* v_res_138_; 
v_res_138_ = lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_947237925____hygCtx___hyg_4_();
return v_res_138_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2792787194____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_158_; lean_object* v___x_159_; lean_object* v___x_160_; lean_object* v___x_161_; 
v___x_158_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2792787194____hygCtx___hyg_4_));
v___x_159_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2792787194____hygCtx___hyg_4_));
v___x_160_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2792787194____hygCtx___hyg_4_));
v___x_161_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__spec__0(v___x_158_, v___x_159_, v___x_160_);
return v___x_161_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2792787194____hygCtx___hyg_4____boxed(lean_object* v_a_162_){
_start:
{
lean_object* v_res_163_; 
v_res_163_ = lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2792787194____hygCtx___hyg_4_();
return v_res_163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2098674011____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_179_; lean_object* v_name_180_; lean_object* v___x_181_; uint8_t v___x_182_; lean_object* v___x_183_; lean_object* v___x_184_; lean_object* v___x_185_; lean_object* v___x_186_; lean_object* v___x_187_; lean_object* v___x_188_; lean_object* v___x_189_; lean_object* v___x_190_; lean_object* v___x_191_; lean_object* v___x_192_; 
v___x_179_ = lp_mathlib_Mathlib_Linter_Style_linter_style_native;
v_name_180_ = lean_ctor_get(v___x_179_, 0);
v___x_181_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2098674011____hygCtx___hyg_4_));
v___x_182_ = 0;
v___x_183_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2098674011____hygCtx___hyg_4_));
v___x_184_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2098674011____hygCtx___hyg_4_));
v___x_185_ = lean_box(0);
lean_inc(v_name_180_);
v___x_186_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_186_, 0, v_name_180_);
v___x_187_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_187_, 0, v___x_184_);
lean_ctor_set(v___x_187_, 1, v___x_185_);
lean_ctor_set(v___x_187_, 2, v___x_186_);
v___x_188_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_188_, 0, v___x_187_);
v___x_189_ = lean_box(v___x_182_);
v___x_190_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_190_, 0, v___x_189_);
lean_ctor_set(v___x_190_, 1, v___x_183_);
lean_ctor_set(v___x_190_, 2, v___x_188_);
v___x_191_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2098674011____hygCtx___hyg_4_));
v___x_192_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__spec__0(v___x_181_, v___x_190_, v___x_191_);
lean_dec_ref_known(v___x_190_, 3);
return v___x_192_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2098674011____hygCtx___hyg_4____boxed(lean_object* v_a_193_){
_start:
{
lean_object* v_res_194_; 
v_res_194_ = lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2098674011____hygCtx___hyg_4_();
return v_res_194_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_303104340____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_214_; lean_object* v___x_215_; lean_object* v___x_216_; lean_object* v___x_217_; 
v___x_214_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_303104340____hygCtx___hyg_4_));
v___x_215_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_303104340____hygCtx___hyg_4_));
v___x_216_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_303104340____hygCtx___hyg_4_));
v___x_217_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4__spec__0(v___x_214_, v___x_215_, v___x_216_);
return v___x_217_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_303104340____hygCtx___hyg_4____boxed(lean_object* v_a_218_){
_start:
{
lean_object* v_res_219_; 
v_res_219_ = lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_303104340____hygCtx___hyg_4_();
return v_res_219_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___lam__0(lean_object* v___x_220_, lean_object* v_x_221_){
_start:
{
lean_object* v___x_222_; uint8_t v___x_223_; 
v___x_222_ = l_Lean_Syntax_getAtomVal(v_x_221_);
v___x_223_ = lean_string_dec_eq(v___x_222_, v___x_220_);
lean_dec_ref(v___x_222_);
return v___x_223_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___lam__0___boxed(lean_object* v___x_224_, lean_object* v_x_225_){
_start:
{
uint8_t v_res_226_; lean_object* v_r_227_; 
v_res_226_ = lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___lam__0(v___x_224_, v_x_225_);
lean_dec(v_x_225_);
lean_dec_ref(v___x_224_);
v_r_227_ = lean_box(v_res_226_);
return v_r_227_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_elem___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment_spec__0(lean_object* v_a_228_, lean_object* v_x_229_){
_start:
{
if (lean_obj_tag(v_x_229_) == 0)
{
uint8_t v___x_230_; 
v___x_230_ = 0;
return v___x_230_;
}
else
{
lean_object* v_head_231_; lean_object* v_tail_232_; uint8_t v___x_233_; 
v_head_231_ = lean_ctor_get(v_x_229_, 0);
v_tail_232_ = lean_ctor_get(v_x_229_, 1);
v___x_233_ = lean_name_eq(v_a_228_, v_head_231_);
if (v___x_233_ == 0)
{
v_x_229_ = v_tail_232_;
goto _start;
}
else
{
return v___x_233_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_elem___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment_spec__0___boxed(lean_object* v_a_235_, lean_object* v_x_236_){
_start:
{
uint8_t v_res_237_; lean_object* v_r_238_; 
v_res_237_ = lp_mathlib_List_elem___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment_spec__0(v_a_235_, v_x_236_);
lean_dec(v_x_236_);
lean_dec(v_a_235_);
v_r_238_ = lean_box(v_res_237_);
return v_r_238_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment(lean_object* v_x_273_){
_start:
{
lean_object* v___x_274_; uint8_t v___x_275_; 
v___x_274_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__4));
lean_inc(v_x_273_);
v___x_275_ = l_Lean_Syntax_isOfKind(v_x_273_, v___x_274_);
if (v___x_275_ == 0)
{
lean_object* v___x_276_; 
lean_dec(v_x_273_);
v___x_276_ = lean_box(0);
return v___x_276_;
}
else
{
lean_object* v___x_277_; lean_object* v___x_278_; lean_object* v___x_279_; uint8_t v___x_280_; 
v___x_277_ = lean_unsigned_to_nat(0u);
v___x_278_ = l_Lean_Syntax_getArg(v_x_273_, v___x_277_);
v___x_279_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__6));
lean_inc(v___x_278_);
v___x_280_ = l_Lean_Syntax_isOfKind(v___x_278_, v___x_279_);
if (v___x_280_ == 0)
{
lean_object* v___x_281_; 
lean_dec(v___x_278_);
lean_dec(v_x_273_);
v___x_281_ = lean_box(0);
return v___x_281_;
}
else
{
lean_object* v___x_282_; lean_object* v___x_283_; uint8_t v___x_284_; 
v___x_282_ = lean_unsigned_to_nat(2u);
v___x_283_ = l_Lean_Syntax_getArg(v___x_278_, v___x_282_);
v___x_284_ = l_Lean_Syntax_matchesNull(v___x_283_, v___x_277_);
if (v___x_284_ == 0)
{
lean_object* v___x_285_; 
lean_dec(v___x_278_);
lean_dec(v_x_273_);
v___x_285_ = lean_box(0);
return v___x_285_;
}
else
{
lean_object* v___x_286_; lean_object* v_n_287_; lean_object* v___x_288_; uint8_t v___x_289_; 
v___x_286_ = lean_unsigned_to_nat(3u);
v_n_287_ = l_Lean_Syntax_getArg(v___x_278_, v___x_286_);
v___x_288_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__8));
lean_inc(v_n_287_);
v___x_289_ = l_Lean_Syntax_isOfKind(v_n_287_, v___x_288_);
if (v___x_289_ == 0)
{
lean_object* v___x_290_; 
lean_dec(v_n_287_);
lean_dec(v___x_278_);
lean_dec(v_x_273_);
v___x_290_ = lean_box(0);
return v___x_290_;
}
else
{
lean_object* v___f_291_; lean_object* v___x_292_; lean_object* v_mh_293_; lean_object* v_opt_294_; lean_object* v___x_312_; lean_object* v___x_313_; uint8_t v___x_314_; 
v___f_291_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__9));
v___x_292_ = lean_unsigned_to_nat(1u);
v_mh_293_ = l_Lean_Syntax_getArg(v___x_278_, v___x_292_);
lean_dec(v___x_278_);
v_opt_294_ = l_Lean_TSyntax_getId(v_mh_293_);
lean_dec(v_mh_293_);
lean_inc(v_opt_294_);
v___x_312_ = l_Lean_Name_components(v_opt_294_);
v___x_313_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__15));
v___x_314_ = lp_mathlib_List_elem___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment_spec__0(v___x_313_, v___x_312_);
lean_dec(v___x_312_);
if (v___x_314_ == 0)
{
if (v___x_289_ == 0)
{
goto v___jp_295_;
}
else
{
lean_object* v___x_315_; 
lean_dec(v_opt_294_);
lean_dec(v_n_287_);
lean_dec(v_x_273_);
v___x_315_ = lean_box(0);
return v___x_315_;
}
}
else
{
goto v___jp_295_;
}
v___jp_295_:
{
lean_object* v___x_296_; 
v___x_296_ = l_Lean_Syntax_find_x3f(v_x_273_, v___f_291_);
if (lean_obj_tag(v___x_296_) == 1)
{
lean_object* v_val_297_; lean_object* v___x_298_; 
v_val_297_ = lean_ctor_get(v___x_296_, 0);
lean_inc(v_val_297_);
lean_dec_ref_known(v___x_296_, 1);
v___x_298_ = l_Lean_Syntax_getTrailing_x3f(v_val_297_);
lean_dec(v_val_297_);
if (lean_obj_tag(v___x_298_) == 0)
{
lean_object* v___x_299_; 
lean_dec(v_opt_294_);
lean_dec(v_n_287_);
v___x_299_ = lean_box(0);
return v___x_299_;
}
else
{
lean_object* v_val_300_; lean_object* v___x_302_; uint8_t v_isShared_303_; uint8_t v_isSharedCheck_310_; 
v_val_300_ = lean_ctor_get(v___x_298_, 0);
v_isSharedCheck_310_ = !lean_is_exclusive(v___x_298_);
if (v_isSharedCheck_310_ == 0)
{
v___x_302_ = v___x_298_;
v_isShared_303_ = v_isSharedCheck_310_;
goto v_resetjp_301_;
}
else
{
lean_inc(v_val_300_);
lean_dec(v___x_298_);
v___x_302_ = lean_box(0);
v_isShared_303_ = v_isSharedCheck_310_;
goto v_resetjp_301_;
}
v_resetjp_301_:
{
lean_object* v___x_304_; lean_object* v___x_305_; lean_object* v___x_306_; lean_object* v___x_308_; 
v___x_304_ = l_Lean_TSyntax_getNat(v_n_287_);
lean_dec(v_n_287_);
v___x_305_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_305_, 0, v___x_304_);
lean_ctor_set(v___x_305_, 1, v_val_300_);
v___x_306_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_306_, 0, v_opt_294_);
lean_ctor_set(v___x_306_, 1, v___x_305_);
if (v_isShared_303_ == 0)
{
lean_ctor_set(v___x_302_, 0, v___x_306_);
v___x_308_ = v___x_302_;
goto v_reusejp_307_;
}
else
{
lean_object* v_reuseFailAlloc_309_; 
v_reuseFailAlloc_309_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_309_, 0, v___x_306_);
v___x_308_ = v_reuseFailAlloc_309_;
goto v_reusejp_307_;
}
v_reusejp_307_:
{
return v___x_308_;
}
}
}
}
else
{
lean_object* v___x_311_; 
lean_dec(v___x_296_);
lean_dec(v_opt_294_);
lean_dec(v_n_287_);
v___x_311_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__14));
return v___x_311_;
}
}
}
}
}
}
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0(lean_object* v_as_340_, size_t v_i_341_, size_t v_stop_342_){
_start:
{
uint8_t v___x_343_; 
v___x_343_ = lean_usize_dec_eq(v_i_341_, v_stop_342_);
if (v___x_343_ == 0)
{
lean_object* v___x_344_; uint8_t v___x_345_; uint8_t v___y_347_; lean_object* v___x_351_; uint8_t v___x_352_; 
v___x_344_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__2));
v___x_345_ = 1;
v___x_351_ = lean_array_uget_borrowed(v_as_340_, v_i_341_);
lean_inc(v___x_351_);
v___x_352_ = l_Lean_Syntax_isOfKind(v___x_351_, v___x_344_);
if (v___x_352_ == 0)
{
v___y_347_ = v___x_352_;
goto v___jp_346_;
}
else
{
lean_object* v___x_353_; lean_object* v___x_354_; lean_object* v___x_355_; uint8_t v___x_356_; 
v___x_353_ = lean_unsigned_to_nat(0u);
v___x_354_ = l_Lean_Syntax_getArg(v___x_351_, v___x_353_);
v___x_355_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__4));
lean_inc(v___x_354_);
v___x_356_ = l_Lean_Syntax_isOfKind(v___x_354_, v___x_355_);
if (v___x_356_ == 0)
{
lean_dec(v___x_354_);
v___y_347_ = v___x_356_;
goto v___jp_346_;
}
else
{
lean_object* v___x_357_; lean_object* v___x_358_; uint8_t v___x_359_; 
v___x_357_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__5));
v___x_358_ = l_Lean_Syntax_getArg(v___x_354_, v___x_353_);
v___x_359_ = l_Lean_Syntax_matchesIdent(v___x_358_, v___x_357_);
lean_dec(v___x_358_);
if (v___x_359_ == 0)
{
lean_dec(v___x_354_);
v___y_347_ = v___x_359_;
goto v___jp_346_;
}
else
{
lean_object* v___x_360_; lean_object* v___x_361_; uint8_t v___x_362_; 
v___x_360_ = lean_unsigned_to_nat(1u);
v___x_361_ = l_Lean_Syntax_getArg(v___x_354_, v___x_360_);
lean_dec(v___x_354_);
v___x_362_ = l_Lean_Syntax_matchesNull(v___x_361_, v___x_353_);
if (v___x_362_ == 0)
{
v___y_347_ = v___x_362_;
goto v___jp_346_;
}
else
{
lean_object* v___x_363_; lean_object* v___x_364_; uint8_t v___x_365_; 
v___x_363_ = lean_unsigned_to_nat(3u);
v___x_364_ = l_Lean_Syntax_getArg(v___x_351_, v___x_360_);
lean_inc(v___x_364_);
v___x_365_ = l_Lean_Syntax_matchesNull(v___x_364_, v___x_363_);
if (v___x_365_ == 0)
{
lean_dec(v___x_364_);
v___y_347_ = v___x_365_;
goto v___jp_346_;
}
else
{
lean_object* v___x_366_; uint8_t v___x_367_; 
v___x_366_ = l_Lean_Syntax_getArg(v___x_364_, v___x_353_);
v___x_367_ = l_Lean_Syntax_matchesNull(v___x_366_, v___x_353_);
if (v___x_367_ == 0)
{
lean_dec(v___x_364_);
v___y_347_ = v___x_367_;
goto v___jp_346_;
}
else
{
lean_object* v___x_368_; uint8_t v___x_369_; 
v___x_368_ = l_Lean_Syntax_getArg(v___x_364_, v___x_360_);
v___x_369_ = l_Lean_Syntax_matchesNull(v___x_368_, v___x_353_);
if (v___x_369_ == 0)
{
lean_dec(v___x_364_);
v___y_347_ = v___x_369_;
goto v___jp_346_;
}
else
{
lean_object* v___x_370_; lean_object* v___x_371_; lean_object* v___x_372_; uint8_t v___x_373_; 
v___x_370_ = lean_unsigned_to_nat(2u);
v___x_371_ = l_Lean_Syntax_getArg(v___x_364_, v___x_370_);
lean_dec(v___x_364_);
v___x_372_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__7));
lean_inc(v___x_371_);
v___x_373_ = l_Lean_Syntax_isOfKind(v___x_371_, v___x_372_);
if (v___x_373_ == 0)
{
lean_dec(v___x_371_);
v___y_347_ = v___x_373_;
goto v___jp_346_;
}
else
{
lean_object* v___x_374_; uint8_t v___x_375_; 
v___x_374_ = l_Lean_Syntax_getArg(v___x_371_, v___x_360_);
v___x_375_ = l_Lean_Syntax_matchesNull(v___x_374_, v___x_353_);
if (v___x_375_ == 0)
{
lean_dec(v___x_371_);
v___y_347_ = v___x_375_;
goto v___jp_346_;
}
else
{
lean_object* v___x_376_; lean_object* v___x_377_; uint8_t v___x_378_; 
v___x_376_ = l_Lean_Syntax_getArg(v___x_371_, v___x_370_);
lean_dec(v___x_371_);
v___x_377_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__9));
v___x_378_ = l_Lean_Syntax_matchesIdent(v___x_376_, v___x_377_);
lean_dec(v___x_376_);
if (v___x_378_ == 0)
{
v___y_347_ = v___x_378_;
goto v___jp_346_;
}
else
{
return v___x_345_;
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
v___jp_346_:
{
if (v___y_347_ == 0)
{
size_t v___x_348_; size_t v___x_349_; 
v___x_348_ = ((size_t)1ULL);
v___x_349_ = lean_usize_add(v_i_341_, v___x_348_);
v_i_341_ = v___x_349_;
goto _start;
}
else
{
return v___x_345_;
}
}
}
else
{
uint8_t v___x_379_; 
v___x_379_ = 0;
return v___x_379_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___boxed(lean_object* v_as_380_, lean_object* v_i_381_, lean_object* v_stop_382_){
_start:
{
size_t v_i_boxed_383_; size_t v_stop_boxed_384_; uint8_t v_res_385_; lean_object* v_r_386_; 
v_i_boxed_383_ = lean_unbox_usize(v_i_381_);
lean_dec(v_i_381_);
v_stop_boxed_384_ = lean_unbox_usize(v_stop_382_);
lean_dec(v_stop_382_);
v_res_385_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0(v_as_380_, v_i_boxed_383_, v_stop_boxed_384_);
lean_dec_ref(v_as_380_);
v_r_386_ = lean_box(v_res_385_);
return v_r_386_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig(lean_object* v_x_442_){
_start:
{
lean_object* v___x_443_; uint8_t v___x_444_; uint8_t v___x_445_; 
v___x_443_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__1));
lean_inc(v_x_442_);
v___x_444_ = l_Lean_Syntax_isOfKind(v_x_442_, v___x_443_);
v___x_445_ = 1;
if (v___x_444_ == 0)
{
lean_object* v___x_446_; uint8_t v___x_447_; 
v___x_446_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__3));
lean_inc(v_x_442_);
v___x_447_ = l_Lean_Syntax_isOfKind(v_x_442_, v___x_446_);
if (v___x_447_ == 0)
{
lean_dec(v_x_442_);
return v___x_447_;
}
else
{
lean_object* v___x_448_; lean_object* v___x_449_; lean_object* v___x_450_; uint8_t v___x_451_; 
v___x_448_ = lean_unsigned_to_nat(0u);
v___x_449_ = l_Lean_Syntax_getArg(v_x_442_, v___x_448_);
lean_dec(v_x_442_);
v___x_450_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__5));
lean_inc(v___x_449_);
v___x_451_ = l_Lean_Syntax_isOfKind(v___x_449_, v___x_450_);
if (v___x_451_ == 0)
{
lean_object* v___x_452_; uint8_t v___x_453_; 
v___x_452_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__7));
lean_inc(v___x_449_);
v___x_453_ = l_Lean_Syntax_isOfKind(v___x_449_, v___x_452_);
if (v___x_453_ == 0)
{
lean_dec(v___x_449_);
return v___x_453_;
}
else
{
lean_object* v___x_454_; lean_object* v___x_455_; lean_object* v___x_456_; uint8_t v___x_457_; 
v___x_454_ = lean_unsigned_to_nat(1u);
v___x_455_ = l_Lean_Syntax_getArg(v___x_449_, v___x_454_);
v___x_456_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__5));
v___x_457_ = l_Lean_Syntax_matchesIdent(v___x_455_, v___x_456_);
if (v___x_457_ == 0)
{
lean_object* v___x_458_; uint8_t v___x_459_; 
v___x_458_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__9));
v___x_459_ = l_Lean_Syntax_matchesIdent(v___x_455_, v___x_458_);
lean_dec(v___x_455_);
if (v___x_459_ == 0)
{
lean_dec(v___x_449_);
return v___x_459_;
}
else
{
lean_object* v___x_460_; lean_object* v___x_461_; lean_object* v___x_462_; uint8_t v___x_463_; 
v___x_460_ = lean_unsigned_to_nat(3u);
v___x_461_ = l_Lean_Syntax_getArg(v___x_449_, v___x_460_);
lean_dec(v___x_449_);
v___x_462_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__11));
lean_inc(v___x_461_);
v___x_463_ = l_Lean_Syntax_isOfKind(v___x_461_, v___x_462_);
if (v___x_463_ == 0)
{
lean_dec(v___x_461_);
return v___x_463_;
}
else
{
lean_object* v___x_464_; uint8_t v___x_465_; 
v___x_464_ = l_Lean_Syntax_getArg(v___x_461_, v___x_454_);
v___x_465_ = l_Lean_Syntax_matchesNull(v___x_464_, v___x_448_);
if (v___x_465_ == 0)
{
lean_dec(v___x_461_);
return v___x_465_;
}
else
{
lean_object* v___x_466_; lean_object* v___x_467_; lean_object* v___x_468_; uint8_t v___x_469_; 
v___x_466_ = lean_unsigned_to_nat(2u);
v___x_467_ = l_Lean_Syntax_getArg(v___x_461_, v___x_466_);
v___x_468_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__13));
lean_inc(v___x_467_);
v___x_469_ = l_Lean_Syntax_isOfKind(v___x_467_, v___x_468_);
if (v___x_469_ == 0)
{
lean_dec(v___x_467_);
lean_dec(v___x_461_);
return v___x_469_;
}
else
{
lean_object* v___x_470_; lean_object* v___x_471_; uint8_t v___x_472_; 
v___x_470_ = l_Lean_Syntax_getArg(v___x_461_, v___x_460_);
v___x_471_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__15));
lean_inc(v___x_470_);
v___x_472_ = l_Lean_Syntax_isOfKind(v___x_470_, v___x_471_);
if (v___x_472_ == 0)
{
lean_dec(v___x_470_);
lean_dec(v___x_467_);
lean_dec(v___x_461_);
return v___x_472_;
}
else
{
lean_object* v___x_473_; uint8_t v___x_474_; 
v___x_473_ = l_Lean_Syntax_getArg(v___x_470_, v___x_448_);
lean_dec(v___x_470_);
v___x_474_ = l_Lean_Syntax_matchesNull(v___x_473_, v___x_448_);
if (v___x_474_ == 0)
{
lean_dec(v___x_467_);
lean_dec(v___x_461_);
return v___x_474_;
}
else
{
lean_object* v___x_475_; lean_object* v___x_476_; uint8_t v___x_477_; 
v___x_475_ = lean_unsigned_to_nat(4u);
v___x_476_ = l_Lean_Syntax_getArg(v___x_461_, v___x_475_);
lean_dec(v___x_461_);
v___x_477_ = l_Lean_Syntax_matchesNull(v___x_476_, v___x_448_);
if (v___x_477_ == 0)
{
lean_dec(v___x_467_);
return v___x_477_;
}
else
{
lean_object* v___x_478_; lean_object* v_t_479_; lean_object* v___x_480_; lean_object* v___x_481_; uint8_t v___x_482_; 
v___x_478_ = l_Lean_Syntax_getArg(v___x_467_, v___x_448_);
lean_dec(v___x_467_);
v_t_479_ = l_Lean_Syntax_getArgs(v___x_478_);
lean_dec(v___x_478_);
v___x_480_ = l_Lean_Syntax_TSepArray_getElems___redArg(v_t_479_);
lean_dec_ref(v_t_479_);
v___x_481_ = lean_array_get_size(v___x_480_);
v___x_482_ = lean_nat_dec_lt(v___x_448_, v___x_481_);
if (v___x_482_ == 0)
{
lean_dec_ref(v___x_480_);
return v___x_457_;
}
else
{
if (v___x_482_ == 0)
{
lean_dec_ref(v___x_480_);
return v___x_457_;
}
else
{
size_t v___x_483_; size_t v___x_484_; uint8_t v___x_485_; 
v___x_483_ = ((size_t)0ULL);
v___x_484_ = lean_usize_of_nat(v___x_481_);
v___x_485_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0(v___x_480_, v___x_483_, v___x_484_);
lean_dec_ref(v___x_480_);
return v___x_485_;
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
else
{
lean_object* v___x_486_; lean_object* v___x_487_; lean_object* v___x_488_; uint8_t v___x_489_; 
lean_dec(v___x_455_);
v___x_486_ = lean_unsigned_to_nat(3u);
v___x_487_ = l_Lean_Syntax_getArg(v___x_449_, v___x_486_);
lean_dec(v___x_449_);
v___x_488_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__9));
v___x_489_ = l_Lean_Syntax_matchesIdent(v___x_487_, v___x_488_);
lean_dec(v___x_487_);
if (v___x_489_ == 0)
{
return v___x_489_;
}
else
{
return v___x_445_;
}
}
}
}
else
{
lean_object* v___x_490_; lean_object* v___x_491_; lean_object* v___x_492_; uint8_t v___x_493_; 
v___x_490_ = lean_unsigned_to_nat(1u);
v___x_491_ = l_Lean_Syntax_getArg(v___x_449_, v___x_490_);
lean_dec(v___x_449_);
v___x_492_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__5));
v___x_493_ = l_Lean_Syntax_matchesIdent(v___x_491_, v___x_492_);
lean_dec(v___x_491_);
if (v___x_493_ == 0)
{
return v___x_493_;
}
else
{
return v___x_445_;
}
}
}
}
else
{
lean_object* v___x_494_; lean_object* v___x_495_; lean_object* v___x_496_; uint8_t v___x_497_; 
v___x_494_ = lean_unsigned_to_nat(0u);
v___x_495_ = l_Lean_Syntax_getArg(v_x_442_, v___x_494_);
lean_dec(v_x_442_);
v___x_496_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__16));
lean_inc(v___x_495_);
v___x_497_ = l_Lean_Syntax_isOfKind(v___x_495_, v___x_496_);
if (v___x_497_ == 0)
{
lean_object* v___x_498_; uint8_t v___x_499_; 
v___x_498_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__17));
lean_inc(v___x_495_);
v___x_499_ = l_Lean_Syntax_isOfKind(v___x_495_, v___x_498_);
if (v___x_499_ == 0)
{
lean_dec(v___x_495_);
return v___x_499_;
}
else
{
lean_object* v___x_500_; lean_object* v___x_501_; lean_object* v___x_502_; uint8_t v___x_503_; 
v___x_500_ = lean_unsigned_to_nat(1u);
v___x_501_ = l_Lean_Syntax_getArg(v___x_495_, v___x_500_);
v___x_502_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__5));
v___x_503_ = l_Lean_Syntax_matchesIdent(v___x_501_, v___x_502_);
if (v___x_503_ == 0)
{
lean_object* v___x_504_; uint8_t v___x_505_; 
v___x_504_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__9));
v___x_505_ = l_Lean_Syntax_matchesIdent(v___x_501_, v___x_504_);
lean_dec(v___x_501_);
if (v___x_505_ == 0)
{
lean_dec(v___x_495_);
return v___x_505_;
}
else
{
lean_object* v___x_506_; lean_object* v___x_507_; lean_object* v___x_508_; uint8_t v___x_509_; 
v___x_506_ = lean_unsigned_to_nat(3u);
v___x_507_ = l_Lean_Syntax_getArg(v___x_495_, v___x_506_);
lean_dec(v___x_495_);
v___x_508_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__11));
lean_inc(v___x_507_);
v___x_509_ = l_Lean_Syntax_isOfKind(v___x_507_, v___x_508_);
if (v___x_509_ == 0)
{
lean_dec(v___x_507_);
return v___x_509_;
}
else
{
lean_object* v___x_510_; uint8_t v___x_511_; 
v___x_510_ = l_Lean_Syntax_getArg(v___x_507_, v___x_500_);
v___x_511_ = l_Lean_Syntax_matchesNull(v___x_510_, v___x_494_);
if (v___x_511_ == 0)
{
lean_dec(v___x_507_);
return v___x_511_;
}
else
{
lean_object* v___x_512_; lean_object* v___x_513_; lean_object* v___x_514_; uint8_t v___x_515_; 
v___x_512_ = lean_unsigned_to_nat(2u);
v___x_513_ = l_Lean_Syntax_getArg(v___x_507_, v___x_512_);
v___x_514_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__13));
lean_inc(v___x_513_);
v___x_515_ = l_Lean_Syntax_isOfKind(v___x_513_, v___x_514_);
if (v___x_515_ == 0)
{
lean_dec(v___x_513_);
lean_dec(v___x_507_);
return v___x_515_;
}
else
{
lean_object* v___x_516_; lean_object* v___x_517_; uint8_t v___x_518_; 
v___x_516_ = l_Lean_Syntax_getArg(v___x_507_, v___x_506_);
v___x_517_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__15));
lean_inc(v___x_516_);
v___x_518_ = l_Lean_Syntax_isOfKind(v___x_516_, v___x_517_);
if (v___x_518_ == 0)
{
lean_dec(v___x_516_);
lean_dec(v___x_513_);
lean_dec(v___x_507_);
return v___x_518_;
}
else
{
lean_object* v___x_519_; uint8_t v___x_520_; 
v___x_519_ = l_Lean_Syntax_getArg(v___x_516_, v___x_494_);
lean_dec(v___x_516_);
v___x_520_ = l_Lean_Syntax_matchesNull(v___x_519_, v___x_494_);
if (v___x_520_ == 0)
{
lean_dec(v___x_513_);
lean_dec(v___x_507_);
return v___x_520_;
}
else
{
lean_object* v___x_521_; lean_object* v___x_522_; uint8_t v___x_523_; 
v___x_521_ = lean_unsigned_to_nat(4u);
v___x_522_ = l_Lean_Syntax_getArg(v___x_507_, v___x_521_);
lean_dec(v___x_507_);
v___x_523_ = l_Lean_Syntax_matchesNull(v___x_522_, v___x_494_);
if (v___x_523_ == 0)
{
lean_dec(v___x_513_);
return v___x_523_;
}
else
{
lean_object* v___x_524_; lean_object* v_t_525_; lean_object* v___x_526_; lean_object* v___x_527_; uint8_t v___x_528_; 
v___x_524_ = l_Lean_Syntax_getArg(v___x_513_, v___x_494_);
lean_dec(v___x_513_);
v_t_525_ = l_Lean_Syntax_getArgs(v___x_524_);
lean_dec(v___x_524_);
v___x_526_ = l_Lean_Syntax_TSepArray_getElems___redArg(v_t_525_);
lean_dec_ref(v_t_525_);
v___x_527_ = lean_array_get_size(v___x_526_);
v___x_528_ = lean_nat_dec_lt(v___x_494_, v___x_527_);
if (v___x_528_ == 0)
{
lean_dec_ref(v___x_526_);
return v___x_503_;
}
else
{
if (v___x_528_ == 0)
{
lean_dec_ref(v___x_526_);
return v___x_503_;
}
else
{
size_t v___x_529_; size_t v___x_530_; uint8_t v___x_531_; 
v___x_529_ = ((size_t)0ULL);
v___x_530_ = lean_usize_of_nat(v___x_527_);
v___x_531_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0(v___x_526_, v___x_529_, v___x_530_);
lean_dec_ref(v___x_526_);
return v___x_531_;
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
else
{
lean_object* v___x_532_; lean_object* v___x_533_; lean_object* v___x_534_; uint8_t v___x_535_; 
lean_dec(v___x_501_);
v___x_532_ = lean_unsigned_to_nat(3u);
v___x_533_ = l_Lean_Syntax_getArg(v___x_495_, v___x_532_);
lean_dec(v___x_495_);
v___x_534_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__9));
v___x_535_ = l_Lean_Syntax_matchesIdent(v___x_533_, v___x_534_);
lean_dec(v___x_533_);
if (v___x_535_ == 0)
{
return v___x_535_;
}
else
{
return v___x_445_;
}
}
}
}
else
{
lean_object* v___x_536_; lean_object* v___x_537_; lean_object* v___x_538_; uint8_t v___x_539_; 
v___x_536_ = lean_unsigned_to_nat(1u);
v___x_537_ = l_Lean_Syntax_getArg(v___x_495_, v___x_536_);
lean_dec(v___x_495_);
v___x_538_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__5));
v___x_539_ = l_Lean_Syntax_matchesIdent(v___x_537_, v___x_538_);
lean_dec(v___x_537_);
if (v___x_539_ == 0)
{
return v___x_539_;
}
else
{
return v___x_445_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___boxed(lean_object* v_x_540_){
_start:
{
uint8_t v_res_541_; lean_object* v_r_542_; 
v_res_541_ = lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig(v_x_540_);
v_r_542_ = lean_box(v_res_541_);
return v_r_542_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax_spec__1(lean_object* v_as_546_, size_t v_i_547_, size_t v_stop_548_, lean_object* v_b_549_){
_start:
{
lean_object* v___y_551_; uint8_t v___x_555_; 
v___x_555_ = lean_usize_dec_eq(v_i_547_, v_stop_548_);
if (v___x_555_ == 0)
{
lean_object* v___x_556_; lean_object* v_fst_557_; lean_object* v___x_558_; uint8_t v___x_559_; 
v___x_556_ = lean_array_uget_borrowed(v_as_546_, v_i_547_);
v_fst_557_ = lean_ctor_get(v___x_556_, 0);
v___x_558_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax_spec__1___closed__1));
v___x_559_ = lean_name_eq(v_fst_557_, v___x_558_);
if (v___x_559_ == 0)
{
lean_object* v___x_560_; 
lean_inc(v___x_556_);
v___x_560_ = lean_array_push(v_b_549_, v___x_556_);
v___y_551_ = v___x_560_;
goto v___jp_550_;
}
else
{
v___y_551_ = v_b_549_;
goto v___jp_550_;
}
}
else
{
return v_b_549_;
}
v___jp_550_:
{
size_t v___x_552_; size_t v___x_553_; 
v___x_552_ = ((size_t)1ULL);
v___x_553_ = lean_usize_add(v_i_547_, v___x_552_);
v_i_547_ = v___x_553_;
v_b_549_ = v___y_551_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax_spec__1___boxed(lean_object* v_as_561_, lean_object* v_i_562_, lean_object* v_stop_563_, lean_object* v_b_564_){
_start:
{
size_t v_i_boxed_565_; size_t v_stop_boxed_566_; lean_object* v_res_567_; 
v_i_boxed_565_ = lean_unbox_usize(v_i_562_);
lean_dec(v_i_562_);
v_stop_boxed_566_ = lean_unbox_usize(v_stop_563_);
lean_dec(v_stop_563_);
v_res_567_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax_spec__1(v_as_561_, v_i_boxed_565_, v_stop_boxed_566_, v_b_564_);
lean_dec_ref(v_as_561_);
return v_res_567_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_Pos_skipWhile___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax_spec__0(lean_object* v_s_568_, lean_object* v_pos_569_){
_start:
{
lean_object* v_str_570_; lean_object* v_startInclusive_571_; lean_object* v_endExclusive_572_; lean_object* v___x_573_; uint8_t v___y_581_; lean_object* v___x_582_; lean_object* v___x_583_; uint8_t v___x_584_; 
v_str_570_ = lean_ctor_get(v_s_568_, 0);
v_startInclusive_571_ = lean_ctor_get(v_s_568_, 1);
v_endExclusive_572_ = lean_ctor_get(v_s_568_, 2);
v___x_573_ = lean_nat_add(v_startInclusive_571_, v_pos_569_);
v___x_582_ = lean_unsigned_to_nat(0u);
v___x_583_ = lean_nat_sub(v_endExclusive_572_, v___x_573_);
v___x_584_ = lean_nat_dec_eq(v___x_582_, v___x_583_);
lean_dec(v___x_583_);
if (v___x_584_ == 0)
{
uint32_t v___x_585_; uint8_t v___y_587_; uint32_t v___x_592_; uint8_t v___x_593_; 
v___x_585_ = lean_string_utf8_get_fast(v_str_570_, v___x_573_);
v___x_592_ = 32;
v___x_593_ = lean_uint32_dec_eq(v___x_585_, v___x_592_);
if (v___x_593_ == 0)
{
uint32_t v___x_594_; uint8_t v___x_595_; 
v___x_594_ = 9;
v___x_595_ = lean_uint32_dec_eq(v___x_585_, v___x_594_);
v___y_587_ = v___x_595_;
goto v___jp_586_;
}
else
{
v___y_587_ = v___x_593_;
goto v___jp_586_;
}
v___jp_586_:
{
if (v___y_587_ == 0)
{
uint32_t v___x_588_; uint8_t v___x_589_; 
v___x_588_ = 13;
v___x_589_ = lean_uint32_dec_eq(v___x_585_, v___x_588_);
if (v___x_589_ == 0)
{
uint32_t v___x_590_; uint8_t v___x_591_; 
v___x_590_ = 10;
v___x_591_ = lean_uint32_dec_eq(v___x_585_, v___x_590_);
v___y_581_ = v___x_591_;
goto v___jp_580_;
}
else
{
v___y_581_ = v___x_589_;
goto v___jp_580_;
}
}
else
{
goto v___jp_574_;
}
}
}
else
{
lean_dec(v___x_573_);
return v_pos_569_;
}
v___jp_574_:
{
lean_object* v___x_575_; lean_object* v___x_576_; lean_object* v___x_577_; uint8_t v___x_578_; 
v___x_575_ = lean_string_utf8_next_fast(v_str_570_, v___x_573_);
v___x_576_ = lean_nat_sub(v___x_575_, v___x_573_);
lean_dec(v___x_573_);
v___x_577_ = lean_nat_add(v_pos_569_, v___x_576_);
lean_dec(v___x_576_);
v___x_578_ = lean_nat_dec_lt(v_pos_569_, v___x_577_);
if (v___x_578_ == 0)
{
lean_dec(v___x_577_);
return v_pos_569_;
}
else
{
lean_dec(v_pos_569_);
v_pos_569_ = v___x_577_;
goto _start;
}
}
v___jp_580_:
{
if (v___y_581_ == 0)
{
lean_dec(v___x_573_);
return v_pos_569_;
}
else
{
goto v___jp_574_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_Pos_skipWhile___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax_spec__0___boxed(lean_object* v_s_596_, lean_object* v_pos_597_){
_start:
{
lean_object* v_res_598_; 
v_res_598_ = lp_mathlib_String_Slice_Pos_skipWhile___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax_spec__0(v_s_596_, v_pos_597_);
lean_dec_ref(v_s_596_);
return v_res_598_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__5(void){
_start:
{
lean_object* v___x_605_; lean_object* v___x_606_; 
v___x_605_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__4));
v___x_606_ = l_Lean_stringToMessageData(v___x_605_);
return v___x_606_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__9(void){
_start:
{
lean_object* v___x_611_; lean_object* v___x_612_; 
v___x_611_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__8));
v___x_612_ = l_Lean_stringToMessageData(v___x_611_);
return v___x_612_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__13(void){
_start:
{
lean_object* v___x_616_; lean_object* v___x_617_; 
v___x_616_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__12));
v___x_617_ = l_Lean_stringToMessageData(v___x_616_);
return v___x_617_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__16(void){
_start:
{
lean_object* v___x_621_; lean_object* v___x_622_; 
v___x_621_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__15));
v___x_622_ = l_Lean_MessageData_ofFormat(v___x_621_);
return v___x_622_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__19(void){
_start:
{
lean_object* v___x_626_; lean_object* v___x_627_; 
v___x_626_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__18));
v___x_627_ = l_Lean_MessageData_ofFormat(v___x_626_);
return v___x_627_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__24(void){
_start:
{
lean_object* v___x_633_; lean_object* v___x_634_; 
v___x_633_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__23));
v___x_634_ = l_Lean_MessageData_ofFormat(v___x_633_);
return v___x_634_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__27(void){
_start:
{
lean_object* v___x_638_; lean_object* v___x_639_; 
v___x_638_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__26));
v___x_639_ = l_Lean_MessageData_ofFormat(v___x_638_);
return v___x_639_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax(lean_object* v_x_642_){
_start:
{
lean_object* v___y_644_; lean_object* v___y_645_; lean_object* v___y_646_; lean_object* v___y_647_; 
if (lean_obj_tag(v_x_642_) == 1)
{
lean_object* v_kind_679_; lean_object* v_args_680_; lean_object* v___y_682_; lean_object* v___x_786_; lean_object* v___x_787_; lean_object* v___x_788_; uint8_t v___x_789_; 
v_kind_679_ = lean_ctor_get(v_x_642_, 1);
v_args_680_ = lean_ctor_get(v_x_642_, 2);
v___x_786_ = lean_unsigned_to_nat(0u);
v___x_787_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__28));
v___x_788_ = lean_array_get_size(v_args_680_);
v___x_789_ = lean_nat_dec_lt(v___x_786_, v___x_788_);
if (v___x_789_ == 0)
{
v___y_682_ = v___x_787_;
goto v___jp_681_;
}
else
{
uint8_t v___x_790_; 
v___x_790_ = lean_nat_dec_le(v___x_788_, v___x_788_);
if (v___x_790_ == 0)
{
if (v___x_789_ == 0)
{
v___y_682_ = v___x_787_;
goto v___jp_681_;
}
else
{
size_t v___x_791_; size_t v___x_792_; lean_object* v___x_793_; 
v___x_791_ = ((size_t)0ULL);
v___x_792_ = lean_usize_of_nat(v___x_788_);
v___x_793_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax_spec__2(v_args_680_, v___x_791_, v___x_792_, v___x_787_);
v___y_682_ = v___x_793_;
goto v___jp_681_;
}
}
else
{
size_t v___x_794_; size_t v___x_795_; lean_object* v___x_796_; 
v___x_794_ = ((size_t)0ULL);
v___x_795_ = lean_usize_of_nat(v___x_788_);
v___x_796_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax_spec__2(v_args_680_, v___x_794_, v___x_795_, v___x_787_);
v___y_682_ = v___x_796_;
goto v___jp_681_;
}
}
v___jp_681_:
{
if (lean_obj_tag(v_kind_679_) == 1)
{
lean_object* v_pre_683_; 
v_pre_683_ = lean_ctor_get(v_kind_679_, 0);
if (lean_obj_tag(v_pre_683_) == 1)
{
lean_object* v_pre_684_; 
v_pre_684_ = lean_ctor_get(v_pre_683_, 0);
if (lean_obj_tag(v_pre_684_) == 1)
{
lean_object* v_pre_685_; 
v_pre_685_ = lean_ctor_get(v_pre_684_, 0);
switch(lean_obj_tag(v_pre_685_))
{
case 1:
{
lean_object* v_pre_686_; 
v_pre_686_ = lean_ctor_get(v_pre_685_, 0);
if (lean_obj_tag(v_pre_686_) == 0)
{
lean_object* v_str_687_; lean_object* v_str_688_; lean_object* v_str_689_; lean_object* v_str_690_; lean_object* v___x_691_; uint8_t v___x_692_; 
v_str_687_ = lean_ctor_get(v_kind_679_, 1);
v_str_688_ = lean_ctor_get(v_pre_683_, 1);
v_str_689_ = lean_ctor_get(v_pre_684_, 1);
v_str_690_ = lean_ctor_get(v_pre_685_, 1);
v___x_691_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__0));
v___x_692_ = lean_string_dec_eq(v_str_690_, v___x_691_);
if (v___x_692_ == 0)
{
lean_dec_ref_known(v_x_642_, 3);
return v___y_682_;
}
else
{
lean_object* v___x_693_; uint8_t v___x_694_; 
v___x_693_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__1));
v___x_694_ = lean_string_dec_eq(v_str_689_, v___x_693_);
if (v___x_694_ == 0)
{
lean_dec_ref_known(v_x_642_, 3);
return v___y_682_;
}
else
{
lean_object* v___x_695_; uint8_t v___x_696_; 
v___x_695_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__2));
v___x_696_ = lean_string_dec_eq(v_str_688_, v___x_695_);
if (v___x_696_ == 0)
{
lean_object* v___x_697_; uint8_t v___x_698_; 
v___x_697_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__0));
v___x_698_ = lean_string_dec_eq(v_str_688_, v___x_697_);
if (v___x_698_ == 0)
{
lean_object* v___x_699_; uint8_t v___x_700_; 
v___x_699_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__2));
v___x_700_ = lean_string_dec_eq(v_str_688_, v___x_699_);
if (v___x_700_ == 0)
{
lean_dec_ref_known(v_x_642_, 3);
return v___y_682_;
}
else
{
lean_object* v___x_701_; uint8_t v___x_702_; 
v___x_701_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__3));
v___x_702_ = lean_string_dec_eq(v_str_687_, v___x_701_);
if (v___x_702_ == 0)
{
lean_dec_ref_known(v_x_642_, 3);
return v___y_682_;
}
else
{
lean_object* v___x_703_; 
lean_inc_ref(v_x_642_);
v___x_703_ = lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment(v_x_642_);
if (lean_obj_tag(v___x_703_) == 0)
{
lean_dec_ref_known(v_x_642_, 3);
return v___y_682_;
}
else
{
lean_object* v_val_704_; lean_object* v_snd_705_; lean_object* v_fst_706_; lean_object* v_fst_707_; lean_object* v_snd_708_; lean_object* v___x_709_; lean_object* v___x_710_; lean_object* v___x_711_; uint8_t v___x_712_; 
v_val_704_ = lean_ctor_get(v___x_703_, 0);
lean_inc(v_val_704_);
lean_dec_ref_known(v___x_703_, 1);
v_snd_705_ = lean_ctor_get(v_val_704_, 1);
lean_inc(v_snd_705_);
v_fst_706_ = lean_ctor_get(v_val_704_, 0);
lean_inc(v_fst_706_);
lean_dec(v_val_704_);
v_fst_707_ = lean_ctor_get(v_snd_705_, 0);
lean_inc(v_fst_707_);
v_snd_708_ = lean_ctor_get(v_snd_705_, 1);
lean_inc(v_snd_708_);
lean_dec(v_snd_705_);
v___x_709_ = lean_unsigned_to_nat(0u);
v___x_710_ = lean_array_get_size(v___y_682_);
v___x_711_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__3));
v___x_712_ = lean_nat_dec_lt(v___x_709_, v___x_710_);
if (v___x_712_ == 0)
{
lean_dec_ref(v___y_682_);
v___y_644_ = v_snd_708_;
v___y_645_ = v_fst_707_;
v___y_646_ = v_fst_706_;
v___y_647_ = v___x_711_;
goto v___jp_643_;
}
else
{
uint8_t v___x_713_; 
v___x_713_ = lean_nat_dec_le(v___x_710_, v___x_710_);
if (v___x_713_ == 0)
{
if (v___x_712_ == 0)
{
lean_dec_ref(v___y_682_);
v___y_644_ = v_snd_708_;
v___y_645_ = v_fst_707_;
v___y_646_ = v_fst_706_;
v___y_647_ = v___x_711_;
goto v___jp_643_;
}
else
{
size_t v___x_714_; size_t v___x_715_; lean_object* v___x_716_; 
v___x_714_ = ((size_t)0ULL);
v___x_715_ = lean_usize_of_nat(v___x_710_);
v___x_716_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax_spec__1(v___y_682_, v___x_714_, v___x_715_, v___x_711_);
lean_dec_ref(v___y_682_);
v___y_644_ = v_snd_708_;
v___y_645_ = v_fst_707_;
v___y_646_ = v_fst_706_;
v___y_647_ = v___x_716_;
goto v___jp_643_;
}
}
else
{
size_t v___x_717_; size_t v___x_718_; lean_object* v___x_719_; 
v___x_717_ = ((size_t)0ULL);
v___x_718_ = lean_usize_of_nat(v___x_710_);
v___x_719_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax_spec__1(v___y_682_, v___x_717_, v___x_718_, v___x_711_);
lean_dec_ref(v___y_682_);
v___y_644_ = v_snd_708_;
v___y_645_ = v_fst_707_;
v___y_646_ = v_fst_706_;
v___y_647_ = v___x_719_;
goto v___jp_643_;
}
}
}
}
}
}
else
{
lean_object* v___x_720_; uint8_t v___x_721_; 
lean_inc_ref(v_kind_679_);
v___x_720_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__0));
v___x_721_ = lean_string_dec_eq(v_str_687_, v___x_720_);
if (v___x_721_ == 0)
{
lean_dec_ref_known(v_kind_679_, 2);
lean_dec_ref_known(v_x_642_, 3);
return v___y_682_;
}
else
{
uint8_t v___x_722_; 
lean_inc_ref(v_x_642_);
v___x_722_ = lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig(v_x_642_);
if (v___x_722_ == 0)
{
lean_dec_ref_known(v_kind_679_, 2);
lean_dec_ref_known(v_x_642_, 3);
return v___y_682_;
}
else
{
lean_object* v___x_723_; lean_object* v___x_724_; lean_object* v___x_725_; lean_object* v___x_726_; lean_object* v___x_727_; lean_object* v___x_728_; lean_object* v___x_729_; lean_object* v___x_730_; lean_object* v___x_731_; 
v___x_723_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__5, &lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__5_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__5);
v___x_724_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__7));
v___x_725_ = l_Lean_MessageData_ofConstName(v___x_724_, v___x_696_);
v___x_726_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_726_, 0, v___x_723_);
lean_ctor_set(v___x_726_, 1, v___x_725_);
v___x_727_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__9, &lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__9_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__9);
v___x_728_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_728_, 0, v___x_726_);
lean_ctor_set(v___x_728_, 1, v___x_727_);
v___x_729_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_729_, 0, v_x_642_);
lean_ctor_set(v___x_729_, 1, v___x_728_);
v___x_730_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_730_, 0, v_kind_679_);
lean_ctor_set(v___x_730_, 1, v___x_729_);
v___x_731_ = lean_array_push(v___y_682_, v___x_730_);
return v___x_731_;
}
}
}
}
else
{
lean_object* v___x_732_; uint8_t v___x_733_; 
lean_inc_ref(v_kind_679_);
v___x_732_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__10));
v___x_733_ = lean_string_dec_eq(v_str_687_, v___x_732_);
if (v___x_733_ == 0)
{
lean_object* v___x_734_; uint8_t v___x_735_; 
v___x_734_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__11));
v___x_735_ = lean_string_dec_eq(v_str_687_, v___x_734_);
if (v___x_735_ == 0)
{
lean_object* v___x_736_; uint8_t v___x_737_; 
v___x_736_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__0));
v___x_737_ = lean_string_dec_eq(v_str_687_, v___x_736_);
if (v___x_737_ == 0)
{
lean_object* v___x_738_; uint8_t v___x_739_; 
v___x_738_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2098674011____hygCtx___hyg_4_));
v___x_739_ = lean_string_dec_eq(v_str_687_, v___x_738_);
if (v___x_739_ == 0)
{
lean_dec_ref_known(v_kind_679_, 2);
lean_dec_ref_known(v_x_642_, 3);
return v___y_682_;
}
else
{
lean_object* v___x_740_; lean_object* v___x_741_; lean_object* v___x_742_; lean_object* v___x_743_; lean_object* v___x_744_; lean_object* v___x_745_; lean_object* v___x_746_; lean_object* v___x_747_; lean_object* v___x_748_; 
v___x_740_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__13, &lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__13_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__13);
v___x_741_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__7));
v___x_742_ = l_Lean_MessageData_ofConstName(v___x_741_, v___x_737_);
v___x_743_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_743_, 0, v___x_740_);
lean_ctor_set(v___x_743_, 1, v___x_742_);
v___x_744_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__9, &lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__9_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__9);
v___x_745_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_745_, 0, v___x_743_);
lean_ctor_set(v___x_745_, 1, v___x_744_);
v___x_746_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_746_, 0, v_x_642_);
lean_ctor_set(v___x_746_, 1, v___x_745_);
v___x_747_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_747_, 0, v_kind_679_);
lean_ctor_set(v___x_747_, 1, v___x_746_);
v___x_748_ = lean_array_push(v___y_682_, v___x_747_);
return v___x_748_;
}
}
else
{
uint8_t v___x_749_; 
lean_inc_ref(v_x_642_);
v___x_749_ = lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig(v_x_642_);
if (v___x_749_ == 0)
{
lean_dec_ref_known(v_kind_679_, 2);
lean_dec_ref_known(v_x_642_, 3);
return v___y_682_;
}
else
{
lean_object* v___x_750_; lean_object* v___x_751_; lean_object* v___x_752_; lean_object* v___x_753_; lean_object* v___x_754_; lean_object* v___x_755_; lean_object* v___x_756_; lean_object* v___x_757_; lean_object* v___x_758_; 
v___x_750_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__5, &lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__5_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__5);
v___x_751_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__7));
v___x_752_ = l_Lean_MessageData_ofConstName(v___x_751_, v___x_735_);
v___x_753_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_753_, 0, v___x_750_);
lean_ctor_set(v___x_753_, 1, v___x_752_);
v___x_754_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__9, &lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__9_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__9);
v___x_755_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_755_, 0, v___x_753_);
lean_ctor_set(v___x_755_, 1, v___x_754_);
v___x_756_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_756_, 0, v_x_642_);
lean_ctor_set(v___x_756_, 1, v___x_755_);
v___x_757_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_757_, 0, v_kind_679_);
lean_ctor_set(v___x_757_, 1, v___x_756_);
v___x_758_ = lean_array_push(v___y_682_, v___x_757_);
return v___x_758_;
}
}
}
else
{
lean_object* v___x_759_; lean_object* v___x_760_; lean_object* v___x_761_; lean_object* v___x_762_; 
v___x_759_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__16, &lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__16_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__16);
v___x_760_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_760_, 0, v_x_642_);
lean_ctor_set(v___x_760_, 1, v___x_759_);
v___x_761_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_761_, 0, v_kind_679_);
lean_ctor_set(v___x_761_, 1, v___x_760_);
v___x_762_ = lean_array_push(v___y_682_, v___x_761_);
return v___x_762_;
}
}
else
{
lean_object* v___x_763_; lean_object* v___x_764_; lean_object* v___x_765_; lean_object* v___x_766_; 
v___x_763_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__19, &lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__19_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__19);
v___x_764_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_764_, 0, v_x_642_);
lean_ctor_set(v___x_764_, 1, v___x_763_);
v___x_765_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_765_, 0, v_kind_679_);
lean_ctor_set(v___x_765_, 1, v___x_764_);
v___x_766_ = lean_array_push(v___y_682_, v___x_765_);
return v___x_766_;
}
}
}
}
}
else
{
lean_dec_ref_known(v_x_642_, 3);
return v___y_682_;
}
}
case 0:
{
lean_object* v_str_767_; lean_object* v_str_768_; lean_object* v_str_769_; lean_object* v___x_770_; uint8_t v___x_771_; 
lean_inc_ref(v_kind_679_);
v_str_767_ = lean_ctor_get(v_kind_679_, 1);
v_str_768_ = lean_ctor_get(v_pre_683_, 1);
v_str_769_ = lean_ctor_get(v_pre_684_, 1);
v___x_770_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4_));
v___x_771_ = lean_string_dec_eq(v_str_769_, v___x_770_);
if (v___x_771_ == 0)
{
lean_dec_ref_known(v_kind_679_, 2);
lean_dec_ref_known(v_x_642_, 3);
return v___y_682_;
}
else
{
lean_object* v___x_772_; uint8_t v___x_773_; 
v___x_772_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__2));
v___x_773_ = lean_string_dec_eq(v_str_768_, v___x_772_);
if (v___x_773_ == 0)
{
lean_dec_ref_known(v_kind_679_, 2);
lean_dec_ref_known(v_x_642_, 3);
return v___y_682_;
}
else
{
lean_object* v___x_774_; uint8_t v___x_775_; 
v___x_774_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__20));
v___x_775_ = lean_string_dec_eq(v_str_767_, v___x_774_);
if (v___x_775_ == 0)
{
lean_object* v___x_776_; uint8_t v___x_777_; 
v___x_776_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__21));
v___x_777_ = lean_string_dec_eq(v_str_767_, v___x_776_);
if (v___x_777_ == 0)
{
lean_dec_ref_known(v_kind_679_, 2);
lean_dec_ref_known(v_x_642_, 3);
return v___y_682_;
}
else
{
lean_object* v___x_778_; lean_object* v___x_779_; lean_object* v___x_780_; lean_object* v___x_781_; 
v___x_778_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__24, &lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__24_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__24);
v___x_779_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_779_, 0, v_x_642_);
lean_ctor_set(v___x_779_, 1, v___x_778_);
v___x_780_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_780_, 0, v_kind_679_);
lean_ctor_set(v___x_780_, 1, v___x_779_);
v___x_781_ = lean_array_push(v___y_682_, v___x_780_);
return v___x_781_;
}
}
else
{
lean_object* v___x_782_; lean_object* v___x_783_; lean_object* v___x_784_; lean_object* v___x_785_; 
v___x_782_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__27, &lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__27_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__27);
v___x_783_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_783_, 0, v_x_642_);
lean_ctor_set(v___x_783_, 1, v___x_782_);
v___x_784_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_784_, 0, v_kind_679_);
lean_ctor_set(v___x_784_, 1, v___x_783_);
v___x_785_ = lean_array_push(v___y_682_, v___x_784_);
return v___x_785_;
}
}
}
}
default: 
{
lean_dec_ref_known(v_x_642_, 3);
return v___y_682_;
}
}
}
else
{
lean_dec_ref_known(v_x_642_, 3);
return v___y_682_;
}
}
else
{
lean_dec_ref_known(v_x_642_, 3);
return v___y_682_;
}
}
else
{
lean_dec_ref_known(v_x_642_, 3);
return v___y_682_;
}
}
}
else
{
lean_object* v___x_797_; 
lean_dec(v_x_642_);
v___x_797_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__28));
return v___x_797_;
}
v___jp_643_:
{
lean_object* v_str_648_; lean_object* v_startPos_649_; lean_object* v_stopPos_650_; lean_object* v___x_652_; uint8_t v_isShared_653_; uint8_t v_isSharedCheck_678_; 
v_str_648_ = lean_ctor_get(v___y_644_, 0);
v_startPos_649_ = lean_ctor_get(v___y_644_, 1);
v_stopPos_650_ = lean_ctor_get(v___y_644_, 2);
v_isSharedCheck_678_ = !lean_is_exclusive(v___y_644_);
if (v_isSharedCheck_678_ == 0)
{
v___x_652_ = v___y_644_;
v_isShared_653_ = v_isSharedCheck_678_;
goto v_resetjp_651_;
}
else
{
lean_inc(v_stopPos_650_);
lean_inc(v_startPos_649_);
lean_inc(v_str_648_);
lean_dec(v___y_644_);
v___x_652_ = lean_box(0);
v_isShared_653_ = v_isSharedCheck_678_;
goto v_resetjp_651_;
}
v_resetjp_651_:
{
lean_object* v___x_654_; lean_object* v___x_655_; lean_object* v___x_656_; lean_object* v___x_658_; 
v___x_654_ = lean_string_utf8_extract(v_str_648_, v_startPos_649_, v_stopPos_650_);
lean_dec(v_stopPos_650_);
lean_dec(v_startPos_649_);
lean_dec_ref(v_str_648_);
v___x_655_ = lean_unsigned_to_nat(0u);
v___x_656_ = lean_string_utf8_byte_size(v___x_654_);
if (v_isShared_653_ == 0)
{
lean_ctor_set(v___x_652_, 2, v___x_656_);
lean_ctor_set(v___x_652_, 1, v___x_655_);
lean_ctor_set(v___x_652_, 0, v___x_654_);
v___x_658_ = v___x_652_;
goto v_reusejp_657_;
}
else
{
lean_object* v_reuseFailAlloc_677_; 
v_reuseFailAlloc_677_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_677_, 0, v___x_654_);
lean_ctor_set(v_reuseFailAlloc_677_, 1, v___x_655_);
lean_ctor_set(v_reuseFailAlloc_677_, 2, v___x_656_);
v___x_658_ = v_reuseFailAlloc_677_;
goto v_reusejp_657_;
}
v_reusejp_657_:
{
lean_object* v___x_659_; lean_object* v___x_660_; uint8_t v___x_661_; 
v___x_659_ = lp_mathlib_String_Slice_Pos_skipWhile___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax_spec__0(v___x_658_, v___x_655_);
lean_dec_ref(v___x_658_);
v___x_660_ = lean_nat_sub(v___x_656_, v___x_659_);
lean_dec(v___x_659_);
v___x_661_ = lean_nat_dec_eq(v___x_660_, v___x_655_);
lean_dec(v___x_660_);
if (v___x_661_ == 0)
{
lean_dec(v___y_646_);
lean_dec(v___y_645_);
lean_dec(v_x_642_);
return v___y_647_;
}
else
{
lean_object* v___x_662_; lean_object* v___x_663_; lean_object* v___x_664_; lean_object* v___x_665_; lean_object* v___x_666_; lean_object* v___x_667_; lean_object* v___x_668_; lean_object* v___x_669_; lean_object* v___x_670_; lean_object* v___x_671_; lean_object* v___x_672_; lean_object* v___x_673_; lean_object* v___x_674_; lean_object* v___x_675_; lean_object* v___x_676_; 
v___x_662_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax_spec__1___closed__1));
v___x_663_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__0));
v___x_664_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v___y_646_, v___x_661_);
v___x_665_ = lean_string_append(v___x_663_, v___x_664_);
lean_dec_ref(v___x_664_);
v___x_666_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__1));
v___x_667_ = lean_string_append(v___x_665_, v___x_666_);
v___x_668_ = l_Nat_reprFast(v___y_645_);
v___x_669_ = lean_string_append(v___x_667_, v___x_668_);
lean_dec_ref(v___x_668_);
v___x_670_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__2));
v___x_671_ = lean_string_append(v___x_669_, v___x_670_);
v___x_672_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_672_, 0, v___x_671_);
v___x_673_ = l_Lean_MessageData_ofFormat(v___x_672_);
v___x_674_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_674_, 0, v_x_642_);
lean_ctor_set(v___x_674_, 1, v___x_673_);
v___x_675_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_675_, 0, v___x_662_);
lean_ctor_set(v___x_675_, 1, v___x_674_);
v___x_676_ = lean_array_push(v___y_647_, v___x_675_);
return v___x_676_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax_spec__2(lean_object* v_as_798_, size_t v_i_799_, size_t v_stop_800_, lean_object* v_b_801_){
_start:
{
uint8_t v___x_802_; 
v___x_802_ = lean_usize_dec_eq(v_i_799_, v_stop_800_);
if (v___x_802_ == 0)
{
lean_object* v___x_803_; lean_object* v___x_804_; lean_object* v___x_805_; size_t v___x_806_; size_t v___x_807_; 
v___x_803_ = lean_array_uget_borrowed(v_as_798_, v_i_799_);
lean_inc(v___x_803_);
v___x_804_ = lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax(v___x_803_);
v___x_805_ = l_Array_append___redArg(v_b_801_, v___x_804_);
lean_dec_ref(v___x_804_);
v___x_806_ = ((size_t)1ULL);
v___x_807_ = lean_usize_add(v_i_799_, v___x_806_);
v_i_799_ = v___x_807_;
v_b_801_ = v___x_805_;
goto _start;
}
else
{
return v_b_801_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax_spec__2___boxed(lean_object* v_as_809_, lean_object* v_i_810_, lean_object* v_stop_811_, lean_object* v_b_812_){
_start:
{
size_t v_i_boxed_813_; size_t v_stop_boxed_814_; lean_object* v_res_815_; 
v_i_boxed_813_ = lean_unbox_usize(v_i_810_);
lean_dec(v_i_810_);
v_stop_boxed_814_ = lean_unbox_usize(v_stop_811_);
lean_dec(v_stop_811_);
v_res_815_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax_spec__2(v_as_809_, v_i_boxed_813_, v_stop_boxed_814_, v_b_812_);
lean_dec_ref(v_as_809_);
return v_res_815_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__0(void){
_start:
{
lean_object* v___x_816_; 
v___x_816_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_816_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__1(void){
_start:
{
lean_object* v___x_817_; lean_object* v___x_818_; 
v___x_817_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__0, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__0_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__0);
v___x_818_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_818_, 0, v___x_817_);
return v___x_818_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__2(void){
_start:
{
lean_object* v___x_819_; lean_object* v___x_820_; lean_object* v___x_821_; 
v___x_819_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__1);
v___x_820_ = lean_unsigned_to_nat(0u);
v___x_821_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_821_, 0, v___x_820_);
lean_ctor_set(v___x_821_, 1, v___x_820_);
lean_ctor_set(v___x_821_, 2, v___x_820_);
lean_ctor_set(v___x_821_, 3, v___x_820_);
lean_ctor_set(v___x_821_, 4, v___x_819_);
lean_ctor_set(v___x_821_, 5, v___x_819_);
lean_ctor_set(v___x_821_, 6, v___x_819_);
lean_ctor_set(v___x_821_, 7, v___x_819_);
lean_ctor_set(v___x_821_, 8, v___x_819_);
lean_ctor_set(v___x_821_, 9, v___x_819_);
return v___x_821_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__3(void){
_start:
{
lean_object* v___x_822_; lean_object* v___x_823_; lean_object* v___x_824_; 
v___x_822_ = lean_unsigned_to_nat(32u);
v___x_823_ = lean_mk_empty_array_with_capacity(v___x_822_);
v___x_824_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_824_, 0, v___x_823_);
return v___x_824_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__4(void){
_start:
{
size_t v___x_825_; lean_object* v___x_826_; lean_object* v___x_827_; lean_object* v___x_828_; lean_object* v___x_829_; lean_object* v___x_830_; 
v___x_825_ = ((size_t)5ULL);
v___x_826_ = lean_unsigned_to_nat(0u);
v___x_827_ = lean_unsigned_to_nat(32u);
v___x_828_ = lean_mk_empty_array_with_capacity(v___x_827_);
v___x_829_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__3, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__3_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__3);
v___x_830_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_830_, 0, v___x_829_);
lean_ctor_set(v___x_830_, 1, v___x_828_);
lean_ctor_set(v___x_830_, 2, v___x_826_);
lean_ctor_set(v___x_830_, 3, v___x_826_);
lean_ctor_set_usize(v___x_830_, 4, v___x_825_);
return v___x_830_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__5(void){
_start:
{
lean_object* v___x_831_; lean_object* v___x_832_; lean_object* v___x_833_; lean_object* v___x_834_; 
v___x_831_ = lean_box(1);
v___x_832_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__4, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__4_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__4);
v___x_833_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__1);
v___x_834_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_834_, 0, v___x_833_);
lean_ctor_set(v___x_834_, 1, v___x_832_);
lean_ctor_set(v___x_834_, 2, v___x_831_);
return v___x_834_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3_spec__6___redArg(lean_object* v_msgData_835_, lean_object* v___y_836_){
_start:
{
lean_object* v___x_838_; lean_object* v_env_839_; lean_object* v___x_840_; lean_object* v_scopes_841_; lean_object* v___x_842_; lean_object* v___x_843_; lean_object* v_opts_844_; lean_object* v___x_845_; lean_object* v___x_846_; lean_object* v___x_847_; lean_object* v___x_848_; lean_object* v___x_849_; 
v___x_838_ = lean_st_ref_get(v___y_836_);
v_env_839_ = lean_ctor_get(v___x_838_, 0);
lean_inc_ref(v_env_839_);
lean_dec(v___x_838_);
v___x_840_ = lean_st_ref_get(v___y_836_);
v_scopes_841_ = lean_ctor_get(v___x_840_, 2);
lean_inc(v_scopes_841_);
lean_dec(v___x_840_);
v___x_842_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_843_ = l_List_head_x21___redArg(v___x_842_, v_scopes_841_);
lean_dec(v_scopes_841_);
v_opts_844_ = lean_ctor_get(v___x_843_, 1);
lean_inc_ref(v_opts_844_);
lean_dec(v___x_843_);
v___x_845_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__2, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__2_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__2);
v___x_846_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__5, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__5_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__5);
v___x_847_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_847_, 0, v_env_839_);
lean_ctor_set(v___x_847_, 1, v___x_845_);
lean_ctor_set(v___x_847_, 2, v___x_846_);
lean_ctor_set(v___x_847_, 3, v_opts_844_);
v___x_848_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_848_, 0, v___x_847_);
lean_ctor_set(v___x_848_, 1, v_msgData_835_);
v___x_849_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_849_, 0, v___x_848_);
return v___x_849_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3_spec__6___redArg___boxed(lean_object* v_msgData_850_, lean_object* v___y_851_, lean_object* v___y_852_){
_start:
{
lean_object* v_res_853_; 
v_res_853_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3_spec__6___redArg(v_msgData_850_, v___y_851_);
lean_dec(v___y_851_);
return v_res_853_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3_spec__7(lean_object* v_opts_854_, lean_object* v_opt_855_){
_start:
{
lean_object* v_name_856_; lean_object* v_defValue_857_; lean_object* v_map_858_; lean_object* v___x_859_; 
v_name_856_ = lean_ctor_get(v_opt_855_, 0);
v_defValue_857_ = lean_ctor_get(v_opt_855_, 1);
v_map_858_ = lean_ctor_get(v_opts_854_, 0);
v___x_859_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_858_, v_name_856_);
if (lean_obj_tag(v___x_859_) == 0)
{
uint8_t v___x_860_; 
v___x_860_ = lean_unbox(v_defValue_857_);
return v___x_860_;
}
else
{
lean_object* v_val_861_; 
v_val_861_ = lean_ctor_get(v___x_859_, 0);
lean_inc(v_val_861_);
lean_dec_ref_known(v___x_859_, 1);
if (lean_obj_tag(v_val_861_) == 1)
{
uint8_t v_v_862_; 
v_v_862_ = lean_ctor_get_uint8(v_val_861_, 0);
lean_dec_ref_known(v_val_861_, 0);
return v_v_862_;
}
else
{
uint8_t v___x_863_; 
lean_dec(v_val_861_);
v___x_863_ = lean_unbox(v_defValue_857_);
return v___x_863_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3_spec__7___boxed(lean_object* v_opts_864_, lean_object* v_opt_865_){
_start:
{
uint8_t v_res_866_; lean_object* v_r_867_; 
v_res_866_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3_spec__7(v_opts_864_, v_opt_865_);
lean_dec_ref(v_opt_865_);
lean_dec_ref(v_opts_864_);
v_r_867_ = lean_box(v_res_866_);
return v_r_867_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3___lam__0(uint8_t v___y_869_, uint8_t v_suppressElabErrors_870_, lean_object* v_x_871_){
_start:
{
if (lean_obj_tag(v_x_871_) == 1)
{
lean_object* v_pre_872_; 
v_pre_872_ = lean_ctor_get(v_x_871_, 0);
if (lean_obj_tag(v_pre_872_) == 0)
{
lean_object* v_str_873_; lean_object* v___x_874_; uint8_t v___x_875_; 
v_str_873_ = lean_ctor_get(v_x_871_, 1);
v___x_874_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3___lam__0___closed__0));
v___x_875_ = lean_string_dec_eq(v_str_873_, v___x_874_);
if (v___x_875_ == 0)
{
return v___y_869_;
}
else
{
return v_suppressElabErrors_870_;
}
}
else
{
return v___y_869_;
}
}
else
{
return v___y_869_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3___lam__0___boxed(lean_object* v___y_876_, lean_object* v_suppressElabErrors_877_, lean_object* v_x_878_){
_start:
{
uint8_t v___y_12478__boxed_879_; uint8_t v_suppressElabErrors_boxed_880_; uint8_t v_res_881_; lean_object* v_r_882_; 
v___y_12478__boxed_879_ = lean_unbox(v___y_876_);
v_suppressElabErrors_boxed_880_ = lean_unbox(v_suppressElabErrors_877_);
v_res_881_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3___lam__0(v___y_12478__boxed_879_, v_suppressElabErrors_boxed_880_, v_x_878_);
lean_dec(v_x_878_);
v_r_882_ = lean_box(v_res_881_);
return v_r_882_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3(lean_object* v_ref_883_, lean_object* v_msgData_884_, uint8_t v_severity_885_, uint8_t v_isSilent_886_, lean_object* v___y_887_, lean_object* v___y_888_){
_start:
{
lean_object* v___y_891_; uint8_t v___y_892_; uint8_t v___y_893_; lean_object* v___y_894_; lean_object* v___y_895_; lean_object* v___y_896_; lean_object* v___y_897_; lean_object* v___y_898_; uint8_t v___y_955_; lean_object* v___y_956_; uint8_t v___y_957_; uint8_t v___y_958_; lean_object* v___y_959_; uint8_t v___y_983_; uint8_t v___y_984_; uint8_t v___y_985_; lean_object* v___y_986_; lean_object* v___y_987_; uint8_t v___y_991_; uint8_t v___y_992_; uint8_t v___y_993_; uint8_t v___x_1008_; uint8_t v___y_1010_; uint8_t v___y_1011_; uint8_t v___y_1012_; uint8_t v___y_1014_; uint8_t v___x_1026_; 
v___x_1008_ = 2;
v___x_1026_ = l_Lean_instBEqMessageSeverity_beq(v_severity_885_, v___x_1008_);
if (v___x_1026_ == 0)
{
v___y_1014_ = v___x_1026_;
goto v___jp_1013_;
}
else
{
uint8_t v___x_1027_; 
lean_inc_ref(v_msgData_884_);
v___x_1027_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_884_);
v___y_1014_ = v___x_1027_;
goto v___jp_1013_;
}
v___jp_890_:
{
lean_object* v___x_899_; 
v___x_899_ = l_Lean_Elab_Command_getScope___redArg(v___y_898_);
if (lean_obj_tag(v___x_899_) == 0)
{
lean_object* v_a_900_; lean_object* v___x_901_; 
v_a_900_ = lean_ctor_get(v___x_899_, 0);
lean_inc(v_a_900_);
lean_dec_ref_known(v___x_899_, 1);
v___x_901_ = l_Lean_Elab_Command_getScope___redArg(v___y_898_);
if (lean_obj_tag(v___x_901_) == 0)
{
lean_object* v_a_902_; lean_object* v___x_904_; uint8_t v_isShared_905_; uint8_t v_isSharedCheck_937_; 
v_a_902_ = lean_ctor_get(v___x_901_, 0);
v_isSharedCheck_937_ = !lean_is_exclusive(v___x_901_);
if (v_isSharedCheck_937_ == 0)
{
v___x_904_ = v___x_901_;
v_isShared_905_ = v_isSharedCheck_937_;
goto v_resetjp_903_;
}
else
{
lean_inc(v_a_902_);
lean_dec(v___x_901_);
v___x_904_ = lean_box(0);
v_isShared_905_ = v_isSharedCheck_937_;
goto v_resetjp_903_;
}
v_resetjp_903_:
{
lean_object* v___x_906_; lean_object* v_currNamespace_907_; lean_object* v_openDecls_908_; lean_object* v_env_909_; lean_object* v_messages_910_; lean_object* v_scopes_911_; lean_object* v_usedQuotCtxts_912_; lean_object* v_nextMacroScope_913_; lean_object* v_maxRecDepth_914_; lean_object* v_ngen_915_; lean_object* v_auxDeclNGen_916_; lean_object* v_infoState_917_; lean_object* v_traceState_918_; lean_object* v_snapshotTasks_919_; lean_object* v_prevLinterStates_920_; lean_object* v___x_922_; uint8_t v_isShared_923_; uint8_t v_isSharedCheck_936_; 
v___x_906_ = lean_st_ref_take(v___y_898_);
v_currNamespace_907_ = lean_ctor_get(v_a_900_, 2);
lean_inc(v_currNamespace_907_);
lean_dec(v_a_900_);
v_openDecls_908_ = lean_ctor_get(v_a_902_, 3);
lean_inc(v_openDecls_908_);
lean_dec(v_a_902_);
v_env_909_ = lean_ctor_get(v___x_906_, 0);
v_messages_910_ = lean_ctor_get(v___x_906_, 1);
v_scopes_911_ = lean_ctor_get(v___x_906_, 2);
v_usedQuotCtxts_912_ = lean_ctor_get(v___x_906_, 3);
v_nextMacroScope_913_ = lean_ctor_get(v___x_906_, 4);
v_maxRecDepth_914_ = lean_ctor_get(v___x_906_, 5);
v_ngen_915_ = lean_ctor_get(v___x_906_, 6);
v_auxDeclNGen_916_ = lean_ctor_get(v___x_906_, 7);
v_infoState_917_ = lean_ctor_get(v___x_906_, 8);
v_traceState_918_ = lean_ctor_get(v___x_906_, 9);
v_snapshotTasks_919_ = lean_ctor_get(v___x_906_, 10);
v_prevLinterStates_920_ = lean_ctor_get(v___x_906_, 11);
v_isSharedCheck_936_ = !lean_is_exclusive(v___x_906_);
if (v_isSharedCheck_936_ == 0)
{
v___x_922_ = v___x_906_;
v_isShared_923_ = v_isSharedCheck_936_;
goto v_resetjp_921_;
}
else
{
lean_inc(v_prevLinterStates_920_);
lean_inc(v_snapshotTasks_919_);
lean_inc(v_traceState_918_);
lean_inc(v_infoState_917_);
lean_inc(v_auxDeclNGen_916_);
lean_inc(v_ngen_915_);
lean_inc(v_maxRecDepth_914_);
lean_inc(v_nextMacroScope_913_);
lean_inc(v_usedQuotCtxts_912_);
lean_inc(v_scopes_911_);
lean_inc(v_messages_910_);
lean_inc(v_env_909_);
lean_dec(v___x_906_);
v___x_922_ = lean_box(0);
v_isShared_923_ = v_isSharedCheck_936_;
goto v_resetjp_921_;
}
v_resetjp_921_:
{
lean_object* v___x_924_; lean_object* v___x_925_; lean_object* v___x_926_; lean_object* v___x_927_; lean_object* v___x_929_; 
v___x_924_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_924_, 0, v_currNamespace_907_);
lean_ctor_set(v___x_924_, 1, v_openDecls_908_);
v___x_925_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_925_, 0, v___x_924_);
lean_ctor_set(v___x_925_, 1, v___y_891_);
lean_inc_ref(v___y_896_);
lean_inc_ref(v___y_897_);
v___x_926_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_926_, 0, v___y_897_);
lean_ctor_set(v___x_926_, 1, v___y_895_);
lean_ctor_set(v___x_926_, 2, v___y_894_);
lean_ctor_set(v___x_926_, 3, v___y_896_);
lean_ctor_set(v___x_926_, 4, v___x_925_);
lean_ctor_set_uint8(v___x_926_, sizeof(void*)*5, v___y_893_);
lean_ctor_set_uint8(v___x_926_, sizeof(void*)*5 + 1, v___y_892_);
lean_ctor_set_uint8(v___x_926_, sizeof(void*)*5 + 2, v_isSilent_886_);
v___x_927_ = l_Lean_MessageLog_add(v___x_926_, v_messages_910_);
if (v_isShared_923_ == 0)
{
lean_ctor_set(v___x_922_, 1, v___x_927_);
v___x_929_ = v___x_922_;
goto v_reusejp_928_;
}
else
{
lean_object* v_reuseFailAlloc_935_; 
v_reuseFailAlloc_935_ = lean_alloc_ctor(0, 12, 0);
lean_ctor_set(v_reuseFailAlloc_935_, 0, v_env_909_);
lean_ctor_set(v_reuseFailAlloc_935_, 1, v___x_927_);
lean_ctor_set(v_reuseFailAlloc_935_, 2, v_scopes_911_);
lean_ctor_set(v_reuseFailAlloc_935_, 3, v_usedQuotCtxts_912_);
lean_ctor_set(v_reuseFailAlloc_935_, 4, v_nextMacroScope_913_);
lean_ctor_set(v_reuseFailAlloc_935_, 5, v_maxRecDepth_914_);
lean_ctor_set(v_reuseFailAlloc_935_, 6, v_ngen_915_);
lean_ctor_set(v_reuseFailAlloc_935_, 7, v_auxDeclNGen_916_);
lean_ctor_set(v_reuseFailAlloc_935_, 8, v_infoState_917_);
lean_ctor_set(v_reuseFailAlloc_935_, 9, v_traceState_918_);
lean_ctor_set(v_reuseFailAlloc_935_, 10, v_snapshotTasks_919_);
lean_ctor_set(v_reuseFailAlloc_935_, 11, v_prevLinterStates_920_);
v___x_929_ = v_reuseFailAlloc_935_;
goto v_reusejp_928_;
}
v_reusejp_928_:
{
lean_object* v___x_930_; lean_object* v___x_931_; lean_object* v___x_933_; 
v___x_930_ = lean_st_ref_set(v___y_898_, v___x_929_);
v___x_931_ = lean_box(0);
if (v_isShared_905_ == 0)
{
lean_ctor_set(v___x_904_, 0, v___x_931_);
v___x_933_ = v___x_904_;
goto v_reusejp_932_;
}
else
{
lean_object* v_reuseFailAlloc_934_; 
v_reuseFailAlloc_934_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_934_, 0, v___x_931_);
v___x_933_ = v_reuseFailAlloc_934_;
goto v_reusejp_932_;
}
v_reusejp_932_:
{
return v___x_933_;
}
}
}
}
}
else
{
lean_object* v_a_938_; lean_object* v___x_940_; uint8_t v_isShared_941_; uint8_t v_isSharedCheck_945_; 
lean_dec(v_a_900_);
lean_dec_ref(v___y_895_);
lean_dec(v___y_894_);
lean_dec_ref(v___y_891_);
v_a_938_ = lean_ctor_get(v___x_901_, 0);
v_isSharedCheck_945_ = !lean_is_exclusive(v___x_901_);
if (v_isSharedCheck_945_ == 0)
{
v___x_940_ = v___x_901_;
v_isShared_941_ = v_isSharedCheck_945_;
goto v_resetjp_939_;
}
else
{
lean_inc(v_a_938_);
lean_dec(v___x_901_);
v___x_940_ = lean_box(0);
v_isShared_941_ = v_isSharedCheck_945_;
goto v_resetjp_939_;
}
v_resetjp_939_:
{
lean_object* v___x_943_; 
if (v_isShared_941_ == 0)
{
v___x_943_ = v___x_940_;
goto v_reusejp_942_;
}
else
{
lean_object* v_reuseFailAlloc_944_; 
v_reuseFailAlloc_944_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_944_, 0, v_a_938_);
v___x_943_ = v_reuseFailAlloc_944_;
goto v_reusejp_942_;
}
v_reusejp_942_:
{
return v___x_943_;
}
}
}
}
else
{
lean_object* v_a_946_; lean_object* v___x_948_; uint8_t v_isShared_949_; uint8_t v_isSharedCheck_953_; 
lean_dec_ref(v___y_895_);
lean_dec(v___y_894_);
lean_dec_ref(v___y_891_);
v_a_946_ = lean_ctor_get(v___x_899_, 0);
v_isSharedCheck_953_ = !lean_is_exclusive(v___x_899_);
if (v_isSharedCheck_953_ == 0)
{
v___x_948_ = v___x_899_;
v_isShared_949_ = v_isSharedCheck_953_;
goto v_resetjp_947_;
}
else
{
lean_inc(v_a_946_);
lean_dec(v___x_899_);
v___x_948_ = lean_box(0);
v_isShared_949_ = v_isSharedCheck_953_;
goto v_resetjp_947_;
}
v_resetjp_947_:
{
lean_object* v___x_951_; 
if (v_isShared_949_ == 0)
{
v___x_951_ = v___x_948_;
goto v_reusejp_950_;
}
else
{
lean_object* v_reuseFailAlloc_952_; 
v_reuseFailAlloc_952_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_952_, 0, v_a_946_);
v___x_951_ = v_reuseFailAlloc_952_;
goto v_reusejp_950_;
}
v_reusejp_950_:
{
return v___x_951_;
}
}
}
}
v___jp_954_:
{
lean_object* v_fileName_960_; lean_object* v_fileMap_961_; uint8_t v_suppressElabErrors_962_; lean_object* v___x_963_; lean_object* v___x_964_; lean_object* v_a_965_; lean_object* v___x_967_; uint8_t v_isShared_968_; uint8_t v_isSharedCheck_981_; 
v_fileName_960_ = lean_ctor_get(v___y_887_, 0);
v_fileMap_961_ = lean_ctor_get(v___y_887_, 1);
v_suppressElabErrors_962_ = lean_ctor_get_uint8(v___y_887_, sizeof(void*)*10);
v___x_963_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_884_);
v___x_964_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3_spec__6___redArg(v___x_963_, v___y_888_);
v_a_965_ = lean_ctor_get(v___x_964_, 0);
v_isSharedCheck_981_ = !lean_is_exclusive(v___x_964_);
if (v_isSharedCheck_981_ == 0)
{
v___x_967_ = v___x_964_;
v_isShared_968_ = v_isSharedCheck_981_;
goto v_resetjp_966_;
}
else
{
lean_inc(v_a_965_);
lean_dec(v___x_964_);
v___x_967_ = lean_box(0);
v_isShared_968_ = v_isSharedCheck_981_;
goto v_resetjp_966_;
}
v_resetjp_966_:
{
lean_object* v___x_969_; lean_object* v___x_970_; lean_object* v___x_971_; lean_object* v___x_972_; 
lean_inc_ref_n(v_fileMap_961_, 2);
v___x_969_ = l_Lean_FileMap_toPosition(v_fileMap_961_, v___y_956_);
lean_dec(v___y_956_);
v___x_970_ = l_Lean_FileMap_toPosition(v_fileMap_961_, v___y_959_);
lean_dec(v___y_959_);
v___x_971_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_971_, 0, v___x_970_);
v___x_972_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__10));
if (v_suppressElabErrors_962_ == 0)
{
lean_del_object(v___x_967_);
v___y_891_ = v_a_965_;
v___y_892_ = v___y_957_;
v___y_893_ = v___y_958_;
v___y_894_ = v___x_971_;
v___y_895_ = v___x_969_;
v___y_896_ = v___x_972_;
v___y_897_ = v_fileName_960_;
v___y_898_ = v___y_888_;
goto v___jp_890_;
}
else
{
lean_object* v___x_973_; lean_object* v___x_974_; lean_object* v___f_975_; uint8_t v___x_976_; 
v___x_973_ = lean_box(v___y_955_);
v___x_974_ = lean_box(v_suppressElabErrors_962_);
v___f_975_ = lean_alloc_closure((void*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3___lam__0___boxed), 3, 2);
lean_closure_set(v___f_975_, 0, v___x_973_);
lean_closure_set(v___f_975_, 1, v___x_974_);
lean_inc(v_a_965_);
v___x_976_ = l_Lean_MessageData_hasTag(v___f_975_, v_a_965_);
if (v___x_976_ == 0)
{
lean_object* v___x_977_; lean_object* v___x_979_; 
lean_dec_ref_known(v___x_971_, 1);
lean_dec_ref(v___x_969_);
lean_dec(v_a_965_);
v___x_977_ = lean_box(0);
if (v_isShared_968_ == 0)
{
lean_ctor_set(v___x_967_, 0, v___x_977_);
v___x_979_ = v___x_967_;
goto v_reusejp_978_;
}
else
{
lean_object* v_reuseFailAlloc_980_; 
v_reuseFailAlloc_980_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_980_, 0, v___x_977_);
v___x_979_ = v_reuseFailAlloc_980_;
goto v_reusejp_978_;
}
v_reusejp_978_:
{
return v___x_979_;
}
}
else
{
lean_del_object(v___x_967_);
v___y_891_ = v_a_965_;
v___y_892_ = v___y_957_;
v___y_893_ = v___y_958_;
v___y_894_ = v___x_971_;
v___y_895_ = v___x_969_;
v___y_896_ = v___x_972_;
v___y_897_ = v_fileName_960_;
v___y_898_ = v___y_888_;
goto v___jp_890_;
}
}
}
}
v___jp_982_:
{
lean_object* v___x_988_; 
v___x_988_ = l_Lean_Syntax_getTailPos_x3f(v___y_986_, v___y_985_);
lean_dec(v___y_986_);
if (lean_obj_tag(v___x_988_) == 0)
{
lean_inc(v___y_987_);
v___y_955_ = v___y_983_;
v___y_956_ = v___y_987_;
v___y_957_ = v___y_984_;
v___y_958_ = v___y_985_;
v___y_959_ = v___y_987_;
goto v___jp_954_;
}
else
{
lean_object* v_val_989_; 
v_val_989_ = lean_ctor_get(v___x_988_, 0);
lean_inc(v_val_989_);
lean_dec_ref_known(v___x_988_, 1);
v___y_955_ = v___y_983_;
v___y_956_ = v___y_987_;
v___y_957_ = v___y_984_;
v___y_958_ = v___y_985_;
v___y_959_ = v_val_989_;
goto v___jp_954_;
}
}
v___jp_990_:
{
lean_object* v___x_994_; 
v___x_994_ = l_Lean_Elab_Command_getRef___redArg(v___y_887_);
if (lean_obj_tag(v___x_994_) == 0)
{
lean_object* v_a_995_; lean_object* v_ref_996_; lean_object* v___x_997_; 
v_a_995_ = lean_ctor_get(v___x_994_, 0);
lean_inc(v_a_995_);
lean_dec_ref_known(v___x_994_, 1);
v_ref_996_ = l_Lean_replaceRef(v_ref_883_, v_a_995_);
lean_dec(v_a_995_);
v___x_997_ = l_Lean_Syntax_getPos_x3f(v_ref_996_, v___y_992_);
if (lean_obj_tag(v___x_997_) == 0)
{
lean_object* v___x_998_; 
v___x_998_ = lean_unsigned_to_nat(0u);
v___y_983_ = v___y_991_;
v___y_984_ = v___y_993_;
v___y_985_ = v___y_992_;
v___y_986_ = v_ref_996_;
v___y_987_ = v___x_998_;
goto v___jp_982_;
}
else
{
lean_object* v_val_999_; 
v_val_999_ = lean_ctor_get(v___x_997_, 0);
lean_inc(v_val_999_);
lean_dec_ref_known(v___x_997_, 1);
v___y_983_ = v___y_991_;
v___y_984_ = v___y_993_;
v___y_985_ = v___y_992_;
v___y_986_ = v_ref_996_;
v___y_987_ = v_val_999_;
goto v___jp_982_;
}
}
else
{
lean_object* v_a_1000_; lean_object* v___x_1002_; uint8_t v_isShared_1003_; uint8_t v_isSharedCheck_1007_; 
lean_dec_ref(v_msgData_884_);
v_a_1000_ = lean_ctor_get(v___x_994_, 0);
v_isSharedCheck_1007_ = !lean_is_exclusive(v___x_994_);
if (v_isSharedCheck_1007_ == 0)
{
v___x_1002_ = v___x_994_;
v_isShared_1003_ = v_isSharedCheck_1007_;
goto v_resetjp_1001_;
}
else
{
lean_inc(v_a_1000_);
lean_dec(v___x_994_);
v___x_1002_ = lean_box(0);
v_isShared_1003_ = v_isSharedCheck_1007_;
goto v_resetjp_1001_;
}
v_resetjp_1001_:
{
lean_object* v___x_1005_; 
if (v_isShared_1003_ == 0)
{
v___x_1005_ = v___x_1002_;
goto v_reusejp_1004_;
}
else
{
lean_object* v_reuseFailAlloc_1006_; 
v_reuseFailAlloc_1006_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1006_, 0, v_a_1000_);
v___x_1005_ = v_reuseFailAlloc_1006_;
goto v_reusejp_1004_;
}
v_reusejp_1004_:
{
return v___x_1005_;
}
}
}
}
v___jp_1009_:
{
if (v___y_1012_ == 0)
{
v___y_991_ = v___y_1010_;
v___y_992_ = v___y_1011_;
v___y_993_ = v_severity_885_;
goto v___jp_990_;
}
else
{
v___y_991_ = v___y_1010_;
v___y_992_ = v___y_1011_;
v___y_993_ = v___x_1008_;
goto v___jp_990_;
}
}
v___jp_1013_:
{
if (v___y_1014_ == 0)
{
lean_object* v___x_1015_; lean_object* v_scopes_1016_; lean_object* v___x_1017_; lean_object* v___x_1018_; lean_object* v_opts_1019_; uint8_t v___x_1020_; uint8_t v___x_1021_; 
v___x_1015_ = lean_st_ref_get(v___y_888_);
v_scopes_1016_ = lean_ctor_get(v___x_1015_, 2);
lean_inc(v_scopes_1016_);
lean_dec(v___x_1015_);
v___x_1017_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_1018_ = l_List_head_x21___redArg(v___x_1017_, v_scopes_1016_);
lean_dec(v_scopes_1016_);
v_opts_1019_ = lean_ctor_get(v___x_1018_, 1);
lean_inc_ref(v_opts_1019_);
lean_dec(v___x_1018_);
v___x_1020_ = 1;
v___x_1021_ = l_Lean_instBEqMessageSeverity_beq(v_severity_885_, v___x_1020_);
if (v___x_1021_ == 0)
{
lean_dec_ref(v_opts_1019_);
v___y_1010_ = v___y_1014_;
v___y_1011_ = v___y_1014_;
v___y_1012_ = v___x_1021_;
goto v___jp_1009_;
}
else
{
lean_object* v___x_1022_; uint8_t v___x_1023_; 
v___x_1022_ = l_Lean_warningAsError;
v___x_1023_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3_spec__7(v_opts_1019_, v___x_1022_);
lean_dec_ref(v_opts_1019_);
v___y_1010_ = v___y_1014_;
v___y_1011_ = v___y_1014_;
v___y_1012_ = v___x_1023_;
goto v___jp_1009_;
}
}
else
{
lean_object* v___x_1024_; lean_object* v___x_1025_; 
lean_dec_ref(v_msgData_884_);
v___x_1024_ = lean_box(0);
v___x_1025_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1025_, 0, v___x_1024_);
return v___x_1025_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3___boxed(lean_object* v_ref_1028_, lean_object* v_msgData_1029_, lean_object* v_severity_1030_, lean_object* v_isSilent_1031_, lean_object* v___y_1032_, lean_object* v___y_1033_, lean_object* v___y_1034_){
_start:
{
uint8_t v_severity_boxed_1035_; uint8_t v_isSilent_boxed_1036_; lean_object* v_res_1037_; 
v_severity_boxed_1035_ = lean_unbox(v_severity_1030_);
v_isSilent_boxed_1036_ = lean_unbox(v_isSilent_1031_);
v_res_1037_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3(v_ref_1028_, v_msgData_1029_, v_severity_boxed_1035_, v_isSilent_boxed_1036_, v___y_1032_, v___y_1033_);
lean_dec(v___y_1033_);
lean_dec_ref(v___y_1032_);
lean_dec(v_ref_1028_);
return v_res_1037_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2(lean_object* v_ref_1038_, lean_object* v_msgData_1039_, lean_object* v___y_1040_, lean_object* v___y_1041_){
_start:
{
uint8_t v___x_1043_; uint8_t v___x_1044_; lean_object* v___x_1045_; 
v___x_1043_ = 1;
v___x_1044_ = 0;
v___x_1045_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3(v_ref_1038_, v_msgData_1039_, v___x_1043_, v___x_1044_, v___y_1040_, v___y_1041_);
return v___x_1045_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2___boxed(lean_object* v_ref_1046_, lean_object* v_msgData_1047_, lean_object* v___y_1048_, lean_object* v___y_1049_, lean_object* v___y_1050_){
_start:
{
lean_object* v_res_1051_; 
v_res_1051_ = lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2(v_ref_1046_, v_msgData_1047_, v___y_1048_, v___y_1049_);
lean_dec(v___y_1049_);
lean_dec_ref(v___y_1048_);
lean_dec(v_ref_1046_);
return v_res_1051_;
}
}
static lean_object* _init_lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1___closed__1(void){
_start:
{
lean_object* v___x_1053_; lean_object* v___x_1054_; 
v___x_1053_ = ((lean_object*)(lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1___closed__0));
v___x_1054_ = l_Lean_stringToMessageData(v___x_1053_);
return v___x_1054_;
}
}
static lean_object* _init_lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1___closed__3(void){
_start:
{
lean_object* v___x_1056_; lean_object* v___x_1057_; 
v___x_1056_ = ((lean_object*)(lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1___closed__2));
v___x_1057_ = l_Lean_stringToMessageData(v___x_1056_);
return v___x_1057_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1(lean_object* v_linterOption_1058_, lean_object* v_stx_1059_, lean_object* v_msg_1060_, lean_object* v___y_1061_, lean_object* v___y_1062_){
_start:
{
lean_object* v_name_1064_; lean_object* v___x_1066_; uint8_t v_isShared_1067_; uint8_t v_isSharedCheck_1082_; 
v_name_1064_ = lean_ctor_get(v_linterOption_1058_, 0);
v_isSharedCheck_1082_ = !lean_is_exclusive(v_linterOption_1058_);
if (v_isSharedCheck_1082_ == 0)
{
lean_object* v_unused_1083_; 
v_unused_1083_ = lean_ctor_get(v_linterOption_1058_, 1);
lean_dec(v_unused_1083_);
v___x_1066_ = v_linterOption_1058_;
v_isShared_1067_ = v_isSharedCheck_1082_;
goto v_resetjp_1065_;
}
else
{
lean_inc(v_name_1064_);
lean_dec(v_linterOption_1058_);
v___x_1066_ = lean_box(0);
v_isShared_1067_ = v_isSharedCheck_1082_;
goto v_resetjp_1065_;
}
v_resetjp_1065_:
{
lean_object* v___x_1068_; lean_object* v___x_1069_; lean_object* v___x_1071_; 
v___x_1068_ = lean_obj_once(&lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1___closed__1, &lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1___closed__1_once, _init_lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1___closed__1);
lean_inc(v_name_1064_);
v___x_1069_ = l_Lean_MessageData_ofName(v_name_1064_);
if (v_isShared_1067_ == 0)
{
lean_ctor_set_tag(v___x_1066_, 7);
lean_ctor_set(v___x_1066_, 1, v___x_1069_);
lean_ctor_set(v___x_1066_, 0, v___x_1068_);
v___x_1071_ = v___x_1066_;
goto v_reusejp_1070_;
}
else
{
lean_object* v_reuseFailAlloc_1081_; 
v_reuseFailAlloc_1081_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1081_, 0, v___x_1068_);
lean_ctor_set(v_reuseFailAlloc_1081_, 1, v___x_1069_);
v___x_1071_ = v_reuseFailAlloc_1081_;
goto v_reusejp_1070_;
}
v_reusejp_1070_:
{
lean_object* v___x_1072_; lean_object* v___x_1073_; lean_object* v_disable_1074_; lean_object* v___x_1075_; lean_object* v___x_1076_; lean_object* v___x_1077_; lean_object* v___x_1078_; lean_object* v___x_1079_; lean_object* v___x_1080_; 
v___x_1072_ = lean_obj_once(&lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1___closed__3, &lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1___closed__3_once, _init_lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1___closed__3);
v___x_1073_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1073_, 0, v___x_1071_);
lean_ctor_set(v___x_1073_, 1, v___x_1072_);
v_disable_1074_ = l_Lean_MessageData_note(v___x_1073_);
v___x_1075_ = l_Lean_Linter_linterMessageTag;
v___x_1076_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1076_, 0, v_msg_1060_);
lean_ctor_set(v___x_1076_, 1, v_disable_1074_);
v___x_1077_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_1077_, 0, v___x_1075_);
lean_ctor_set(v___x_1077_, 1, v___x_1076_);
v___x_1078_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_1078_, 0, v_name_1064_);
lean_ctor_set(v___x_1078_, 1, v___x_1077_);
lean_inc(v_stx_1059_);
v___x_1079_ = lean_alloc_ctor(11, 2, 0);
lean_ctor_set(v___x_1079_, 0, v_stx_1059_);
lean_ctor_set(v___x_1079_, 1, v___x_1078_);
v___x_1080_ = lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2(v_stx_1059_, v___x_1079_, v___y_1061_, v___y_1062_);
lean_dec(v_stx_1059_);
return v___x_1080_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1___boxed(lean_object* v_linterOption_1084_, lean_object* v_stx_1085_, lean_object* v_msg_1086_, lean_object* v___y_1087_, lean_object* v___y_1088_, lean_object* v___y_1089_){
_start:
{
lean_object* v_res_1090_; 
v_res_1090_ = lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1(v_linterOption_1084_, v_stx_1085_, v_msg_1086_, v___y_1087_, v___y_1088_);
lean_dec(v___y_1088_);
lean_dec_ref(v___y_1087_);
return v_res_1090_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__0_spec__0___redArg(lean_object* v_o_1091_, lean_object* v___y_1092_){
_start:
{
lean_object* v___x_1094_; lean_object* v_env_1095_; lean_object* v___x_1096_; lean_object* v_toEnvExtension_1097_; lean_object* v_asyncMode_1098_; lean_object* v___x_1099_; lean_object* v___x_1100_; lean_object* v___x_1101_; lean_object* v_merged_1102_; lean_object* v___x_1104_; uint8_t v_isShared_1105_; uint8_t v_isSharedCheck_1110_; 
v___x_1094_ = lean_st_ref_get(v___y_1092_);
v_env_1095_ = lean_ctor_get(v___x_1094_, 0);
lean_inc_ref(v_env_1095_);
lean_dec(v___x_1094_);
v___x_1096_ = l_Lean_Linter_linterSetsExt;
v_toEnvExtension_1097_ = lean_ctor_get(v___x_1096_, 0);
v_asyncMode_1098_ = lean_ctor_get(v_toEnvExtension_1097_, 2);
v___x_1099_ = l_Lean_Linter_instInhabitedLinterSetsState_default;
v___x_1100_ = lean_box(0);
v___x_1101_ = l_Lean_PersistentEnvExtension_getState___redArg(v___x_1099_, v___x_1096_, v_env_1095_, v_asyncMode_1098_, v___x_1100_);
v_merged_1102_ = lean_ctor_get(v___x_1101_, 0);
v_isSharedCheck_1110_ = !lean_is_exclusive(v___x_1101_);
if (v_isSharedCheck_1110_ == 0)
{
lean_object* v_unused_1111_; 
v_unused_1111_ = lean_ctor_get(v___x_1101_, 1);
lean_dec(v_unused_1111_);
v___x_1104_ = v___x_1101_;
v_isShared_1105_ = v_isSharedCheck_1110_;
goto v_resetjp_1103_;
}
else
{
lean_inc(v_merged_1102_);
lean_dec(v___x_1101_);
v___x_1104_ = lean_box(0);
v_isShared_1105_ = v_isSharedCheck_1110_;
goto v_resetjp_1103_;
}
v_resetjp_1103_:
{
lean_object* v___x_1107_; 
if (v_isShared_1105_ == 0)
{
lean_ctor_set(v___x_1104_, 1, v_merged_1102_);
lean_ctor_set(v___x_1104_, 0, v_o_1091_);
v___x_1107_ = v___x_1104_;
goto v_reusejp_1106_;
}
else
{
lean_object* v_reuseFailAlloc_1109_; 
v_reuseFailAlloc_1109_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1109_, 0, v_o_1091_);
lean_ctor_set(v_reuseFailAlloc_1109_, 1, v_merged_1102_);
v___x_1107_ = v_reuseFailAlloc_1109_;
goto v_reusejp_1106_;
}
v_reusejp_1106_:
{
lean_object* v___x_1108_; 
v___x_1108_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1108_, 0, v___x_1107_);
return v___x_1108_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__0_spec__0___redArg___boxed(lean_object* v_o_1112_, lean_object* v___y_1113_, lean_object* v___y_1114_){
_start:
{
lean_object* v_res_1115_; 
v_res_1115_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__0_spec__0___redArg(v_o_1112_, v___y_1113_);
lean_dec(v___y_1113_);
return v_res_1115_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__0(lean_object* v___y_1116_, lean_object* v___y_1117_){
_start:
{
lean_object* v___x_1119_; lean_object* v_scopes_1120_; lean_object* v___x_1121_; lean_object* v___x_1122_; lean_object* v_opts_1123_; lean_object* v___x_1124_; 
v___x_1119_ = lean_st_ref_get(v___y_1117_);
v_scopes_1120_ = lean_ctor_get(v___x_1119_, 2);
lean_inc(v_scopes_1120_);
lean_dec(v___x_1119_);
v___x_1121_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_1122_ = l_List_head_x21___redArg(v___x_1121_, v_scopes_1120_);
lean_dec(v_scopes_1120_);
v_opts_1123_ = lean_ctor_get(v___x_1122_, 1);
lean_inc_ref(v_opts_1123_);
lean_dec(v___x_1122_);
v___x_1124_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__0_spec__0___redArg(v_opts_1123_, v___y_1117_);
return v___x_1124_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__0___boxed(lean_object* v___y_1125_, lean_object* v___y_1126_, lean_object* v___y_1127_){
_start:
{
lean_object* v_res_1128_; 
v_res_1128_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__0(v___y_1125_, v___y_1126_);
lean_dec(v___y_1126_);
lean_dec_ref(v___y_1125_);
return v_res_1128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__2(lean_object* v_linterOption_1129_, lean_object* v_stx_1130_, lean_object* v_msg_1131_, lean_object* v___y_1132_, lean_object* v___y_1133_){
_start:
{
lean_object* v___x_1135_; lean_object* v_a_1136_; lean_object* v___x_1138_; uint8_t v_isShared_1139_; uint8_t v_isSharedCheck_1146_; 
v___x_1135_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__0(v___y_1132_, v___y_1133_);
v_a_1136_ = lean_ctor_get(v___x_1135_, 0);
v_isSharedCheck_1146_ = !lean_is_exclusive(v___x_1135_);
if (v_isSharedCheck_1146_ == 0)
{
v___x_1138_ = v___x_1135_;
v_isShared_1139_ = v_isSharedCheck_1146_;
goto v_resetjp_1137_;
}
else
{
lean_inc(v_a_1136_);
lean_dec(v___x_1135_);
v___x_1138_ = lean_box(0);
v_isShared_1139_ = v_isSharedCheck_1146_;
goto v_resetjp_1137_;
}
v_resetjp_1137_:
{
uint8_t v___x_1140_; 
v___x_1140_ = l_Lean_Linter_getLinterValue(v_linterOption_1129_, v_a_1136_);
lean_dec(v_a_1136_);
if (v___x_1140_ == 0)
{
lean_object* v___x_1141_; lean_object* v___x_1143_; 
lean_dec_ref(v_msg_1131_);
lean_dec(v_stx_1130_);
lean_dec_ref(v_linterOption_1129_);
v___x_1141_ = lean_box(0);
if (v_isShared_1139_ == 0)
{
lean_ctor_set(v___x_1138_, 0, v___x_1141_);
v___x_1143_ = v___x_1138_;
goto v_reusejp_1142_;
}
else
{
lean_object* v_reuseFailAlloc_1144_; 
v_reuseFailAlloc_1144_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1144_, 0, v___x_1141_);
v___x_1143_ = v_reuseFailAlloc_1144_;
goto v_reusejp_1142_;
}
v_reusejp_1142_:
{
return v___x_1143_;
}
}
else
{
lean_object* v___x_1145_; 
lean_del_object(v___x_1138_);
v___x_1145_ = lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1(v_linterOption_1129_, v_stx_1130_, v_msg_1131_, v___y_1132_, v___y_1133_);
return v___x_1145_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__2___boxed(lean_object* v_linterOption_1147_, lean_object* v_stx_1148_, lean_object* v_msg_1149_, lean_object* v___y_1150_, lean_object* v___y_1151_, lean_object* v___y_1152_){
_start:
{
lean_object* v_res_1153_; 
v_res_1153_ = lp_mathlib_Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__2(v_linterOption_1147_, v_stx_1148_, v_msg_1149_, v___y_1150_, v___y_1151_);
lean_dec(v___y_1151_);
lean_dec_ref(v___y_1150_);
return v_res_1153_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__3(lean_object* v_as_1154_, size_t v_sz_1155_, size_t v_i_1156_, lean_object* v_b_1157_, lean_object* v___y_1158_, lean_object* v___y_1159_){
_start:
{
lean_object* v_a_1162_; uint8_t v___x_1166_; 
v___x_1166_ = lean_usize_dec_lt(v_i_1156_, v_sz_1155_);
if (v___x_1166_ == 0)
{
lean_object* v___x_1167_; 
v___x_1167_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1167_, 0, v_b_1157_);
return v___x_1167_;
}
else
{
lean_object* v_a_1168_; lean_object* v_snd_1169_; lean_object* v_fst_1170_; lean_object* v_fst_1171_; lean_object* v_snd_1172_; lean_object* v___x_1173_; lean_object* v___y_1175_; lean_object* v___y_1176_; 
v_a_1168_ = lean_array_uget_borrowed(v_as_1154_, v_i_1156_);
v_snd_1169_ = lean_ctor_get(v_a_1168_, 1);
v_fst_1170_ = lean_ctor_get(v_a_1168_, 0);
v_fst_1171_ = lean_ctor_get(v_snd_1169_, 0);
v_snd_1172_ = lean_ctor_get(v_snd_1169_, 1);
v___x_1173_ = lean_box(0);
if (lean_obj_tag(v_fst_1170_) == 1)
{
lean_object* v_pre_1193_; 
v_pre_1193_ = lean_ctor_get(v_fst_1170_, 0);
switch(lean_obj_tag(v_pre_1193_))
{
case 1:
{
lean_object* v_pre_1194_; 
v_pre_1194_ = lean_ctor_get(v_pre_1193_, 0);
if (lean_obj_tag(v_pre_1194_) == 1)
{
lean_object* v_pre_1195_; 
v_pre_1195_ = lean_ctor_get(v_pre_1194_, 0);
switch(lean_obj_tag(v_pre_1195_))
{
case 1:
{
lean_object* v_pre_1196_; 
v_pre_1196_ = lean_ctor_get(v_pre_1195_, 0);
if (lean_obj_tag(v_pre_1196_) == 0)
{
lean_object* v_str_1197_; lean_object* v_str_1198_; lean_object* v_str_1199_; lean_object* v_str_1200_; lean_object* v___x_1201_; uint8_t v___x_1202_; 
v_str_1197_ = lean_ctor_get(v_fst_1170_, 1);
v_str_1198_ = lean_ctor_get(v_pre_1193_, 1);
v_str_1199_ = lean_ctor_get(v_pre_1194_, 1);
v_str_1200_ = lean_ctor_get(v_pre_1195_, 1);
v___x_1201_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__0));
v___x_1202_ = lean_string_dec_eq(v_str_1200_, v___x_1201_);
if (v___x_1202_ == 0)
{
v_a_1162_ = v___x_1173_;
goto v___jp_1161_;
}
else
{
lean_object* v___x_1203_; uint8_t v___x_1204_; 
v___x_1203_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getSetOptionMaxHeartbeatsComment___closed__1));
v___x_1204_ = lean_string_dec_eq(v_str_1199_, v___x_1203_);
if (v___x_1204_ == 0)
{
v_a_1162_ = v___x_1173_;
goto v___jp_1161_;
}
else
{
lean_object* v___x_1205_; uint8_t v___x_1206_; 
v___x_1205_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__2));
v___x_1206_ = lean_string_dec_eq(v_str_1198_, v___x_1205_);
if (v___x_1206_ == 0)
{
lean_object* v___x_1207_; uint8_t v___x_1208_; 
v___x_1207_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig_spec__0___closed__0));
v___x_1208_ = lean_string_dec_eq(v_str_1198_, v___x_1207_);
if (v___x_1208_ == 0)
{
v_a_1162_ = v___x_1173_;
goto v___jp_1161_;
}
else
{
lean_object* v___x_1209_; uint8_t v___x_1210_; 
v___x_1209_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__0));
v___x_1210_ = lean_string_dec_eq(v_str_1197_, v___x_1209_);
if (v___x_1210_ == 0)
{
v_a_1162_ = v___x_1173_;
goto v___jp_1161_;
}
else
{
v___y_1175_ = v___y_1158_;
v___y_1176_ = v___y_1159_;
goto v___jp_1174_;
}
}
}
else
{
lean_object* v___x_1211_; uint8_t v___x_1212_; 
v___x_1211_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__10));
v___x_1212_ = lean_string_dec_eq(v_str_1197_, v___x_1211_);
if (v___x_1212_ == 0)
{
lean_object* v___x_1213_; uint8_t v___x_1214_; 
v___x_1213_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__11));
v___x_1214_ = lean_string_dec_eq(v_str_1197_, v___x_1213_);
if (v___x_1214_ == 0)
{
lean_object* v___x_1215_; uint8_t v___x_1216_; 
v___x_1215_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2098674011____hygCtx___hyg_4_));
v___x_1216_ = lean_string_dec_eq(v_str_1197_, v___x_1215_);
if (v___x_1216_ == 0)
{
lean_object* v___x_1217_; uint8_t v___x_1218_; 
v___x_1217_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__0));
v___x_1218_ = lean_string_dec_eq(v_str_1197_, v___x_1217_);
if (v___x_1218_ == 0)
{
v_a_1162_ = v___x_1173_;
goto v___jp_1161_;
}
else
{
v___y_1175_ = v___y_1158_;
v___y_1176_ = v___y_1159_;
goto v___jp_1174_;
}
}
else
{
v___y_1175_ = v___y_1158_;
v___y_1176_ = v___y_1159_;
goto v___jp_1174_;
}
}
else
{
lean_object* v___x_1219_; lean_object* v___x_1220_; 
v___x_1219_ = lp_mathlib_Mathlib_Linter_Style_linter_style_admit;
lean_inc(v_snd_1172_);
lean_inc(v_fst_1171_);
v___x_1220_ = lp_mathlib_Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__2(v___x_1219_, v_fst_1171_, v_snd_1172_, v___y_1158_, v___y_1159_);
if (lean_obj_tag(v___x_1220_) == 0)
{
lean_dec_ref_known(v___x_1220_, 1);
v_a_1162_ = v___x_1173_;
goto v___jp_1161_;
}
else
{
return v___x_1220_;
}
}
}
else
{
lean_object* v___x_1221_; lean_object* v___x_1222_; 
v___x_1221_ = lp_mathlib_Mathlib_Linter_Style_linter_style_refine;
lean_inc(v_snd_1172_);
lean_inc(v_fst_1171_);
v___x_1222_ = lp_mathlib_Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__2(v___x_1221_, v_fst_1171_, v_snd_1172_, v___y_1158_, v___y_1159_);
if (lean_obj_tag(v___x_1222_) == 0)
{
lean_dec_ref_known(v___x_1222_, 1);
v_a_1162_ = v___x_1173_;
goto v___jp_1161_;
}
else
{
return v___x_1222_;
}
}
}
}
}
}
else
{
v_a_1162_ = v___x_1173_;
goto v___jp_1161_;
}
}
case 0:
{
lean_object* v_str_1223_; lean_object* v_str_1224_; lean_object* v_str_1225_; lean_object* v___x_1226_; uint8_t v___x_1227_; 
v_str_1223_ = lean_ctor_get(v_fst_1170_, 1);
v_str_1224_ = lean_ctor_get(v_pre_1193_, 1);
v_str_1225_ = lean_ctor_get(v_pre_1194_, 1);
v___x_1226_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4_));
v___x_1227_ = lean_string_dec_eq(v_str_1225_, v___x_1226_);
if (v___x_1227_ == 0)
{
v_a_1162_ = v___x_1173_;
goto v___jp_1161_;
}
else
{
lean_object* v___x_1228_; uint8_t v___x_1229_; 
v___x_1228_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_usesNativeConfig___closed__2));
v___x_1229_ = lean_string_dec_eq(v_str_1224_, v___x_1228_);
if (v___x_1229_ == 0)
{
v_a_1162_ = v___x_1173_;
goto v___jp_1161_;
}
else
{
lean_object* v___x_1230_; uint8_t v___x_1231_; 
v___x_1230_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__20));
v___x_1231_ = lean_string_dec_eq(v_str_1223_, v___x_1230_);
if (v___x_1231_ == 0)
{
lean_object* v___x_1232_; uint8_t v___x_1233_; 
v___x_1232_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax___closed__21));
v___x_1233_ = lean_string_dec_eq(v_str_1223_, v___x_1232_);
if (v___x_1233_ == 0)
{
v_a_1162_ = v___x_1173_;
goto v___jp_1161_;
}
else
{
lean_object* v___x_1234_; lean_object* v___x_1235_; 
v___x_1234_ = lp_mathlib_Mathlib_Linter_Style_linter_style_induction;
lean_inc(v_snd_1172_);
lean_inc(v_fst_1171_);
v___x_1235_ = lp_mathlib_Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__2(v___x_1234_, v_fst_1171_, v_snd_1172_, v___y_1158_, v___y_1159_);
if (lean_obj_tag(v___x_1235_) == 0)
{
lean_dec_ref_known(v___x_1235_, 1);
v_a_1162_ = v___x_1173_;
goto v___jp_1161_;
}
else
{
return v___x_1235_;
}
}
}
else
{
lean_object* v___x_1236_; lean_object* v___x_1237_; 
v___x_1236_ = lp_mathlib_Mathlib_Linter_Style_linter_style_cases;
lean_inc(v_snd_1172_);
lean_inc(v_fst_1171_);
v___x_1237_ = lp_mathlib_Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__2(v___x_1236_, v_fst_1171_, v_snd_1172_, v___y_1158_, v___y_1159_);
if (lean_obj_tag(v___x_1237_) == 0)
{
lean_dec_ref_known(v___x_1237_, 1);
v_a_1162_ = v___x_1173_;
goto v___jp_1161_;
}
else
{
return v___x_1237_;
}
}
}
}
}
default: 
{
v_a_1162_ = v___x_1173_;
goto v___jp_1161_;
}
}
}
else
{
v_a_1162_ = v___x_1173_;
goto v___jp_1161_;
}
}
case 0:
{
lean_object* v_str_1238_; lean_object* v___x_1239_; uint8_t v___x_1240_; 
v_str_1238_ = lean_ctor_get(v_fst_1170_, 1);
v___x_1239_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax_spec__1___closed__0));
v___x_1240_ = lean_string_dec_eq(v_str_1238_, v___x_1239_);
if (v___x_1240_ == 0)
{
v_a_1162_ = v___x_1173_;
goto v___jp_1161_;
}
else
{
lean_object* v___x_1241_; lean_object* v___x_1242_; 
v___x_1241_ = lp_mathlib_Mathlib_Linter_Style_linter_style_maxHeartbeats;
lean_inc(v_snd_1172_);
lean_inc(v_fst_1171_);
v___x_1242_ = lp_mathlib_Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__2(v___x_1241_, v_fst_1171_, v_snd_1172_, v___y_1158_, v___y_1159_);
if (lean_obj_tag(v___x_1242_) == 0)
{
lean_dec_ref_known(v___x_1242_, 1);
v_a_1162_ = v___x_1173_;
goto v___jp_1161_;
}
else
{
return v___x_1242_;
}
}
}
default: 
{
v_a_1162_ = v___x_1173_;
goto v___jp_1161_;
}
}
}
else
{
v_a_1162_ = v___x_1173_;
goto v___jp_1161_;
}
v___jp_1174_:
{
lean_object* v___x_1177_; 
v___x_1177_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__0(v___y_1175_, v___y_1176_);
if (lean_obj_tag(v___x_1177_) == 0)
{
lean_object* v_a_1178_; lean_object* v___x_1179_; uint8_t v___x_1180_; 
v_a_1178_ = lean_ctor_get(v___x_1177_, 0);
lean_inc(v_a_1178_);
lean_dec_ref_known(v___x_1177_, 1);
v___x_1179_ = lp_mathlib_Mathlib_Linter_Style_linter_style_native;
v___x_1180_ = l_Lean_Linter_getLinterValue(v___x_1179_, v_a_1178_);
if (v___x_1180_ == 0)
{
lean_object* v___x_1181_; uint8_t v___x_1182_; 
v___x_1181_ = lp_mathlib_Mathlib_Linter_Style_linter_style_nativeDecide;
v___x_1182_ = l_Lean_Linter_getLinterValue(v___x_1181_, v_a_1178_);
lean_dec(v_a_1178_);
if (v___x_1182_ == 0)
{
v_a_1162_ = v___x_1173_;
goto v___jp_1161_;
}
else
{
lean_object* v___x_1183_; 
lean_inc(v_snd_1172_);
lean_inc(v_fst_1171_);
v___x_1183_ = lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1(v___x_1181_, v_fst_1171_, v_snd_1172_, v___y_1175_, v___y_1176_);
if (lean_obj_tag(v___x_1183_) == 0)
{
lean_dec_ref_known(v___x_1183_, 1);
v_a_1162_ = v___x_1173_;
goto v___jp_1161_;
}
else
{
return v___x_1183_;
}
}
}
else
{
lean_object* v___x_1184_; 
lean_dec(v_a_1178_);
lean_inc(v_snd_1172_);
lean_inc(v_fst_1171_);
v___x_1184_ = lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1(v___x_1179_, v_fst_1171_, v_snd_1172_, v___y_1175_, v___y_1176_);
if (lean_obj_tag(v___x_1184_) == 0)
{
lean_dec_ref_known(v___x_1184_, 1);
v_a_1162_ = v___x_1173_;
goto v___jp_1161_;
}
else
{
return v___x_1184_;
}
}
}
else
{
lean_object* v_a_1185_; lean_object* v___x_1187_; uint8_t v_isShared_1188_; uint8_t v_isSharedCheck_1192_; 
v_a_1185_ = lean_ctor_get(v___x_1177_, 0);
v_isSharedCheck_1192_ = !lean_is_exclusive(v___x_1177_);
if (v_isSharedCheck_1192_ == 0)
{
v___x_1187_ = v___x_1177_;
v_isShared_1188_ = v_isSharedCheck_1192_;
goto v_resetjp_1186_;
}
else
{
lean_inc(v_a_1185_);
lean_dec(v___x_1177_);
v___x_1187_ = lean_box(0);
v_isShared_1188_ = v_isSharedCheck_1192_;
goto v_resetjp_1186_;
}
v_resetjp_1186_:
{
lean_object* v___x_1190_; 
if (v_isShared_1188_ == 0)
{
v___x_1190_ = v___x_1187_;
goto v_reusejp_1189_;
}
else
{
lean_object* v_reuseFailAlloc_1191_; 
v_reuseFailAlloc_1191_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1191_, 0, v_a_1185_);
v___x_1190_ = v_reuseFailAlloc_1191_;
goto v_reusejp_1189_;
}
v_reusejp_1189_:
{
return v___x_1190_;
}
}
}
}
}
v___jp_1161_:
{
size_t v___x_1163_; size_t v___x_1164_; 
v___x_1163_ = ((size_t)1ULL);
v___x_1164_ = lean_usize_add(v_i_1156_, v___x_1163_);
v_i_1156_ = v___x_1164_;
v_b_1157_ = v_a_1162_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__3___boxed(lean_object* v_as_1243_, lean_object* v_sz_1244_, lean_object* v_i_1245_, lean_object* v_b_1246_, lean_object* v___y_1247_, lean_object* v___y_1248_, lean_object* v___y_1249_){
_start:
{
size_t v_sz_boxed_1250_; size_t v_i_boxed_1251_; lean_object* v_res_1252_; 
v_sz_boxed_1250_ = lean_unbox_usize(v_sz_1244_);
lean_dec(v_sz_1244_);
v_i_boxed_1251_ = lean_unbox_usize(v_i_1245_);
lean_dec(v_i_1245_);
v_res_1252_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__3(v_as_1243_, v_sz_boxed_1250_, v_i_boxed_1251_, v_b_1246_, v___y_1247_, v___y_1248_);
lean_dec(v___y_1248_);
lean_dec_ref(v___y_1247_);
lean_dec_ref(v_as_1243_);
return v_res_1252_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___lam__0(lean_object* v___x_1253_, lean_object* v_x_1254_, lean_object* v___y_1255_, lean_object* v___y_1256_){
_start:
{
lean_object* v___x_1258_; size_t v_sz_1259_; size_t v___x_1260_; lean_object* v___x_1261_; 
v___x_1258_ = lean_box(0);
v_sz_1259_ = lean_array_size(v___x_1253_);
v___x_1260_ = ((size_t)0ULL);
v___x_1261_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__3(v___x_1253_, v_sz_1259_, v___x_1260_, v___x_1258_, v___y_1255_, v___y_1256_);
if (lean_obj_tag(v___x_1261_) == 0)
{
lean_object* v___x_1263_; uint8_t v_isShared_1264_; uint8_t v_isSharedCheck_1268_; 
v_isSharedCheck_1268_ = !lean_is_exclusive(v___x_1261_);
if (v_isSharedCheck_1268_ == 0)
{
lean_object* v_unused_1269_; 
v_unused_1269_ = lean_ctor_get(v___x_1261_, 0);
lean_dec(v_unused_1269_);
v___x_1263_ = v___x_1261_;
v_isShared_1264_ = v_isSharedCheck_1268_;
goto v_resetjp_1262_;
}
else
{
lean_dec(v___x_1261_);
v___x_1263_ = lean_box(0);
v_isShared_1264_ = v_isSharedCheck_1268_;
goto v_resetjp_1262_;
}
v_resetjp_1262_:
{
lean_object* v___x_1266_; 
if (v_isShared_1264_ == 0)
{
lean_ctor_set(v___x_1263_, 0, v___x_1258_);
v___x_1266_ = v___x_1263_;
goto v_reusejp_1265_;
}
else
{
lean_object* v_reuseFailAlloc_1267_; 
v_reuseFailAlloc_1267_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1267_, 0, v___x_1258_);
v___x_1266_ = v_reuseFailAlloc_1267_;
goto v_reusejp_1265_;
}
v_reusejp_1265_:
{
return v___x_1266_;
}
}
}
else
{
return v___x_1261_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___lam__0___boxed(lean_object* v___x_1270_, lean_object* v_x_1271_, lean_object* v___y_1272_, lean_object* v___y_1273_, lean_object* v___y_1274_){
_start:
{
lean_object* v_res_1275_; 
v_res_1275_ = lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___lam__0(v___x_1270_, v_x_1271_, v___y_1272_, v___y_1273_);
lean_dec(v___y_1273_);
lean_dec_ref(v___y_1272_);
lean_dec(v_x_1271_);
lean_dec_ref(v___x_1270_);
return v_res_1275_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___lam__1(lean_object* v_stx_1276_, lean_object* v___y_1277_, lean_object* v___y_1278_){
_start:
{
lean_object* v___x_1280_; lean_object* v_a_1281_; lean_object* v___x_1282_; lean_object* v_a_1283_; lean_object* v___x_1284_; lean_object* v_a_1285_; lean_object* v___x_1286_; lean_object* v_a_1287_; lean_object* v___x_1288_; lean_object* v_a_1289_; lean_object* v___x_1290_; lean_object* v_a_1291_; lean_object* v___x_1293_; uint8_t v_isShared_1294_; uint8_t v_isSharedCheck_1332_; 
v___x_1280_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__0(v___y_1277_, v___y_1278_);
v_a_1281_ = lean_ctor_get(v___x_1280_, 0);
lean_inc(v_a_1281_);
lean_dec_ref(v___x_1280_);
v___x_1282_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__0(v___y_1277_, v___y_1278_);
v_a_1283_ = lean_ctor_get(v___x_1282_, 0);
lean_inc(v_a_1283_);
lean_dec_ref(v___x_1282_);
v___x_1284_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__0(v___y_1277_, v___y_1278_);
v_a_1285_ = lean_ctor_get(v___x_1284_, 0);
lean_inc(v_a_1285_);
lean_dec_ref(v___x_1284_);
v___x_1286_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__0(v___y_1277_, v___y_1278_);
v_a_1287_ = lean_ctor_get(v___x_1286_, 0);
lean_inc(v_a_1287_);
lean_dec_ref(v___x_1286_);
v___x_1288_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__0(v___y_1277_, v___y_1278_);
v_a_1289_ = lean_ctor_get(v___x_1288_, 0);
lean_inc(v_a_1289_);
lean_dec_ref(v___x_1288_);
v___x_1290_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__0(v___y_1277_, v___y_1278_);
v_a_1291_ = lean_ctor_get(v___x_1290_, 0);
v_isSharedCheck_1332_ = !lean_is_exclusive(v___x_1290_);
if (v_isSharedCheck_1332_ == 0)
{
v___x_1293_ = v___x_1290_;
v_isShared_1294_ = v_isSharedCheck_1332_;
goto v_resetjp_1292_;
}
else
{
lean_inc(v_a_1291_);
lean_dec(v___x_1290_);
v___x_1293_ = lean_box(0);
v_isShared_1294_ = v_isSharedCheck_1332_;
goto v_resetjp_1292_;
}
v_resetjp_1292_:
{
lean_object* v___x_1295_; lean_object* v_a_1296_; lean_object* v___x_1298_; uint8_t v_isShared_1299_; uint8_t v_isSharedCheck_1331_; 
v___x_1295_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__0(v___y_1277_, v___y_1278_);
v_a_1296_ = lean_ctor_get(v___x_1295_, 0);
v_isSharedCheck_1331_ = !lean_is_exclusive(v___x_1295_);
if (v_isSharedCheck_1331_ == 0)
{
v___x_1298_ = v___x_1295_;
v_isShared_1299_ = v_isSharedCheck_1331_;
goto v_resetjp_1297_;
}
else
{
lean_inc(v_a_1296_);
lean_dec(v___x_1295_);
v___x_1298_ = lean_box(0);
v_isShared_1299_ = v_isSharedCheck_1331_;
goto v_resetjp_1297_;
}
v_resetjp_1297_:
{
uint8_t v___y_1312_; lean_object* v___x_1327_; uint8_t v___x_1328_; 
v___x_1327_ = lp_mathlib_Mathlib_Linter_Style_linter_style_refine;
v___x_1328_ = l_Lean_Linter_getLinterValue(v___x_1327_, v_a_1281_);
lean_dec(v_a_1281_);
if (v___x_1328_ == 0)
{
lean_object* v___x_1329_; uint8_t v___x_1330_; 
v___x_1329_ = lp_mathlib_Mathlib_Linter_Style_linter_style_cases;
v___x_1330_ = l_Lean_Linter_getLinterValue(v___x_1329_, v_a_1283_);
lean_dec(v_a_1283_);
v___y_1312_ = v___x_1330_;
goto v___jp_1311_;
}
else
{
lean_dec(v_a_1283_);
v___y_1312_ = v___x_1328_;
goto v___jp_1311_;
}
v___jp_1300_:
{
lean_object* v___x_1301_; lean_object* v_messages_1302_; uint8_t v___x_1303_; 
v___x_1301_ = lean_st_ref_get(v___y_1278_);
v_messages_1302_ = lean_ctor_get(v___x_1301_, 1);
lean_inc_ref(v_messages_1302_);
lean_dec(v___x_1301_);
v___x_1303_ = l_Lean_MessageLog_hasErrors(v_messages_1302_);
lean_dec_ref(v_messages_1302_);
if (v___x_1303_ == 0)
{
lean_object* v___x_1304_; lean_object* v___f_1305_; lean_object* v___x_1306_; 
lean_del_object(v___x_1298_);
lean_inc(v_stx_1276_);
v___x_1304_ = lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_getDeprecatedSyntax(v_stx_1276_);
v___f_1305_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___lam__0___boxed), 5, 1);
lean_closure_set(v___f_1305_, 0, v___x_1304_);
v___x_1306_ = l_Lean_withSetOptionIn___redArg(v___f_1305_, v_stx_1276_, v___y_1277_, v___y_1278_);
return v___x_1306_;
}
else
{
lean_object* v___x_1307_; lean_object* v___x_1309_; 
lean_dec(v_stx_1276_);
v___x_1307_ = lean_box(0);
if (v_isShared_1299_ == 0)
{
lean_ctor_set(v___x_1298_, 0, v___x_1307_);
v___x_1309_ = v___x_1298_;
goto v_reusejp_1308_;
}
else
{
lean_object* v_reuseFailAlloc_1310_; 
v_reuseFailAlloc_1310_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1310_, 0, v___x_1307_);
v___x_1309_ = v_reuseFailAlloc_1310_;
goto v_reusejp_1308_;
}
v_reusejp_1308_:
{
return v___x_1309_;
}
}
}
v___jp_1311_:
{
if (v___y_1312_ == 0)
{
lean_object* v___x_1313_; uint8_t v___x_1314_; 
v___x_1313_ = lp_mathlib_Mathlib_Linter_Style_linter_style_induction;
v___x_1314_ = l_Lean_Linter_getLinterValue(v___x_1313_, v_a_1285_);
lean_dec(v_a_1285_);
if (v___x_1314_ == 0)
{
lean_object* v___x_1315_; uint8_t v___x_1316_; 
v___x_1315_ = lp_mathlib_Mathlib_Linter_Style_linter_style_admit;
v___x_1316_ = l_Lean_Linter_getLinterValue(v___x_1315_, v_a_1287_);
lean_dec(v_a_1287_);
if (v___x_1316_ == 0)
{
lean_object* v___x_1317_; uint8_t v___x_1318_; 
v___x_1317_ = lp_mathlib_Mathlib_Linter_Style_linter_style_maxHeartbeats;
v___x_1318_ = l_Lean_Linter_getLinterValue(v___x_1317_, v_a_1289_);
lean_dec(v_a_1289_);
if (v___x_1318_ == 0)
{
lean_object* v___x_1319_; uint8_t v___x_1320_; 
v___x_1319_ = lp_mathlib_Mathlib_Linter_Style_linter_style_native;
v___x_1320_ = l_Lean_Linter_getLinterValue(v___x_1319_, v_a_1291_);
lean_dec(v_a_1291_);
if (v___x_1320_ == 0)
{
lean_object* v___x_1321_; uint8_t v___x_1322_; 
v___x_1321_ = lp_mathlib_Mathlib_Linter_Style_linter_style_nativeDecide;
v___x_1322_ = l_Lean_Linter_getLinterValue(v___x_1321_, v_a_1296_);
lean_dec(v_a_1296_);
if (v___x_1322_ == 0)
{
lean_object* v___x_1323_; lean_object* v___x_1325_; 
lean_del_object(v___x_1298_);
lean_dec(v_stx_1276_);
v___x_1323_ = lean_box(0);
if (v_isShared_1294_ == 0)
{
lean_ctor_set(v___x_1293_, 0, v___x_1323_);
v___x_1325_ = v___x_1293_;
goto v_reusejp_1324_;
}
else
{
lean_object* v_reuseFailAlloc_1326_; 
v_reuseFailAlloc_1326_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1326_, 0, v___x_1323_);
v___x_1325_ = v_reuseFailAlloc_1326_;
goto v_reusejp_1324_;
}
v_reusejp_1324_:
{
return v___x_1325_;
}
}
else
{
lean_del_object(v___x_1293_);
goto v___jp_1300_;
}
}
else
{
lean_dec(v_a_1296_);
lean_del_object(v___x_1293_);
goto v___jp_1300_;
}
}
else
{
lean_dec(v_a_1296_);
lean_del_object(v___x_1293_);
lean_dec(v_a_1291_);
goto v___jp_1300_;
}
}
else
{
lean_dec(v_a_1296_);
lean_del_object(v___x_1293_);
lean_dec(v_a_1291_);
lean_dec(v_a_1289_);
goto v___jp_1300_;
}
}
else
{
lean_dec(v_a_1296_);
lean_del_object(v___x_1293_);
lean_dec(v_a_1291_);
lean_dec(v_a_1289_);
lean_dec(v_a_1287_);
goto v___jp_1300_;
}
}
else
{
lean_dec(v_a_1296_);
lean_del_object(v___x_1293_);
lean_dec(v_a_1291_);
lean_dec(v_a_1289_);
lean_dec(v_a_1287_);
lean_dec(v_a_1285_);
goto v___jp_1300_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___lam__1___boxed(lean_object* v_stx_1333_, lean_object* v___y_1334_, lean_object* v___y_1335_, lean_object* v___y_1336_){
_start:
{
lean_object* v_res_1337_; 
v_res_1337_ = lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter___lam__1(v_stx_1333_, v___y_1334_, v___y_1335_);
lean_dec(v___y_1335_);
lean_dec_ref(v___y_1334_);
return v_res_1337_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__0_spec__0(lean_object* v_o_1376_, lean_object* v___y_1377_, lean_object* v___y_1378_){
_start:
{
lean_object* v___x_1380_; 
v___x_1380_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__0_spec__0___redArg(v_o_1376_, v___y_1378_);
return v___x_1380_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__0_spec__0___boxed(lean_object* v_o_1381_, lean_object* v___y_1382_, lean_object* v___y_1383_, lean_object* v___y_1384_){
_start:
{
lean_object* v_res_1385_; 
v_res_1385_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__0_spec__0(v_o_1381_, v___y_1382_, v___y_1383_);
lean_dec(v___y_1383_);
lean_dec_ref(v___y_1382_);
return v_res_1385_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3_spec__6(lean_object* v_msgData_1386_, lean_object* v___y_1387_, lean_object* v___y_1388_){
_start:
{
lean_object* v___x_1390_; 
v___x_1390_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3_spec__6___redArg(v_msgData_1386_, v___y_1388_);
return v___x_1390_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3_spec__6___boxed(lean_object* v_msgData_1391_, lean_object* v___y_1392_, lean_object* v___y_1393_, lean_object* v___y_1394_){
_start:
{
lean_object* v_res_1395_; 
v_res_1395_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter_spec__1_spec__2_spec__3_spec__6(v_msgData_1391_, v___y_1392_, v___y_1393_);
lean_dec(v___y_1393_);
lean_dec_ref(v___y_1392_);
return v_res_1395_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2374602524____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_1397_; lean_object* v___x_1398_; 
v___x_1397_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_deprecatedSyntaxLinter));
v___x_1398_ = l_Lean_Elab_Command_addLinter(v___x_1397_);
return v___x_1398_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2374602524____hygCtx___hyg_2____boxed(lean_object* v_a_1399_){
_start:
{
lean_object* v_res_1400_; 
v_res_1400_ = lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2374602524____hygCtx___hyg_2_();
return v_res_1400_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Parser_Command(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Parser_Command(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Command(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_Header(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Command(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_Header(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_854069892____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Linter_Style_linter_style_refine = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Linter_Style_linter_style_refine);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_278512193____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Linter_Style_linter_style_cases = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Linter_Style_linter_style_cases);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_3986038175____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Linter_Style_linter_style_induction = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Linter_Style_linter_style_induction);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_947237925____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Linter_Style_linter_style_admit = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Linter_Style_linter_style_admit);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2792787194____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Linter_Style_linter_style_native = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Linter_Style_linter_style_native);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2098674011____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Linter_Style_linter_style_nativeDecide = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Linter_Style_linter_style_nativeDecide);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_303104340____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Linter_Style_linter_style_maxHeartbeats = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Linter_Style_linter_style_maxHeartbeats);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter_2374602524____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Command(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_Header(uint8_t builtin);
lean_object* initialize_Lean_Parser_Command(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Command(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linter_Header(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Parser_Command(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter(builtin);
}
#ifdef __cplusplus
}
#endif
