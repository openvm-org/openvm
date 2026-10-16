// Lean compiler output
// Module: Mathlib.Tactic.Linter.EmptyLine
// Imports: public import Init public meta import Init public meta import Mathlib.Tactic.Linter.Header public import Lean.Parser.Command
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
lean_object* l_Id_instMonad___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
extern lean_object* l_Lean_Elab_Command_instInhabitedScope_default;
lean_object* l_List_head_x21___redArg(lean_object*, lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_String_Slice_posLE(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint32_t lean_string_utf8_get_fast(lean_object*, lean_object*);
uint8_t lean_uint32_dec_eq(uint32_t, uint32_t);
uint8_t l_Lean_Syntax_instBEqRange_beq(lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_get_size(lean_object*);
uint64_t l_Lean_Syntax_instHashableRange_hash(lean_object*);
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* l_Lean_Environment_header(lean_object*);
lean_object* l_Id_instMonad___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* l_Lean_MessageData_note(lean_object*);
extern lean_object* l_Lean_Linter_linterMessageTag;
lean_object* l_Lean_Elab_Command_getScope___redArg(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Lean_MessageLog_add(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(lean_object*);
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
lean_object* l_Lean_Syntax_getRange_x3f(lean_object*, uint8_t);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_register_option(lean_object*, lean_object*);
extern lean_object* l_Lean_Linter_linterSetsExt;
extern lean_object* l_Lean_Linter_instInhabitedLinterSetsState_default;
lean_object* l_Lean_PersistentEnvExtension_getState___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_string_utf8_extract_fast(lean_object*, lean_object*, lean_object*);
lean_object* lean_string_utf8_byte_size(lean_object*);
lean_object* lean_string_utf8_next_fast(lean_object*, lean_object*);
lean_object* lean_string_length(lean_object*);
lean_object* l_List_replicateTR___redArg(lean_object*, lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_ofRange(lean_object*, uint8_t);
lean_object* lean_string_push(lean_object*, uint32_t);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_Slice_Pattern_ForwardSliceSearcher_buildTable(lean_object*);
lean_object* l_Lean_Name_hash___override___boxed(lean_object*);
lean_object* l_Lean_Syntax_getKind(lean_object*);
uint8_t l_Lean_Linter_getLinterValue(lean_object*, lean_object*);
lean_object* l_Lean_MessageLog_reportedPlusUnreported(lean_object*);
lean_object* l_Lean_Syntax_getTrailing_x3f(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* lean_string_utf8_next(lean_object*, lean_object*);
uint32_t lean_string_utf8_get(lean_object*, lean_object*);
lean_object* lean_string_utf8_prev(lean_object*, lean_object*);
lean_object* lean_string_utf8_extract(lean_object*, lean_object*, lean_object*);
uint8_t lean_string_get_byte_fast(lean_object*, lean_object*);
uint8_t lean_uint8_dec_eq(uint8_t, uint8_t);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* l_String_Slice_posGE___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getAtomVal(lean_object*);
lean_object* l_Lean_Syntax_unsetTrailing(lean_object*);
lean_object* l_Lean_Syntax_getSubstring_x3f(lean_object*, uint8_t, uint8_t);
lean_object* l_String_splitOnAux(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* l_Lean_Name_components(lean_object*);
lean_object* l_Lean_withSetOptionIn___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_beq___boxed(lean_object*, lean_object*);
lean_object* l_Std_DHashMap_Internal_Raw_u2080_insertIfNew___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__6(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__2___boxed(lean_object*, lean_object*);
lean_object* l_Array_append___redArg___boxed(lean_object*, lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
lean_object* l_Lean_Elab_Command_addLinter(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Substring_Raw_getRange(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Substring_Raw_getRange___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_filterMapM___redArg___lam__1(lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Lean_Syntax_filterMapM___redArg___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_Syntax_filterMapM___redArg___lam__2___closed__0 = (const lean_object*)&lp_mathlib_Lean_Syntax_filterMapM___redArg___lam__2___closed__0_value;
static const lean_closure_object lp_mathlib_Lean_Syntax_filterMapM___redArg___lam__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Syntax_filterMapM___redArg___lam__2___closed__1 = (const lean_object*)&lp_mathlib_Lean_Syntax_filterMapM___redArg___lam__2___closed__1_value;
static const lean_closure_object lp_mathlib_Lean_Syntax_filterMapM___redArg___lam__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__1___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Syntax_filterMapM___redArg___lam__2___closed__2 = (const lean_object*)&lp_mathlib_Lean_Syntax_filterMapM___redArg___lam__2___closed__2_value;
static const lean_closure_object lp_mathlib_Lean_Syntax_filterMapM___redArg___lam__2___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__2___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Syntax_filterMapM___redArg___lam__2___closed__3 = (const lean_object*)&lp_mathlib_Lean_Syntax_filterMapM___redArg___lam__2___closed__3_value;
static const lean_closure_object lp_mathlib_Lean_Syntax_filterMapM___redArg___lam__2___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__3, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Syntax_filterMapM___redArg___lam__2___closed__4 = (const lean_object*)&lp_mathlib_Lean_Syntax_filterMapM___redArg___lam__2___closed__4_value;
static const lean_closure_object lp_mathlib_Lean_Syntax_filterMapM___redArg___lam__2___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__4___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Syntax_filterMapM___redArg___lam__2___closed__5 = (const lean_object*)&lp_mathlib_Lean_Syntax_filterMapM___redArg___lam__2___closed__5_value;
static const lean_closure_object lp_mathlib_Lean_Syntax_filterMapM___redArg___lam__2___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__5___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Syntax_filterMapM___redArg___lam__2___closed__6 = (const lean_object*)&lp_mathlib_Lean_Syntax_filterMapM___redArg___lam__2___closed__6_value;
static const lean_closure_object lp_mathlib_Lean_Syntax_filterMapM___redArg___lam__2___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__6, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Syntax_filterMapM___redArg___lam__2___closed__7 = (const lean_object*)&lp_mathlib_Lean_Syntax_filterMapM___redArg___lam__2___closed__7_value;
static const lean_ctor_object lp_mathlib_Lean_Syntax_filterMapM___redArg___lam__2___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Syntax_filterMapM___redArg___lam__2___closed__1_value),((lean_object*)&lp_mathlib_Lean_Syntax_filterMapM___redArg___lam__2___closed__2_value)}};
static const lean_object* lp_mathlib_Lean_Syntax_filterMapM___redArg___lam__2___closed__8 = (const lean_object*)&lp_mathlib_Lean_Syntax_filterMapM___redArg___lam__2___closed__8_value;
static const lean_ctor_object lp_mathlib_Lean_Syntax_filterMapM___redArg___lam__2___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Syntax_filterMapM___redArg___lam__2___closed__8_value),((lean_object*)&lp_mathlib_Lean_Syntax_filterMapM___redArg___lam__2___closed__3_value),((lean_object*)&lp_mathlib_Lean_Syntax_filterMapM___redArg___lam__2___closed__4_value),((lean_object*)&lp_mathlib_Lean_Syntax_filterMapM___redArg___lam__2___closed__5_value),((lean_object*)&lp_mathlib_Lean_Syntax_filterMapM___redArg___lam__2___closed__6_value)}};
static const lean_object* lp_mathlib_Lean_Syntax_filterMapM___redArg___lam__2___closed__9 = (const lean_object*)&lp_mathlib_Lean_Syntax_filterMapM___redArg___lam__2___closed__9_value;
static const lean_ctor_object lp_mathlib_Lean_Syntax_filterMapM___redArg___lam__2___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Syntax_filterMapM___redArg___lam__2___closed__9_value),((lean_object*)&lp_mathlib_Lean_Syntax_filterMapM___redArg___lam__2___closed__7_value)}};
static const lean_object* lp_mathlib_Lean_Syntax_filterMapM___redArg___lam__2___closed__10 = (const lean_object*)&lp_mathlib_Lean_Syntax_filterMapM___redArg___lam__2___closed__10_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_filterMapM___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Lean_Syntax_filterMapM___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Array_append___redArg___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Syntax_filterMapM___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_Syntax_filterMapM___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_filterMapM___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_filterMapM___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_filterMapM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_filterMap___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Syntax_filterMapM___at___00Lean_Syntax_filterMap_spec__0_spec__1___redArg(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Syntax_filterMapM___at___00Lean_Syntax_filterMap_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_filterMapM___at___00Lean_Syntax_filterMap_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Syntax_filterMapM___at___00Lean_Syntax_filterMap_spec__0_spec__0___redArg(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Syntax_filterMapM___at___00Lean_Syntax_filterMap_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_filterMap___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_filterMap(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_filterMapM___at___00Lean_Syntax_filterMap_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Syntax_filterMapM___at___00Lean_Syntax_filterMap_spec__0_spec__0(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Syntax_filterMapM___at___00Lean_Syntax_filterMap_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Syntax_filterMapM___at___00Lean_Syntax_filterMap_spec__0_spec__1(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Syntax_filterMapM___at___00Lean_Syntax_filterMap_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_filter___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_filter(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4__spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "style"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "emptyLine"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(186, 218, 113, 226, 101, 176, 32, 79)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(105, 62, 218, 153, 100, 142, 29, 251)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(37, 110, 105, 252, 240, 181, 112, 5)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 28, .m_capacity = 28, .m_length = 27, .m_data = "enable the emptyLine linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(120, 131, 127, 204, 79, 169, 80, 92)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(101, 237, 90, 120, 51, 59, 46, 172)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4__value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(98, 189, 128, 85, 154, 50, 252, 160)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4__value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(98, 143, 180, 149, 182, 23, 123, 111)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_linter_style_emptyLine;
static const lean_closure_object lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Name_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__0_value;
static const lean_closure_object lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Name_hash___override___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__2;
static lean_once_cell_t lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__3;
static const lean_string_object lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Command"};
static const lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "docComment"};
static const lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__8_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__6_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__8_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__7_value),LEAN_SCALAR_PTR_LITERAL(44, 76, 179, 33, 27, 4, 201, 125)}};
static const lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__8_value;
static lean_once_cell_t lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__9;
static const lean_string_object lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "moduleDoc"};
static const lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__11_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__11_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__11_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__6_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__11_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__10_value),LEAN_SCALAR_PTR_LITERAL(249, 71, 187, 113, 90, 175, 60, 199)}};
static const lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__11_value;
static lean_once_cell_t lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__12;
static const lean_string_object lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "mutual"};
static const lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__14_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__14_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__14_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__14_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__6_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__14_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__13_value),LEAN_SCALAR_PTR_LITERAL(55, 205, 8, 5, 164, 77, 17, 1)}};
static const lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__14_value;
static lean_once_cell_t lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__15;
static const lean_string_object lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "str"};
static const lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__16_value),LEAN_SCALAR_PTR_LITERAL(255, 188, 142, 1, 190, 33, 34, 128)}};
static const lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__17_value;
static lean_once_cell_t lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__18;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines;
static const lean_string_object lp_mathlib_Mathlib_Linter_EmptyLine_SkippedFileSegments___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_SkippedFileSegments___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_SkippedFileSegments___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_EmptyLine_SkippedFileSegments___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_SkippedFileSegments___closed__0_value),LEAN_SCALAR_PTR_LITERAL(186, 205, 46, 93, 234, 75, 44, 75)}};
static const lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_SkippedFileSegments___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_SkippedFileSegments___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Linter_EmptyLine_SkippedFileSegments___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_SkippedFileSegments___closed__2;
static const lean_string_object lp_mathlib_Mathlib_Linter_EmptyLine_SkippedFileSegments___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Util"};
static const lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_SkippedFileSegments___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_SkippedFileSegments___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_EmptyLine_SkippedFileSegments___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_SkippedFileSegments___closed__3_value),LEAN_SCALAR_PTR_LITERAL(97, 52, 201, 108, 45, 150, 112, 31)}};
static const lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_SkippedFileSegments___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_SkippedFileSegments___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Linter_EmptyLine_SkippedFileSegments___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_SkippedFileSegments___closed__5;
static const lean_string_object lp_mathlib_Mathlib_Linter_EmptyLine_SkippedFileSegments___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Meta"};
static const lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_SkippedFileSegments___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_SkippedFileSegments___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_EmptyLine_SkippedFileSegments___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_SkippedFileSegments___closed__6_value),LEAN_SCALAR_PTR_LITERAL(211, 174, 49, 251, 64, 24, 251, 1)}};
static const lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_SkippedFileSegments___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_SkippedFileSegments___closed__7_value;
static lean_once_cell_t lp_mathlib_Mathlib_Linter_EmptyLine_SkippedFileSegments___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_SkippedFileSegments___closed__8;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_SkippedFileSegments;
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__2___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__2___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__6_spec__10___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__6_spec__10___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__7___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__7___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__6_spec__11_spec__13_spec__27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__6_spec__11_spec__13___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__6_spec__11___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__6___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__0___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__0___closed__1;
static lean_once_cell_t lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__0___closed__2;
static lean_once_cell_t lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__0___closed__3;
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Substring_Raw_takeRightWhileAux___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Substring_Raw_takeRightWhileAux___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__4___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Substring_Raw_takeWhileAux___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Substring_Raw_takeWhileAux___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5_spec__8___redArg(lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5_spec__8___redArg___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "\n\n"};
static const lean_object* lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5___closed__0 = (const lean_object*)&lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5___closed__0_value;
static lean_once_cell_t lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5___closed__1;
static lean_once_cell_t lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static uint8_t lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5___closed__2;
static lean_once_cell_t lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5___closed__3;
static lean_once_cell_t lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5___closed__4;
static lean_once_cell_t lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5___closed__5;
static const lean_ctor_object lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5___closed__6 = (const lean_object*)&lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5___closed__6_value;
LEAN_EXPORT uint8_t lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5___boxed(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "structSimpleBinder"};
static const lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__1___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__1___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__1___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__6_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__1___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(24, 230, 214, 182, 254, 52, 213, 225)}};
static const lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "where"};
static const lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__1___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__1___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "structInstField"};
static const lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__1___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__1___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__1___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__1___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__1___closed__5_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__1___closed__5_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__1___closed__5_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(50, 77, 20, 88, 28, 210, 230, 84)}};
static const lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__1___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__1(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_Pos_revSkipWhile___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__8(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_Pos_revSkipWhile___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__8___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Lean_PersistentArray_anyM___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__1_spec__3(uint8_t, lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Lean_PersistentArray_anyM___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentArray_anyMAux___at___00Lean_PersistentArray_anyM___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__1_spec__2(uint8_t, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Lean_PersistentArray_anyMAux___at___00Lean_PersistentArray_anyM___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__1_spec__2_spec__4(uint8_t, lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Lean_PersistentArray_anyMAux___at___00Lean_PersistentArray_anyM___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__1_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_anyMAux___at___00Lean_PersistentArray_anyM___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__1_spec__2___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentArray_anyM___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__1(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_anyM___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__14(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__14___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29_spec__37(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29_spec__37___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29___lam__0___closed__0_value;
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29___lam__0___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29_spec__36___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29_spec__36___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29_spec__36___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29_spec__36___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29_spec__36___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29_spec__36___redArg___closed__2;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29_spec__36___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29_spec__36___redArg___closed__3;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29_spec__36___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29_spec__36___redArg___closed__4;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29_spec__36___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29_spec__36___redArg___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29_spec__36___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29_spec__36___redArg___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 46, .m_capacity = 46, .m_length = 45, .m_data = "This linter can be disabled with `set_option "};
static const lean_object* lp_mathlib_Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15___closed__0 = (const lean_object*)&lp_mathlib_Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15___closed__1;
static const lean_string_object lp_mathlib_Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = " false`"};
static const lean_object* lp_mathlib_Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15___closed__2 = (const lean_object*)&lp_mathlib_Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__16___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__16___closed__0 = (const lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__16___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__16(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__16___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__17(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__17___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__18___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = " "};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__18___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__18___closed__0_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__18___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 105, .m_capacity = 105, .m_length = 104, .m_data = "Please, write a comment here or remove this line, but do not place empty lines within commands!\nContext:"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__18___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__18___closed__1_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__18___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__18___closed__2;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__18___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "⏎"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__18___closed__3 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__18___closed__3_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__18___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 2, .m_data = "⏎⏎"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__18___closed__4 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__18___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__18(uint8_t, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__18___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_Pos_revSkipWhile___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__10(uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_Pos_revSkipWhile___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__10___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_Pos_skipWhile___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__9(uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_Pos_skipWhile___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__9___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__11___redArg(uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__11___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__13_spec__20_spec__23___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__13_spec__20_spec__23___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__13_spec__20_spec__24_spec__32_spec__35___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__13_spec__20_spec__24_spec__32___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__13_spec__20_spec__24___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__13_spec__20___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__13_spec__21(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__13_spec__21___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__13(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__13___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_List_find_x3f___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__19___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_find_x3f___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__19___closed__0;
static lean_once_cell_t lp_mathlib_List_find_x3f___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__19___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_find_x3f___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__19___closed__1;
static lean_once_cell_t lp_mathlib_List_find_x3f___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__19___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_find_x3f___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__19___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_List_find_x3f___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__19(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_find_x3f___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__19___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__12_spec__18(uint8_t, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__12_spec__18___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Array_filterMapM___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__12___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Array_filterMapM___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__12___closed__0 = (const lean_object*)&lp_mathlib_Array_filterMapM___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__12___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Array_filterMapM___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__12(uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_filterMapM___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__12___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__2___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__2___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__2___closed__1;
static lean_once_cell_t lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__2___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__2___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___closed__0_value;
static const lean_closure_object lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__2___boxed, .m_arity = 5, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___closed__0_value)} };
static const lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___closed__1_value;
static const lean_closure_object lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_withSetOptionIn___boxed, .m_arity = 6, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___closed__1_value)} };
static const lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "EmptyLine"};
static const lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "emptyLineLinter"};
static const lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___closed__5_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(120, 131, 127, 204, 79, 169, 80, 92)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___closed__5_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___closed__5_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___closed__3_value),LEAN_SCALAR_PTR_LITERAL(50, 168, 190, 222, 179, 169, 140, 36)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___closed__5_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___closed__4_value),LEAN_SCALAR_PTR_LITERAL(90, 219, 173, 224, 111, 113, 63, 118)}};
static const lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___closed__5_value)}};
static const lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___closed__6_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter = (const lean_object*)&lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__6(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__7(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__7___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__11(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__6_spec__10(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__6_spec__10___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__6_spec__11(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__13_spec__20(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__6_spec__11_spec__13(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__13_spec__20_spec__23(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__13_spec__20_spec__23___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__13_spec__20_spec__24(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29_spec__36(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29_spec__36___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__6_spec__11_spec__13_spec__27(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__13_spec__20_spec__24_spec__32(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__13_spec__20_spec__24_spec__32_spec__35(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_EmptyLine_initFn_00___x40_Mathlib_Tactic_Linter_EmptyLine_1134681162____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_EmptyLine_initFn_00___x40_Mathlib_Tactic_Linter_EmptyLine_1134681162____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Substring_Raw_getRange(lean_object* v_x_1_){
_start:
{
lean_object* v_startPos_2_; lean_object* v_stopPos_3_; lean_object* v___x_4_; 
v_startPos_2_ = lean_ctor_get(v_x_1_, 1);
v_stopPos_3_ = lean_ctor_get(v_x_1_, 2);
lean_inc(v_stopPos_3_);
lean_inc(v_startPos_2_);
v___x_4_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4_, 0, v_startPos_2_);
lean_ctor_set(v___x_4_, 1, v_stopPos_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Substring_Raw_getRange___boxed(lean_object* v_x_5_){
_start:
{
lean_object* v_res_6_; 
v_res_6_ = lp_mathlib_Lean_Substring_Raw_getRange(v_x_5_);
lean_dec_ref(v_x_5_);
return v_res_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_filterMapM___redArg___lam__1(lean_object* v_toPure_7_, lean_object* v___y_8_, lean_object* v_____do__lift_9_){
_start:
{
if (lean_obj_tag(v_____do__lift_9_) == 0)
{
lean_object* v___x_10_; 
v___x_10_ = lean_apply_2(v_toPure_7_, lean_box(0), v___y_8_);
return v___x_10_;
}
else
{
lean_object* v_val_11_; lean_object* v___x_12_; lean_object* v___x_13_; 
v_val_11_ = lean_ctor_get(v_____do__lift_9_, 0);
lean_inc(v_val_11_);
lean_dec_ref_known(v_____do__lift_9_, 1);
v___x_12_ = lean_array_push(v___y_8_, v_val_11_);
v___x_13_ = lean_apply_2(v_toPure_7_, lean_box(0), v___x_12_);
return v___x_13_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_filterMapM___redArg___lam__2(lean_object* v_toPure_35_, lean_object* v_f_36_, lean_object* v_stx_37_, lean_object* v_toBind_38_, lean_object* v___f_39_, lean_object* v_____do__lift_40_){
_start:
{
lean_object* v___y_42_; lean_object* v___x_46_; lean_object* v___x_47_; lean_object* v___x_48_; lean_object* v___x_49_; uint8_t v___x_50_; 
v___x_46_ = lean_unsigned_to_nat(0u);
v___x_47_ = ((lean_object*)(lp_mathlib_Lean_Syntax_filterMapM___redArg___lam__2___closed__0));
v___x_48_ = lean_array_get_size(v_____do__lift_40_);
v___x_49_ = ((lean_object*)(lp_mathlib_Lean_Syntax_filterMapM___redArg___lam__2___closed__10));
v___x_50_ = lean_nat_dec_lt(v___x_46_, v___x_48_);
if (v___x_50_ == 0)
{
lean_dec_ref(v_____do__lift_40_);
lean_dec_ref(v___f_39_);
v___y_42_ = v___x_47_;
goto v___jp_41_;
}
else
{
uint8_t v___x_51_; 
v___x_51_ = lean_nat_dec_le(v___x_48_, v___x_48_);
if (v___x_51_ == 0)
{
if (v___x_50_ == 0)
{
lean_dec_ref(v_____do__lift_40_);
lean_dec_ref(v___f_39_);
v___y_42_ = v___x_47_;
goto v___jp_41_;
}
else
{
size_t v___x_52_; size_t v___x_53_; lean_object* v___x_54_; 
v___x_52_ = ((size_t)0ULL);
v___x_53_ = lean_usize_of_nat(v___x_48_);
v___x_54_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_49_, v___f_39_, v_____do__lift_40_, v___x_52_, v___x_53_, v___x_47_);
v___y_42_ = v___x_54_;
goto v___jp_41_;
}
}
else
{
size_t v___x_55_; size_t v___x_56_; lean_object* v___x_57_; 
v___x_55_ = ((size_t)0ULL);
v___x_56_ = lean_usize_of_nat(v___x_48_);
v___x_57_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_49_, v___f_39_, v_____do__lift_40_, v___x_55_, v___x_56_, v___x_47_);
v___y_42_ = v___x_57_;
goto v___jp_41_;
}
}
v___jp_41_:
{
lean_object* v___f_43_; lean_object* v___x_44_; lean_object* v___x_45_; 
v___f_43_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Syntax_filterMapM___redArg___lam__1), 3, 2);
lean_closure_set(v___f_43_, 0, v_toPure_35_);
lean_closure_set(v___f_43_, 1, v___y_42_);
v___x_44_ = lean_apply_1(v_f_36_, v_stx_37_);
v___x_45_ = lean_apply_4(v_toBind_38_, lean_box(0), lean_box(0), v___x_44_, v___f_43_);
return v___x_45_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_filterMapM___redArg(lean_object* v_inst_59_, lean_object* v_stx_60_, lean_object* v_f_61_){
_start:
{
lean_object* v_toApplicative_62_; lean_object* v_toBind_63_; lean_object* v_toPure_64_; lean_object* v___f_65_; lean_object* v___f_66_; lean_object* v___f_67_; lean_object* v___x_68_; size_t v_sz_69_; size_t v___x_70_; lean_object* v___x_71_; lean_object* v___x_72_; 
v_toApplicative_62_ = lean_ctor_get(v_inst_59_, 0);
v_toBind_63_ = lean_ctor_get(v_inst_59_, 1);
lean_inc_n(v_toBind_63_, 2);
v_toPure_64_ = lean_ctor_get(v_toApplicative_62_, 1);
lean_inc(v_f_61_);
lean_inc_ref(v_inst_59_);
v___f_65_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Syntax_filterMapM___redArg___lam__0), 3, 2);
lean_closure_set(v___f_65_, 0, v_inst_59_);
lean_closure_set(v___f_65_, 1, v_f_61_);
v___f_66_ = ((lean_object*)(lp_mathlib_Lean_Syntax_filterMapM___redArg___closed__0));
lean_inc(v_stx_60_);
lean_inc(v_toPure_64_);
v___f_67_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Syntax_filterMapM___redArg___lam__2), 6, 5);
lean_closure_set(v___f_67_, 0, v_toPure_64_);
lean_closure_set(v___f_67_, 1, v_f_61_);
lean_closure_set(v___f_67_, 2, v_stx_60_);
lean_closure_set(v___f_67_, 3, v_toBind_63_);
lean_closure_set(v___f_67_, 4, v___f_66_);
v___x_68_ = l_Lean_Syntax_getArgs(v_stx_60_);
lean_dec(v_stx_60_);
v_sz_69_ = lean_array_size(v___x_68_);
v___x_70_ = ((size_t)0ULL);
v___x_71_ = l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_box(0), lean_box(0), lean_box(0), v_inst_59_, v___f_65_, v_sz_69_, v___x_70_, v___x_68_);
v___x_72_ = lean_apply_4(v_toBind_63_, lean_box(0), lean_box(0), v___x_71_, v___f_67_);
return v___x_72_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_filterMapM___redArg___lam__0(lean_object* v_inst_73_, lean_object* v_f_74_, lean_object* v_x_75_){
_start:
{
lean_object* v___x_76_; 
v___x_76_ = lp_mathlib_Lean_Syntax_filterMapM___redArg(v_inst_73_, v_x_75_, v_f_74_);
return v___x_76_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_filterMapM(lean_object* v_m_77_, lean_object* v_inst_78_, lean_object* v_00_u03b1_79_, lean_object* v_stx_80_, lean_object* v_f_81_){
_start:
{
lean_object* v___x_82_; 
v___x_82_ = lp_mathlib_Lean_Syntax_filterMapM___redArg(v_inst_78_, v_stx_80_, v_f_81_);
return v___x_82_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_filterMap___redArg___lam__0(lean_object* v_f_83_, lean_object* v_x_84_){
_start:
{
lean_object* v___x_85_; 
v___x_85_ = lean_apply_1(v_f_83_, v_x_84_);
return v___x_85_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Syntax_filterMapM___at___00Lean_Syntax_filterMap_spec__0_spec__1___redArg(lean_object* v_as_86_, size_t v_i_87_, size_t v_stop_88_, lean_object* v_b_89_){
_start:
{
uint8_t v___x_90_; 
v___x_90_ = lean_usize_dec_eq(v_i_87_, v_stop_88_);
if (v___x_90_ == 0)
{
lean_object* v___x_91_; lean_object* v___x_92_; size_t v___x_93_; size_t v___x_94_; 
v___x_91_ = lean_array_uget_borrowed(v_as_86_, v_i_87_);
v___x_92_ = l_Array_append___redArg(v_b_89_, v___x_91_);
v___x_93_ = ((size_t)1ULL);
v___x_94_ = lean_usize_add(v_i_87_, v___x_93_);
v_i_87_ = v___x_94_;
v_b_89_ = v___x_92_;
goto _start;
}
else
{
return v_b_89_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Syntax_filterMapM___at___00Lean_Syntax_filterMap_spec__0_spec__1___redArg___boxed(lean_object* v_as_96_, lean_object* v_i_97_, lean_object* v_stop_98_, lean_object* v_b_99_){
_start:
{
size_t v_i_boxed_100_; size_t v_stop_boxed_101_; lean_object* v_res_102_; 
v_i_boxed_100_ = lean_unbox_usize(v_i_97_);
lean_dec(v_i_97_);
v_stop_boxed_101_ = lean_unbox_usize(v_stop_98_);
lean_dec(v_stop_98_);
v_res_102_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Syntax_filterMapM___at___00Lean_Syntax_filterMap_spec__0_spec__1___redArg(v_as_96_, v_i_boxed_100_, v_stop_boxed_101_, v_b_99_);
lean_dec_ref(v_as_96_);
return v_res_102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_filterMapM___at___00Lean_Syntax_filterMap_spec__0___redArg(lean_object* v_stx_103_, lean_object* v_f_104_){
_start:
{
lean_object* v___y_106_; lean_object* v___x_110_; size_t v_sz_111_; size_t v___x_112_; lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; lean_object* v___x_116_; uint8_t v___x_117_; 
v___x_110_ = l_Lean_Syntax_getArgs(v_stx_103_);
v_sz_111_ = lean_array_size(v___x_110_);
v___x_112_ = ((size_t)0ULL);
lean_inc_ref(v_f_104_);
v___x_113_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Syntax_filterMapM___at___00Lean_Syntax_filterMap_spec__0_spec__0___redArg(v_f_104_, v_sz_111_, v___x_112_, v___x_110_);
v___x_114_ = lean_unsigned_to_nat(0u);
v___x_115_ = ((lean_object*)(lp_mathlib_Lean_Syntax_filterMapM___redArg___lam__2___closed__0));
v___x_116_ = lean_array_get_size(v___x_113_);
v___x_117_ = lean_nat_dec_lt(v___x_114_, v___x_116_);
if (v___x_117_ == 0)
{
lean_dec_ref(v___x_113_);
v___y_106_ = v___x_115_;
goto v___jp_105_;
}
else
{
uint8_t v___x_118_; 
v___x_118_ = lean_nat_dec_le(v___x_116_, v___x_116_);
if (v___x_118_ == 0)
{
if (v___x_117_ == 0)
{
lean_dec_ref(v___x_113_);
v___y_106_ = v___x_115_;
goto v___jp_105_;
}
else
{
size_t v___x_119_; lean_object* v___x_120_; 
v___x_119_ = lean_usize_of_nat(v___x_116_);
v___x_120_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Syntax_filterMapM___at___00Lean_Syntax_filterMap_spec__0_spec__1___redArg(v___x_113_, v___x_112_, v___x_119_, v___x_115_);
lean_dec_ref(v___x_113_);
v___y_106_ = v___x_120_;
goto v___jp_105_;
}
}
else
{
size_t v___x_121_; lean_object* v___x_122_; 
v___x_121_ = lean_usize_of_nat(v___x_116_);
v___x_122_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Syntax_filterMapM___at___00Lean_Syntax_filterMap_spec__0_spec__1___redArg(v___x_113_, v___x_112_, v___x_121_, v___x_115_);
lean_dec_ref(v___x_113_);
v___y_106_ = v___x_122_;
goto v___jp_105_;
}
}
v___jp_105_:
{
lean_object* v___x_107_; 
v___x_107_ = lean_apply_1(v_f_104_, v_stx_103_);
if (lean_obj_tag(v___x_107_) == 0)
{
return v___y_106_;
}
else
{
lean_object* v_val_108_; lean_object* v___x_109_; 
v_val_108_ = lean_ctor_get(v___x_107_, 0);
lean_inc(v_val_108_);
lean_dec_ref_known(v___x_107_, 1);
v___x_109_ = lean_array_push(v___y_106_, v_val_108_);
return v___x_109_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Syntax_filterMapM___at___00Lean_Syntax_filterMap_spec__0_spec__0___redArg(lean_object* v_f_123_, size_t v_sz_124_, size_t v_i_125_, lean_object* v_bs_126_){
_start:
{
uint8_t v___x_127_; 
v___x_127_ = lean_usize_dec_lt(v_i_125_, v_sz_124_);
if (v___x_127_ == 0)
{
lean_dec_ref(v_f_123_);
return v_bs_126_;
}
else
{
lean_object* v_v_128_; lean_object* v___x_129_; lean_object* v_bs_x27_130_; lean_object* v___x_131_; size_t v___x_132_; size_t v___x_133_; lean_object* v___x_134_; 
v_v_128_ = lean_array_uget(v_bs_126_, v_i_125_);
v___x_129_ = lean_unsigned_to_nat(0u);
v_bs_x27_130_ = lean_array_uset(v_bs_126_, v_i_125_, v___x_129_);
lean_inc_ref(v_f_123_);
v___x_131_ = lp_mathlib_Lean_Syntax_filterMapM___at___00Lean_Syntax_filterMap_spec__0___redArg(v_v_128_, v_f_123_);
v___x_132_ = ((size_t)1ULL);
v___x_133_ = lean_usize_add(v_i_125_, v___x_132_);
v___x_134_ = lean_array_uset(v_bs_x27_130_, v_i_125_, v___x_131_);
v_i_125_ = v___x_133_;
v_bs_126_ = v___x_134_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Syntax_filterMapM___at___00Lean_Syntax_filterMap_spec__0_spec__0___redArg___boxed(lean_object* v_f_136_, lean_object* v_sz_137_, lean_object* v_i_138_, lean_object* v_bs_139_){
_start:
{
size_t v_sz_boxed_140_; size_t v_i_boxed_141_; lean_object* v_res_142_; 
v_sz_boxed_140_ = lean_unbox_usize(v_sz_137_);
lean_dec(v_sz_137_);
v_i_boxed_141_ = lean_unbox_usize(v_i_138_);
lean_dec(v_i_138_);
v_res_142_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Syntax_filterMapM___at___00Lean_Syntax_filterMap_spec__0_spec__0___redArg(v_f_136_, v_sz_boxed_140_, v_i_boxed_141_, v_bs_139_);
return v_res_142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_filterMap___redArg(lean_object* v_stx_143_, lean_object* v_f_144_){
_start:
{
lean_object* v___f_145_; lean_object* v___x_146_; 
v___f_145_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Syntax_filterMap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_145_, 0, v_f_144_);
v___x_146_ = lp_mathlib_Lean_Syntax_filterMapM___at___00Lean_Syntax_filterMap_spec__0___redArg(v_stx_143_, v___f_145_);
return v___x_146_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_filterMap(lean_object* v_00_u03b1_147_, lean_object* v_stx_148_, lean_object* v_f_149_){
_start:
{
lean_object* v___x_150_; 
v___x_150_ = lp_mathlib_Lean_Syntax_filterMap___redArg(v_stx_148_, v_f_149_);
return v___x_150_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_filterMapM___at___00Lean_Syntax_filterMap_spec__0(lean_object* v_00_u03b1_151_, lean_object* v_stx_152_, lean_object* v_f_153_){
_start:
{
lean_object* v___x_154_; 
v___x_154_ = lp_mathlib_Lean_Syntax_filterMapM___at___00Lean_Syntax_filterMap_spec__0___redArg(v_stx_152_, v_f_153_);
return v___x_154_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Syntax_filterMapM___at___00Lean_Syntax_filterMap_spec__0_spec__0(lean_object* v_00_u03b1_155_, lean_object* v_f_156_, size_t v_sz_157_, size_t v_i_158_, lean_object* v_bs_159_){
_start:
{
lean_object* v___x_160_; 
v___x_160_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Syntax_filterMapM___at___00Lean_Syntax_filterMap_spec__0_spec__0___redArg(v_f_156_, v_sz_157_, v_i_158_, v_bs_159_);
return v___x_160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Syntax_filterMapM___at___00Lean_Syntax_filterMap_spec__0_spec__0___boxed(lean_object* v_00_u03b1_161_, lean_object* v_f_162_, lean_object* v_sz_163_, lean_object* v_i_164_, lean_object* v_bs_165_){
_start:
{
size_t v_sz_boxed_166_; size_t v_i_boxed_167_; lean_object* v_res_168_; 
v_sz_boxed_166_ = lean_unbox_usize(v_sz_163_);
lean_dec(v_sz_163_);
v_i_boxed_167_ = lean_unbox_usize(v_i_164_);
lean_dec(v_i_164_);
v_res_168_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Syntax_filterMapM___at___00Lean_Syntax_filterMap_spec__0_spec__0(v_00_u03b1_161_, v_f_162_, v_sz_boxed_166_, v_i_boxed_167_, v_bs_165_);
return v_res_168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Syntax_filterMapM___at___00Lean_Syntax_filterMap_spec__0_spec__1(lean_object* v_00_u03b1_169_, lean_object* v_as_170_, size_t v_i_171_, size_t v_stop_172_, lean_object* v_b_173_){
_start:
{
lean_object* v___x_174_; 
v___x_174_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Syntax_filterMapM___at___00Lean_Syntax_filterMap_spec__0_spec__1___redArg(v_as_170_, v_i_171_, v_stop_172_, v_b_173_);
return v___x_174_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Syntax_filterMapM___at___00Lean_Syntax_filterMap_spec__0_spec__1___boxed(lean_object* v_00_u03b1_175_, lean_object* v_as_176_, lean_object* v_i_177_, lean_object* v_stop_178_, lean_object* v_b_179_){
_start:
{
size_t v_i_boxed_180_; size_t v_stop_boxed_181_; lean_object* v_res_182_; 
v_i_boxed_180_ = lean_unbox_usize(v_i_177_);
lean_dec(v_i_177_);
v_stop_boxed_181_ = lean_unbox_usize(v_stop_178_);
lean_dec(v_stop_178_);
v_res_182_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Syntax_filterMapM___at___00Lean_Syntax_filterMap_spec__0_spec__1(v_00_u03b1_175_, v_as_176_, v_i_boxed_180_, v_stop_boxed_181_, v_b_179_);
lean_dec_ref(v_as_176_);
return v_res_182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_filter___lam__0(lean_object* v_f_183_, lean_object* v_s_184_){
_start:
{
lean_object* v___x_185_; uint8_t v___x_186_; 
lean_inc(v_s_184_);
v___x_185_ = lean_apply_1(v_f_183_, v_s_184_);
v___x_186_ = lean_unbox(v___x_185_);
if (v___x_186_ == 0)
{
lean_object* v___x_187_; 
lean_dec(v_s_184_);
v___x_187_ = lean_box(0);
return v___x_187_;
}
else
{
lean_object* v___x_188_; 
v___x_188_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_188_, 0, v_s_184_);
return v___x_188_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_filter(lean_object* v_stx_189_, lean_object* v_f_190_){
_start:
{
lean_object* v___f_191_; lean_object* v___x_192_; 
v___f_191_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Syntax_filter___lam__0), 2, 1);
lean_closure_set(v___f_191_, 0, v_f_190_);
v___x_192_ = lp_mathlib_Lean_Syntax_filterMap___redArg(v_stx_189_, v___f_191_);
return v___x_192_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4__spec__0(lean_object* v_name_193_, lean_object* v_decl_194_, lean_object* v_ref_195_){
_start:
{
lean_object* v_defValue_197_; lean_object* v_descr_198_; lean_object* v_deprecation_x3f_199_; lean_object* v___x_200_; uint8_t v___x_201_; lean_object* v___x_202_; lean_object* v___x_203_; 
v_defValue_197_ = lean_ctor_get(v_decl_194_, 0);
v_descr_198_ = lean_ctor_get(v_decl_194_, 1);
v_deprecation_x3f_199_ = lean_ctor_get(v_decl_194_, 2);
v___x_200_ = lean_alloc_ctor(1, 0, 1);
v___x_201_ = lean_unbox(v_defValue_197_);
lean_ctor_set_uint8(v___x_200_, 0, v___x_201_);
lean_inc(v_deprecation_x3f_199_);
lean_inc_ref(v_descr_198_);
lean_inc_n(v_name_193_, 2);
v___x_202_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_202_, 0, v_name_193_);
lean_ctor_set(v___x_202_, 1, v_ref_195_);
lean_ctor_set(v___x_202_, 2, v___x_200_);
lean_ctor_set(v___x_202_, 3, v_descr_198_);
lean_ctor_set(v___x_202_, 4, v_deprecation_x3f_199_);
v___x_203_ = lean_register_option(v_name_193_, v___x_202_);
if (lean_obj_tag(v___x_203_) == 0)
{
lean_object* v___x_205_; uint8_t v_isShared_206_; uint8_t v_isSharedCheck_211_; 
v_isSharedCheck_211_ = !lean_is_exclusive(v___x_203_);
if (v_isSharedCheck_211_ == 0)
{
lean_object* v_unused_212_; 
v_unused_212_ = lean_ctor_get(v___x_203_, 0);
lean_dec(v_unused_212_);
v___x_205_ = v___x_203_;
v_isShared_206_ = v_isSharedCheck_211_;
goto v_resetjp_204_;
}
else
{
lean_dec(v___x_203_);
v___x_205_ = lean_box(0);
v_isShared_206_ = v_isSharedCheck_211_;
goto v_resetjp_204_;
}
v_resetjp_204_:
{
lean_object* v___x_207_; lean_object* v___x_209_; 
lean_inc(v_defValue_197_);
v___x_207_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_207_, 0, v_name_193_);
lean_ctor_set(v___x_207_, 1, v_defValue_197_);
if (v_isShared_206_ == 0)
{
lean_ctor_set(v___x_205_, 0, v___x_207_);
v___x_209_ = v___x_205_;
goto v_reusejp_208_;
}
else
{
lean_object* v_reuseFailAlloc_210_; 
v_reuseFailAlloc_210_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_210_, 0, v___x_207_);
v___x_209_ = v_reuseFailAlloc_210_;
goto v_reusejp_208_;
}
v_reusejp_208_:
{
return v___x_209_;
}
}
}
else
{
lean_object* v_a_213_; lean_object* v___x_215_; uint8_t v_isShared_216_; uint8_t v_isSharedCheck_220_; 
lean_dec(v_name_193_);
v_a_213_ = lean_ctor_get(v___x_203_, 0);
v_isSharedCheck_220_ = !lean_is_exclusive(v___x_203_);
if (v_isSharedCheck_220_ == 0)
{
v___x_215_ = v___x_203_;
v_isShared_216_ = v_isSharedCheck_220_;
goto v_resetjp_214_;
}
else
{
lean_inc(v_a_213_);
lean_dec(v___x_203_);
v___x_215_ = lean_box(0);
v_isShared_216_ = v_isSharedCheck_220_;
goto v_resetjp_214_;
}
v_resetjp_214_:
{
lean_object* v___x_218_; 
if (v_isShared_216_ == 0)
{
v___x_218_ = v___x_215_;
goto v_reusejp_217_;
}
else
{
lean_object* v_reuseFailAlloc_219_; 
v_reuseFailAlloc_219_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_219_, 0, v_a_213_);
v___x_218_ = v_reuseFailAlloc_219_;
goto v_reusejp_217_;
}
v_reusejp_217_:
{
return v___x_218_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4__spec__0___boxed(lean_object* v_name_221_, lean_object* v_decl_222_, lean_object* v_ref_223_, lean_object* v_a_224_){
_start:
{
lean_object* v_res_225_; 
v_res_225_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4__spec__0(v_name_221_, v_decl_222_, v_ref_223_);
lean_dec_ref(v_decl_222_);
return v_res_225_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_248_; lean_object* v___x_249_; lean_object* v___x_250_; lean_object* v___x_251_; 
v___x_248_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4_));
v___x_249_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4_));
v___x_250_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4_));
v___x_251_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4__spec__0(v___x_248_, v___x_249_, v___x_250_);
return v___x_251_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4____boxed(lean_object* v_a_252_){
_start:
{
lean_object* v_res_253_; 
v_res_253_ = lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4_();
return v_res_253_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__2(void){
_start:
{
lean_object* v___x_256_; lean_object* v___x_257_; lean_object* v___x_258_; 
v___x_256_ = lean_box(0);
v___x_257_ = lean_unsigned_to_nat(16u);
v___x_258_ = lean_mk_array(v___x_257_, v___x_256_);
return v___x_258_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__3(void){
_start:
{
lean_object* v___x_259_; lean_object* v___x_260_; lean_object* v___x_261_; 
v___x_259_ = lean_obj_once(&lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__2, &lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__2_once, _init_lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__2);
v___x_260_ = lean_unsigned_to_nat(0u);
v___x_261_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_261_, 0, v___x_260_);
lean_ctor_set(v___x_261_, 1, v___x_259_);
return v___x_261_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__9(void){
_start:
{
lean_object* v___x_271_; lean_object* v___x_272_; lean_object* v___x_273_; lean_object* v___x_274_; lean_object* v___x_275_; lean_object* v___x_276_; 
v___x_271_ = lean_box(0);
v___x_272_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__8));
v___x_273_ = lean_obj_once(&lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__3, &lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__3_once, _init_lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__3);
v___x_274_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__1));
v___x_275_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__0));
v___x_276_ = l_Std_DHashMap_Internal_Raw_u2080_insertIfNew___redArg(v___x_275_, v___x_274_, v___x_273_, v___x_272_, v___x_271_);
return v___x_276_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__12(void){
_start:
{
lean_object* v___x_283_; lean_object* v___x_284_; lean_object* v___x_285_; lean_object* v___x_286_; lean_object* v___x_287_; lean_object* v___x_288_; 
v___x_283_ = lean_box(0);
v___x_284_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__11));
v___x_285_ = lean_obj_once(&lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__9, &lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__9_once, _init_lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__9);
v___x_286_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__1));
v___x_287_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__0));
v___x_288_ = l_Std_DHashMap_Internal_Raw_u2080_insertIfNew___redArg(v___x_287_, v___x_286_, v___x_285_, v___x_284_, v___x_283_);
return v___x_288_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__15(void){
_start:
{
lean_object* v___x_295_; lean_object* v___x_296_; lean_object* v___x_297_; lean_object* v___x_298_; lean_object* v___x_299_; lean_object* v___x_300_; 
v___x_295_ = lean_box(0);
v___x_296_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__14));
v___x_297_ = lean_obj_once(&lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__12, &lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__12_once, _init_lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__12);
v___x_298_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__1));
v___x_299_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__0));
v___x_300_ = l_Std_DHashMap_Internal_Raw_u2080_insertIfNew___redArg(v___x_299_, v___x_298_, v___x_297_, v___x_296_, v___x_295_);
return v___x_300_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__18(void){
_start:
{
lean_object* v___x_304_; lean_object* v___x_305_; lean_object* v___x_306_; lean_object* v___x_307_; lean_object* v___x_308_; lean_object* v___x_309_; 
v___x_304_ = lean_box(0);
v___x_305_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__17));
v___x_306_ = lean_obj_once(&lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__15, &lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__15_once, _init_lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__15);
v___x_307_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__1));
v___x_308_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__0));
v___x_309_ = l_Std_DHashMap_Internal_Raw_u2080_insertIfNew___redArg(v___x_308_, v___x_307_, v___x_306_, v___x_305_, v___x_304_);
return v___x_309_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines(void){
_start:
{
lean_object* v___x_310_; 
v___x_310_ = lean_obj_once(&lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__18, &lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__18_once, _init_lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__18);
return v___x_310_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Linter_EmptyLine_SkippedFileSegments___closed__2(void){
_start:
{
lean_object* v___x_314_; lean_object* v___x_315_; lean_object* v___x_316_; lean_object* v___x_317_; lean_object* v___x_318_; lean_object* v___x_319_; 
v___x_314_ = lean_box(0);
v___x_315_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_EmptyLine_SkippedFileSegments___closed__1));
v___x_316_ = lean_obj_once(&lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__3, &lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__3_once, _init_lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__3);
v___x_317_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__1));
v___x_318_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__0));
v___x_319_ = l_Std_DHashMap_Internal_Raw_u2080_insertIfNew___redArg(v___x_318_, v___x_317_, v___x_316_, v___x_315_, v___x_314_);
return v___x_319_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Linter_EmptyLine_SkippedFileSegments___closed__5(void){
_start:
{
lean_object* v___x_323_; lean_object* v___x_324_; lean_object* v___x_325_; lean_object* v___x_326_; lean_object* v___x_327_; lean_object* v___x_328_; 
v___x_323_ = lean_box(0);
v___x_324_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_EmptyLine_SkippedFileSegments___closed__4));
v___x_325_ = lean_obj_once(&lp_mathlib_Mathlib_Linter_EmptyLine_SkippedFileSegments___closed__2, &lp_mathlib_Mathlib_Linter_EmptyLine_SkippedFileSegments___closed__2_once, _init_lp_mathlib_Mathlib_Linter_EmptyLine_SkippedFileSegments___closed__2);
v___x_326_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__1));
v___x_327_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__0));
v___x_328_ = l_Std_DHashMap_Internal_Raw_u2080_insertIfNew___redArg(v___x_327_, v___x_326_, v___x_325_, v___x_324_, v___x_323_);
return v___x_328_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Linter_EmptyLine_SkippedFileSegments___closed__8(void){
_start:
{
lean_object* v___x_332_; lean_object* v___x_333_; lean_object* v___x_334_; lean_object* v___x_335_; lean_object* v___x_336_; lean_object* v___x_337_; 
v___x_332_ = lean_box(0);
v___x_333_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_EmptyLine_SkippedFileSegments___closed__7));
v___x_334_ = lean_obj_once(&lp_mathlib_Mathlib_Linter_EmptyLine_SkippedFileSegments___closed__5, &lp_mathlib_Mathlib_Linter_EmptyLine_SkippedFileSegments___closed__5_once, _init_lp_mathlib_Mathlib_Linter_EmptyLine_SkippedFileSegments___closed__5);
v___x_335_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__1));
v___x_336_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__0));
v___x_337_ = l_Std_DHashMap_Internal_Raw_u2080_insertIfNew___redArg(v___x_336_, v___x_335_, v___x_334_, v___x_333_, v___x_332_);
return v___x_337_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Linter_EmptyLine_SkippedFileSegments(void){
_start:
{
lean_object* v___x_338_; 
v___x_338_ = lean_obj_once(&lp_mathlib_Mathlib_Linter_EmptyLine_SkippedFileSegments___closed__8, &lp_mathlib_Mathlib_Linter_EmptyLine_SkippedFileSegments___closed__8_once, _init_lp_mathlib_Mathlib_Linter_EmptyLine_SkippedFileSegments___closed__8);
return v___x_338_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__2___redArg(lean_object* v___y_339_){
_start:
{
lean_object* v___x_341_; lean_object* v_env_342_; lean_object* v___x_343_; lean_object* v_mainModule_344_; lean_object* v___x_345_; 
v___x_341_ = lean_st_ref_get(v___y_339_);
v_env_342_ = lean_ctor_get(v___x_341_, 0);
lean_inc_ref(v_env_342_);
lean_dec(v___x_341_);
v___x_343_ = l_Lean_Environment_header(v_env_342_);
lean_dec_ref(v_env_342_);
v_mainModule_344_ = lean_ctor_get(v___x_343_, 0);
lean_inc(v_mainModule_344_);
lean_dec_ref(v___x_343_);
v___x_345_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_345_, 0, v_mainModule_344_);
return v___x_345_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__2___redArg___boxed(lean_object* v___y_346_, lean_object* v___y_347_){
_start:
{
lean_object* v_res_348_; 
v_res_348_ = lp_mathlib_Lean_getMainModule___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__2___redArg(v___y_346_);
lean_dec(v___y_346_);
return v_res_348_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__2(lean_object* v___y_349_, lean_object* v___y_350_){
_start:
{
lean_object* v___x_352_; 
v___x_352_ = lp_mathlib_Lean_getMainModule___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__2___redArg(v___y_350_);
return v___x_352_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__2___boxed(lean_object* v___y_353_, lean_object* v___y_354_, lean_object* v___y_355_){
_start:
{
lean_object* v_res_356_; 
v_res_356_ = lp_mathlib_Lean_getMainModule___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__2(v___y_353_, v___y_354_);
lean_dec(v___y_354_);
lean_dec_ref(v___y_353_);
return v_res_356_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__6_spec__10___redArg(lean_object* v_a_357_, lean_object* v_x_358_){
_start:
{
if (lean_obj_tag(v_x_358_) == 0)
{
uint8_t v___x_359_; 
v___x_359_ = 0;
return v___x_359_;
}
else
{
lean_object* v_key_360_; lean_object* v_tail_361_; uint8_t v___x_362_; 
v_key_360_ = lean_ctor_get(v_x_358_, 0);
v_tail_361_ = lean_ctor_get(v_x_358_, 2);
v___x_362_ = lean_name_eq(v_key_360_, v_a_357_);
if (v___x_362_ == 0)
{
v_x_358_ = v_tail_361_;
goto _start;
}
else
{
return v___x_362_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__6_spec__10___redArg___boxed(lean_object* v_a_364_, lean_object* v_x_365_){
_start:
{
uint8_t v_res_366_; lean_object* v_r_367_; 
v_res_366_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__6_spec__10___redArg(v_a_364_, v_x_365_);
lean_dec(v_x_365_);
lean_dec(v_a_364_);
v_r_367_ = lean_box(v_res_366_);
return v_r_367_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__7___redArg(lean_object* v_m_368_, lean_object* v_a_369_){
_start:
{
lean_object* v_buckets_370_; lean_object* v___x_371_; uint64_t v___y_373_; 
v_buckets_370_ = lean_ctor_get(v_m_368_, 1);
v___x_371_ = lean_array_get_size(v_buckets_370_);
if (lean_obj_tag(v_a_369_) == 0)
{
uint64_t v___x_387_; 
v___x_387_ = 1723ULL;
v___y_373_ = v___x_387_;
goto v___jp_372_;
}
else
{
uint64_t v_hash_388_; 
v_hash_388_ = lean_ctor_get_uint64(v_a_369_, sizeof(void*)*2);
v___y_373_ = v_hash_388_;
goto v___jp_372_;
}
v___jp_372_:
{
uint64_t v___x_374_; uint64_t v___x_375_; uint64_t v_fold_376_; uint64_t v___x_377_; uint64_t v___x_378_; uint64_t v___x_379_; size_t v___x_380_; size_t v___x_381_; size_t v___x_382_; size_t v___x_383_; size_t v___x_384_; lean_object* v___x_385_; uint8_t v___x_386_; 
v___x_374_ = 32ULL;
v___x_375_ = lean_uint64_shift_right(v___y_373_, v___x_374_);
v_fold_376_ = lean_uint64_xor(v___y_373_, v___x_375_);
v___x_377_ = 16ULL;
v___x_378_ = lean_uint64_shift_right(v_fold_376_, v___x_377_);
v___x_379_ = lean_uint64_xor(v_fold_376_, v___x_378_);
v___x_380_ = lean_uint64_to_usize(v___x_379_);
v___x_381_ = lean_usize_of_nat(v___x_371_);
v___x_382_ = ((size_t)1ULL);
v___x_383_ = lean_usize_sub(v___x_381_, v___x_382_);
v___x_384_ = lean_usize_land(v___x_380_, v___x_383_);
v___x_385_ = lean_array_uget_borrowed(v_buckets_370_, v___x_384_);
v___x_386_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__6_spec__10___redArg(v_a_369_, v___x_385_);
return v___x_386_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__7___redArg___boxed(lean_object* v_m_389_, lean_object* v_a_390_){
_start:
{
uint8_t v_res_391_; lean_object* v_r_392_; 
v_res_391_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__7___redArg(v_m_389_, v_a_390_);
lean_dec(v_a_390_);
lean_dec_ref(v_m_389_);
v_r_392_ = lean_box(v_res_391_);
return v_r_392_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__6_spec__11_spec__13_spec__27___redArg(lean_object* v_x_393_, lean_object* v_x_394_){
_start:
{
if (lean_obj_tag(v_x_394_) == 0)
{
return v_x_393_;
}
else
{
lean_object* v_key_395_; lean_object* v_value_396_; lean_object* v_tail_397_; lean_object* v___x_399_; uint8_t v_isShared_400_; uint8_t v_isSharedCheck_423_; 
v_key_395_ = lean_ctor_get(v_x_394_, 0);
v_value_396_ = lean_ctor_get(v_x_394_, 1);
v_tail_397_ = lean_ctor_get(v_x_394_, 2);
v_isSharedCheck_423_ = !lean_is_exclusive(v_x_394_);
if (v_isSharedCheck_423_ == 0)
{
v___x_399_ = v_x_394_;
v_isShared_400_ = v_isSharedCheck_423_;
goto v_resetjp_398_;
}
else
{
lean_inc(v_tail_397_);
lean_inc(v_value_396_);
lean_inc(v_key_395_);
lean_dec(v_x_394_);
v___x_399_ = lean_box(0);
v_isShared_400_ = v_isSharedCheck_423_;
goto v_resetjp_398_;
}
v_resetjp_398_:
{
lean_object* v___x_401_; uint64_t v___y_403_; 
v___x_401_ = lean_array_get_size(v_x_393_);
if (lean_obj_tag(v_key_395_) == 0)
{
uint64_t v___x_421_; 
v___x_421_ = 1723ULL;
v___y_403_ = v___x_421_;
goto v___jp_402_;
}
else
{
uint64_t v_hash_422_; 
v_hash_422_ = lean_ctor_get_uint64(v_key_395_, sizeof(void*)*2);
v___y_403_ = v_hash_422_;
goto v___jp_402_;
}
v___jp_402_:
{
uint64_t v___x_404_; uint64_t v___x_405_; uint64_t v_fold_406_; uint64_t v___x_407_; uint64_t v___x_408_; uint64_t v___x_409_; size_t v___x_410_; size_t v___x_411_; size_t v___x_412_; size_t v___x_413_; size_t v___x_414_; lean_object* v___x_415_; lean_object* v___x_417_; 
v___x_404_ = 32ULL;
v___x_405_ = lean_uint64_shift_right(v___y_403_, v___x_404_);
v_fold_406_ = lean_uint64_xor(v___y_403_, v___x_405_);
v___x_407_ = 16ULL;
v___x_408_ = lean_uint64_shift_right(v_fold_406_, v___x_407_);
v___x_409_ = lean_uint64_xor(v_fold_406_, v___x_408_);
v___x_410_ = lean_uint64_to_usize(v___x_409_);
v___x_411_ = lean_usize_of_nat(v___x_401_);
v___x_412_ = ((size_t)1ULL);
v___x_413_ = lean_usize_sub(v___x_411_, v___x_412_);
v___x_414_ = lean_usize_land(v___x_410_, v___x_413_);
v___x_415_ = lean_array_uget_borrowed(v_x_393_, v___x_414_);
lean_inc(v___x_415_);
if (v_isShared_400_ == 0)
{
lean_ctor_set(v___x_399_, 2, v___x_415_);
v___x_417_ = v___x_399_;
goto v_reusejp_416_;
}
else
{
lean_object* v_reuseFailAlloc_420_; 
v_reuseFailAlloc_420_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_420_, 0, v_key_395_);
lean_ctor_set(v_reuseFailAlloc_420_, 1, v_value_396_);
lean_ctor_set(v_reuseFailAlloc_420_, 2, v___x_415_);
v___x_417_ = v_reuseFailAlloc_420_;
goto v_reusejp_416_;
}
v_reusejp_416_:
{
lean_object* v___x_418_; 
v___x_418_ = lean_array_uset(v_x_393_, v___x_414_, v___x_417_);
v_x_393_ = v___x_418_;
v_x_394_ = v_tail_397_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__6_spec__11_spec__13___redArg(lean_object* v_i_424_, lean_object* v_source_425_, lean_object* v_target_426_){
_start:
{
lean_object* v___x_427_; uint8_t v___x_428_; 
v___x_427_ = lean_array_get_size(v_source_425_);
v___x_428_ = lean_nat_dec_lt(v_i_424_, v___x_427_);
if (v___x_428_ == 0)
{
lean_dec_ref(v_source_425_);
lean_dec(v_i_424_);
return v_target_426_;
}
else
{
lean_object* v_es_429_; lean_object* v___x_430_; lean_object* v_source_431_; lean_object* v_target_432_; lean_object* v___x_433_; lean_object* v___x_434_; 
v_es_429_ = lean_array_fget(v_source_425_, v_i_424_);
v___x_430_ = lean_box(0);
v_source_431_ = lean_array_fset(v_source_425_, v_i_424_, v___x_430_);
v_target_432_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__6_spec__11_spec__13_spec__27___redArg(v_target_426_, v_es_429_);
v___x_433_ = lean_unsigned_to_nat(1u);
v___x_434_ = lean_nat_add(v_i_424_, v___x_433_);
lean_dec(v_i_424_);
v_i_424_ = v___x_434_;
v_source_425_ = v_source_431_;
v_target_426_ = v_target_432_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__6_spec__11___redArg(lean_object* v_data_436_){
_start:
{
lean_object* v___x_437_; lean_object* v___x_438_; lean_object* v_nbuckets_439_; lean_object* v___x_440_; lean_object* v___x_441_; lean_object* v___x_442_; lean_object* v___x_443_; 
v___x_437_ = lean_array_get_size(v_data_436_);
v___x_438_ = lean_unsigned_to_nat(2u);
v_nbuckets_439_ = lean_nat_mul(v___x_437_, v___x_438_);
v___x_440_ = lean_unsigned_to_nat(0u);
v___x_441_ = lean_box(0);
v___x_442_ = lean_mk_array(v_nbuckets_439_, v___x_441_);
v___x_443_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__6_spec__11_spec__13___redArg(v___x_440_, v_data_436_, v___x_442_);
return v___x_443_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__6___redArg(lean_object* v_m_444_, lean_object* v_a_445_, lean_object* v_b_446_){
_start:
{
lean_object* v_size_447_; lean_object* v_buckets_448_; lean_object* v___x_449_; uint64_t v___y_451_; 
v_size_447_ = lean_ctor_get(v_m_444_, 0);
v_buckets_448_ = lean_ctor_get(v_m_444_, 1);
v___x_449_ = lean_array_get_size(v_buckets_448_);
if (lean_obj_tag(v_a_445_) == 0)
{
uint64_t v___x_488_; 
v___x_488_ = 1723ULL;
v___y_451_ = v___x_488_;
goto v___jp_450_;
}
else
{
uint64_t v_hash_489_; 
v_hash_489_ = lean_ctor_get_uint64(v_a_445_, sizeof(void*)*2);
v___y_451_ = v_hash_489_;
goto v___jp_450_;
}
v___jp_450_:
{
uint64_t v___x_452_; uint64_t v___x_453_; uint64_t v_fold_454_; uint64_t v___x_455_; uint64_t v___x_456_; uint64_t v___x_457_; size_t v___x_458_; size_t v___x_459_; size_t v___x_460_; size_t v___x_461_; size_t v___x_462_; lean_object* v_bkt_463_; uint8_t v___x_464_; 
v___x_452_ = 32ULL;
v___x_453_ = lean_uint64_shift_right(v___y_451_, v___x_452_);
v_fold_454_ = lean_uint64_xor(v___y_451_, v___x_453_);
v___x_455_ = 16ULL;
v___x_456_ = lean_uint64_shift_right(v_fold_454_, v___x_455_);
v___x_457_ = lean_uint64_xor(v_fold_454_, v___x_456_);
v___x_458_ = lean_uint64_to_usize(v___x_457_);
v___x_459_ = lean_usize_of_nat(v___x_449_);
v___x_460_ = ((size_t)1ULL);
v___x_461_ = lean_usize_sub(v___x_459_, v___x_460_);
v___x_462_ = lean_usize_land(v___x_458_, v___x_461_);
v_bkt_463_ = lean_array_uget_borrowed(v_buckets_448_, v___x_462_);
v___x_464_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__6_spec__10___redArg(v_a_445_, v_bkt_463_);
if (v___x_464_ == 0)
{
lean_object* v___x_466_; uint8_t v_isShared_467_; uint8_t v_isSharedCheck_485_; 
lean_inc_ref(v_buckets_448_);
lean_inc(v_size_447_);
v_isSharedCheck_485_ = !lean_is_exclusive(v_m_444_);
if (v_isSharedCheck_485_ == 0)
{
lean_object* v_unused_486_; lean_object* v_unused_487_; 
v_unused_486_ = lean_ctor_get(v_m_444_, 1);
lean_dec(v_unused_486_);
v_unused_487_ = lean_ctor_get(v_m_444_, 0);
lean_dec(v_unused_487_);
v___x_466_ = v_m_444_;
v_isShared_467_ = v_isSharedCheck_485_;
goto v_resetjp_465_;
}
else
{
lean_dec(v_m_444_);
v___x_466_ = lean_box(0);
v_isShared_467_ = v_isSharedCheck_485_;
goto v_resetjp_465_;
}
v_resetjp_465_:
{
lean_object* v___x_468_; lean_object* v_size_x27_469_; lean_object* v___x_470_; lean_object* v_buckets_x27_471_; lean_object* v___x_472_; lean_object* v___x_473_; lean_object* v___x_474_; lean_object* v___x_475_; lean_object* v___x_476_; uint8_t v___x_477_; 
v___x_468_ = lean_unsigned_to_nat(1u);
v_size_x27_469_ = lean_nat_add(v_size_447_, v___x_468_);
lean_dec(v_size_447_);
lean_inc(v_bkt_463_);
v___x_470_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_470_, 0, v_a_445_);
lean_ctor_set(v___x_470_, 1, v_b_446_);
lean_ctor_set(v___x_470_, 2, v_bkt_463_);
v_buckets_x27_471_ = lean_array_uset(v_buckets_448_, v___x_462_, v___x_470_);
v___x_472_ = lean_unsigned_to_nat(4u);
v___x_473_ = lean_nat_mul(v_size_x27_469_, v___x_472_);
v___x_474_ = lean_unsigned_to_nat(3u);
v___x_475_ = lean_nat_div(v___x_473_, v___x_474_);
lean_dec(v___x_473_);
v___x_476_ = lean_array_get_size(v_buckets_x27_471_);
v___x_477_ = lean_nat_dec_le(v___x_475_, v___x_476_);
lean_dec(v___x_475_);
if (v___x_477_ == 0)
{
lean_object* v_val_478_; lean_object* v___x_480_; 
v_val_478_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__6_spec__11___redArg(v_buckets_x27_471_);
if (v_isShared_467_ == 0)
{
lean_ctor_set(v___x_466_, 1, v_val_478_);
lean_ctor_set(v___x_466_, 0, v_size_x27_469_);
v___x_480_ = v___x_466_;
goto v_reusejp_479_;
}
else
{
lean_object* v_reuseFailAlloc_481_; 
v_reuseFailAlloc_481_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_481_, 0, v_size_x27_469_);
lean_ctor_set(v_reuseFailAlloc_481_, 1, v_val_478_);
v___x_480_ = v_reuseFailAlloc_481_;
goto v_reusejp_479_;
}
v_reusejp_479_:
{
return v___x_480_;
}
}
else
{
lean_object* v___x_483_; 
if (v_isShared_467_ == 0)
{
lean_ctor_set(v___x_466_, 1, v_buckets_x27_471_);
lean_ctor_set(v___x_466_, 0, v_size_x27_469_);
v___x_483_ = v___x_466_;
goto v_reusejp_482_;
}
else
{
lean_object* v_reuseFailAlloc_484_; 
v_reuseFailAlloc_484_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_484_, 0, v_size_x27_469_);
lean_ctor_set(v_reuseFailAlloc_484_, 1, v_buckets_x27_471_);
v___x_483_ = v_reuseFailAlloc_484_;
goto v_reusejp_482_;
}
v_reusejp_482_:
{
return v___x_483_;
}
}
}
}
else
{
lean_dec(v_b_446_);
lean_dec(v_a_445_);
return v_m_444_;
}
}
}
}
static lean_object* _init_lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__0___closed__0(void){
_start:
{
lean_object* v___x_490_; lean_object* v___x_491_; lean_object* v___x_492_; lean_object* v___x_493_; 
v___x_490_ = lean_box(0);
v___x_491_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__8));
v___x_492_ = lean_obj_once(&lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__3, &lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__3_once, _init_lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__3);
v___x_493_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__6___redArg(v___x_492_, v___x_491_, v___x_490_);
return v___x_493_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__0___closed__1(void){
_start:
{
lean_object* v___x_494_; lean_object* v___x_495_; lean_object* v___x_496_; lean_object* v___x_497_; 
v___x_494_ = lean_box(0);
v___x_495_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__11));
v___x_496_ = lean_obj_once(&lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__0___closed__0, &lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__0___closed__0_once, _init_lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__0___closed__0);
v___x_497_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__6___redArg(v___x_496_, v___x_495_, v___x_494_);
return v___x_497_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__0___closed__2(void){
_start:
{
lean_object* v___x_498_; lean_object* v___x_499_; lean_object* v___x_500_; lean_object* v___x_501_; 
v___x_498_ = lean_box(0);
v___x_499_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__14));
v___x_500_ = lean_obj_once(&lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__0___closed__1, &lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__0___closed__1);
v___x_501_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__6___redArg(v___x_500_, v___x_499_, v___x_498_);
return v___x_501_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__0___closed__3(void){
_start:
{
lean_object* v___x_502_; lean_object* v___x_503_; lean_object* v___x_504_; lean_object* v___x_505_; 
v___x_502_ = lean_box(0);
v___x_503_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__17));
v___x_504_ = lean_obj_once(&lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__0___closed__2, &lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__0___closed__2_once, _init_lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__0___closed__2);
v___x_505_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__6___redArg(v___x_504_, v___x_503_, v___x_502_);
return v___x_505_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__0(lean_object* v_x_506_){
_start:
{
lean_object* v___x_507_; lean_object* v___x_508_; uint8_t v___x_509_; 
v___x_507_ = lean_obj_once(&lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__0___closed__3, &lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__0___closed__3_once, _init_lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__0___closed__3);
v___x_508_ = l_Lean_Syntax_getKind(v_x_506_);
v___x_509_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__7___redArg(v___x_507_, v___x_508_);
lean_dec(v___x_508_);
return v___x_509_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__0___boxed(lean_object* v_x_510_){
_start:
{
uint8_t v_res_511_; lean_object* v_r_512_; 
v_res_511_ = lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__0(v_x_510_);
v_r_512_ = lean_box(v_res_511_);
return v_r_512_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Substring_Raw_takeRightWhileAux___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__4(lean_object* v_s_513_, lean_object* v_begPos_514_, lean_object* v_i_515_){
_start:
{
uint8_t v___x_516_; 
v___x_516_ = lean_nat_dec_lt(v_begPos_514_, v_i_515_);
if (v___x_516_ == 0)
{
return v_i_515_;
}
else
{
lean_object* v_i_x27_517_; uint8_t v___y_519_; uint32_t v_c_521_; uint8_t v___y_523_; uint32_t v___x_529_; uint8_t v___x_530_; 
v_i_x27_517_ = lean_string_utf8_prev(v_s_513_, v_i_515_);
v_c_521_ = lean_string_utf8_get(v_s_513_, v_i_x27_517_);
v___x_529_ = 32;
v___x_530_ = lean_uint32_dec_eq(v_c_521_, v___x_529_);
if (v___x_530_ == 0)
{
uint32_t v___x_531_; uint8_t v___x_532_; 
v___x_531_ = 9;
v___x_532_ = lean_uint32_dec_eq(v_c_521_, v___x_531_);
v___y_523_ = v___x_532_;
goto v___jp_522_;
}
else
{
v___y_523_ = v___x_530_;
goto v___jp_522_;
}
v___jp_518_:
{
if (v___y_519_ == 0)
{
lean_dec(v_i_x27_517_);
return v_i_515_;
}
else
{
lean_dec(v_i_515_);
v_i_515_ = v_i_x27_517_;
goto _start;
}
}
v___jp_522_:
{
if (v___y_523_ == 0)
{
uint32_t v___x_524_; uint8_t v___x_525_; 
v___x_524_ = 13;
v___x_525_ = lean_uint32_dec_eq(v_c_521_, v___x_524_);
if (v___x_525_ == 0)
{
uint32_t v___x_526_; uint8_t v___x_527_; 
v___x_526_ = 10;
v___x_527_ = lean_uint32_dec_eq(v_c_521_, v___x_526_);
v___y_519_ = v___x_527_;
goto v___jp_518_;
}
else
{
v___y_519_ = v___x_525_;
goto v___jp_518_;
}
}
else
{
lean_dec(v_i_515_);
v_i_515_ = v_i_x27_517_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Substring_Raw_takeRightWhileAux___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__4___boxed(lean_object* v_s_533_, lean_object* v_begPos_534_, lean_object* v_i_535_){
_start:
{
lean_object* v_res_536_; 
v_res_536_ = lp_mathlib_Substring_Raw_takeRightWhileAux___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__4(v_s_533_, v_begPos_534_, v_i_535_);
lean_dec(v_begPos_534_);
lean_dec_ref(v_s_533_);
return v_res_536_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Substring_Raw_takeWhileAux___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__3(lean_object* v_s_537_, lean_object* v_stopPos_538_, lean_object* v_i_539_){
_start:
{
uint8_t v___y_544_; uint8_t v___x_545_; 
v___x_545_ = lean_nat_dec_lt(v_i_539_, v_stopPos_538_);
if (v___x_545_ == 0)
{
return v_i_539_;
}
else
{
uint32_t v___x_546_; uint8_t v___y_548_; uint32_t v___x_553_; uint8_t v___x_554_; 
v___x_546_ = lean_string_utf8_get(v_s_537_, v_i_539_);
v___x_553_ = 32;
v___x_554_ = lean_uint32_dec_eq(v___x_546_, v___x_553_);
if (v___x_554_ == 0)
{
uint32_t v___x_555_; uint8_t v___x_556_; 
v___x_555_ = 9;
v___x_556_ = lean_uint32_dec_eq(v___x_546_, v___x_555_);
v___y_548_ = v___x_556_;
goto v___jp_547_;
}
else
{
v___y_548_ = v___x_554_;
goto v___jp_547_;
}
v___jp_547_:
{
if (v___y_548_ == 0)
{
uint32_t v___x_549_; uint8_t v___x_550_; 
v___x_549_ = 13;
v___x_550_ = lean_uint32_dec_eq(v___x_546_, v___x_549_);
if (v___x_550_ == 0)
{
uint32_t v___x_551_; uint8_t v___x_552_; 
v___x_551_ = 10;
v___x_552_ = lean_uint32_dec_eq(v___x_546_, v___x_551_);
v___y_544_ = v___x_552_;
goto v___jp_543_;
}
else
{
v___y_544_ = v___x_550_;
goto v___jp_543_;
}
}
else
{
goto v___jp_540_;
}
}
}
v___jp_540_:
{
lean_object* v___x_541_; 
v___x_541_ = lean_string_utf8_next(v_s_537_, v_i_539_);
lean_dec(v_i_539_);
v_i_539_ = v___x_541_;
goto _start;
}
v___jp_543_:
{
if (v___y_544_ == 0)
{
return v_i_539_;
}
else
{
goto v___jp_540_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Substring_Raw_takeWhileAux___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__3___boxed(lean_object* v_s_557_, lean_object* v_stopPos_558_, lean_object* v_i_559_){
_start:
{
lean_object* v_res_560_; 
v_res_560_ = lp_mathlib_Substring_Raw_takeWhileAux___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__3(v_s_557_, v_stopPos_558_, v_i_559_);
lean_dec(v_stopPos_558_);
lean_dec_ref(v_s_557_);
return v_res_560_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5_spec__8___redArg(lean_object* v_s_561_, lean_object* v_a_562_, uint8_t v_b_563_){
_start:
{
uint8_t v___x_564_; 
v___x_564_ = 0;
switch(lean_obj_tag(v_a_562_))
{
case 0:
{
uint8_t v___x_565_; 
lean_dec_ref_known(v_a_562_, 1);
v___x_565_ = 1;
return v___x_565_;
}
case 1:
{
lean_object* v_pos_566_; lean_object* v___x_568_; uint8_t v_isShared_569_; uint8_t v_isSharedCheck_579_; 
v_pos_566_ = lean_ctor_get(v_a_562_, 0);
v_isSharedCheck_579_ = !lean_is_exclusive(v_a_562_);
if (v_isSharedCheck_579_ == 0)
{
v___x_568_ = v_a_562_;
v_isShared_569_ = v_isSharedCheck_579_;
goto v_resetjp_567_;
}
else
{
lean_inc(v_pos_566_);
lean_dec(v_a_562_);
v___x_568_ = lean_box(0);
v_isShared_569_ = v_isSharedCheck_579_;
goto v_resetjp_567_;
}
v_resetjp_567_:
{
lean_object* v_str_570_; lean_object* v_startInclusive_571_; lean_object* v___x_572_; lean_object* v___x_573_; lean_object* v___x_574_; lean_object* v___x_576_; 
v_str_570_ = lean_ctor_get(v_s_561_, 0);
v_startInclusive_571_ = lean_ctor_get(v_s_561_, 1);
v___x_572_ = lean_nat_add(v_startInclusive_571_, v_pos_566_);
lean_dec(v_pos_566_);
v___x_573_ = lean_string_utf8_next_fast(v_str_570_, v___x_572_);
lean_dec(v___x_572_);
v___x_574_ = lean_nat_sub(v___x_573_, v_startInclusive_571_);
if (v_isShared_569_ == 0)
{
lean_ctor_set_tag(v___x_568_, 0);
lean_ctor_set(v___x_568_, 0, v___x_574_);
v___x_576_ = v___x_568_;
goto v_reusejp_575_;
}
else
{
lean_object* v_reuseFailAlloc_578_; 
v_reuseFailAlloc_578_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_578_, 0, v___x_574_);
v___x_576_ = v_reuseFailAlloc_578_;
goto v_reusejp_575_;
}
v_reusejp_575_:
{
v_a_562_ = v___x_576_;
v_b_563_ = v___x_564_;
goto _start;
}
}
}
case 2:
{
lean_object* v_needle_580_; lean_object* v_table_581_; lean_object* v_stackPos_582_; lean_object* v_needlePos_583_; lean_object* v___x_585_; uint8_t v_isShared_586_; uint8_t v_isSharedCheck_636_; 
v_needle_580_ = lean_ctor_get(v_a_562_, 0);
v_table_581_ = lean_ctor_get(v_a_562_, 1);
v_stackPos_582_ = lean_ctor_get(v_a_562_, 2);
v_needlePos_583_ = lean_ctor_get(v_a_562_, 3);
v_isSharedCheck_636_ = !lean_is_exclusive(v_a_562_);
if (v_isSharedCheck_636_ == 0)
{
v___x_585_ = v_a_562_;
v_isShared_586_ = v_isSharedCheck_636_;
goto v_resetjp_584_;
}
else
{
lean_inc(v_needlePos_583_);
lean_inc(v_stackPos_582_);
lean_inc(v_table_581_);
lean_inc(v_needle_580_);
lean_dec(v_a_562_);
v___x_585_ = lean_box(0);
v_isShared_586_ = v_isSharedCheck_636_;
goto v_resetjp_584_;
}
v_resetjp_584_:
{
lean_object* v_str_587_; lean_object* v_startInclusive_588_; lean_object* v_endExclusive_589_; lean_object* v_str_590_; lean_object* v_startInclusive_591_; lean_object* v_endExclusive_592_; lean_object* v_basePos_593_; lean_object* v___x_594_; lean_object* v___x_595_; lean_object* v___x_596_; uint8_t v___x_597_; 
v_str_587_ = lean_ctor_get(v_needle_580_, 0);
v_startInclusive_588_ = lean_ctor_get(v_needle_580_, 1);
v_endExclusive_589_ = lean_ctor_get(v_needle_580_, 2);
v_str_590_ = lean_ctor_get(v_s_561_, 0);
v_startInclusive_591_ = lean_ctor_get(v_s_561_, 1);
v_endExclusive_592_ = lean_ctor_get(v_s_561_, 2);
v_basePos_593_ = lean_nat_sub(v_stackPos_582_, v_needlePos_583_);
v___x_594_ = lean_nat_sub(v_endExclusive_589_, v_startInclusive_588_);
v___x_595_ = lean_nat_add(v_basePos_593_, v___x_594_);
v___x_596_ = lean_nat_sub(v_endExclusive_592_, v_startInclusive_591_);
v___x_597_ = lean_nat_dec_le(v___x_595_, v___x_596_);
lean_dec(v___x_595_);
if (v___x_597_ == 0)
{
uint8_t v___x_598_; 
lean_dec(v___x_594_);
lean_del_object(v___x_585_);
lean_dec(v_needlePos_583_);
lean_dec(v_stackPos_582_);
lean_dec_ref(v_table_581_);
lean_dec_ref(v_needle_580_);
v___x_598_ = lean_nat_dec_lt(v_basePos_593_, v___x_596_);
lean_dec(v___x_596_);
lean_dec(v_basePos_593_);
if (v___x_598_ == 0)
{
return v_b_563_;
}
else
{
lean_object* v___x_599_; 
v___x_599_ = lean_box(3);
v_a_562_ = v___x_599_;
v_b_563_ = v___x_564_;
goto _start;
}
}
else
{
lean_object* v___x_601_; uint8_t v_stackByte_602_; lean_object* v___x_603_; uint8_t v_patByte_604_; uint8_t v___x_605_; 
lean_dec(v___x_596_);
lean_dec(v_basePos_593_);
v___x_601_ = lean_nat_add(v_startInclusive_591_, v_stackPos_582_);
v_stackByte_602_ = lean_string_get_byte_fast(v_str_590_, v___x_601_);
v___x_603_ = lean_nat_add(v_startInclusive_588_, v_needlePos_583_);
v_patByte_604_ = lean_string_get_byte_fast(v_str_587_, v___x_603_);
v___x_605_ = lean_uint8_dec_eq(v_stackByte_602_, v_patByte_604_);
if (v___x_605_ == 0)
{
lean_object* v___x_606_; uint8_t v___x_607_; 
lean_dec(v___x_594_);
v___x_606_ = lean_unsigned_to_nat(0u);
v___x_607_ = lean_nat_dec_eq(v_needlePos_583_, v___x_606_);
if (v___x_607_ == 0)
{
lean_object* v___x_608_; lean_object* v___x_609_; lean_object* v_newNeedlePos_610_; uint8_t v___x_611_; 
v___x_608_ = lean_unsigned_to_nat(1u);
v___x_609_ = lean_nat_sub(v_needlePos_583_, v___x_608_);
lean_dec(v_needlePos_583_);
v_newNeedlePos_610_ = lean_array_fget_borrowed(v_table_581_, v___x_609_);
lean_dec(v___x_609_);
v___x_611_ = lean_nat_dec_eq(v_newNeedlePos_610_, v___x_606_);
if (v___x_611_ == 0)
{
lean_object* v___x_613_; 
lean_inc(v_newNeedlePos_610_);
if (v_isShared_586_ == 0)
{
lean_ctor_set(v___x_585_, 3, v_newNeedlePos_610_);
v___x_613_ = v___x_585_;
goto v_reusejp_612_;
}
else
{
lean_object* v_reuseFailAlloc_615_; 
v_reuseFailAlloc_615_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v_reuseFailAlloc_615_, 0, v_needle_580_);
lean_ctor_set(v_reuseFailAlloc_615_, 1, v_table_581_);
lean_ctor_set(v_reuseFailAlloc_615_, 2, v_stackPos_582_);
lean_ctor_set(v_reuseFailAlloc_615_, 3, v_newNeedlePos_610_);
v___x_613_ = v_reuseFailAlloc_615_;
goto v_reusejp_612_;
}
v_reusejp_612_:
{
v_a_562_ = v___x_613_;
v_b_563_ = v___x_564_;
goto _start;
}
}
else
{
lean_object* v_nextStackPos_616_; lean_object* v___x_618_; 
v_nextStackPos_616_ = l_String_Slice_posGE___redArg(v_s_561_, v_stackPos_582_);
if (v_isShared_586_ == 0)
{
lean_ctor_set(v___x_585_, 3, v___x_606_);
lean_ctor_set(v___x_585_, 2, v_nextStackPos_616_);
v___x_618_ = v___x_585_;
goto v_reusejp_617_;
}
else
{
lean_object* v_reuseFailAlloc_620_; 
v_reuseFailAlloc_620_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v_reuseFailAlloc_620_, 0, v_needle_580_);
lean_ctor_set(v_reuseFailAlloc_620_, 1, v_table_581_);
lean_ctor_set(v_reuseFailAlloc_620_, 2, v_nextStackPos_616_);
lean_ctor_set(v_reuseFailAlloc_620_, 3, v___x_606_);
v___x_618_ = v_reuseFailAlloc_620_;
goto v_reusejp_617_;
}
v_reusejp_617_:
{
v_a_562_ = v___x_618_;
v_b_563_ = v___x_564_;
goto _start;
}
}
}
else
{
lean_object* v___x_621_; lean_object* v___x_622_; lean_object* v_nextStackPos_623_; lean_object* v___x_625_; 
lean_dec(v_needlePos_583_);
v___x_621_ = lean_unsigned_to_nat(1u);
v___x_622_ = lean_nat_add(v_stackPos_582_, v___x_621_);
lean_dec(v_stackPos_582_);
v_nextStackPos_623_ = l_String_Slice_posGE___redArg(v_s_561_, v___x_622_);
if (v_isShared_586_ == 0)
{
lean_ctor_set(v___x_585_, 3, v___x_606_);
lean_ctor_set(v___x_585_, 2, v_nextStackPos_623_);
v___x_625_ = v___x_585_;
goto v_reusejp_624_;
}
else
{
lean_object* v_reuseFailAlloc_627_; 
v_reuseFailAlloc_627_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v_reuseFailAlloc_627_, 0, v_needle_580_);
lean_ctor_set(v_reuseFailAlloc_627_, 1, v_table_581_);
lean_ctor_set(v_reuseFailAlloc_627_, 2, v_nextStackPos_623_);
lean_ctor_set(v_reuseFailAlloc_627_, 3, v___x_606_);
v___x_625_ = v_reuseFailAlloc_627_;
goto v_reusejp_624_;
}
v_reusejp_624_:
{
v_a_562_ = v___x_625_;
v_b_563_ = v___x_564_;
goto _start;
}
}
}
else
{
lean_object* v___x_628_; lean_object* v_nextNeedlePos_629_; uint8_t v___x_630_; 
v___x_628_ = lean_unsigned_to_nat(1u);
v_nextNeedlePos_629_ = lean_nat_add(v_needlePos_583_, v___x_628_);
lean_dec(v_needlePos_583_);
v___x_630_ = lean_nat_dec_eq(v_nextNeedlePos_629_, v___x_594_);
lean_dec(v___x_594_);
if (v___x_630_ == 0)
{
lean_object* v_nextStackPos_631_; lean_object* v___x_633_; 
v_nextStackPos_631_ = lean_nat_add(v_stackPos_582_, v___x_628_);
lean_dec(v_stackPos_582_);
if (v_isShared_586_ == 0)
{
lean_ctor_set(v___x_585_, 3, v_nextNeedlePos_629_);
lean_ctor_set(v___x_585_, 2, v_nextStackPos_631_);
v___x_633_ = v___x_585_;
goto v_reusejp_632_;
}
else
{
lean_object* v_reuseFailAlloc_635_; 
v_reuseFailAlloc_635_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v_reuseFailAlloc_635_, 0, v_needle_580_);
lean_ctor_set(v_reuseFailAlloc_635_, 1, v_table_581_);
lean_ctor_set(v_reuseFailAlloc_635_, 2, v_nextStackPos_631_);
lean_ctor_set(v_reuseFailAlloc_635_, 3, v_nextNeedlePos_629_);
v___x_633_ = v_reuseFailAlloc_635_;
goto v_reusejp_632_;
}
v_reusejp_632_:
{
v_a_562_ = v___x_633_;
goto _start;
}
}
else
{
lean_dec(v_nextNeedlePos_629_);
lean_del_object(v___x_585_);
lean_dec(v_stackPos_582_);
lean_dec_ref(v_table_581_);
lean_dec_ref(v_needle_580_);
return v___x_630_;
}
}
}
}
}
default: 
{
return v_b_563_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5_spec__8___redArg___boxed(lean_object* v_s_637_, lean_object* v_a_638_, lean_object* v_b_639_){
_start:
{
uint8_t v_b_boxed_640_; uint8_t v_res_641_; lean_object* v_r_642_; 
v_b_boxed_640_ = lean_unbox(v_b_639_);
v_res_641_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5_spec__8___redArg(v_s_637_, v_a_638_, v_b_boxed_640_);
lean_dec_ref(v_s_637_);
v_r_642_ = lean_box(v_res_641_);
return v_r_642_;
}
}
static lean_object* _init_lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5___closed__1(void){
_start:
{
lean_object* v___x_644_; lean_object* v___x_645_; 
v___x_644_ = ((lean_object*)(lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5___closed__0));
v___x_645_ = lean_string_utf8_byte_size(v___x_644_);
return v___x_645_;
}
}
static uint8_t _init_lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5___closed__2(void){
_start:
{
lean_object* v___x_646_; lean_object* v___x_647_; uint8_t v___x_648_; 
v___x_646_ = lean_unsigned_to_nat(0u);
v___x_647_ = lean_obj_once(&lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5___closed__1, &lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5___closed__1_once, _init_lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5___closed__1);
v___x_648_ = lean_nat_dec_eq(v___x_647_, v___x_646_);
return v___x_648_;
}
}
static lean_object* _init_lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5___closed__3(void){
_start:
{
lean_object* v___x_649_; lean_object* v___x_650_; lean_object* v___x_651_; lean_object* v___x_652_; 
v___x_649_ = lean_obj_once(&lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5___closed__1, &lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5___closed__1_once, _init_lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5___closed__1);
v___x_650_ = lean_unsigned_to_nat(0u);
v___x_651_ = ((lean_object*)(lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5___closed__0));
v___x_652_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_652_, 0, v___x_651_);
lean_ctor_set(v___x_652_, 1, v___x_650_);
lean_ctor_set(v___x_652_, 2, v___x_649_);
return v___x_652_;
}
}
static lean_object* _init_lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5___closed__4(void){
_start:
{
lean_object* v___x_653_; lean_object* v___x_654_; 
v___x_653_ = lean_obj_once(&lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5___closed__3, &lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5___closed__3_once, _init_lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5___closed__3);
v___x_654_ = l_String_Slice_Pattern_ForwardSliceSearcher_buildTable(v___x_653_);
return v___x_654_;
}
}
static lean_object* _init_lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5___closed__5(void){
_start:
{
lean_object* v___x_655_; lean_object* v___x_656_; lean_object* v___x_657_; lean_object* v___x_658_; 
v___x_655_ = lean_unsigned_to_nat(0u);
v___x_656_ = lean_obj_once(&lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5___closed__4, &lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5___closed__4_once, _init_lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5___closed__4);
v___x_657_ = lean_obj_once(&lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5___closed__3, &lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5___closed__3_once, _init_lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5___closed__3);
v___x_658_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v___x_658_, 0, v___x_657_);
lean_ctor_set(v___x_658_, 1, v___x_656_);
lean_ctor_set(v___x_658_, 2, v___x_655_);
lean_ctor_set(v___x_658_, 3, v___x_655_);
return v___x_658_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5(lean_object* v_s_661_){
_start:
{
lean_object* v___y_663_; uint8_t v___x_666_; 
v___x_666_ = lean_uint8_once(&lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5___closed__2, &lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5___closed__2_once, _init_lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5___closed__2);
if (v___x_666_ == 0)
{
lean_object* v___x_667_; 
v___x_667_ = lean_obj_once(&lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5___closed__5, &lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5___closed__5_once, _init_lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5___closed__5);
v___y_663_ = v___x_667_;
goto v___jp_662_;
}
else
{
lean_object* v___x_668_; 
v___x_668_ = ((lean_object*)(lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5___closed__6));
v___y_663_ = v___x_668_;
goto v___jp_662_;
}
v___jp_662_:
{
uint8_t v___x_664_; uint8_t v___x_665_; 
v___x_664_ = 0;
lean_inc(v___y_663_);
v___x_665_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5_spec__8___redArg(v_s_661_, v___y_663_, v___x_664_);
return v___x_665_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5___boxed(lean_object* v_s_669_){
_start:
{
uint8_t v_res_670_; lean_object* v_r_671_; 
v_res_670_ = lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5(v_s_669_);
lean_dec_ref(v_s_669_);
v_r_671_ = lean_box(v_res_670_);
return v_r_671_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__1(uint8_t v___x_686_, lean_object* v_s_687_){
_start:
{
lean_object* v___x_700_; 
v___x_700_ = l_Lean_Syntax_getTrailing_x3f(v_s_687_);
if (lean_obj_tag(v___x_700_) == 1)
{
lean_object* v_val_701_; lean_object* v___x_703_; uint8_t v_isShared_704_; uint8_t v_isSharedCheck_736_; 
v_val_701_ = lean_ctor_get(v___x_700_, 0);
v_isSharedCheck_736_ = !lean_is_exclusive(v___x_700_);
if (v_isSharedCheck_736_ == 0)
{
v___x_703_ = v___x_700_;
v_isShared_704_ = v_isSharedCheck_736_;
goto v_resetjp_702_;
}
else
{
lean_inc(v_val_701_);
lean_dec(v___x_700_);
v___x_703_ = lean_box(0);
v_isShared_704_ = v_isSharedCheck_736_;
goto v_resetjp_702_;
}
v_resetjp_702_:
{
uint8_t v___y_706_; lean_object* v___x_731_; lean_object* v___x_732_; uint8_t v___x_733_; 
v___x_731_ = l_Lean_Syntax_getAtomVal(v_s_687_);
v___x_732_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__1___closed__2));
v___x_733_ = lean_string_dec_eq(v___x_731_, v___x_732_);
lean_dec_ref(v___x_731_);
if (v___x_733_ == 0)
{
lean_object* v___x_734_; uint8_t v___x_735_; 
v___x_734_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__1___closed__5));
lean_inc(v_s_687_);
v___x_735_ = l_Lean_Syntax_isOfKind(v_s_687_, v___x_734_);
v___y_706_ = v___x_735_;
goto v___jp_705_;
}
else
{
v___y_706_ = v___x_686_;
goto v___jp_705_;
}
v___jp_705_:
{
if (v___y_706_ == 0)
{
lean_object* v___x_707_; uint8_t v___x_708_; 
v___x_707_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__1___closed__1));
lean_inc(v_s_687_);
v___x_708_ = l_Lean_Syntax_isOfKind(v_s_687_, v___x_707_);
if (v___x_708_ == 0)
{
lean_object* v_str_709_; lean_object* v_startPos_710_; lean_object* v_stopPos_711_; lean_object* v___x_713_; uint8_t v_isShared_714_; uint8_t v_isSharedCheck_730_; 
lean_dec(v_s_687_);
v_str_709_ = lean_ctor_get(v_val_701_, 0);
v_startPos_710_ = lean_ctor_get(v_val_701_, 1);
v_stopPos_711_ = lean_ctor_get(v_val_701_, 2);
v_isSharedCheck_730_ = !lean_is_exclusive(v_val_701_);
if (v_isSharedCheck_730_ == 0)
{
v___x_713_ = v_val_701_;
v_isShared_714_ = v_isSharedCheck_730_;
goto v_resetjp_712_;
}
else
{
lean_inc(v_stopPos_711_);
lean_inc(v_startPos_710_);
lean_inc(v_str_709_);
lean_dec(v_val_701_);
v___x_713_ = lean_box(0);
v_isShared_714_ = v_isSharedCheck_730_;
goto v_resetjp_712_;
}
v_resetjp_712_:
{
lean_object* v_b_715_; lean_object* v_e_716_; lean_object* v___x_717_; lean_object* v___x_718_; lean_object* v___x_719_; lean_object* v___x_720_; uint8_t v___x_721_; 
v_b_715_ = lp_mathlib_Substring_Raw_takeWhileAux___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__3(v_str_709_, v_stopPos_711_, v_startPos_710_);
v_e_716_ = lp_mathlib_Substring_Raw_takeRightWhileAux___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__4(v_str_709_, v_b_715_, v_stopPos_711_);
v___x_717_ = lean_string_utf8_extract(v_str_709_, v_b_715_, v_e_716_);
v___x_718_ = lean_unsigned_to_nat(0u);
v___x_719_ = lean_string_utf8_byte_size(v___x_717_);
v___x_720_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_720_, 0, v___x_717_);
lean_ctor_set(v___x_720_, 1, v___x_718_);
lean_ctor_set(v___x_720_, 2, v___x_719_);
v___x_721_ = lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5(v___x_720_);
lean_dec_ref_known(v___x_720_, 3);
if (v___x_721_ == 0)
{
lean_object* v___x_722_; 
lean_dec(v_e_716_);
lean_dec(v_b_715_);
lean_del_object(v___x_713_);
lean_dec_ref(v_str_709_);
lean_del_object(v___x_703_);
v___x_722_ = lean_box(0);
return v___x_722_;
}
else
{
lean_object* v___x_724_; 
if (v_isShared_714_ == 0)
{
lean_ctor_set(v___x_713_, 2, v_e_716_);
lean_ctor_set(v___x_713_, 1, v_b_715_);
v___x_724_ = v___x_713_;
goto v_reusejp_723_;
}
else
{
lean_object* v_reuseFailAlloc_729_; 
v_reuseFailAlloc_729_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_729_, 0, v_str_709_);
lean_ctor_set(v_reuseFailAlloc_729_, 1, v_b_715_);
lean_ctor_set(v_reuseFailAlloc_729_, 2, v_e_716_);
v___x_724_ = v_reuseFailAlloc_729_;
goto v_reusejp_723_;
}
v_reusejp_723_:
{
lean_object* v___x_725_; lean_object* v___x_727_; 
v___x_725_ = lp_mathlib_Lean_Substring_Raw_getRange(v___x_724_);
lean_dec_ref(v___x_724_);
if (v_isShared_704_ == 0)
{
lean_ctor_set(v___x_703_, 0, v___x_725_);
v___x_727_ = v___x_703_;
goto v_reusejp_726_;
}
else
{
lean_object* v_reuseFailAlloc_728_; 
v_reuseFailAlloc_728_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_728_, 0, v___x_725_);
v___x_727_ = v_reuseFailAlloc_728_;
goto v_reusejp_726_;
}
v_reusejp_726_:
{
return v___x_727_;
}
}
}
}
}
else
{
lean_del_object(v___x_703_);
lean_dec(v_val_701_);
goto v___jp_688_;
}
}
else
{
lean_del_object(v___x_703_);
lean_dec(v_val_701_);
goto v___jp_688_;
}
}
}
}
else
{
lean_object* v___x_737_; 
lean_dec(v___x_700_);
lean_dec(v_s_687_);
v___x_737_ = lean_box(0);
return v___x_737_;
}
v___jp_688_:
{
lean_object* v___x_689_; 
v___x_689_ = l_Lean_Syntax_getTrailing_x3f(v_s_687_);
lean_dec(v_s_687_);
if (lean_obj_tag(v___x_689_) == 0)
{
lean_object* v___x_690_; 
v___x_690_ = lean_box(0);
return v___x_690_;
}
else
{
lean_object* v_val_691_; lean_object* v___x_693_; uint8_t v_isShared_694_; uint8_t v_isSharedCheck_699_; 
v_val_691_ = lean_ctor_get(v___x_689_, 0);
v_isSharedCheck_699_ = !lean_is_exclusive(v___x_689_);
if (v_isSharedCheck_699_ == 0)
{
v___x_693_ = v___x_689_;
v_isShared_694_ = v_isSharedCheck_699_;
goto v_resetjp_692_;
}
else
{
lean_inc(v_val_691_);
lean_dec(v___x_689_);
v___x_693_ = lean_box(0);
v_isShared_694_ = v_isSharedCheck_699_;
goto v_resetjp_692_;
}
v_resetjp_692_:
{
lean_object* v___x_695_; lean_object* v___x_697_; 
v___x_695_ = lp_mathlib_Lean_Substring_Raw_getRange(v_val_691_);
lean_dec(v_val_691_);
if (v_isShared_694_ == 0)
{
lean_ctor_set(v___x_693_, 0, v___x_695_);
v___x_697_ = v___x_693_;
goto v_reusejp_696_;
}
else
{
lean_object* v_reuseFailAlloc_698_; 
v_reuseFailAlloc_698_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_698_, 0, v___x_695_);
v___x_697_ = v_reuseFailAlloc_698_;
goto v_reusejp_696_;
}
v_reusejp_696_:
{
return v___x_697_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__1___boxed(lean_object* v___x_738_, lean_object* v_s_739_){
_start:
{
uint8_t v___x_17735__boxed_740_; lean_object* v_res_741_; 
v___x_17735__boxed_740_ = lean_unbox(v___x_738_);
v_res_741_ = lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__1(v___x_17735__boxed_740_, v_s_739_);
return v_res_741_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_Pos_revSkipWhile___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__8(lean_object* v_s_742_, lean_object* v_pos_743_){
_start:
{
lean_object* v_str_744_; lean_object* v_startInclusive_745_; lean_object* v___x_746_; lean_object* v___x_747_; lean_object* v___x_748_; uint8_t v___x_749_; 
v_str_744_ = lean_ctor_get(v_s_742_, 0);
v_startInclusive_745_ = lean_ctor_get(v_s_742_, 1);
v___x_746_ = lean_nat_add(v_startInclusive_745_, v_pos_743_);
v___x_747_ = lean_nat_sub(v___x_746_, v_startInclusive_745_);
v___x_748_ = lean_unsigned_to_nat(0u);
v___x_749_ = lean_nat_dec_eq(v___x_747_, v___x_748_);
if (v___x_749_ == 0)
{
lean_object* v___x_750_; lean_object* v___x_751_; lean_object* v___x_752_; lean_object* v___x_753_; uint8_t v___y_758_; lean_object* v___x_759_; uint32_t v___x_760_; uint8_t v___y_762_; uint32_t v___x_767_; uint8_t v___x_768_; 
lean_inc(v_startInclusive_745_);
lean_inc_ref(v_str_744_);
v___x_750_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_750_, 0, v_str_744_);
lean_ctor_set(v___x_750_, 1, v_startInclusive_745_);
lean_ctor_set(v___x_750_, 2, v___x_746_);
v___x_751_ = lean_unsigned_to_nat(1u);
v___x_752_ = lean_nat_sub(v___x_747_, v___x_751_);
lean_dec(v___x_747_);
v___x_753_ = l_String_Slice_posLE(v___x_750_, v___x_752_);
lean_dec_ref_known(v___x_750_, 3);
v___x_759_ = lean_nat_add(v_startInclusive_745_, v___x_753_);
v___x_760_ = lean_string_utf8_get_fast(v_str_744_, v___x_759_);
lean_dec(v___x_759_);
v___x_767_ = 32;
v___x_768_ = lean_uint32_dec_eq(v___x_760_, v___x_767_);
if (v___x_768_ == 0)
{
uint32_t v___x_769_; uint8_t v___x_770_; 
v___x_769_ = 9;
v___x_770_ = lean_uint32_dec_eq(v___x_760_, v___x_769_);
v___y_762_ = v___x_770_;
goto v___jp_761_;
}
else
{
v___y_762_ = v___x_768_;
goto v___jp_761_;
}
v___jp_754_:
{
uint8_t v___x_755_; 
v___x_755_ = lean_nat_dec_lt(v___x_753_, v_pos_743_);
if (v___x_755_ == 0)
{
lean_dec(v___x_753_);
return v_pos_743_;
}
else
{
lean_dec(v_pos_743_);
v_pos_743_ = v___x_753_;
goto _start;
}
}
v___jp_757_:
{
if (v___y_758_ == 0)
{
lean_dec(v___x_753_);
return v_pos_743_;
}
else
{
goto v___jp_754_;
}
}
v___jp_761_:
{
if (v___y_762_ == 0)
{
uint32_t v___x_763_; uint8_t v___x_764_; 
v___x_763_ = 13;
v___x_764_ = lean_uint32_dec_eq(v___x_760_, v___x_763_);
if (v___x_764_ == 0)
{
uint32_t v___x_765_; uint8_t v___x_766_; 
v___x_765_ = 10;
v___x_766_ = lean_uint32_dec_eq(v___x_760_, v___x_765_);
v___y_758_ = v___x_766_;
goto v___jp_757_;
}
else
{
v___y_758_ = v___x_764_;
goto v___jp_757_;
}
}
else
{
goto v___jp_754_;
}
}
}
else
{
lean_dec(v___x_747_);
lean_dec(v___x_746_);
return v_pos_743_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_Pos_revSkipWhile___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__8___boxed(lean_object* v_s_771_, lean_object* v_pos_772_){
_start:
{
lean_object* v_res_773_; 
v_res_773_ = lp_mathlib_String_Slice_Pos_revSkipWhile___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__8(v_s_771_, v_pos_772_);
lean_dec_ref(v_s_771_);
return v_res_773_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Lean_PersistentArray_anyM___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__1_spec__3(uint8_t v___x_774_, lean_object* v_as_775_, size_t v_i_776_, size_t v_stop_777_){
_start:
{
uint8_t v___x_778_; 
v___x_778_ = lean_usize_dec_eq(v_i_776_, v_stop_777_);
if (v___x_778_ == 0)
{
lean_object* v___x_779_; uint8_t v_severity_780_; uint8_t v___x_781_; uint8_t v___y_783_; 
v___x_779_ = lean_array_uget_borrowed(v_as_775_, v_i_776_);
v_severity_780_ = lean_ctor_get_uint8(v___x_779_, sizeof(void*)*5 + 1);
v___x_781_ = 1;
if (v_severity_780_ == 0)
{
v___y_783_ = v___x_778_;
goto v___jp_782_;
}
else
{
v___y_783_ = v___x_774_;
goto v___jp_782_;
}
v___jp_782_:
{
if (v___y_783_ == 0)
{
size_t v___x_784_; size_t v___x_785_; 
v___x_784_ = ((size_t)1ULL);
v___x_785_ = lean_usize_add(v_i_776_, v___x_784_);
v_i_776_ = v___x_785_;
goto _start;
}
else
{
return v___x_781_;
}
}
}
else
{
uint8_t v___x_787_; 
v___x_787_ = 0;
return v___x_787_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Lean_PersistentArray_anyM___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__1_spec__3___boxed(lean_object* v___x_788_, lean_object* v_as_789_, lean_object* v_i_790_, lean_object* v_stop_791_){
_start:
{
uint8_t v___x_17898__boxed_792_; size_t v_i_boxed_793_; size_t v_stop_boxed_794_; uint8_t v_res_795_; lean_object* v_r_796_; 
v___x_17898__boxed_792_ = lean_unbox(v___x_788_);
v_i_boxed_793_ = lean_unbox_usize(v_i_790_);
lean_dec(v_i_790_);
v_stop_boxed_794_ = lean_unbox_usize(v_stop_791_);
lean_dec(v_stop_791_);
v_res_795_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Lean_PersistentArray_anyM___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__1_spec__3(v___x_17898__boxed_792_, v_as_789_, v_i_boxed_793_, v_stop_boxed_794_);
lean_dec_ref(v_as_789_);
v_r_796_ = lean_box(v_res_795_);
return v_r_796_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentArray_anyMAux___at___00Lean_PersistentArray_anyM___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__1_spec__2(uint8_t v___x_797_, lean_object* v_x_798_){
_start:
{
if (lean_obj_tag(v_x_798_) == 0)
{
lean_object* v_cs_799_; lean_object* v___x_800_; lean_object* v___x_801_; uint8_t v___x_802_; 
v_cs_799_ = lean_ctor_get(v_x_798_, 0);
v___x_800_ = lean_unsigned_to_nat(0u);
v___x_801_ = lean_array_get_size(v_cs_799_);
v___x_802_ = lean_nat_dec_lt(v___x_800_, v___x_801_);
if (v___x_802_ == 0)
{
return v___x_802_;
}
else
{
if (v___x_802_ == 0)
{
return v___x_802_;
}
else
{
size_t v___x_803_; size_t v___x_804_; uint8_t v___x_805_; 
v___x_803_ = ((size_t)0ULL);
v___x_804_ = lean_usize_of_nat(v___x_801_);
v___x_805_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Lean_PersistentArray_anyMAux___at___00Lean_PersistentArray_anyM___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__1_spec__2_spec__4(v___x_797_, v_cs_799_, v___x_803_, v___x_804_);
return v___x_805_;
}
}
}
else
{
lean_object* v_vs_806_; lean_object* v___x_807_; lean_object* v___x_808_; uint8_t v___x_809_; 
v_vs_806_ = lean_ctor_get(v_x_798_, 0);
v___x_807_ = lean_unsigned_to_nat(0u);
v___x_808_ = lean_array_get_size(v_vs_806_);
v___x_809_ = lean_nat_dec_lt(v___x_807_, v___x_808_);
if (v___x_809_ == 0)
{
return v___x_809_;
}
else
{
if (v___x_809_ == 0)
{
return v___x_809_;
}
else
{
size_t v___x_810_; size_t v___x_811_; uint8_t v___x_812_; 
v___x_810_ = ((size_t)0ULL);
v___x_811_ = lean_usize_of_nat(v___x_808_);
v___x_812_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Lean_PersistentArray_anyM___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__1_spec__3(v___x_797_, v_vs_806_, v___x_810_, v___x_811_);
return v___x_812_;
}
}
}
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Lean_PersistentArray_anyMAux___at___00Lean_PersistentArray_anyM___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__1_spec__2_spec__4(uint8_t v___x_813_, lean_object* v_as_814_, size_t v_i_815_, size_t v_stop_816_){
_start:
{
uint8_t v___x_817_; 
v___x_817_ = lean_usize_dec_eq(v_i_815_, v_stop_816_);
if (v___x_817_ == 0)
{
lean_object* v___x_818_; uint8_t v___x_819_; 
v___x_818_ = lean_array_uget_borrowed(v_as_814_, v_i_815_);
v___x_819_ = lp_mathlib_Lean_PersistentArray_anyMAux___at___00Lean_PersistentArray_anyM___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__1_spec__2(v___x_813_, v___x_818_);
if (v___x_819_ == 0)
{
size_t v___x_820_; size_t v___x_821_; 
v___x_820_ = ((size_t)1ULL);
v___x_821_ = lean_usize_add(v_i_815_, v___x_820_);
v_i_815_ = v___x_821_;
goto _start;
}
else
{
return v___x_819_;
}
}
else
{
uint8_t v___x_823_; 
v___x_823_ = 0;
return v___x_823_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Lean_PersistentArray_anyMAux___at___00Lean_PersistentArray_anyM___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__1_spec__2_spec__4___boxed(lean_object* v___x_824_, lean_object* v_as_825_, lean_object* v_i_826_, lean_object* v_stop_827_){
_start:
{
uint8_t v___x_17919__boxed_828_; size_t v_i_boxed_829_; size_t v_stop_boxed_830_; uint8_t v_res_831_; lean_object* v_r_832_; 
v___x_17919__boxed_828_ = lean_unbox(v___x_824_);
v_i_boxed_829_ = lean_unbox_usize(v_i_826_);
lean_dec(v_i_826_);
v_stop_boxed_830_ = lean_unbox_usize(v_stop_827_);
lean_dec(v_stop_827_);
v_res_831_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Lean_PersistentArray_anyMAux___at___00Lean_PersistentArray_anyM___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__1_spec__2_spec__4(v___x_17919__boxed_828_, v_as_825_, v_i_boxed_829_, v_stop_boxed_830_);
lean_dec_ref(v_as_825_);
v_r_832_ = lean_box(v_res_831_);
return v_r_832_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_anyMAux___at___00Lean_PersistentArray_anyM___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__1_spec__2___boxed(lean_object* v___x_833_, lean_object* v_x_834_){
_start:
{
uint8_t v___x_17927__boxed_835_; uint8_t v_res_836_; lean_object* v_r_837_; 
v___x_17927__boxed_835_ = lean_unbox(v___x_833_);
v_res_836_ = lp_mathlib_Lean_PersistentArray_anyMAux___at___00Lean_PersistentArray_anyM___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__1_spec__2(v___x_17927__boxed_835_, v_x_834_);
lean_dec_ref(v_x_834_);
v_r_837_ = lean_box(v_res_836_);
return v_r_837_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentArray_anyM___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__1(uint8_t v___x_838_, lean_object* v_t_839_){
_start:
{
lean_object* v_root_840_; lean_object* v_tail_841_; uint8_t v___x_842_; 
v_root_840_ = lean_ctor_get(v_t_839_, 0);
v_tail_841_ = lean_ctor_get(v_t_839_, 1);
v___x_842_ = lp_mathlib_Lean_PersistentArray_anyMAux___at___00Lean_PersistentArray_anyM___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__1_spec__2(v___x_838_, v_root_840_);
if (v___x_842_ == 0)
{
lean_object* v___x_843_; lean_object* v___x_844_; uint8_t v___x_845_; 
v___x_843_ = lean_unsigned_to_nat(0u);
v___x_844_ = lean_array_get_size(v_tail_841_);
v___x_845_ = lean_nat_dec_lt(v___x_843_, v___x_844_);
if (v___x_845_ == 0)
{
return v___x_842_;
}
else
{
if (v___x_845_ == 0)
{
return v___x_842_;
}
else
{
size_t v___x_846_; size_t v___x_847_; uint8_t v___x_848_; 
v___x_846_ = ((size_t)0ULL);
v___x_847_ = lean_usize_of_nat(v___x_844_);
v___x_848_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Lean_PersistentArray_anyM___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__1_spec__3(v___x_838_, v_tail_841_, v___x_846_, v___x_847_);
return v___x_848_;
}
}
}
else
{
return v___x_842_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_anyM___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__1___boxed(lean_object* v___x_849_, lean_object* v_t_850_){
_start:
{
uint8_t v___x_17970__boxed_851_; uint8_t v_res_852_; lean_object* v_r_853_; 
v___x_17970__boxed_851_ = lean_unbox(v___x_849_);
v_res_852_ = lp_mathlib_Lean_PersistentArray_anyM___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__1(v___x_17970__boxed_851_, v_t_850_);
lean_dec_ref(v_t_850_);
v_r_853_ = lean_box(v_res_852_);
return v_r_853_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__0_spec__0___redArg(lean_object* v_o_854_, lean_object* v___y_855_){
_start:
{
lean_object* v___x_857_; lean_object* v_env_858_; lean_object* v___x_859_; lean_object* v_toEnvExtension_860_; lean_object* v_asyncMode_861_; lean_object* v___x_862_; lean_object* v___x_863_; lean_object* v___x_864_; lean_object* v_merged_865_; lean_object* v___x_867_; uint8_t v_isShared_868_; uint8_t v_isSharedCheck_873_; 
v___x_857_ = lean_st_ref_get(v___y_855_);
v_env_858_ = lean_ctor_get(v___x_857_, 0);
lean_inc_ref(v_env_858_);
lean_dec(v___x_857_);
v___x_859_ = l_Lean_Linter_linterSetsExt;
v_toEnvExtension_860_ = lean_ctor_get(v___x_859_, 0);
v_asyncMode_861_ = lean_ctor_get(v_toEnvExtension_860_, 2);
v___x_862_ = l_Lean_Linter_instInhabitedLinterSetsState_default;
v___x_863_ = lean_box(0);
v___x_864_ = l_Lean_PersistentEnvExtension_getState___redArg(v___x_862_, v___x_859_, v_env_858_, v_asyncMode_861_, v___x_863_);
v_merged_865_ = lean_ctor_get(v___x_864_, 0);
v_isSharedCheck_873_ = !lean_is_exclusive(v___x_864_);
if (v_isSharedCheck_873_ == 0)
{
lean_object* v_unused_874_; 
v_unused_874_ = lean_ctor_get(v___x_864_, 1);
lean_dec(v_unused_874_);
v___x_867_ = v___x_864_;
v_isShared_868_ = v_isSharedCheck_873_;
goto v_resetjp_866_;
}
else
{
lean_inc(v_merged_865_);
lean_dec(v___x_864_);
v___x_867_ = lean_box(0);
v_isShared_868_ = v_isSharedCheck_873_;
goto v_resetjp_866_;
}
v_resetjp_866_:
{
lean_object* v___x_870_; 
if (v_isShared_868_ == 0)
{
lean_ctor_set(v___x_867_, 1, v_merged_865_);
lean_ctor_set(v___x_867_, 0, v_o_854_);
v___x_870_ = v___x_867_;
goto v_reusejp_869_;
}
else
{
lean_object* v_reuseFailAlloc_872_; 
v_reuseFailAlloc_872_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_872_, 0, v_o_854_);
lean_ctor_set(v_reuseFailAlloc_872_, 1, v_merged_865_);
v___x_870_ = v_reuseFailAlloc_872_;
goto v_reusejp_869_;
}
v_reusejp_869_:
{
lean_object* v___x_871_; 
v___x_871_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_871_, 0, v___x_870_);
return v___x_871_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__0_spec__0___redArg___boxed(lean_object* v_o_875_, lean_object* v___y_876_, lean_object* v___y_877_){
_start:
{
lean_object* v_res_878_; 
v_res_878_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__0_spec__0___redArg(v_o_875_, v___y_876_);
lean_dec(v___y_876_);
return v_res_878_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__0(lean_object* v___y_879_, lean_object* v___y_880_){
_start:
{
lean_object* v___x_882_; lean_object* v_scopes_883_; lean_object* v___x_884_; lean_object* v___x_885_; lean_object* v_opts_886_; lean_object* v___x_887_; 
v___x_882_ = lean_st_ref_get(v___y_880_);
v_scopes_883_ = lean_ctor_get(v___x_882_, 2);
lean_inc(v_scopes_883_);
lean_dec(v___x_882_);
v___x_884_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_885_ = l_List_head_x21___redArg(v___x_884_, v_scopes_883_);
lean_dec(v_scopes_883_);
v_opts_886_ = lean_ctor_get(v___x_885_, 1);
lean_inc_ref(v_opts_886_);
lean_dec(v___x_885_);
v___x_887_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__0_spec__0___redArg(v_opts_886_, v___y_880_);
return v___x_887_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__0___boxed(lean_object* v___y_888_, lean_object* v___y_889_, lean_object* v___y_890_){
_start:
{
lean_object* v_res_891_; 
v_res_891_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__0(v___y_888_, v___y_889_);
lean_dec(v___y_889_);
lean_dec_ref(v___y_888_);
return v_res_891_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__14(lean_object* v_x_892_, lean_object* v_x_893_){
_start:
{
if (lean_obj_tag(v_x_893_) == 0)
{
return v_x_892_;
}
else
{
lean_object* v_head_894_; lean_object* v_tail_895_; lean_object* v___x_896_; 
v_head_894_ = lean_ctor_get(v_x_893_, 0);
v_tail_895_ = lean_ctor_get(v_x_893_, 1);
v___x_896_ = lean_string_append(v_x_892_, v_head_894_);
v_x_892_ = v___x_896_;
v_x_893_ = v_tail_895_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__14___boxed(lean_object* v_x_898_, lean_object* v_x_899_){
_start:
{
lean_object* v_res_900_; 
v_res_900_ = lp_mathlib_List_foldl___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__14(v_x_898_, v_x_899_);
lean_dec(v_x_899_);
return v_res_900_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29_spec__37(lean_object* v_opts_901_, lean_object* v_opt_902_){
_start:
{
lean_object* v_name_903_; lean_object* v_defValue_904_; lean_object* v_map_905_; lean_object* v___x_906_; 
v_name_903_ = lean_ctor_get(v_opt_902_, 0);
v_defValue_904_ = lean_ctor_get(v_opt_902_, 1);
v_map_905_ = lean_ctor_get(v_opts_901_, 0);
v___x_906_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_905_, v_name_903_);
if (lean_obj_tag(v___x_906_) == 0)
{
uint8_t v___x_907_; 
v___x_907_ = lean_unbox(v_defValue_904_);
return v___x_907_;
}
else
{
lean_object* v_val_908_; 
v_val_908_ = lean_ctor_get(v___x_906_, 0);
lean_inc(v_val_908_);
lean_dec_ref_known(v___x_906_, 1);
if (lean_obj_tag(v_val_908_) == 1)
{
uint8_t v_v_909_; 
v_v_909_ = lean_ctor_get_uint8(v_val_908_, 0);
lean_dec_ref_known(v_val_908_, 0);
return v_v_909_;
}
else
{
uint8_t v___x_910_; 
lean_dec(v_val_908_);
v___x_910_ = lean_unbox(v_defValue_904_);
return v___x_910_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29_spec__37___boxed(lean_object* v_opts_911_, lean_object* v_opt_912_){
_start:
{
uint8_t v_res_913_; lean_object* v_r_914_; 
v_res_913_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29_spec__37(v_opts_911_, v_opt_912_);
lean_dec_ref(v_opt_912_);
lean_dec_ref(v_opts_911_);
v_r_914_ = lean_box(v_res_913_);
return v_r_914_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29___lam__0(uint8_t v___y_916_, uint8_t v_suppressElabErrors_917_, lean_object* v_x_918_){
_start:
{
if (lean_obj_tag(v_x_918_) == 1)
{
lean_object* v_pre_919_; 
v_pre_919_ = lean_ctor_get(v_x_918_, 0);
if (lean_obj_tag(v_pre_919_) == 0)
{
lean_object* v_str_920_; lean_object* v___x_921_; uint8_t v___x_922_; 
v_str_920_ = lean_ctor_get(v_x_918_, 1);
v___x_921_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29___lam__0___closed__0));
v___x_922_ = lean_string_dec_eq(v_str_920_, v___x_921_);
if (v___x_922_ == 0)
{
return v___y_916_;
}
else
{
return v_suppressElabErrors_917_;
}
}
else
{
return v___y_916_;
}
}
else
{
return v___y_916_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29___lam__0___boxed(lean_object* v___y_923_, lean_object* v_suppressElabErrors_924_, lean_object* v_x_925_){
_start:
{
uint8_t v___y_18064__boxed_926_; uint8_t v_suppressElabErrors_boxed_927_; uint8_t v_res_928_; lean_object* v_r_929_; 
v___y_18064__boxed_926_ = lean_unbox(v___y_923_);
v_suppressElabErrors_boxed_927_ = lean_unbox(v_suppressElabErrors_924_);
v_res_928_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29___lam__0(v___y_18064__boxed_926_, v_suppressElabErrors_boxed_927_, v_x_925_);
lean_dec(v_x_925_);
v_r_929_ = lean_box(v_res_928_);
return v_r_929_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29_spec__36___redArg___closed__0(void){
_start:
{
lean_object* v___x_930_; 
v___x_930_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_930_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29_spec__36___redArg___closed__1(void){
_start:
{
lean_object* v___x_931_; lean_object* v___x_932_; 
v___x_931_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29_spec__36___redArg___closed__0, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29_spec__36___redArg___closed__0_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29_spec__36___redArg___closed__0);
v___x_932_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_932_, 0, v___x_931_);
return v___x_932_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29_spec__36___redArg___closed__2(void){
_start:
{
lean_object* v___x_933_; lean_object* v___x_934_; lean_object* v___x_935_; 
v___x_933_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29_spec__36___redArg___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29_spec__36___redArg___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29_spec__36___redArg___closed__1);
v___x_934_ = lean_unsigned_to_nat(0u);
v___x_935_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_935_, 0, v___x_934_);
lean_ctor_set(v___x_935_, 1, v___x_934_);
lean_ctor_set(v___x_935_, 2, v___x_934_);
lean_ctor_set(v___x_935_, 3, v___x_934_);
lean_ctor_set(v___x_935_, 4, v___x_933_);
lean_ctor_set(v___x_935_, 5, v___x_933_);
lean_ctor_set(v___x_935_, 6, v___x_933_);
lean_ctor_set(v___x_935_, 7, v___x_933_);
lean_ctor_set(v___x_935_, 8, v___x_933_);
lean_ctor_set(v___x_935_, 9, v___x_933_);
return v___x_935_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29_spec__36___redArg___closed__3(void){
_start:
{
lean_object* v___x_936_; lean_object* v___x_937_; lean_object* v___x_938_; 
v___x_936_ = lean_unsigned_to_nat(32u);
v___x_937_ = lean_mk_empty_array_with_capacity(v___x_936_);
v___x_938_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_938_, 0, v___x_937_);
return v___x_938_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29_spec__36___redArg___closed__4(void){
_start:
{
size_t v___x_939_; lean_object* v___x_940_; lean_object* v___x_941_; lean_object* v___x_942_; lean_object* v___x_943_; lean_object* v___x_944_; 
v___x_939_ = ((size_t)5ULL);
v___x_940_ = lean_unsigned_to_nat(0u);
v___x_941_ = lean_unsigned_to_nat(32u);
v___x_942_ = lean_mk_empty_array_with_capacity(v___x_941_);
v___x_943_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29_spec__36___redArg___closed__3, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29_spec__36___redArg___closed__3_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29_spec__36___redArg___closed__3);
v___x_944_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_944_, 0, v___x_943_);
lean_ctor_set(v___x_944_, 1, v___x_942_);
lean_ctor_set(v___x_944_, 2, v___x_940_);
lean_ctor_set(v___x_944_, 3, v___x_940_);
lean_ctor_set_usize(v___x_944_, 4, v___x_939_);
return v___x_944_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29_spec__36___redArg___closed__5(void){
_start:
{
lean_object* v___x_945_; lean_object* v___x_946_; lean_object* v___x_947_; lean_object* v___x_948_; 
v___x_945_ = lean_box(1);
v___x_946_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29_spec__36___redArg___closed__4, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29_spec__36___redArg___closed__4_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29_spec__36___redArg___closed__4);
v___x_947_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29_spec__36___redArg___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29_spec__36___redArg___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29_spec__36___redArg___closed__1);
v___x_948_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_948_, 0, v___x_947_);
lean_ctor_set(v___x_948_, 1, v___x_946_);
lean_ctor_set(v___x_948_, 2, v___x_945_);
return v___x_948_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29_spec__36___redArg(lean_object* v_msgData_949_, lean_object* v___y_950_){
_start:
{
lean_object* v___x_952_; lean_object* v_env_953_; lean_object* v___x_954_; lean_object* v_scopes_955_; lean_object* v___x_956_; lean_object* v___x_957_; lean_object* v_opts_958_; lean_object* v___x_959_; lean_object* v___x_960_; lean_object* v___x_961_; lean_object* v___x_962_; lean_object* v___x_963_; 
v___x_952_ = lean_st_ref_get(v___y_950_);
v_env_953_ = lean_ctor_get(v___x_952_, 0);
lean_inc_ref(v_env_953_);
lean_dec(v___x_952_);
v___x_954_ = lean_st_ref_get(v___y_950_);
v_scopes_955_ = lean_ctor_get(v___x_954_, 2);
lean_inc(v_scopes_955_);
lean_dec(v___x_954_);
v___x_956_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_957_ = l_List_head_x21___redArg(v___x_956_, v_scopes_955_);
lean_dec(v_scopes_955_);
v_opts_958_ = lean_ctor_get(v___x_957_, 1);
lean_inc_ref(v_opts_958_);
lean_dec(v___x_957_);
v___x_959_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29_spec__36___redArg___closed__2, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29_spec__36___redArg___closed__2_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29_spec__36___redArg___closed__2);
v___x_960_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29_spec__36___redArg___closed__5, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29_spec__36___redArg___closed__5_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29_spec__36___redArg___closed__5);
v___x_961_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_961_, 0, v_env_953_);
lean_ctor_set(v___x_961_, 1, v___x_959_);
lean_ctor_set(v___x_961_, 2, v___x_960_);
lean_ctor_set(v___x_961_, 3, v_opts_958_);
v___x_962_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_962_, 0, v___x_961_);
lean_ctor_set(v___x_962_, 1, v_msgData_949_);
v___x_963_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_963_, 0, v___x_962_);
return v___x_963_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29_spec__36___redArg___boxed(lean_object* v_msgData_964_, lean_object* v___y_965_, lean_object* v___y_966_){
_start:
{
lean_object* v_res_967_; 
v_res_967_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29_spec__36___redArg(v_msgData_964_, v___y_965_);
lean_dec(v___y_965_);
return v_res_967_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29(lean_object* v_ref_969_, lean_object* v_msgData_970_, uint8_t v_severity_971_, uint8_t v_isSilent_972_, lean_object* v___y_973_, lean_object* v___y_974_){
_start:
{
uint8_t v___y_977_; uint8_t v___y_978_; lean_object* v___y_979_; lean_object* v___y_980_; lean_object* v___y_981_; lean_object* v___y_982_; lean_object* v___y_983_; lean_object* v___y_984_; uint8_t v___y_1041_; uint8_t v___y_1042_; uint8_t v___y_1043_; lean_object* v___y_1044_; lean_object* v___y_1045_; uint8_t v___y_1069_; uint8_t v___y_1070_; uint8_t v___y_1071_; lean_object* v___y_1072_; lean_object* v___y_1073_; uint8_t v___y_1077_; uint8_t v___y_1078_; uint8_t v___y_1079_; uint8_t v___x_1094_; uint8_t v___y_1096_; uint8_t v___y_1097_; uint8_t v___y_1098_; uint8_t v___y_1100_; uint8_t v___x_1112_; 
v___x_1094_ = 2;
v___x_1112_ = l_Lean_instBEqMessageSeverity_beq(v_severity_971_, v___x_1094_);
if (v___x_1112_ == 0)
{
v___y_1100_ = v___x_1112_;
goto v___jp_1099_;
}
else
{
uint8_t v___x_1113_; 
lean_inc_ref(v_msgData_970_);
v___x_1113_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_970_);
v___y_1100_ = v___x_1113_;
goto v___jp_1099_;
}
v___jp_976_:
{
lean_object* v___x_985_; 
v___x_985_ = l_Lean_Elab_Command_getScope___redArg(v___y_984_);
if (lean_obj_tag(v___x_985_) == 0)
{
lean_object* v_a_986_; lean_object* v___x_987_; 
v_a_986_ = lean_ctor_get(v___x_985_, 0);
lean_inc(v_a_986_);
lean_dec_ref_known(v___x_985_, 1);
v___x_987_ = l_Lean_Elab_Command_getScope___redArg(v___y_984_);
if (lean_obj_tag(v___x_987_) == 0)
{
lean_object* v_a_988_; lean_object* v___x_990_; uint8_t v_isShared_991_; uint8_t v_isSharedCheck_1023_; 
v_a_988_ = lean_ctor_get(v___x_987_, 0);
v_isSharedCheck_1023_ = !lean_is_exclusive(v___x_987_);
if (v_isSharedCheck_1023_ == 0)
{
v___x_990_ = v___x_987_;
v_isShared_991_ = v_isSharedCheck_1023_;
goto v_resetjp_989_;
}
else
{
lean_inc(v_a_988_);
lean_dec(v___x_987_);
v___x_990_ = lean_box(0);
v_isShared_991_ = v_isSharedCheck_1023_;
goto v_resetjp_989_;
}
v_resetjp_989_:
{
lean_object* v___x_992_; lean_object* v_currNamespace_993_; lean_object* v_openDecls_994_; lean_object* v_env_995_; lean_object* v_messages_996_; lean_object* v_scopes_997_; lean_object* v_usedQuotCtxts_998_; lean_object* v_nextMacroScope_999_; lean_object* v_maxRecDepth_1000_; lean_object* v_ngen_1001_; lean_object* v_auxDeclNGen_1002_; lean_object* v_infoState_1003_; lean_object* v_traceState_1004_; lean_object* v_snapshotTasks_1005_; lean_object* v_prevLinterStates_1006_; lean_object* v___x_1008_; uint8_t v_isShared_1009_; uint8_t v_isSharedCheck_1022_; 
v___x_992_ = lean_st_ref_take(v___y_984_);
v_currNamespace_993_ = lean_ctor_get(v_a_986_, 2);
lean_inc(v_currNamespace_993_);
lean_dec(v_a_986_);
v_openDecls_994_ = lean_ctor_get(v_a_988_, 3);
lean_inc(v_openDecls_994_);
lean_dec(v_a_988_);
v_env_995_ = lean_ctor_get(v___x_992_, 0);
v_messages_996_ = lean_ctor_get(v___x_992_, 1);
v_scopes_997_ = lean_ctor_get(v___x_992_, 2);
v_usedQuotCtxts_998_ = lean_ctor_get(v___x_992_, 3);
v_nextMacroScope_999_ = lean_ctor_get(v___x_992_, 4);
v_maxRecDepth_1000_ = lean_ctor_get(v___x_992_, 5);
v_ngen_1001_ = lean_ctor_get(v___x_992_, 6);
v_auxDeclNGen_1002_ = lean_ctor_get(v___x_992_, 7);
v_infoState_1003_ = lean_ctor_get(v___x_992_, 8);
v_traceState_1004_ = lean_ctor_get(v___x_992_, 9);
v_snapshotTasks_1005_ = lean_ctor_get(v___x_992_, 10);
v_prevLinterStates_1006_ = lean_ctor_get(v___x_992_, 11);
v_isSharedCheck_1022_ = !lean_is_exclusive(v___x_992_);
if (v_isSharedCheck_1022_ == 0)
{
v___x_1008_ = v___x_992_;
v_isShared_1009_ = v_isSharedCheck_1022_;
goto v_resetjp_1007_;
}
else
{
lean_inc(v_prevLinterStates_1006_);
lean_inc(v_snapshotTasks_1005_);
lean_inc(v_traceState_1004_);
lean_inc(v_infoState_1003_);
lean_inc(v_auxDeclNGen_1002_);
lean_inc(v_ngen_1001_);
lean_inc(v_maxRecDepth_1000_);
lean_inc(v_nextMacroScope_999_);
lean_inc(v_usedQuotCtxts_998_);
lean_inc(v_scopes_997_);
lean_inc(v_messages_996_);
lean_inc(v_env_995_);
lean_dec(v___x_992_);
v___x_1008_ = lean_box(0);
v_isShared_1009_ = v_isSharedCheck_1022_;
goto v_resetjp_1007_;
}
v_resetjp_1007_:
{
lean_object* v___x_1010_; lean_object* v___x_1011_; lean_object* v___x_1012_; lean_object* v___x_1013_; lean_object* v___x_1015_; 
v___x_1010_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1010_, 0, v_currNamespace_993_);
lean_ctor_set(v___x_1010_, 1, v_openDecls_994_);
v___x_1011_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1011_, 0, v___x_1010_);
lean_ctor_set(v___x_1011_, 1, v___y_981_);
lean_inc_ref(v___y_982_);
lean_inc_ref(v___y_980_);
v___x_1012_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_1012_, 0, v___y_980_);
lean_ctor_set(v___x_1012_, 1, v___y_983_);
lean_ctor_set(v___x_1012_, 2, v___y_979_);
lean_ctor_set(v___x_1012_, 3, v___y_982_);
lean_ctor_set(v___x_1012_, 4, v___x_1011_);
lean_ctor_set_uint8(v___x_1012_, sizeof(void*)*5, v___y_978_);
lean_ctor_set_uint8(v___x_1012_, sizeof(void*)*5 + 1, v___y_977_);
lean_ctor_set_uint8(v___x_1012_, sizeof(void*)*5 + 2, v_isSilent_972_);
v___x_1013_ = l_Lean_MessageLog_add(v___x_1012_, v_messages_996_);
if (v_isShared_1009_ == 0)
{
lean_ctor_set(v___x_1008_, 1, v___x_1013_);
v___x_1015_ = v___x_1008_;
goto v_reusejp_1014_;
}
else
{
lean_object* v_reuseFailAlloc_1021_; 
v_reuseFailAlloc_1021_ = lean_alloc_ctor(0, 12, 0);
lean_ctor_set(v_reuseFailAlloc_1021_, 0, v_env_995_);
lean_ctor_set(v_reuseFailAlloc_1021_, 1, v___x_1013_);
lean_ctor_set(v_reuseFailAlloc_1021_, 2, v_scopes_997_);
lean_ctor_set(v_reuseFailAlloc_1021_, 3, v_usedQuotCtxts_998_);
lean_ctor_set(v_reuseFailAlloc_1021_, 4, v_nextMacroScope_999_);
lean_ctor_set(v_reuseFailAlloc_1021_, 5, v_maxRecDepth_1000_);
lean_ctor_set(v_reuseFailAlloc_1021_, 6, v_ngen_1001_);
lean_ctor_set(v_reuseFailAlloc_1021_, 7, v_auxDeclNGen_1002_);
lean_ctor_set(v_reuseFailAlloc_1021_, 8, v_infoState_1003_);
lean_ctor_set(v_reuseFailAlloc_1021_, 9, v_traceState_1004_);
lean_ctor_set(v_reuseFailAlloc_1021_, 10, v_snapshotTasks_1005_);
lean_ctor_set(v_reuseFailAlloc_1021_, 11, v_prevLinterStates_1006_);
v___x_1015_ = v_reuseFailAlloc_1021_;
goto v_reusejp_1014_;
}
v_reusejp_1014_:
{
lean_object* v___x_1016_; lean_object* v___x_1017_; lean_object* v___x_1019_; 
v___x_1016_ = lean_st_ref_set(v___y_984_, v___x_1015_);
v___x_1017_ = lean_box(0);
if (v_isShared_991_ == 0)
{
lean_ctor_set(v___x_990_, 0, v___x_1017_);
v___x_1019_ = v___x_990_;
goto v_reusejp_1018_;
}
else
{
lean_object* v_reuseFailAlloc_1020_; 
v_reuseFailAlloc_1020_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1020_, 0, v___x_1017_);
v___x_1019_ = v_reuseFailAlloc_1020_;
goto v_reusejp_1018_;
}
v_reusejp_1018_:
{
return v___x_1019_;
}
}
}
}
}
else
{
lean_object* v_a_1024_; lean_object* v___x_1026_; uint8_t v_isShared_1027_; uint8_t v_isSharedCheck_1031_; 
lean_dec(v_a_986_);
lean_dec_ref(v___y_983_);
lean_dec_ref(v___y_981_);
lean_dec(v___y_979_);
v_a_1024_ = lean_ctor_get(v___x_987_, 0);
v_isSharedCheck_1031_ = !lean_is_exclusive(v___x_987_);
if (v_isSharedCheck_1031_ == 0)
{
v___x_1026_ = v___x_987_;
v_isShared_1027_ = v_isSharedCheck_1031_;
goto v_resetjp_1025_;
}
else
{
lean_inc(v_a_1024_);
lean_dec(v___x_987_);
v___x_1026_ = lean_box(0);
v_isShared_1027_ = v_isSharedCheck_1031_;
goto v_resetjp_1025_;
}
v_resetjp_1025_:
{
lean_object* v___x_1029_; 
if (v_isShared_1027_ == 0)
{
v___x_1029_ = v___x_1026_;
goto v_reusejp_1028_;
}
else
{
lean_object* v_reuseFailAlloc_1030_; 
v_reuseFailAlloc_1030_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1030_, 0, v_a_1024_);
v___x_1029_ = v_reuseFailAlloc_1030_;
goto v_reusejp_1028_;
}
v_reusejp_1028_:
{
return v___x_1029_;
}
}
}
}
else
{
lean_object* v_a_1032_; lean_object* v___x_1034_; uint8_t v_isShared_1035_; uint8_t v_isSharedCheck_1039_; 
lean_dec_ref(v___y_983_);
lean_dec_ref(v___y_981_);
lean_dec(v___y_979_);
v_a_1032_ = lean_ctor_get(v___x_985_, 0);
v_isSharedCheck_1039_ = !lean_is_exclusive(v___x_985_);
if (v_isSharedCheck_1039_ == 0)
{
v___x_1034_ = v___x_985_;
v_isShared_1035_ = v_isSharedCheck_1039_;
goto v_resetjp_1033_;
}
else
{
lean_inc(v_a_1032_);
lean_dec(v___x_985_);
v___x_1034_ = lean_box(0);
v_isShared_1035_ = v_isSharedCheck_1039_;
goto v_resetjp_1033_;
}
v_resetjp_1033_:
{
lean_object* v___x_1037_; 
if (v_isShared_1035_ == 0)
{
v___x_1037_ = v___x_1034_;
goto v_reusejp_1036_;
}
else
{
lean_object* v_reuseFailAlloc_1038_; 
v_reuseFailAlloc_1038_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1038_, 0, v_a_1032_);
v___x_1037_ = v_reuseFailAlloc_1038_;
goto v_reusejp_1036_;
}
v_reusejp_1036_:
{
return v___x_1037_;
}
}
}
}
v___jp_1040_:
{
lean_object* v_fileName_1046_; lean_object* v_fileMap_1047_; uint8_t v_suppressElabErrors_1048_; lean_object* v___x_1049_; lean_object* v___x_1050_; lean_object* v_a_1051_; lean_object* v___x_1053_; uint8_t v_isShared_1054_; uint8_t v_isSharedCheck_1067_; 
v_fileName_1046_ = lean_ctor_get(v___y_973_, 0);
v_fileMap_1047_ = lean_ctor_get(v___y_973_, 1);
v_suppressElabErrors_1048_ = lean_ctor_get_uint8(v___y_973_, sizeof(void*)*10);
v___x_1049_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_970_);
v___x_1050_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29_spec__36___redArg(v___x_1049_, v___y_974_);
v_a_1051_ = lean_ctor_get(v___x_1050_, 0);
v_isSharedCheck_1067_ = !lean_is_exclusive(v___x_1050_);
if (v_isSharedCheck_1067_ == 0)
{
v___x_1053_ = v___x_1050_;
v_isShared_1054_ = v_isSharedCheck_1067_;
goto v_resetjp_1052_;
}
else
{
lean_inc(v_a_1051_);
lean_dec(v___x_1050_);
v___x_1053_ = lean_box(0);
v_isShared_1054_ = v_isSharedCheck_1067_;
goto v_resetjp_1052_;
}
v_resetjp_1052_:
{
lean_object* v___x_1055_; lean_object* v___x_1056_; lean_object* v___x_1057_; lean_object* v___x_1058_; 
lean_inc_ref_n(v_fileMap_1047_, 2);
v___x_1055_ = l_Lean_FileMap_toPosition(v_fileMap_1047_, v___y_1044_);
lean_dec(v___y_1044_);
v___x_1056_ = l_Lean_FileMap_toPosition(v_fileMap_1047_, v___y_1045_);
lean_dec(v___y_1045_);
v___x_1057_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1057_, 0, v___x_1056_);
v___x_1058_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29___closed__0));
if (v_suppressElabErrors_1048_ == 0)
{
lean_del_object(v___x_1053_);
v___y_977_ = v___y_1042_;
v___y_978_ = v___y_1043_;
v___y_979_ = v___x_1057_;
v___y_980_ = v_fileName_1046_;
v___y_981_ = v_a_1051_;
v___y_982_ = v___x_1058_;
v___y_983_ = v___x_1055_;
v___y_984_ = v___y_974_;
goto v___jp_976_;
}
else
{
lean_object* v___x_1059_; lean_object* v___x_1060_; lean_object* v___f_1061_; uint8_t v___x_1062_; 
v___x_1059_ = lean_box(v___y_1041_);
v___x_1060_ = lean_box(v_suppressElabErrors_1048_);
v___f_1061_ = lean_alloc_closure((void*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29___lam__0___boxed), 3, 2);
lean_closure_set(v___f_1061_, 0, v___x_1059_);
lean_closure_set(v___f_1061_, 1, v___x_1060_);
lean_inc(v_a_1051_);
v___x_1062_ = l_Lean_MessageData_hasTag(v___f_1061_, v_a_1051_);
if (v___x_1062_ == 0)
{
lean_object* v___x_1063_; lean_object* v___x_1065_; 
lean_dec_ref_known(v___x_1057_, 1);
lean_dec_ref(v___x_1055_);
lean_dec(v_a_1051_);
v___x_1063_ = lean_box(0);
if (v_isShared_1054_ == 0)
{
lean_ctor_set(v___x_1053_, 0, v___x_1063_);
v___x_1065_ = v___x_1053_;
goto v_reusejp_1064_;
}
else
{
lean_object* v_reuseFailAlloc_1066_; 
v_reuseFailAlloc_1066_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1066_, 0, v___x_1063_);
v___x_1065_ = v_reuseFailAlloc_1066_;
goto v_reusejp_1064_;
}
v_reusejp_1064_:
{
return v___x_1065_;
}
}
else
{
lean_del_object(v___x_1053_);
v___y_977_ = v___y_1042_;
v___y_978_ = v___y_1043_;
v___y_979_ = v___x_1057_;
v___y_980_ = v_fileName_1046_;
v___y_981_ = v_a_1051_;
v___y_982_ = v___x_1058_;
v___y_983_ = v___x_1055_;
v___y_984_ = v___y_974_;
goto v___jp_976_;
}
}
}
}
v___jp_1068_:
{
lean_object* v___x_1074_; 
v___x_1074_ = l_Lean_Syntax_getTailPos_x3f(v___y_1072_, v___y_1071_);
lean_dec(v___y_1072_);
if (lean_obj_tag(v___x_1074_) == 0)
{
lean_inc(v___y_1073_);
v___y_1041_ = v___y_1069_;
v___y_1042_ = v___y_1070_;
v___y_1043_ = v___y_1071_;
v___y_1044_ = v___y_1073_;
v___y_1045_ = v___y_1073_;
goto v___jp_1040_;
}
else
{
lean_object* v_val_1075_; 
v_val_1075_ = lean_ctor_get(v___x_1074_, 0);
lean_inc(v_val_1075_);
lean_dec_ref_known(v___x_1074_, 1);
v___y_1041_ = v___y_1069_;
v___y_1042_ = v___y_1070_;
v___y_1043_ = v___y_1071_;
v___y_1044_ = v___y_1073_;
v___y_1045_ = v_val_1075_;
goto v___jp_1040_;
}
}
v___jp_1076_:
{
lean_object* v___x_1080_; 
v___x_1080_ = l_Lean_Elab_Command_getRef___redArg(v___y_973_);
if (lean_obj_tag(v___x_1080_) == 0)
{
lean_object* v_a_1081_; lean_object* v_ref_1082_; lean_object* v___x_1083_; 
v_a_1081_ = lean_ctor_get(v___x_1080_, 0);
lean_inc(v_a_1081_);
lean_dec_ref_known(v___x_1080_, 1);
v_ref_1082_ = l_Lean_replaceRef(v_ref_969_, v_a_1081_);
lean_dec(v_a_1081_);
v___x_1083_ = l_Lean_Syntax_getPos_x3f(v_ref_1082_, v___y_1078_);
if (lean_obj_tag(v___x_1083_) == 0)
{
lean_object* v___x_1084_; 
v___x_1084_ = lean_unsigned_to_nat(0u);
v___y_1069_ = v___y_1077_;
v___y_1070_ = v___y_1079_;
v___y_1071_ = v___y_1078_;
v___y_1072_ = v_ref_1082_;
v___y_1073_ = v___x_1084_;
goto v___jp_1068_;
}
else
{
lean_object* v_val_1085_; 
v_val_1085_ = lean_ctor_get(v___x_1083_, 0);
lean_inc(v_val_1085_);
lean_dec_ref_known(v___x_1083_, 1);
v___y_1069_ = v___y_1077_;
v___y_1070_ = v___y_1079_;
v___y_1071_ = v___y_1078_;
v___y_1072_ = v_ref_1082_;
v___y_1073_ = v_val_1085_;
goto v___jp_1068_;
}
}
else
{
lean_object* v_a_1086_; lean_object* v___x_1088_; uint8_t v_isShared_1089_; uint8_t v_isSharedCheck_1093_; 
lean_dec_ref(v_msgData_970_);
v_a_1086_ = lean_ctor_get(v___x_1080_, 0);
v_isSharedCheck_1093_ = !lean_is_exclusive(v___x_1080_);
if (v_isSharedCheck_1093_ == 0)
{
v___x_1088_ = v___x_1080_;
v_isShared_1089_ = v_isSharedCheck_1093_;
goto v_resetjp_1087_;
}
else
{
lean_inc(v_a_1086_);
lean_dec(v___x_1080_);
v___x_1088_ = lean_box(0);
v_isShared_1089_ = v_isSharedCheck_1093_;
goto v_resetjp_1087_;
}
v_resetjp_1087_:
{
lean_object* v___x_1091_; 
if (v_isShared_1089_ == 0)
{
v___x_1091_ = v___x_1088_;
goto v_reusejp_1090_;
}
else
{
lean_object* v_reuseFailAlloc_1092_; 
v_reuseFailAlloc_1092_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1092_, 0, v_a_1086_);
v___x_1091_ = v_reuseFailAlloc_1092_;
goto v_reusejp_1090_;
}
v_reusejp_1090_:
{
return v___x_1091_;
}
}
}
}
v___jp_1095_:
{
if (v___y_1098_ == 0)
{
v___y_1077_ = v___y_1096_;
v___y_1078_ = v___y_1097_;
v___y_1079_ = v_severity_971_;
goto v___jp_1076_;
}
else
{
v___y_1077_ = v___y_1096_;
v___y_1078_ = v___y_1097_;
v___y_1079_ = v___x_1094_;
goto v___jp_1076_;
}
}
v___jp_1099_:
{
if (v___y_1100_ == 0)
{
lean_object* v___x_1101_; lean_object* v_scopes_1102_; lean_object* v___x_1103_; lean_object* v___x_1104_; lean_object* v_opts_1105_; uint8_t v___x_1106_; uint8_t v___x_1107_; 
v___x_1101_ = lean_st_ref_get(v___y_974_);
v_scopes_1102_ = lean_ctor_get(v___x_1101_, 2);
lean_inc(v_scopes_1102_);
lean_dec(v___x_1101_);
v___x_1103_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_1104_ = l_List_head_x21___redArg(v___x_1103_, v_scopes_1102_);
lean_dec(v_scopes_1102_);
v_opts_1105_ = lean_ctor_get(v___x_1104_, 1);
lean_inc_ref(v_opts_1105_);
lean_dec(v___x_1104_);
v___x_1106_ = 1;
v___x_1107_ = l_Lean_instBEqMessageSeverity_beq(v_severity_971_, v___x_1106_);
if (v___x_1107_ == 0)
{
lean_dec_ref(v_opts_1105_);
v___y_1096_ = v___y_1100_;
v___y_1097_ = v___y_1100_;
v___y_1098_ = v___x_1107_;
goto v___jp_1095_;
}
else
{
lean_object* v___x_1108_; uint8_t v___x_1109_; 
v___x_1108_ = l_Lean_warningAsError;
v___x_1109_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29_spec__37(v_opts_1105_, v___x_1108_);
lean_dec_ref(v_opts_1105_);
v___y_1096_ = v___y_1100_;
v___y_1097_ = v___y_1100_;
v___y_1098_ = v___x_1109_;
goto v___jp_1095_;
}
}
else
{
lean_object* v___x_1110_; lean_object* v___x_1111_; 
lean_dec_ref(v_msgData_970_);
v___x_1110_ = lean_box(0);
v___x_1111_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1111_, 0, v___x_1110_);
return v___x_1111_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29___boxed(lean_object* v_ref_1114_, lean_object* v_msgData_1115_, lean_object* v_severity_1116_, lean_object* v_isSilent_1117_, lean_object* v___y_1118_, lean_object* v___y_1119_, lean_object* v___y_1120_){
_start:
{
uint8_t v_severity_boxed_1121_; uint8_t v_isSilent_boxed_1122_; lean_object* v_res_1123_; 
v_severity_boxed_1121_ = lean_unbox(v_severity_1116_);
v_isSilent_boxed_1122_ = lean_unbox(v_isSilent_1117_);
v_res_1123_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29(v_ref_1114_, v_msgData_1115_, v_severity_boxed_1121_, v_isSilent_boxed_1122_, v___y_1118_, v___y_1119_);
lean_dec(v___y_1119_);
lean_dec_ref(v___y_1118_);
lean_dec(v_ref_1114_);
return v_res_1123_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24(lean_object* v_ref_1124_, lean_object* v_msgData_1125_, lean_object* v___y_1126_, lean_object* v___y_1127_){
_start:
{
uint8_t v___x_1129_; uint8_t v___x_1130_; lean_object* v___x_1131_; 
v___x_1129_ = 1;
v___x_1130_ = 0;
v___x_1131_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29(v_ref_1124_, v_msgData_1125_, v___x_1129_, v___x_1130_, v___y_1126_, v___y_1127_);
return v___x_1131_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24___boxed(lean_object* v_ref_1132_, lean_object* v_msgData_1133_, lean_object* v___y_1134_, lean_object* v___y_1135_, lean_object* v___y_1136_){
_start:
{
lean_object* v_res_1137_; 
v_res_1137_ = lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24(v_ref_1132_, v_msgData_1133_, v___y_1134_, v___y_1135_);
lean_dec(v___y_1135_);
lean_dec_ref(v___y_1134_);
lean_dec(v_ref_1132_);
return v_res_1137_;
}
}
static lean_object* _init_lp_mathlib_Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15___closed__1(void){
_start:
{
lean_object* v___x_1139_; lean_object* v___x_1140_; 
v___x_1139_ = ((lean_object*)(lp_mathlib_Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15___closed__0));
v___x_1140_ = l_Lean_stringToMessageData(v___x_1139_);
return v___x_1140_;
}
}
static lean_object* _init_lp_mathlib_Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15___closed__3(void){
_start:
{
lean_object* v___x_1142_; lean_object* v___x_1143_; 
v___x_1142_ = ((lean_object*)(lp_mathlib_Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15___closed__2));
v___x_1143_ = l_Lean_stringToMessageData(v___x_1142_);
return v___x_1143_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15(lean_object* v_linterOption_1144_, lean_object* v_stx_1145_, lean_object* v_msg_1146_, lean_object* v___y_1147_, lean_object* v___y_1148_){
_start:
{
lean_object* v_name_1150_; lean_object* v___x_1152_; uint8_t v_isShared_1153_; uint8_t v_isSharedCheck_1168_; 
v_name_1150_ = lean_ctor_get(v_linterOption_1144_, 0);
v_isSharedCheck_1168_ = !lean_is_exclusive(v_linterOption_1144_);
if (v_isSharedCheck_1168_ == 0)
{
lean_object* v_unused_1169_; 
v_unused_1169_ = lean_ctor_get(v_linterOption_1144_, 1);
lean_dec(v_unused_1169_);
v___x_1152_ = v_linterOption_1144_;
v_isShared_1153_ = v_isSharedCheck_1168_;
goto v_resetjp_1151_;
}
else
{
lean_inc(v_name_1150_);
lean_dec(v_linterOption_1144_);
v___x_1152_ = lean_box(0);
v_isShared_1153_ = v_isSharedCheck_1168_;
goto v_resetjp_1151_;
}
v_resetjp_1151_:
{
lean_object* v___x_1154_; lean_object* v___x_1155_; lean_object* v___x_1157_; 
v___x_1154_ = lean_obj_once(&lp_mathlib_Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15___closed__1, &lp_mathlib_Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15___closed__1_once, _init_lp_mathlib_Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15___closed__1);
lean_inc(v_name_1150_);
v___x_1155_ = l_Lean_MessageData_ofName(v_name_1150_);
if (v_isShared_1153_ == 0)
{
lean_ctor_set_tag(v___x_1152_, 7);
lean_ctor_set(v___x_1152_, 1, v___x_1155_);
lean_ctor_set(v___x_1152_, 0, v___x_1154_);
v___x_1157_ = v___x_1152_;
goto v_reusejp_1156_;
}
else
{
lean_object* v_reuseFailAlloc_1167_; 
v_reuseFailAlloc_1167_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1167_, 0, v___x_1154_);
lean_ctor_set(v_reuseFailAlloc_1167_, 1, v___x_1155_);
v___x_1157_ = v_reuseFailAlloc_1167_;
goto v_reusejp_1156_;
}
v_reusejp_1156_:
{
lean_object* v___x_1158_; lean_object* v___x_1159_; lean_object* v_disable_1160_; lean_object* v___x_1161_; lean_object* v___x_1162_; lean_object* v___x_1163_; lean_object* v___x_1164_; lean_object* v___x_1165_; lean_object* v___x_1166_; 
v___x_1158_ = lean_obj_once(&lp_mathlib_Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15___closed__3, &lp_mathlib_Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15___closed__3_once, _init_lp_mathlib_Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15___closed__3);
v___x_1159_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1159_, 0, v___x_1157_);
lean_ctor_set(v___x_1159_, 1, v___x_1158_);
v_disable_1160_ = l_Lean_MessageData_note(v___x_1159_);
v___x_1161_ = l_Lean_Linter_linterMessageTag;
v___x_1162_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1162_, 0, v_msg_1146_);
lean_ctor_set(v___x_1162_, 1, v_disable_1160_);
v___x_1163_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_1163_, 0, v___x_1161_);
lean_ctor_set(v___x_1163_, 1, v___x_1162_);
v___x_1164_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_1164_, 0, v_name_1150_);
lean_ctor_set(v___x_1164_, 1, v___x_1163_);
lean_inc(v_stx_1145_);
v___x_1165_ = lean_alloc_ctor(11, 2, 0);
lean_ctor_set(v___x_1165_, 0, v_stx_1145_);
lean_ctor_set(v___x_1165_, 1, v___x_1164_);
v___x_1166_ = lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24(v_stx_1145_, v___x_1165_, v___y_1147_, v___y_1148_);
lean_dec(v_stx_1145_);
return v___x_1166_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15___boxed(lean_object* v_linterOption_1170_, lean_object* v_stx_1171_, lean_object* v_msg_1172_, lean_object* v___y_1173_, lean_object* v___y_1174_, lean_object* v___y_1175_){
_start:
{
lean_object* v_res_1176_; 
v_res_1176_ = lp_mathlib_Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15(v_linterOption_1170_, v_stx_1171_, v_msg_1172_, v___y_1173_, v___y_1174_);
lean_dec(v___y_1174_);
lean_dec_ref(v___y_1173_);
return v_res_1176_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__16(lean_object* v_fst_1180_, lean_object* v_a_1181_, lean_object* v_a_1182_){
_start:
{
if (lean_obj_tag(v_a_1181_) == 0)
{
lean_object* v___x_1183_; 
lean_dec_ref(v_fst_1180_);
v___x_1183_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1183_, 0, v_a_1182_);
return v___x_1183_;
}
else
{
lean_object* v_key_1184_; lean_object* v_tail_1185_; lean_object* v_start_1186_; lean_object* v_stop_1187_; lean_object* v_start_1188_; lean_object* v_stop_1189_; lean_object* v___x_1190_; lean_object* v___x_1191_; uint8_t v___y_1193_; uint8_t v___x_1207_; 
lean_dec_ref(v_a_1182_);
v_key_1184_ = lean_ctor_get(v_a_1181_, 0);
v_tail_1185_ = lean_ctor_get(v_a_1181_, 2);
v_start_1186_ = lean_ctor_get(v_key_1184_, 0);
v_stop_1187_ = lean_ctor_get(v_key_1184_, 1);
v_start_1188_ = lean_ctor_get(v_fst_1180_, 0);
v_stop_1189_ = lean_ctor_get(v_fst_1180_, 1);
v___x_1190_ = lean_box(0);
v___x_1191_ = ((lean_object*)(lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__16___closed__0));
v___x_1207_ = lean_nat_dec_le(v_start_1186_, v_start_1188_);
if (v___x_1207_ == 0)
{
v___y_1193_ = v___x_1207_;
goto v___jp_1192_;
}
else
{
uint8_t v___x_1208_; 
v___x_1208_ = lean_nat_dec_le(v_stop_1189_, v_stop_1187_);
v___y_1193_ = v___x_1208_;
goto v___jp_1192_;
}
v___jp_1192_:
{
if (v___y_1193_ == 0)
{
v_a_1181_ = v_tail_1185_;
v_a_1182_ = v___x_1191_;
goto _start;
}
else
{
lean_object* v___x_1196_; uint8_t v_isShared_1197_; uint8_t v_isSharedCheck_1204_; 
v_isSharedCheck_1204_ = !lean_is_exclusive(v_fst_1180_);
if (v_isSharedCheck_1204_ == 0)
{
lean_object* v_unused_1205_; lean_object* v_unused_1206_; 
v_unused_1205_ = lean_ctor_get(v_fst_1180_, 1);
lean_dec(v_unused_1205_);
v_unused_1206_ = lean_ctor_get(v_fst_1180_, 0);
lean_dec(v_unused_1206_);
v___x_1196_ = v_fst_1180_;
v_isShared_1197_ = v_isSharedCheck_1204_;
goto v_resetjp_1195_;
}
else
{
lean_dec(v_fst_1180_);
v___x_1196_ = lean_box(0);
v_isShared_1197_ = v_isSharedCheck_1204_;
goto v_resetjp_1195_;
}
v_resetjp_1195_:
{
lean_object* v___x_1198_; lean_object* v___x_1199_; lean_object* v___x_1201_; 
v___x_1198_ = lean_box(v___y_1193_);
v___x_1199_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1199_, 0, v___x_1198_);
if (v_isShared_1197_ == 0)
{
lean_ctor_set(v___x_1196_, 1, v___x_1190_);
lean_ctor_set(v___x_1196_, 0, v___x_1199_);
v___x_1201_ = v___x_1196_;
goto v_reusejp_1200_;
}
else
{
lean_object* v_reuseFailAlloc_1203_; 
v_reuseFailAlloc_1203_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1203_, 0, v___x_1199_);
lean_ctor_set(v_reuseFailAlloc_1203_, 1, v___x_1190_);
v___x_1201_ = v_reuseFailAlloc_1203_;
goto v_reusejp_1200_;
}
v_reusejp_1200_:
{
lean_object* v___x_1202_; 
v___x_1202_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1202_, 0, v___x_1201_);
return v___x_1202_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__16___boxed(lean_object* v_fst_1209_, lean_object* v_a_1210_, lean_object* v_a_1211_){
_start:
{
lean_object* v_res_1212_; 
v_res_1212_ = lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__16(v_fst_1209_, v_a_1210_, v_a_1211_);
lean_dec(v_a_1210_);
return v_res_1212_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__17(lean_object* v_fst_1213_, lean_object* v_as_1214_, size_t v_sz_1215_, size_t v_i_1216_, lean_object* v_b_1217_){
_start:
{
uint8_t v___x_1218_; 
v___x_1218_ = lean_usize_dec_lt(v_i_1216_, v_sz_1215_);
if (v___x_1218_ == 0)
{
lean_dec_ref(v_fst_1213_);
return v_b_1217_;
}
else
{
lean_object* v_a_1219_; lean_object* v___x_1220_; 
v_a_1219_ = lean_array_uget_borrowed(v_as_1214_, v_i_1216_);
lean_inc_ref(v_fst_1213_);
v___x_1220_ = lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__16(v_fst_1213_, v_a_1219_, v_b_1217_);
if (lean_obj_tag(v___x_1220_) == 0)
{
lean_object* v_a_1221_; 
lean_dec_ref(v_fst_1213_);
v_a_1221_ = lean_ctor_get(v___x_1220_, 0);
lean_inc(v_a_1221_);
lean_dec_ref_known(v___x_1220_, 1);
return v_a_1221_;
}
else
{
lean_object* v_a_1222_; size_t v___x_1223_; size_t v___x_1224_; 
v_a_1222_ = lean_ctor_get(v___x_1220_, 0);
lean_inc(v_a_1222_);
lean_dec_ref_known(v___x_1220_, 1);
v___x_1223_ = ((size_t)1ULL);
v___x_1224_ = lean_usize_add(v_i_1216_, v___x_1223_);
v_i_1216_ = v___x_1224_;
v_b_1217_ = v_a_1222_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__17___boxed(lean_object* v_fst_1226_, lean_object* v_as_1227_, lean_object* v_sz_1228_, lean_object* v_i_1229_, lean_object* v_b_1230_){
_start:
{
size_t v_sz_boxed_1231_; size_t v_i_boxed_1232_; lean_object* v_res_1233_; 
v_sz_boxed_1231_ = lean_unbox_usize(v_sz_1228_);
lean_dec(v_sz_1228_);
v_i_boxed_1232_ = lean_unbox_usize(v_i_1229_);
lean_dec(v_i_1229_);
v_res_1233_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__17(v_fst_1226_, v_as_1227_, v_sz_boxed_1231_, v_i_boxed_1232_, v_b_1230_);
lean_dec_ref(v_as_1227_);
return v_res_1233_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__18___closed__2(void){
_start:
{
lean_object* v___x_1236_; lean_object* v___x_1237_; 
v___x_1236_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__18___closed__1));
v___x_1237_ = l_Lean_stringToMessageData(v___x_1236_);
return v___x_1237_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__18(uint8_t v___x_1240_, lean_object* v___x_1241_, lean_object* v_as_1242_, size_t v_sz_1243_, size_t v_i_1244_, lean_object* v_b_1245_, lean_object* v___y_1246_, lean_object* v___y_1247_){
_start:
{
lean_object* v_a_1250_; uint8_t v___x_1254_; 
v___x_1254_ = lean_usize_dec_lt(v_i_1244_, v_sz_1243_);
if (v___x_1254_ == 0)
{
lean_object* v___x_1255_; 
v___x_1255_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1255_, 0, v_b_1245_);
return v___x_1255_;
}
else
{
lean_object* v_a_1256_; lean_object* v_snd_1257_; lean_object* v_fst_1258_; lean_object* v_fst_1259_; lean_object* v_snd_1260_; lean_object* v___x_1262_; uint8_t v_isShared_1263_; uint8_t v_isSharedCheck_1310_; 
v_a_1256_ = lean_array_uget_borrowed(v_as_1242_, v_i_1244_);
v_snd_1257_ = lean_ctor_get(v_a_1256_, 1);
lean_inc(v_snd_1257_);
v_fst_1258_ = lean_ctor_get(v_a_1256_, 0);
v_fst_1259_ = lean_ctor_get(v_snd_1257_, 0);
v_snd_1260_ = lean_ctor_get(v_snd_1257_, 1);
v_isSharedCheck_1310_ = !lean_is_exclusive(v_snd_1257_);
if (v_isSharedCheck_1310_ == 0)
{
v___x_1262_ = v_snd_1257_;
v_isShared_1263_ = v_isSharedCheck_1310_;
goto v_resetjp_1261_;
}
else
{
lean_inc(v_snd_1260_);
lean_inc(v_fst_1259_);
lean_dec(v_snd_1257_);
v___x_1262_ = lean_box(0);
v_isShared_1263_ = v_isSharedCheck_1310_;
goto v_resetjp_1261_;
}
v_resetjp_1261_:
{
lean_object* v_buckets_1264_; lean_object* v___x_1265_; lean_object* v___x_1266_; size_t v_sz_1267_; size_t v___x_1268_; lean_object* v___x_1269_; lean_object* v_fst_1270_; lean_object* v___x_1272_; uint8_t v_isShared_1273_; uint8_t v_isSharedCheck_1308_; 
v_buckets_1264_ = lean_ctor_get(v___x_1241_, 1);
v___x_1265_ = lean_box(0);
v___x_1266_ = ((lean_object*)(lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__16___closed__0));
v_sz_1267_ = lean_array_size(v_buckets_1264_);
v___x_1268_ = ((size_t)0ULL);
lean_inc(v_fst_1258_);
v___x_1269_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__17(v_fst_1258_, v_buckets_1264_, v_sz_1267_, v___x_1268_, v___x_1266_);
v_fst_1270_ = lean_ctor_get(v___x_1269_, 0);
v_isSharedCheck_1308_ = !lean_is_exclusive(v___x_1269_);
if (v_isSharedCheck_1308_ == 0)
{
lean_object* v_unused_1309_; 
v_unused_1309_ = lean_ctor_get(v___x_1269_, 1);
lean_dec(v_unused_1309_);
v___x_1272_ = v___x_1269_;
v_isShared_1273_ = v_isSharedCheck_1308_;
goto v_resetjp_1271_;
}
else
{
lean_inc(v_fst_1270_);
lean_dec(v___x_1269_);
v___x_1272_ = lean_box(0);
v_isShared_1273_ = v_isSharedCheck_1308_;
goto v_resetjp_1271_;
}
v_resetjp_1271_:
{
lean_object* v___x_1274_; lean_object* v___x_1275_; lean_object* v___x_1276_; 
v___x_1274_ = lean_unsigned_to_nat(1u);
v___x_1275_ = lp_mathlib_Mathlib_Linter_linter_style_emptyLine;
v___x_1276_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29___closed__0));
if (lean_obj_tag(v_fst_1270_) == 0)
{
goto v___jp_1277_;
}
else
{
lean_object* v_val_1306_; uint8_t v___x_1307_; 
v_val_1306_ = lean_ctor_get(v_fst_1270_, 0);
lean_inc(v_val_1306_);
lean_dec_ref_known(v_fst_1270_, 1);
v___x_1307_ = lean_unbox(v_val_1306_);
lean_dec(v_val_1306_);
if (v___x_1307_ == 0)
{
goto v___jp_1277_;
}
else
{
lean_del_object(v___x_1272_);
lean_del_object(v___x_1262_);
lean_dec(v_snd_1260_);
lean_dec(v_fst_1259_);
v_a_1250_ = v___x_1265_;
goto v___jp_1249_;
}
}
v___jp_1277_:
{
lean_object* v___x_1278_; lean_object* v___x_1279_; lean_object* v___x_1280_; lean_object* v___x_1281_; lean_object* v___x_1282_; lean_object* v___x_1283_; lean_object* v___x_1284_; uint32_t v___x_1285_; lean_object* v___x_1286_; lean_object* v___x_1287_; lean_object* v___x_1288_; lean_object* v___x_1289_; lean_object* v___x_1291_; 
v___x_1278_ = lean_string_length(v_fst_1259_);
v___x_1279_ = lean_nat_add(v___x_1278_, v___x_1274_);
v___x_1280_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__18___closed__0));
v___x_1281_ = l_List_replicateTR___redArg(v___x_1279_, v___x_1280_);
v___x_1282_ = lp_mathlib_List_foldl___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__14(v___x_1276_, v___x_1281_);
lean_dec(v___x_1281_);
lean_inc(v_fst_1258_);
v___x_1283_ = l_Lean_Syntax_ofRange(v_fst_1258_, v___x_1240_);
v___x_1284_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__18___closed__2, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__18___closed__2_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__18___closed__2);
v___x_1285_ = 8595;
v___x_1286_ = lean_string_push(v___x_1282_, v___x_1285_);
v___x_1287_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1287_, 0, v___x_1286_);
v___x_1288_ = l_Lean_MessageData_ofFormat(v___x_1287_);
v___x_1289_ = l_Lean_indentD(v___x_1288_);
if (v_isShared_1273_ == 0)
{
lean_ctor_set_tag(v___x_1272_, 7);
lean_ctor_set(v___x_1272_, 1, v___x_1289_);
lean_ctor_set(v___x_1272_, 0, v___x_1284_);
v___x_1291_ = v___x_1272_;
goto v_reusejp_1290_;
}
else
{
lean_object* v_reuseFailAlloc_1305_; 
v_reuseFailAlloc_1305_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1305_, 0, v___x_1284_);
lean_ctor_set(v_reuseFailAlloc_1305_, 1, v___x_1289_);
v___x_1291_ = v_reuseFailAlloc_1305_;
goto v_reusejp_1290_;
}
v_reusejp_1290_:
{
lean_object* v___x_1292_; lean_object* v___x_1293_; lean_object* v___x_1294_; lean_object* v___x_1295_; lean_object* v___x_1296_; lean_object* v___x_1297_; lean_object* v___x_1298_; lean_object* v___x_1299_; lean_object* v___x_1300_; lean_object* v___x_1302_; 
v___x_1292_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__18___closed__3));
v___x_1293_ = lean_string_append(v___x_1292_, v_fst_1259_);
lean_dec(v_fst_1259_);
v___x_1294_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__18___closed__4));
v___x_1295_ = lean_string_append(v___x_1293_, v___x_1294_);
v___x_1296_ = lean_string_append(v___x_1295_, v_snd_1260_);
lean_dec(v_snd_1260_);
v___x_1297_ = lean_string_append(v___x_1296_, v___x_1292_);
v___x_1298_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1298_, 0, v___x_1297_);
v___x_1299_ = l_Lean_MessageData_ofFormat(v___x_1298_);
v___x_1300_ = l_Lean_indentD(v___x_1299_);
if (v_isShared_1263_ == 0)
{
lean_ctor_set_tag(v___x_1262_, 7);
lean_ctor_set(v___x_1262_, 1, v___x_1300_);
lean_ctor_set(v___x_1262_, 0, v___x_1291_);
v___x_1302_ = v___x_1262_;
goto v_reusejp_1301_;
}
else
{
lean_object* v_reuseFailAlloc_1304_; 
v_reuseFailAlloc_1304_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1304_, 0, v___x_1291_);
lean_ctor_set(v_reuseFailAlloc_1304_, 1, v___x_1300_);
v___x_1302_ = v_reuseFailAlloc_1304_;
goto v_reusejp_1301_;
}
v_reusejp_1301_:
{
lean_object* v___x_1303_; 
v___x_1303_ = lp_mathlib_Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15(v___x_1275_, v___x_1283_, v___x_1302_, v___y_1246_, v___y_1247_);
if (lean_obj_tag(v___x_1303_) == 0)
{
lean_dec_ref_known(v___x_1303_, 1);
v_a_1250_ = v___x_1265_;
goto v___jp_1249_;
}
else
{
return v___x_1303_;
}
}
}
}
}
}
}
v___jp_1249_:
{
size_t v___x_1251_; size_t v___x_1252_; 
v___x_1251_ = ((size_t)1ULL);
v___x_1252_ = lean_usize_add(v_i_1244_, v___x_1251_);
v_i_1244_ = v___x_1252_;
v_b_1245_ = v_a_1250_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__18___boxed(lean_object* v___x_1311_, lean_object* v___x_1312_, lean_object* v_as_1313_, lean_object* v_sz_1314_, lean_object* v_i_1315_, lean_object* v_b_1316_, lean_object* v___y_1317_, lean_object* v___y_1318_, lean_object* v___y_1319_){
_start:
{
uint8_t v___x_18604__boxed_1320_; size_t v_sz_boxed_1321_; size_t v_i_boxed_1322_; lean_object* v_res_1323_; 
v___x_18604__boxed_1320_ = lean_unbox(v___x_1311_);
v_sz_boxed_1321_ = lean_unbox_usize(v_sz_1314_);
lean_dec(v_sz_1314_);
v_i_boxed_1322_ = lean_unbox_usize(v_i_1315_);
lean_dec(v_i_1315_);
v_res_1323_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__18(v___x_18604__boxed_1320_, v___x_1312_, v_as_1313_, v_sz_boxed_1321_, v_i_boxed_1322_, v_b_1316_, v___y_1317_, v___y_1318_);
lean_dec(v___y_1318_);
lean_dec_ref(v___y_1317_);
lean_dec_ref(v_as_1313_);
lean_dec_ref(v___x_1312_);
return v_res_1323_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_Pos_revSkipWhile___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__10(uint8_t v___x_1324_, lean_object* v_s_1325_, lean_object* v_pos_1326_){
_start:
{
lean_object* v_str_1327_; lean_object* v_startInclusive_1328_; lean_object* v___x_1329_; lean_object* v___x_1330_; lean_object* v___x_1331_; uint8_t v___x_1332_; 
v_str_1327_ = lean_ctor_get(v_s_1325_, 0);
v_startInclusive_1328_ = lean_ctor_get(v_s_1325_, 1);
v___x_1329_ = lean_nat_add(v_startInclusive_1328_, v_pos_1326_);
v___x_1330_ = lean_nat_sub(v___x_1329_, v_startInclusive_1328_);
v___x_1331_ = lean_unsigned_to_nat(0u);
v___x_1332_ = lean_nat_dec_eq(v___x_1330_, v___x_1331_);
if (v___x_1332_ == 0)
{
lean_object* v___x_1333_; lean_object* v___x_1334_; lean_object* v___x_1335_; lean_object* v___x_1336_; lean_object* v___x_1337_; uint32_t v___x_1338_; uint32_t v___x_1339_; uint8_t v___x_1340_; 
lean_inc(v_startInclusive_1328_);
lean_inc_ref(v_str_1327_);
v___x_1333_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1333_, 0, v_str_1327_);
lean_ctor_set(v___x_1333_, 1, v_startInclusive_1328_);
lean_ctor_set(v___x_1333_, 2, v___x_1329_);
v___x_1334_ = lean_unsigned_to_nat(1u);
v___x_1335_ = lean_nat_sub(v___x_1330_, v___x_1334_);
lean_dec(v___x_1330_);
v___x_1336_ = l_String_Slice_posLE(v___x_1333_, v___x_1335_);
lean_dec_ref_known(v___x_1333_, 3);
v___x_1337_ = lean_nat_add(v_startInclusive_1328_, v___x_1336_);
v___x_1338_ = lean_string_utf8_get_fast(v_str_1327_, v___x_1337_);
lean_dec(v___x_1337_);
v___x_1339_ = 10;
v___x_1340_ = lean_uint32_dec_eq(v___x_1338_, v___x_1339_);
if (v___x_1340_ == 0)
{
if (v___x_1324_ == 0)
{
lean_dec(v___x_1336_);
return v_pos_1326_;
}
else
{
uint8_t v___x_1341_; 
v___x_1341_ = lean_nat_dec_lt(v___x_1336_, v_pos_1326_);
if (v___x_1341_ == 0)
{
lean_dec(v___x_1336_);
return v_pos_1326_;
}
else
{
lean_dec(v_pos_1326_);
v_pos_1326_ = v___x_1336_;
goto _start;
}
}
}
else
{
lean_dec(v___x_1336_);
return v_pos_1326_;
}
}
else
{
lean_dec(v___x_1330_);
lean_dec(v___x_1329_);
return v_pos_1326_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_Pos_revSkipWhile___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__10___boxed(lean_object* v___x_1343_, lean_object* v_s_1344_, lean_object* v_pos_1345_){
_start:
{
uint8_t v___x_18743__boxed_1346_; lean_object* v_res_1347_; 
v___x_18743__boxed_1346_ = lean_unbox(v___x_1343_);
v_res_1347_ = lp_mathlib_String_Slice_Pos_revSkipWhile___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__10(v___x_18743__boxed_1346_, v_s_1344_, v_pos_1345_);
lean_dec_ref(v_s_1344_);
return v_res_1347_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_Pos_skipWhile___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__9(uint8_t v___x_1348_, lean_object* v_s_1349_, lean_object* v_pos_1350_){
_start:
{
lean_object* v_str_1351_; lean_object* v_startInclusive_1352_; lean_object* v_endExclusive_1353_; lean_object* v___x_1354_; lean_object* v___x_1355_; lean_object* v___x_1356_; uint8_t v___x_1357_; 
v_str_1351_ = lean_ctor_get(v_s_1349_, 0);
v_startInclusive_1352_ = lean_ctor_get(v_s_1349_, 1);
v_endExclusive_1353_ = lean_ctor_get(v_s_1349_, 2);
v___x_1354_ = lean_nat_add(v_startInclusive_1352_, v_pos_1350_);
v___x_1355_ = lean_unsigned_to_nat(0u);
v___x_1356_ = lean_nat_sub(v_endExclusive_1353_, v___x_1354_);
v___x_1357_ = lean_nat_dec_eq(v___x_1355_, v___x_1356_);
lean_dec(v___x_1356_);
if (v___x_1357_ == 0)
{
uint32_t v___x_1358_; uint32_t v___x_1359_; uint8_t v___x_1360_; 
v___x_1358_ = lean_string_utf8_get_fast(v_str_1351_, v___x_1354_);
v___x_1359_ = 10;
v___x_1360_ = lean_uint32_dec_eq(v___x_1358_, v___x_1359_);
if (v___x_1360_ == 0)
{
if (v___x_1348_ == 0)
{
lean_dec(v___x_1354_);
return v_pos_1350_;
}
else
{
lean_object* v___x_1361_; lean_object* v___x_1362_; lean_object* v___x_1363_; uint8_t v___x_1364_; 
v___x_1361_ = lean_string_utf8_next_fast(v_str_1351_, v___x_1354_);
v___x_1362_ = lean_nat_sub(v___x_1361_, v___x_1354_);
lean_dec(v___x_1354_);
v___x_1363_ = lean_nat_add(v_pos_1350_, v___x_1362_);
lean_dec(v___x_1362_);
v___x_1364_ = lean_nat_dec_lt(v_pos_1350_, v___x_1363_);
if (v___x_1364_ == 0)
{
lean_dec(v___x_1363_);
return v_pos_1350_;
}
else
{
lean_dec(v_pos_1350_);
v_pos_1350_ = v___x_1363_;
goto _start;
}
}
}
else
{
lean_dec(v___x_1354_);
return v_pos_1350_;
}
}
else
{
lean_dec(v___x_1354_);
return v_pos_1350_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_Pos_skipWhile___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__9___boxed(lean_object* v___x_1366_, lean_object* v_s_1367_, lean_object* v_pos_1368_){
_start:
{
uint8_t v___x_18774__boxed_1369_; lean_object* v_res_1370_; 
v___x_18774__boxed_1369_ = lean_unbox(v___x_1366_);
v_res_1370_ = lp_mathlib_String_Slice_Pos_skipWhile___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__9(v___x_18774__boxed_1369_, v_s_1367_, v_pos_1368_);
lean_dec_ref(v_s_1367_);
return v_res_1370_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__11___redArg(uint8_t v___x_1371_, lean_object* v_as_x27_1372_, lean_object* v_b_1373_){
_start:
{
if (lean_obj_tag(v_as_x27_1372_) == 0)
{
lean_object* v___x_1375_; 
v___x_1375_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1375_, 0, v_b_1373_);
return v___x_1375_;
}
else
{
lean_object* v_snd_1376_; lean_object* v_head_1377_; lean_object* v_tail_1378_; lean_object* v_fst_1379_; lean_object* v___x_1381_; uint8_t v_isShared_1382_; uint8_t v_isSharedCheck_1420_; 
v_snd_1376_ = lean_ctor_get(v_b_1373_, 1);
lean_inc(v_snd_1376_);
v_head_1377_ = lean_ctor_get(v_as_x27_1372_, 0);
v_tail_1378_ = lean_ctor_get(v_as_x27_1372_, 1);
v_fst_1379_ = lean_ctor_get(v_b_1373_, 0);
v_isSharedCheck_1420_ = !lean_is_exclusive(v_b_1373_);
if (v_isSharedCheck_1420_ == 0)
{
lean_object* v_unused_1421_; 
v_unused_1421_ = lean_ctor_get(v_b_1373_, 1);
lean_dec(v_unused_1421_);
v___x_1381_ = v_b_1373_;
v_isShared_1382_ = v_isSharedCheck_1420_;
goto v_resetjp_1380_;
}
else
{
lean_inc(v_fst_1379_);
lean_dec(v_b_1373_);
v___x_1381_ = lean_box(0);
v_isShared_1382_ = v_isSharedCheck_1420_;
goto v_resetjp_1380_;
}
v_resetjp_1380_:
{
lean_object* v_fst_1383_; lean_object* v_snd_1384_; lean_object* v___x_1386_; uint8_t v_isShared_1387_; uint8_t v_isSharedCheck_1419_; 
v_fst_1383_ = lean_ctor_get(v_snd_1376_, 0);
v_snd_1384_ = lean_ctor_get(v_snd_1376_, 1);
v_isSharedCheck_1419_ = !lean_is_exclusive(v_snd_1376_);
if (v_isSharedCheck_1419_ == 0)
{
v___x_1386_ = v_snd_1376_;
v_isShared_1387_ = v_isSharedCheck_1419_;
goto v_resetjp_1385_;
}
else
{
lean_inc(v_snd_1384_);
lean_inc(v_fst_1383_);
lean_dec(v_snd_1376_);
v___x_1386_ = lean_box(0);
v_isShared_1387_ = v_isSharedCheck_1419_;
goto v_resetjp_1385_;
}
v_resetjp_1385_:
{
lean_object* v___x_1388_; lean_object* v_str_1389_; lean_object* v_startInclusive_1390_; lean_object* v_endExclusive_1391_; lean_object* v___x_1393_; uint8_t v_isShared_1394_; uint8_t v_isSharedCheck_1418_; 
lean_inc_n(v_fst_1383_, 2);
v___x_1388_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1388_, 0, v_fst_1383_);
lean_ctor_set(v___x_1388_, 1, v_fst_1383_);
v_str_1389_ = lean_ctor_get(v_snd_1384_, 0);
v_startInclusive_1390_ = lean_ctor_get(v_snd_1384_, 1);
v_endExclusive_1391_ = lean_ctor_get(v_snd_1384_, 2);
v_isSharedCheck_1418_ = !lean_is_exclusive(v_snd_1384_);
if (v_isSharedCheck_1418_ == 0)
{
v___x_1393_ = v_snd_1384_;
v_isShared_1394_ = v_isSharedCheck_1418_;
goto v_resetjp_1392_;
}
else
{
lean_inc(v_endExclusive_1391_);
lean_inc(v_startInclusive_1390_);
lean_inc(v_str_1389_);
lean_dec(v_snd_1384_);
v___x_1393_ = lean_box(0);
v_isShared_1394_ = v_isSharedCheck_1418_;
goto v_resetjp_1392_;
}
v_resetjp_1392_:
{
lean_object* v___x_1395_; lean_object* v___x_1396_; lean_object* v___x_1397_; lean_object* v___x_1399_; 
v___x_1395_ = lean_string_utf8_extract_fast(v_str_1389_, v_startInclusive_1390_, v_endExclusive_1391_);
lean_dec(v_endExclusive_1391_);
lean_dec(v_startInclusive_1390_);
lean_dec_ref(v_str_1389_);
v___x_1396_ = lean_unsigned_to_nat(0u);
v___x_1397_ = lean_string_utf8_byte_size(v_head_1377_);
lean_inc(v_head_1377_);
if (v_isShared_1394_ == 0)
{
lean_ctor_set(v___x_1393_, 2, v___x_1397_);
lean_ctor_set(v___x_1393_, 1, v___x_1396_);
lean_ctor_set(v___x_1393_, 0, v_head_1377_);
v___x_1399_ = v___x_1393_;
goto v_reusejp_1398_;
}
else
{
lean_object* v_reuseFailAlloc_1417_; 
v_reuseFailAlloc_1417_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1417_, 0, v_head_1377_);
lean_ctor_set(v_reuseFailAlloc_1417_, 1, v___x_1396_);
lean_ctor_set(v_reuseFailAlloc_1417_, 2, v___x_1397_);
v___x_1399_ = v_reuseFailAlloc_1417_;
goto v_reusejp_1398_;
}
v_reusejp_1398_:
{
lean_object* v___x_1400_; lean_object* v___x_1401_; lean_object* v___x_1403_; 
v___x_1400_ = lp_mathlib_String_Slice_Pos_skipWhile___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__9(v___x_1371_, v___x_1399_, v___x_1396_);
v___x_1401_ = lean_string_utf8_extract_fast(v_head_1377_, v___x_1396_, v___x_1400_);
lean_dec(v___x_1400_);
if (v_isShared_1387_ == 0)
{
lean_ctor_set(v___x_1386_, 1, v___x_1401_);
lean_ctor_set(v___x_1386_, 0, v___x_1395_);
v___x_1403_ = v___x_1386_;
goto v_reusejp_1402_;
}
else
{
lean_object* v_reuseFailAlloc_1416_; 
v_reuseFailAlloc_1416_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1416_, 0, v___x_1395_);
lean_ctor_set(v_reuseFailAlloc_1416_, 1, v___x_1401_);
v___x_1403_ = v_reuseFailAlloc_1416_;
goto v_reusejp_1402_;
}
v_reusejp_1402_:
{
lean_object* v___x_1405_; 
if (v_isShared_1382_ == 0)
{
lean_ctor_set(v___x_1381_, 1, v___x_1403_);
lean_ctor_set(v___x_1381_, 0, v___x_1388_);
v___x_1405_ = v___x_1381_;
goto v_reusejp_1404_;
}
else
{
lean_object* v_reuseFailAlloc_1415_; 
v_reuseFailAlloc_1415_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1415_, 0, v___x_1388_);
lean_ctor_set(v_reuseFailAlloc_1415_, 1, v___x_1403_);
v___x_1405_ = v_reuseFailAlloc_1415_;
goto v_reusejp_1404_;
}
v_reusejp_1404_:
{
lean_object* v___x_1406_; lean_object* v___x_1407_; lean_object* v___x_1408_; lean_object* v___x_1409_; lean_object* v___x_1410_; lean_object* v___x_1411_; lean_object* v___x_1412_; lean_object* v___x_1413_; 
v___x_1406_ = lean_array_push(v_fst_1379_, v___x_1405_);
v___x_1407_ = lean_unsigned_to_nat(2u);
v___x_1408_ = lean_nat_add(v___x_1397_, v___x_1407_);
v___x_1409_ = lean_nat_add(v___x_1408_, v_fst_1383_);
lean_dec(v_fst_1383_);
lean_dec(v___x_1408_);
v___x_1410_ = lp_mathlib_String_Slice_Pos_revSkipWhile___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__10(v___x_1371_, v___x_1399_, v___x_1397_);
lean_dec_ref(v___x_1399_);
lean_inc(v_head_1377_);
v___x_1411_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1411_, 0, v_head_1377_);
lean_ctor_set(v___x_1411_, 1, v___x_1410_);
lean_ctor_set(v___x_1411_, 2, v___x_1397_);
v___x_1412_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1412_, 0, v___x_1409_);
lean_ctor_set(v___x_1412_, 1, v___x_1411_);
v___x_1413_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1413_, 0, v___x_1406_);
lean_ctor_set(v___x_1413_, 1, v___x_1412_);
v_as_x27_1372_ = v_tail_1378_;
v_b_1373_ = v___x_1413_;
goto _start;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__11___redArg___boxed(lean_object* v___x_1422_, lean_object* v_as_x27_1423_, lean_object* v_b_1424_, lean_object* v___y_1425_){
_start:
{
uint8_t v___x_18801__boxed_1426_; lean_object* v_res_1427_; 
v___x_18801__boxed_1426_ = lean_unbox(v___x_1422_);
v_res_1427_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__11___redArg(v___x_18801__boxed_1426_, v_as_x27_1423_, v_b_1424_);
lean_dec(v_as_x27_1423_);
return v_res_1427_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__13_spec__20_spec__23___redArg(lean_object* v_a_1428_, lean_object* v_x_1429_){
_start:
{
if (lean_obj_tag(v_x_1429_) == 0)
{
uint8_t v___x_1430_; 
v___x_1430_ = 0;
return v___x_1430_;
}
else
{
lean_object* v_key_1431_; lean_object* v_tail_1432_; uint8_t v___x_1433_; 
v_key_1431_ = lean_ctor_get(v_x_1429_, 0);
v_tail_1432_ = lean_ctor_get(v_x_1429_, 2);
v___x_1433_ = l_Lean_Syntax_instBEqRange_beq(v_key_1431_, v_a_1428_);
if (v___x_1433_ == 0)
{
v_x_1429_ = v_tail_1432_;
goto _start;
}
else
{
return v___x_1433_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__13_spec__20_spec__23___redArg___boxed(lean_object* v_a_1435_, lean_object* v_x_1436_){
_start:
{
uint8_t v_res_1437_; lean_object* v_r_1438_; 
v_res_1437_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__13_spec__20_spec__23___redArg(v_a_1435_, v_x_1436_);
lean_dec(v_x_1436_);
lean_dec_ref(v_a_1435_);
v_r_1438_ = lean_box(v_res_1437_);
return v_r_1438_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__13_spec__20_spec__24_spec__32_spec__35___redArg(lean_object* v_x_1439_, lean_object* v_x_1440_){
_start:
{
if (lean_obj_tag(v_x_1440_) == 0)
{
return v_x_1439_;
}
else
{
lean_object* v_key_1441_; lean_object* v_value_1442_; lean_object* v_tail_1443_; lean_object* v___x_1445_; uint8_t v_isShared_1446_; uint8_t v_isSharedCheck_1466_; 
v_key_1441_ = lean_ctor_get(v_x_1440_, 0);
v_value_1442_ = lean_ctor_get(v_x_1440_, 1);
v_tail_1443_ = lean_ctor_get(v_x_1440_, 2);
v_isSharedCheck_1466_ = !lean_is_exclusive(v_x_1440_);
if (v_isSharedCheck_1466_ == 0)
{
v___x_1445_ = v_x_1440_;
v_isShared_1446_ = v_isSharedCheck_1466_;
goto v_resetjp_1444_;
}
else
{
lean_inc(v_tail_1443_);
lean_inc(v_value_1442_);
lean_inc(v_key_1441_);
lean_dec(v_x_1440_);
v___x_1445_ = lean_box(0);
v_isShared_1446_ = v_isSharedCheck_1466_;
goto v_resetjp_1444_;
}
v_resetjp_1444_:
{
lean_object* v___x_1447_; uint64_t v___x_1448_; uint64_t v___x_1449_; uint64_t v___x_1450_; uint64_t v_fold_1451_; uint64_t v___x_1452_; uint64_t v___x_1453_; uint64_t v___x_1454_; size_t v___x_1455_; size_t v___x_1456_; size_t v___x_1457_; size_t v___x_1458_; size_t v___x_1459_; lean_object* v___x_1460_; lean_object* v___x_1462_; 
v___x_1447_ = lean_array_get_size(v_x_1439_);
v___x_1448_ = l_Lean_Syntax_instHashableRange_hash(v_key_1441_);
v___x_1449_ = 32ULL;
v___x_1450_ = lean_uint64_shift_right(v___x_1448_, v___x_1449_);
v_fold_1451_ = lean_uint64_xor(v___x_1448_, v___x_1450_);
v___x_1452_ = 16ULL;
v___x_1453_ = lean_uint64_shift_right(v_fold_1451_, v___x_1452_);
v___x_1454_ = lean_uint64_xor(v_fold_1451_, v___x_1453_);
v___x_1455_ = lean_uint64_to_usize(v___x_1454_);
v___x_1456_ = lean_usize_of_nat(v___x_1447_);
v___x_1457_ = ((size_t)1ULL);
v___x_1458_ = lean_usize_sub(v___x_1456_, v___x_1457_);
v___x_1459_ = lean_usize_land(v___x_1455_, v___x_1458_);
v___x_1460_ = lean_array_uget_borrowed(v_x_1439_, v___x_1459_);
lean_inc(v___x_1460_);
if (v_isShared_1446_ == 0)
{
lean_ctor_set(v___x_1445_, 2, v___x_1460_);
v___x_1462_ = v___x_1445_;
goto v_reusejp_1461_;
}
else
{
lean_object* v_reuseFailAlloc_1465_; 
v_reuseFailAlloc_1465_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1465_, 0, v_key_1441_);
lean_ctor_set(v_reuseFailAlloc_1465_, 1, v_value_1442_);
lean_ctor_set(v_reuseFailAlloc_1465_, 2, v___x_1460_);
v___x_1462_ = v_reuseFailAlloc_1465_;
goto v_reusejp_1461_;
}
v_reusejp_1461_:
{
lean_object* v___x_1463_; 
v___x_1463_ = lean_array_uset(v_x_1439_, v___x_1459_, v___x_1462_);
v_x_1439_ = v___x_1463_;
v_x_1440_ = v_tail_1443_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__13_spec__20_spec__24_spec__32___redArg(lean_object* v_i_1467_, lean_object* v_source_1468_, lean_object* v_target_1469_){
_start:
{
lean_object* v___x_1470_; uint8_t v___x_1471_; 
v___x_1470_ = lean_array_get_size(v_source_1468_);
v___x_1471_ = lean_nat_dec_lt(v_i_1467_, v___x_1470_);
if (v___x_1471_ == 0)
{
lean_dec_ref(v_source_1468_);
lean_dec(v_i_1467_);
return v_target_1469_;
}
else
{
lean_object* v_es_1472_; lean_object* v___x_1473_; lean_object* v_source_1474_; lean_object* v_target_1475_; lean_object* v___x_1476_; lean_object* v___x_1477_; 
v_es_1472_ = lean_array_fget(v_source_1468_, v_i_1467_);
v___x_1473_ = lean_box(0);
v_source_1474_ = lean_array_fset(v_source_1468_, v_i_1467_, v___x_1473_);
v_target_1475_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__13_spec__20_spec__24_spec__32_spec__35___redArg(v_target_1469_, v_es_1472_);
v___x_1476_ = lean_unsigned_to_nat(1u);
v___x_1477_ = lean_nat_add(v_i_1467_, v___x_1476_);
lean_dec(v_i_1467_);
v_i_1467_ = v___x_1477_;
v_source_1468_ = v_source_1474_;
v_target_1469_ = v_target_1475_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__13_spec__20_spec__24___redArg(lean_object* v_data_1479_){
_start:
{
lean_object* v___x_1480_; lean_object* v___x_1481_; lean_object* v_nbuckets_1482_; lean_object* v___x_1483_; lean_object* v___x_1484_; lean_object* v___x_1485_; lean_object* v___x_1486_; 
v___x_1480_ = lean_array_get_size(v_data_1479_);
v___x_1481_ = lean_unsigned_to_nat(2u);
v_nbuckets_1482_ = lean_nat_mul(v___x_1480_, v___x_1481_);
v___x_1483_ = lean_unsigned_to_nat(0u);
v___x_1484_ = lean_box(0);
v___x_1485_ = lean_mk_array(v_nbuckets_1482_, v___x_1484_);
v___x_1486_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__13_spec__20_spec__24_spec__32___redArg(v___x_1483_, v_data_1479_, v___x_1485_);
return v___x_1486_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__13_spec__20___redArg(lean_object* v_m_1487_, lean_object* v_a_1488_, lean_object* v_b_1489_){
_start:
{
lean_object* v_size_1490_; lean_object* v_buckets_1491_; lean_object* v___x_1492_; uint64_t v___x_1493_; uint64_t v___x_1494_; uint64_t v___x_1495_; uint64_t v_fold_1496_; uint64_t v___x_1497_; uint64_t v___x_1498_; uint64_t v___x_1499_; size_t v___x_1500_; size_t v___x_1501_; size_t v___x_1502_; size_t v___x_1503_; size_t v___x_1504_; lean_object* v_bkt_1505_; uint8_t v___x_1506_; 
v_size_1490_ = lean_ctor_get(v_m_1487_, 0);
v_buckets_1491_ = lean_ctor_get(v_m_1487_, 1);
v___x_1492_ = lean_array_get_size(v_buckets_1491_);
v___x_1493_ = l_Lean_Syntax_instHashableRange_hash(v_a_1488_);
v___x_1494_ = 32ULL;
v___x_1495_ = lean_uint64_shift_right(v___x_1493_, v___x_1494_);
v_fold_1496_ = lean_uint64_xor(v___x_1493_, v___x_1495_);
v___x_1497_ = 16ULL;
v___x_1498_ = lean_uint64_shift_right(v_fold_1496_, v___x_1497_);
v___x_1499_ = lean_uint64_xor(v_fold_1496_, v___x_1498_);
v___x_1500_ = lean_uint64_to_usize(v___x_1499_);
v___x_1501_ = lean_usize_of_nat(v___x_1492_);
v___x_1502_ = ((size_t)1ULL);
v___x_1503_ = lean_usize_sub(v___x_1501_, v___x_1502_);
v___x_1504_ = lean_usize_land(v___x_1500_, v___x_1503_);
v_bkt_1505_ = lean_array_uget_borrowed(v_buckets_1491_, v___x_1504_);
v___x_1506_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__13_spec__20_spec__23___redArg(v_a_1488_, v_bkt_1505_);
if (v___x_1506_ == 0)
{
lean_object* v___x_1508_; uint8_t v_isShared_1509_; uint8_t v_isSharedCheck_1527_; 
lean_inc_ref(v_buckets_1491_);
lean_inc(v_size_1490_);
v_isSharedCheck_1527_ = !lean_is_exclusive(v_m_1487_);
if (v_isSharedCheck_1527_ == 0)
{
lean_object* v_unused_1528_; lean_object* v_unused_1529_; 
v_unused_1528_ = lean_ctor_get(v_m_1487_, 1);
lean_dec(v_unused_1528_);
v_unused_1529_ = lean_ctor_get(v_m_1487_, 0);
lean_dec(v_unused_1529_);
v___x_1508_ = v_m_1487_;
v_isShared_1509_ = v_isSharedCheck_1527_;
goto v_resetjp_1507_;
}
else
{
lean_dec(v_m_1487_);
v___x_1508_ = lean_box(0);
v_isShared_1509_ = v_isSharedCheck_1527_;
goto v_resetjp_1507_;
}
v_resetjp_1507_:
{
lean_object* v___x_1510_; lean_object* v_size_x27_1511_; lean_object* v___x_1512_; lean_object* v_buckets_x27_1513_; lean_object* v___x_1514_; lean_object* v___x_1515_; lean_object* v___x_1516_; lean_object* v___x_1517_; lean_object* v___x_1518_; uint8_t v___x_1519_; 
v___x_1510_ = lean_unsigned_to_nat(1u);
v_size_x27_1511_ = lean_nat_add(v_size_1490_, v___x_1510_);
lean_dec(v_size_1490_);
lean_inc(v_bkt_1505_);
v___x_1512_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1512_, 0, v_a_1488_);
lean_ctor_set(v___x_1512_, 1, v_b_1489_);
lean_ctor_set(v___x_1512_, 2, v_bkt_1505_);
v_buckets_x27_1513_ = lean_array_uset(v_buckets_1491_, v___x_1504_, v___x_1512_);
v___x_1514_ = lean_unsigned_to_nat(4u);
v___x_1515_ = lean_nat_mul(v_size_x27_1511_, v___x_1514_);
v___x_1516_ = lean_unsigned_to_nat(3u);
v___x_1517_ = lean_nat_div(v___x_1515_, v___x_1516_);
lean_dec(v___x_1515_);
v___x_1518_ = lean_array_get_size(v_buckets_x27_1513_);
v___x_1519_ = lean_nat_dec_le(v___x_1517_, v___x_1518_);
lean_dec(v___x_1517_);
if (v___x_1519_ == 0)
{
lean_object* v_val_1520_; lean_object* v___x_1522_; 
v_val_1520_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__13_spec__20_spec__24___redArg(v_buckets_x27_1513_);
if (v_isShared_1509_ == 0)
{
lean_ctor_set(v___x_1508_, 1, v_val_1520_);
lean_ctor_set(v___x_1508_, 0, v_size_x27_1511_);
v___x_1522_ = v___x_1508_;
goto v_reusejp_1521_;
}
else
{
lean_object* v_reuseFailAlloc_1523_; 
v_reuseFailAlloc_1523_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1523_, 0, v_size_x27_1511_);
lean_ctor_set(v_reuseFailAlloc_1523_, 1, v_val_1520_);
v___x_1522_ = v_reuseFailAlloc_1523_;
goto v_reusejp_1521_;
}
v_reusejp_1521_:
{
return v___x_1522_;
}
}
else
{
lean_object* v___x_1525_; 
if (v_isShared_1509_ == 0)
{
lean_ctor_set(v___x_1508_, 1, v_buckets_x27_1513_);
lean_ctor_set(v___x_1508_, 0, v_size_x27_1511_);
v___x_1525_ = v___x_1508_;
goto v_reusejp_1524_;
}
else
{
lean_object* v_reuseFailAlloc_1526_; 
v_reuseFailAlloc_1526_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1526_, 0, v_size_x27_1511_);
lean_ctor_set(v_reuseFailAlloc_1526_, 1, v_buckets_x27_1513_);
v___x_1525_ = v_reuseFailAlloc_1526_;
goto v_reusejp_1524_;
}
v_reusejp_1524_:
{
return v___x_1525_;
}
}
}
}
else
{
lean_dec(v_b_1489_);
lean_dec_ref(v_a_1488_);
return v_m_1487_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__13_spec__21(lean_object* v_as_1530_, size_t v_sz_1531_, size_t v_i_1532_, lean_object* v_b_1533_){
_start:
{
uint8_t v___x_1534_; 
v___x_1534_ = lean_usize_dec_lt(v_i_1532_, v_sz_1531_);
if (v___x_1534_ == 0)
{
return v_b_1533_;
}
else
{
lean_object* v_a_1535_; lean_object* v___x_1536_; lean_object* v_r_1537_; size_t v___x_1538_; size_t v___x_1539_; 
v_a_1535_ = lean_array_uget_borrowed(v_as_1530_, v_i_1532_);
v___x_1536_ = lean_box(0);
lean_inc(v_a_1535_);
v_r_1537_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__13_spec__20___redArg(v_b_1533_, v_a_1535_, v___x_1536_);
v___x_1538_ = ((size_t)1ULL);
v___x_1539_ = lean_usize_add(v_i_1532_, v___x_1538_);
v_i_1532_ = v___x_1539_;
v_b_1533_ = v_r_1537_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__13_spec__21___boxed(lean_object* v_as_1541_, lean_object* v_sz_1542_, lean_object* v_i_1543_, lean_object* v_b_1544_){
_start:
{
size_t v_sz_boxed_1545_; size_t v_i_boxed_1546_; lean_object* v_res_1547_; 
v_sz_boxed_1545_ = lean_unbox_usize(v_sz_1542_);
lean_dec(v_sz_1542_);
v_i_boxed_1546_ = lean_unbox_usize(v_i_1543_);
lean_dec(v_i_1543_);
v_res_1547_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__13_spec__21(v_as_1541_, v_sz_boxed_1545_, v_i_boxed_1546_, v_b_1544_);
lean_dec_ref(v_as_1541_);
return v_res_1547_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__13(lean_object* v_m_1548_, lean_object* v_l_1549_){
_start:
{
size_t v_sz_1550_; size_t v___x_1551_; lean_object* v___x_1552_; 
v_sz_1550_ = lean_array_size(v_l_1549_);
v___x_1551_ = ((size_t)0ULL);
v___x_1552_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__13_spec__21(v_l_1549_, v_sz_1550_, v___x_1551_, v_m_1548_);
return v___x_1552_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__13___boxed(lean_object* v_m_1553_, lean_object* v_l_1554_){
_start:
{
lean_object* v_res_1555_; 
v_res_1555_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__13(v_m_1553_, v_l_1554_);
lean_dec_ref(v_l_1554_);
return v_res_1555_;
}
}
static lean_object* _init_lp_mathlib_List_find_x3f___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__19___closed__0(void){
_start:
{
lean_object* v___x_1556_; lean_object* v___x_1557_; lean_object* v___x_1558_; lean_object* v___x_1559_; 
v___x_1556_ = lean_box(0);
v___x_1557_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_EmptyLine_SkippedFileSegments___closed__1));
v___x_1558_ = lean_obj_once(&lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__3, &lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__3_once, _init_lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines___closed__3);
v___x_1559_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__6___redArg(v___x_1558_, v___x_1557_, v___x_1556_);
return v___x_1559_;
}
}
static lean_object* _init_lp_mathlib_List_find_x3f___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__19___closed__1(void){
_start:
{
lean_object* v___x_1560_; lean_object* v___x_1561_; lean_object* v___x_1562_; lean_object* v___x_1563_; 
v___x_1560_ = lean_box(0);
v___x_1561_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_EmptyLine_SkippedFileSegments___closed__4));
v___x_1562_ = lean_obj_once(&lp_mathlib_List_find_x3f___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__19___closed__0, &lp_mathlib_List_find_x3f___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__19___closed__0_once, _init_lp_mathlib_List_find_x3f___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__19___closed__0);
v___x_1563_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__6___redArg(v___x_1562_, v___x_1561_, v___x_1560_);
return v___x_1563_;
}
}
static lean_object* _init_lp_mathlib_List_find_x3f___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__19___closed__2(void){
_start:
{
lean_object* v___x_1564_; lean_object* v___x_1565_; lean_object* v___x_1566_; lean_object* v___x_1567_; 
v___x_1564_ = lean_box(0);
v___x_1565_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_EmptyLine_SkippedFileSegments___closed__7));
v___x_1566_ = lean_obj_once(&lp_mathlib_List_find_x3f___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__19___closed__1, &lp_mathlib_List_find_x3f___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__19___closed__1_once, _init_lp_mathlib_List_find_x3f___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__19___closed__1);
v___x_1567_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__6___redArg(v___x_1566_, v___x_1565_, v___x_1564_);
return v___x_1567_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_find_x3f___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__19(lean_object* v_x_1568_){
_start:
{
if (lean_obj_tag(v_x_1568_) == 0)
{
lean_object* v___x_1569_; 
v___x_1569_ = lean_box(0);
return v___x_1569_;
}
else
{
lean_object* v_head_1570_; lean_object* v_tail_1571_; lean_object* v___x_1572_; uint8_t v___x_1573_; 
v_head_1570_ = lean_ctor_get(v_x_1568_, 0);
v_tail_1571_ = lean_ctor_get(v_x_1568_, 1);
v___x_1572_ = lean_obj_once(&lp_mathlib_List_find_x3f___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__19___closed__2, &lp_mathlib_List_find_x3f___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__19___closed__2_once, _init_lp_mathlib_List_find_x3f___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__19___closed__2);
v___x_1573_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__7___redArg(v___x_1572_, v_head_1570_);
if (v___x_1573_ == 0)
{
v_x_1568_ = v_tail_1571_;
goto _start;
}
else
{
lean_object* v___x_1575_; 
lean_inc(v_head_1570_);
v___x_1575_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1575_, 0, v_head_1570_);
return v___x_1575_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_find_x3f___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__19___boxed(lean_object* v_x_1576_){
_start:
{
lean_object* v_res_1577_; 
v_res_1577_ = lp_mathlib_List_find_x3f___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__19(v_x_1576_);
lean_dec(v_x_1576_);
return v_res_1577_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__12_spec__18(uint8_t v___y_1578_, lean_object* v_as_1579_, size_t v_i_1580_, size_t v_stop_1581_, lean_object* v_b_1582_){
_start:
{
lean_object* v___y_1584_; uint8_t v___x_1588_; 
v___x_1588_ = lean_usize_dec_eq(v_i_1580_, v_stop_1581_);
if (v___x_1588_ == 0)
{
lean_object* v___x_1589_; lean_object* v___x_1590_; 
v___x_1589_ = lean_array_uget_borrowed(v_as_1579_, v_i_1580_);
v___x_1590_ = l_Lean_Syntax_getRange_x3f(v___x_1589_, v___y_1578_);
if (lean_obj_tag(v___x_1590_) == 0)
{
v___y_1584_ = v_b_1582_;
goto v___jp_1583_;
}
else
{
lean_object* v_val_1591_; lean_object* v___x_1592_; 
v_val_1591_ = lean_ctor_get(v___x_1590_, 0);
lean_inc(v_val_1591_);
lean_dec_ref_known(v___x_1590_, 1);
v___x_1592_ = lean_array_push(v_b_1582_, v_val_1591_);
v___y_1584_ = v___x_1592_;
goto v___jp_1583_;
}
}
else
{
return v_b_1582_;
}
v___jp_1583_:
{
size_t v___x_1585_; size_t v___x_1586_; 
v___x_1585_ = ((size_t)1ULL);
v___x_1586_ = lean_usize_add(v_i_1580_, v___x_1585_);
v_i_1580_ = v___x_1586_;
v_b_1582_ = v___y_1584_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__12_spec__18___boxed(lean_object* v___y_1593_, lean_object* v_as_1594_, lean_object* v_i_1595_, lean_object* v_stop_1596_, lean_object* v_b_1597_){
_start:
{
uint8_t v___y_19133__boxed_1598_; size_t v_i_boxed_1599_; size_t v_stop_boxed_1600_; lean_object* v_res_1601_; 
v___y_19133__boxed_1598_ = lean_unbox(v___y_1593_);
v_i_boxed_1599_ = lean_unbox_usize(v_i_1595_);
lean_dec(v_i_1595_);
v_stop_boxed_1600_ = lean_unbox_usize(v_stop_1596_);
lean_dec(v_stop_1596_);
v_res_1601_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__12_spec__18(v___y_19133__boxed_1598_, v_as_1594_, v_i_boxed_1599_, v_stop_boxed_1600_, v_b_1597_);
lean_dec_ref(v_as_1594_);
return v_res_1601_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_filterMapM___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__12(uint8_t v___y_1604_, lean_object* v_as_1605_, lean_object* v_start_1606_, lean_object* v_stop_1607_){
_start:
{
lean_object* v___x_1608_; uint8_t v___x_1609_; 
v___x_1608_ = ((lean_object*)(lp_mathlib_Array_filterMapM___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__12___closed__0));
v___x_1609_ = lean_nat_dec_lt(v_start_1606_, v_stop_1607_);
if (v___x_1609_ == 0)
{
return v___x_1608_;
}
else
{
lean_object* v___x_1610_; uint8_t v___x_1611_; 
v___x_1610_ = lean_array_get_size(v_as_1605_);
v___x_1611_ = lean_nat_dec_le(v_stop_1607_, v___x_1610_);
if (v___x_1611_ == 0)
{
uint8_t v___x_1612_; 
v___x_1612_ = lean_nat_dec_lt(v_start_1606_, v___x_1610_);
if (v___x_1612_ == 0)
{
return v___x_1608_;
}
else
{
size_t v___x_1613_; size_t v___x_1614_; lean_object* v___x_1615_; 
v___x_1613_ = lean_usize_of_nat(v_start_1606_);
v___x_1614_ = lean_usize_of_nat(v___x_1610_);
v___x_1615_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__12_spec__18(v___y_1604_, v_as_1605_, v___x_1613_, v___x_1614_, v___x_1608_);
return v___x_1615_;
}
}
else
{
size_t v___x_1616_; size_t v___x_1617_; lean_object* v___x_1618_; 
v___x_1616_ = lean_usize_of_nat(v_start_1606_);
v___x_1617_ = lean_usize_of_nat(v_stop_1607_);
v___x_1618_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__12_spec__18(v___y_1604_, v_as_1605_, v___x_1616_, v___x_1617_, v___x_1608_);
return v___x_1618_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_filterMapM___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__12___boxed(lean_object* v___y_1619_, lean_object* v_as_1620_, lean_object* v_start_1621_, lean_object* v_stop_1622_){
_start:
{
uint8_t v___y_19160__boxed_1623_; lean_object* v_res_1624_; 
v___y_19160__boxed_1623_ = lean_unbox(v___y_1619_);
v_res_1624_ = lp_mathlib_Array_filterMapM___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__12(v___y_19160__boxed_1623_, v_as_1620_, v_start_1621_, v_stop_1622_);
lean_dec(v_stop_1622_);
lean_dec(v_start_1621_);
lean_dec_ref(v_as_1620_);
return v_res_1624_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__2___closed__1(void){
_start:
{
lean_object* v___x_1627_; lean_object* v___x_1628_; lean_object* v___x_1629_; 
v___x_1627_ = lean_box(0);
v___x_1628_ = lean_unsigned_to_nat(16u);
v___x_1629_ = lean_mk_array(v___x_1628_, v___x_1627_);
return v___x_1629_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__2___closed__2(void){
_start:
{
lean_object* v___x_1630_; lean_object* v___x_1631_; lean_object* v___x_1632_; 
v___x_1630_ = lean_obj_once(&lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__2___closed__1, &lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__2___closed__1_once, _init_lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__2___closed__1);
v___x_1631_ = lean_unsigned_to_nat(0u);
v___x_1632_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1632_, 0, v___x_1631_);
lean_ctor_set(v___x_1632_, 1, v___x_1630_);
return v___x_1632_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__2(lean_object* v___f_1633_, lean_object* v_stx_1634_, lean_object* v___y_1635_, lean_object* v___y_1636_){
_start:
{
lean_object* v___x_1638_; lean_object* v_a_1639_; lean_object* v___x_1641_; uint8_t v_isShared_1642_; uint8_t v_isSharedCheck_1744_; 
v___x_1638_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__0(v___y_1635_, v___y_1636_);
v_a_1639_ = lean_ctor_get(v___x_1638_, 0);
v_isSharedCheck_1744_ = !lean_is_exclusive(v___x_1638_);
if (v_isSharedCheck_1744_ == 0)
{
v___x_1641_ = v___x_1638_;
v_isShared_1642_ = v_isSharedCheck_1744_;
goto v_resetjp_1640_;
}
else
{
lean_inc(v_a_1639_);
lean_dec(v___x_1638_);
v___x_1641_ = lean_box(0);
v_isShared_1642_ = v_isSharedCheck_1744_;
goto v_resetjp_1640_;
}
v_resetjp_1640_:
{
lean_object* v___x_1643_; uint8_t v___x_1644_; 
v___x_1643_ = lp_mathlib_Mathlib_Linter_linter_style_emptyLine;
v___x_1644_ = l_Lean_Linter_getLinterValue(v___x_1643_, v_a_1639_);
lean_dec(v_a_1639_);
if (v___x_1644_ == 0)
{
lean_object* v___x_1645_; lean_object* v___x_1647_; 
lean_dec(v_stx_1634_);
lean_dec_ref(v___f_1633_);
v___x_1645_ = lean_box(0);
if (v_isShared_1642_ == 0)
{
lean_ctor_set(v___x_1641_, 0, v___x_1645_);
v___x_1647_ = v___x_1641_;
goto v_reusejp_1646_;
}
else
{
lean_object* v_reuseFailAlloc_1648_; 
v_reuseFailAlloc_1648_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1648_, 0, v___x_1645_);
v___x_1647_ = v_reuseFailAlloc_1648_;
goto v_reusejp_1646_;
}
v_reusejp_1646_:
{
return v___x_1647_;
}
}
else
{
lean_object* v___x_1649_; lean_object* v_messages_1650_; lean_object* v___x_1651_; uint8_t v___x_1652_; 
v___x_1649_ = lean_st_ref_get(v___y_1636_);
v_messages_1650_ = lean_ctor_get(v___x_1649_, 1);
lean_inc_ref(v_messages_1650_);
lean_dec(v___x_1649_);
v___x_1651_ = l_Lean_MessageLog_reportedPlusUnreported(v_messages_1650_);
v___x_1652_ = lp_mathlib_Lean_PersistentArray_anyM___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__1(v___x_1644_, v___x_1651_);
lean_dec_ref(v___x_1651_);
if (v___x_1652_ == 0)
{
lean_object* v___x_1653_; lean_object* v_a_1654_; lean_object* v___x_1656_; uint8_t v_isShared_1657_; uint8_t v_isSharedCheck_1739_; 
v___x_1653_ = lp_mathlib_Lean_getMainModule___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__2___redArg(v___y_1636_);
v_a_1654_ = lean_ctor_get(v___x_1653_, 0);
v_isSharedCheck_1739_ = !lean_is_exclusive(v___x_1653_);
if (v_isSharedCheck_1739_ == 0)
{
v___x_1656_ = v___x_1653_;
v_isShared_1657_ = v_isSharedCheck_1739_;
goto v_resetjp_1655_;
}
else
{
lean_inc(v_a_1654_);
lean_dec(v___x_1653_);
v___x_1656_ = lean_box(0);
v_isShared_1657_ = v_isSharedCheck_1739_;
goto v_resetjp_1655_;
}
v_resetjp_1655_:
{
lean_object* v___x_1663_; lean_object* v___f_1664_; uint8_t v___y_1666_; lean_object* v___x_1737_; lean_object* v___x_1738_; 
v___x_1663_ = lean_box(v___x_1644_);
v___f_1664_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__1___boxed), 2, 1);
lean_closure_set(v___f_1664_, 0, v___x_1663_);
v___x_1737_ = l_Lean_Name_components(v_a_1654_);
v___x_1738_ = lp_mathlib_List_find_x3f___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__19(v___x_1737_);
lean_dec(v___x_1737_);
if (lean_obj_tag(v___x_1738_) == 0)
{
v___y_1666_ = v___x_1652_;
goto v___jp_1665_;
}
else
{
lean_dec_ref_known(v___x_1738_, 1);
v___y_1666_ = v___x_1644_;
goto v___jp_1665_;
}
v___jp_1658_:
{
lean_object* v___x_1659_; lean_object* v___x_1661_; 
v___x_1659_ = lean_box(0);
if (v_isShared_1657_ == 0)
{
lean_ctor_set(v___x_1656_, 0, v___x_1659_);
v___x_1661_ = v___x_1656_;
goto v_reusejp_1660_;
}
else
{
lean_object* v_reuseFailAlloc_1662_; 
v_reuseFailAlloc_1662_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1662_, 0, v___x_1659_);
v___x_1661_ = v_reuseFailAlloc_1662_;
goto v_reusejp_1660_;
}
v_reusejp_1660_:
{
return v___x_1661_;
}
}
v___jp_1665_:
{
if (v___y_1666_ == 0)
{
lean_object* v___x_1667_; lean_object* v___x_1668_; 
v___x_1667_ = l_Lean_Syntax_unsetTrailing(v_stx_1634_);
v___x_1668_ = l_Lean_Syntax_getSubstring_x3f(v___x_1667_, v___x_1644_, v___x_1644_);
if (lean_obj_tag(v___x_1668_) == 1)
{
lean_object* v_val_1669_; lean_object* v_str_1670_; lean_object* v_startPos_1671_; lean_object* v_stopPos_1672_; lean_object* v___x_1674_; uint8_t v_isShared_1675_; uint8_t v_isSharedCheck_1728_; 
lean_del_object(v___x_1641_);
v_val_1669_ = lean_ctor_get(v___x_1668_, 0);
lean_inc(v_val_1669_);
lean_dec_ref_known(v___x_1668_, 1);
v_str_1670_ = lean_ctor_get(v_val_1669_, 0);
v_startPos_1671_ = lean_ctor_get(v_val_1669_, 1);
v_stopPos_1672_ = lean_ctor_get(v_val_1669_, 2);
v_isSharedCheck_1728_ = !lean_is_exclusive(v_val_1669_);
if (v_isSharedCheck_1728_ == 0)
{
v___x_1674_ = v_val_1669_;
v_isShared_1675_ = v_isSharedCheck_1728_;
goto v_resetjp_1673_;
}
else
{
lean_inc(v_stopPos_1672_);
lean_inc(v_startPos_1671_);
lean_inc(v_str_1670_);
lean_dec(v_val_1669_);
v___x_1674_ = lean_box(0);
v_isShared_1675_ = v_isSharedCheck_1728_;
goto v_resetjp_1673_;
}
v_resetjp_1673_:
{
lean_object* v___x_1676_; lean_object* v___x_1677_; lean_object* v___x_1678_; lean_object* v___x_1680_; 
v___x_1676_ = lean_string_utf8_extract(v_str_1670_, v_startPos_1671_, v_stopPos_1672_);
lean_dec(v_stopPos_1672_);
lean_dec_ref(v_str_1670_);
v___x_1677_ = lean_unsigned_to_nat(0u);
v___x_1678_ = lean_string_utf8_byte_size(v___x_1676_);
lean_inc_ref(v___x_1676_);
if (v_isShared_1675_ == 0)
{
lean_ctor_set(v___x_1674_, 2, v___x_1678_);
lean_ctor_set(v___x_1674_, 1, v___x_1677_);
lean_ctor_set(v___x_1674_, 0, v___x_1676_);
v___x_1680_ = v___x_1674_;
goto v_reusejp_1679_;
}
else
{
lean_object* v_reuseFailAlloc_1727_; 
v_reuseFailAlloc_1727_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1727_, 0, v___x_1676_);
lean_ctor_set(v_reuseFailAlloc_1727_, 1, v___x_1677_);
lean_ctor_set(v_reuseFailAlloc_1727_, 2, v___x_1678_);
v___x_1680_ = v_reuseFailAlloc_1727_;
goto v_reusejp_1679_;
}
v_reusejp_1679_:
{
lean_object* v___x_1681_; lean_object* v___x_1682_; lean_object* v___x_1683_; lean_object* v___x_1684_; lean_object* v___x_1685_; 
v___x_1681_ = lp_mathlib_String_Slice_Pos_revSkipWhile___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__8(v___x_1680_, v___x_1678_);
lean_dec_ref(v___x_1680_);
v___x_1682_ = lean_string_utf8_extract_fast(v___x_1676_, v___x_1677_, v___x_1681_);
lean_dec(v___x_1681_);
lean_dec_ref(v___x_1676_);
v___x_1683_ = ((lean_object*)(lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5___closed__0));
v___x_1684_ = lean_box(0);
v___x_1685_ = l_String_splitOnAux(v___x_1682_, v___x_1683_, v___x_1677_, v___x_1677_, v___x_1677_, v___x_1684_);
lean_dec_ref(v___x_1682_);
if (lean_obj_tag(v___x_1685_) == 1)
{
lean_object* v_tail_1686_; 
v_tail_1686_ = lean_ctor_get(v___x_1685_, 1);
lean_inc(v_tail_1686_);
if (lean_obj_tag(v_tail_1686_) == 1)
{
lean_object* v_head_1687_; lean_object* v___x_1689_; uint8_t v_isShared_1690_; uint8_t v_isSharedCheck_1725_; 
lean_del_object(v___x_1656_);
v_head_1687_ = lean_ctor_get(v___x_1685_, 0);
v_isSharedCheck_1725_ = !lean_is_exclusive(v___x_1685_);
if (v_isSharedCheck_1725_ == 0)
{
lean_object* v_unused_1726_; 
v_unused_1726_ = lean_ctor_get(v___x_1685_, 1);
lean_dec(v_unused_1726_);
v___x_1689_ = v___x_1685_;
v_isShared_1690_ = v_isSharedCheck_1725_;
goto v_resetjp_1688_;
}
else
{
lean_inc(v_head_1687_);
lean_dec(v___x_1685_);
v___x_1689_ = lean_box(0);
v_isShared_1690_ = v_isSharedCheck_1725_;
goto v_resetjp_1688_;
}
v_resetjp_1688_:
{
lean_object* v___x_1691_; lean_object* v___x_1692_; lean_object* v___x_1693_; lean_object* v___x_1694_; lean_object* v___x_1695_; lean_object* v___x_1696_; lean_object* v___x_1697_; lean_object* v___x_1698_; lean_object* v___x_1700_; 
v___x_1691_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__2___closed__0));
v___x_1692_ = lean_string_utf8_byte_size(v_head_1687_);
v___x_1693_ = lean_unsigned_to_nat(1u);
v___x_1694_ = lean_nat_add(v___x_1692_, v___x_1693_);
v___x_1695_ = lean_nat_add(v___x_1694_, v_startPos_1671_);
lean_dec(v_startPos_1671_);
lean_dec(v___x_1694_);
lean_inc(v_head_1687_);
v___x_1696_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1696_, 0, v_head_1687_);
lean_ctor_set(v___x_1696_, 1, v___x_1677_);
lean_ctor_set(v___x_1696_, 2, v___x_1692_);
v___x_1697_ = lp_mathlib_String_Slice_Pos_revSkipWhile___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__10(v___x_1644_, v___x_1696_, v___x_1692_);
lean_dec_ref_known(v___x_1696_, 3);
v___x_1698_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1698_, 0, v_head_1687_);
lean_ctor_set(v___x_1698_, 1, v___x_1697_);
lean_ctor_set(v___x_1698_, 2, v___x_1692_);
if (v_isShared_1690_ == 0)
{
lean_ctor_set_tag(v___x_1689_, 0);
lean_ctor_set(v___x_1689_, 1, v___x_1698_);
lean_ctor_set(v___x_1689_, 0, v___x_1695_);
v___x_1700_ = v___x_1689_;
goto v_reusejp_1699_;
}
else
{
lean_object* v_reuseFailAlloc_1724_; 
v_reuseFailAlloc_1724_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1724_, 0, v___x_1695_);
lean_ctor_set(v_reuseFailAlloc_1724_, 1, v___x_1698_);
v___x_1700_ = v_reuseFailAlloc_1724_;
goto v_reusejp_1699_;
}
v_reusejp_1699_:
{
lean_object* v___x_1701_; lean_object* v___x_1702_; lean_object* v_a_1703_; lean_object* v___x_1704_; lean_object* v___x_1705_; lean_object* v_fst_1706_; lean_object* v___x_1707_; lean_object* v___x_1708_; lean_object* v___x_1709_; lean_object* v___x_1710_; lean_object* v___x_1711_; lean_object* v___x_1712_; size_t v_sz_1713_; size_t v___x_1714_; lean_object* v___x_1715_; 
v___x_1701_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1701_, 0, v___x_1691_);
lean_ctor_set(v___x_1701_, 1, v___x_1700_);
v___x_1702_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__11___redArg(v___x_1644_, v_tail_1686_, v___x_1701_);
lean_dec_ref_known(v_tail_1686_, 2);
v_a_1703_ = lean_ctor_get(v___x_1702_, 0);
lean_inc(v_a_1703_);
lean_dec_ref(v___x_1702_);
lean_inc(v___x_1667_);
v___x_1704_ = lp_mathlib_Lean_Syntax_filter(v___x_1667_, v___f_1633_);
v___x_1705_ = lean_array_get_size(v___x_1704_);
v_fst_1706_ = lean_ctor_get(v_a_1703_, 0);
lean_inc(v_fst_1706_);
lean_dec(v_a_1703_);
v___x_1707_ = lp_mathlib_Array_filterMapM___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__12(v___y_1666_, v___x_1704_, v___x_1677_, v___x_1705_);
lean_dec_ref(v___x_1704_);
v___x_1708_ = lp_mathlib_Lean_Syntax_filterMap___redArg(v___x_1667_, v___f_1664_);
v___x_1709_ = lean_obj_once(&lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__2___closed__2, &lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__2___closed__2_once, _init_lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__2___closed__2);
v___x_1710_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__13(v___x_1709_, v___x_1708_);
lean_dec_ref(v___x_1708_);
v___x_1711_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__13(v___x_1710_, v___x_1707_);
lean_dec_ref(v___x_1707_);
v___x_1712_ = lean_box(0);
v_sz_1713_ = lean_array_size(v_fst_1706_);
v___x_1714_ = ((size_t)0ULL);
v___x_1715_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__18(v___x_1644_, v___x_1711_, v_fst_1706_, v_sz_1713_, v___x_1714_, v___x_1712_, v___y_1635_, v___y_1636_);
lean_dec(v_fst_1706_);
lean_dec_ref(v___x_1711_);
if (lean_obj_tag(v___x_1715_) == 0)
{
lean_object* v___x_1717_; uint8_t v_isShared_1718_; uint8_t v_isSharedCheck_1722_; 
v_isSharedCheck_1722_ = !lean_is_exclusive(v___x_1715_);
if (v_isSharedCheck_1722_ == 0)
{
lean_object* v_unused_1723_; 
v_unused_1723_ = lean_ctor_get(v___x_1715_, 0);
lean_dec(v_unused_1723_);
v___x_1717_ = v___x_1715_;
v_isShared_1718_ = v_isSharedCheck_1722_;
goto v_resetjp_1716_;
}
else
{
lean_dec(v___x_1715_);
v___x_1717_ = lean_box(0);
v_isShared_1718_ = v_isSharedCheck_1722_;
goto v_resetjp_1716_;
}
v_resetjp_1716_:
{
lean_object* v___x_1720_; 
if (v_isShared_1718_ == 0)
{
lean_ctor_set(v___x_1717_, 0, v___x_1712_);
v___x_1720_ = v___x_1717_;
goto v_reusejp_1719_;
}
else
{
lean_object* v_reuseFailAlloc_1721_; 
v_reuseFailAlloc_1721_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1721_, 0, v___x_1712_);
v___x_1720_ = v_reuseFailAlloc_1721_;
goto v_reusejp_1719_;
}
v_reusejp_1719_:
{
return v___x_1720_;
}
}
}
else
{
return v___x_1715_;
}
}
}
}
else
{
lean_dec(v_tail_1686_);
lean_dec_ref_known(v___x_1685_, 2);
lean_dec(v_startPos_1671_);
lean_dec(v___x_1667_);
lean_dec_ref(v___f_1664_);
lean_dec_ref(v___f_1633_);
goto v___jp_1658_;
}
}
else
{
lean_dec(v___x_1685_);
lean_dec(v_startPos_1671_);
lean_dec(v___x_1667_);
lean_dec_ref(v___f_1664_);
lean_dec_ref(v___f_1633_);
goto v___jp_1658_;
}
}
}
}
else
{
lean_object* v___x_1729_; lean_object* v___x_1731_; 
lean_dec(v___x_1668_);
lean_dec(v___x_1667_);
lean_dec_ref(v___f_1664_);
lean_del_object(v___x_1656_);
lean_dec_ref(v___f_1633_);
v___x_1729_ = lean_box(0);
if (v_isShared_1642_ == 0)
{
lean_ctor_set(v___x_1641_, 0, v___x_1729_);
v___x_1731_ = v___x_1641_;
goto v_reusejp_1730_;
}
else
{
lean_object* v_reuseFailAlloc_1732_; 
v_reuseFailAlloc_1732_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1732_, 0, v___x_1729_);
v___x_1731_ = v_reuseFailAlloc_1732_;
goto v_reusejp_1730_;
}
v_reusejp_1730_:
{
return v___x_1731_;
}
}
}
else
{
lean_object* v___x_1733_; lean_object* v___x_1735_; 
lean_dec_ref(v___f_1664_);
lean_del_object(v___x_1656_);
lean_dec(v_stx_1634_);
lean_dec_ref(v___f_1633_);
v___x_1733_ = lean_box(0);
if (v_isShared_1642_ == 0)
{
lean_ctor_set(v___x_1641_, 0, v___x_1733_);
v___x_1735_ = v___x_1641_;
goto v_reusejp_1734_;
}
else
{
lean_object* v_reuseFailAlloc_1736_; 
v_reuseFailAlloc_1736_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1736_, 0, v___x_1733_);
v___x_1735_ = v_reuseFailAlloc_1736_;
goto v_reusejp_1734_;
}
v_reusejp_1734_:
{
return v___x_1735_;
}
}
}
}
}
else
{
lean_object* v___x_1740_; lean_object* v___x_1742_; 
lean_dec(v_stx_1634_);
lean_dec_ref(v___f_1633_);
v___x_1740_ = lean_box(0);
if (v_isShared_1642_ == 0)
{
lean_ctor_set(v___x_1641_, 0, v___x_1740_);
v___x_1742_ = v___x_1641_;
goto v_reusejp_1741_;
}
else
{
lean_object* v_reuseFailAlloc_1743_; 
v_reuseFailAlloc_1743_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1743_, 0, v___x_1740_);
v___x_1742_ = v_reuseFailAlloc_1743_;
goto v_reusejp_1741_;
}
v_reusejp_1741_:
{
return v___x_1742_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__2___boxed(lean_object* v___f_1745_, lean_object* v_stx_1746_, lean_object* v___y_1747_, lean_object* v___y_1748_, lean_object* v___y_1749_){
_start:
{
lean_object* v_res_1750_; 
v_res_1750_ = lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter___lam__2(v___f_1745_, v_stx_1746_, v___y_1747_, v___y_1748_);
lean_dec(v___y_1748_);
lean_dec_ref(v___y_1747_);
return v_res_1750_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__0_spec__0(lean_object* v_o_1767_, lean_object* v___y_1768_, lean_object* v___y_1769_){
_start:
{
lean_object* v___x_1771_; 
v___x_1771_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__0_spec__0___redArg(v_o_1767_, v___y_1769_);
return v___x_1771_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__0_spec__0___boxed(lean_object* v_o_1772_, lean_object* v___y_1773_, lean_object* v___y_1774_, lean_object* v___y_1775_){
_start:
{
lean_object* v_res_1776_; 
v_res_1776_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__0_spec__0(v_o_1772_, v___y_1773_, v___y_1774_);
lean_dec(v___y_1774_);
lean_dec_ref(v___y_1773_);
return v_res_1776_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__6(lean_object* v_00_u03b2_1777_, lean_object* v_m_1778_, lean_object* v_a_1779_, lean_object* v_b_1780_){
_start:
{
lean_object* v___x_1781_; 
v___x_1781_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__6___redArg(v_m_1778_, v_a_1779_, v_b_1780_);
return v___x_1781_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__7(lean_object* v_00_u03b2_1782_, lean_object* v_m_1783_, lean_object* v_a_1784_){
_start:
{
uint8_t v___x_1785_; 
v___x_1785_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__7___redArg(v_m_1783_, v_a_1784_);
return v___x_1785_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__7___boxed(lean_object* v_00_u03b2_1786_, lean_object* v_m_1787_, lean_object* v_a_1788_){
_start:
{
uint8_t v_res_1789_; lean_object* v_r_1790_; 
v_res_1789_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__7(v_00_u03b2_1786_, v_m_1787_, v_a_1788_);
lean_dec(v_a_1788_);
lean_dec_ref(v_m_1787_);
v_r_1790_ = lean_box(v_res_1789_);
return v_r_1790_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__11(uint8_t v___x_1791_, lean_object* v_as_1792_, lean_object* v_as_x27_1793_, lean_object* v_b_1794_, lean_object* v_a_1795_, lean_object* v___y_1796_, lean_object* v___y_1797_){
_start:
{
lean_object* v___x_1799_; 
v___x_1799_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__11___redArg(v___x_1791_, v_as_x27_1793_, v_b_1794_);
return v___x_1799_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__11___boxed(lean_object* v___x_1800_, lean_object* v_as_1801_, lean_object* v_as_x27_1802_, lean_object* v_b_1803_, lean_object* v_a_1804_, lean_object* v___y_1805_, lean_object* v___y_1806_, lean_object* v___y_1807_){
_start:
{
uint8_t v___x_19504__boxed_1808_; lean_object* v_res_1809_; 
v___x_19504__boxed_1808_ = lean_unbox(v___x_1800_);
v_res_1809_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__11(v___x_19504__boxed_1808_, v_as_1801_, v_as_x27_1802_, v_b_1803_, v_a_1804_, v___y_1805_, v___y_1806_);
lean_dec(v___y_1806_);
lean_dec_ref(v___y_1805_);
lean_dec(v_as_x27_1802_);
lean_dec(v_as_1801_);
return v_res_1809_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5_spec__8(lean_object* v_s_1810_, lean_object* v_inst_1811_, lean_object* v_R_1812_, lean_object* v_a_1813_, uint8_t v_b_1814_, lean_object* v_c_1815_){
_start:
{
uint8_t v___x_1816_; 
v___x_1816_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5_spec__8___redArg(v_s_1810_, v_a_1813_, v_b_1814_);
return v___x_1816_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5_spec__8___boxed(lean_object* v_s_1817_, lean_object* v_inst_1818_, lean_object* v_R_1819_, lean_object* v_a_1820_, lean_object* v_b_1821_, lean_object* v_c_1822_){
_start:
{
uint8_t v_b_boxed_1823_; uint8_t v_res_1824_; lean_object* v_r_1825_; 
v_b_boxed_1823_ = lean_unbox(v_b_1821_);
v_res_1824_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__5_spec__8(v_s_1817_, v_inst_1818_, v_R_1819_, v_a_1820_, v_b_boxed_1823_, v_c_1822_);
lean_dec_ref(v_s_1817_);
v_r_1825_ = lean_box(v_res_1824_);
return v_r_1825_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__6_spec__10(lean_object* v_00_u03b2_1826_, lean_object* v_a_1827_, lean_object* v_x_1828_){
_start:
{
uint8_t v___x_1829_; 
v___x_1829_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__6_spec__10___redArg(v_a_1827_, v_x_1828_);
return v___x_1829_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__6_spec__10___boxed(lean_object* v_00_u03b2_1830_, lean_object* v_a_1831_, lean_object* v_x_1832_){
_start:
{
uint8_t v_res_1833_; lean_object* v_r_1834_; 
v_res_1833_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__6_spec__10(v_00_u03b2_1830_, v_a_1831_, v_x_1832_);
lean_dec(v_x_1832_);
lean_dec(v_a_1831_);
v_r_1834_ = lean_box(v_res_1833_);
return v_r_1834_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__6_spec__11(lean_object* v_00_u03b2_1835_, lean_object* v_data_1836_){
_start:
{
lean_object* v___x_1837_; 
v___x_1837_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__6_spec__11___redArg(v_data_1836_);
return v___x_1837_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__13_spec__20(lean_object* v_00_u03b2_1838_, lean_object* v_m_1839_, lean_object* v_a_1840_, lean_object* v_b_1841_){
_start:
{
lean_object* v___x_1842_; 
v___x_1842_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__13_spec__20___redArg(v_m_1839_, v_a_1840_, v_b_1841_);
return v___x_1842_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__6_spec__11_spec__13(lean_object* v_00_u03b2_1843_, lean_object* v_i_1844_, lean_object* v_source_1845_, lean_object* v_target_1846_){
_start:
{
lean_object* v___x_1847_; 
v___x_1847_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__6_spec__11_spec__13___redArg(v_i_1844_, v_source_1845_, v_target_1846_);
return v___x_1847_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__13_spec__20_spec__23(lean_object* v_00_u03b2_1848_, lean_object* v_a_1849_, lean_object* v_x_1850_){
_start:
{
uint8_t v___x_1851_; 
v___x_1851_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__13_spec__20_spec__23___redArg(v_a_1849_, v_x_1850_);
return v___x_1851_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__13_spec__20_spec__23___boxed(lean_object* v_00_u03b2_1852_, lean_object* v_a_1853_, lean_object* v_x_1854_){
_start:
{
uint8_t v_res_1855_; lean_object* v_r_1856_; 
v_res_1855_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__13_spec__20_spec__23(v_00_u03b2_1852_, v_a_1853_, v_x_1854_);
lean_dec(v_x_1854_);
lean_dec_ref(v_a_1853_);
v_r_1856_ = lean_box(v_res_1855_);
return v_r_1856_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__13_spec__20_spec__24(lean_object* v_00_u03b2_1857_, lean_object* v_data_1858_){
_start:
{
lean_object* v___x_1859_; 
v___x_1859_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__13_spec__20_spec__24___redArg(v_data_1858_);
return v___x_1859_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29_spec__36(lean_object* v_msgData_1860_, lean_object* v___y_1861_, lean_object* v___y_1862_){
_start:
{
lean_object* v___x_1864_; 
v___x_1864_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29_spec__36___redArg(v_msgData_1860_, v___y_1862_);
return v___x_1864_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29_spec__36___boxed(lean_object* v_msgData_1865_, lean_object* v___y_1866_, lean_object* v___y_1867_, lean_object* v___y_1868_){
_start:
{
lean_object* v_res_1869_; 
v_res_1869_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__15_spec__24_spec__29_spec__36(v_msgData_1865_, v___y_1866_, v___y_1867_);
lean_dec(v___y_1867_);
lean_dec_ref(v___y_1866_);
return v_res_1869_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__6_spec__11_spec__13_spec__27(lean_object* v_00_u03b2_1870_, lean_object* v_x_1871_, lean_object* v_x_1872_){
_start:
{
lean_object* v___x_1873_; 
v___x_1873_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__6_spec__11_spec__13_spec__27___redArg(v_x_1871_, v_x_1872_);
return v___x_1873_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__13_spec__20_spec__24_spec__32(lean_object* v_00_u03b2_1874_, lean_object* v_i_1875_, lean_object* v_source_1876_, lean_object* v_target_1877_){
_start:
{
lean_object* v___x_1878_; 
v___x_1878_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__13_spec__20_spec__24_spec__32___redArg(v_i_1875_, v_source_1876_, v_target_1877_);
return v___x_1878_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__13_spec__20_spec__24_spec__32_spec__35(lean_object* v_00_u03b2_1879_, lean_object* v_x_1880_, lean_object* v_x_1881_){
_start:
{
lean_object* v___x_1882_; 
v___x_1882_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Linter_EmptyLine_emptyLineLinter_spec__13_spec__20_spec__24_spec__32_spec__35___redArg(v_x_1880_, v_x_1881_);
return v___x_1882_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_EmptyLine_initFn_00___x40_Mathlib_Tactic_Linter_EmptyLine_1134681162____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_1884_; lean_object* v___x_1885_; 
v___x_1884_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_EmptyLine_emptyLineLinter));
v___x_1885_ = l_Lean_Elab_Command_addLinter(v___x_1884_);
return v___x_1885_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_EmptyLine_initFn_00___x40_Mathlib_Tactic_Linter_EmptyLine_1134681162____hygCtx___hyg_2____boxed(lean_object* v_a_1886_){
_start:
{
lean_object* v_res_1887_; 
v_res_1887_ = lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_EmptyLine_initFn_00___x40_Mathlib_Tactic_Linter_EmptyLine_1134681162____hygCtx___hyg_2_();
return v_res_1887_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Parser_Command(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_EmptyLine(uint8_t builtin) {
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
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_Header(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Linter_EmptyLine(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_Header(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_EmptyLine_35929507____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Linter_linter_style_emptyLine = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Linter_linter_style_emptyLine);
lean_dec_ref(res);
lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines = _init_lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines();
lean_mark_persistent(lp_mathlib_Mathlib_Linter_EmptyLine_AllowEmptyLines);
lp_mathlib_Mathlib_Linter_EmptyLine_SkippedFileSegments = _init_lp_mathlib_Mathlib_Linter_EmptyLine_SkippedFileSegments();
lean_mark_persistent(lp_mathlib_Mathlib_Linter_EmptyLine_SkippedFileSegments);
res = lp_mathlib___private_Mathlib_Tactic_Linter_EmptyLine_0__Mathlib_Linter_EmptyLine_initFn_00___x40_Mathlib_Tactic_Linter_EmptyLine_1134681162____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_Header(uint8_t builtin);
lean_object* initialize_Lean_Parser_Command(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Linter_EmptyLine(uint8_t builtin) {
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
res = initialize_Lean_Parser_Command(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_EmptyLine(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Linter_EmptyLine(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Linter_EmptyLine(builtin);
}
#ifdef __cplusplus
}
#endif
