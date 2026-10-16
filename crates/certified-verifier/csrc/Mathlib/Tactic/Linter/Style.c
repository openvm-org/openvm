// Lean compiler output
// Module: Mathlib.Tactic.Linter.Style
// Imports: public import Init public meta import Init public meta import Lean.Elab.Command public meta import Mathlib.Tactic.Linter.Header public import Lean.Parser.Command public import Mathlib.Tactic.DeclarationNames public import Batteries.Tactic.Lint.Basic public import Lean.Parser.Module
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
lean_object* lean_st_ref_get(lean_object*);
extern lean_object* l_Lean_Linter_linterSetsExt;
extern lean_object* l_Lean_Linter_instInhabitedLinterSetsState_default;
lean_object* l_Lean_PersistentEnvExtension_getState___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Lean_MessageLog_add(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(lean_object*);
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
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_Command_instInhabitedScope_default;
lean_object* l_List_head_x21___redArg(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_register_option(lean_object*, lean_object*);
uint8_t l_Lean_Linter_getLinterValue(lean_object*, lean_object*);
uint8_t l_Lean_MessageLog_hasErrors(lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
size_t lean_usize_add(size_t, size_t);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getId(lean_object*);
uint8_t l_Lean_Name_hasMacroScopes(lean_object*);
lean_object* l_Lean_Syntax_getKind(lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* l_Lean_Name_toString(lean_object*, uint8_t);
lean_object* l_String_splitOnAux(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_lengthTR___redArg(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* l_Lean_MessageData_note(lean_object*);
extern lean_object* l_Lean_Linter_linterMessageTag;
lean_object* l_Lean_Elab_Command_getScope___redArg(lean_object*);
lean_object* l_Lean_Elab_Command_getRef___redArg(lean_object*);
lean_object* l_Lean_Syntax_find_x3f(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
lean_object* l_Lean_TSyntax_getId(lean_object*);
lean_object* l_Lean_Syntax_ofRange(lean_object*, uint8_t);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
lean_object* l_Lean_mkIdentFrom(lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_Syntax_getRange_x3f(lean_object*, uint8_t);
extern lean_object* l_Lean_Syntax_instInhabitedRange_default;
lean_object* l_Lean_withSetOptionIn___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_addLinter(lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Substring_Raw_splitOn(lean_object*, lean_object*);
lean_object* l_Substring_Raw_nextn(lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_String_instInhabitedSlice;
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
uint32_t lean_string_utf8_get_fast(lean_object*, lean_object*);
uint8_t lean_uint32_dec_eq(uint32_t, uint32_t);
lean_object* lean_string_utf8_next_fast(lean_object*, lean_object*);
uint8_t lean_string_is_valid_pos(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lean_string_utf8_extract(lean_object*, lean_object*, lean_object*);
lean_object* lean_string_utf8_byte_size(lean_object*);
uint8_t lean_string_memcmp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getSubstring_x3f(lean_object*, uint8_t, uint8_t);
lean_object* l_Lean_Name_mkStr6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Parser_parseHeader(lean_object*);
lean_object* lean_io_error_to_string(lean_object*);
lean_object* l_Array_instInhabited(lean_object*);
lean_object* lean_array_get_size(lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_Name_getRoot(lean_object*);
lean_object* l_Lean_Name_components(lean_object*);
lean_object* l_String_intercalate(lean_object*, lean_object*);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l_Lean_ConstantInfo_type(lean_object*);
uint8_t l_Lean_Expr_isConstOf(lean_object*, lean_object*);
lean_object* l_String_Slice_posLE(lean_object*, lean_object*);
uint8_t lean_string_get_byte_fast(lean_object*, lean_object*);
uint8_t lean_uint8_dec_eq(uint8_t, uint8_t);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* l_String_Slice_posGE___redArg(lean_object*, lean_object*);
lean_object* l_String_Slice_Pattern_ForwardSliceSearcher_buildTable(lean_object*);
uint8_t l_Lean_Name_isPrefixOf(lean_object*, lean_object*);
lean_object* l_List_getLast_x3f___redArg(lean_object*);
lean_object* l_Lean_MessageData_ofConstName(lean_object*, uint8_t);
lean_object* l_Lean_Environment_find_x3f(lean_object*, lean_object*, uint8_t);
extern lean_object* l_Lean_instInhabitedConstantInfo_default;
uint8_t l_Lean_ConstantInfo_isDefinition(lean_object*);
uint8_t l_Lean_isPrivateName(lean_object*);
uint8_t lp_batteries_Lean_Environment_isAutoDecl(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* lean_nat_div(lean_object*, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
uint8_t l_Lean_Parser_isTerminalCommand(lean_object*);
lean_object* l_Lean_Environment_header(lean_object*);
uint8_t l_Lean_Syntax_matchesIdent(lean_object*, lean_object*);
lean_object* l_Lean_TSyntax_getNat(lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_getScopes___redArg(lean_object*);
lean_object* lean_array_mk(lean_object*);
lean_object* lean_array_pop(lean_object*);
lean_object* lean_array_to_list(lean_object*);
uint8_t l_Lean_Name_isAnonymous(lean_object*);
uint8_t l_List_isEmpty___redArg(lean_object*);
lean_object* l_Lean_Elab_Tactic_getGoals___redArg(lean_object*);
lean_object* l_Lean_MVarId_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_evalTactic(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_expr_eqv(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_throwNoGoalsToBeSolved___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getTrailing_x3f(lean_object*);
lean_object* lean_string_utf8_next(lean_object*, lean_object*);
uint32_t lean_string_utf8_get(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "style"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "setOption"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(186, 218, 113, 226, 101, 176, 32, 79)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(105, 62, 218, 153, 100, 142, 29, 251)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(87, 45, 214, 20, 57, 151, 205, 161)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 30, .m_capacity = 30, .m_length = 29, .m_data = "enable the `setOption` linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(120, 131, 127, 204, 79, 169, 80, 92)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(101, 237, 90, 120, 51, 59, 46, 172)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(98, 189, 128, 85, 154, 50, 252, 160)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(240, 98, 53, 178, 254, 101, 252, 200)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_linter_style_setOption;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Command"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "set_option"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__4_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__2_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__4_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__3_value),LEAN_SCALAR_PTR_LITERAL(216, 223, 149, 245, 150, 86, 134, 198)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__6_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__6_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__5_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__6_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__3_value),LEAN_SCALAR_PTR_LITERAL(73, 146, 13, 195, 238, 126, 231, 237)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__6_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__8_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__8_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__7_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__8_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__3_value),LEAN_SCALAR_PTR_LITERAL(168, 188, 37, 234, 109, 221, 79, 218)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__8_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__9_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__10_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Linter_Style_setOption_isSetOption(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_Style_setOption_isSetOption___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__6(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__6___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4___lam__0___closed__0_value;
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4___lam__0___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__5___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__5___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__5___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__5___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__5___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__5___redArg___closed__2;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__5___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__5___redArg___closed__3;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__5___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__5___redArg___closed__4;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__5___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__5___redArg___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__5___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 46, .m_capacity = 46, .m_length = 45, .m_data = "This linter can be disabled with `set_option "};
static const lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__1___closed__0 = (const lean_object*)&lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__1___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__1___closed__1;
static const lean_string_object lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = " false`"};
static const lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__1___closed__2 = (const lean_object*)&lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__1___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__1___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__4(uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__4___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_elem___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_elem___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__2___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Linter_Style_setOption_isSetOption___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "Unscoped option "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__1_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__2;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 85, .m_capacity = 85, .m_length = 84, .m_data = " is not allowed:\nPlease scope this to individual declarations, as in\n```\nset_option "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__3_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__4;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 73, .m_capacity = 73, .m_length = 72, .m_data = " in\n-- comment explaining why this is necessary\nexample : ... := ...\n```"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__5_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__6;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "debug"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__7_value),LEAN_SCALAR_PTR_LITERAL(40, 215, 222, 176, 152, 52, 0, 225)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__8_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "pp"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__9_value),LEAN_SCALAR_PTR_LITERAL(249, 51, 192, 169, 230, 180, 160, 93)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__10_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "profiler"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__11_value),LEAN_SCALAR_PTR_LITERAL(55, 199, 104, 147, 160, 34, 129, 33)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__12_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__13_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__13_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__14_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__12_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__14_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__15_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__10_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__15_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__16_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__8_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__16_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__17_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "maxHeartbeats"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__18 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__18_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__18_value),LEAN_SCALAR_PTR_LITERAL(163, 202, 216, 251, 148, 187, 135, 206)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__19 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__19_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "flexible"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__20 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__20_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__21_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(186, 218, 113, 226, 101, 176, 32, 79)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__21_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__20_value),LEAN_SCALAR_PTR_LITERAL(87, 87, 124, 175, 51, 209, 186, 58)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__21 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__21_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "backward"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__22 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__22_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "inferInstanceAs"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__23 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__23_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "wrap"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__24 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__24_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "reuseSubInstances"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__25 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__25_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__26_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__22_value),LEAN_SCALAR_PTR_LITERAL(77, 196, 98, 49, 58, 220, 29, 220)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__26_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__26_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__23_value),LEAN_SCALAR_PTR_LITERAL(6, 203, 50, 196, 213, 242, 67, 10)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__26_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__26_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__24_value),LEAN_SCALAR_PTR_LITERAL(208, 252, 45, 86, 202, 182, 131, 2)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__26_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__25_value),LEAN_SCALAR_PTR_LITERAL(10, 196, 243, 125, 230, 240, 101, 207)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__26 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__26_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 127, .m_capacity = 127, .m_length = 126, .m_data = "The `backward.inferInstanceAs.wrap.reuseSubInstances` option marks the introduction of technical debt, so please don't use it."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__27 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__27_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__27_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__28 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__28_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__29_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__29;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 32, .m_capacity = 32, .m_length = 31, .m_data = "Setting options starting with '"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__30 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__30_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__31_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__31;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "', '"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__32 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__32_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 152, .m_capacity = 152, .m_length = 151, .m_data = "' is only intended for development and not for final code. If you intend to submit this contribution to the Mathlib project, please remove 'set_option "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__33 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__33_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__34_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__34;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "'."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__35 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__35_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__36_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__36;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__0_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_withSetOptionIn___boxed, .m_arity = 6, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__0_value)} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__2_value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__4_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__7_value),LEAN_SCALAR_PTR_LITERAL(191, 70, 156, 159, 11, 54, 216, 94)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__5_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(37, 204, 154, 235, 250, 222, 148, 114)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__6_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Style"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__6_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__7_value),LEAN_SCALAR_PTR_LITERAL(209, 156, 94, 142, 17, 36, 158, 124)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(148, 36, 190, 198, 95, 170, 234, 115)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__9_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(133, 161, 131, 209, 60, 194, 20, 234)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__10_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(71, 208, 143, 109, 249, 238, 16, 214)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__11_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__7_value),LEAN_SCALAR_PTR_LITERAL(91, 145, 50, 232, 6, 147, 242, 15)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__12_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__12_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(61, 217, 103, 205, 223, 187, 47, 248)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__13_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "setOptionLinter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__14_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__13_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__14_value),LEAN_SCALAR_PTR_LITERAL(164, 125, 101, 250, 140, 218, 90, 186)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__15_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__15_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__16_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__16_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__5(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_initFn_00___x40_Mathlib_Tactic_Linter_Style_3512398344____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_initFn_00___x40_Mathlib_Tactic_Linter_Style_3512398344____hygCtx___hyg_2____boxed(lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_3409032198____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "missingEnd"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_3409032198____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_3409032198____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_3409032198____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(186, 218, 113, 226, 101, 176, 32, 79)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_3409032198____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_3409032198____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(105, 62, 218, 153, 100, 142, 29, 251)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_3409032198____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_3409032198____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_3409032198____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(37, 188, 88, 204, 36, 17, 29, 235)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_3409032198____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_3409032198____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Style_3409032198____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 30, .m_capacity = 30, .m_length = 29, .m_data = "enable the missing end linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Style_3409032198____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Style_3409032198____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Style_3409032198____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Style_3409032198____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Style_3409032198____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Style_3409032198____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_3409032198____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_3409032198____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_3409032198____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(120, 131, 127, 204, 79, 169, 80, 92)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_3409032198____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_3409032198____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(101, 237, 90, 120, 51, 59, 46, 172)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_3409032198____hygCtx___hyg_4__value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_3409032198____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(98, 189, 128, 85, 154, 50, 252, 160)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_3409032198____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_3409032198____hygCtx___hyg_4__value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_3409032198____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(98, 49, 62, 174, 48, 19, 51, 119)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_3409032198____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_3409032198____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_3409032198____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_3409032198____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_linter_style_missingEnd;
LEAN_EXPORT lean_object* lp_mathlib_List_dropWhile___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter_spec__0(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_dropWhile___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter_spec__1(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_List_foldl___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "\n\nend"};
static const lean_object* lp_mathlib_List_foldl___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter_spec__2___closed__0 = (const lean_object*)&lp_mathlib_List_foldl___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter_spec__2___closed__0_value;
static const lean_string_object lp_mathlib_List_foldl___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter_spec__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = " "};
static const lean_object* lp_mathlib_List_foldl___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter_spec__2___closed__1 = (const lean_object*)&lp_mathlib_List_foldl___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter_spec__2___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter_spec__2___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "eoi"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___lam__0___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___lam__0___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___lam__0___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___lam__0___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___lam__0___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__2_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___lam__0___closed__1_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(26, 206, 8, 118, 9, 188, 233, 7)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___lam__0___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___lam__0___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 45, .m_capacity = 45, .m_length = 44, .m_data = "unclosed sections or namespaces; expected: '"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___lam__0___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___lam__0___closed__2_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___lam__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___lam__0___closed__3;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "'"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___lam__0___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___lam__0___closed__4_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___lam__0___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___lam__0___closed__5;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___lam__0___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___closed__0_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_withSetOptionIn___boxed, .m_arity = 6, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___closed__0_value)} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__12_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_3409032198____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(159, 91, 30, 45, 143, 28, 29, 223)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "missingEndLinter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___closed__2_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___closed__3_value),LEAN_SCALAR_PTR_LITERAL(239, 33, 60, 199, 80, 109, 116, 116)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___closed__4_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_initFn_00___x40_Mathlib_Tactic_Linter_Style_3360231377____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_initFn_00___x40_Mathlib_Tactic_Linter_Style_3360231377____hygCtx___hyg_2____boxed(lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_3789867222____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "cdot"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_3789867222____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_3789867222____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_3789867222____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(186, 218, 113, 226, 101, 176, 32, 79)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_3789867222____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_3789867222____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(105, 62, 218, 153, 100, 142, 29, 251)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_3789867222____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_3789867222____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_3789867222____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(13, 125, 98, 226, 154, 238, 151, 97)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_3789867222____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_3789867222____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Style_3789867222____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "enable the `cdot` linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Style_3789867222____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Style_3789867222____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Style_3789867222____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Style_3789867222____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Style_3789867222____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Style_3789867222____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_3789867222____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_3789867222____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_3789867222____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(120, 131, 127, 204, 79, 169, 80, 92)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_3789867222____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_3789867222____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(101, 237, 90, 120, 51, 59, 46, 172)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_3789867222____hygCtx___hyg_4__value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_3789867222____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(98, 189, 128, 85, 154, 50, 252, 160)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_3789867222____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_3789867222____hygCtx___hyg_4__value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_3789867222____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(250, 218, 91, 117, 91, 229, 132, 235)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_3789867222____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_3789867222____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_3789867222____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_3789867222____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_linter_style_cdot;
static const lean_string_object lp_mathlib_Mathlib_Linter_isCDot_x3f___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "cdotTk"};
static const lean_object* lp_mathlib_Mathlib_Linter_isCDot_x3f___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Linter_isCDot_x3f___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_isCDot_x3f___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 1, .m_data = "·"};
static const lean_object* lp_mathlib_Mathlib_Linter_isCDot_x3f___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Linter_isCDot_x3f___closed__1_value;
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Linter_isCDot_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_isCDot_x3f___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_findCDot_spec__1(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_findCDot_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_findCDot___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_findCDot___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_findCDot___closed__0_value;
static const lean_array_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_findCDot___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_findCDot___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_findCDot___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_findCDot(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_findCDot_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_findCDot_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_unwanted__cdot_spec__0(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_unwanted__cdot_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_unwanted__cdot(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__1(lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 59, .m_capacity = 59, .m_length = 57, .m_data = "Please, use '·' (typed as `\\.`) instead of '.' as 'cdot'."};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__0___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__0___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__0(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Substring_Raw_takeWhileAux___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Substring_Raw_takeWhileAux___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__2_spec__2___redArg(lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__2_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__2(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__2___boxed(lean_object*);
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Linter_isCDot_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(117, 126, 44, 217, 38, 3, 69, 145)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___closed__1_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 71, .m_capacity = 71, .m_length = 69, .m_data = "This central dot `·` is isolated; please merge it with the next line."};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___closed__2 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___closed__2_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___closed__3;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "Init.Data.Option.BasicAux"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___closed__4 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___closed__4_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "Option.get!"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___closed__5 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___closed__5_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "value is none"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___closed__6 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___closed__6_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___closed__7;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___closed__8 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___closed__9 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___closed__9_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter___lam__0___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter___closed__0_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_withSetOptionIn___boxed, .m_arity = 6, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter___closed__0_value)} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "cdotLinter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__12_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter___closed__2_value),LEAN_SCALAR_PTR_LITERAL(16, 207, 53, 178, 20, 170, 81, 193)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter___closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter___closed__3_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter___closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter___closed__4_value;
LEAN_EXPORT uint8_t lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__2_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__2_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_Style_1823831825____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_Style_1823831825____hygCtx___hyg_2____boxed(lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_830885783____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "dollarSyntax"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_830885783____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_830885783____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_830885783____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(186, 218, 113, 226, 101, 176, 32, 79)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_830885783____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_830885783____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(105, 62, 218, 153, 100, 142, 29, 251)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_830885783____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_830885783____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_830885783____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(10, 147, 135, 6, 59, 145, 162, 17)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_830885783____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_830885783____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Style_830885783____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 33, .m_capacity = 33, .m_length = 32, .m_data = "enable the `dollarSyntax` linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Style_830885783____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Style_830885783____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Style_830885783____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Style_830885783____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Style_830885783____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Style_830885783____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_830885783____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_830885783____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_830885783____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(120, 131, 127, 204, 79, 169, 80, 92)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_830885783____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_830885783____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(101, 237, 90, 120, 51, 59, 46, 172)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_830885783____hygCtx___hyg_4__value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_830885783____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(98, 189, 128, 85, 154, 50, 252, 160)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_830885783____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_830885783____hygCtx___hyg_4__value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_830885783____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(29, 53, 153, 148, 212, 90, 174, 87)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_830885783____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_830885783____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_830885783____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_830885783____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_linter_style_dollarSyntax;
static const lean_string_object lp_mathlib_Mathlib_Linter_Style_dollarSyntax_findDollarSyntax___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "term_$__"};
static const lean_object* lp_mathlib_Mathlib_Linter_Style_dollarSyntax_findDollarSyntax___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Linter_Style_dollarSyntax_findDollarSyntax___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_Style_dollarSyntax_findDollarSyntax(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Linter_Style_dollarSyntax_findDollarSyntax_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Linter_Style_dollarSyntax_findDollarSyntax_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_dollarSyntax_dollarSyntaxLinter_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 54, .m_capacity = 54, .m_length = 53, .m_data = "Please use '<|' instead of '$' for the pipe operator."};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_dollarSyntax_dollarSyntaxLinter_spec__0___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_dollarSyntax_dollarSyntaxLinter_spec__0___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_dollarSyntax_dollarSyntaxLinter_spec__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_dollarSyntax_dollarSyntaxLinter_spec__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_dollarSyntax_dollarSyntaxLinter_spec__0(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_dollarSyntax_dollarSyntaxLinter_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_dollarSyntax_dollarSyntaxLinter___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_dollarSyntax_dollarSyntaxLinter___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_dollarSyntax_dollarSyntaxLinter___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_dollarSyntax_dollarSyntaxLinter___lam__0___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_dollarSyntax_dollarSyntaxLinter___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_dollarSyntax_dollarSyntaxLinter___closed__0_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_dollarSyntax_dollarSyntaxLinter___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_withSetOptionIn___boxed, .m_arity = 6, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_dollarSyntax_dollarSyntaxLinter___closed__0_value)} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_dollarSyntax_dollarSyntaxLinter___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_dollarSyntax_dollarSyntaxLinter___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_dollarSyntax_dollarSyntaxLinter___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__12_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_830885783____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(144, 177, 75, 231, 182, 226, 234, 251)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_dollarSyntax_dollarSyntaxLinter___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_dollarSyntax_dollarSyntaxLinter___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_dollarSyntax_dollarSyntaxLinter___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "dollarSyntaxLinter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_dollarSyntax_dollarSyntaxLinter___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_dollarSyntax_dollarSyntaxLinter___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_dollarSyntax_dollarSyntaxLinter___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_dollarSyntax_dollarSyntaxLinter___closed__2_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_dollarSyntax_dollarSyntaxLinter___closed__3_value),LEAN_SCALAR_PTR_LITERAL(108, 184, 185, 255, 105, 88, 90, 77)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_dollarSyntax_dollarSyntaxLinter___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_dollarSyntax_dollarSyntaxLinter___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_dollarSyntax_dollarSyntaxLinter___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_dollarSyntax_dollarSyntaxLinter___closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_dollarSyntax_dollarSyntaxLinter___closed__4_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_dollarSyntax_dollarSyntaxLinter___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_dollarSyntax_dollarSyntaxLinter___closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_dollarSyntax_dollarSyntaxLinter = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_dollarSyntax_dollarSyntaxLinter___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_dollarSyntax_initFn_00___x40_Mathlib_Tactic_Linter_Style_1303717341____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_dollarSyntax_initFn_00___x40_Mathlib_Tactic_Linter_Style_1303717341____hygCtx___hyg_2____boxed(lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_450967313____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "lambdaSyntax"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_450967313____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_450967313____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_450967313____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(186, 218, 113, 226, 101, 176, 32, 79)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_450967313____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_450967313____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(105, 62, 218, 153, 100, 142, 29, 251)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_450967313____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_450967313____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_450967313____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(156, 213, 94, 197, 181, 156, 171, 221)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_450967313____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_450967313____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Style_450967313____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 33, .m_capacity = 33, .m_length = 32, .m_data = "enable the `lambdaSyntax` linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Style_450967313____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Style_450967313____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Style_450967313____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Style_450967313____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Style_450967313____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Style_450967313____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_450967313____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_450967313____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_450967313____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(120, 131, 127, 204, 79, 169, 80, 92)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_450967313____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_450967313____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(101, 237, 90, 120, 51, 59, 46, 172)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_450967313____hygCtx___hyg_4__value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_450967313____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(98, 189, 128, 85, 154, 50, 252, 160)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_450967313____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_450967313____hygCtx___hyg_4__value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_450967313____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(179, 132, 37, 226, 121, 172, 157, 144)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_450967313____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_450967313____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_450967313____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_450967313____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_linter_style_lambdaSyntax;
static const lean_string_object lp_mathlib_Mathlib_Linter_Style_lambdaSyntax_findLambdaSyntax___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "fun"};
static const lean_object* lp_mathlib_Mathlib_Linter_Style_lambdaSyntax_findLambdaSyntax___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Linter_Style_lambdaSyntax_findLambdaSyntax___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_Style_lambdaSyntax_findLambdaSyntax(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Linter_Style_lambdaSyntax_findLambdaSyntax_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Linter_Style_lambdaSyntax_findLambdaSyntax_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_lambdaSyntaxLinter_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 1, .m_data = "λ"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_lambdaSyntaxLinter_spec__0___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_lambdaSyntaxLinter_spec__0___closed__0_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_lambdaSyntaxLinter_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 104, .m_capacity = 104, .m_length = 101, .m_data = "Please use 'fun' and not 'λ' to define anonymous functions.\nThe 'λ' syntax is deprecated in mathlib4."};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_lambdaSyntaxLinter_spec__0___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_lambdaSyntaxLinter_spec__0___closed__1_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_lambdaSyntaxLinter_spec__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_lambdaSyntaxLinter_spec__0___closed__2;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_lambdaSyntaxLinter_spec__0(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_lambdaSyntaxLinter_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_lambdaSyntaxLinter___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_lambdaSyntaxLinter___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_lambdaSyntaxLinter___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_lambdaSyntaxLinter___lam__0___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_lambdaSyntaxLinter___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_lambdaSyntaxLinter___closed__0_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_lambdaSyntaxLinter___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_withSetOptionIn___boxed, .m_arity = 6, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_lambdaSyntaxLinter___closed__0_value)} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_lambdaSyntaxLinter___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_lambdaSyntaxLinter___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_lambdaSyntaxLinter___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__12_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_450967313____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 170, 30, 58, 0, 178, 226, 80)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_lambdaSyntaxLinter___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_lambdaSyntaxLinter___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_lambdaSyntaxLinter___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "lambdaSyntaxLinter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_lambdaSyntaxLinter___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_lambdaSyntaxLinter___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_lambdaSyntaxLinter___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_lambdaSyntaxLinter___closed__2_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_lambdaSyntaxLinter___closed__3_value),LEAN_SCALAR_PTR_LITERAL(228, 41, 188, 209, 92, 41, 125, 2)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_lambdaSyntaxLinter___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_lambdaSyntaxLinter___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_lambdaSyntaxLinter___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_lambdaSyntaxLinter___closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_lambdaSyntaxLinter___closed__4_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_lambdaSyntaxLinter___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_lambdaSyntaxLinter___closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_lambdaSyntaxLinter = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_lambdaSyntaxLinter___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_initFn_00___x40_Mathlib_Tactic_Linter_Style_1166937461____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_initFn_00___x40_Mathlib_Tactic_Linter_Style_1166937461____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_695976056____hygCtx___hyg_4__spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_695976056____hygCtx___hyg_4__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_695976056____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "longFile"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_695976056____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_695976056____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_695976056____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(186, 218, 113, 226, 101, 176, 32, 79)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_695976056____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_695976056____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(105, 62, 218, 153, 100, 142, 29, 251)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_695976056____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_695976056____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_695976056____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(62, 22, 134, 123, 169, 68, 60, 115)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_695976056____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_695976056____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Style_695976056____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "enable the longFile linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Style_695976056____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Style_695976056____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Style_695976056____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Style_695976056____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Style_695976056____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Style_695976056____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_695976056____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_695976056____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_695976056____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(120, 131, 127, 204, 79, 169, 80, 92)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_695976056____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_695976056____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(101, 237, 90, 120, 51, 59, 46, 172)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_695976056____hygCtx___hyg_4__value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_695976056____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(98, 189, 128, 85, 154, 50, 252, 160)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_695976056____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_695976056____hygCtx___hyg_4__value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_695976056____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(81, 142, 185, 196, 84, 109, 5, 84)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_695976056____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_695976056____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_695976056____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_695976056____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_linter_style_longFile;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_583422302____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "longFileDefValue"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_583422302____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_583422302____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_583422302____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(186, 218, 113, 226, 101, 176, 32, 79)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_583422302____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_583422302____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(105, 62, 218, 153, 100, 142, 29, 251)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_583422302____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_583422302____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_583422302____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(224, 197, 148, 22, 222, 122, 246, 164)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_583422302____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_583422302____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Style_583422302____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 55, .m_capacity = 55, .m_length = 54, .m_data = "a soft upper bound on the number of lines of each file"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Style_583422302____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Style_583422302____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Style_583422302____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(1500) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Style_583422302____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Style_583422302____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Style_583422302____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_583422302____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_583422302____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_583422302____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(120, 131, 127, 204, 79, 169, 80, 92)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_583422302____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_583422302____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(101, 237, 90, 120, 51, 59, 46, 172)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_583422302____hygCtx___hyg_4__value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_583422302____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(98, 189, 128, 85, 154, 50, 252, 160)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_583422302____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_583422302____hygCtx___hyg_4__value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_583422302____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(71, 186, 119, 56, 21, 134, 28, 73)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_583422302____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_583422302____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_583422302____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_583422302____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_linter_style_longFileDefValue;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter_spec__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter_spec__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = " 0`"};
static const lean_object* lp_mathlib_Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter_spec__2___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter_spec__2___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter_spec__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter_spec__2___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "This file is "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__1;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 35, .m_capacity = 35, .m_length = 34, .m_data = " lines long. The current limit is "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__2_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__3;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 28, .m_capacity = 28, .m_length = 27, .m_data = ", but it is expected to be "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__4_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__5;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 37, .m_capacity = 37, .m_length = 36, .m_data = ":\n`set_option linter.style.longFile "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__6_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__7;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "`."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__8_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__9;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 31, .m_capacity = 31, .m_length = 30, .m_data = " lines long, but the limit is "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__10_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__11;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 90, .m_capacity = 90, .m_length = 89, .m_data = ".\n\nYou can extend the allowed length of the file using `set_option linter.style.longFile "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__12_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__13;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 78, .m_capacity = 78, .m_length = 77, .m_data = "`.\nYou can completely disable this linter by setting the length limit to `0`."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__14_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__15;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 47, .m_capacity = 47, .m_length = 46, .m_data = "The default value of the `longFile` linter is "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__16_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__17;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = ".\nThis file is "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__18 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__18_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__19;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 107, .m_capacity = 107, .m_length = 106, .m_data = " lines long which does not exceed the allowed bound.\nPlease, remove the `set_option linter.style.longFile "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__20 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__20_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__21;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__22 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__22_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = ".\nThe current value of "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__23 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__23_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__24;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 90, .m_capacity = 90, .m_length = 89, .m_data = " does not exceed the allowed bound.\nPlease, remove the `set_option linter.style.longFile "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__25 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__25_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__26_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__26;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___closed__0_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_withSetOptionIn___boxed, .m_arity = 6, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___closed__0_value)} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__12_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_695976056____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(4, 48, 225, 172, 124, 230, 200, 181)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "longFileLinter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___closed__2_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___closed__3_value),LEAN_SCALAR_PTR_LITERAL(244, 79, 12, 246, 33, 235, 156, 39)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___closed__4_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_initFn_00___x40_Mathlib_Tactic_Linter_Style_3199531762____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_initFn_00___x40_Mathlib_Tactic_Linter_Style_3199531762____hygCtx___hyg_2____boxed(lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_1043171623____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "longLine"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_1043171623____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_1043171623____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_1043171623____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(186, 218, 113, 226, 101, 176, 32, 79)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_1043171623____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_1043171623____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(105, 62, 218, 153, 100, 142, 29, 251)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_1043171623____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_1043171623____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_1043171623____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(32, 223, 195, 105, 147, 227, 220, 189)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_1043171623____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_1043171623____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Style_1043171623____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "enable the longLine linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Style_1043171623____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Style_1043171623____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Style_1043171623____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Style_1043171623____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Style_1043171623____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Style_1043171623____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_1043171623____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_1043171623____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_1043171623____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(120, 131, 127, 204, 79, 169, 80, 92)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_1043171623____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_1043171623____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(101, 237, 90, 120, 51, 59, 46, 172)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_1043171623____hygCtx___hyg_4__value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_1043171623____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(98, 189, 128, 85, 154, 50, 252, 160)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_1043171623____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_1043171623____hygCtx___hyg_4__value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_1043171623____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(135, 211, 246, 44, 254, 98, 40, 218)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_1043171623____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_1043171623____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_1043171623____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_1043171623____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_linter_style_longLine;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_690386724____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "maxLineLength"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_690386724____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_690386724____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_690386724____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(186, 218, 113, 226, 101, 176, 32, 79)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_690386724____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_690386724____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(105, 62, 218, 153, 100, 142, 29, 251)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_690386724____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_690386724____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_1043171623____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(32, 223, 195, 105, 147, 227, 220, 189)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_690386724____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_690386724____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_690386724____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(144, 221, 58, 156, 195, 104, 100, 37)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_690386724____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_690386724____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Style_690386724____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 63, .m_capacity = 63, .m_length = 62, .m_data = "maximum line length before the longLine linter emits a warning"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Style_690386724____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Style_690386724____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Style_690386724____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(100) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Style_690386724____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Style_690386724____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Style_690386724____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_690386724____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_690386724____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_690386724____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(120, 131, 127, 204, 79, 169, 80, 92)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_690386724____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_690386724____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(101, 237, 90, 120, 51, 59, 46, 172)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_690386724____hygCtx___hyg_4__value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_690386724____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(98, 189, 128, 85, 154, 50, 252, 160)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_690386724____hygCtx___hyg_4__value_aux_4 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_690386724____hygCtx___hyg_4__value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_1043171623____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(135, 211, 246, 44, 254, 98, 40, 218)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_690386724____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_690386724____hygCtx___hyg_4__value_aux_4),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_690386724____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(195, 57, 246, 11, 46, 53, 74, 162)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_690386724____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_690386724____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_690386724____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_690386724____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_linter_style_longLine_maxLineLength;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "meta import all "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__1;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "import all "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__2_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__3;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "public meta import "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__4_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__5;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "meta import "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__6_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__7;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "public import "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__8_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__9;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "import "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__10_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__11;
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_filterTR_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_filterTR_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__1_spec__1___redArg(lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__1___boxed(lean_object*);
static const lean_string_object lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__2___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "This line exceeds the "};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__2___redArg___closed__0 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__2___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__2___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__2___redArg___closed__1;
static const lean_string_object lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__2___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 37, .m_capacity = 37, .m_length = 36, .m_data = " character limit, please shorten it!"};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__2___redArg___closed__2 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__2___redArg___closed__2_value;
static lean_once_cell_t lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__2___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__2___redArg___closed__3;
static const lean_string_object lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__2___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 205, .m_capacity = 205, .m_length = 204, .m_data = "\nYou can use \"string gaps\" to format long strings: within a string quotation, using a '\\' at the end of a line allows you to continue the string on the following line, removing all intervening whitespace."};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__2___redArg___closed__4 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__2___redArg___closed__4_value;
static const lean_string_object lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__2___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "http"};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__2___redArg___closed__5 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__2___redArg___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__2___redArg(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "guardMsgsCmd"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___lam__0___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___lam__0___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(80, 121, 62, 112, 73, 11, 102, 99)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___lam__0___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___lam__0___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Module"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___lam__0___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___lam__0___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "header"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___lam__0___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___lam__0___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___lam__0___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___lam__0___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___lam__0___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___lam__0___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___lam__0___closed__4_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(239, 68, 245, 129, 233, 83, 45, 77)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___lam__0___closed__4_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___lam__0___closed__3_value),LEAN_SCALAR_PTR_LITERAL(40, 173, 92, 3, 94, 219, 131, 202)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___lam__0___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___lam__0___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "\n"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___lam__0___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___lam__0___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___lam__0___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___lam__0___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___lam__0___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___closed__0_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_withSetOptionIn___boxed, .m_arity = 6, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___closed__0_value)} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__12_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_1043171623____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(250, 79, 169, 231, 92, 191, 161, 244)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "longLineLinter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___closed__2_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___closed__3_value),LEAN_SCALAR_PTR_LITERAL(5, 108, 131, 109, 78, 216, 216, 144)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___closed__4_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__2(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_initFn_00___x40_Mathlib_Tactic_Linter_Style_724867545____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_initFn_00___x40_Mathlib_Tactic_Linter_Style_724867545____hygCtx___hyg_2____boxed(lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_3276061806____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "nameCheck"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_3276061806____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_3276061806____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_3276061806____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(186, 218, 113, 226, 101, 176, 32, 79)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_3276061806____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_3276061806____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(105, 62, 218, 153, 100, 142, 29, 251)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_3276061806____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_3276061806____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_3276061806____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(68, 248, 175, 114, 162, 177, 8, 43)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_3276061806____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_3276061806____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Style_3276061806____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 30, .m_capacity = 30, .m_length = 29, .m_data = "enable the `nameCheck` linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Style_3276061806____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Style_3276061806____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Style_3276061806____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Style_3276061806____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Style_3276061806____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Style_3276061806____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_3276061806____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_3276061806____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_3276061806____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(120, 131, 127, 204, 79, 169, 80, 92)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_3276061806____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_3276061806____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(101, 237, 90, 120, 51, 59, 46, 172)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_3276061806____hygCtx___hyg_4__value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_3276061806____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(98, 189, 128, 85, 154, 50, 252, 160)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_3276061806____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_3276061806____hygCtx___hyg_4__value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_3276061806____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(171, 236, 66, 123, 89, 237, 3, 177)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_3276061806____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_3276061806____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_3276061806____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_3276061806____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_linter_style_nameCheck;
LEAN_EXPORT uint8_t lp_mathlib_Option_instBEq_beq___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Option_instBEq_beq___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__0___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "declId"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___lam__0___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___lam__0___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___lam__0___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___lam__0___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___lam__0___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__2_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___lam__0___closed__1_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(243, 92, 136, 33, 216, 98, 92, 25)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___lam__0___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___lam__0___closed__1_value;
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___lam__0___boxed(lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "export"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___lam__1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___lam__1___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___lam__1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___lam__1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___lam__1___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___lam__1___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___lam__1___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__2_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___lam__1___closed__1_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(6, 73, 228, 195, 89, 60, 49, 127)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___lam__1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___lam__1___closed__1_value;
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___lam__1___boxed(lean_object*);
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__1___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__1___closed__0_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "__"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__1___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__1___closed__1_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "The declaration '"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__1___closed__2 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__1___closed__2_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__1___closed__3;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 114, .m_capacity = 114, .m_length = 113, .m_data = "' contains '__', which does not follow the mathlib naming conventions. Consider using single underscores instead."};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__1___closed__4 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__1___closed__4_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__1___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__1___closed__5;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__1(uint8_t, uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_getAliasSyntax___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__2_spec__2___redArg(uint8_t, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_getAliasSyntax___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__2_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_getAliasSyntax___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_getAliasSyntax___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___closed__0_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___closed__1_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___lam__2___boxed, .m_arity = 6, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___closed__0_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___closed__1_value)} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___closed__2_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_withSetOptionIn___boxed, .m_arity = 6, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___closed__2_value)} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__12_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_3276061806____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(126, 78, 166, 193, 50, 75, 194, 128)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "doubleUnderscore"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___closed__4_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___closed__5_value),LEAN_SCALAR_PTR_LITERAL(47, 14, 37, 172, 88, 180, 51, 106)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___closed__6_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___closed__7_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___closed__7_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_getAliasSyntax___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__2_spec__2(uint8_t, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_getAliasSyntax___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__2_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_initFn_00___x40_Mathlib_Tactic_Linter_Style_1285535146____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_initFn_00___x40_Mathlib_Tactic_Linter_Style_1285535146____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_any___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__0(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_any___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1_spec__1___redArg(lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "_"};
static const lean_object* lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1___closed__0 = (const lean_object*)&lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1___closed__0_value;
static lean_once_cell_t lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1___closed__1;
static lean_once_cell_t lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static uint8_t lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1___closed__2;
static lean_once_cell_t lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1___closed__3;
static lean_once_cell_t lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1___closed__4;
static lean_once_cell_t lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1___closed__5;
static const lean_ctor_object lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1___closed__6 = (const lean_object*)&lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1___closed__6_value;
LEAN_EXPORT uint8_t lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1___boxed(lean_object*);
static const lean_string_object lp_mathlib_List_any___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_List_any___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__2___closed__0 = (const lean_object*)&lp_mathlib_List_any___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__2___closed__0_value;
static lean_once_cell_t lp_mathlib_List_any___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_any___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__2___closed__1;
LEAN_EXPORT uint8_t lp_mathlib_List_any___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__2(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_any___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__2___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_List_any___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Simps"};
static const lean_object* lp_mathlib_List_any___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__3___closed__0 = (const lean_object*)&lp_mathlib_List_any___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__3___closed__0_value;
static const lean_ctor_object lp_mathlib_List_any___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_List_any___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__3___closed__0_value),LEAN_SCALAR_PTR_LITERAL(62, 14, 47, 163, 29, 240, 87, 177)}};
static const lean_object* lp_mathlib_List_any___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__3___closed__1 = (const lean_object*)&lp_mathlib_List_any___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__3___closed__1_value;
LEAN_EXPORT uint8_t lp_mathlib_List_any___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__3(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_any___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__3___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__4_spec__5___redArg(lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__4_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__4(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__4___boxed(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_mathlib"};
static const lean_object* lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_2"};
static const lean_object* lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__3;
static const lean_string_object lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "LibraryNote"};
static const lean_object* lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__4_value),LEAN_SCALAR_PTR_LITERAL(134, 197, 35, 181, 239, 168, 57, 237)}};
static const lean_object* lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__6_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__7_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_object* lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__1_value),LEAN_SCALAR_PTR_LITERAL(46, 201, 23, 171, 41, 77, 220, 95)}};
static const lean_object* lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_1"};
static const lean_object* lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__8_value;
static lean_once_cell_t lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__9;
static const lean_string_object lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Default"};
static const lean_object* lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__10_value),LEAN_SCALAR_PTR_LITERAL(139, 129, 26, 102, 182, 182, 197, 45)}};
static const lean_object* lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__11_value;
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__4_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__4_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Linter_Style_nameCheck_defsWithUnderscore_spec__0(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Meta"};
static const lean_object* lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___lam__0___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Simp"};
static const lean_object* lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Simproc"};
static const lean_object* lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___lam__0___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___lam__0___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___lam__0___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___lam__0___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(194, 50, 106, 158, 41, 60, 103, 214)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___lam__0___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___lam__0___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(54, 38, 229, 237, 143, 62, 212, 6)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___lam__0___closed__3_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(18, 160, 179, 254, 130, 82, 156, 255)}};
static const lean_object* lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___lam__0___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "The definition `"};
static const lean_object* lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___lam__0___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___lam__0___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___lam__0___closed__5;
static const lean_string_object lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 129, .m_capacity = 129, .m_length = 128, .m_data = "` contains an underscore. This almost surely violates mathlib's naming convention; use lowerCamelCase or UpperCamelCase instead."};
static const lean_object* lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___lam__0___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___lam__0___closed__6_value;
static lean_once_cell_t lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___lam__0___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___lam__0___closed__7;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___lam__0___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 55, .m_capacity = 55, .m_length = 54, .m_data = "no definitions with an underscore in their name found."};
static const lean_object* lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___closed__1_value)}};
static const lean_object* lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___closed__3;
static const lean_string_object lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 52, .m_capacity = 52, .m_length = 51, .m_data = "FOUND definitions with an underscore in their name."};
static const lean_object* lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___closed__5_value;
static lean_once_cell_t lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___closed__6;
static lean_once_cell_t lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___closed__7;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_856385564____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "openClassical"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_856385564____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_856385564____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_856385564____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(186, 218, 113, 226, 101, 176, 32, 79)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_856385564____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_856385564____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(105, 62, 218, 153, 100, 142, 29, 251)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_856385564____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_856385564____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_856385564____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(207, 117, 47, 58, 5, 18, 169, 211)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_856385564____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_856385564____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Style_856385564____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 32, .m_capacity = 32, .m_length = 31, .m_data = "enable the openClassical linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Style_856385564____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Style_856385564____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Style_856385564____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Style_856385564____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Style_856385564____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Style_856385564____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_856385564____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_856385564____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_856385564____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(120, 131, 127, 204, 79, 169, 80, 92)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_856385564____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_856385564____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(101, 237, 90, 120, 51, 59, 46, 172)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_856385564____hygCtx___hyg_4__value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_856385564____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(98, 189, 128, 85, 154, 50, 252, 160)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_856385564____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_856385564____hygCtx___hyg_4__value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_856385564____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(184, 68, 178, 215, 49, 160, 213, 237)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_856385564____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_856385564____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_856385564____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_856385564____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_linter_style_openClassical;
static lean_once_cell_t lp_mathlib_panic___at___00Mathlib_Linter_Style_openClassical_extractOpenNames_spec__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_panic___at___00Mathlib_Linter_Style_openClassical_extractOpenNames_spec__0___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Linter_Style_openClassical_extractOpenNames_spec__0(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "in"};
static const lean_object* lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__2_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__0_value),LEAN_SCALAR_PTR_LITERAL(65, 79, 35, 19, 21, 38, 89, 10)}};
static const lean_object* lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "open"};
static const lean_object* lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__3_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__3_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__2_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__3_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__2_value),LEAN_SCALAR_PTR_LITERAL(148, 8, 226, 43, 107, 167, 95, 157)}};
static const lean_object* lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "openHiding"};
static const lean_object* lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__5_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__5_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__5_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__2_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__5_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__4_value),LEAN_SCALAR_PTR_LITERAL(250, 7, 251, 101, 7, 65, 209, 251)}};
static const lean_object* lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "openRenaming"};
static const lean_object* lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__7_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__7_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__2_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__7_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__6_value),LEAN_SCALAR_PTR_LITERAL(19, 98, 60, 198, 229, 139, 11, 192)}};
static const lean_object* lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "openOnly"};
static const lean_object* lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__9_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__9_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__9_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__9_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__2_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__9_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__8_value),LEAN_SCALAR_PTR_LITERAL(240, 111, 143, 230, 204, 176, 68, 125)}};
static const lean_object* lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "openSimple"};
static const lean_object* lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__11_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__11_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__11_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__2_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__11_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__10_value),LEAN_SCALAR_PTR_LITERAL(171, 238, 134, 92, 162, 110, 43, 67)}};
static const lean_object* lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "openScoped"};
static const lean_object* lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__13_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__13_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__13_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__13_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__13_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__2_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__13_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__12_value),LEAN_SCALAR_PTR_LITERAL(55, 166, 237, 23, 37, 47, 5, 133)}};
static const lean_object* lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__13_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 28, .m_capacity = 28, .m_length = 27, .m_data = "Mathlib.Tactic.Linter.Style"};
static const lean_object* lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 52, .m_capacity = 52, .m_length = 51, .m_data = "Mathlib.Linter.Style.openClassical.extractOpenNames"};
static const lean_object* lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__15_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "unreachable code has been reached"};
static const lean_object* lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__16_value;
static lean_once_cell_t lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__17;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames(lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 346, .m_capacity = 346, .m_length = 345, .m_data = "please avoid 'open (scoped) Classical' statements: this can hide theorem statements which would be better stated with explicit decidability statements.\nInstead, use `open Classical in` for definitions or instances, the `classical` tactic for proofs.\nFor theorem statements, either add missing decidability assumptions or use `open Classical in`."};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter_spec__0___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter_spec__0___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter_spec__0___closed__0_value)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter_spec__0___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter_spec__0___closed__1_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter_spec__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter_spec__0___closed__2;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter_spec__0(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Classical"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter_spec__1___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter_spec__1___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter_spec__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter_spec__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(40, 236, 220, 79, 38, 141, 161, 150)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter_spec__1___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter_spec__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter_spec__1(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter___lam__0___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__12_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_856385564____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(245, 214, 57, 83, 37, 217, 248, 109)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "openClassicalLinter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter___closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter___closed__2_value),LEAN_SCALAR_PTR_LITERAL(69, 106, 54, 136, 154, 15, 247, 240)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter___closed__0_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter___closed__3_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter___closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_initFn_00___x40_Mathlib_Tactic_Linter_Style_273924139____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_initFn_00___x40_Mathlib_Tactic_Linter_Style_273924139____hygCtx___hyg_2____boxed(lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_4166288182____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "show"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_4166288182____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_4166288182____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_4166288182____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(186, 218, 113, 226, 101, 176, 32, 79)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_4166288182____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_4166288182____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(105, 62, 218, 153, 100, 142, 29, 251)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_4166288182____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_4166288182____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_4166288182____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(196, 65, 200, 223, 135, 9, 255, 253)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_4166288182____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_4166288182____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Style_4166288182____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "enable the show linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Style_4166288182____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Style_4166288182____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Style_4166288182____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Style_4166288182____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Style_4166288182____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Style_4166288182____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_4166288182____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_4166288182____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_4166288182____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(120, 131, 127, 204, 79, 169, 80, 92)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_4166288182____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_4166288182____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(101, 237, 90, 120, 51, 59, 46, 172)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_4166288182____hygCtx___hyg_4__value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_4166288182____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(98, 189, 128, 85, 154, 50, 252, 160)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_4166288182____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_4166288182____hygCtx___hyg_4__value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_4166288182____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(43, 96, 46, 188, 165, 19, 126, 47)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_4166288182____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_4166288182____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_4166288182____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_4166288182____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_linter_style_show;
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2_spec__3_spec__4_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2_spec__3_spec__4_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2_spec__3_spec__4___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2_spec__3_spec__4___redArg___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2_spec__3_spec__4___redArg___lam__0___closed__0_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2_spec__3_spec__4___redArg___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "unsolvedGoals"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2_spec__3_spec__4___redArg___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2_spec__3_spec__4___redArg___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2_spec__3_spec__4___redArg___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "synthPlaceholder"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2_spec__3_spec__4___redArg___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2_spec__3_spec__4___redArg___lam__0___closed__2_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2_spec__3_spec__4___redArg___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "lean"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2_spec__3_spec__4___redArg___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2_spec__3_spec__4___redArg___lam__0___closed__3_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2_spec__3_spec__4___redArg___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "inductionWithNoAlts"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2_spec__3_spec__4___redArg___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2_spec__3_spec__4___redArg___lam__0___closed__4_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2_spec__3_spec__4___redArg___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "_namedError"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2_spec__3_spec__4___redArg___lam__0___closed__5 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2_spec__3_spec__4___redArg___lam__0___closed__5_value;
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2_spec__3_spec__4___redArg___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2_spec__3_spec__4___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2_spec__3_spec__4___redArg(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2_spec__3_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__1_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_beq___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_beq___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__3___boxed(lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow___closed__0_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow___closed__0_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow___closed__0_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__7_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow___closed__0_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_4166288182____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(151, 147, 62, 103, 130, 224, 84, 63)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 190, .m_capacity = 190, .m_length = 189, .m_data = "The `show` tactic should only be used to indicate intermediate goal states for readability.\nHowever, this tactic invocation changed the goal. Please use `change` instead for these purposes."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow___closed__1_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow___closed__2;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2_spec__3_spec__4(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Linter_Style_show___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_Style_show___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_Style_show___closed__0_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(120, 131, 127, 204, 79, 169, 80, 92)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_Style_show___closed__0_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_Style_show___closed__0_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___closed__7_value),LEAN_SCALAR_PTR_LITERAL(80, 121, 70, 102, 253, 204, 74, 141)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_Style_show___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_Style_show___closed__0_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_4166288182____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(17, 213, 238, 12, 92, 30, 133, 106)}};
static const lean_object* lp_mathlib_Mathlib_Linter_Style_show___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Linter_Style_show___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_Style_show___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Linter_Style_show___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Linter_Style_show___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_Style_show___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Linter_Style_show___closed__1_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Linter_Style_show___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Linter_Style_show___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_Style_show___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "show "};
static const lean_object* lp_mathlib_Mathlib_Linter_Style_show___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Linter_Style_show___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_Style_show___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_Style_show___closed__3_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Linter_Style_show___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Linter_Style_show___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_Style_show___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_List_any___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__2___closed__0_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Mathlib_Linter_Style_show___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Linter_Style_show___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_Style_show___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_Style_show___closed__5_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Linter_Style_show___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Linter_Style_show___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_Style_show___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_Style_show___closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Linter_Style_show___closed__4_value),((lean_object*)&lp_mathlib_Mathlib_Linter_Style_show___closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_Linter_Style_show___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Linter_Style_show___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_Style_show___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_Style_show___closed__0_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Linter_Style_show___closed__7_value)}};
static const lean_object* lp_mathlib_Mathlib_Linter_Style_show___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Linter_Style_show___closed__8_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Linter_Style_show = (const lean_object*)&lp_mathlib_Mathlib_Linter_Style_show___closed__8_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Linter_Style___aux__Mathlib__Tactic__Linter__Style______elabRules__Mathlib__Linter__Style__show__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Linter_Style___aux__Mathlib__Tactic__Linter__Style______elabRules__Mathlib__Linter__Style__show__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Linter_Style___aux__Mathlib__Tactic__Linter__Style______elabRules__Mathlib__Linter__Style__show__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Linter_Style___aux__Mathlib__Tactic__Linter__Style______elabRules__Mathlib__Linter__Style__show__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Linter_Style___aux__Mathlib__Tactic__Linter__Style______elabRules__Mathlib__Linter__Style__show__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Linter_Style___aux__Mathlib__Tactic__Linter__Style______elabRules__Mathlib__Linter__Style__show__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_Style___aux__Mathlib__Tactic__Linter__Style______elabRules__Mathlib__Linter__Style__show__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_Style___aux__Mathlib__Tactic__Linter__Style______elabRules__Mathlib__Linter__Style__show__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__spec__0(lean_object* v_name_1_, lean_object* v_decl_2_, lean_object* v_ref_3_){
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__spec__0___boxed(lean_object* v_name_29_, lean_object* v_decl_30_, lean_object* v_ref_31_, lean_object* v_a_32_){
_start:
{
lean_object* v_res_33_; 
v_res_33_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__spec__0(v_name_29_, v_decl_30_, v_ref_31_);
lean_dec_ref(v_decl_30_);
return v_res_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_56_; lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v___x_59_; 
v___x_56_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4_));
v___x_57_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4_));
v___x_58_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4_));
v___x_59_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__spec__0(v___x_56_, v___x_57_, v___x_58_);
return v___x_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4____boxed(lean_object* v_a_60_){
_start:
{
lean_object* v_res_61_; 
v_res_61_ = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4_();
return v_res_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption(lean_object* v_x_86_){
_start:
{
lean_object* v___x_87_; uint8_t v___x_88_; 
v___x_87_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__4));
lean_inc(v_x_86_);
v___x_88_ = l_Lean_Syntax_isOfKind(v_x_86_, v___x_87_);
if (v___x_88_ == 0)
{
lean_object* v___x_89_; uint8_t v___x_90_; 
v___x_89_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__6));
lean_inc(v_x_86_);
v___x_90_ = l_Lean_Syntax_isOfKind(v_x_86_, v___x_89_);
if (v___x_90_ == 0)
{
lean_object* v___x_91_; uint8_t v___x_92_; 
v___x_91_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__8));
lean_inc(v_x_86_);
v___x_92_ = l_Lean_Syntax_isOfKind(v_x_86_, v___x_91_);
if (v___x_92_ == 0)
{
lean_object* v___x_93_; 
lean_dec(v_x_86_);
v___x_93_ = lean_box(0);
return v___x_93_;
}
else
{
lean_object* v___x_94_; lean_object* v_name_95_; lean_object* v___x_96_; uint8_t v___x_97_; 
v___x_94_ = lean_unsigned_to_nat(1u);
v_name_95_ = l_Lean_Syntax_getArg(v_x_86_, v___x_94_);
v___x_96_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__10));
lean_inc(v_name_95_);
v___x_97_ = l_Lean_Syntax_isOfKind(v_name_95_, v___x_96_);
if (v___x_97_ == 0)
{
lean_object* v___x_98_; 
lean_dec(v_name_95_);
lean_dec(v_x_86_);
v___x_98_ = lean_box(0);
return v___x_98_;
}
else
{
lean_object* v___x_99_; lean_object* v___x_100_; lean_object* v___x_101_; uint8_t v___x_102_; 
v___x_99_ = lean_unsigned_to_nat(0u);
v___x_100_ = lean_unsigned_to_nat(2u);
v___x_101_ = l_Lean_Syntax_getArg(v_x_86_, v___x_100_);
lean_dec(v_x_86_);
v___x_102_ = l_Lean_Syntax_matchesNull(v___x_101_, v___x_99_);
if (v___x_102_ == 0)
{
lean_object* v___x_103_; 
lean_dec(v_name_95_);
v___x_103_ = lean_box(0);
return v___x_103_;
}
else
{
lean_object* v___x_104_; lean_object* v___x_105_; 
v___x_104_ = l_Lean_TSyntax_getId(v_name_95_);
lean_dec(v_name_95_);
v___x_105_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_105_, 0, v___x_104_);
return v___x_105_;
}
}
}
}
else
{
lean_object* v___x_106_; lean_object* v_name_107_; lean_object* v___x_108_; uint8_t v___x_109_; 
v___x_106_ = lean_unsigned_to_nat(1u);
v_name_107_ = l_Lean_Syntax_getArg(v_x_86_, v___x_106_);
v___x_108_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__10));
lean_inc(v_name_107_);
v___x_109_ = l_Lean_Syntax_isOfKind(v_name_107_, v___x_108_);
if (v___x_109_ == 0)
{
lean_object* v___x_110_; 
lean_dec(v_name_107_);
lean_dec(v_x_86_);
v___x_110_ = lean_box(0);
return v___x_110_;
}
else
{
lean_object* v___x_111_; lean_object* v___x_112_; lean_object* v___x_113_; uint8_t v___x_114_; 
v___x_111_ = lean_unsigned_to_nat(0u);
v___x_112_ = lean_unsigned_to_nat(2u);
v___x_113_ = l_Lean_Syntax_getArg(v_x_86_, v___x_112_);
lean_dec(v_x_86_);
v___x_114_ = l_Lean_Syntax_matchesNull(v___x_113_, v___x_111_);
if (v___x_114_ == 0)
{
lean_object* v___x_115_; 
lean_dec(v_name_107_);
v___x_115_ = lean_box(0);
return v___x_115_;
}
else
{
lean_object* v___x_116_; lean_object* v___x_117_; 
v___x_116_ = l_Lean_TSyntax_getId(v_name_107_);
lean_dec(v_name_107_);
v___x_117_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_117_, 0, v___x_116_);
return v___x_117_;
}
}
}
}
else
{
lean_object* v___x_118_; lean_object* v_name_119_; lean_object* v___x_120_; uint8_t v___x_121_; 
v___x_118_ = lean_unsigned_to_nat(1u);
v_name_119_ = l_Lean_Syntax_getArg(v_x_86_, v___x_118_);
v___x_120_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__10));
lean_inc(v_name_119_);
v___x_121_ = l_Lean_Syntax_isOfKind(v_name_119_, v___x_120_);
if (v___x_121_ == 0)
{
lean_object* v___x_122_; 
lean_dec(v_name_119_);
lean_dec(v_x_86_);
v___x_122_ = lean_box(0);
return v___x_122_;
}
else
{
lean_object* v___x_123_; lean_object* v___x_124_; lean_object* v___x_125_; uint8_t v___x_126_; 
v___x_123_ = lean_unsigned_to_nat(0u);
v___x_124_ = lean_unsigned_to_nat(2u);
v___x_125_ = l_Lean_Syntax_getArg(v_x_86_, v___x_124_);
lean_dec(v_x_86_);
v___x_126_ = l_Lean_Syntax_matchesNull(v___x_125_, v___x_123_);
if (v___x_126_ == 0)
{
lean_object* v___x_127_; 
lean_dec(v_name_119_);
v___x_127_ = lean_box(0);
return v___x_127_;
}
else
{
lean_object* v___x_128_; lean_object* v___x_129_; 
v___x_128_ = l_Lean_TSyntax_getId(v_name_119_);
lean_dec(v_name_119_);
v___x_129_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_129_, 0, v___x_128_);
return v___x_129_;
}
}
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Linter_Style_setOption_isSetOption(lean_object* v_stx_130_){
_start:
{
lean_object* v___x_131_; 
v___x_131_ = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption(v_stx_130_);
if (lean_obj_tag(v___x_131_) == 1)
{
uint8_t v___x_132_; 
lean_dec_ref_known(v___x_131_, 1);
v___x_132_ = 1;
return v___x_132_;
}
else
{
uint8_t v___x_133_; 
lean_dec(v___x_131_);
v___x_133_ = 0;
return v___x_133_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_Style_setOption_isSetOption___boxed(lean_object* v_stx_134_){
_start:
{
uint8_t v_res_135_; lean_object* v_r_136_; 
v_res_135_ = lp_mathlib_Mathlib_Linter_Style_setOption_isSetOption(v_stx_134_);
v_r_136_ = lean_box(v_res_135_);
return v_r_136_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__6(lean_object* v_opts_137_, lean_object* v_opt_138_){
_start:
{
lean_object* v_name_139_; lean_object* v_defValue_140_; lean_object* v_map_141_; lean_object* v___x_142_; 
v_name_139_ = lean_ctor_get(v_opt_138_, 0);
v_defValue_140_ = lean_ctor_get(v_opt_138_, 1);
v_map_141_ = lean_ctor_get(v_opts_137_, 0);
v___x_142_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_141_, v_name_139_);
if (lean_obj_tag(v___x_142_) == 0)
{
uint8_t v___x_143_; 
v___x_143_ = lean_unbox(v_defValue_140_);
return v___x_143_;
}
else
{
lean_object* v_val_144_; 
v_val_144_ = lean_ctor_get(v___x_142_, 0);
lean_inc(v_val_144_);
lean_dec_ref_known(v___x_142_, 1);
if (lean_obj_tag(v_val_144_) == 1)
{
uint8_t v_v_145_; 
v_v_145_ = lean_ctor_get_uint8(v_val_144_, 0);
lean_dec_ref_known(v_val_144_, 0);
return v_v_145_;
}
else
{
uint8_t v___x_146_; 
lean_dec(v_val_144_);
v___x_146_ = lean_unbox(v_defValue_140_);
return v___x_146_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__6___boxed(lean_object* v_opts_147_, lean_object* v_opt_148_){
_start:
{
uint8_t v_res_149_; lean_object* v_r_150_; 
v_res_149_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__6(v_opts_147_, v_opt_148_);
lean_dec_ref(v_opt_148_);
lean_dec_ref(v_opts_147_);
v_r_150_ = lean_box(v_res_149_);
return v_r_150_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4___lam__0(uint8_t v___y_152_, uint8_t v_suppressElabErrors_153_, lean_object* v_x_154_){
_start:
{
if (lean_obj_tag(v_x_154_) == 1)
{
lean_object* v_pre_155_; 
v_pre_155_ = lean_ctor_get(v_x_154_, 0);
if (lean_obj_tag(v_pre_155_) == 0)
{
lean_object* v_str_156_; lean_object* v___x_157_; uint8_t v___x_158_; 
v_str_156_ = lean_ctor_get(v_x_154_, 1);
v___x_157_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4___lam__0___closed__0));
v___x_158_ = lean_string_dec_eq(v_str_156_, v___x_157_);
if (v___x_158_ == 0)
{
return v___y_152_;
}
else
{
return v_suppressElabErrors_153_;
}
}
else
{
return v___y_152_;
}
}
else
{
return v___y_152_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4___lam__0___boxed(lean_object* v___y_159_, lean_object* v_suppressElabErrors_160_, lean_object* v_x_161_){
_start:
{
uint8_t v___y_5177__boxed_162_; uint8_t v_suppressElabErrors_boxed_163_; uint8_t v_res_164_; lean_object* v_r_165_; 
v___y_5177__boxed_162_ = lean_unbox(v___y_159_);
v_suppressElabErrors_boxed_163_ = lean_unbox(v_suppressElabErrors_160_);
v_res_164_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4___lam__0(v___y_5177__boxed_162_, v_suppressElabErrors_boxed_163_, v_x_161_);
lean_dec(v_x_161_);
v_r_165_ = lean_box(v_res_164_);
return v_r_165_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__5___redArg___closed__0(void){
_start:
{
lean_object* v___x_166_; 
v___x_166_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_166_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__5___redArg___closed__1(void){
_start:
{
lean_object* v___x_167_; lean_object* v___x_168_; 
v___x_167_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__5___redArg___closed__0, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__5___redArg___closed__0_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__5___redArg___closed__0);
v___x_168_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_168_, 0, v___x_167_);
return v___x_168_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__5___redArg___closed__2(void){
_start:
{
lean_object* v___x_169_; lean_object* v___x_170_; lean_object* v___x_171_; 
v___x_169_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__5___redArg___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__5___redArg___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__5___redArg___closed__1);
v___x_170_ = lean_unsigned_to_nat(0u);
v___x_171_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_171_, 0, v___x_170_);
lean_ctor_set(v___x_171_, 1, v___x_170_);
lean_ctor_set(v___x_171_, 2, v___x_170_);
lean_ctor_set(v___x_171_, 3, v___x_170_);
lean_ctor_set(v___x_171_, 4, v___x_169_);
lean_ctor_set(v___x_171_, 5, v___x_169_);
lean_ctor_set(v___x_171_, 6, v___x_169_);
lean_ctor_set(v___x_171_, 7, v___x_169_);
lean_ctor_set(v___x_171_, 8, v___x_169_);
lean_ctor_set(v___x_171_, 9, v___x_169_);
return v___x_171_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__5___redArg___closed__3(void){
_start:
{
lean_object* v___x_172_; lean_object* v___x_173_; lean_object* v___x_174_; 
v___x_172_ = lean_unsigned_to_nat(32u);
v___x_173_ = lean_mk_empty_array_with_capacity(v___x_172_);
v___x_174_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_174_, 0, v___x_173_);
return v___x_174_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__5___redArg___closed__4(void){
_start:
{
size_t v___x_175_; lean_object* v___x_176_; lean_object* v___x_177_; lean_object* v___x_178_; lean_object* v___x_179_; lean_object* v___x_180_; 
v___x_175_ = ((size_t)5ULL);
v___x_176_ = lean_unsigned_to_nat(0u);
v___x_177_ = lean_unsigned_to_nat(32u);
v___x_178_ = lean_mk_empty_array_with_capacity(v___x_177_);
v___x_179_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__5___redArg___closed__3, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__5___redArg___closed__3_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__5___redArg___closed__3);
v___x_180_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_180_, 0, v___x_179_);
lean_ctor_set(v___x_180_, 1, v___x_178_);
lean_ctor_set(v___x_180_, 2, v___x_176_);
lean_ctor_set(v___x_180_, 3, v___x_176_);
lean_ctor_set_usize(v___x_180_, 4, v___x_175_);
return v___x_180_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__5___redArg___closed__5(void){
_start:
{
lean_object* v___x_181_; lean_object* v___x_182_; lean_object* v___x_183_; lean_object* v___x_184_; 
v___x_181_ = lean_box(1);
v___x_182_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__5___redArg___closed__4, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__5___redArg___closed__4_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__5___redArg___closed__4);
v___x_183_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__5___redArg___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__5___redArg___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__5___redArg___closed__1);
v___x_184_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_184_, 0, v___x_183_);
lean_ctor_set(v___x_184_, 1, v___x_182_);
lean_ctor_set(v___x_184_, 2, v___x_181_);
return v___x_184_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__5___redArg(lean_object* v_msgData_185_, lean_object* v___y_186_){
_start:
{
lean_object* v___x_188_; lean_object* v_env_189_; lean_object* v___x_190_; lean_object* v_scopes_191_; lean_object* v___x_192_; lean_object* v___x_193_; lean_object* v_opts_194_; lean_object* v___x_195_; lean_object* v___x_196_; lean_object* v___x_197_; lean_object* v___x_198_; lean_object* v___x_199_; 
v___x_188_ = lean_st_ref_get(v___y_186_);
v_env_189_ = lean_ctor_get(v___x_188_, 0);
lean_inc_ref(v_env_189_);
lean_dec(v___x_188_);
v___x_190_ = lean_st_ref_get(v___y_186_);
v_scopes_191_ = lean_ctor_get(v___x_190_, 2);
lean_inc(v_scopes_191_);
lean_dec(v___x_190_);
v___x_192_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_193_ = l_List_head_x21___redArg(v___x_192_, v_scopes_191_);
lean_dec(v_scopes_191_);
v_opts_194_ = lean_ctor_get(v___x_193_, 1);
lean_inc_ref(v_opts_194_);
lean_dec(v___x_193_);
v___x_195_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__5___redArg___closed__2, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__5___redArg___closed__2_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__5___redArg___closed__2);
v___x_196_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__5___redArg___closed__5, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__5___redArg___closed__5_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__5___redArg___closed__5);
v___x_197_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_197_, 0, v_env_189_);
lean_ctor_set(v___x_197_, 1, v___x_195_);
lean_ctor_set(v___x_197_, 2, v___x_196_);
lean_ctor_set(v___x_197_, 3, v_opts_194_);
v___x_198_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_198_, 0, v___x_197_);
lean_ctor_set(v___x_198_, 1, v_msgData_185_);
v___x_199_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_199_, 0, v___x_198_);
return v___x_199_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__5___redArg___boxed(lean_object* v_msgData_200_, lean_object* v___y_201_, lean_object* v___y_202_){
_start:
{
lean_object* v_res_203_; 
v_res_203_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__5___redArg(v_msgData_200_, v___y_201_);
lean_dec(v___y_201_);
return v_res_203_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4(lean_object* v_ref_205_, lean_object* v_msgData_206_, uint8_t v_severity_207_, uint8_t v_isSilent_208_, lean_object* v___y_209_, lean_object* v___y_210_){
_start:
{
uint8_t v___y_213_; lean_object* v___y_214_; lean_object* v___y_215_; lean_object* v___y_216_; lean_object* v___y_217_; uint8_t v___y_218_; lean_object* v___y_219_; lean_object* v___y_220_; uint8_t v___y_277_; uint8_t v___y_278_; lean_object* v___y_279_; uint8_t v___y_280_; lean_object* v___y_281_; uint8_t v___y_305_; uint8_t v___y_306_; lean_object* v___y_307_; uint8_t v___y_308_; lean_object* v___y_309_; uint8_t v___y_313_; uint8_t v___y_314_; uint8_t v___y_315_; uint8_t v___x_330_; uint8_t v___y_332_; uint8_t v___y_333_; uint8_t v___y_334_; uint8_t v___y_336_; uint8_t v___x_348_; 
v___x_330_ = 2;
v___x_348_ = l_Lean_instBEqMessageSeverity_beq(v_severity_207_, v___x_330_);
if (v___x_348_ == 0)
{
v___y_336_ = v___x_348_;
goto v___jp_335_;
}
else
{
uint8_t v___x_349_; 
lean_inc_ref(v_msgData_206_);
v___x_349_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_206_);
v___y_336_ = v___x_349_;
goto v___jp_335_;
}
v___jp_212_:
{
lean_object* v___x_221_; 
v___x_221_ = l_Lean_Elab_Command_getScope___redArg(v___y_220_);
if (lean_obj_tag(v___x_221_) == 0)
{
lean_object* v_a_222_; lean_object* v___x_223_; 
v_a_222_ = lean_ctor_get(v___x_221_, 0);
lean_inc(v_a_222_);
lean_dec_ref_known(v___x_221_, 1);
v___x_223_ = l_Lean_Elab_Command_getScope___redArg(v___y_220_);
if (lean_obj_tag(v___x_223_) == 0)
{
lean_object* v_a_224_; lean_object* v___x_226_; uint8_t v_isShared_227_; uint8_t v_isSharedCheck_259_; 
v_a_224_ = lean_ctor_get(v___x_223_, 0);
v_isSharedCheck_259_ = !lean_is_exclusive(v___x_223_);
if (v_isSharedCheck_259_ == 0)
{
v___x_226_ = v___x_223_;
v_isShared_227_ = v_isSharedCheck_259_;
goto v_resetjp_225_;
}
else
{
lean_inc(v_a_224_);
lean_dec(v___x_223_);
v___x_226_ = lean_box(0);
v_isShared_227_ = v_isSharedCheck_259_;
goto v_resetjp_225_;
}
v_resetjp_225_:
{
lean_object* v___x_228_; lean_object* v_currNamespace_229_; lean_object* v_openDecls_230_; lean_object* v_env_231_; lean_object* v_messages_232_; lean_object* v_scopes_233_; lean_object* v_usedQuotCtxts_234_; lean_object* v_nextMacroScope_235_; lean_object* v_maxRecDepth_236_; lean_object* v_ngen_237_; lean_object* v_auxDeclNGen_238_; lean_object* v_infoState_239_; lean_object* v_traceState_240_; lean_object* v_snapshotTasks_241_; lean_object* v_prevLinterStates_242_; lean_object* v___x_244_; uint8_t v_isShared_245_; uint8_t v_isSharedCheck_258_; 
v___x_228_ = lean_st_ref_take(v___y_220_);
v_currNamespace_229_ = lean_ctor_get(v_a_222_, 2);
lean_inc(v_currNamespace_229_);
lean_dec(v_a_222_);
v_openDecls_230_ = lean_ctor_get(v_a_224_, 3);
lean_inc(v_openDecls_230_);
lean_dec(v_a_224_);
v_env_231_ = lean_ctor_get(v___x_228_, 0);
v_messages_232_ = lean_ctor_get(v___x_228_, 1);
v_scopes_233_ = lean_ctor_get(v___x_228_, 2);
v_usedQuotCtxts_234_ = lean_ctor_get(v___x_228_, 3);
v_nextMacroScope_235_ = lean_ctor_get(v___x_228_, 4);
v_maxRecDepth_236_ = lean_ctor_get(v___x_228_, 5);
v_ngen_237_ = lean_ctor_get(v___x_228_, 6);
v_auxDeclNGen_238_ = lean_ctor_get(v___x_228_, 7);
v_infoState_239_ = lean_ctor_get(v___x_228_, 8);
v_traceState_240_ = lean_ctor_get(v___x_228_, 9);
v_snapshotTasks_241_ = lean_ctor_get(v___x_228_, 10);
v_prevLinterStates_242_ = lean_ctor_get(v___x_228_, 11);
v_isSharedCheck_258_ = !lean_is_exclusive(v___x_228_);
if (v_isSharedCheck_258_ == 0)
{
v___x_244_ = v___x_228_;
v_isShared_245_ = v_isSharedCheck_258_;
goto v_resetjp_243_;
}
else
{
lean_inc(v_prevLinterStates_242_);
lean_inc(v_snapshotTasks_241_);
lean_inc(v_traceState_240_);
lean_inc(v_infoState_239_);
lean_inc(v_auxDeclNGen_238_);
lean_inc(v_ngen_237_);
lean_inc(v_maxRecDepth_236_);
lean_inc(v_nextMacroScope_235_);
lean_inc(v_usedQuotCtxts_234_);
lean_inc(v_scopes_233_);
lean_inc(v_messages_232_);
lean_inc(v_env_231_);
lean_dec(v___x_228_);
v___x_244_ = lean_box(0);
v_isShared_245_ = v_isSharedCheck_258_;
goto v_resetjp_243_;
}
v_resetjp_243_:
{
lean_object* v___x_246_; lean_object* v___x_247_; lean_object* v___x_248_; lean_object* v___x_249_; lean_object* v___x_251_; 
v___x_246_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_246_, 0, v_currNamespace_229_);
lean_ctor_set(v___x_246_, 1, v_openDecls_230_);
v___x_247_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_247_, 0, v___x_246_);
lean_ctor_set(v___x_247_, 1, v___y_215_);
lean_inc_ref(v___y_219_);
lean_inc_ref(v___y_214_);
v___x_248_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_248_, 0, v___y_214_);
lean_ctor_set(v___x_248_, 1, v___y_216_);
lean_ctor_set(v___x_248_, 2, v___y_217_);
lean_ctor_set(v___x_248_, 3, v___y_219_);
lean_ctor_set(v___x_248_, 4, v___x_247_);
lean_ctor_set_uint8(v___x_248_, sizeof(void*)*5, v___y_218_);
lean_ctor_set_uint8(v___x_248_, sizeof(void*)*5 + 1, v___y_213_);
lean_ctor_set_uint8(v___x_248_, sizeof(void*)*5 + 2, v_isSilent_208_);
v___x_249_ = l_Lean_MessageLog_add(v___x_248_, v_messages_232_);
if (v_isShared_245_ == 0)
{
lean_ctor_set(v___x_244_, 1, v___x_249_);
v___x_251_ = v___x_244_;
goto v_reusejp_250_;
}
else
{
lean_object* v_reuseFailAlloc_257_; 
v_reuseFailAlloc_257_ = lean_alloc_ctor(0, 12, 0);
lean_ctor_set(v_reuseFailAlloc_257_, 0, v_env_231_);
lean_ctor_set(v_reuseFailAlloc_257_, 1, v___x_249_);
lean_ctor_set(v_reuseFailAlloc_257_, 2, v_scopes_233_);
lean_ctor_set(v_reuseFailAlloc_257_, 3, v_usedQuotCtxts_234_);
lean_ctor_set(v_reuseFailAlloc_257_, 4, v_nextMacroScope_235_);
lean_ctor_set(v_reuseFailAlloc_257_, 5, v_maxRecDepth_236_);
lean_ctor_set(v_reuseFailAlloc_257_, 6, v_ngen_237_);
lean_ctor_set(v_reuseFailAlloc_257_, 7, v_auxDeclNGen_238_);
lean_ctor_set(v_reuseFailAlloc_257_, 8, v_infoState_239_);
lean_ctor_set(v_reuseFailAlloc_257_, 9, v_traceState_240_);
lean_ctor_set(v_reuseFailAlloc_257_, 10, v_snapshotTasks_241_);
lean_ctor_set(v_reuseFailAlloc_257_, 11, v_prevLinterStates_242_);
v___x_251_ = v_reuseFailAlloc_257_;
goto v_reusejp_250_;
}
v_reusejp_250_:
{
lean_object* v___x_252_; lean_object* v___x_253_; lean_object* v___x_255_; 
v___x_252_ = lean_st_ref_set(v___y_220_, v___x_251_);
v___x_253_ = lean_box(0);
if (v_isShared_227_ == 0)
{
lean_ctor_set(v___x_226_, 0, v___x_253_);
v___x_255_ = v___x_226_;
goto v_reusejp_254_;
}
else
{
lean_object* v_reuseFailAlloc_256_; 
v_reuseFailAlloc_256_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_256_, 0, v___x_253_);
v___x_255_ = v_reuseFailAlloc_256_;
goto v_reusejp_254_;
}
v_reusejp_254_:
{
return v___x_255_;
}
}
}
}
}
else
{
lean_object* v_a_260_; lean_object* v___x_262_; uint8_t v_isShared_263_; uint8_t v_isSharedCheck_267_; 
lean_dec(v_a_222_);
lean_dec(v___y_217_);
lean_dec_ref(v___y_216_);
lean_dec_ref(v___y_215_);
v_a_260_ = lean_ctor_get(v___x_223_, 0);
v_isSharedCheck_267_ = !lean_is_exclusive(v___x_223_);
if (v_isSharedCheck_267_ == 0)
{
v___x_262_ = v___x_223_;
v_isShared_263_ = v_isSharedCheck_267_;
goto v_resetjp_261_;
}
else
{
lean_inc(v_a_260_);
lean_dec(v___x_223_);
v___x_262_ = lean_box(0);
v_isShared_263_ = v_isSharedCheck_267_;
goto v_resetjp_261_;
}
v_resetjp_261_:
{
lean_object* v___x_265_; 
if (v_isShared_263_ == 0)
{
v___x_265_ = v___x_262_;
goto v_reusejp_264_;
}
else
{
lean_object* v_reuseFailAlloc_266_; 
v_reuseFailAlloc_266_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_266_, 0, v_a_260_);
v___x_265_ = v_reuseFailAlloc_266_;
goto v_reusejp_264_;
}
v_reusejp_264_:
{
return v___x_265_;
}
}
}
}
else
{
lean_object* v_a_268_; lean_object* v___x_270_; uint8_t v_isShared_271_; uint8_t v_isSharedCheck_275_; 
lean_dec(v___y_217_);
lean_dec_ref(v___y_216_);
lean_dec_ref(v___y_215_);
v_a_268_ = lean_ctor_get(v___x_221_, 0);
v_isSharedCheck_275_ = !lean_is_exclusive(v___x_221_);
if (v_isSharedCheck_275_ == 0)
{
v___x_270_ = v___x_221_;
v_isShared_271_ = v_isSharedCheck_275_;
goto v_resetjp_269_;
}
else
{
lean_inc(v_a_268_);
lean_dec(v___x_221_);
v___x_270_ = lean_box(0);
v_isShared_271_ = v_isSharedCheck_275_;
goto v_resetjp_269_;
}
v_resetjp_269_:
{
lean_object* v___x_273_; 
if (v_isShared_271_ == 0)
{
v___x_273_ = v___x_270_;
goto v_reusejp_272_;
}
else
{
lean_object* v_reuseFailAlloc_274_; 
v_reuseFailAlloc_274_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_274_, 0, v_a_268_);
v___x_273_ = v_reuseFailAlloc_274_;
goto v_reusejp_272_;
}
v_reusejp_272_:
{
return v___x_273_;
}
}
}
}
v___jp_276_:
{
lean_object* v_fileName_282_; lean_object* v_fileMap_283_; uint8_t v_suppressElabErrors_284_; lean_object* v___x_285_; lean_object* v___x_286_; lean_object* v_a_287_; lean_object* v___x_289_; uint8_t v_isShared_290_; uint8_t v_isSharedCheck_303_; 
v_fileName_282_ = lean_ctor_get(v___y_209_, 0);
v_fileMap_283_ = lean_ctor_get(v___y_209_, 1);
v_suppressElabErrors_284_ = lean_ctor_get_uint8(v___y_209_, sizeof(void*)*10);
v___x_285_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_206_);
v___x_286_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__5___redArg(v___x_285_, v___y_210_);
v_a_287_ = lean_ctor_get(v___x_286_, 0);
v_isSharedCheck_303_ = !lean_is_exclusive(v___x_286_);
if (v_isSharedCheck_303_ == 0)
{
v___x_289_ = v___x_286_;
v_isShared_290_ = v_isSharedCheck_303_;
goto v_resetjp_288_;
}
else
{
lean_inc(v_a_287_);
lean_dec(v___x_286_);
v___x_289_ = lean_box(0);
v_isShared_290_ = v_isSharedCheck_303_;
goto v_resetjp_288_;
}
v_resetjp_288_:
{
lean_object* v___x_291_; lean_object* v___x_292_; lean_object* v___x_293_; lean_object* v___x_294_; 
lean_inc_ref_n(v_fileMap_283_, 2);
v___x_291_ = l_Lean_FileMap_toPosition(v_fileMap_283_, v___y_279_);
lean_dec(v___y_279_);
v___x_292_ = l_Lean_FileMap_toPosition(v_fileMap_283_, v___y_281_);
lean_dec(v___y_281_);
v___x_293_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_293_, 0, v___x_292_);
v___x_294_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4___closed__0));
if (v_suppressElabErrors_284_ == 0)
{
lean_del_object(v___x_289_);
v___y_213_ = v___y_278_;
v___y_214_ = v_fileName_282_;
v___y_215_ = v_a_287_;
v___y_216_ = v___x_291_;
v___y_217_ = v___x_293_;
v___y_218_ = v___y_280_;
v___y_219_ = v___x_294_;
v___y_220_ = v___y_210_;
goto v___jp_212_;
}
else
{
lean_object* v___x_295_; lean_object* v___x_296_; lean_object* v___f_297_; uint8_t v___x_298_; 
v___x_295_ = lean_box(v___y_277_);
v___x_296_ = lean_box(v_suppressElabErrors_284_);
v___f_297_ = lean_alloc_closure((void*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4___lam__0___boxed), 3, 2);
lean_closure_set(v___f_297_, 0, v___x_295_);
lean_closure_set(v___f_297_, 1, v___x_296_);
lean_inc(v_a_287_);
v___x_298_ = l_Lean_MessageData_hasTag(v___f_297_, v_a_287_);
if (v___x_298_ == 0)
{
lean_object* v___x_299_; lean_object* v___x_301_; 
lean_dec_ref_known(v___x_293_, 1);
lean_dec_ref(v___x_291_);
lean_dec(v_a_287_);
v___x_299_ = lean_box(0);
if (v_isShared_290_ == 0)
{
lean_ctor_set(v___x_289_, 0, v___x_299_);
v___x_301_ = v___x_289_;
goto v_reusejp_300_;
}
else
{
lean_object* v_reuseFailAlloc_302_; 
v_reuseFailAlloc_302_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_302_, 0, v___x_299_);
v___x_301_ = v_reuseFailAlloc_302_;
goto v_reusejp_300_;
}
v_reusejp_300_:
{
return v___x_301_;
}
}
else
{
lean_del_object(v___x_289_);
v___y_213_ = v___y_278_;
v___y_214_ = v_fileName_282_;
v___y_215_ = v_a_287_;
v___y_216_ = v___x_291_;
v___y_217_ = v___x_293_;
v___y_218_ = v___y_280_;
v___y_219_ = v___x_294_;
v___y_220_ = v___y_210_;
goto v___jp_212_;
}
}
}
}
v___jp_304_:
{
lean_object* v___x_310_; 
v___x_310_ = l_Lean_Syntax_getTailPos_x3f(v___y_307_, v___y_308_);
lean_dec(v___y_307_);
if (lean_obj_tag(v___x_310_) == 0)
{
lean_inc(v___y_309_);
v___y_277_ = v___y_305_;
v___y_278_ = v___y_306_;
v___y_279_ = v___y_309_;
v___y_280_ = v___y_308_;
v___y_281_ = v___y_309_;
goto v___jp_276_;
}
else
{
lean_object* v_val_311_; 
v_val_311_ = lean_ctor_get(v___x_310_, 0);
lean_inc(v_val_311_);
lean_dec_ref_known(v___x_310_, 1);
v___y_277_ = v___y_305_;
v___y_278_ = v___y_306_;
v___y_279_ = v___y_309_;
v___y_280_ = v___y_308_;
v___y_281_ = v_val_311_;
goto v___jp_276_;
}
}
v___jp_312_:
{
lean_object* v___x_316_; 
v___x_316_ = l_Lean_Elab_Command_getRef___redArg(v___y_209_);
if (lean_obj_tag(v___x_316_) == 0)
{
lean_object* v_a_317_; lean_object* v_ref_318_; lean_object* v___x_319_; 
v_a_317_ = lean_ctor_get(v___x_316_, 0);
lean_inc(v_a_317_);
lean_dec_ref_known(v___x_316_, 1);
v_ref_318_ = l_Lean_replaceRef(v_ref_205_, v_a_317_);
lean_dec(v_a_317_);
v___x_319_ = l_Lean_Syntax_getPos_x3f(v_ref_318_, v___y_314_);
if (lean_obj_tag(v___x_319_) == 0)
{
lean_object* v___x_320_; 
v___x_320_ = lean_unsigned_to_nat(0u);
v___y_305_ = v___y_313_;
v___y_306_ = v___y_315_;
v___y_307_ = v_ref_318_;
v___y_308_ = v___y_314_;
v___y_309_ = v___x_320_;
goto v___jp_304_;
}
else
{
lean_object* v_val_321_; 
v_val_321_ = lean_ctor_get(v___x_319_, 0);
lean_inc(v_val_321_);
lean_dec_ref_known(v___x_319_, 1);
v___y_305_ = v___y_313_;
v___y_306_ = v___y_315_;
v___y_307_ = v_ref_318_;
v___y_308_ = v___y_314_;
v___y_309_ = v_val_321_;
goto v___jp_304_;
}
}
else
{
lean_object* v_a_322_; lean_object* v___x_324_; uint8_t v_isShared_325_; uint8_t v_isSharedCheck_329_; 
lean_dec_ref(v_msgData_206_);
v_a_322_ = lean_ctor_get(v___x_316_, 0);
v_isSharedCheck_329_ = !lean_is_exclusive(v___x_316_);
if (v_isSharedCheck_329_ == 0)
{
v___x_324_ = v___x_316_;
v_isShared_325_ = v_isSharedCheck_329_;
goto v_resetjp_323_;
}
else
{
lean_inc(v_a_322_);
lean_dec(v___x_316_);
v___x_324_ = lean_box(0);
v_isShared_325_ = v_isSharedCheck_329_;
goto v_resetjp_323_;
}
v_resetjp_323_:
{
lean_object* v___x_327_; 
if (v_isShared_325_ == 0)
{
v___x_327_ = v___x_324_;
goto v_reusejp_326_;
}
else
{
lean_object* v_reuseFailAlloc_328_; 
v_reuseFailAlloc_328_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_328_, 0, v_a_322_);
v___x_327_ = v_reuseFailAlloc_328_;
goto v_reusejp_326_;
}
v_reusejp_326_:
{
return v___x_327_;
}
}
}
}
v___jp_331_:
{
if (v___y_334_ == 0)
{
v___y_313_ = v___y_332_;
v___y_314_ = v___y_333_;
v___y_315_ = v_severity_207_;
goto v___jp_312_;
}
else
{
v___y_313_ = v___y_332_;
v___y_314_ = v___y_333_;
v___y_315_ = v___x_330_;
goto v___jp_312_;
}
}
v___jp_335_:
{
if (v___y_336_ == 0)
{
lean_object* v___x_337_; lean_object* v_scopes_338_; lean_object* v___x_339_; lean_object* v___x_340_; lean_object* v_opts_341_; uint8_t v___x_342_; uint8_t v___x_343_; 
v___x_337_ = lean_st_ref_get(v___y_210_);
v_scopes_338_ = lean_ctor_get(v___x_337_, 2);
lean_inc(v_scopes_338_);
lean_dec(v___x_337_);
v___x_339_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_340_ = l_List_head_x21___redArg(v___x_339_, v_scopes_338_);
lean_dec(v_scopes_338_);
v_opts_341_ = lean_ctor_get(v___x_340_, 1);
lean_inc_ref(v_opts_341_);
lean_dec(v___x_340_);
v___x_342_ = 1;
v___x_343_ = l_Lean_instBEqMessageSeverity_beq(v_severity_207_, v___x_342_);
if (v___x_343_ == 0)
{
lean_dec_ref(v_opts_341_);
v___y_332_ = v___y_336_;
v___y_333_ = v___y_336_;
v___y_334_ = v___x_343_;
goto v___jp_331_;
}
else
{
lean_object* v___x_344_; uint8_t v___x_345_; 
v___x_344_ = l_Lean_warningAsError;
v___x_345_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__6(v_opts_341_, v___x_344_);
lean_dec_ref(v_opts_341_);
v___y_332_ = v___y_336_;
v___y_333_ = v___y_336_;
v___y_334_ = v___x_345_;
goto v___jp_331_;
}
}
else
{
lean_object* v___x_346_; lean_object* v___x_347_; 
lean_dec_ref(v_msgData_206_);
v___x_346_ = lean_box(0);
v___x_347_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_347_, 0, v___x_346_);
return v___x_347_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4___boxed(lean_object* v_ref_350_, lean_object* v_msgData_351_, lean_object* v_severity_352_, lean_object* v_isSilent_353_, lean_object* v___y_354_, lean_object* v___y_355_, lean_object* v___y_356_){
_start:
{
uint8_t v_severity_boxed_357_; uint8_t v_isSilent_boxed_358_; lean_object* v_res_359_; 
v_severity_boxed_357_ = lean_unbox(v_severity_352_);
v_isSilent_boxed_358_ = lean_unbox(v_isSilent_353_);
v_res_359_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4(v_ref_350_, v_msgData_351_, v_severity_boxed_357_, v_isSilent_boxed_358_, v___y_354_, v___y_355_);
lean_dec(v___y_355_);
lean_dec_ref(v___y_354_);
lean_dec(v_ref_350_);
return v_res_359_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3(lean_object* v_ref_360_, lean_object* v_msgData_361_, lean_object* v___y_362_, lean_object* v___y_363_){
_start:
{
uint8_t v___x_365_; uint8_t v___x_366_; lean_object* v___x_367_; 
v___x_365_ = 1;
v___x_366_ = 0;
v___x_367_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4(v_ref_360_, v_msgData_361_, v___x_365_, v___x_366_, v___y_362_, v___y_363_);
return v___x_367_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3___boxed(lean_object* v_ref_368_, lean_object* v_msgData_369_, lean_object* v___y_370_, lean_object* v___y_371_, lean_object* v___y_372_){
_start:
{
lean_object* v_res_373_; 
v_res_373_ = lp_mathlib_Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3(v_ref_368_, v_msgData_369_, v___y_370_, v___y_371_);
lean_dec(v___y_371_);
lean_dec_ref(v___y_370_);
lean_dec(v_ref_368_);
return v_res_373_;
}
}
static lean_object* _init_lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__1___closed__1(void){
_start:
{
lean_object* v___x_375_; lean_object* v___x_376_; 
v___x_375_ = ((lean_object*)(lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__1___closed__0));
v___x_376_ = l_Lean_stringToMessageData(v___x_375_);
return v___x_376_;
}
}
static lean_object* _init_lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__1___closed__3(void){
_start:
{
lean_object* v___x_378_; lean_object* v___x_379_; 
v___x_378_ = ((lean_object*)(lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__1___closed__2));
v___x_379_ = l_Lean_stringToMessageData(v___x_378_);
return v___x_379_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__1(lean_object* v_linterOption_380_, lean_object* v_stx_381_, lean_object* v_msg_382_, lean_object* v___y_383_, lean_object* v___y_384_){
_start:
{
lean_object* v_name_386_; lean_object* v___x_388_; uint8_t v_isShared_389_; uint8_t v_isSharedCheck_404_; 
v_name_386_ = lean_ctor_get(v_linterOption_380_, 0);
v_isSharedCheck_404_ = !lean_is_exclusive(v_linterOption_380_);
if (v_isSharedCheck_404_ == 0)
{
lean_object* v_unused_405_; 
v_unused_405_ = lean_ctor_get(v_linterOption_380_, 1);
lean_dec(v_unused_405_);
v___x_388_ = v_linterOption_380_;
v_isShared_389_ = v_isSharedCheck_404_;
goto v_resetjp_387_;
}
else
{
lean_inc(v_name_386_);
lean_dec(v_linterOption_380_);
v___x_388_ = lean_box(0);
v_isShared_389_ = v_isSharedCheck_404_;
goto v_resetjp_387_;
}
v_resetjp_387_:
{
lean_object* v___x_390_; lean_object* v___x_391_; lean_object* v___x_393_; 
v___x_390_ = lean_obj_once(&lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__1___closed__1, &lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__1___closed__1_once, _init_lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__1___closed__1);
lean_inc(v_name_386_);
v___x_391_ = l_Lean_MessageData_ofName(v_name_386_);
if (v_isShared_389_ == 0)
{
lean_ctor_set_tag(v___x_388_, 7);
lean_ctor_set(v___x_388_, 1, v___x_391_);
lean_ctor_set(v___x_388_, 0, v___x_390_);
v___x_393_ = v___x_388_;
goto v_reusejp_392_;
}
else
{
lean_object* v_reuseFailAlloc_403_; 
v_reuseFailAlloc_403_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_403_, 0, v___x_390_);
lean_ctor_set(v_reuseFailAlloc_403_, 1, v___x_391_);
v___x_393_ = v_reuseFailAlloc_403_;
goto v_reusejp_392_;
}
v_reusejp_392_:
{
lean_object* v___x_394_; lean_object* v___x_395_; lean_object* v_disable_396_; lean_object* v___x_397_; lean_object* v___x_398_; lean_object* v___x_399_; lean_object* v___x_400_; lean_object* v___x_401_; lean_object* v___x_402_; 
v___x_394_ = lean_obj_once(&lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__1___closed__3, &lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__1___closed__3_once, _init_lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__1___closed__3);
v___x_395_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_395_, 0, v___x_393_);
lean_ctor_set(v___x_395_, 1, v___x_394_);
v_disable_396_ = l_Lean_MessageData_note(v___x_395_);
v___x_397_ = l_Lean_Linter_linterMessageTag;
v___x_398_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_398_, 0, v_msg_382_);
lean_ctor_set(v___x_398_, 1, v_disable_396_);
v___x_399_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_399_, 0, v___x_397_);
lean_ctor_set(v___x_399_, 1, v___x_398_);
v___x_400_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_400_, 0, v_name_386_);
lean_ctor_set(v___x_400_, 1, v___x_399_);
lean_inc(v_stx_381_);
v___x_401_ = lean_alloc_ctor(11, 2, 0);
lean_ctor_set(v___x_401_, 0, v_stx_381_);
lean_ctor_set(v___x_401_, 1, v___x_400_);
v___x_402_ = lp_mathlib_Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3(v_stx_381_, v___x_401_, v___y_383_, v___y_384_);
lean_dec(v_stx_381_);
return v___x_402_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__1___boxed(lean_object* v_linterOption_406_, lean_object* v_stx_407_, lean_object* v_msg_408_, lean_object* v___y_409_, lean_object* v___y_410_, lean_object* v___y_411_){
_start:
{
lean_object* v_res_412_; 
v_res_412_ = lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__1(v_linterOption_406_, v_stx_407_, v_msg_408_, v___y_409_, v___y_410_);
lean_dec(v___y_410_);
lean_dec_ref(v___y_409_);
return v_res_412_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__0_spec__0___redArg(lean_object* v_o_413_, lean_object* v___y_414_){
_start:
{
lean_object* v___x_416_; lean_object* v_env_417_; lean_object* v___x_418_; lean_object* v_toEnvExtension_419_; lean_object* v_asyncMode_420_; lean_object* v___x_421_; lean_object* v___x_422_; lean_object* v___x_423_; lean_object* v_merged_424_; lean_object* v___x_426_; uint8_t v_isShared_427_; uint8_t v_isSharedCheck_432_; 
v___x_416_ = lean_st_ref_get(v___y_414_);
v_env_417_ = lean_ctor_get(v___x_416_, 0);
lean_inc_ref(v_env_417_);
lean_dec(v___x_416_);
v___x_418_ = l_Lean_Linter_linterSetsExt;
v_toEnvExtension_419_ = lean_ctor_get(v___x_418_, 0);
v_asyncMode_420_ = lean_ctor_get(v_toEnvExtension_419_, 2);
v___x_421_ = l_Lean_Linter_instInhabitedLinterSetsState_default;
v___x_422_ = lean_box(0);
v___x_423_ = l_Lean_PersistentEnvExtension_getState___redArg(v___x_421_, v___x_418_, v_env_417_, v_asyncMode_420_, v___x_422_);
v_merged_424_ = lean_ctor_get(v___x_423_, 0);
v_isSharedCheck_432_ = !lean_is_exclusive(v___x_423_);
if (v_isSharedCheck_432_ == 0)
{
lean_object* v_unused_433_; 
v_unused_433_ = lean_ctor_get(v___x_423_, 1);
lean_dec(v_unused_433_);
v___x_426_ = v___x_423_;
v_isShared_427_ = v_isSharedCheck_432_;
goto v_resetjp_425_;
}
else
{
lean_inc(v_merged_424_);
lean_dec(v___x_423_);
v___x_426_ = lean_box(0);
v_isShared_427_ = v_isSharedCheck_432_;
goto v_resetjp_425_;
}
v_resetjp_425_:
{
lean_object* v___x_429_; 
if (v_isShared_427_ == 0)
{
lean_ctor_set(v___x_426_, 1, v_merged_424_);
lean_ctor_set(v___x_426_, 0, v_o_413_);
v___x_429_ = v___x_426_;
goto v_reusejp_428_;
}
else
{
lean_object* v_reuseFailAlloc_431_; 
v_reuseFailAlloc_431_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_431_, 0, v_o_413_);
lean_ctor_set(v_reuseFailAlloc_431_, 1, v_merged_424_);
v___x_429_ = v_reuseFailAlloc_431_;
goto v_reusejp_428_;
}
v_reusejp_428_:
{
lean_object* v___x_430_; 
v___x_430_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_430_, 0, v___x_429_);
return v___x_430_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__0_spec__0___redArg___boxed(lean_object* v_o_434_, lean_object* v___y_435_, lean_object* v___y_436_){
_start:
{
lean_object* v_res_437_; 
v_res_437_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__0_spec__0___redArg(v_o_434_, v___y_435_);
lean_dec(v___y_435_);
return v_res_437_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__0(lean_object* v___y_438_, lean_object* v___y_439_){
_start:
{
lean_object* v___x_441_; lean_object* v_scopes_442_; lean_object* v___x_443_; lean_object* v___x_444_; lean_object* v_opts_445_; lean_object* v___x_446_; 
v___x_441_ = lean_st_ref_get(v___y_439_);
v_scopes_442_ = lean_ctor_get(v___x_441_, 2);
lean_inc(v_scopes_442_);
lean_dec(v___x_441_);
v___x_443_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_444_ = l_List_head_x21___redArg(v___x_443_, v_scopes_442_);
lean_dec(v_scopes_442_);
v_opts_445_ = lean_ctor_get(v___x_444_, 1);
lean_inc_ref(v_opts_445_);
lean_dec(v___x_444_);
v___x_446_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__0_spec__0___redArg(v_opts_445_, v___y_439_);
return v___x_446_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__0___boxed(lean_object* v___y_447_, lean_object* v___y_448_, lean_object* v___y_449_){
_start:
{
lean_object* v_res_450_; 
v_res_450_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__0(v___y_447_, v___y_448_);
lean_dec(v___y_448_);
lean_dec_ref(v___y_447_);
return v_res_450_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__4(uint8_t v___x_451_, lean_object* v_a_452_, lean_object* v_a_453_){
_start:
{
if (lean_obj_tag(v_a_452_) == 0)
{
lean_object* v___x_454_; 
v___x_454_ = l_List_reverse___redArg(v_a_453_);
return v___x_454_;
}
else
{
lean_object* v_head_455_; lean_object* v_tail_456_; lean_object* v___x_458_; uint8_t v_isShared_459_; uint8_t v_isSharedCheck_465_; 
v_head_455_ = lean_ctor_get(v_a_452_, 0);
v_tail_456_ = lean_ctor_get(v_a_452_, 1);
v_isSharedCheck_465_ = !lean_is_exclusive(v_a_452_);
if (v_isSharedCheck_465_ == 0)
{
v___x_458_ = v_a_452_;
v_isShared_459_ = v_isSharedCheck_465_;
goto v_resetjp_457_;
}
else
{
lean_inc(v_tail_456_);
lean_inc(v_head_455_);
lean_dec(v_a_452_);
v___x_458_ = lean_box(0);
v_isShared_459_ = v_isSharedCheck_465_;
goto v_resetjp_457_;
}
v_resetjp_457_:
{
lean_object* v___x_460_; lean_object* v___x_462_; 
v___x_460_ = l_Lean_Name_toString(v_head_455_, v___x_451_);
if (v_isShared_459_ == 0)
{
lean_ctor_set(v___x_458_, 1, v_a_453_);
lean_ctor_set(v___x_458_, 0, v___x_460_);
v___x_462_ = v___x_458_;
goto v_reusejp_461_;
}
else
{
lean_object* v_reuseFailAlloc_464_; 
v_reuseFailAlloc_464_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_464_, 0, v___x_460_);
lean_ctor_set(v_reuseFailAlloc_464_, 1, v_a_453_);
v___x_462_ = v_reuseFailAlloc_464_;
goto v_reusejp_461_;
}
v_reusejp_461_:
{
v_a_452_ = v_tail_456_;
v_a_453_ = v___x_462_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__4___boxed(lean_object* v___x_466_, lean_object* v_a_467_, lean_object* v_a_468_){
_start:
{
uint8_t v___x_5673__boxed_469_; lean_object* v_res_470_; 
v___x_5673__boxed_469_ = lean_unbox(v___x_466_);
v_res_470_ = lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__4(v___x_5673__boxed_469_, v_a_467_, v_a_468_);
return v_res_470_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_elem___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__2(lean_object* v_a_471_, lean_object* v_x_472_){
_start:
{
if (lean_obj_tag(v_x_472_) == 0)
{
uint8_t v___x_473_; 
v___x_473_ = 0;
return v___x_473_;
}
else
{
lean_object* v_head_474_; lean_object* v_tail_475_; uint8_t v___x_476_; 
v_head_474_ = lean_ctor_get(v_x_472_, 0);
v_tail_475_ = lean_ctor_get(v_x_472_, 1);
v___x_476_ = lean_name_eq(v_a_471_, v_head_474_);
if (v___x_476_ == 0)
{
v_x_472_ = v_tail_475_;
goto _start;
}
else
{
return v___x_476_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_elem___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__2___boxed(lean_object* v_a_478_, lean_object* v_x_479_){
_start:
{
uint8_t v_res_480_; lean_object* v_r_481_; 
v_res_480_ = lp_mathlib_List_elem___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__2(v_a_478_, v_x_479_);
lean_dec(v_x_479_);
lean_dec(v_a_478_);
v_r_481_ = lean_box(v_res_480_);
return v_r_481_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__2(void){
_start:
{
lean_object* v___x_484_; lean_object* v___x_485_; 
v___x_484_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__1));
v___x_485_ = l_Lean_stringToMessageData(v___x_484_);
return v___x_485_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__4(void){
_start:
{
lean_object* v___x_487_; lean_object* v___x_488_; 
v___x_487_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__3));
v___x_488_ = l_Lean_stringToMessageData(v___x_487_);
return v___x_488_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__6(void){
_start:
{
lean_object* v___x_490_; lean_object* v___x_491_; 
v___x_490_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__5));
v___x_491_ = l_Lean_stringToMessageData(v___x_490_);
return v___x_491_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__29(void){
_start:
{
lean_object* v___x_534_; lean_object* v___x_535_; 
v___x_534_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__28));
v___x_535_ = l_Lean_MessageData_ofFormat(v___x_534_);
return v___x_535_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__31(void){
_start:
{
lean_object* v___x_537_; lean_object* v___x_538_; 
v___x_537_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__30));
v___x_538_ = l_Lean_stringToMessageData(v___x_537_);
return v___x_538_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__34(void){
_start:
{
lean_object* v___x_541_; lean_object* v___x_542_; 
v___x_541_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__33));
v___x_542_ = l_Lean_stringToMessageData(v___x_541_);
return v___x_542_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__36(void){
_start:
{
lean_object* v___x_544_; lean_object* v___x_545_; 
v___x_544_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__35));
v___x_545_ = l_Lean_stringToMessageData(v___x_544_);
return v___x_545_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0(lean_object* v_stx_546_, lean_object* v___y_547_, lean_object* v___y_548_){
_start:
{
lean_object* v___x_550_; lean_object* v_a_551_; lean_object* v___x_553_; uint8_t v_isShared_554_; uint8_t v_isSharedCheck_621_; 
v___x_550_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__0(v___y_547_, v___y_548_);
v_a_551_ = lean_ctor_get(v___x_550_, 0);
v_isSharedCheck_621_ = !lean_is_exclusive(v___x_550_);
if (v_isSharedCheck_621_ == 0)
{
v___x_553_ = v___x_550_;
v_isShared_554_ = v_isSharedCheck_621_;
goto v_resetjp_552_;
}
else
{
lean_inc(v_a_551_);
lean_dec(v___x_550_);
v___x_553_ = lean_box(0);
v_isShared_554_ = v_isSharedCheck_621_;
goto v_resetjp_552_;
}
v_resetjp_552_:
{
lean_object* v___x_555_; uint8_t v___x_556_; 
v___x_555_ = lp_mathlib_Mathlib_Linter_linter_style_setOption;
v___x_556_ = l_Lean_Linter_getLinterValue(v___x_555_, v_a_551_);
lean_dec(v_a_551_);
if (v___x_556_ == 0)
{
lean_object* v___x_557_; lean_object* v___x_559_; 
lean_dec(v_stx_546_);
v___x_557_ = lean_box(0);
if (v_isShared_554_ == 0)
{
lean_ctor_set(v___x_553_, 0, v___x_557_);
v___x_559_ = v___x_553_;
goto v_reusejp_558_;
}
else
{
lean_object* v_reuseFailAlloc_560_; 
v_reuseFailAlloc_560_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_560_, 0, v___x_557_);
v___x_559_ = v_reuseFailAlloc_560_;
goto v_reusejp_558_;
}
v_reusejp_558_:
{
return v___x_559_;
}
}
else
{
lean_object* v___x_561_; lean_object* v_messages_562_; uint8_t v___x_563_; 
v___x_561_ = lean_st_ref_get(v___y_548_);
v_messages_562_ = lean_ctor_get(v___x_561_, 1);
lean_inc_ref(v_messages_562_);
lean_dec(v___x_561_);
v___x_563_ = l_Lean_MessageLog_hasErrors(v_messages_562_);
lean_dec_ref(v_messages_562_);
if (v___x_563_ == 0)
{
lean_object* v___x_564_; lean_object* v___x_565_; 
v___x_564_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__0));
lean_inc(v_stx_546_);
v___x_565_ = l_Lean_Syntax_find_x3f(v_stx_546_, v___x_564_);
if (lean_obj_tag(v___x_565_) == 1)
{
lean_object* v_val_566_; lean_object* v___x_567_; 
v_val_566_ = lean_ctor_get(v___x_565_, 0);
lean_inc_n(v_val_566_, 2);
lean_dec_ref_known(v___x_565_, 1);
v___x_567_ = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption(v_val_566_);
if (lean_obj_tag(v___x_567_) == 1)
{
lean_object* v_val_568_; lean_object* v___x_579_; lean_object* v___x_580_; lean_object* v___x_581_; uint8_t v___x_582_; 
v_val_568_ = lean_ctor_get(v___x_567_, 0);
lean_inc(v_val_568_);
lean_dec_ref_known(v___x_567_, 1);
v___x_579_ = lean_box(0);
v___x_580_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__17));
v___x_581_ = l_Lean_Name_getRoot(v_val_568_);
v___x_582_ = lp_mathlib_List_elem___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__2(v___x_581_, v___x_580_);
lean_dec(v___x_581_);
if (v___x_582_ == 0)
{
lean_object* v___x_583_; lean_object* v___x_584_; uint8_t v___x_585_; 
lean_inc(v_val_568_);
v___x_583_ = l_Lean_Name_components(v_val_568_);
v___x_584_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__19));
v___x_585_ = lp_mathlib_List_elem___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__2(v___x_584_, v___x_583_);
lean_dec(v___x_583_);
if (v___x_585_ == 0)
{
lean_object* v___x_586_; uint8_t v___x_587_; 
v___x_586_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__21));
v___x_587_ = lean_name_eq(v_val_568_, v___x_586_);
if (v___x_587_ == 0)
{
lean_object* v___x_588_; uint8_t v___x_589_; 
lean_dec(v_val_566_);
v___x_588_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__26));
v___x_589_ = lean_name_eq(v_val_568_, v___x_588_);
lean_dec(v_val_568_);
if (v___x_589_ == 0)
{
lean_object* v___x_590_; lean_object* v___x_592_; 
lean_dec(v_stx_546_);
v___x_590_ = lean_box(0);
if (v_isShared_554_ == 0)
{
lean_ctor_set(v___x_553_, 0, v___x_590_);
v___x_592_ = v___x_553_;
goto v_reusejp_591_;
}
else
{
lean_object* v_reuseFailAlloc_593_; 
v_reuseFailAlloc_593_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_593_, 0, v___x_590_);
v___x_592_ = v_reuseFailAlloc_593_;
goto v_reusejp_591_;
}
v_reusejp_591_:
{
return v___x_592_;
}
}
else
{
lean_object* v___x_594_; lean_object* v___x_595_; 
lean_del_object(v___x_553_);
v___x_594_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__29, &lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__29_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__29);
v___x_595_ = lp_mathlib_Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3(v_stx_546_, v___x_594_, v___y_547_, v___y_548_);
lean_dec(v_stx_546_);
return v___x_595_;
}
}
else
{
lean_del_object(v___x_553_);
lean_dec(v_stx_546_);
goto v___jp_569_;
}
}
else
{
lean_del_object(v___x_553_);
lean_dec(v_stx_546_);
goto v___jp_569_;
}
}
else
{
lean_object* v___x_596_; lean_object* v___x_597_; lean_object* v___x_598_; lean_object* v___x_599_; lean_object* v___x_600_; lean_object* v___x_601_; lean_object* v___x_602_; lean_object* v___x_603_; lean_object* v___x_604_; lean_object* v___x_605_; lean_object* v___x_606_; lean_object* v___x_607_; lean_object* v___x_608_; 
lean_del_object(v___x_553_);
lean_dec(v_stx_546_);
v___x_596_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__31, &lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__31_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__31);
v___x_597_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__32));
v___x_598_ = lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__4(v___x_582_, v___x_580_, v___x_579_);
v___x_599_ = l_String_intercalate(v___x_597_, v___x_598_);
v___x_600_ = l_Lean_stringToMessageData(v___x_599_);
v___x_601_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_601_, 0, v___x_596_);
lean_ctor_set(v___x_601_, 1, v___x_600_);
v___x_602_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__34, &lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__34_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__34);
v___x_603_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_603_, 0, v___x_601_);
lean_ctor_set(v___x_603_, 1, v___x_602_);
v___x_604_ = l_Lean_MessageData_ofName(v_val_568_);
v___x_605_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_605_, 0, v___x_603_);
lean_ctor_set(v___x_605_, 1, v___x_604_);
v___x_606_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__36, &lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__36_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__36);
v___x_607_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_607_, 0, v___x_605_);
lean_ctor_set(v___x_607_, 1, v___x_606_);
v___x_608_ = lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__1(v___x_555_, v_val_566_, v___x_607_, v___y_547_, v___y_548_);
return v___x_608_;
}
v___jp_569_:
{
lean_object* v___x_570_; lean_object* v___x_571_; lean_object* v___x_572_; lean_object* v___x_573_; lean_object* v___x_574_; lean_object* v___x_575_; lean_object* v___x_576_; lean_object* v___x_577_; lean_object* v___x_578_; 
v___x_570_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__2, &lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__2);
v___x_571_ = l_Lean_MessageData_ofName(v_val_568_);
lean_inc_ref(v___x_571_);
v___x_572_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_572_, 0, v___x_570_);
lean_ctor_set(v___x_572_, 1, v___x_571_);
v___x_573_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__4, &lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__4_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__4);
v___x_574_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_574_, 0, v___x_572_);
lean_ctor_set(v___x_574_, 1, v___x_573_);
v___x_575_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_575_, 0, v___x_574_);
lean_ctor_set(v___x_575_, 1, v___x_571_);
v___x_576_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__6, &lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__6_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___closed__6);
v___x_577_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_577_, 0, v___x_575_);
lean_ctor_set(v___x_577_, 1, v___x_576_);
v___x_578_ = lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__1(v___x_555_, v_val_566_, v___x_577_, v___y_547_, v___y_548_);
return v___x_578_;
}
}
else
{
lean_object* v___x_609_; lean_object* v___x_611_; 
lean_dec(v___x_567_);
lean_dec(v_val_566_);
lean_dec(v_stx_546_);
v___x_609_ = lean_box(0);
if (v_isShared_554_ == 0)
{
lean_ctor_set(v___x_553_, 0, v___x_609_);
v___x_611_ = v___x_553_;
goto v_reusejp_610_;
}
else
{
lean_object* v_reuseFailAlloc_612_; 
v_reuseFailAlloc_612_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_612_, 0, v___x_609_);
v___x_611_ = v_reuseFailAlloc_612_;
goto v_reusejp_610_;
}
v_reusejp_610_:
{
return v___x_611_;
}
}
}
else
{
lean_object* v___x_613_; lean_object* v___x_615_; 
lean_dec(v___x_565_);
lean_dec(v_stx_546_);
v___x_613_ = lean_box(0);
if (v_isShared_554_ == 0)
{
lean_ctor_set(v___x_553_, 0, v___x_613_);
v___x_615_ = v___x_553_;
goto v_reusejp_614_;
}
else
{
lean_object* v_reuseFailAlloc_616_; 
v_reuseFailAlloc_616_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_616_, 0, v___x_613_);
v___x_615_ = v_reuseFailAlloc_616_;
goto v_reusejp_614_;
}
v_reusejp_614_:
{
return v___x_615_;
}
}
}
else
{
lean_object* v___x_617_; lean_object* v___x_619_; 
lean_dec(v_stx_546_);
v___x_617_ = lean_box(0);
if (v_isShared_554_ == 0)
{
lean_ctor_set(v___x_553_, 0, v___x_617_);
v___x_619_ = v___x_553_;
goto v_reusejp_618_;
}
else
{
lean_object* v_reuseFailAlloc_620_; 
v_reuseFailAlloc_620_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_620_, 0, v___x_617_);
v___x_619_ = v_reuseFailAlloc_620_;
goto v_reusejp_618_;
}
v_reusejp_618_:
{
return v___x_619_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0___boxed(lean_object* v_stx_622_, lean_object* v___y_623_, lean_object* v___y_624_, lean_object* v___y_625_){
_start:
{
lean_object* v_res_626_; 
v_res_626_ = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter___lam__0(v_stx_622_, v___y_623_, v___y_624_);
lean_dec(v___y_624_);
lean_dec_ref(v___y_623_);
return v_res_626_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__0_spec__0(lean_object* v_o_670_, lean_object* v___y_671_, lean_object* v___y_672_){
_start:
{
lean_object* v___x_674_; 
v___x_674_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__0_spec__0___redArg(v_o_670_, v___y_672_);
return v___x_674_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__0_spec__0___boxed(lean_object* v_o_675_, lean_object* v___y_676_, lean_object* v___y_677_, lean_object* v___y_678_){
_start:
{
lean_object* v_res_679_; 
v_res_679_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__0_spec__0(v_o_675_, v___y_676_, v___y_677_);
lean_dec(v___y_677_);
lean_dec_ref(v___y_676_);
return v_res_679_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__5(lean_object* v_msgData_680_, lean_object* v___y_681_, lean_object* v___y_682_){
_start:
{
lean_object* v___x_684_; 
v___x_684_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__5___redArg(v_msgData_680_, v___y_682_);
return v___x_684_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__5___boxed(lean_object* v_msgData_685_, lean_object* v___y_686_, lean_object* v___y_687_, lean_object* v___y_688_){
_start:
{
lean_object* v_res_689_; 
v_res_689_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__5(v_msgData_685_, v___y_686_, v___y_687_);
lean_dec(v___y_687_);
lean_dec_ref(v___y_686_);
return v_res_689_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_initFn_00___x40_Mathlib_Tactic_Linter_Style_3512398344____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_691_; lean_object* v___x_692_; 
v___x_691_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter));
v___x_692_ = l_Lean_Elab_Command_addLinter(v___x_691_);
return v___x_692_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_initFn_00___x40_Mathlib_Tactic_Linter_Style_3512398344____hygCtx___hyg_2____boxed(lean_object* v_a_693_){
_start:
{
lean_object* v_res_694_; 
v_res_694_ = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_initFn_00___x40_Mathlib_Tactic_Linter_Style_3512398344____hygCtx___hyg_2_();
return v_res_694_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_3409032198____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_713_; lean_object* v___x_714_; lean_object* v___x_715_; lean_object* v___x_716_; 
v___x_713_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_3409032198____hygCtx___hyg_4_));
v___x_714_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Style_3409032198____hygCtx___hyg_4_));
v___x_715_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_3409032198____hygCtx___hyg_4_));
v___x_716_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__spec__0(v___x_713_, v___x_714_, v___x_715_);
return v___x_716_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_3409032198____hygCtx___hyg_4____boxed(lean_object* v_a_717_){
_start:
{
lean_object* v_res_718_; 
v_res_718_ = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_3409032198____hygCtx___hyg_4_();
return v_res_718_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_dropWhile___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter_spec__0(uint8_t v___x_719_, lean_object* v_x_720_){
_start:
{
if (lean_obj_tag(v_x_720_) == 0)
{
return v_x_720_;
}
else
{
lean_object* v_head_721_; lean_object* v_tail_722_; uint8_t v___y_724_; lean_object* v_currNamespace_726_; uint8_t v_isNoncomputable_727_; uint8_t v_isPublic_728_; uint8_t v_isMeta_729_; uint8_t v___x_730_; 
v_head_721_ = lean_ctor_get(v_x_720_, 0);
v_tail_722_ = lean_ctor_get(v_x_720_, 1);
v_currNamespace_726_ = lean_ctor_get(v_head_721_, 2);
v_isNoncomputable_727_ = lean_ctor_get_uint8(v_head_721_, sizeof(void*)*10);
v_isPublic_728_ = lean_ctor_get_uint8(v_head_721_, sizeof(void*)*10 + 1);
v_isMeta_729_ = lean_ctor_get_uint8(v_head_721_, sizeof(void*)*10 + 2);
v___x_730_ = l_Lean_Name_isAnonymous(v_currNamespace_726_);
if (v___x_730_ == 0)
{
v___y_724_ = v___x_730_;
goto v___jp_723_;
}
else
{
if (v_isMeta_729_ == 0)
{
if (v_isPublic_728_ == 0)
{
v___y_724_ = v_isNoncomputable_727_;
goto v___jp_723_;
}
else
{
v___y_724_ = v___x_719_;
goto v___jp_723_;
}
}
else
{
v___y_724_ = v___x_719_;
goto v___jp_723_;
}
}
v___jp_723_:
{
if (v___y_724_ == 0)
{
lean_inc_ref(v_x_720_);
return v_x_720_;
}
else
{
v_x_720_ = v_tail_722_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_dropWhile___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter_spec__0___boxed(lean_object* v___x_731_, lean_object* v_x_732_){
_start:
{
uint8_t v___x_2313__boxed_733_; lean_object* v_res_734_; 
v___x_2313__boxed_733_ = lean_unbox(v___x_731_);
v_res_734_ = lp_mathlib_List_dropWhile___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter_spec__0(v___x_2313__boxed_733_, v_x_732_);
lean_dec(v_x_732_);
return v_res_734_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter_spec__1(lean_object* v_a_735_, lean_object* v_a_736_){
_start:
{
if (lean_obj_tag(v_a_735_) == 0)
{
lean_object* v___x_737_; 
v___x_737_ = l_List_reverse___redArg(v_a_736_);
return v___x_737_;
}
else
{
lean_object* v_head_738_; lean_object* v_tail_739_; lean_object* v___x_741_; uint8_t v_isShared_742_; uint8_t v_isSharedCheck_748_; 
v_head_738_ = lean_ctor_get(v_a_735_, 0);
v_tail_739_ = lean_ctor_get(v_a_735_, 1);
v_isSharedCheck_748_ = !lean_is_exclusive(v_a_735_);
if (v_isSharedCheck_748_ == 0)
{
v___x_741_ = v_a_735_;
v_isShared_742_ = v_isSharedCheck_748_;
goto v_resetjp_740_;
}
else
{
lean_inc(v_tail_739_);
lean_inc(v_head_738_);
lean_dec(v_a_735_);
v___x_741_ = lean_box(0);
v_isShared_742_ = v_isSharedCheck_748_;
goto v_resetjp_740_;
}
v_resetjp_740_:
{
lean_object* v_header_743_; lean_object* v___x_745_; 
v_header_743_ = lean_ctor_get(v_head_738_, 0);
lean_inc_ref(v_header_743_);
lean_dec(v_head_738_);
if (v_isShared_742_ == 0)
{
lean_ctor_set(v___x_741_, 1, v_a_736_);
lean_ctor_set(v___x_741_, 0, v_header_743_);
v___x_745_ = v___x_741_;
goto v_reusejp_744_;
}
else
{
lean_object* v_reuseFailAlloc_747_; 
v_reuseFailAlloc_747_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_747_, 0, v_header_743_);
lean_ctor_set(v_reuseFailAlloc_747_, 1, v_a_736_);
v___x_745_ = v_reuseFailAlloc_747_;
goto v_reusejp_744_;
}
v_reusejp_744_:
{
v_a_735_ = v_tail_739_;
v_a_736_ = v___x_745_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter_spec__2(lean_object* v_x_751_, lean_object* v_x_752_){
_start:
{
if (lean_obj_tag(v_x_752_) == 0)
{
return v_x_751_;
}
else
{
lean_object* v_head_753_; lean_object* v_tail_754_; lean_object* v___x_755_; lean_object* v___y_757_; lean_object* v___x_762_; uint8_t v___x_763_; 
v_head_753_ = lean_ctor_get(v_x_752_, 0);
v_tail_754_ = lean_ctor_get(v_x_752_, 1);
v___x_755_ = ((lean_object*)(lp_mathlib_List_foldl___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter_spec__2___closed__0));
v___x_762_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4___closed__0));
v___x_763_ = lean_string_dec_eq(v_head_753_, v___x_762_);
if (v___x_763_ == 0)
{
lean_object* v___x_764_; 
v___x_764_ = ((lean_object*)(lp_mathlib_List_foldl___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter_spec__2___closed__1));
v___y_757_ = v___x_764_;
goto v___jp_756_;
}
else
{
v___y_757_ = v___x_762_;
goto v___jp_756_;
}
v___jp_756_:
{
lean_object* v___x_758_; lean_object* v___x_759_; lean_object* v___x_760_; 
v___x_758_ = lean_string_append(v___x_755_, v___y_757_);
v___x_759_ = lean_string_append(v___x_758_, v_head_753_);
v___x_760_ = lean_string_append(v_x_751_, v___x_759_);
lean_dec_ref(v___x_759_);
v_x_751_ = v___x_760_;
v_x_752_ = v_tail_754_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter_spec__2___boxed(lean_object* v_x_765_, lean_object* v_x_766_){
_start:
{
lean_object* v_res_767_; 
v_res_767_ = lp_mathlib_List_foldl___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter_spec__2(v_x_765_, v_x_766_);
lean_dec(v_x_766_);
return v_res_767_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___lam__0___closed__3(void){
_start:
{
lean_object* v___x_775_; lean_object* v___x_776_; 
v___x_775_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___lam__0___closed__2));
v___x_776_ = l_Lean_stringToMessageData(v___x_775_);
return v___x_776_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___lam__0___closed__5(void){
_start:
{
lean_object* v___x_778_; lean_object* v___x_779_; 
v___x_778_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___lam__0___closed__4));
v___x_779_ = l_Lean_stringToMessageData(v___x_778_);
return v___x_779_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___lam__0(lean_object* v_stx_780_, lean_object* v___y_781_, lean_object* v___y_782_){
_start:
{
lean_object* v___x_784_; uint8_t v___x_785_; 
v___x_784_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___lam__0___closed__1));
lean_inc(v_stx_780_);
v___x_785_ = l_Lean_Syntax_isOfKind(v_stx_780_, v___x_784_);
if (v___x_785_ == 0)
{
lean_object* v___x_786_; lean_object* v___x_787_; 
lean_dec(v_stx_780_);
v___x_786_ = lean_box(0);
v___x_787_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_787_, 0, v___x_786_);
return v___x_787_;
}
else
{
lean_object* v___x_788_; lean_object* v_a_789_; lean_object* v___x_791_; uint8_t v_isShared_792_; uint8_t v_isSharedCheck_849_; 
v___x_788_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__0(v___y_781_, v___y_782_);
v_a_789_ = lean_ctor_get(v___x_788_, 0);
v_isSharedCheck_849_ = !lean_is_exclusive(v___x_788_);
if (v_isSharedCheck_849_ == 0)
{
v___x_791_ = v___x_788_;
v_isShared_792_ = v_isSharedCheck_849_;
goto v_resetjp_790_;
}
else
{
lean_inc(v_a_789_);
lean_dec(v___x_788_);
v___x_791_ = lean_box(0);
v_isShared_792_ = v_isSharedCheck_849_;
goto v_resetjp_790_;
}
v_resetjp_790_:
{
lean_object* v___x_793_; lean_object* v___x_799_; lean_object* v___y_801_; uint8_t v___y_813_; uint8_t v___x_846_; 
v___x_793_ = lean_st_ref_get(v___y_782_);
v___x_799_ = lp_mathlib_Mathlib_Linter_linter_style_missingEnd;
v___x_846_ = l_Lean_Linter_getLinterValue(v___x_799_, v_a_789_);
lean_dec(v_a_789_);
if (v___x_846_ == 0)
{
lean_dec(v___x_793_);
v___y_813_ = v___x_846_;
goto v___jp_812_;
}
else
{
lean_object* v_messages_847_; uint8_t v___x_848_; 
v_messages_847_ = lean_ctor_get(v___x_793_, 1);
lean_inc_ref(v_messages_847_);
lean_dec(v___x_793_);
v___x_848_ = l_Lean_MessageLog_hasErrors(v_messages_847_);
lean_dec_ref(v_messages_847_);
if (v___x_848_ == 0)
{
v___y_813_ = v___x_846_;
goto v___jp_812_;
}
else
{
lean_dec(v_stx_780_);
goto v___jp_794_;
}
}
v___jp_794_:
{
lean_object* v___x_795_; lean_object* v___x_797_; 
v___x_795_ = lean_box(0);
if (v_isShared_792_ == 0)
{
lean_ctor_set(v___x_791_, 0, v___x_795_);
v___x_797_ = v___x_791_;
goto v_reusejp_796_;
}
else
{
lean_object* v_reuseFailAlloc_798_; 
v_reuseFailAlloc_798_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_798_, 0, v___x_795_);
v___x_797_ = v_reuseFailAlloc_798_;
goto v_reusejp_796_;
}
v_reusejp_796_:
{
return v___x_797_;
}
}
v___jp_800_:
{
lean_object* v___x_802_; lean_object* v___x_803_; lean_object* v___x_804_; lean_object* v___x_805_; lean_object* v___x_806_; lean_object* v___x_807_; lean_object* v___x_808_; lean_object* v___x_809_; lean_object* v___x_810_; lean_object* v___x_811_; 
v___x_802_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4___closed__0));
v___x_803_ = lean_box(0);
v___x_804_ = lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter_spec__1(v___y_801_, v___x_803_);
v___x_805_ = lp_mathlib_List_foldl___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter_spec__2(v___x_802_, v___x_804_);
lean_dec(v___x_804_);
v___x_806_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___lam__0___closed__3, &lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___lam__0___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___lam__0___closed__3);
v___x_807_ = l_Lean_stringToMessageData(v___x_805_);
v___x_808_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_808_, 0, v___x_806_);
lean_ctor_set(v___x_808_, 1, v___x_807_);
v___x_809_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___lam__0___closed__5, &lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___lam__0___closed__5_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___lam__0___closed__5);
v___x_810_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_810_, 0, v___x_808_);
lean_ctor_set(v___x_810_, 1, v___x_809_);
v___x_811_ = lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__1(v___x_799_, v_stx_780_, v___x_810_, v___y_781_, v___y_782_);
return v___x_811_;
}
v___jp_812_:
{
if (v___y_813_ == 0)
{
lean_dec(v_stx_780_);
goto v___jp_794_;
}
else
{
lean_object* v___x_814_; 
lean_del_object(v___x_791_);
v___x_814_ = l_Lean_Elab_Command_getScopes___redArg(v___y_782_);
if (lean_obj_tag(v___x_814_) == 0)
{
lean_object* v_a_815_; lean_object* v___x_817_; uint8_t v_isShared_818_; uint8_t v_isSharedCheck_837_; 
v_a_815_ = lean_ctor_get(v___x_814_, 0);
v_isSharedCheck_837_ = !lean_is_exclusive(v___x_814_);
if (v_isSharedCheck_837_ == 0)
{
v___x_817_ = v___x_814_;
v_isShared_818_ = v_isSharedCheck_837_;
goto v_resetjp_816_;
}
else
{
lean_inc(v_a_815_);
lean_dec(v___x_814_);
v___x_817_ = lean_box(0);
v_isShared_818_ = v_isSharedCheck_837_;
goto v_resetjp_816_;
}
v_resetjp_816_:
{
lean_object* v___x_819_; lean_object* v___x_820_; uint8_t v___x_821_; 
v___x_819_ = l_List_lengthTR___redArg(v_a_815_);
v___x_820_ = lean_unsigned_to_nat(1u);
v___x_821_ = lean_nat_dec_eq(v___x_819_, v___x_820_);
lean_dec(v___x_819_);
if (v___x_821_ == 0)
{
lean_object* v___x_822_; lean_object* v___x_823_; lean_object* v___x_824_; lean_object* v___x_825_; lean_object* v___x_826_; lean_object* v___x_827_; uint8_t v___x_828_; 
v___x_822_ = lean_array_mk(v_a_815_);
v___x_823_ = lean_array_pop(v___x_822_);
v___x_824_ = lean_array_to_list(v___x_823_);
v___x_825_ = l_List_reverse___redArg(v___x_824_);
v___x_826_ = lp_mathlib_List_dropWhile___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter_spec__0(v___x_785_, v___x_825_);
lean_dec(v___x_825_);
v___x_827_ = l_List_reverse___redArg(v___x_826_);
v___x_828_ = l_List_isEmpty___redArg(v___x_827_);
if (v___x_828_ == 0)
{
lean_del_object(v___x_817_);
v___y_801_ = v___x_827_;
goto v___jp_800_;
}
else
{
if (v___x_821_ == 0)
{
lean_object* v___x_829_; lean_object* v___x_831_; 
lean_dec(v___x_827_);
lean_dec(v_stx_780_);
v___x_829_ = lean_box(0);
if (v_isShared_818_ == 0)
{
lean_ctor_set(v___x_817_, 0, v___x_829_);
v___x_831_ = v___x_817_;
goto v_reusejp_830_;
}
else
{
lean_object* v_reuseFailAlloc_832_; 
v_reuseFailAlloc_832_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_832_, 0, v___x_829_);
v___x_831_ = v_reuseFailAlloc_832_;
goto v_reusejp_830_;
}
v_reusejp_830_:
{
return v___x_831_;
}
}
else
{
lean_del_object(v___x_817_);
v___y_801_ = v___x_827_;
goto v___jp_800_;
}
}
}
else
{
lean_object* v___x_833_; lean_object* v___x_835_; 
lean_dec(v_a_815_);
lean_dec(v_stx_780_);
v___x_833_ = lean_box(0);
if (v_isShared_818_ == 0)
{
lean_ctor_set(v___x_817_, 0, v___x_833_);
v___x_835_ = v___x_817_;
goto v_reusejp_834_;
}
else
{
lean_object* v_reuseFailAlloc_836_; 
v_reuseFailAlloc_836_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_836_, 0, v___x_833_);
v___x_835_ = v_reuseFailAlloc_836_;
goto v_reusejp_834_;
}
v_reusejp_834_:
{
return v___x_835_;
}
}
}
}
else
{
lean_object* v_a_838_; lean_object* v___x_840_; uint8_t v_isShared_841_; uint8_t v_isSharedCheck_845_; 
lean_dec(v_stx_780_);
v_a_838_ = lean_ctor_get(v___x_814_, 0);
v_isSharedCheck_845_ = !lean_is_exclusive(v___x_814_);
if (v_isSharedCheck_845_ == 0)
{
v___x_840_ = v___x_814_;
v_isShared_841_ = v_isSharedCheck_845_;
goto v_resetjp_839_;
}
else
{
lean_inc(v_a_838_);
lean_dec(v___x_814_);
v___x_840_ = lean_box(0);
v_isShared_841_ = v_isSharedCheck_845_;
goto v_resetjp_839_;
}
v_resetjp_839_:
{
lean_object* v___x_843_; 
if (v_isShared_841_ == 0)
{
v___x_843_ = v___x_840_;
goto v_reusejp_842_;
}
else
{
lean_object* v_reuseFailAlloc_844_; 
v_reuseFailAlloc_844_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_844_, 0, v_a_838_);
v___x_843_ = v_reuseFailAlloc_844_;
goto v_reusejp_842_;
}
v_reusejp_842_:
{
return v___x_843_;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___lam__0___boxed(lean_object* v_stx_850_, lean_object* v___y_851_, lean_object* v___y_852_, lean_object* v___y_853_){
_start:
{
lean_object* v_res_854_; 
v_res_854_ = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___lam__0(v_stx_850_, v___y_851_, v___y_852_);
lean_dec(v___y_852_);
lean_dec_ref(v___y_851_);
return v_res_854_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_initFn_00___x40_Mathlib_Tactic_Linter_Style_3360231377____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_870_; lean_object* v___x_871_; 
v___x_870_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter));
v___x_871_ = l_Lean_Elab_Command_addLinter(v___x_870_);
return v___x_871_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_initFn_00___x40_Mathlib_Tactic_Linter_Style_3360231377____hygCtx___hyg_2____boxed(lean_object* v_a_872_){
_start:
{
lean_object* v_res_873_; 
v_res_873_ = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_initFn_00___x40_Mathlib_Tactic_Linter_Style_3360231377____hygCtx___hyg_2_();
return v_res_873_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_3789867222____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_892_; lean_object* v___x_893_; lean_object* v___x_894_; lean_object* v___x_895_; 
v___x_892_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_3789867222____hygCtx___hyg_4_));
v___x_893_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Style_3789867222____hygCtx___hyg_4_));
v___x_894_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_3789867222____hygCtx___hyg_4_));
v___x_895_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__spec__0(v___x_892_, v___x_893_, v___x_894_);
return v___x_895_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_3789867222____hygCtx___hyg_4____boxed(lean_object* v_a_896_){
_start:
{
lean_object* v_res_897_; 
v_res_897_ = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_3789867222____hygCtx___hyg_4_();
return v_res_897_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Linter_isCDot_x3f(lean_object* v_x_900_){
_start:
{
if (lean_obj_tag(v_x_900_) == 1)
{
lean_object* v_kind_901_; 
v_kind_901_ = lean_ctor_get(v_x_900_, 1);
if (lean_obj_tag(v_kind_901_) == 1)
{
lean_object* v_pre_902_; 
v_pre_902_ = lean_ctor_get(v_kind_901_, 0);
if (lean_obj_tag(v_pre_902_) == 1)
{
lean_object* v_pre_903_; 
v_pre_903_ = lean_ctor_get(v_pre_902_, 0);
switch(lean_obj_tag(v_pre_903_))
{
case 0:
{
lean_object* v_args_904_; lean_object* v_str_905_; lean_object* v_str_906_; lean_object* v___x_907_; uint8_t v___x_908_; 
v_args_904_ = lean_ctor_get(v_x_900_, 2);
v_str_905_ = lean_ctor_get(v_kind_901_, 1);
v_str_906_ = lean_ctor_get(v_pre_902_, 1);
v___x_907_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__0));
v___x_908_ = lean_string_dec_eq(v_str_906_, v___x_907_);
if (v___x_908_ == 0)
{
return v___x_908_;
}
else
{
lean_object* v___x_909_; uint8_t v___x_910_; 
v___x_909_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_isCDot_x3f___closed__0));
v___x_910_ = lean_string_dec_eq(v_str_905_, v___x_909_);
if (v___x_910_ == 0)
{
return v___x_910_;
}
else
{
lean_object* v___x_911_; lean_object* v___x_912_; uint8_t v___x_913_; 
v___x_911_ = lean_array_get_size(v_args_904_);
v___x_912_ = lean_unsigned_to_nat(1u);
v___x_913_ = lean_nat_dec_eq(v___x_911_, v___x_912_);
if (v___x_913_ == 0)
{
return v___x_913_;
}
else
{
lean_object* v___x_914_; lean_object* v___x_915_; 
v___x_914_ = lean_unsigned_to_nat(0u);
v___x_915_ = lean_array_fget_borrowed(v_args_904_, v___x_914_);
if (lean_obj_tag(v___x_915_) == 2)
{
lean_object* v_val_916_; lean_object* v___x_917_; uint8_t v___x_918_; 
v_val_916_ = lean_ctor_get(v___x_915_, 1);
v___x_917_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_isCDot_x3f___closed__1));
v___x_918_ = lean_string_dec_eq(v_val_916_, v___x_917_);
return v___x_918_;
}
else
{
uint8_t v___x_919_; 
v___x_919_ = 0;
return v___x_919_;
}
}
}
}
}
case 1:
{
lean_object* v_pre_920_; 
v_pre_920_ = lean_ctor_get(v_pre_903_, 0);
if (lean_obj_tag(v_pre_920_) == 1)
{
lean_object* v_pre_921_; 
v_pre_921_ = lean_ctor_get(v_pre_920_, 0);
if (lean_obj_tag(v_pre_921_) == 0)
{
lean_object* v_args_922_; lean_object* v_str_923_; lean_object* v_str_924_; lean_object* v_str_925_; lean_object* v_str_926_; lean_object* v___x_927_; uint8_t v___x_928_; 
v_args_922_ = lean_ctor_get(v_x_900_, 2);
v_str_923_ = lean_ctor_get(v_kind_901_, 1);
v_str_924_ = lean_ctor_get(v_pre_902_, 1);
v_str_925_ = lean_ctor_get(v_pre_903_, 1);
v_str_926_ = lean_ctor_get(v_pre_920_, 1);
v___x_927_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__0));
v___x_928_ = lean_string_dec_eq(v_str_926_, v___x_927_);
if (v___x_928_ == 0)
{
return v___x_928_;
}
else
{
lean_object* v___x_929_; uint8_t v___x_930_; 
v___x_929_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__1));
v___x_930_ = lean_string_dec_eq(v_str_925_, v___x_929_);
if (v___x_930_ == 0)
{
return v___x_930_;
}
else
{
lean_object* v___x_931_; uint8_t v___x_932_; 
v___x_931_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__5));
v___x_932_ = lean_string_dec_eq(v_str_924_, v___x_931_);
if (v___x_932_ == 0)
{
return v___x_932_;
}
else
{
lean_object* v___x_933_; uint8_t v___x_934_; 
v___x_933_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_3789867222____hygCtx___hyg_4_));
v___x_934_ = lean_string_dec_eq(v_str_923_, v___x_933_);
if (v___x_934_ == 0)
{
return v___x_934_;
}
else
{
lean_object* v___x_935_; lean_object* v___x_936_; uint8_t v___x_937_; 
v___x_935_ = lean_array_get_size(v_args_922_);
v___x_936_ = lean_unsigned_to_nat(2u);
v___x_937_ = lean_nat_dec_eq(v___x_935_, v___x_936_);
if (v___x_937_ == 0)
{
return v___x_937_;
}
else
{
lean_object* v___x_938_; lean_object* v___x_939_; 
v___x_938_ = lean_unsigned_to_nat(0u);
v___x_939_ = lean_array_fget_borrowed(v_args_922_, v___x_938_);
if (lean_obj_tag(v___x_939_) == 2)
{
lean_object* v_val_940_; lean_object* v___x_941_; uint8_t v___x_942_; 
v_val_940_ = lean_ctor_get(v___x_939_, 1);
v___x_941_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_isCDot_x3f___closed__1));
v___x_942_ = lean_string_dec_eq(v_val_940_, v___x_941_);
return v___x_942_;
}
else
{
uint8_t v___x_943_; 
v___x_943_ = 0;
return v___x_943_;
}
}
}
}
}
}
}
else
{
uint8_t v___x_944_; 
v___x_944_ = 0;
return v___x_944_;
}
}
else
{
uint8_t v___x_945_; 
v___x_945_ = 0;
return v___x_945_;
}
}
default: 
{
uint8_t v___x_946_; 
v___x_946_ = 0;
return v___x_946_;
}
}
}
else
{
uint8_t v___x_947_; 
v___x_947_ = 0;
return v___x_947_;
}
}
else
{
uint8_t v___x_948_; 
v___x_948_ = 0;
return v___x_948_;
}
}
else
{
uint8_t v___x_949_; 
v___x_949_ = 0;
return v___x_949_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_isCDot_x3f___boxed(lean_object* v_x_950_){
_start:
{
uint8_t v_res_951_; lean_object* v_r_952_; 
v_res_951_ = lp_mathlib_Mathlib_Linter_isCDot_x3f(v_x_950_);
lean_dec(v_x_950_);
v_r_952_ = lean_box(v_res_951_);
return v_r_952_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_findCDot_spec__1(lean_object* v_as_953_, size_t v_i_954_, size_t v_stop_955_, lean_object* v_b_956_){
_start:
{
uint8_t v___x_957_; 
v___x_957_ = lean_usize_dec_eq(v_i_954_, v_stop_955_);
if (v___x_957_ == 0)
{
lean_object* v___x_958_; lean_object* v___x_959_; size_t v___x_960_; size_t v___x_961_; 
v___x_958_ = lean_array_uget_borrowed(v_as_953_, v_i_954_);
v___x_959_ = l_Array_append___redArg(v_b_956_, v___x_958_);
v___x_960_ = ((size_t)1ULL);
v___x_961_ = lean_usize_add(v_i_954_, v___x_960_);
v_i_954_ = v___x_961_;
v_b_956_ = v___x_959_;
goto _start;
}
else
{
return v_b_956_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_findCDot_spec__1___boxed(lean_object* v_as_963_, lean_object* v_i_964_, lean_object* v_stop_965_, lean_object* v_b_966_){
_start:
{
size_t v_i_boxed_967_; size_t v_stop_boxed_968_; lean_object* v_res_969_; 
v_i_boxed_967_ = lean_unbox_usize(v_i_964_);
lean_dec(v_i_964_);
v_stop_boxed_968_ = lean_unbox_usize(v_stop_965_);
lean_dec(v_stop_965_);
v_res_969_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_findCDot_spec__1(v_as_963_, v_i_boxed_967_, v_stop_boxed_968_, v_b_966_);
lean_dec_ref(v_as_963_);
return v_res_969_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_findCDot(lean_object* v_x_974_){
_start:
{
if (lean_obj_tag(v_x_974_) == 1)
{
lean_object* v_kind_975_; lean_object* v_args_976_; lean_object* v___y_978_; size_t v_sz_1003_; size_t v___x_1004_; lean_object* v___x_1005_; lean_object* v___x_1006_; lean_object* v___x_1007_; lean_object* v___x_1008_; uint8_t v___x_1009_; 
v_kind_975_ = lean_ctor_get(v_x_974_, 1);
v_args_976_ = lean_ctor_get(v_x_974_, 2);
v_sz_1003_ = lean_array_size(v_args_976_);
v___x_1004_ = ((size_t)0ULL);
lean_inc_ref(v_args_976_);
v___x_1005_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_findCDot_spec__0(v_sz_1003_, v___x_1004_, v_args_976_);
v___x_1006_ = lean_unsigned_to_nat(0u);
v___x_1007_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_findCDot___closed__0));
v___x_1008_ = lean_array_get_size(v___x_1005_);
v___x_1009_ = lean_nat_dec_lt(v___x_1006_, v___x_1008_);
if (v___x_1009_ == 0)
{
lean_dec_ref(v___x_1005_);
v___y_978_ = v___x_1007_;
goto v___jp_977_;
}
else
{
uint8_t v___x_1010_; 
v___x_1010_ = lean_nat_dec_le(v___x_1008_, v___x_1008_);
if (v___x_1010_ == 0)
{
if (v___x_1009_ == 0)
{
lean_dec_ref(v___x_1005_);
v___y_978_ = v___x_1007_;
goto v___jp_977_;
}
else
{
size_t v___x_1011_; lean_object* v___x_1012_; 
v___x_1011_ = lean_usize_of_nat(v___x_1008_);
v___x_1012_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_findCDot_spec__1(v___x_1005_, v___x_1004_, v___x_1011_, v___x_1007_);
lean_dec_ref(v___x_1005_);
v___y_978_ = v___x_1012_;
goto v___jp_977_;
}
}
else
{
size_t v___x_1013_; lean_object* v___x_1014_; 
v___x_1013_ = lean_usize_of_nat(v___x_1008_);
v___x_1014_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_findCDot_spec__1(v___x_1005_, v___x_1004_, v___x_1013_, v___x_1007_);
lean_dec_ref(v___x_1005_);
v___y_978_ = v___x_1014_;
goto v___jp_977_;
}
}
v___jp_977_:
{
if (lean_obj_tag(v_kind_975_) == 1)
{
lean_object* v_pre_979_; 
v_pre_979_ = lean_ctor_get(v_kind_975_, 0);
if (lean_obj_tag(v_pre_979_) == 1)
{
lean_object* v_pre_980_; 
v_pre_980_ = lean_ctor_get(v_pre_979_, 0);
switch(lean_obj_tag(v_pre_980_))
{
case 1:
{
lean_object* v_pre_981_; 
v_pre_981_ = lean_ctor_get(v_pre_980_, 0);
if (lean_obj_tag(v_pre_981_) == 1)
{
lean_object* v_pre_982_; 
v_pre_982_ = lean_ctor_get(v_pre_981_, 0);
if (lean_obj_tag(v_pre_982_) == 0)
{
lean_object* v_str_983_; lean_object* v_str_984_; lean_object* v_str_985_; lean_object* v_str_986_; lean_object* v___x_987_; uint8_t v___x_988_; 
v_str_983_ = lean_ctor_get(v_kind_975_, 1);
v_str_984_ = lean_ctor_get(v_pre_979_, 1);
v_str_985_ = lean_ctor_get(v_pre_980_, 1);
v_str_986_ = lean_ctor_get(v_pre_981_, 1);
v___x_987_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__0));
v___x_988_ = lean_string_dec_eq(v_str_986_, v___x_987_);
if (v___x_988_ == 0)
{
lean_dec_ref_known(v_x_974_, 3);
return v___y_978_;
}
else
{
lean_object* v___x_989_; uint8_t v___x_990_; 
v___x_989_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__1));
v___x_990_ = lean_string_dec_eq(v_str_985_, v___x_989_);
if (v___x_990_ == 0)
{
lean_dec_ref_known(v_x_974_, 3);
return v___y_978_;
}
else
{
lean_object* v___x_991_; uint8_t v___x_992_; 
v___x_991_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__5));
v___x_992_ = lean_string_dec_eq(v_str_984_, v___x_991_);
if (v___x_992_ == 0)
{
lean_dec_ref_known(v_x_974_, 3);
return v___y_978_;
}
else
{
lean_object* v___x_993_; uint8_t v___x_994_; 
v___x_993_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_3789867222____hygCtx___hyg_4_));
v___x_994_ = lean_string_dec_eq(v_str_983_, v___x_993_);
if (v___x_994_ == 0)
{
lean_dec_ref_known(v_x_974_, 3);
return v___y_978_;
}
else
{
lean_object* v___x_995_; 
v___x_995_ = lean_array_push(v___y_978_, v_x_974_);
return v___x_995_;
}
}
}
}
}
else
{
lean_dec_ref_known(v_x_974_, 3);
return v___y_978_;
}
}
else
{
lean_dec_ref_known(v_x_974_, 3);
return v___y_978_;
}
}
case 0:
{
lean_object* v_str_996_; lean_object* v_str_997_; lean_object* v___x_998_; uint8_t v___x_999_; 
v_str_996_ = lean_ctor_get(v_kind_975_, 1);
v_str_997_ = lean_ctor_get(v_pre_979_, 1);
v___x_998_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__0));
v___x_999_ = lean_string_dec_eq(v_str_997_, v___x_998_);
if (v___x_999_ == 0)
{
lean_dec_ref_known(v_x_974_, 3);
return v___y_978_;
}
else
{
lean_object* v___x_1000_; uint8_t v___x_1001_; 
v___x_1000_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_isCDot_x3f___closed__0));
v___x_1001_ = lean_string_dec_eq(v_str_996_, v___x_1000_);
if (v___x_1001_ == 0)
{
lean_dec_ref_known(v_x_974_, 3);
return v___y_978_;
}
else
{
lean_object* v___x_1002_; 
v___x_1002_ = lean_array_push(v___y_978_, v_x_974_);
return v___x_1002_;
}
}
}
default: 
{
lean_dec_ref_known(v_x_974_, 3);
return v___y_978_;
}
}
}
else
{
lean_dec_ref_known(v_x_974_, 3);
return v___y_978_;
}
}
else
{
lean_dec_ref_known(v_x_974_, 3);
return v___y_978_;
}
}
}
else
{
lean_object* v___x_1015_; 
lean_dec(v_x_974_);
v___x_1015_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_findCDot___closed__1));
return v___x_1015_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_findCDot_spec__0(size_t v_sz_1016_, size_t v_i_1017_, lean_object* v_bs_1018_){
_start:
{
uint8_t v___x_1019_; 
v___x_1019_ = lean_usize_dec_lt(v_i_1017_, v_sz_1016_);
if (v___x_1019_ == 0)
{
return v_bs_1018_;
}
else
{
lean_object* v_v_1020_; lean_object* v___x_1021_; lean_object* v_bs_x27_1022_; lean_object* v___x_1023_; size_t v___x_1024_; size_t v___x_1025_; lean_object* v___x_1026_; 
v_v_1020_ = lean_array_uget(v_bs_1018_, v_i_1017_);
v___x_1021_ = lean_unsigned_to_nat(0u);
v_bs_x27_1022_ = lean_array_uset(v_bs_1018_, v_i_1017_, v___x_1021_);
v___x_1023_ = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_findCDot(v_v_1020_);
v___x_1024_ = ((size_t)1ULL);
v___x_1025_ = lean_usize_add(v_i_1017_, v___x_1024_);
v___x_1026_ = lean_array_uset(v_bs_x27_1022_, v_i_1017_, v___x_1023_);
v_i_1017_ = v___x_1025_;
v_bs_1018_ = v___x_1026_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_findCDot_spec__0___boxed(lean_object* v_sz_1028_, lean_object* v_i_1029_, lean_object* v_bs_1030_){
_start:
{
size_t v_sz_boxed_1031_; size_t v_i_boxed_1032_; lean_object* v_res_1033_; 
v_sz_boxed_1031_ = lean_unbox_usize(v_sz_1028_);
lean_dec(v_sz_1028_);
v_i_boxed_1032_ = lean_unbox_usize(v_i_1029_);
lean_dec(v_i_1029_);
v_res_1033_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_findCDot_spec__0(v_sz_boxed_1031_, v_i_boxed_1032_, v_bs_1030_);
return v_res_1033_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_unwanted__cdot_spec__0(lean_object* v_as_1034_, size_t v_i_1035_, size_t v_stop_1036_, lean_object* v_b_1037_){
_start:
{
lean_object* v___y_1039_; uint8_t v___x_1043_; 
v___x_1043_ = lean_usize_dec_eq(v_i_1035_, v_stop_1036_);
if (v___x_1043_ == 0)
{
lean_object* v___x_1044_; uint8_t v___x_1045_; 
v___x_1044_ = lean_array_uget_borrowed(v_as_1034_, v_i_1035_);
v___x_1045_ = lp_mathlib_Mathlib_Linter_isCDot_x3f(v___x_1044_);
if (v___x_1045_ == 0)
{
lean_object* v___x_1046_; 
lean_inc(v___x_1044_);
v___x_1046_ = lean_array_push(v_b_1037_, v___x_1044_);
v___y_1039_ = v___x_1046_;
goto v___jp_1038_;
}
else
{
v___y_1039_ = v_b_1037_;
goto v___jp_1038_;
}
}
else
{
return v_b_1037_;
}
v___jp_1038_:
{
size_t v___x_1040_; size_t v___x_1041_; 
v___x_1040_ = ((size_t)1ULL);
v___x_1041_ = lean_usize_add(v_i_1035_, v___x_1040_);
v_i_1035_ = v___x_1041_;
v_b_1037_ = v___y_1039_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_unwanted__cdot_spec__0___boxed(lean_object* v_as_1047_, lean_object* v_i_1048_, lean_object* v_stop_1049_, lean_object* v_b_1050_){
_start:
{
size_t v_i_boxed_1051_; size_t v_stop_boxed_1052_; lean_object* v_res_1053_; 
v_i_boxed_1051_ = lean_unbox_usize(v_i_1048_);
lean_dec(v_i_1048_);
v_stop_boxed_1052_ = lean_unbox_usize(v_stop_1049_);
lean_dec(v_stop_1049_);
v_res_1053_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_unwanted__cdot_spec__0(v_as_1047_, v_i_boxed_1051_, v_stop_boxed_1052_, v_b_1050_);
lean_dec_ref(v_as_1047_);
return v_res_1053_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_unwanted__cdot(lean_object* v_stx_1054_){
_start:
{
lean_object* v___x_1055_; lean_object* v___x_1056_; lean_object* v___x_1057_; lean_object* v___x_1058_; uint8_t v___x_1059_; 
v___x_1055_ = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_findCDot(v_stx_1054_);
v___x_1056_ = lean_unsigned_to_nat(0u);
v___x_1057_ = lean_array_get_size(v___x_1055_);
v___x_1058_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_findCDot___closed__1));
v___x_1059_ = lean_nat_dec_lt(v___x_1056_, v___x_1057_);
if (v___x_1059_ == 0)
{
lean_dec_ref(v___x_1055_);
return v___x_1058_;
}
else
{
uint8_t v___x_1060_; 
v___x_1060_ = lean_nat_dec_le(v___x_1057_, v___x_1057_);
if (v___x_1060_ == 0)
{
if (v___x_1059_ == 0)
{
lean_dec_ref(v___x_1055_);
return v___x_1058_;
}
else
{
size_t v___x_1061_; size_t v___x_1062_; lean_object* v___x_1063_; 
v___x_1061_ = ((size_t)0ULL);
v___x_1062_ = lean_usize_of_nat(v___x_1057_);
v___x_1063_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_unwanted__cdot_spec__0(v___x_1055_, v___x_1061_, v___x_1062_, v___x_1058_);
lean_dec_ref(v___x_1055_);
return v___x_1063_;
}
}
else
{
size_t v___x_1064_; size_t v___x_1065_; lean_object* v___x_1066_; 
v___x_1064_ = ((size_t)0ULL);
v___x_1065_ = lean_usize_of_nat(v___x_1057_);
v___x_1066_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_unwanted__cdot_spec__0(v___x_1055_, v___x_1064_, v___x_1065_, v___x_1058_);
lean_dec_ref(v___x_1055_);
return v___x_1066_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__1(lean_object* v_msg_1067_){
_start:
{
lean_object* v___x_1068_; lean_object* v___x_1069_; 
v___x_1068_ = l_String_instInhabitedSlice;
v___x_1069_ = lean_panic_fn_borrowed(v___x_1068_, v_msg_1067_);
return v___x_1069_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__0___closed__1(void){
_start:
{
lean_object* v___x_1071_; lean_object* v___x_1072_; 
v___x_1071_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__0___closed__0));
v___x_1072_ = l_Lean_stringToMessageData(v___x_1071_);
return v___x_1072_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__0(lean_object* v_as_1073_, size_t v_sz_1074_, size_t v_i_1075_, lean_object* v_b_1076_, lean_object* v___y_1077_, lean_object* v___y_1078_){
_start:
{
uint8_t v___x_1080_; 
v___x_1080_ = lean_usize_dec_lt(v_i_1075_, v_sz_1074_);
if (v___x_1080_ == 0)
{
lean_object* v___x_1081_; 
v___x_1081_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1081_, 0, v_b_1076_);
return v___x_1081_;
}
else
{
lean_object* v___x_1082_; lean_object* v_a_1083_; lean_object* v___x_1084_; lean_object* v___x_1085_; 
v___x_1082_ = lp_mathlib_Mathlib_Linter_linter_style_cdot;
v_a_1083_ = lean_array_uget_borrowed(v_as_1073_, v_i_1075_);
v___x_1084_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__0___closed__1, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__0___closed__1_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__0___closed__1);
lean_inc(v_a_1083_);
v___x_1085_ = lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__1(v___x_1082_, v_a_1083_, v___x_1084_, v___y_1077_, v___y_1078_);
if (lean_obj_tag(v___x_1085_) == 0)
{
lean_object* v___x_1086_; size_t v___x_1087_; size_t v___x_1088_; 
lean_dec_ref_known(v___x_1085_, 1);
v___x_1086_ = lean_box(0);
v___x_1087_ = ((size_t)1ULL);
v___x_1088_ = lean_usize_add(v_i_1075_, v___x_1087_);
v_i_1075_ = v___x_1088_;
v_b_1076_ = v___x_1086_;
goto _start;
}
else
{
return v___x_1085_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__0___boxed(lean_object* v_as_1090_, lean_object* v_sz_1091_, lean_object* v_i_1092_, lean_object* v_b_1093_, lean_object* v___y_1094_, lean_object* v___y_1095_, lean_object* v___y_1096_){
_start:
{
size_t v_sz_boxed_1097_; size_t v_i_boxed_1098_; lean_object* v_res_1099_; 
v_sz_boxed_1097_ = lean_unbox_usize(v_sz_1091_);
lean_dec(v_sz_1091_);
v_i_boxed_1098_ = lean_unbox_usize(v_i_1092_);
lean_dec(v_i_1092_);
v_res_1099_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__0(v_as_1090_, v_sz_boxed_1097_, v_i_boxed_1098_, v_b_1093_, v___y_1094_, v___y_1095_);
lean_dec(v___y_1095_);
lean_dec_ref(v___y_1094_);
lean_dec_ref(v_as_1090_);
return v_res_1099_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Substring_Raw_takeWhileAux___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__3(lean_object* v_s_1100_, lean_object* v_stopPos_1101_, lean_object* v_i_1102_){
_start:
{
uint8_t v___y_1107_; uint8_t v___x_1108_; 
v___x_1108_ = lean_nat_dec_lt(v_i_1102_, v_stopPos_1101_);
if (v___x_1108_ == 0)
{
return v_i_1102_;
}
else
{
uint32_t v___x_1109_; uint8_t v___y_1111_; uint32_t v___x_1116_; uint8_t v___x_1117_; 
v___x_1109_ = lean_string_utf8_get(v_s_1100_, v_i_1102_);
v___x_1116_ = 32;
v___x_1117_ = lean_uint32_dec_eq(v___x_1109_, v___x_1116_);
if (v___x_1117_ == 0)
{
uint32_t v___x_1118_; uint8_t v___x_1119_; 
v___x_1118_ = 9;
v___x_1119_ = lean_uint32_dec_eq(v___x_1109_, v___x_1118_);
v___y_1111_ = v___x_1119_;
goto v___jp_1110_;
}
else
{
v___y_1111_ = v___x_1117_;
goto v___jp_1110_;
}
v___jp_1110_:
{
if (v___y_1111_ == 0)
{
uint32_t v___x_1112_; uint8_t v___x_1113_; 
v___x_1112_ = 13;
v___x_1113_ = lean_uint32_dec_eq(v___x_1109_, v___x_1112_);
if (v___x_1113_ == 0)
{
uint32_t v___x_1114_; uint8_t v___x_1115_; 
v___x_1114_ = 10;
v___x_1115_ = lean_uint32_dec_eq(v___x_1109_, v___x_1114_);
v___y_1107_ = v___x_1115_;
goto v___jp_1106_;
}
else
{
v___y_1107_ = v___x_1113_;
goto v___jp_1106_;
}
}
else
{
goto v___jp_1103_;
}
}
}
v___jp_1103_:
{
lean_object* v___x_1104_; 
v___x_1104_ = lean_string_utf8_next(v_s_1100_, v_i_1102_);
lean_dec(v_i_1102_);
v_i_1102_ = v___x_1104_;
goto _start;
}
v___jp_1106_:
{
if (v___y_1107_ == 0)
{
return v_i_1102_;
}
else
{
goto v___jp_1103_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Substring_Raw_takeWhileAux___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__3___boxed(lean_object* v_s_1120_, lean_object* v_stopPos_1121_, lean_object* v_i_1122_){
_start:
{
lean_object* v_res_1123_; 
v_res_1123_ = lp_mathlib_Substring_Raw_takeWhileAux___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__3(v_s_1120_, v_stopPos_1121_, v_i_1122_);
lean_dec(v_stopPos_1121_);
lean_dec_ref(v_s_1120_);
return v_res_1123_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__2_spec__2___redArg(lean_object* v_s_1124_, lean_object* v_a_1125_, uint8_t v_b_1126_){
_start:
{
lean_object* v_str_1127_; lean_object* v_startInclusive_1128_; lean_object* v_endExclusive_1129_; lean_object* v___x_1130_; uint8_t v___x_1131_; 
v_str_1127_ = lean_ctor_get(v_s_1124_, 0);
v_startInclusive_1128_ = lean_ctor_get(v_s_1124_, 1);
v_endExclusive_1129_ = lean_ctor_get(v_s_1124_, 2);
v___x_1130_ = lean_nat_sub(v_endExclusive_1129_, v_startInclusive_1128_);
v___x_1131_ = lean_nat_dec_eq(v_a_1125_, v___x_1130_);
lean_dec(v___x_1130_);
if (v___x_1131_ == 0)
{
uint32_t v___x_1132_; lean_object* v___x_1133_; uint32_t v___x_1134_; uint8_t v___x_1135_; 
v___x_1132_ = 10;
v___x_1133_ = lean_nat_add(v_startInclusive_1128_, v_a_1125_);
lean_dec(v_a_1125_);
v___x_1134_ = lean_string_utf8_get_fast(v_str_1127_, v___x_1133_);
v___x_1135_ = lean_uint32_dec_eq(v___x_1134_, v___x_1132_);
if (v___x_1135_ == 0)
{
lean_object* v___x_1136_; lean_object* v___x_1137_; 
v___x_1136_ = lean_string_utf8_next_fast(v_str_1127_, v___x_1133_);
lean_dec(v___x_1133_);
v___x_1137_ = lean_nat_sub(v___x_1136_, v_startInclusive_1128_);
v_a_1125_ = v___x_1137_;
v_b_1126_ = v___x_1135_;
goto _start;
}
else
{
lean_dec(v___x_1133_);
return v___x_1135_;
}
}
else
{
lean_dec(v_a_1125_);
return v_b_1126_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__2_spec__2___redArg___boxed(lean_object* v_s_1139_, lean_object* v_a_1140_, lean_object* v_b_1141_){
_start:
{
uint8_t v_b_boxed_1142_; uint8_t v_res_1143_; lean_object* v_r_1144_; 
v_b_boxed_1142_ = lean_unbox(v_b_1141_);
v_res_1143_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__2_spec__2___redArg(v_s_1139_, v_a_1140_, v_b_boxed_1142_);
lean_dec_ref(v_s_1139_);
v_r_1144_ = lean_box(v_res_1143_);
return v_r_1144_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__2(lean_object* v_s_1145_){
_start:
{
lean_object* v_searcher_1146_; uint8_t v___x_1147_; uint8_t v___x_1148_; 
v_searcher_1146_ = lean_unsigned_to_nat(0u);
v___x_1147_ = 0;
v___x_1148_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__2_spec__2___redArg(v_s_1145_, v_searcher_1146_, v___x_1147_);
return v___x_1148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__2___boxed(lean_object* v_s_1149_){
_start:
{
uint8_t v_res_1150_; lean_object* v_r_1151_; 
v_res_1150_ = lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__2(v_s_1149_);
lean_dec_ref(v_s_1149_);
v_r_1151_ = lean_box(v_res_1150_);
return v_r_1151_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___closed__3(void){
_start:
{
lean_object* v___x_1159_; lean_object* v___x_1160_; 
v___x_1159_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___closed__2));
v___x_1160_ = l_Lean_stringToMessageData(v___x_1159_);
return v___x_1160_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___closed__7(void){
_start:
{
lean_object* v___x_1164_; lean_object* v___x_1165_; lean_object* v___x_1166_; lean_object* v___x_1167_; lean_object* v___x_1168_; lean_object* v___x_1169_; 
v___x_1164_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___closed__6));
v___x_1165_ = lean_unsigned_to_nat(14u);
v___x_1166_ = lean_unsigned_to_nat(22u);
v___x_1167_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___closed__5));
v___x_1168_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___closed__4));
v___x_1169_ = l_mkPanicMessageWithDecl(v___x_1168_, v___x_1167_, v___x_1166_, v___x_1165_, v___x_1164_);
return v___x_1169_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4(lean_object* v_as_1175_, size_t v_sz_1176_, size_t v_i_1177_, lean_object* v_b_1178_, lean_object* v___y_1179_, lean_object* v___y_1180_){
_start:
{
lean_object* v_a_1183_; uint8_t v___x_1187_; 
v___x_1187_ = lean_usize_dec_lt(v_i_1177_, v_sz_1176_);
if (v___x_1187_ == 0)
{
lean_object* v___x_1188_; 
v___x_1188_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1188_, 0, v_b_1178_);
return v___x_1188_;
}
else
{
lean_object* v___x_1189_; lean_object* v_a_1190_; lean_object* v___x_1191_; uint8_t v___x_1192_; 
lean_dec_ref(v_b_1178_);
v___x_1189_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___closed__0));
v_a_1190_ = lean_array_uget_borrowed(v_as_1175_, v_i_1177_);
v___x_1191_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___closed__1));
lean_inc(v_a_1190_);
v___x_1192_ = l_Lean_Syntax_isOfKind(v_a_1190_, v___x_1191_);
if (v___x_1192_ == 0)
{
v_a_1183_ = v___x_1189_;
goto v___jp_1182_;
}
else
{
lean_object* v___x_1193_; 
v___x_1193_ = l_Lean_Syntax_getTrailing_x3f(v_a_1190_);
if (lean_obj_tag(v___x_1193_) == 1)
{
lean_object* v_val_1194_; lean_object* v_str_1195_; lean_object* v_startPos_1196_; lean_object* v_stopPos_1197_; lean_object* v___x_1199_; uint8_t v_isShared_1200_; uint8_t v_isSharedCheck_1226_; 
v_val_1194_ = lean_ctor_get(v___x_1193_, 0);
lean_inc(v_val_1194_);
lean_dec_ref_known(v___x_1193_, 1);
v_str_1195_ = lean_ctor_get(v_val_1194_, 0);
v_startPos_1196_ = lean_ctor_get(v_val_1194_, 1);
v_stopPos_1197_ = lean_ctor_get(v_val_1194_, 2);
v_isSharedCheck_1226_ = !lean_is_exclusive(v_val_1194_);
if (v_isSharedCheck_1226_ == 0)
{
v___x_1199_ = v_val_1194_;
v_isShared_1200_ = v_isSharedCheck_1226_;
goto v_resetjp_1198_;
}
else
{
lean_inc(v_stopPos_1197_);
lean_inc(v_startPos_1196_);
lean_inc(v_str_1195_);
lean_dec(v_val_1194_);
v___x_1199_ = lean_box(0);
v_isShared_1200_ = v_isSharedCheck_1226_;
goto v_resetjp_1198_;
}
v_resetjp_1198_:
{
lean_object* v___x_1201_; uint8_t v___y_1203_; uint8_t v___x_1218_; 
v___x_1201_ = lp_mathlib_Mathlib_Linter_linter_style_cdot;
v___x_1218_ = lean_string_is_valid_pos(v_str_1195_, v_startPos_1196_);
if (v___x_1218_ == 0)
{
lean_del_object(v___x_1199_);
lean_dec(v_stopPos_1197_);
lean_dec(v_startPos_1196_);
lean_dec_ref(v_str_1195_);
goto v___jp_1214_;
}
else
{
lean_object* v_e_1219_; uint8_t v___x_1220_; 
lean_inc(v_startPos_1196_);
v_e_1219_ = lp_mathlib_Substring_Raw_takeWhileAux___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__3(v_str_1195_, v_stopPos_1197_, v_startPos_1196_);
lean_dec(v_stopPos_1197_);
v___x_1220_ = lean_string_is_valid_pos(v_str_1195_, v_e_1219_);
if (v___x_1220_ == 0)
{
lean_dec(v_e_1219_);
lean_del_object(v___x_1199_);
lean_dec(v_startPos_1196_);
lean_dec_ref(v_str_1195_);
goto v___jp_1214_;
}
else
{
uint8_t v___x_1221_; 
v___x_1221_ = lean_nat_dec_le(v_startPos_1196_, v_e_1219_);
if (v___x_1221_ == 0)
{
lean_dec(v_e_1219_);
lean_del_object(v___x_1199_);
lean_dec(v_startPos_1196_);
lean_dec_ref(v_str_1195_);
goto v___jp_1214_;
}
else
{
lean_object* v___x_1223_; 
if (v_isShared_1200_ == 0)
{
lean_ctor_set(v___x_1199_, 2, v_e_1219_);
v___x_1223_ = v___x_1199_;
goto v_reusejp_1222_;
}
else
{
lean_object* v_reuseFailAlloc_1225_; 
v_reuseFailAlloc_1225_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1225_, 0, v_str_1195_);
lean_ctor_set(v_reuseFailAlloc_1225_, 1, v_startPos_1196_);
lean_ctor_set(v_reuseFailAlloc_1225_, 2, v_e_1219_);
v___x_1223_ = v_reuseFailAlloc_1225_;
goto v_reusejp_1222_;
}
v_reusejp_1222_:
{
uint8_t v___x_1224_; 
v___x_1224_ = lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__2(v___x_1223_);
lean_dec_ref(v___x_1223_);
v___y_1203_ = v___x_1224_;
goto v___jp_1202_;
}
}
}
}
v___jp_1202_:
{
if (v___y_1203_ == 0)
{
v_a_1183_ = v___x_1189_;
goto v___jp_1182_;
}
else
{
lean_object* v___x_1204_; lean_object* v___x_1205_; 
v___x_1204_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___closed__3, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___closed__3_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___closed__3);
lean_inc(v_a_1190_);
v___x_1205_ = lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__1(v___x_1201_, v_a_1190_, v___x_1204_, v___y_1179_, v___y_1180_);
if (lean_obj_tag(v___x_1205_) == 0)
{
lean_dec_ref_known(v___x_1205_, 1);
v_a_1183_ = v___x_1189_;
goto v___jp_1182_;
}
else
{
lean_object* v_a_1206_; lean_object* v___x_1208_; uint8_t v_isShared_1209_; uint8_t v_isSharedCheck_1213_; 
v_a_1206_ = lean_ctor_get(v___x_1205_, 0);
v_isSharedCheck_1213_ = !lean_is_exclusive(v___x_1205_);
if (v_isSharedCheck_1213_ == 0)
{
v___x_1208_ = v___x_1205_;
v_isShared_1209_ = v_isSharedCheck_1213_;
goto v_resetjp_1207_;
}
else
{
lean_inc(v_a_1206_);
lean_dec(v___x_1205_);
v___x_1208_ = lean_box(0);
v_isShared_1209_ = v_isSharedCheck_1213_;
goto v_resetjp_1207_;
}
v_resetjp_1207_:
{
lean_object* v___x_1211_; 
if (v_isShared_1209_ == 0)
{
v___x_1211_ = v___x_1208_;
goto v_reusejp_1210_;
}
else
{
lean_object* v_reuseFailAlloc_1212_; 
v_reuseFailAlloc_1212_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1212_, 0, v_a_1206_);
v___x_1211_ = v_reuseFailAlloc_1212_;
goto v_reusejp_1210_;
}
v_reusejp_1210_:
{
return v___x_1211_;
}
}
}
}
}
v___jp_1214_:
{
lean_object* v___x_1215_; lean_object* v___x_1216_; uint8_t v___x_1217_; 
v___x_1215_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___closed__7, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___closed__7_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___closed__7);
v___x_1216_ = lp_mathlib_panic___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__1(v___x_1215_);
v___x_1217_ = lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__2(v___x_1216_);
lean_dec_ref(v___x_1216_);
v___y_1203_ = v___x_1217_;
goto v___jp_1202_;
}
}
}
else
{
lean_object* v___x_1227_; lean_object* v___x_1228_; 
lean_dec(v___x_1193_);
v___x_1227_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___closed__9));
v___x_1228_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1228_, 0, v___x_1227_);
return v___x_1228_;
}
}
}
v___jp_1182_:
{
size_t v___x_1184_; size_t v___x_1185_; 
v___x_1184_ = ((size_t)1ULL);
v___x_1185_ = lean_usize_add(v_i_1177_, v___x_1184_);
lean_inc_ref(v_a_1183_);
v_i_1177_ = v___x_1185_;
v_b_1178_ = v_a_1183_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___boxed(lean_object* v_as_1229_, lean_object* v_sz_1230_, lean_object* v_i_1231_, lean_object* v_b_1232_, lean_object* v___y_1233_, lean_object* v___y_1234_, lean_object* v___y_1235_){
_start:
{
size_t v_sz_boxed_1236_; size_t v_i_boxed_1237_; lean_object* v_res_1238_; 
v_sz_boxed_1236_ = lean_unbox_usize(v_sz_1230_);
lean_dec(v_sz_1230_);
v_i_boxed_1237_ = lean_unbox_usize(v_i_1231_);
lean_dec(v_i_1231_);
v_res_1238_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4(v_as_1229_, v_sz_boxed_1236_, v_i_boxed_1237_, v_b_1232_, v___y_1233_, v___y_1234_);
lean_dec(v___y_1234_);
lean_dec_ref(v___y_1233_);
lean_dec_ref(v_as_1229_);
return v_res_1238_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter___lam__0(lean_object* v_stx_1239_, lean_object* v___y_1240_, lean_object* v___y_1241_){
_start:
{
lean_object* v___x_1243_; lean_object* v_a_1244_; lean_object* v___x_1246_; uint8_t v_isShared_1247_; uint8_t v_isSharedCheck_1291_; 
v___x_1243_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__0(v___y_1240_, v___y_1241_);
v_a_1244_ = lean_ctor_get(v___x_1243_, 0);
v_isSharedCheck_1291_ = !lean_is_exclusive(v___x_1243_);
if (v_isSharedCheck_1291_ == 0)
{
v___x_1246_ = v___x_1243_;
v_isShared_1247_ = v_isSharedCheck_1291_;
goto v_resetjp_1245_;
}
else
{
lean_inc(v_a_1244_);
lean_dec(v___x_1243_);
v___x_1246_ = lean_box(0);
v_isShared_1247_ = v_isSharedCheck_1291_;
goto v_resetjp_1245_;
}
v_resetjp_1245_:
{
lean_object* v___x_1248_; uint8_t v___x_1249_; 
v___x_1248_ = lp_mathlib_Mathlib_Linter_linter_style_cdot;
v___x_1249_ = l_Lean_Linter_getLinterValue(v___x_1248_, v_a_1244_);
lean_dec(v_a_1244_);
if (v___x_1249_ == 0)
{
lean_object* v___x_1250_; lean_object* v___x_1252_; 
lean_dec(v_stx_1239_);
v___x_1250_ = lean_box(0);
if (v_isShared_1247_ == 0)
{
lean_ctor_set(v___x_1246_, 0, v___x_1250_);
v___x_1252_ = v___x_1246_;
goto v_reusejp_1251_;
}
else
{
lean_object* v_reuseFailAlloc_1253_; 
v_reuseFailAlloc_1253_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1253_, 0, v___x_1250_);
v___x_1252_ = v_reuseFailAlloc_1253_;
goto v_reusejp_1251_;
}
v_reusejp_1251_:
{
return v___x_1252_;
}
}
else
{
lean_object* v___x_1254_; lean_object* v_messages_1255_; uint8_t v___x_1256_; 
v___x_1254_ = lean_st_ref_get(v___y_1241_);
v_messages_1255_ = lean_ctor_get(v___x_1254_, 1);
lean_inc_ref(v_messages_1255_);
lean_dec(v___x_1254_);
v___x_1256_ = l_Lean_MessageLog_hasErrors(v_messages_1255_);
lean_dec_ref(v_messages_1255_);
if (v___x_1256_ == 0)
{
lean_object* v___x_1257_; lean_object* v___x_1258_; size_t v_sz_1259_; size_t v___x_1260_; lean_object* v___x_1261_; 
lean_del_object(v___x_1246_);
lean_inc(v_stx_1239_);
v___x_1257_ = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_unwanted__cdot(v_stx_1239_);
v___x_1258_ = lean_box(0);
v_sz_1259_ = lean_array_size(v___x_1257_);
v___x_1260_ = ((size_t)0ULL);
v___x_1261_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__0(v___x_1257_, v_sz_1259_, v___x_1260_, v___x_1258_, v___y_1240_, v___y_1241_);
lean_dec_ref(v___x_1257_);
if (lean_obj_tag(v___x_1261_) == 0)
{
lean_object* v___x_1262_; lean_object* v___x_1263_; size_t v_sz_1264_; lean_object* v___x_1265_; 
lean_dec_ref_known(v___x_1261_, 1);
v___x_1262_ = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_findCDot(v_stx_1239_);
v___x_1263_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___closed__0));
v_sz_1264_ = lean_array_size(v___x_1262_);
v___x_1265_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4(v___x_1262_, v_sz_1264_, v___x_1260_, v___x_1263_, v___y_1240_, v___y_1241_);
lean_dec_ref(v___x_1262_);
if (lean_obj_tag(v___x_1265_) == 0)
{
lean_object* v_a_1266_; lean_object* v___x_1268_; uint8_t v_isShared_1269_; uint8_t v_isSharedCheck_1278_; 
v_a_1266_ = lean_ctor_get(v___x_1265_, 0);
v_isSharedCheck_1278_ = !lean_is_exclusive(v___x_1265_);
if (v_isSharedCheck_1278_ == 0)
{
v___x_1268_ = v___x_1265_;
v_isShared_1269_ = v_isSharedCheck_1278_;
goto v_resetjp_1267_;
}
else
{
lean_inc(v_a_1266_);
lean_dec(v___x_1265_);
v___x_1268_ = lean_box(0);
v_isShared_1269_ = v_isSharedCheck_1278_;
goto v_resetjp_1267_;
}
v_resetjp_1267_:
{
lean_object* v_fst_1270_; 
v_fst_1270_ = lean_ctor_get(v_a_1266_, 0);
lean_inc(v_fst_1270_);
lean_dec(v_a_1266_);
if (lean_obj_tag(v_fst_1270_) == 0)
{
lean_object* v___x_1272_; 
if (v_isShared_1269_ == 0)
{
lean_ctor_set(v___x_1268_, 0, v___x_1258_);
v___x_1272_ = v___x_1268_;
goto v_reusejp_1271_;
}
else
{
lean_object* v_reuseFailAlloc_1273_; 
v_reuseFailAlloc_1273_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1273_, 0, v___x_1258_);
v___x_1272_ = v_reuseFailAlloc_1273_;
goto v_reusejp_1271_;
}
v_reusejp_1271_:
{
return v___x_1272_;
}
}
else
{
lean_object* v_val_1274_; lean_object* v___x_1276_; 
v_val_1274_ = lean_ctor_get(v_fst_1270_, 0);
lean_inc(v_val_1274_);
lean_dec_ref_known(v_fst_1270_, 1);
if (v_isShared_1269_ == 0)
{
lean_ctor_set(v___x_1268_, 0, v_val_1274_);
v___x_1276_ = v___x_1268_;
goto v_reusejp_1275_;
}
else
{
lean_object* v_reuseFailAlloc_1277_; 
v_reuseFailAlloc_1277_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1277_, 0, v_val_1274_);
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
else
{
lean_object* v_a_1279_; lean_object* v___x_1281_; uint8_t v_isShared_1282_; uint8_t v_isSharedCheck_1286_; 
v_a_1279_ = lean_ctor_get(v___x_1265_, 0);
v_isSharedCheck_1286_ = !lean_is_exclusive(v___x_1265_);
if (v_isSharedCheck_1286_ == 0)
{
v___x_1281_ = v___x_1265_;
v_isShared_1282_ = v_isSharedCheck_1286_;
goto v_resetjp_1280_;
}
else
{
lean_inc(v_a_1279_);
lean_dec(v___x_1265_);
v___x_1281_ = lean_box(0);
v_isShared_1282_ = v_isSharedCheck_1286_;
goto v_resetjp_1280_;
}
v_resetjp_1280_:
{
lean_object* v___x_1284_; 
if (v_isShared_1282_ == 0)
{
v___x_1284_ = v___x_1281_;
goto v_reusejp_1283_;
}
else
{
lean_object* v_reuseFailAlloc_1285_; 
v_reuseFailAlloc_1285_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1285_, 0, v_a_1279_);
v___x_1284_ = v_reuseFailAlloc_1285_;
goto v_reusejp_1283_;
}
v_reusejp_1283_:
{
return v___x_1284_;
}
}
}
}
else
{
lean_dec(v_stx_1239_);
return v___x_1261_;
}
}
else
{
lean_object* v___x_1287_; lean_object* v___x_1289_; 
lean_dec(v_stx_1239_);
v___x_1287_ = lean_box(0);
if (v_isShared_1247_ == 0)
{
lean_ctor_set(v___x_1246_, 0, v___x_1287_);
v___x_1289_ = v___x_1246_;
goto v_reusejp_1288_;
}
else
{
lean_object* v_reuseFailAlloc_1290_; 
v_reuseFailAlloc_1290_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1290_, 0, v___x_1287_);
v___x_1289_ = v_reuseFailAlloc_1290_;
goto v_reusejp_1288_;
}
v_reusejp_1288_:
{
return v___x_1289_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter___lam__0___boxed(lean_object* v_stx_1292_, lean_object* v___y_1293_, lean_object* v___y_1294_, lean_object* v___y_1295_){
_start:
{
lean_object* v_res_1296_; 
v_res_1296_ = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter___lam__0(v_stx_1292_, v___y_1293_, v___y_1294_);
lean_dec(v___y_1294_);
lean_dec_ref(v___y_1293_);
return v_res_1296_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__2_spec__2(lean_object* v_s_1308_, lean_object* v_inst_1309_, lean_object* v_R_1310_, lean_object* v_a_1311_, uint8_t v_b_1312_, lean_object* v_c_1313_){
_start:
{
uint8_t v___x_1314_; 
v___x_1314_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__2_spec__2___redArg(v_s_1308_, v_a_1311_, v_b_1312_);
return v___x_1314_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__2_spec__2___boxed(lean_object* v_s_1315_, lean_object* v_inst_1316_, lean_object* v_R_1317_, lean_object* v_a_1318_, lean_object* v_b_1319_, lean_object* v_c_1320_){
_start:
{
uint8_t v_b_boxed_1321_; uint8_t v_res_1322_; lean_object* v_r_1323_; 
v_b_boxed_1321_ = lean_unbox(v_b_1319_);
v_res_1322_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__2_spec__2(v_s_1315_, v_inst_1316_, v_R_1317_, v_a_1318_, v_b_boxed_1321_, v_c_1320_);
lean_dec_ref(v_s_1315_);
v_r_1323_ = lean_box(v_res_1322_);
return v_r_1323_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_Style_1823831825____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_1325_; lean_object* v___x_1326_; 
v___x_1325_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter));
v___x_1326_ = l_Lean_Elab_Command_addLinter(v___x_1325_);
return v___x_1326_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_Style_1823831825____hygCtx___hyg_2____boxed(lean_object* v_a_1327_){
_start:
{
lean_object* v_res_1328_; 
v_res_1328_ = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_Style_1823831825____hygCtx___hyg_2_();
return v_res_1328_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_830885783____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_1347_; lean_object* v___x_1348_; lean_object* v___x_1349_; lean_object* v___x_1350_; 
v___x_1347_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_830885783____hygCtx___hyg_4_));
v___x_1348_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Style_830885783____hygCtx___hyg_4_));
v___x_1349_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_830885783____hygCtx___hyg_4_));
v___x_1350_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__spec__0(v___x_1347_, v___x_1348_, v___x_1349_);
return v___x_1350_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_830885783____hygCtx___hyg_4____boxed(lean_object* v_a_1351_){
_start:
{
lean_object* v_res_1352_; 
v_res_1352_ = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_830885783____hygCtx___hyg_4_();
return v_res_1352_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_Style_dollarSyntax_findDollarSyntax(lean_object* v_x_1354_){
_start:
{
if (lean_obj_tag(v_x_1354_) == 1)
{
lean_object* v_kind_1355_; lean_object* v_args_1356_; lean_object* v___y_1358_; size_t v_sz_1364_; size_t v___x_1365_; lean_object* v___x_1366_; lean_object* v___x_1367_; lean_object* v___x_1368_; lean_object* v___x_1369_; uint8_t v___x_1370_; 
v_kind_1355_ = lean_ctor_get(v_x_1354_, 1);
v_args_1356_ = lean_ctor_get(v_x_1354_, 2);
v_sz_1364_ = lean_array_size(v_args_1356_);
v___x_1365_ = ((size_t)0ULL);
lean_inc_ref(v_args_1356_);
v___x_1366_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Linter_Style_dollarSyntax_findDollarSyntax_spec__0(v_sz_1364_, v___x_1365_, v_args_1356_);
v___x_1367_ = lean_unsigned_to_nat(0u);
v___x_1368_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_findCDot___closed__0));
v___x_1369_ = lean_array_get_size(v___x_1366_);
v___x_1370_ = lean_nat_dec_lt(v___x_1367_, v___x_1369_);
if (v___x_1370_ == 0)
{
lean_dec_ref(v___x_1366_);
v___y_1358_ = v___x_1368_;
goto v___jp_1357_;
}
else
{
uint8_t v___x_1371_; 
v___x_1371_ = lean_nat_dec_le(v___x_1369_, v___x_1369_);
if (v___x_1371_ == 0)
{
if (v___x_1370_ == 0)
{
lean_dec_ref(v___x_1366_);
v___y_1358_ = v___x_1368_;
goto v___jp_1357_;
}
else
{
size_t v___x_1372_; lean_object* v___x_1373_; 
v___x_1372_ = lean_usize_of_nat(v___x_1369_);
v___x_1373_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_findCDot_spec__1(v___x_1366_, v___x_1365_, v___x_1372_, v___x_1368_);
lean_dec_ref(v___x_1366_);
v___y_1358_ = v___x_1373_;
goto v___jp_1357_;
}
}
else
{
size_t v___x_1374_; lean_object* v___x_1375_; 
v___x_1374_ = lean_usize_of_nat(v___x_1369_);
v___x_1375_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_findCDot_spec__1(v___x_1366_, v___x_1365_, v___x_1374_, v___x_1368_);
lean_dec_ref(v___x_1366_);
v___y_1358_ = v___x_1375_;
goto v___jp_1357_;
}
}
v___jp_1357_:
{
if (lean_obj_tag(v_kind_1355_) == 1)
{
lean_object* v_pre_1359_; 
v_pre_1359_ = lean_ctor_get(v_kind_1355_, 0);
if (lean_obj_tag(v_pre_1359_) == 0)
{
lean_object* v_str_1360_; lean_object* v___x_1361_; uint8_t v___x_1362_; 
v_str_1360_ = lean_ctor_get(v_kind_1355_, 1);
v___x_1361_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_Style_dollarSyntax_findDollarSyntax___closed__0));
v___x_1362_ = lean_string_dec_eq(v_str_1360_, v___x_1361_);
if (v___x_1362_ == 0)
{
lean_dec_ref_known(v_x_1354_, 3);
return v___y_1358_;
}
else
{
lean_object* v___x_1363_; 
v___x_1363_ = lean_array_push(v___y_1358_, v_x_1354_);
return v___x_1363_;
}
}
else
{
lean_dec_ref_known(v_x_1354_, 3);
return v___y_1358_;
}
}
else
{
lean_dec_ref_known(v_x_1354_, 3);
return v___y_1358_;
}
}
}
else
{
lean_object* v___x_1376_; 
lean_dec(v_x_1354_);
v___x_1376_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_findCDot___closed__1));
return v___x_1376_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Linter_Style_dollarSyntax_findDollarSyntax_spec__0(size_t v_sz_1377_, size_t v_i_1378_, lean_object* v_bs_1379_){
_start:
{
uint8_t v___x_1380_; 
v___x_1380_ = lean_usize_dec_lt(v_i_1378_, v_sz_1377_);
if (v___x_1380_ == 0)
{
return v_bs_1379_;
}
else
{
lean_object* v_v_1381_; lean_object* v___x_1382_; lean_object* v_bs_x27_1383_; lean_object* v___x_1384_; size_t v___x_1385_; size_t v___x_1386_; lean_object* v___x_1387_; 
v_v_1381_ = lean_array_uget(v_bs_1379_, v_i_1378_);
v___x_1382_ = lean_unsigned_to_nat(0u);
v_bs_x27_1383_ = lean_array_uset(v_bs_1379_, v_i_1378_, v___x_1382_);
v___x_1384_ = lp_mathlib_Mathlib_Linter_Style_dollarSyntax_findDollarSyntax(v_v_1381_);
v___x_1385_ = ((size_t)1ULL);
v___x_1386_ = lean_usize_add(v_i_1378_, v___x_1385_);
v___x_1387_ = lean_array_uset(v_bs_x27_1383_, v_i_1378_, v___x_1384_);
v_i_1378_ = v___x_1386_;
v_bs_1379_ = v___x_1387_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Linter_Style_dollarSyntax_findDollarSyntax_spec__0___boxed(lean_object* v_sz_1389_, lean_object* v_i_1390_, lean_object* v_bs_1391_){
_start:
{
size_t v_sz_boxed_1392_; size_t v_i_boxed_1393_; lean_object* v_res_1394_; 
v_sz_boxed_1392_ = lean_unbox_usize(v_sz_1389_);
lean_dec(v_sz_1389_);
v_i_boxed_1393_ = lean_unbox_usize(v_i_1390_);
lean_dec(v_i_1390_);
v_res_1394_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Linter_Style_dollarSyntax_findDollarSyntax_spec__0(v_sz_boxed_1392_, v_i_boxed_1393_, v_bs_1391_);
return v_res_1394_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_dollarSyntax_dollarSyntaxLinter_spec__0___closed__1(void){
_start:
{
lean_object* v___x_1396_; lean_object* v___x_1397_; 
v___x_1396_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_dollarSyntax_dollarSyntaxLinter_spec__0___closed__0));
v___x_1397_ = l_Lean_stringToMessageData(v___x_1396_);
return v___x_1397_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_dollarSyntax_dollarSyntaxLinter_spec__0(lean_object* v_as_1398_, size_t v_sz_1399_, size_t v_i_1400_, lean_object* v_b_1401_, lean_object* v___y_1402_, lean_object* v___y_1403_){
_start:
{
uint8_t v___x_1405_; 
v___x_1405_ = lean_usize_dec_lt(v_i_1400_, v_sz_1399_);
if (v___x_1405_ == 0)
{
lean_object* v___x_1406_; 
v___x_1406_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1406_, 0, v_b_1401_);
return v___x_1406_;
}
else
{
lean_object* v___x_1407_; lean_object* v_a_1408_; lean_object* v___x_1409_; lean_object* v___x_1410_; 
v___x_1407_ = lp_mathlib_Mathlib_Linter_linter_style_dollarSyntax;
v_a_1408_ = lean_array_uget_borrowed(v_as_1398_, v_i_1400_);
v___x_1409_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_dollarSyntax_dollarSyntaxLinter_spec__0___closed__1, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_dollarSyntax_dollarSyntaxLinter_spec__0___closed__1_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_dollarSyntax_dollarSyntaxLinter_spec__0___closed__1);
lean_inc(v_a_1408_);
v___x_1410_ = lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__1(v___x_1407_, v_a_1408_, v___x_1409_, v___y_1402_, v___y_1403_);
if (lean_obj_tag(v___x_1410_) == 0)
{
lean_object* v___x_1411_; size_t v___x_1412_; size_t v___x_1413_; 
lean_dec_ref_known(v___x_1410_, 1);
v___x_1411_ = lean_box(0);
v___x_1412_ = ((size_t)1ULL);
v___x_1413_ = lean_usize_add(v_i_1400_, v___x_1412_);
v_i_1400_ = v___x_1413_;
v_b_1401_ = v___x_1411_;
goto _start;
}
else
{
return v___x_1410_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_dollarSyntax_dollarSyntaxLinter_spec__0___boxed(lean_object* v_as_1415_, lean_object* v_sz_1416_, lean_object* v_i_1417_, lean_object* v_b_1418_, lean_object* v___y_1419_, lean_object* v___y_1420_, lean_object* v___y_1421_){
_start:
{
size_t v_sz_boxed_1422_; size_t v_i_boxed_1423_; lean_object* v_res_1424_; 
v_sz_boxed_1422_ = lean_unbox_usize(v_sz_1416_);
lean_dec(v_sz_1416_);
v_i_boxed_1423_ = lean_unbox_usize(v_i_1417_);
lean_dec(v_i_1417_);
v_res_1424_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_dollarSyntax_dollarSyntaxLinter_spec__0(v_as_1415_, v_sz_boxed_1422_, v_i_boxed_1423_, v_b_1418_, v___y_1419_, v___y_1420_);
lean_dec(v___y_1420_);
lean_dec_ref(v___y_1419_);
lean_dec_ref(v_as_1415_);
return v_res_1424_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_dollarSyntax_dollarSyntaxLinter___lam__0(lean_object* v_stx_1425_, lean_object* v___y_1426_, lean_object* v___y_1427_){
_start:
{
lean_object* v___x_1429_; lean_object* v_a_1430_; lean_object* v___x_1432_; uint8_t v_isShared_1433_; uint8_t v_isSharedCheck_1460_; 
v___x_1429_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__0(v___y_1426_, v___y_1427_);
v_a_1430_ = lean_ctor_get(v___x_1429_, 0);
v_isSharedCheck_1460_ = !lean_is_exclusive(v___x_1429_);
if (v_isSharedCheck_1460_ == 0)
{
v___x_1432_ = v___x_1429_;
v_isShared_1433_ = v_isSharedCheck_1460_;
goto v_resetjp_1431_;
}
else
{
lean_inc(v_a_1430_);
lean_dec(v___x_1429_);
v___x_1432_ = lean_box(0);
v_isShared_1433_ = v_isSharedCheck_1460_;
goto v_resetjp_1431_;
}
v_resetjp_1431_:
{
lean_object* v___x_1434_; uint8_t v___x_1435_; 
v___x_1434_ = lp_mathlib_Mathlib_Linter_linter_style_dollarSyntax;
v___x_1435_ = l_Lean_Linter_getLinterValue(v___x_1434_, v_a_1430_);
lean_dec(v_a_1430_);
if (v___x_1435_ == 0)
{
lean_object* v___x_1436_; lean_object* v___x_1438_; 
lean_dec(v_stx_1425_);
v___x_1436_ = lean_box(0);
if (v_isShared_1433_ == 0)
{
lean_ctor_set(v___x_1432_, 0, v___x_1436_);
v___x_1438_ = v___x_1432_;
goto v_reusejp_1437_;
}
else
{
lean_object* v_reuseFailAlloc_1439_; 
v_reuseFailAlloc_1439_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1439_, 0, v___x_1436_);
v___x_1438_ = v_reuseFailAlloc_1439_;
goto v_reusejp_1437_;
}
v_reusejp_1437_:
{
return v___x_1438_;
}
}
else
{
lean_object* v___x_1440_; lean_object* v_messages_1441_; uint8_t v___x_1442_; 
v___x_1440_ = lean_st_ref_get(v___y_1427_);
v_messages_1441_ = lean_ctor_get(v___x_1440_, 1);
lean_inc_ref(v_messages_1441_);
lean_dec(v___x_1440_);
v___x_1442_ = l_Lean_MessageLog_hasErrors(v_messages_1441_);
lean_dec_ref(v_messages_1441_);
if (v___x_1442_ == 0)
{
lean_object* v___x_1443_; lean_object* v___x_1444_; size_t v_sz_1445_; size_t v___x_1446_; lean_object* v___x_1447_; 
lean_del_object(v___x_1432_);
v___x_1443_ = lp_mathlib_Mathlib_Linter_Style_dollarSyntax_findDollarSyntax(v_stx_1425_);
v___x_1444_ = lean_box(0);
v_sz_1445_ = lean_array_size(v___x_1443_);
v___x_1446_ = ((size_t)0ULL);
v___x_1447_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_dollarSyntax_dollarSyntaxLinter_spec__0(v___x_1443_, v_sz_1445_, v___x_1446_, v___x_1444_, v___y_1426_, v___y_1427_);
lean_dec_ref(v___x_1443_);
if (lean_obj_tag(v___x_1447_) == 0)
{
lean_object* v___x_1449_; uint8_t v_isShared_1450_; uint8_t v_isSharedCheck_1454_; 
v_isSharedCheck_1454_ = !lean_is_exclusive(v___x_1447_);
if (v_isSharedCheck_1454_ == 0)
{
lean_object* v_unused_1455_; 
v_unused_1455_ = lean_ctor_get(v___x_1447_, 0);
lean_dec(v_unused_1455_);
v___x_1449_ = v___x_1447_;
v_isShared_1450_ = v_isSharedCheck_1454_;
goto v_resetjp_1448_;
}
else
{
lean_dec(v___x_1447_);
v___x_1449_ = lean_box(0);
v_isShared_1450_ = v_isSharedCheck_1454_;
goto v_resetjp_1448_;
}
v_resetjp_1448_:
{
lean_object* v___x_1452_; 
if (v_isShared_1450_ == 0)
{
lean_ctor_set(v___x_1449_, 0, v___x_1444_);
v___x_1452_ = v___x_1449_;
goto v_reusejp_1451_;
}
else
{
lean_object* v_reuseFailAlloc_1453_; 
v_reuseFailAlloc_1453_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1453_, 0, v___x_1444_);
v___x_1452_ = v_reuseFailAlloc_1453_;
goto v_reusejp_1451_;
}
v_reusejp_1451_:
{
return v___x_1452_;
}
}
}
else
{
return v___x_1447_;
}
}
else
{
lean_object* v___x_1456_; lean_object* v___x_1458_; 
lean_dec(v_stx_1425_);
v___x_1456_ = lean_box(0);
if (v_isShared_1433_ == 0)
{
lean_ctor_set(v___x_1432_, 0, v___x_1456_);
v___x_1458_ = v___x_1432_;
goto v_reusejp_1457_;
}
else
{
lean_object* v_reuseFailAlloc_1459_; 
v_reuseFailAlloc_1459_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1459_, 0, v___x_1456_);
v___x_1458_ = v_reuseFailAlloc_1459_;
goto v_reusejp_1457_;
}
v_reusejp_1457_:
{
return v___x_1458_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_dollarSyntax_dollarSyntaxLinter___lam__0___boxed(lean_object* v_stx_1461_, lean_object* v___y_1462_, lean_object* v___y_1463_, lean_object* v___y_1464_){
_start:
{
lean_object* v_res_1465_; 
v_res_1465_ = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_dollarSyntax_dollarSyntaxLinter___lam__0(v_stx_1461_, v___y_1462_, v___y_1463_);
lean_dec(v___y_1463_);
lean_dec_ref(v___y_1462_);
return v_res_1465_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_dollarSyntax_initFn_00___x40_Mathlib_Tactic_Linter_Style_1303717341____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_1481_; lean_object* v___x_1482_; 
v___x_1481_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_dollarSyntax_dollarSyntaxLinter));
v___x_1482_ = l_Lean_Elab_Command_addLinter(v___x_1481_);
return v___x_1482_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_dollarSyntax_initFn_00___x40_Mathlib_Tactic_Linter_Style_1303717341____hygCtx___hyg_2____boxed(lean_object* v_a_1483_){
_start:
{
lean_object* v_res_1484_; 
v_res_1484_ = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_dollarSyntax_initFn_00___x40_Mathlib_Tactic_Linter_Style_1303717341____hygCtx___hyg_2_();
return v_res_1484_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_450967313____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_1503_; lean_object* v___x_1504_; lean_object* v___x_1505_; lean_object* v___x_1506_; 
v___x_1503_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_450967313____hygCtx___hyg_4_));
v___x_1504_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Style_450967313____hygCtx___hyg_4_));
v___x_1505_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_450967313____hygCtx___hyg_4_));
v___x_1506_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__spec__0(v___x_1503_, v___x_1504_, v___x_1505_);
return v___x_1506_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_450967313____hygCtx___hyg_4____boxed(lean_object* v_a_1507_){
_start:
{
lean_object* v_res_1508_; 
v_res_1508_ = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_450967313____hygCtx___hyg_4_();
return v_res_1508_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_Style_lambdaSyntax_findLambdaSyntax(lean_object* v_x_1510_){
_start:
{
if (lean_obj_tag(v_x_1510_) == 1)
{
lean_object* v_kind_1511_; lean_object* v_args_1512_; lean_object* v___y_1514_; size_t v_sz_1532_; size_t v___x_1533_; lean_object* v___x_1534_; lean_object* v___x_1535_; lean_object* v___x_1536_; lean_object* v___x_1537_; uint8_t v___x_1538_; 
v_kind_1511_ = lean_ctor_get(v_x_1510_, 1);
v_args_1512_ = lean_ctor_get(v_x_1510_, 2);
v_sz_1532_ = lean_array_size(v_args_1512_);
v___x_1533_ = ((size_t)0ULL);
lean_inc_ref(v_args_1512_);
v___x_1534_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Linter_Style_lambdaSyntax_findLambdaSyntax_spec__0(v_sz_1532_, v___x_1533_, v_args_1512_);
v___x_1535_ = lean_unsigned_to_nat(0u);
v___x_1536_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_findCDot___closed__0));
v___x_1537_ = lean_array_get_size(v___x_1534_);
v___x_1538_ = lean_nat_dec_lt(v___x_1535_, v___x_1537_);
if (v___x_1538_ == 0)
{
lean_dec_ref(v___x_1534_);
v___y_1514_ = v___x_1536_;
goto v___jp_1513_;
}
else
{
uint8_t v___x_1539_; 
v___x_1539_ = lean_nat_dec_le(v___x_1537_, v___x_1537_);
if (v___x_1539_ == 0)
{
if (v___x_1538_ == 0)
{
lean_dec_ref(v___x_1534_);
v___y_1514_ = v___x_1536_;
goto v___jp_1513_;
}
else
{
size_t v___x_1540_; lean_object* v___x_1541_; 
v___x_1540_ = lean_usize_of_nat(v___x_1537_);
v___x_1541_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_findCDot_spec__1(v___x_1534_, v___x_1533_, v___x_1540_, v___x_1536_);
lean_dec_ref(v___x_1534_);
v___y_1514_ = v___x_1541_;
goto v___jp_1513_;
}
}
else
{
size_t v___x_1542_; lean_object* v___x_1543_; 
v___x_1542_ = lean_usize_of_nat(v___x_1537_);
v___x_1543_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_findCDot_spec__1(v___x_1534_, v___x_1533_, v___x_1542_, v___x_1536_);
lean_dec_ref(v___x_1534_);
v___y_1514_ = v___x_1543_;
goto v___jp_1513_;
}
}
v___jp_1513_:
{
if (lean_obj_tag(v_kind_1511_) == 1)
{
lean_object* v_pre_1515_; 
v_pre_1515_ = lean_ctor_get(v_kind_1511_, 0);
if (lean_obj_tag(v_pre_1515_) == 1)
{
lean_object* v_pre_1516_; 
v_pre_1516_ = lean_ctor_get(v_pre_1515_, 0);
if (lean_obj_tag(v_pre_1516_) == 1)
{
lean_object* v_pre_1517_; 
v_pre_1517_ = lean_ctor_get(v_pre_1516_, 0);
if (lean_obj_tag(v_pre_1517_) == 1)
{
lean_object* v_pre_1518_; 
v_pre_1518_ = lean_ctor_get(v_pre_1517_, 0);
if (lean_obj_tag(v_pre_1518_) == 0)
{
lean_object* v_str_1519_; lean_object* v_str_1520_; lean_object* v_str_1521_; lean_object* v_str_1522_; lean_object* v___x_1523_; uint8_t v___x_1524_; 
v_str_1519_ = lean_ctor_get(v_kind_1511_, 1);
v_str_1520_ = lean_ctor_get(v_pre_1515_, 1);
v_str_1521_ = lean_ctor_get(v_pre_1516_, 1);
v_str_1522_ = lean_ctor_get(v_pre_1517_, 1);
v___x_1523_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__0));
v___x_1524_ = lean_string_dec_eq(v_str_1522_, v___x_1523_);
if (v___x_1524_ == 0)
{
lean_dec_ref_known(v_x_1510_, 3);
return v___y_1514_;
}
else
{
lean_object* v___x_1525_; uint8_t v___x_1526_; 
v___x_1525_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__1));
v___x_1526_ = lean_string_dec_eq(v_str_1521_, v___x_1525_);
if (v___x_1526_ == 0)
{
lean_dec_ref_known(v_x_1510_, 3);
return v___y_1514_;
}
else
{
lean_object* v___x_1527_; uint8_t v___x_1528_; 
v___x_1527_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__5));
v___x_1528_ = lean_string_dec_eq(v_str_1520_, v___x_1527_);
if (v___x_1528_ == 0)
{
lean_dec_ref_known(v_x_1510_, 3);
return v___y_1514_;
}
else
{
lean_object* v___x_1529_; uint8_t v___x_1530_; 
v___x_1529_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_Style_lambdaSyntax_findLambdaSyntax___closed__0));
v___x_1530_ = lean_string_dec_eq(v_str_1519_, v___x_1529_);
if (v___x_1530_ == 0)
{
lean_dec_ref_known(v_x_1510_, 3);
return v___y_1514_;
}
else
{
lean_object* v___x_1531_; 
v___x_1531_ = lean_array_push(v___y_1514_, v_x_1510_);
return v___x_1531_;
}
}
}
}
}
else
{
lean_dec_ref_known(v_x_1510_, 3);
return v___y_1514_;
}
}
else
{
lean_dec_ref_known(v_x_1510_, 3);
return v___y_1514_;
}
}
else
{
lean_dec_ref_known(v_x_1510_, 3);
return v___y_1514_;
}
}
else
{
lean_dec_ref_known(v_x_1510_, 3);
return v___y_1514_;
}
}
else
{
lean_dec_ref_known(v_x_1510_, 3);
return v___y_1514_;
}
}
}
else
{
lean_object* v___x_1544_; 
lean_dec(v_x_1510_);
v___x_1544_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_findCDot___closed__1));
return v___x_1544_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Linter_Style_lambdaSyntax_findLambdaSyntax_spec__0(size_t v_sz_1545_, size_t v_i_1546_, lean_object* v_bs_1547_){
_start:
{
uint8_t v___x_1548_; 
v___x_1548_ = lean_usize_dec_lt(v_i_1546_, v_sz_1545_);
if (v___x_1548_ == 0)
{
return v_bs_1547_;
}
else
{
lean_object* v_v_1549_; lean_object* v___x_1550_; lean_object* v_bs_x27_1551_; lean_object* v___x_1552_; size_t v___x_1553_; size_t v___x_1554_; lean_object* v___x_1555_; 
v_v_1549_ = lean_array_uget(v_bs_1547_, v_i_1546_);
v___x_1550_ = lean_unsigned_to_nat(0u);
v_bs_x27_1551_ = lean_array_uset(v_bs_1547_, v_i_1546_, v___x_1550_);
v___x_1552_ = lp_mathlib_Mathlib_Linter_Style_lambdaSyntax_findLambdaSyntax(v_v_1549_);
v___x_1553_ = ((size_t)1ULL);
v___x_1554_ = lean_usize_add(v_i_1546_, v___x_1553_);
v___x_1555_ = lean_array_uset(v_bs_x27_1551_, v_i_1546_, v___x_1552_);
v_i_1546_ = v___x_1554_;
v_bs_1547_ = v___x_1555_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Linter_Style_lambdaSyntax_findLambdaSyntax_spec__0___boxed(lean_object* v_sz_1557_, lean_object* v_i_1558_, lean_object* v_bs_1559_){
_start:
{
size_t v_sz_boxed_1560_; size_t v_i_boxed_1561_; lean_object* v_res_1562_; 
v_sz_boxed_1560_ = lean_unbox_usize(v_sz_1557_);
lean_dec(v_sz_1557_);
v_i_boxed_1561_ = lean_unbox_usize(v_i_1558_);
lean_dec(v_i_1558_);
v_res_1562_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Linter_Style_lambdaSyntax_findLambdaSyntax_spec__0(v_sz_boxed_1560_, v_i_boxed_1561_, v_bs_1559_);
return v_res_1562_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_lambdaSyntaxLinter_spec__0___closed__2(void){
_start:
{
lean_object* v___x_1565_; lean_object* v___x_1566_; 
v___x_1565_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_lambdaSyntaxLinter_spec__0___closed__1));
v___x_1566_ = l_Lean_stringToMessageData(v___x_1565_);
return v___x_1566_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_lambdaSyntaxLinter_spec__0(lean_object* v_as_1567_, size_t v_sz_1568_, size_t v_i_1569_, lean_object* v_b_1570_, lean_object* v___y_1571_, lean_object* v___y_1572_){
_start:
{
lean_object* v_a_1575_; uint8_t v___x_1579_; 
v___x_1579_ = lean_usize_dec_lt(v_i_1569_, v_sz_1568_);
if (v___x_1579_ == 0)
{
lean_object* v___x_1580_; 
v___x_1580_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1580_, 0, v_b_1570_);
return v___x_1580_;
}
else
{
lean_object* v___x_1581_; lean_object* v_a_1582_; lean_object* v___x_1583_; lean_object* v___x_1584_; 
v___x_1581_ = lean_box(0);
v_a_1582_ = lean_array_uget_borrowed(v_as_1567_, v_i_1569_);
v___x_1583_ = lean_unsigned_to_nat(0u);
v___x_1584_ = l_Lean_Syntax_getArg(v_a_1582_, v___x_1583_);
if (lean_obj_tag(v___x_1584_) == 2)
{
lean_object* v_val_1585_; lean_object* v___x_1586_; uint8_t v___x_1587_; 
v_val_1585_ = lean_ctor_get(v___x_1584_, 1);
lean_inc_ref(v_val_1585_);
v___x_1586_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_lambdaSyntaxLinter_spec__0___closed__0));
v___x_1587_ = lean_string_dec_eq(v_val_1585_, v___x_1586_);
lean_dec_ref(v_val_1585_);
if (v___x_1587_ == 0)
{
lean_dec_ref_known(v___x_1584_, 2);
v_a_1575_ = v___x_1581_;
goto v___jp_1574_;
}
else
{
lean_object* v___x_1588_; lean_object* v___x_1589_; lean_object* v___x_1590_; 
v___x_1588_ = lp_mathlib_Mathlib_Linter_linter_style_lambdaSyntax;
v___x_1589_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_lambdaSyntaxLinter_spec__0___closed__2, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_lambdaSyntaxLinter_spec__0___closed__2_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_lambdaSyntaxLinter_spec__0___closed__2);
v___x_1590_ = lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__1(v___x_1588_, v___x_1584_, v___x_1589_, v___y_1571_, v___y_1572_);
if (lean_obj_tag(v___x_1590_) == 0)
{
lean_dec_ref_known(v___x_1590_, 1);
v_a_1575_ = v___x_1581_;
goto v___jp_1574_;
}
else
{
return v___x_1590_;
}
}
}
else
{
lean_dec(v___x_1584_);
v_a_1575_ = v___x_1581_;
goto v___jp_1574_;
}
}
v___jp_1574_:
{
size_t v___x_1576_; size_t v___x_1577_; 
v___x_1576_ = ((size_t)1ULL);
v___x_1577_ = lean_usize_add(v_i_1569_, v___x_1576_);
v_i_1569_ = v___x_1577_;
v_b_1570_ = v_a_1575_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_lambdaSyntaxLinter_spec__0___boxed(lean_object* v_as_1591_, lean_object* v_sz_1592_, lean_object* v_i_1593_, lean_object* v_b_1594_, lean_object* v___y_1595_, lean_object* v___y_1596_, lean_object* v___y_1597_){
_start:
{
size_t v_sz_boxed_1598_; size_t v_i_boxed_1599_; lean_object* v_res_1600_; 
v_sz_boxed_1598_ = lean_unbox_usize(v_sz_1592_);
lean_dec(v_sz_1592_);
v_i_boxed_1599_ = lean_unbox_usize(v_i_1593_);
lean_dec(v_i_1593_);
v_res_1600_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_lambdaSyntaxLinter_spec__0(v_as_1591_, v_sz_boxed_1598_, v_i_boxed_1599_, v_b_1594_, v___y_1595_, v___y_1596_);
lean_dec(v___y_1596_);
lean_dec_ref(v___y_1595_);
lean_dec_ref(v_as_1591_);
return v_res_1600_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_lambdaSyntaxLinter___lam__0(lean_object* v_stx_1601_, lean_object* v___y_1602_, lean_object* v___y_1603_){
_start:
{
lean_object* v___x_1605_; lean_object* v_a_1606_; lean_object* v___x_1608_; uint8_t v_isShared_1609_; uint8_t v_isSharedCheck_1636_; 
v___x_1605_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__0(v___y_1602_, v___y_1603_);
v_a_1606_ = lean_ctor_get(v___x_1605_, 0);
v_isSharedCheck_1636_ = !lean_is_exclusive(v___x_1605_);
if (v_isSharedCheck_1636_ == 0)
{
v___x_1608_ = v___x_1605_;
v_isShared_1609_ = v_isSharedCheck_1636_;
goto v_resetjp_1607_;
}
else
{
lean_inc(v_a_1606_);
lean_dec(v___x_1605_);
v___x_1608_ = lean_box(0);
v_isShared_1609_ = v_isSharedCheck_1636_;
goto v_resetjp_1607_;
}
v_resetjp_1607_:
{
lean_object* v___x_1610_; uint8_t v___x_1611_; 
v___x_1610_ = lp_mathlib_Mathlib_Linter_linter_style_lambdaSyntax;
v___x_1611_ = l_Lean_Linter_getLinterValue(v___x_1610_, v_a_1606_);
lean_dec(v_a_1606_);
if (v___x_1611_ == 0)
{
lean_object* v___x_1612_; lean_object* v___x_1614_; 
lean_dec(v_stx_1601_);
v___x_1612_ = lean_box(0);
if (v_isShared_1609_ == 0)
{
lean_ctor_set(v___x_1608_, 0, v___x_1612_);
v___x_1614_ = v___x_1608_;
goto v_reusejp_1613_;
}
else
{
lean_object* v_reuseFailAlloc_1615_; 
v_reuseFailAlloc_1615_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1615_, 0, v___x_1612_);
v___x_1614_ = v_reuseFailAlloc_1615_;
goto v_reusejp_1613_;
}
v_reusejp_1613_:
{
return v___x_1614_;
}
}
else
{
lean_object* v___x_1616_; lean_object* v_messages_1617_; uint8_t v___x_1618_; 
v___x_1616_ = lean_st_ref_get(v___y_1603_);
v_messages_1617_ = lean_ctor_get(v___x_1616_, 1);
lean_inc_ref(v_messages_1617_);
lean_dec(v___x_1616_);
v___x_1618_ = l_Lean_MessageLog_hasErrors(v_messages_1617_);
lean_dec_ref(v_messages_1617_);
if (v___x_1618_ == 0)
{
lean_object* v___x_1619_; lean_object* v___x_1620_; size_t v_sz_1621_; size_t v___x_1622_; lean_object* v___x_1623_; 
lean_del_object(v___x_1608_);
v___x_1619_ = lp_mathlib_Mathlib_Linter_Style_lambdaSyntax_findLambdaSyntax(v_stx_1601_);
v___x_1620_ = lean_box(0);
v_sz_1621_ = lean_array_size(v___x_1619_);
v___x_1622_ = ((size_t)0ULL);
v___x_1623_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_lambdaSyntaxLinter_spec__0(v___x_1619_, v_sz_1621_, v___x_1622_, v___x_1620_, v___y_1602_, v___y_1603_);
lean_dec_ref(v___x_1619_);
if (lean_obj_tag(v___x_1623_) == 0)
{
lean_object* v___x_1625_; uint8_t v_isShared_1626_; uint8_t v_isSharedCheck_1630_; 
v_isSharedCheck_1630_ = !lean_is_exclusive(v___x_1623_);
if (v_isSharedCheck_1630_ == 0)
{
lean_object* v_unused_1631_; 
v_unused_1631_ = lean_ctor_get(v___x_1623_, 0);
lean_dec(v_unused_1631_);
v___x_1625_ = v___x_1623_;
v_isShared_1626_ = v_isSharedCheck_1630_;
goto v_resetjp_1624_;
}
else
{
lean_dec(v___x_1623_);
v___x_1625_ = lean_box(0);
v_isShared_1626_ = v_isSharedCheck_1630_;
goto v_resetjp_1624_;
}
v_resetjp_1624_:
{
lean_object* v___x_1628_; 
if (v_isShared_1626_ == 0)
{
lean_ctor_set(v___x_1625_, 0, v___x_1620_);
v___x_1628_ = v___x_1625_;
goto v_reusejp_1627_;
}
else
{
lean_object* v_reuseFailAlloc_1629_; 
v_reuseFailAlloc_1629_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1629_, 0, v___x_1620_);
v___x_1628_ = v_reuseFailAlloc_1629_;
goto v_reusejp_1627_;
}
v_reusejp_1627_:
{
return v___x_1628_;
}
}
}
else
{
return v___x_1623_;
}
}
else
{
lean_object* v___x_1632_; lean_object* v___x_1634_; 
lean_dec(v_stx_1601_);
v___x_1632_ = lean_box(0);
if (v_isShared_1609_ == 0)
{
lean_ctor_set(v___x_1608_, 0, v___x_1632_);
v___x_1634_ = v___x_1608_;
goto v_reusejp_1633_;
}
else
{
lean_object* v_reuseFailAlloc_1635_; 
v_reuseFailAlloc_1635_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1635_, 0, v___x_1632_);
v___x_1634_ = v_reuseFailAlloc_1635_;
goto v_reusejp_1633_;
}
v_reusejp_1633_:
{
return v___x_1634_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_lambdaSyntaxLinter___lam__0___boxed(lean_object* v_stx_1637_, lean_object* v___y_1638_, lean_object* v___y_1639_, lean_object* v___y_1640_){
_start:
{
lean_object* v_res_1641_; 
v_res_1641_ = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_lambdaSyntaxLinter___lam__0(v_stx_1637_, v___y_1638_, v___y_1639_);
lean_dec(v___y_1639_);
lean_dec_ref(v___y_1638_);
return v_res_1641_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_initFn_00___x40_Mathlib_Tactic_Linter_Style_1166937461____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_1657_; lean_object* v___x_1658_; 
v___x_1657_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_lambdaSyntaxLinter));
v___x_1658_ = l_Lean_Elab_Command_addLinter(v___x_1657_);
return v___x_1658_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_initFn_00___x40_Mathlib_Tactic_Linter_Style_1166937461____hygCtx___hyg_2____boxed(lean_object* v_a_1659_){
_start:
{
lean_object* v_res_1660_; 
v_res_1660_ = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_initFn_00___x40_Mathlib_Tactic_Linter_Style_1166937461____hygCtx___hyg_2_();
return v_res_1660_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_695976056____hygCtx___hyg_4__spec__0(lean_object* v_name_1661_, lean_object* v_decl_1662_, lean_object* v_ref_1663_){
_start:
{
lean_object* v_defValue_1665_; lean_object* v_descr_1666_; lean_object* v_deprecation_x3f_1667_; lean_object* v___x_1668_; lean_object* v___x_1669_; lean_object* v___x_1670_; 
v_defValue_1665_ = lean_ctor_get(v_decl_1662_, 0);
v_descr_1666_ = lean_ctor_get(v_decl_1662_, 1);
v_deprecation_x3f_1667_ = lean_ctor_get(v_decl_1662_, 2);
lean_inc(v_defValue_1665_);
v___x_1668_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1668_, 0, v_defValue_1665_);
lean_inc(v_deprecation_x3f_1667_);
lean_inc_ref(v_descr_1666_);
lean_inc_n(v_name_1661_, 2);
v___x_1669_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1669_, 0, v_name_1661_);
lean_ctor_set(v___x_1669_, 1, v_ref_1663_);
lean_ctor_set(v___x_1669_, 2, v___x_1668_);
lean_ctor_set(v___x_1669_, 3, v_descr_1666_);
lean_ctor_set(v___x_1669_, 4, v_deprecation_x3f_1667_);
v___x_1670_ = lean_register_option(v_name_1661_, v___x_1669_);
if (lean_obj_tag(v___x_1670_) == 0)
{
lean_object* v___x_1672_; uint8_t v_isShared_1673_; uint8_t v_isSharedCheck_1678_; 
v_isSharedCheck_1678_ = !lean_is_exclusive(v___x_1670_);
if (v_isSharedCheck_1678_ == 0)
{
lean_object* v_unused_1679_; 
v_unused_1679_ = lean_ctor_get(v___x_1670_, 0);
lean_dec(v_unused_1679_);
v___x_1672_ = v___x_1670_;
v_isShared_1673_ = v_isSharedCheck_1678_;
goto v_resetjp_1671_;
}
else
{
lean_dec(v___x_1670_);
v___x_1672_ = lean_box(0);
v_isShared_1673_ = v_isSharedCheck_1678_;
goto v_resetjp_1671_;
}
v_resetjp_1671_:
{
lean_object* v___x_1674_; lean_object* v___x_1676_; 
lean_inc(v_defValue_1665_);
v___x_1674_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1674_, 0, v_name_1661_);
lean_ctor_set(v___x_1674_, 1, v_defValue_1665_);
if (v_isShared_1673_ == 0)
{
lean_ctor_set(v___x_1672_, 0, v___x_1674_);
v___x_1676_ = v___x_1672_;
goto v_reusejp_1675_;
}
else
{
lean_object* v_reuseFailAlloc_1677_; 
v_reuseFailAlloc_1677_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1677_, 0, v___x_1674_);
v___x_1676_ = v_reuseFailAlloc_1677_;
goto v_reusejp_1675_;
}
v_reusejp_1675_:
{
return v___x_1676_;
}
}
}
else
{
lean_object* v_a_1680_; lean_object* v___x_1682_; uint8_t v_isShared_1683_; uint8_t v_isSharedCheck_1687_; 
lean_dec(v_name_1661_);
v_a_1680_ = lean_ctor_get(v___x_1670_, 0);
v_isSharedCheck_1687_ = !lean_is_exclusive(v___x_1670_);
if (v_isSharedCheck_1687_ == 0)
{
v___x_1682_ = v___x_1670_;
v_isShared_1683_ = v_isSharedCheck_1687_;
goto v_resetjp_1681_;
}
else
{
lean_inc(v_a_1680_);
lean_dec(v___x_1670_);
v___x_1682_ = lean_box(0);
v_isShared_1683_ = v_isSharedCheck_1687_;
goto v_resetjp_1681_;
}
v_resetjp_1681_:
{
lean_object* v___x_1685_; 
if (v_isShared_1683_ == 0)
{
v___x_1685_ = v___x_1682_;
goto v_reusejp_1684_;
}
else
{
lean_object* v_reuseFailAlloc_1686_; 
v_reuseFailAlloc_1686_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1686_, 0, v_a_1680_);
v___x_1685_ = v_reuseFailAlloc_1686_;
goto v_reusejp_1684_;
}
v_reusejp_1684_:
{
return v___x_1685_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_695976056____hygCtx___hyg_4__spec__0___boxed(lean_object* v_name_1688_, lean_object* v_decl_1689_, lean_object* v_ref_1690_, lean_object* v_a_1691_){
_start:
{
lean_object* v_res_1692_; 
v_res_1692_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_695976056____hygCtx___hyg_4__spec__0(v_name_1688_, v_decl_1689_, v_ref_1690_);
lean_dec_ref(v_decl_1689_);
return v_res_1692_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_695976056____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_1710_; lean_object* v___x_1711_; lean_object* v___x_1712_; lean_object* v___x_1713_; 
v___x_1710_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_695976056____hygCtx___hyg_4_));
v___x_1711_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Style_695976056____hygCtx___hyg_4_));
v___x_1712_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_695976056____hygCtx___hyg_4_));
v___x_1713_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_695976056____hygCtx___hyg_4__spec__0(v___x_1710_, v___x_1711_, v___x_1712_);
return v___x_1713_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_695976056____hygCtx___hyg_4____boxed(lean_object* v_a_1714_){
_start:
{
lean_object* v_res_1715_; 
v_res_1715_ = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_695976056____hygCtx___hyg_4_();
return v_res_1715_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_583422302____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_1733_; lean_object* v___x_1734_; lean_object* v___x_1735_; lean_object* v___x_1736_; 
v___x_1733_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_583422302____hygCtx___hyg_4_));
v___x_1734_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Style_583422302____hygCtx___hyg_4_));
v___x_1735_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_583422302____hygCtx___hyg_4_));
v___x_1736_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_695976056____hygCtx___hyg_4__spec__0(v___x_1733_, v___x_1734_, v___x_1735_);
return v___x_1736_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_583422302____hygCtx___hyg_4____boxed(lean_object* v_a_1737_){
_start:
{
lean_object* v_res_1738_; 
v_res_1738_ = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_583422302____hygCtx___hyg_4_();
return v_res_1738_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter_spec__0(lean_object* v_opts_1739_, lean_object* v_opt_1740_){
_start:
{
lean_object* v_name_1741_; lean_object* v_defValue_1742_; lean_object* v_map_1743_; lean_object* v___x_1744_; 
v_name_1741_ = lean_ctor_get(v_opt_1740_, 0);
v_defValue_1742_ = lean_ctor_get(v_opt_1740_, 1);
v_map_1743_ = lean_ctor_get(v_opts_1739_, 0);
v___x_1744_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_1743_, v_name_1741_);
if (lean_obj_tag(v___x_1744_) == 0)
{
lean_inc(v_defValue_1742_);
return v_defValue_1742_;
}
else
{
lean_object* v_val_1745_; 
v_val_1745_ = lean_ctor_get(v___x_1744_, 0);
lean_inc(v_val_1745_);
lean_dec_ref_known(v___x_1744_, 1);
if (lean_obj_tag(v_val_1745_) == 3)
{
lean_object* v_v_1746_; 
v_v_1746_ = lean_ctor_get(v_val_1745_, 0);
lean_inc(v_v_1746_);
lean_dec_ref_known(v_val_1745_, 1);
return v_v_1746_;
}
else
{
lean_dec(v_val_1745_);
lean_inc(v_defValue_1742_);
return v_defValue_1742_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter_spec__0___boxed(lean_object* v_opts_1747_, lean_object* v_opt_1748_){
_start:
{
lean_object* v_res_1749_; 
v_res_1749_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter_spec__0(v_opts_1747_, v_opt_1748_);
lean_dec_ref(v_opt_1748_);
lean_dec_ref(v_opts_1747_);
return v_res_1749_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter_spec__1___redArg(lean_object* v___y_1750_){
_start:
{
lean_object* v___x_1752_; lean_object* v_env_1753_; lean_object* v___x_1754_; lean_object* v_mainModule_1755_; lean_object* v___x_1756_; 
v___x_1752_ = lean_st_ref_get(v___y_1750_);
v_env_1753_ = lean_ctor_get(v___x_1752_, 0);
lean_inc_ref(v_env_1753_);
lean_dec(v___x_1752_);
v___x_1754_ = l_Lean_Environment_header(v_env_1753_);
lean_dec_ref(v_env_1753_);
v_mainModule_1755_ = lean_ctor_get(v___x_1754_, 0);
lean_inc(v_mainModule_1755_);
lean_dec_ref(v___x_1754_);
v___x_1756_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1756_, 0, v_mainModule_1755_);
return v___x_1756_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter_spec__1___redArg___boxed(lean_object* v___y_1757_, lean_object* v___y_1758_){
_start:
{
lean_object* v_res_1759_; 
v_res_1759_ = lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter_spec__1___redArg(v___y_1757_);
lean_dec(v___y_1757_);
return v_res_1759_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter_spec__1(lean_object* v___y_1760_, lean_object* v___y_1761_){
_start:
{
lean_object* v___x_1763_; 
v___x_1763_ = lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter_spec__1___redArg(v___y_1761_);
return v___x_1763_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter_spec__1___boxed(lean_object* v___y_1764_, lean_object* v___y_1765_, lean_object* v___y_1766_){
_start:
{
lean_object* v_res_1767_; 
v_res_1767_ = lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter_spec__1(v___y_1764_, v___y_1765_);
lean_dec(v___y_1765_);
lean_dec_ref(v___y_1764_);
return v_res_1767_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter_spec__2___closed__1(void){
_start:
{
lean_object* v___x_1769_; lean_object* v___x_1770_; 
v___x_1769_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter_spec__2___closed__0));
v___x_1770_ = l_Lean_stringToMessageData(v___x_1769_);
return v___x_1770_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter_spec__2(lean_object* v_linterOption_1771_, lean_object* v_stx_1772_, lean_object* v_msg_1773_, lean_object* v___y_1774_, lean_object* v___y_1775_){
_start:
{
lean_object* v_name_1777_; lean_object* v___x_1779_; uint8_t v_isShared_1780_; uint8_t v_isSharedCheck_1792_; 
v_name_1777_ = lean_ctor_get(v_linterOption_1771_, 0);
v_isSharedCheck_1792_ = !lean_is_exclusive(v_linterOption_1771_);
if (v_isSharedCheck_1792_ == 0)
{
lean_object* v_unused_1793_; 
v_unused_1793_ = lean_ctor_get(v_linterOption_1771_, 1);
lean_dec(v_unused_1793_);
v___x_1779_ = v_linterOption_1771_;
v_isShared_1780_ = v_isSharedCheck_1792_;
goto v_resetjp_1778_;
}
else
{
lean_inc(v_name_1777_);
lean_dec(v_linterOption_1771_);
v___x_1779_ = lean_box(0);
v_isShared_1780_ = v_isSharedCheck_1792_;
goto v_resetjp_1778_;
}
v_resetjp_1778_:
{
lean_object* v___x_1781_; lean_object* v___x_1782_; lean_object* v___x_1784_; 
v___x_1781_ = lean_obj_once(&lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__1___closed__1, &lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__1___closed__1_once, _init_lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__1___closed__1);
lean_inc(v_name_1777_);
v___x_1782_ = l_Lean_MessageData_ofName(v_name_1777_);
if (v_isShared_1780_ == 0)
{
lean_ctor_set_tag(v___x_1779_, 7);
lean_ctor_set(v___x_1779_, 1, v___x_1782_);
lean_ctor_set(v___x_1779_, 0, v___x_1781_);
v___x_1784_ = v___x_1779_;
goto v_reusejp_1783_;
}
else
{
lean_object* v_reuseFailAlloc_1791_; 
v_reuseFailAlloc_1791_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1791_, 0, v___x_1781_);
lean_ctor_set(v_reuseFailAlloc_1791_, 1, v___x_1782_);
v___x_1784_ = v_reuseFailAlloc_1791_;
goto v_reusejp_1783_;
}
v_reusejp_1783_:
{
lean_object* v___x_1785_; lean_object* v___x_1786_; lean_object* v_disable_1787_; lean_object* v___x_1788_; lean_object* v___x_1789_; lean_object* v___x_1790_; 
v___x_1785_ = lean_obj_once(&lp_mathlib_Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter_spec__2___closed__1, &lp_mathlib_Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter_spec__2___closed__1_once, _init_lp_mathlib_Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter_spec__2___closed__1);
v___x_1786_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1786_, 0, v___x_1784_);
lean_ctor_set(v___x_1786_, 1, v___x_1785_);
v_disable_1787_ = l_Lean_MessageData_note(v___x_1786_);
v___x_1788_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1788_, 0, v_msg_1773_);
lean_ctor_set(v___x_1788_, 1, v_disable_1787_);
v___x_1789_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_1789_, 0, v_name_1777_);
lean_ctor_set(v___x_1789_, 1, v___x_1788_);
v___x_1790_ = lp_mathlib_Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3(v_stx_1772_, v___x_1789_, v___y_1774_, v___y_1775_);
return v___x_1790_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter_spec__2___boxed(lean_object* v_linterOption_1794_, lean_object* v_stx_1795_, lean_object* v_msg_1796_, lean_object* v___y_1797_, lean_object* v___y_1798_, lean_object* v___y_1799_){
_start:
{
lean_object* v_res_1800_; 
v_res_1800_ = lp_mathlib_Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter_spec__2(v_linterOption_1794_, v_stx_1795_, v_msg_1796_, v___y_1797_, v___y_1798_);
lean_dec(v___y_1798_);
lean_dec_ref(v___y_1797_);
lean_dec(v_stx_1795_);
return v_res_1800_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__1(void){
_start:
{
lean_object* v___x_1802_; lean_object* v___x_1803_; 
v___x_1802_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__0));
v___x_1803_ = l_Lean_stringToMessageData(v___x_1802_);
return v___x_1803_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__3(void){
_start:
{
lean_object* v___x_1805_; lean_object* v___x_1806_; 
v___x_1805_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__2));
v___x_1806_ = l_Lean_stringToMessageData(v___x_1805_);
return v___x_1806_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__5(void){
_start:
{
lean_object* v___x_1808_; lean_object* v___x_1809_; 
v___x_1808_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__4));
v___x_1809_ = l_Lean_stringToMessageData(v___x_1808_);
return v___x_1809_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__7(void){
_start:
{
lean_object* v___x_1811_; lean_object* v___x_1812_; 
v___x_1811_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__6));
v___x_1812_ = l_Lean_stringToMessageData(v___x_1811_);
return v___x_1812_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__9(void){
_start:
{
lean_object* v___x_1814_; lean_object* v___x_1815_; 
v___x_1814_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__8));
v___x_1815_ = l_Lean_stringToMessageData(v___x_1814_);
return v___x_1815_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__11(void){
_start:
{
lean_object* v___x_1817_; lean_object* v___x_1818_; 
v___x_1817_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__10));
v___x_1818_ = l_Lean_stringToMessageData(v___x_1817_);
return v___x_1818_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__13(void){
_start:
{
lean_object* v___x_1820_; lean_object* v___x_1821_; 
v___x_1820_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__12));
v___x_1821_ = l_Lean_stringToMessageData(v___x_1820_);
return v___x_1821_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__15(void){
_start:
{
lean_object* v___x_1823_; lean_object* v___x_1824_; 
v___x_1823_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__14));
v___x_1824_ = l_Lean_stringToMessageData(v___x_1823_);
return v___x_1824_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__17(void){
_start:
{
lean_object* v___x_1826_; lean_object* v___x_1827_; 
v___x_1826_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__16));
v___x_1827_ = l_Lean_stringToMessageData(v___x_1826_);
return v___x_1827_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__19(void){
_start:
{
lean_object* v___x_1829_; lean_object* v___x_1830_; 
v___x_1829_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__18));
v___x_1830_ = l_Lean_stringToMessageData(v___x_1829_);
return v___x_1830_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__21(void){
_start:
{
lean_object* v___x_1832_; lean_object* v___x_1833_; 
v___x_1832_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__20));
v___x_1833_ = l_Lean_stringToMessageData(v___x_1832_);
return v___x_1833_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__24(void){
_start:
{
lean_object* v___x_1837_; lean_object* v___x_1838_; 
v___x_1837_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__23));
v___x_1838_ = l_Lean_stringToMessageData(v___x_1837_);
return v___x_1838_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__26(void){
_start:
{
lean_object* v___x_1840_; lean_object* v___x_1841_; 
v___x_1840_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__25));
v___x_1841_ = l_Lean_stringToMessageData(v___x_1840_);
return v___x_1841_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0(lean_object* v_stx_1842_, lean_object* v___y_1843_, lean_object* v___y_1844_){
_start:
{
lean_object* v___x_1849_; lean_object* v_scopes_1850_; lean_object* v___x_1851_; lean_object* v___x_1852_; lean_object* v_opts_1853_; lean_object* v___x_1854_; lean_object* v___x_1855_; lean_object* v___y_1857_; lean_object* v___y_1858_; lean_object* v___y_1859_; uint8_t v___y_1860_; lean_object* v___x_1907_; uint8_t v___x_1908_; 
v___x_1849_ = lean_st_ref_get(v___y_1844_);
v_scopes_1850_ = lean_ctor_get(v___x_1849_, 2);
lean_inc(v_scopes_1850_);
lean_dec(v___x_1849_);
v___x_1851_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_1852_ = l_List_head_x21___redArg(v___x_1851_, v_scopes_1850_);
lean_dec(v_scopes_1850_);
v_opts_1853_ = lean_ctor_get(v___x_1852_, 1);
lean_inc_ref(v_opts_1853_);
lean_dec(v___x_1852_);
v___x_1854_ = lp_mathlib_Mathlib_Linter_linter_style_longFile;
v___x_1855_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter_spec__0(v_opts_1853_, v___x_1854_);
lean_dec_ref(v_opts_1853_);
v___x_1907_ = lean_unsigned_to_nat(0u);
v___x_1908_ = lean_nat_dec_eq(v___x_1855_, v___x_1907_);
if (v___x_1908_ == 0)
{
lean_object* v___x_1909_; lean_object* v_scopes_1910_; lean_object* v___x_1911_; lean_object* v_opts_1912_; lean_object* v___x_1913_; lean_object* v___x_1914_; lean_object* v___y_1916_; lean_object* v___y_1917_; lean_object* v___y_1918_; lean_object* v___y_1922_; uint8_t v___y_1923_; lean_object* v___x_1977_; uint8_t v___x_1978_; 
v___x_1909_ = lean_st_ref_get(v___y_1844_);
v_scopes_1910_ = lean_ctor_get(v___x_1909_, 2);
lean_inc(v_scopes_1910_);
lean_dec(v___x_1909_);
v___x_1911_ = l_List_head_x21___redArg(v___x_1851_, v_scopes_1910_);
lean_dec(v_scopes_1910_);
v_opts_1912_ = lean_ctor_get(v___x_1911_, 1);
lean_inc_ref(v_opts_1912_);
lean_dec(v___x_1911_);
v___x_1913_ = lp_mathlib_Mathlib_Linter_linter_style_longFileDefValue;
v___x_1914_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter_spec__0(v_opts_1912_, v___x_1913_);
lean_dec_ref(v_opts_1912_);
v___x_1977_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__4));
lean_inc(v_stx_1842_);
v___x_1978_ = l_Lean_Syntax_isOfKind(v_stx_1842_, v___x_1977_);
if (v___x_1978_ == 0)
{
goto v___jp_1950_;
}
else
{
lean_object* v___x_1979_; lean_object* v___x_1980_; lean_object* v___x_1981_; uint8_t v___x_1982_; 
v___x_1979_ = lean_unsigned_to_nat(1u);
v___x_1980_ = l_Lean_Syntax_getArg(v_stx_1842_, v___x_1979_);
v___x_1981_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_695976056____hygCtx___hyg_4_));
v___x_1982_ = l_Lean_Syntax_matchesIdent(v___x_1980_, v___x_1981_);
lean_dec(v___x_1980_);
if (v___x_1982_ == 0)
{
goto v___jp_1950_;
}
else
{
lean_object* v___x_1983_; lean_object* v___x_1984_; uint8_t v___x_1985_; 
v___x_1983_ = lean_unsigned_to_nat(2u);
v___x_1984_ = l_Lean_Syntax_getArg(v_stx_1842_, v___x_1983_);
v___x_1985_ = l_Lean_Syntax_matchesNull(v___x_1984_, v___x_1907_);
if (v___x_1985_ == 0)
{
goto v___jp_1950_;
}
else
{
lean_object* v___x_1986_; lean_object* v___x_1987_; lean_object* v___x_1988_; uint8_t v___x_1989_; 
v___x_1986_ = lean_unsigned_to_nat(3u);
v___x_1987_ = l_Lean_Syntax_getArg(v_stx_1842_, v___x_1986_);
v___x_1988_ = l_Lean_TSyntax_getNat(v___x_1987_);
lean_dec(v___x_1987_);
v___x_1989_ = lean_nat_dec_le(v___x_1988_, v___x_1914_);
lean_dec(v___x_1988_);
if (v___x_1989_ == 0)
{
goto v___jp_1950_;
}
else
{
lean_object* v___x_1990_; lean_object* v___x_1991_; lean_object* v___x_1992_; lean_object* v___x_1993_; lean_object* v___x_1994_; lean_object* v___x_1995_; lean_object* v___x_1996_; lean_object* v___x_1997_; lean_object* v___x_1998_; lean_object* v___x_1999_; lean_object* v___x_2000_; lean_object* v___x_2001_; lean_object* v___x_2002_; lean_object* v___x_2003_; lean_object* v___x_2004_; lean_object* v___x_2005_; lean_object* v___x_2006_; 
v___x_1990_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__17, &lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__17_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__17);
v___x_1991_ = l_Nat_reprFast(v___x_1914_);
v___x_1992_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1992_, 0, v___x_1991_);
v___x_1993_ = l_Lean_MessageData_ofFormat(v___x_1992_);
v___x_1994_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1994_, 0, v___x_1990_);
lean_ctor_set(v___x_1994_, 1, v___x_1993_);
v___x_1995_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__24, &lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__24_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__24);
v___x_1996_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1996_, 0, v___x_1994_);
lean_ctor_set(v___x_1996_, 1, v___x_1995_);
v___x_1997_ = l_Nat_reprFast(v___x_1855_);
v___x_1998_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1998_, 0, v___x_1997_);
v___x_1999_ = l_Lean_MessageData_ofFormat(v___x_1998_);
lean_inc_ref(v___x_1999_);
v___x_2000_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2000_, 0, v___x_1996_);
lean_ctor_set(v___x_2000_, 1, v___x_1999_);
v___x_2001_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__26, &lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__26_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__26);
v___x_2002_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2002_, 0, v___x_2000_);
lean_ctor_set(v___x_2002_, 1, v___x_2001_);
v___x_2003_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2003_, 0, v___x_2002_);
lean_ctor_set(v___x_2003_, 1, v___x_1999_);
v___x_2004_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__9, &lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__9_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__9);
v___x_2005_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2005_, 0, v___x_2003_);
lean_ctor_set(v___x_2005_, 1, v___x_2004_);
v___x_2006_ = lp_mathlib_Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter_spec__2(v___x_1854_, v_stx_1842_, v___x_2005_, v___y_1843_, v___y_1844_);
lean_dec(v_stx_1842_);
return v___x_2006_;
}
}
}
}
v___jp_1915_:
{
uint8_t v___x_1919_; 
v___x_1919_ = lean_nat_dec_le(v___x_1914_, v___x_1855_);
lean_dec(v___x_1914_);
if (v___x_1919_ == 0)
{
v___y_1857_ = v___y_1917_;
v___y_1858_ = v___y_1916_;
v___y_1859_ = v___y_1918_;
v___y_1860_ = v___x_1919_;
goto v___jp_1856_;
}
else
{
uint8_t v___x_1920_; 
v___x_1920_ = lean_nat_dec_lt(v___x_1855_, v___y_1917_);
v___y_1857_ = v___y_1917_;
v___y_1858_ = v___y_1916_;
v___y_1859_ = v___y_1918_;
v___y_1860_ = v___x_1920_;
goto v___jp_1856_;
}
}
v___jp_1921_:
{
if (v___y_1923_ == 0)
{
lean_object* v___x_1924_; lean_object* v___x_1925_; lean_object* v___x_1926_; lean_object* v___x_1927_; lean_object* v___x_1928_; uint8_t v___x_1929_; 
v___x_1924_ = lean_unsigned_to_nat(100u);
v___x_1925_ = lean_nat_div(v___y_1922_, v___x_1924_);
v___x_1926_ = lean_nat_mul(v___x_1925_, v___x_1924_);
lean_dec(v___x_1925_);
v___x_1927_ = lean_unsigned_to_nat(200u);
v___x_1928_ = lean_nat_add(v___x_1926_, v___x_1927_);
lean_dec(v___x_1926_);
v___x_1929_ = lean_nat_dec_le(v___x_1928_, v___x_1914_);
if (v___x_1929_ == 0)
{
v___y_1916_ = v___x_1924_;
v___y_1917_ = v___y_1922_;
v___y_1918_ = v___x_1928_;
goto v___jp_1915_;
}
else
{
lean_dec(v___x_1928_);
lean_inc(v___x_1914_);
v___y_1916_ = v___x_1924_;
v___y_1917_ = v___y_1922_;
v___y_1918_ = v___x_1914_;
goto v___jp_1915_;
}
}
else
{
lean_object* v___x_1930_; lean_object* v___x_1931_; lean_object* v___x_1932_; lean_object* v___x_1933_; lean_object* v___x_1934_; lean_object* v___x_1935_; lean_object* v___x_1936_; lean_object* v___x_1937_; lean_object* v___x_1938_; lean_object* v___x_1939_; lean_object* v___x_1940_; lean_object* v___x_1941_; lean_object* v___x_1942_; lean_object* v___x_1943_; lean_object* v___x_1944_; lean_object* v___x_1945_; lean_object* v___x_1946_; lean_object* v___x_1947_; lean_object* v___x_1948_; lean_object* v___x_1949_; 
v___x_1930_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__17, &lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__17_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__17);
v___x_1931_ = l_Nat_reprFast(v___x_1914_);
v___x_1932_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1932_, 0, v___x_1931_);
v___x_1933_ = l_Lean_MessageData_ofFormat(v___x_1932_);
v___x_1934_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1934_, 0, v___x_1930_);
lean_ctor_set(v___x_1934_, 1, v___x_1933_);
v___x_1935_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__19, &lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__19_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__19);
v___x_1936_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1936_, 0, v___x_1934_);
lean_ctor_set(v___x_1936_, 1, v___x_1935_);
v___x_1937_ = l_Nat_reprFast(v___y_1922_);
v___x_1938_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1938_, 0, v___x_1937_);
v___x_1939_ = l_Lean_MessageData_ofFormat(v___x_1938_);
v___x_1940_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1940_, 0, v___x_1936_);
lean_ctor_set(v___x_1940_, 1, v___x_1939_);
v___x_1941_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__21, &lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__21_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__21);
v___x_1942_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1942_, 0, v___x_1940_);
lean_ctor_set(v___x_1942_, 1, v___x_1941_);
v___x_1943_ = l_Nat_reprFast(v___x_1855_);
v___x_1944_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1944_, 0, v___x_1943_);
v___x_1945_ = l_Lean_MessageData_ofFormat(v___x_1944_);
v___x_1946_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1946_, 0, v___x_1942_);
lean_ctor_set(v___x_1946_, 1, v___x_1945_);
v___x_1947_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__9, &lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__9_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__9);
v___x_1948_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1948_, 0, v___x_1946_);
lean_ctor_set(v___x_1948_, 1, v___x_1947_);
v___x_1949_ = lp_mathlib_Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter_spec__2(v___x_1854_, v_stx_1842_, v___x_1948_, v___y_1843_, v___y_1844_);
lean_dec(v_stx_1842_);
return v___x_1949_;
}
}
v___jp_1950_:
{
uint8_t v___x_1951_; 
lean_inc(v_stx_1842_);
v___x_1951_ = l_Lean_Parser_isTerminalCommand(v_stx_1842_);
if (v___x_1951_ == 0)
{
lean_object* v___x_1952_; lean_object* v___x_1953_; 
lean_dec(v___x_1914_);
lean_dec(v___x_1855_);
lean_dec(v_stx_1842_);
v___x_1952_ = lean_box(0);
v___x_1953_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1953_, 0, v___x_1952_);
return v___x_1953_;
}
else
{
lean_object* v___x_1954_; lean_object* v_a_1955_; lean_object* v___x_1957_; uint8_t v_isShared_1958_; uint8_t v_isSharedCheck_1976_; 
v___x_1954_ = lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter_spec__1___redArg(v___y_1844_);
v_a_1955_ = lean_ctor_get(v___x_1954_, 0);
v_isSharedCheck_1976_ = !lean_is_exclusive(v___x_1954_);
if (v_isSharedCheck_1976_ == 0)
{
v___x_1957_ = v___x_1954_;
v_isShared_1958_ = v_isSharedCheck_1976_;
goto v_resetjp_1956_;
}
else
{
lean_inc(v_a_1955_);
lean_dec(v___x_1954_);
v___x_1957_ = lean_box(0);
v_isShared_1958_ = v_isSharedCheck_1976_;
goto v_resetjp_1956_;
}
v_resetjp_1956_:
{
lean_object* v___x_1959_; uint8_t v___x_1960_; 
v___x_1959_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__22));
v___x_1960_ = lean_name_eq(v_a_1955_, v___x_1959_);
lean_dec(v_a_1955_);
if (v___x_1960_ == 0)
{
lean_object* v___x_1961_; 
v___x_1961_ = l_Lean_Syntax_getTailPos_x3f(v_stx_1842_, v___x_1960_);
if (lean_obj_tag(v___x_1961_) == 1)
{
lean_object* v_val_1962_; lean_object* v_fileMap_1963_; lean_object* v___x_1964_; lean_object* v_line_1965_; uint8_t v___x_1966_; 
lean_del_object(v___x_1957_);
v_val_1962_ = lean_ctor_get(v___x_1961_, 0);
lean_inc(v_val_1962_);
lean_dec_ref_known(v___x_1961_, 1);
v_fileMap_1963_ = lean_ctor_get(v___y_1843_, 1);
lean_inc_ref(v_fileMap_1963_);
v___x_1964_ = l_Lean_FileMap_toPosition(v_fileMap_1963_, v_val_1962_);
lean_dec(v_val_1962_);
v_line_1965_ = lean_ctor_get(v___x_1964_, 0);
lean_inc(v_line_1965_);
lean_dec_ref(v___x_1964_);
v___x_1966_ = lean_nat_dec_le(v_line_1965_, v___x_1914_);
if (v___x_1966_ == 0)
{
v___y_1922_ = v_line_1965_;
v___y_1923_ = v___x_1966_;
goto v___jp_1921_;
}
else
{
uint8_t v___x_1967_; 
v___x_1967_ = lean_nat_dec_lt(v___x_1914_, v___x_1855_);
v___y_1922_ = v_line_1965_;
v___y_1923_ = v___x_1967_;
goto v___jp_1921_;
}
}
else
{
lean_object* v___x_1968_; lean_object* v___x_1970_; 
lean_dec(v___x_1961_);
lean_dec(v___x_1914_);
lean_dec(v___x_1855_);
lean_dec(v_stx_1842_);
v___x_1968_ = lean_box(0);
if (v_isShared_1958_ == 0)
{
lean_ctor_set(v___x_1957_, 0, v___x_1968_);
v___x_1970_ = v___x_1957_;
goto v_reusejp_1969_;
}
else
{
lean_object* v_reuseFailAlloc_1971_; 
v_reuseFailAlloc_1971_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1971_, 0, v___x_1968_);
v___x_1970_ = v_reuseFailAlloc_1971_;
goto v_reusejp_1969_;
}
v_reusejp_1969_:
{
return v___x_1970_;
}
}
}
else
{
lean_object* v___x_1972_; lean_object* v___x_1974_; 
lean_dec(v___x_1914_);
lean_dec(v___x_1855_);
lean_dec(v_stx_1842_);
v___x_1972_ = lean_box(0);
if (v_isShared_1958_ == 0)
{
lean_ctor_set(v___x_1957_, 0, v___x_1972_);
v___x_1974_ = v___x_1957_;
goto v_reusejp_1973_;
}
else
{
lean_object* v_reuseFailAlloc_1975_; 
v_reuseFailAlloc_1975_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1975_, 0, v___x_1972_);
v___x_1974_ = v_reuseFailAlloc_1975_;
goto v_reusejp_1973_;
}
v_reusejp_1973_:
{
return v___x_1974_;
}
}
}
}
}
}
else
{
lean_object* v___x_2007_; lean_object* v___x_2008_; 
lean_dec(v___x_1855_);
lean_dec(v_stx_1842_);
v___x_2007_ = lean_box(0);
v___x_2008_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2008_, 0, v___x_2007_);
return v___x_2008_;
}
v___jp_1846_:
{
lean_object* v___x_1847_; lean_object* v___x_1848_; 
v___x_1847_ = lean_box(0);
v___x_1848_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1848_, 0, v___x_1847_);
return v___x_1848_;
}
v___jp_1856_:
{
if (v___y_1860_ == 0)
{
uint8_t v___x_1861_; 
v___x_1861_ = lean_nat_dec_eq(v___x_1855_, v___y_1859_);
if (v___x_1861_ == 0)
{
lean_object* v___x_1862_; uint8_t v___x_1863_; 
v___x_1862_ = lean_nat_add(v___x_1855_, v___y_1858_);
v___x_1863_ = lean_nat_dec_eq(v___x_1862_, v___y_1859_);
lean_dec(v___x_1862_);
if (v___x_1863_ == 0)
{
lean_object* v___x_1864_; lean_object* v___x_1865_; lean_object* v___x_1866_; lean_object* v___x_1867_; lean_object* v___x_1868_; lean_object* v___x_1869_; lean_object* v___x_1870_; lean_object* v___x_1871_; lean_object* v___x_1872_; lean_object* v___x_1873_; lean_object* v___x_1874_; lean_object* v___x_1875_; lean_object* v___x_1876_; lean_object* v___x_1877_; lean_object* v___x_1878_; lean_object* v___x_1879_; lean_object* v___x_1880_; lean_object* v___x_1881_; lean_object* v___x_1882_; lean_object* v___x_1883_; lean_object* v___x_1884_; lean_object* v___x_1885_; lean_object* v___x_1886_; 
v___x_1864_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__1, &lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__1);
v___x_1865_ = l_Nat_reprFast(v___y_1857_);
v___x_1866_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1866_, 0, v___x_1865_);
v___x_1867_ = l_Lean_MessageData_ofFormat(v___x_1866_);
v___x_1868_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1868_, 0, v___x_1864_);
lean_ctor_set(v___x_1868_, 1, v___x_1867_);
v___x_1869_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__3, &lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__3);
v___x_1870_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1870_, 0, v___x_1868_);
lean_ctor_set(v___x_1870_, 1, v___x_1869_);
v___x_1871_ = l_Nat_reprFast(v___x_1855_);
v___x_1872_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1872_, 0, v___x_1871_);
v___x_1873_ = l_Lean_MessageData_ofFormat(v___x_1872_);
v___x_1874_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1874_, 0, v___x_1870_);
lean_ctor_set(v___x_1874_, 1, v___x_1873_);
v___x_1875_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__5, &lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__5_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__5);
v___x_1876_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1876_, 0, v___x_1874_);
lean_ctor_set(v___x_1876_, 1, v___x_1875_);
v___x_1877_ = l_Nat_reprFast(v___y_1859_);
v___x_1878_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1878_, 0, v___x_1877_);
v___x_1879_ = l_Lean_MessageData_ofFormat(v___x_1878_);
lean_inc_ref(v___x_1879_);
v___x_1880_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1880_, 0, v___x_1876_);
lean_ctor_set(v___x_1880_, 1, v___x_1879_);
v___x_1881_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__7, &lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__7_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__7);
v___x_1882_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1882_, 0, v___x_1880_);
lean_ctor_set(v___x_1882_, 1, v___x_1881_);
v___x_1883_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1883_, 0, v___x_1882_);
lean_ctor_set(v___x_1883_, 1, v___x_1879_);
v___x_1884_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__9, &lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__9_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__9);
v___x_1885_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1885_, 0, v___x_1883_);
lean_ctor_set(v___x_1885_, 1, v___x_1884_);
v___x_1886_ = lp_mathlib_Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter_spec__2(v___x_1854_, v_stx_1842_, v___x_1885_, v___y_1843_, v___y_1844_);
lean_dec(v_stx_1842_);
return v___x_1886_;
}
else
{
lean_dec(v___y_1859_);
lean_dec(v___y_1857_);
lean_dec(v___x_1855_);
lean_dec(v_stx_1842_);
goto v___jp_1846_;
}
}
else
{
lean_dec(v___y_1859_);
lean_dec(v___y_1857_);
lean_dec(v___x_1855_);
lean_dec(v_stx_1842_);
goto v___jp_1846_;
}
}
else
{
lean_object* v___x_1887_; lean_object* v___x_1888_; lean_object* v___x_1889_; lean_object* v___x_1890_; lean_object* v___x_1891_; lean_object* v___x_1892_; lean_object* v___x_1893_; lean_object* v___x_1894_; lean_object* v___x_1895_; lean_object* v___x_1896_; lean_object* v___x_1897_; lean_object* v___x_1898_; lean_object* v___x_1899_; lean_object* v___x_1900_; lean_object* v___x_1901_; lean_object* v___x_1902_; lean_object* v___x_1903_; lean_object* v___x_1904_; lean_object* v___x_1905_; lean_object* v___x_1906_; 
v___x_1887_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__1, &lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__1);
v___x_1888_ = l_Nat_reprFast(v___y_1857_);
v___x_1889_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1889_, 0, v___x_1888_);
v___x_1890_ = l_Lean_MessageData_ofFormat(v___x_1889_);
v___x_1891_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1891_, 0, v___x_1887_);
lean_ctor_set(v___x_1891_, 1, v___x_1890_);
v___x_1892_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__11, &lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__11_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__11);
v___x_1893_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1893_, 0, v___x_1891_);
lean_ctor_set(v___x_1893_, 1, v___x_1892_);
v___x_1894_ = l_Nat_reprFast(v___x_1855_);
v___x_1895_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1895_, 0, v___x_1894_);
v___x_1896_ = l_Lean_MessageData_ofFormat(v___x_1895_);
v___x_1897_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1897_, 0, v___x_1893_);
lean_ctor_set(v___x_1897_, 1, v___x_1896_);
v___x_1898_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__13, &lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__13_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__13);
v___x_1899_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1899_, 0, v___x_1897_);
lean_ctor_set(v___x_1899_, 1, v___x_1898_);
v___x_1900_ = l_Nat_reprFast(v___y_1859_);
v___x_1901_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1901_, 0, v___x_1900_);
v___x_1902_ = l_Lean_MessageData_ofFormat(v___x_1901_);
v___x_1903_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1903_, 0, v___x_1899_);
lean_ctor_set(v___x_1903_, 1, v___x_1902_);
v___x_1904_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__15, &lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__15_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___closed__15);
v___x_1905_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1905_, 0, v___x_1903_);
lean_ctor_set(v___x_1905_, 1, v___x_1904_);
v___x_1906_ = lp_mathlib_Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter_spec__2(v___x_1854_, v_stx_1842_, v___x_1905_, v___y_1843_, v___y_1844_);
lean_dec(v_stx_1842_);
return v___x_1906_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0___boxed(lean_object* v_stx_2009_, lean_object* v___y_2010_, lean_object* v___y_2011_, lean_object* v___y_2012_){
_start:
{
lean_object* v_res_2013_; 
v_res_2013_ = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter___lam__0(v_stx_2009_, v___y_2010_, v___y_2011_);
lean_dec(v___y_2011_);
lean_dec_ref(v___y_2010_);
return v_res_2013_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_initFn_00___x40_Mathlib_Tactic_Linter_Style_3199531762____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_2029_; lean_object* v___x_2030_; 
v___x_2029_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter));
v___x_2030_ = l_Lean_Elab_Command_addLinter(v___x_2029_);
return v___x_2030_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_initFn_00___x40_Mathlib_Tactic_Linter_Style_3199531762____hygCtx___hyg_2____boxed(lean_object* v_a_2031_){
_start:
{
lean_object* v_res_2032_; 
v_res_2032_ = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_initFn_00___x40_Mathlib_Tactic_Linter_Style_3199531762____hygCtx___hyg_2_();
return v_res_2032_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_1043171623____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_2051_; lean_object* v___x_2052_; lean_object* v___x_2053_; lean_object* v___x_2054_; 
v___x_2051_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_1043171623____hygCtx___hyg_4_));
v___x_2052_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Style_1043171623____hygCtx___hyg_4_));
v___x_2053_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_1043171623____hygCtx___hyg_4_));
v___x_2054_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__spec__0(v___x_2051_, v___x_2052_, v___x_2053_);
return v___x_2054_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_1043171623____hygCtx___hyg_4____boxed(lean_object* v_a_2055_){
_start:
{
lean_object* v_res_2056_; 
v_res_2056_ = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_1043171623____hygCtx___hyg_4_();
return v_res_2056_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_690386724____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_2076_; lean_object* v___x_2077_; lean_object* v___x_2078_; lean_object* v___x_2079_; 
v___x_2076_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_690386724____hygCtx___hyg_4_));
v___x_2077_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Style_690386724____hygCtx___hyg_4_));
v___x_2078_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_690386724____hygCtx___hyg_4_));
v___x_2079_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_695976056____hygCtx___hyg_4__spec__0(v___x_2076_, v___x_2077_, v___x_2078_);
return v___x_2079_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_690386724____hygCtx___hyg_4____boxed(lean_object* v_a_2080_){
_start:
{
lean_object* v_res_2081_; 
v_res_2081_ = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_690386724____hygCtx___hyg_4_();
return v_res_2081_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__1(void){
_start:
{
lean_object* v___x_2083_; lean_object* v___x_2084_; 
v___x_2083_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__0));
v___x_2084_ = lean_string_utf8_byte_size(v___x_2083_);
return v___x_2084_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__3(void){
_start:
{
lean_object* v___x_2086_; lean_object* v___x_2087_; 
v___x_2086_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__2));
v___x_2087_ = lean_string_utf8_byte_size(v___x_2086_);
return v___x_2087_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__5(void){
_start:
{
lean_object* v___x_2089_; lean_object* v___x_2090_; 
v___x_2089_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__4));
v___x_2090_ = lean_string_utf8_byte_size(v___x_2089_);
return v___x_2090_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__7(void){
_start:
{
lean_object* v___x_2092_; lean_object* v___x_2093_; 
v___x_2092_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__6));
v___x_2093_ = lean_string_utf8_byte_size(v___x_2092_);
return v___x_2093_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__9(void){
_start:
{
lean_object* v___x_2095_; lean_object* v___x_2096_; 
v___x_2095_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__8));
v___x_2096_ = lean_string_utf8_byte_size(v___x_2095_);
return v___x_2096_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__11(void){
_start:
{
lean_object* v___x_2098_; lean_object* v___x_2099_; 
v___x_2098_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__10));
v___x_2099_ = lean_string_utf8_byte_size(v___x_2098_);
return v___x_2099_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport(lean_object* v_s_2100_){
_start:
{
lean_object* v___x_2136_; lean_object* v___x_2137_; lean_object* v___x_2138_; uint8_t v___x_2139_; 
v___x_2136_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__10));
v___x_2137_ = lean_string_utf8_byte_size(v_s_2100_);
v___x_2138_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__11, &lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__11_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__11);
v___x_2139_ = lean_nat_dec_le(v___x_2138_, v___x_2137_);
if (v___x_2139_ == 0)
{
goto v___jp_2129_;
}
else
{
lean_object* v___x_2140_; uint8_t v___x_2141_; 
v___x_2140_ = lean_unsigned_to_nat(0u);
v___x_2141_ = lean_string_memcmp(v_s_2100_, v___x_2136_, v___x_2140_, v___x_2140_, v___x_2138_);
if (v___x_2141_ == 0)
{
goto v___jp_2129_;
}
else
{
return v___x_2141_;
}
}
v___jp_2101_:
{
lean_object* v___x_2102_; lean_object* v___x_2103_; lean_object* v___x_2104_; uint8_t v___x_2105_; 
v___x_2102_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__0));
v___x_2103_ = lean_string_utf8_byte_size(v_s_2100_);
v___x_2104_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__1, &lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__1);
v___x_2105_ = lean_nat_dec_le(v___x_2104_, v___x_2103_);
if (v___x_2105_ == 0)
{
return v___x_2105_;
}
else
{
lean_object* v___x_2106_; uint8_t v___x_2107_; 
v___x_2106_ = lean_unsigned_to_nat(0u);
v___x_2107_ = lean_string_memcmp(v_s_2100_, v___x_2102_, v___x_2106_, v___x_2106_, v___x_2104_);
return v___x_2107_;
}
}
v___jp_2108_:
{
lean_object* v___x_2109_; lean_object* v___x_2110_; lean_object* v___x_2111_; uint8_t v___x_2112_; 
v___x_2109_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__2));
v___x_2110_ = lean_string_utf8_byte_size(v_s_2100_);
v___x_2111_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__3, &lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__3);
v___x_2112_ = lean_nat_dec_le(v___x_2111_, v___x_2110_);
if (v___x_2112_ == 0)
{
goto v___jp_2101_;
}
else
{
lean_object* v___x_2113_; uint8_t v___x_2114_; 
v___x_2113_ = lean_unsigned_to_nat(0u);
v___x_2114_ = lean_string_memcmp(v_s_2100_, v___x_2109_, v___x_2113_, v___x_2113_, v___x_2111_);
if (v___x_2114_ == 0)
{
goto v___jp_2101_;
}
else
{
return v___x_2114_;
}
}
}
v___jp_2115_:
{
lean_object* v___x_2116_; lean_object* v___x_2117_; lean_object* v___x_2118_; uint8_t v___x_2119_; 
v___x_2116_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__4));
v___x_2117_ = lean_string_utf8_byte_size(v_s_2100_);
v___x_2118_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__5, &lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__5_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__5);
v___x_2119_ = lean_nat_dec_le(v___x_2118_, v___x_2117_);
if (v___x_2119_ == 0)
{
goto v___jp_2108_;
}
else
{
lean_object* v___x_2120_; uint8_t v___x_2121_; 
v___x_2120_ = lean_unsigned_to_nat(0u);
v___x_2121_ = lean_string_memcmp(v_s_2100_, v___x_2116_, v___x_2120_, v___x_2120_, v___x_2118_);
if (v___x_2121_ == 0)
{
goto v___jp_2108_;
}
else
{
return v___x_2121_;
}
}
}
v___jp_2122_:
{
lean_object* v___x_2123_; lean_object* v___x_2124_; lean_object* v___x_2125_; uint8_t v___x_2126_; 
v___x_2123_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__6));
v___x_2124_ = lean_string_utf8_byte_size(v_s_2100_);
v___x_2125_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__7, &lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__7_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__7);
v___x_2126_ = lean_nat_dec_le(v___x_2125_, v___x_2124_);
if (v___x_2126_ == 0)
{
goto v___jp_2115_;
}
else
{
lean_object* v___x_2127_; uint8_t v___x_2128_; 
v___x_2127_ = lean_unsigned_to_nat(0u);
v___x_2128_ = lean_string_memcmp(v_s_2100_, v___x_2123_, v___x_2127_, v___x_2127_, v___x_2125_);
if (v___x_2128_ == 0)
{
goto v___jp_2115_;
}
else
{
return v___x_2128_;
}
}
}
v___jp_2129_:
{
lean_object* v___x_2130_; lean_object* v___x_2131_; lean_object* v___x_2132_; uint8_t v___x_2133_; 
v___x_2130_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__8));
v___x_2131_ = lean_string_utf8_byte_size(v_s_2100_);
v___x_2132_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__9, &lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__9_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___closed__9);
v___x_2133_ = lean_nat_dec_le(v___x_2132_, v___x_2131_);
if (v___x_2133_ == 0)
{
goto v___jp_2122_;
}
else
{
lean_object* v___x_2134_; uint8_t v___x_2135_; 
v___x_2134_ = lean_unsigned_to_nat(0u);
v___x_2135_ = lean_string_memcmp(v_s_2100_, v___x_2130_, v___x_2134_, v___x_2134_, v___x_2132_);
if (v___x_2135_ == 0)
{
goto v___jp_2122_;
}
else
{
return v___x_2135_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport___boxed(lean_object* v_s_2142_){
_start:
{
uint8_t v_res_2143_; lean_object* v_r_2144_; 
v_res_2143_ = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport(v_s_2142_);
lean_dec_ref(v_s_2142_);
v_r_2144_ = lean_box(v_res_2143_);
return v_r_2144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_filterTR_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__0(lean_object* v___x_2145_, lean_object* v___x_2146_, lean_object* v_a_2147_, lean_object* v_a_2148_){
_start:
{
if (lean_obj_tag(v_a_2147_) == 0)
{
lean_object* v___x_2149_; 
lean_dec_ref(v___x_2145_);
v___x_2149_ = l_List_reverse___redArg(v_a_2148_);
return v___x_2149_;
}
else
{
lean_object* v_head_2150_; lean_object* v_tail_2151_; lean_object* v___x_2153_; uint8_t v_isShared_2154_; uint8_t v_isSharedCheck_2164_; 
v_head_2150_ = lean_ctor_get(v_a_2147_, 0);
v_tail_2151_ = lean_ctor_get(v_a_2147_, 1);
v_isSharedCheck_2164_ = !lean_is_exclusive(v_a_2147_);
if (v_isSharedCheck_2164_ == 0)
{
v___x_2153_ = v_a_2147_;
v_isShared_2154_ = v_isSharedCheck_2164_;
goto v_resetjp_2152_;
}
else
{
lean_inc(v_tail_2151_);
lean_inc(v_head_2150_);
lean_dec(v_a_2147_);
v___x_2153_ = lean_box(0);
v_isShared_2154_ = v_isSharedCheck_2164_;
goto v_resetjp_2152_;
}
v_resetjp_2152_:
{
lean_object* v_stopPos_2155_; lean_object* v___x_2156_; lean_object* v_column_2157_; uint8_t v___x_2158_; 
v_stopPos_2155_ = lean_ctor_get(v_head_2150_, 2);
lean_inc_ref(v___x_2145_);
v___x_2156_ = l_Lean_FileMap_toPosition(v___x_2145_, v_stopPos_2155_);
v_column_2157_ = lean_ctor_get(v___x_2156_, 1);
lean_inc(v_column_2157_);
lean_dec_ref(v___x_2156_);
v___x_2158_ = lean_nat_dec_lt(v___x_2146_, v_column_2157_);
lean_dec(v_column_2157_);
if (v___x_2158_ == 0)
{
lean_del_object(v___x_2153_);
lean_dec(v_head_2150_);
v_a_2147_ = v_tail_2151_;
goto _start;
}
else
{
lean_object* v___x_2161_; 
if (v_isShared_2154_ == 0)
{
lean_ctor_set(v___x_2153_, 1, v_a_2148_);
v___x_2161_ = v___x_2153_;
goto v_reusejp_2160_;
}
else
{
lean_object* v_reuseFailAlloc_2163_; 
v_reuseFailAlloc_2163_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2163_, 0, v_head_2150_);
lean_ctor_set(v_reuseFailAlloc_2163_, 1, v_a_2148_);
v___x_2161_ = v_reuseFailAlloc_2163_;
goto v_reusejp_2160_;
}
v_reusejp_2160_:
{
v_a_2147_ = v_tail_2151_;
v_a_2148_ = v___x_2161_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_filterTR_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__0___boxed(lean_object* v___x_2165_, lean_object* v___x_2166_, lean_object* v_a_2167_, lean_object* v_a_2168_){
_start:
{
lean_object* v_res_2169_; 
v_res_2169_ = lp_mathlib_List_filterTR_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__0(v___x_2165_, v___x_2166_, v_a_2167_, v_a_2168_);
lean_dec(v___x_2166_);
return v_res_2169_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__1_spec__1___redArg(lean_object* v_s_2170_, lean_object* v_a_2171_, uint8_t v_b_2172_){
_start:
{
lean_object* v_str_2173_; lean_object* v_startInclusive_2174_; lean_object* v_endExclusive_2175_; lean_object* v___x_2176_; uint8_t v___x_2177_; 
v_str_2173_ = lean_ctor_get(v_s_2170_, 0);
v_startInclusive_2174_ = lean_ctor_get(v_s_2170_, 1);
v_endExclusive_2175_ = lean_ctor_get(v_s_2170_, 2);
v___x_2176_ = lean_nat_sub(v_endExclusive_2175_, v_startInclusive_2174_);
v___x_2177_ = lean_nat_dec_eq(v_a_2171_, v___x_2176_);
lean_dec(v___x_2176_);
if (v___x_2177_ == 0)
{
uint32_t v___x_2178_; lean_object* v___x_2179_; uint32_t v___x_2180_; uint8_t v___x_2181_; 
v___x_2178_ = 34;
v___x_2179_ = lean_nat_add(v_startInclusive_2174_, v_a_2171_);
lean_dec(v_a_2171_);
v___x_2180_ = lean_string_utf8_get_fast(v_str_2173_, v___x_2179_);
v___x_2181_ = lean_uint32_dec_eq(v___x_2180_, v___x_2178_);
if (v___x_2181_ == 0)
{
lean_object* v___x_2182_; lean_object* v___x_2183_; 
v___x_2182_ = lean_string_utf8_next_fast(v_str_2173_, v___x_2179_);
lean_dec(v___x_2179_);
v___x_2183_ = lean_nat_sub(v___x_2182_, v_startInclusive_2174_);
v_a_2171_ = v___x_2183_;
v_b_2172_ = v___x_2181_;
goto _start;
}
else
{
lean_dec(v___x_2179_);
return v___x_2181_;
}
}
else
{
lean_dec(v_a_2171_);
return v_b_2172_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__1_spec__1___redArg___boxed(lean_object* v_s_2185_, lean_object* v_a_2186_, lean_object* v_b_2187_){
_start:
{
uint8_t v_b_boxed_2188_; uint8_t v_res_2189_; lean_object* v_r_2190_; 
v_b_boxed_2188_ = lean_unbox(v_b_2187_);
v_res_2189_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__1_spec__1___redArg(v_s_2185_, v_a_2186_, v_b_boxed_2188_);
lean_dec_ref(v_s_2185_);
v_r_2190_ = lean_box(v_res_2189_);
return v_r_2190_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__1(lean_object* v_s_2191_){
_start:
{
lean_object* v_searcher_2192_; uint8_t v___x_2193_; uint8_t v___x_2194_; 
v_searcher_2192_ = lean_unsigned_to_nat(0u);
v___x_2193_ = 0;
v___x_2194_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__1_spec__1___redArg(v_s_2191_, v_searcher_2192_, v___x_2193_);
return v___x_2194_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__1___boxed(lean_object* v_s_2195_){
_start:
{
uint8_t v_res_2196_; lean_object* v_r_2197_; 
v_res_2196_ = lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__1(v_s_2195_);
lean_dec_ref(v_s_2195_);
v_r_2197_ = lean_box(v_res_2196_);
return v_r_2197_;
}
}
static lean_object* _init_lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__2___redArg___closed__1(void){
_start:
{
lean_object* v___x_2199_; lean_object* v___x_2200_; 
v___x_2199_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__2___redArg___closed__0));
v___x_2200_ = l_Lean_stringToMessageData(v___x_2199_);
return v___x_2200_;
}
}
static lean_object* _init_lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__2___redArg___closed__3(void){
_start:
{
lean_object* v___x_2202_; lean_object* v___x_2203_; 
v___x_2202_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__2___redArg___closed__2));
v___x_2203_ = l_Lean_stringToMessageData(v___x_2202_);
return v___x_2203_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__2___redArg(lean_object* v___x_2206_, uint8_t v___x_2207_, uint8_t v___x_2208_, lean_object* v_as_x27_2209_, lean_object* v_b_2210_, lean_object* v___y_2211_, lean_object* v___y_2212_){
_start:
{
if (lean_obj_tag(v_as_x27_2209_) == 0)
{
lean_object* v___x_2214_; 
lean_dec(v___x_2206_);
v___x_2214_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2214_, 0, v_b_2210_);
return v___x_2214_;
}
else
{
lean_object* v_head_2215_; lean_object* v_tail_2216_; lean_object* v___x_2217_; lean_object* v___x_2218_; lean_object* v___y_2220_; uint8_t v___y_2240_; uint8_t v___y_2248_; lean_object* v___x_2258_; lean_object* v___x_2259_; lean_object* v___x_2260_; lean_object* v___x_2261_; uint8_t v___x_2262_; 
v_head_2215_ = lean_ctor_get(v_as_x27_2209_, 0);
v_tail_2216_ = lean_ctor_get(v_as_x27_2209_, 1);
v___x_2217_ = lp_mathlib_Mathlib_Linter_linter_style_longLine;
v___x_2218_ = lean_box(0);
v___x_2258_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__2___redArg___closed__5));
lean_inc(v_head_2215_);
v___x_2259_ = l_Substring_Raw_splitOn(v_head_2215_, v___x_2258_);
v___x_2260_ = l_List_lengthTR___redArg(v___x_2259_);
lean_dec(v___x_2259_);
v___x_2261_ = lean_unsigned_to_nat(1u);
v___x_2262_ = lean_nat_dec_le(v___x_2260_, v___x_2261_);
lean_dec(v___x_2260_);
if (v___x_2262_ == 0)
{
v___y_2248_ = v___x_2262_;
goto v___jp_2247_;
}
else
{
lean_object* v_str_2263_; lean_object* v_startPos_2264_; lean_object* v_stopPos_2265_; lean_object* v___x_2266_; uint8_t v___x_2267_; 
v_str_2263_ = lean_ctor_get(v_head_2215_, 0);
v_startPos_2264_ = lean_ctor_get(v_head_2215_, 1);
v_stopPos_2265_ = lean_ctor_get(v_head_2215_, 2);
v___x_2266_ = lean_string_utf8_extract(v_str_2263_, v_startPos_2264_, v_stopPos_2265_);
v___x_2267_ = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_isImport(v___x_2266_);
lean_dec_ref(v___x_2266_);
if (v___x_2267_ == 0)
{
v___y_2248_ = v___x_2262_;
goto v___jp_2247_;
}
else
{
v___y_2248_ = v___x_2208_;
goto v___jp_2247_;
}
}
v___jp_2219_:
{
lean_object* v_startPos_2221_; lean_object* v_stopPos_2222_; lean_object* v___x_2223_; lean_object* v___x_2224_; lean_object* v___x_2225_; lean_object* v___x_2226_; lean_object* v___x_2227_; lean_object* v___x_2228_; lean_object* v___x_2229_; lean_object* v___x_2230_; lean_object* v___x_2231_; lean_object* v___x_2232_; lean_object* v___x_2233_; lean_object* v___x_2234_; lean_object* v___x_2235_; lean_object* v___x_2236_; lean_object* v___x_2237_; 
v_startPos_2221_ = lean_ctor_get(v_head_2215_, 1);
v_stopPos_2222_ = lean_ctor_get(v_head_2215_, 2);
v___x_2223_ = lean_unsigned_to_nat(0u);
lean_inc_n(v___x_2206_, 2);
v___x_2224_ = l_Substring_Raw_nextn(v_head_2215_, v___x_2206_, v___x_2223_);
v___x_2225_ = lean_nat_add(v_startPos_2221_, v___x_2224_);
lean_dec(v___x_2224_);
lean_inc(v_stopPos_2222_);
v___x_2226_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2226_, 0, v___x_2225_);
lean_ctor_set(v___x_2226_, 1, v_stopPos_2222_);
v___x_2227_ = l_Lean_Syntax_ofRange(v___x_2226_, v___x_2207_);
v___x_2228_ = lean_obj_once(&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__2___redArg___closed__1, &lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__2___redArg___closed__1_once, _init_lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__2___redArg___closed__1);
v___x_2229_ = l_Nat_reprFast(v___x_2206_);
v___x_2230_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2230_, 0, v___x_2229_);
v___x_2231_ = l_Lean_MessageData_ofFormat(v___x_2230_);
v___x_2232_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2232_, 0, v___x_2228_);
lean_ctor_set(v___x_2232_, 1, v___x_2231_);
v___x_2233_ = lean_obj_once(&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__2___redArg___closed__3, &lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__2___redArg___closed__3_once, _init_lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__2___redArg___closed__3);
v___x_2234_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2234_, 0, v___x_2232_);
lean_ctor_set(v___x_2234_, 1, v___x_2233_);
lean_inc_ref(v___y_2220_);
v___x_2235_ = l_Lean_stringToMessageData(v___y_2220_);
v___x_2236_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2236_, 0, v___x_2234_);
lean_ctor_set(v___x_2236_, 1, v___x_2235_);
v___x_2237_ = lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__1(v___x_2217_, v___x_2227_, v___x_2236_, v___y_2211_, v___y_2212_);
if (lean_obj_tag(v___x_2237_) == 0)
{
lean_dec_ref_known(v___x_2237_, 1);
v_as_x27_2209_ = v_tail_2216_;
v_b_2210_ = v___x_2218_;
goto _start;
}
else
{
lean_dec(v___x_2206_);
return v___x_2237_;
}
}
v___jp_2239_:
{
if (v___y_2240_ == 0)
{
lean_object* v___x_2241_; 
v___x_2241_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4___closed__0));
v___y_2220_ = v___x_2241_;
goto v___jp_2219_;
}
else
{
lean_object* v___x_2242_; 
v___x_2242_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__2___redArg___closed__4));
v___y_2220_ = v___x_2242_;
goto v___jp_2219_;
}
}
v___jp_2243_:
{
lean_object* v___x_2244_; lean_object* v___x_2245_; uint8_t v___x_2246_; 
v___x_2244_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___closed__7, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___closed__7_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___closed__7);
v___x_2245_ = lp_mathlib_panic___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__1(v___x_2244_);
v___x_2246_ = lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__1(v___x_2245_);
lean_dec_ref(v___x_2245_);
v___y_2240_ = v___x_2246_;
goto v___jp_2239_;
}
v___jp_2247_:
{
if (v___y_2248_ == 0)
{
v_as_x27_2209_ = v_tail_2216_;
v_b_2210_ = v___x_2218_;
goto _start;
}
else
{
lean_object* v_str_2250_; lean_object* v_startPos_2251_; lean_object* v_stopPos_2252_; uint8_t v___x_2253_; 
v_str_2250_ = lean_ctor_get(v_head_2215_, 0);
v_startPos_2251_ = lean_ctor_get(v_head_2215_, 1);
v_stopPos_2252_ = lean_ctor_get(v_head_2215_, 2);
v___x_2253_ = lean_string_is_valid_pos(v_str_2250_, v_startPos_2251_);
if (v___x_2253_ == 0)
{
goto v___jp_2243_;
}
else
{
uint8_t v___x_2254_; 
v___x_2254_ = lean_string_is_valid_pos(v_str_2250_, v_stopPos_2252_);
if (v___x_2254_ == 0)
{
goto v___jp_2243_;
}
else
{
uint8_t v___x_2255_; 
v___x_2255_ = lean_nat_dec_le(v_startPos_2251_, v_stopPos_2252_);
if (v___x_2255_ == 0)
{
goto v___jp_2243_;
}
else
{
lean_object* v___x_2256_; uint8_t v___x_2257_; 
lean_inc(v_stopPos_2252_);
lean_inc(v_startPos_2251_);
lean_inc_ref(v_str_2250_);
v___x_2256_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2256_, 0, v_str_2250_);
lean_ctor_set(v___x_2256_, 1, v_startPos_2251_);
lean_ctor_set(v___x_2256_, 2, v_stopPos_2252_);
v___x_2257_ = lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__1(v___x_2256_);
lean_dec_ref_known(v___x_2256_, 3);
v___y_2240_ = v___x_2257_;
goto v___jp_2239_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__2___redArg___boxed(lean_object* v___x_2268_, lean_object* v___x_2269_, lean_object* v___x_2270_, lean_object* v_as_x27_2271_, lean_object* v_b_2272_, lean_object* v___y_2273_, lean_object* v___y_2274_, lean_object* v___y_2275_){
_start:
{
uint8_t v___x_7054__boxed_2276_; uint8_t v___x_7055__boxed_2277_; lean_object* v_res_2278_; 
v___x_7054__boxed_2276_ = lean_unbox(v___x_2269_);
v___x_7055__boxed_2277_ = lean_unbox(v___x_2270_);
v_res_2278_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__2___redArg(v___x_2268_, v___x_7054__boxed_2276_, v___x_7055__boxed_2277_, v_as_x27_2271_, v_b_2272_, v___y_2273_, v___y_2274_);
lean_dec(v___y_2274_);
lean_dec_ref(v___y_2273_);
lean_dec(v_as_x27_2271_);
return v_res_2278_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___lam__0(lean_object* v_stx_2294_, lean_object* v___y_2295_, lean_object* v___y_2296_){
_start:
{
lean_object* v___x_2298_; lean_object* v_a_2299_; lean_object* v___x_2301_; uint8_t v_isShared_2302_; uint8_t v_isSharedCheck_2386_; 
v___x_2298_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__0(v___y_2295_, v___y_2296_);
v_a_2299_ = lean_ctor_get(v___x_2298_, 0);
v_isSharedCheck_2386_ = !lean_is_exclusive(v___x_2298_);
if (v_isSharedCheck_2386_ == 0)
{
v___x_2301_ = v___x_2298_;
v_isShared_2302_ = v_isSharedCheck_2386_;
goto v_resetjp_2300_;
}
else
{
lean_inc(v_a_2299_);
lean_dec(v___x_2298_);
v___x_2301_ = lean_box(0);
v_isShared_2302_ = v_isSharedCheck_2386_;
goto v_resetjp_2300_;
}
v_resetjp_2300_:
{
lean_object* v___x_2303_; uint8_t v___x_2304_; 
v___x_2303_ = lp_mathlib_Mathlib_Linter_linter_style_longLine;
v___x_2304_ = l_Lean_Linter_getLinterValue(v___x_2303_, v_a_2299_);
lean_dec(v_a_2299_);
if (v___x_2304_ == 0)
{
lean_object* v___x_2305_; lean_object* v___x_2307_; 
lean_dec(v_stx_2294_);
v___x_2305_ = lean_box(0);
if (v_isShared_2302_ == 0)
{
lean_ctor_set(v___x_2301_, 0, v___x_2305_);
v___x_2307_ = v___x_2301_;
goto v_reusejp_2306_;
}
else
{
lean_object* v_reuseFailAlloc_2308_; 
v_reuseFailAlloc_2308_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2308_, 0, v___x_2305_);
v___x_2307_ = v_reuseFailAlloc_2308_;
goto v_reusejp_2306_;
}
v_reusejp_2306_:
{
return v___x_2307_;
}
}
else
{
lean_object* v___x_2309_; lean_object* v_messages_2310_; uint8_t v___x_2311_; 
v___x_2309_ = lean_st_ref_get(v___y_2296_);
v_messages_2310_ = lean_ctor_get(v___x_2309_, 1);
lean_inc_ref(v_messages_2310_);
lean_dec(v___x_2309_);
v___x_2311_ = l_Lean_MessageLog_hasErrors(v_messages_2310_);
lean_dec_ref(v_messages_2310_);
if (v___x_2311_ == 0)
{
lean_object* v___x_2312_; uint8_t v___x_2313_; 
v___x_2312_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___lam__0___closed__1));
lean_inc(v_stx_2294_);
v___x_2313_ = l_Lean_Syntax_isOfKind(v_stx_2294_, v___x_2312_);
if (v___x_2313_ == 0)
{
lean_object* v___x_2314_; uint8_t v___x_2315_; lean_object* v___y_2317_; lean_object* v___y_2318_; lean_object* v___y_2319_; lean_object* v___y_2320_; lean_object* v___y_2321_; lean_object* v_stx_2337_; lean_object* v___y_2338_; lean_object* v___y_2339_; 
v___x_2314_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___lam__0___closed__4));
lean_inc(v_stx_2294_);
v___x_2315_ = l_Lean_Syntax_isOfKind(v_stx_2294_, v___x_2314_);
if (v___x_2315_ == 0)
{
lean_object* v___x_2351_; uint8_t v___x_2352_; 
lean_del_object(v___x_2301_);
v___x_2351_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_missingEndLinter___lam__0___closed__1));
lean_inc(v_stx_2294_);
v___x_2352_ = l_Lean_Syntax_isOfKind(v_stx_2294_, v___x_2351_);
if (v___x_2352_ == 0)
{
v_stx_2337_ = v_stx_2294_;
v___y_2338_ = v___y_2295_;
v___y_2339_ = v___y_2296_;
goto v___jp_2336_;
}
else
{
lean_object* v_fileMap_2353_; lean_object* v_fileName_2354_; lean_object* v_ref_2355_; lean_object* v_source_2356_; lean_object* v___x_2357_; lean_object* v___x_2358_; lean_object* v___x_2359_; 
lean_dec(v_stx_2294_);
v_fileMap_2353_ = lean_ctor_get(v___y_2295_, 1);
v_fileName_2354_ = lean_ctor_get(v___y_2295_, 0);
v_ref_2355_ = lean_ctor_get(v___y_2295_, 7);
v_source_2356_ = lean_ctor_get(v_fileMap_2353_, 0);
v___x_2357_ = lean_string_utf8_byte_size(v_source_2356_);
lean_inc_ref(v_fileMap_2353_);
lean_inc_ref(v_fileName_2354_);
lean_inc_ref(v_source_2356_);
v___x_2358_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_2358_, 0, v_source_2356_);
lean_ctor_set(v___x_2358_, 1, v_fileName_2354_);
lean_ctor_set(v___x_2358_, 2, v_fileMap_2353_);
lean_ctor_set(v___x_2358_, 3, v___x_2357_);
v___x_2359_ = l_Lean_Parser_parseHeader(v___x_2358_);
if (lean_obj_tag(v___x_2359_) == 0)
{
lean_object* v_a_2360_; lean_object* v_fst_2361_; 
v_a_2360_ = lean_ctor_get(v___x_2359_, 0);
lean_inc(v_a_2360_);
lean_dec_ref_known(v___x_2359_, 1);
v_fst_2361_ = lean_ctor_get(v_a_2360_, 0);
lean_inc(v_fst_2361_);
lean_dec(v_a_2360_);
v_stx_2337_ = v_fst_2361_;
v___y_2338_ = v___y_2295_;
v___y_2339_ = v___y_2296_;
goto v___jp_2336_;
}
else
{
lean_object* v_a_2362_; lean_object* v___x_2364_; uint8_t v_isShared_2365_; uint8_t v_isSharedCheck_2373_; 
v_a_2362_ = lean_ctor_get(v___x_2359_, 0);
v_isSharedCheck_2373_ = !lean_is_exclusive(v___x_2359_);
if (v_isSharedCheck_2373_ == 0)
{
v___x_2364_ = v___x_2359_;
v_isShared_2365_ = v_isSharedCheck_2373_;
goto v_resetjp_2363_;
}
else
{
lean_inc(v_a_2362_);
lean_dec(v___x_2359_);
v___x_2364_ = lean_box(0);
v_isShared_2365_ = v_isSharedCheck_2373_;
goto v_resetjp_2363_;
}
v_resetjp_2363_:
{
lean_object* v___x_2366_; lean_object* v___x_2367_; lean_object* v___x_2368_; lean_object* v___x_2369_; lean_object* v___x_2371_; 
v___x_2366_ = lean_io_error_to_string(v_a_2362_);
v___x_2367_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2367_, 0, v___x_2366_);
v___x_2368_ = l_Lean_MessageData_ofFormat(v___x_2367_);
lean_inc(v_ref_2355_);
v___x_2369_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2369_, 0, v_ref_2355_);
lean_ctor_set(v___x_2369_, 1, v___x_2368_);
if (v_isShared_2365_ == 0)
{
lean_ctor_set(v___x_2364_, 0, v___x_2369_);
v___x_2371_ = v___x_2364_;
goto v_reusejp_2370_;
}
else
{
lean_object* v_reuseFailAlloc_2372_; 
v_reuseFailAlloc_2372_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2372_, 0, v___x_2369_);
v___x_2371_ = v_reuseFailAlloc_2372_;
goto v_reusejp_2370_;
}
v_reusejp_2370_:
{
return v___x_2371_;
}
}
}
}
}
else
{
lean_object* v___x_2374_; lean_object* v___x_2376_; 
lean_dec(v_stx_2294_);
v___x_2374_ = lean_box(0);
if (v_isShared_2302_ == 0)
{
lean_ctor_set(v___x_2301_, 0, v___x_2374_);
v___x_2376_ = v___x_2301_;
goto v_reusejp_2375_;
}
else
{
lean_object* v_reuseFailAlloc_2377_; 
v_reuseFailAlloc_2377_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2377_, 0, v___x_2374_);
v___x_2376_ = v_reuseFailAlloc_2377_;
goto v_reusejp_2375_;
}
v_reusejp_2375_:
{
return v___x_2376_;
}
}
v___jp_2316_:
{
lean_object* v___x_2322_; lean_object* v___x_2323_; lean_object* v___x_2324_; lean_object* v___x_2325_; lean_object* v___x_2326_; lean_object* v___x_2327_; 
v___x_2322_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___lam__0___closed__5));
v___x_2323_ = l_Substring_Raw_splitOn(v___y_2321_, v___x_2322_);
v___x_2324_ = lean_box(0);
lean_inc_ref(v___y_2319_);
v___x_2325_ = lp_mathlib_List_filterTR_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__0(v___y_2319_, v___y_2318_, v___x_2323_, v___x_2324_);
v___x_2326_ = lean_box(0);
v___x_2327_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__2___redArg(v___y_2318_, v___x_2304_, v___x_2315_, v___x_2325_, v___x_2326_, v___y_2317_, v___y_2320_);
lean_dec(v___x_2325_);
if (lean_obj_tag(v___x_2327_) == 0)
{
lean_object* v___x_2329_; uint8_t v_isShared_2330_; uint8_t v_isSharedCheck_2334_; 
v_isSharedCheck_2334_ = !lean_is_exclusive(v___x_2327_);
if (v_isSharedCheck_2334_ == 0)
{
lean_object* v_unused_2335_; 
v_unused_2335_ = lean_ctor_get(v___x_2327_, 0);
lean_dec(v_unused_2335_);
v___x_2329_ = v___x_2327_;
v_isShared_2330_ = v_isSharedCheck_2334_;
goto v_resetjp_2328_;
}
else
{
lean_dec(v___x_2327_);
v___x_2329_ = lean_box(0);
v_isShared_2330_ = v_isSharedCheck_2334_;
goto v_resetjp_2328_;
}
v_resetjp_2328_:
{
lean_object* v___x_2332_; 
if (v_isShared_2330_ == 0)
{
lean_ctor_set(v___x_2329_, 0, v___x_2326_);
v___x_2332_ = v___x_2329_;
goto v_reusejp_2331_;
}
else
{
lean_object* v_reuseFailAlloc_2333_; 
v_reuseFailAlloc_2333_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2333_, 0, v___x_2326_);
v___x_2332_ = v_reuseFailAlloc_2333_;
goto v_reusejp_2331_;
}
v_reusejp_2331_:
{
return v___x_2332_;
}
}
}
else
{
return v___x_2327_;
}
}
v___jp_2336_:
{
lean_object* v___x_2340_; lean_object* v___x_2341_; lean_object* v_fileMap_2342_; lean_object* v_scopes_2343_; lean_object* v___x_2344_; lean_object* v___x_2345_; lean_object* v_opts_2346_; lean_object* v___x_2347_; lean_object* v___x_2348_; 
v___x_2340_ = l_Lean_Syntax_getSubstring_x3f(v_stx_2337_, v___x_2304_, v___x_2304_);
lean_dec(v_stx_2337_);
v___x_2341_ = lean_st_ref_get(v___y_2339_);
v_fileMap_2342_ = lean_ctor_get(v___y_2338_, 1);
v_scopes_2343_ = lean_ctor_get(v___x_2341_, 2);
lean_inc(v_scopes_2343_);
lean_dec(v___x_2341_);
v___x_2344_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_2345_ = l_List_head_x21___redArg(v___x_2344_, v_scopes_2343_);
lean_dec(v_scopes_2343_);
v_opts_2346_ = lean_ctor_get(v___x_2345_, 1);
lean_inc_ref(v_opts_2346_);
lean_dec(v___x_2345_);
v___x_2347_ = lp_mathlib_Mathlib_Linter_linter_style_longLine_maxLineLength;
v___x_2348_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_longFileLinter_spec__0(v_opts_2346_, v___x_2347_);
lean_dec_ref(v_opts_2346_);
if (lean_obj_tag(v___x_2340_) == 0)
{
lean_object* v___x_2349_; 
v___x_2349_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___lam__0___closed__6));
v___y_2317_ = v___y_2338_;
v___y_2318_ = v___x_2348_;
v___y_2319_ = v_fileMap_2342_;
v___y_2320_ = v___y_2339_;
v___y_2321_ = v___x_2349_;
goto v___jp_2316_;
}
else
{
lean_object* v_val_2350_; 
v_val_2350_ = lean_ctor_get(v___x_2340_, 0);
lean_inc(v_val_2350_);
lean_dec_ref_known(v___x_2340_, 1);
v___y_2317_ = v___y_2338_;
v___y_2318_ = v___x_2348_;
v___y_2319_ = v_fileMap_2342_;
v___y_2320_ = v___y_2339_;
v___y_2321_ = v_val_2350_;
goto v___jp_2316_;
}
}
}
else
{
lean_object* v___x_2378_; lean_object* v___x_2380_; 
lean_dec(v_stx_2294_);
v___x_2378_ = lean_box(0);
if (v_isShared_2302_ == 0)
{
lean_ctor_set(v___x_2301_, 0, v___x_2378_);
v___x_2380_ = v___x_2301_;
goto v_reusejp_2379_;
}
else
{
lean_object* v_reuseFailAlloc_2381_; 
v_reuseFailAlloc_2381_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2381_, 0, v___x_2378_);
v___x_2380_ = v_reuseFailAlloc_2381_;
goto v_reusejp_2379_;
}
v_reusejp_2379_:
{
return v___x_2380_;
}
}
}
else
{
lean_object* v___x_2382_; lean_object* v___x_2384_; 
lean_dec(v_stx_2294_);
v___x_2382_ = lean_box(0);
if (v_isShared_2302_ == 0)
{
lean_ctor_set(v___x_2301_, 0, v___x_2382_);
v___x_2384_ = v___x_2301_;
goto v_reusejp_2383_;
}
else
{
lean_object* v_reuseFailAlloc_2385_; 
v_reuseFailAlloc_2385_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2385_, 0, v___x_2382_);
v___x_2384_ = v_reuseFailAlloc_2385_;
goto v_reusejp_2383_;
}
v_reusejp_2383_:
{
return v___x_2384_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___lam__0___boxed(lean_object* v_stx_2387_, lean_object* v___y_2388_, lean_object* v___y_2389_, lean_object* v___y_2390_){
_start:
{
lean_object* v_res_2391_; 
v_res_2391_ = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter___lam__0(v_stx_2387_, v___y_2388_, v___y_2389_);
lean_dec(v___y_2389_);
lean_dec_ref(v___y_2388_);
return v_res_2391_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__2(lean_object* v___x_2406_, uint8_t v___x_2407_, uint8_t v___x_2408_, lean_object* v_as_2409_, lean_object* v_as_x27_2410_, lean_object* v_b_2411_, lean_object* v_a_2412_, lean_object* v___y_2413_, lean_object* v___y_2414_){
_start:
{
lean_object* v___x_2416_; 
v___x_2416_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__2___redArg(v___x_2406_, v___x_2407_, v___x_2408_, v_as_x27_2410_, v_b_2411_, v___y_2413_, v___y_2414_);
return v___x_2416_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__2___boxed(lean_object* v___x_2417_, lean_object* v___x_2418_, lean_object* v___x_2419_, lean_object* v_as_2420_, lean_object* v_as_x27_2421_, lean_object* v_b_2422_, lean_object* v_a_2423_, lean_object* v___y_2424_, lean_object* v___y_2425_, lean_object* v___y_2426_){
_start:
{
uint8_t v___x_7489__boxed_2427_; uint8_t v___x_7490__boxed_2428_; lean_object* v_res_2429_; 
v___x_7489__boxed_2427_ = lean_unbox(v___x_2418_);
v___x_7490__boxed_2428_ = lean_unbox(v___x_2419_);
v_res_2429_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__2(v___x_2417_, v___x_7489__boxed_2427_, v___x_7490__boxed_2428_, v_as_2420_, v_as_x27_2421_, v_b_2422_, v_a_2423_, v___y_2424_, v___y_2425_);
lean_dec(v___y_2425_);
lean_dec_ref(v___y_2424_);
lean_dec(v_as_x27_2421_);
lean_dec(v_as_2420_);
return v_res_2429_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__1_spec__1(lean_object* v_s_2430_, lean_object* v_inst_2431_, lean_object* v_R_2432_, lean_object* v_a_2433_, uint8_t v_b_2434_, lean_object* v_c_2435_){
_start:
{
uint8_t v___x_2436_; 
v___x_2436_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__1_spec__1___redArg(v_s_2430_, v_a_2433_, v_b_2434_);
return v___x_2436_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__1_spec__1___boxed(lean_object* v_s_2437_, lean_object* v_inst_2438_, lean_object* v_R_2439_, lean_object* v_a_2440_, lean_object* v_b_2441_, lean_object* v_c_2442_){
_start:
{
uint8_t v_b_boxed_2443_; uint8_t v_res_2444_; lean_object* v_r_2445_; 
v_b_boxed_2443_ = lean_unbox(v_b_2441_);
v_res_2444_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter_spec__1_spec__1(v_s_2437_, v_inst_2438_, v_R_2439_, v_a_2440_, v_b_boxed_2443_, v_c_2442_);
lean_dec_ref(v_s_2437_);
v_r_2445_ = lean_box(v_res_2444_);
return v_r_2445_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_initFn_00___x40_Mathlib_Tactic_Linter_Style_724867545____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_2447_; lean_object* v___x_2448_; 
v___x_2447_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_longLineLinter));
v___x_2448_ = l_Lean_Elab_Command_addLinter(v___x_2447_);
return v___x_2448_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_initFn_00___x40_Mathlib_Tactic_Linter_Style_724867545____hygCtx___hyg_2____boxed(lean_object* v_a_2449_){
_start:
{
lean_object* v_res_2450_; 
v_res_2450_ = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_initFn_00___x40_Mathlib_Tactic_Linter_Style_724867545____hygCtx___hyg_2_();
return v_res_2450_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_3276061806____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_2469_; lean_object* v___x_2470_; lean_object* v___x_2471_; lean_object* v___x_2472_; 
v___x_2469_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_3276061806____hygCtx___hyg_4_));
v___x_2470_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Style_3276061806____hygCtx___hyg_4_));
v___x_2471_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_3276061806____hygCtx___hyg_4_));
v___x_2472_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__spec__0(v___x_2469_, v___x_2470_, v___x_2471_);
return v___x_2472_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_3276061806____hygCtx___hyg_4____boxed(lean_object* v_a_2473_){
_start:
{
lean_object* v_res_2474_; 
v_res_2474_ = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_3276061806____hygCtx___hyg_4_();
return v_res_2474_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Option_instBEq_beq___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__0(lean_object* v_x_2475_, lean_object* v_x_2476_){
_start:
{
if (lean_obj_tag(v_x_2475_) == 0)
{
if (lean_obj_tag(v_x_2476_) == 0)
{
uint8_t v___x_2477_; 
v___x_2477_ = 1;
return v___x_2477_;
}
else
{
uint8_t v___x_2478_; 
v___x_2478_ = 0;
return v___x_2478_;
}
}
else
{
if (lean_obj_tag(v_x_2476_) == 0)
{
uint8_t v___x_2479_; 
v___x_2479_ = 0;
return v___x_2479_;
}
else
{
lean_object* v_val_2480_; lean_object* v_val_2481_; uint8_t v___x_2482_; 
v_val_2480_ = lean_ctor_get(v_x_2475_, 0);
v_val_2481_ = lean_ctor_get(v_x_2476_, 0);
v___x_2482_ = lean_nat_dec_eq(v_val_2480_, v_val_2481_);
return v___x_2482_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Option_instBEq_beq___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__0___boxed(lean_object* v_x_2483_, lean_object* v_x_2484_){
_start:
{
uint8_t v_res_2485_; lean_object* v_r_2486_; 
v_res_2485_ = lp_mathlib_Option_instBEq_beq___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__0(v_x_2483_, v_x_2484_);
lean_dec(v_x_2484_);
lean_dec(v_x_2483_);
v_r_2486_ = lean_box(v_res_2485_);
return v_r_2486_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___lam__0(lean_object* v_x_2493_){
_start:
{
lean_object* v___x_2494_; uint8_t v___x_2495_; 
v___x_2494_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___lam__0___closed__1));
v___x_2495_ = l_Lean_Syntax_isOfKind(v_x_2493_, v___x_2494_);
return v___x_2495_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___lam__0___boxed(lean_object* v_x_2496_){
_start:
{
uint8_t v_res_2497_; lean_object* v_r_2498_; 
v_res_2497_ = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___lam__0(v_x_2496_);
v_r_2498_ = lean_box(v_res_2497_);
return v_r_2498_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___lam__1(lean_object* v_x_2505_){
_start:
{
lean_object* v___x_2506_; uint8_t v___x_2507_; 
v___x_2506_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___lam__1___closed__1));
v___x_2507_ = l_Lean_Syntax_isOfKind(v_x_2505_, v___x_2506_);
return v___x_2507_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___lam__1___boxed(lean_object* v_x_2508_){
_start:
{
uint8_t v_res_2509_; lean_object* v_r_2510_; 
v_res_2509_ = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___lam__1(v_x_2508_);
v_r_2510_ = lean_box(v_res_2509_);
return v_r_2510_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__1___closed__3(void){
_start:
{
lean_object* v___x_2515_; lean_object* v___x_2516_; 
v___x_2515_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__1___closed__2));
v___x_2516_ = l_Lean_stringToMessageData(v___x_2515_);
return v___x_2516_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__1___closed__5(void){
_start:
{
lean_object* v___x_2518_; lean_object* v___x_2519_; 
v___x_2518_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__1___closed__4));
v___x_2519_ = l_Lean_stringToMessageData(v___x_2518_);
return v___x_2519_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__1(uint8_t v___x_2520_, uint8_t v___x_2521_, lean_object* v_as_2522_, size_t v_sz_2523_, size_t v_i_2524_, lean_object* v_b_2525_, lean_object* v___y_2526_, lean_object* v___y_2527_){
_start:
{
lean_object* v_a_2530_; uint8_t v___x_2534_; 
v___x_2534_ = lean_usize_dec_lt(v_i_2524_, v_sz_2523_);
if (v___x_2534_ == 0)
{
lean_object* v___x_2535_; 
v___x_2535_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2535_, 0, v_b_2525_);
return v___x_2535_;
}
else
{
lean_object* v___x_2536_; lean_object* v_a_2537_; lean_object* v___x_2538_; lean_object* v___x_2539_; lean_object* v___x_2540_; uint8_t v___x_2541_; 
v___x_2536_ = lean_box(0);
v_a_2537_ = lean_array_uget_borrowed(v_as_2522_, v_i_2524_);
v___x_2538_ = l_Lean_Syntax_getPos_x3f(v_a_2537_, v___x_2520_);
v___x_2539_ = lean_unsigned_to_nat(0u);
v___x_2540_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__1___closed__0));
v___x_2541_ = lp_mathlib_Option_instBEq_beq___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__0(v___x_2538_, v___x_2540_);
lean_dec(v___x_2538_);
if (v___x_2541_ == 0)
{
lean_object* v___x_2542_; uint8_t v___x_2543_; 
v___x_2542_ = l_Lean_Syntax_getId(v_a_2537_);
v___x_2543_ = l_Lean_Name_hasMacroScopes(v___x_2542_);
if (v___x_2543_ == 0)
{
lean_object* v___x_2544_; lean_object* v___x_2545_; uint8_t v___x_2546_; 
lean_inc(v_a_2537_);
v___x_2544_ = l_Lean_Syntax_getKind(v_a_2537_);
v___x_2545_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__10));
v___x_2546_ = lean_name_eq(v___x_2544_, v___x_2545_);
lean_dec(v___x_2544_);
if (v___x_2546_ == 0)
{
lean_dec(v___x_2542_);
v_a_2530_ = v___x_2536_;
goto v___jp_2529_;
}
else
{
lean_object* v___x_2547_; lean_object* v___x_2548_; lean_object* v___x_2549_; lean_object* v___x_2550_; lean_object* v___x_2551_; lean_object* v___x_2552_; uint8_t v___x_2553_; 
v___x_2547_ = lean_unsigned_to_nat(1u);
v___x_2548_ = l_Lean_Name_toString(v___x_2542_, v___x_2521_);
v___x_2549_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__1___closed__1));
v___x_2550_ = lean_box(0);
v___x_2551_ = l_String_splitOnAux(v___x_2548_, v___x_2549_, v___x_2539_, v___x_2539_, v___x_2539_, v___x_2550_);
lean_dec_ref(v___x_2548_);
v___x_2552_ = l_List_lengthTR___redArg(v___x_2551_);
lean_dec(v___x_2551_);
v___x_2553_ = lean_nat_dec_lt(v___x_2547_, v___x_2552_);
lean_dec(v___x_2552_);
if (v___x_2553_ == 0)
{
v_a_2530_ = v___x_2536_;
goto v___jp_2529_;
}
else
{
lean_object* v___x_2554_; lean_object* v___x_2555_; lean_object* v___x_2556_; lean_object* v___x_2557_; lean_object* v___x_2558_; lean_object* v___x_2559_; lean_object* v___x_2560_; 
v___x_2554_ = lp_mathlib_Mathlib_Linter_linter_style_nameCheck;
v___x_2555_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__1___closed__3, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__1___closed__3_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__1___closed__3);
lean_inc_n(v_a_2537_, 2);
v___x_2556_ = l_Lean_MessageData_ofSyntax(v_a_2537_);
v___x_2557_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2557_, 0, v___x_2555_);
lean_ctor_set(v___x_2557_, 1, v___x_2556_);
v___x_2558_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__1___closed__5, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__1___closed__5_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__1___closed__5);
v___x_2559_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2559_, 0, v___x_2557_);
lean_ctor_set(v___x_2559_, 1, v___x_2558_);
v___x_2560_ = lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__1(v___x_2554_, v_a_2537_, v___x_2559_, v___y_2526_, v___y_2527_);
if (lean_obj_tag(v___x_2560_) == 0)
{
lean_dec_ref_known(v___x_2560_, 1);
v_a_2530_ = v___x_2536_;
goto v___jp_2529_;
}
else
{
return v___x_2560_;
}
}
}
}
else
{
lean_dec(v___x_2542_);
v_a_2530_ = v___x_2536_;
goto v___jp_2529_;
}
}
else
{
v_a_2530_ = v___x_2536_;
goto v___jp_2529_;
}
}
v___jp_2529_:
{
size_t v___x_2531_; size_t v___x_2532_; 
v___x_2531_ = ((size_t)1ULL);
v___x_2532_ = lean_usize_add(v_i_2524_, v___x_2531_);
v_i_2524_ = v___x_2532_;
v_b_2525_ = v_a_2530_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__1___boxed(lean_object* v___x_2561_, lean_object* v___x_2562_, lean_object* v_as_2563_, lean_object* v_sz_2564_, lean_object* v_i_2565_, lean_object* v_b_2566_, lean_object* v___y_2567_, lean_object* v___y_2568_, lean_object* v___y_2569_){
_start:
{
uint8_t v___x_4454__boxed_2570_; uint8_t v___x_4455__boxed_2571_; size_t v_sz_boxed_2572_; size_t v_i_boxed_2573_; lean_object* v_res_2574_; 
v___x_4454__boxed_2570_ = lean_unbox(v___x_2561_);
v___x_4455__boxed_2571_ = lean_unbox(v___x_2562_);
v_sz_boxed_2572_ = lean_unbox_usize(v_sz_2564_);
lean_dec(v_sz_2564_);
v_i_boxed_2573_ = lean_unbox_usize(v_i_2565_);
lean_dec(v_i_2565_);
v_res_2574_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__1(v___x_4454__boxed_2570_, v___x_4455__boxed_2571_, v_as_2563_, v_sz_boxed_2572_, v_i_boxed_2573_, v_b_2566_, v___y_2567_, v___y_2568_);
lean_dec(v___y_2568_);
lean_dec_ref(v___y_2567_);
lean_dec_ref(v_as_2563_);
return v_res_2574_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_getAliasSyntax___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__2_spec__2___redArg(uint8_t v___x_2575_, lean_object* v___x_2576_, lean_object* v_as_2577_, size_t v_sz_2578_, size_t v_i_2579_, lean_object* v_b_2580_){
_start:
{
uint8_t v___x_2582_; 
v___x_2582_ = lean_usize_dec_lt(v_i_2579_, v_sz_2578_);
if (v___x_2582_ == 0)
{
lean_object* v___x_2583_; 
lean_dec(v___x_2576_);
v___x_2583_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2583_, 0, v_b_2580_);
return v___x_2583_;
}
else
{
lean_object* v_a_2584_; lean_object* v___x_2585_; uint8_t v___x_2586_; lean_object* v___y_2588_; lean_object* v___x_2596_; 
v_a_2584_ = lean_array_uget_borrowed(v_as_2577_, v_i_2579_);
v___x_2585_ = l_Lean_TSyntax_getId(v_a_2584_);
v___x_2586_ = 0;
v___x_2596_ = l_Lean_Syntax_getRange_x3f(v_a_2584_, v___x_2586_);
if (lean_obj_tag(v___x_2596_) == 0)
{
lean_object* v___x_2597_; 
v___x_2597_ = l_Lean_Syntax_instInhabitedRange_default;
v___y_2588_ = v___x_2597_;
goto v___jp_2587_;
}
else
{
lean_object* v_val_2598_; 
v_val_2598_ = lean_ctor_get(v___x_2596_, 0);
lean_inc(v_val_2598_);
lean_dec_ref_known(v___x_2596_, 1);
v___y_2588_ = v_val_2598_;
goto v___jp_2587_;
}
v___jp_2587_:
{
lean_object* v___x_2589_; lean_object* v___x_2590_; lean_object* v___x_2591_; lean_object* v___x_2592_; size_t v___x_2593_; size_t v___x_2594_; 
v___x_2589_ = l_Lean_Syntax_ofRange(v___y_2588_, v___x_2575_);
lean_inc(v___x_2576_);
v___x_2590_ = l_Lean_Name_append(v___x_2576_, v___x_2585_);
v___x_2591_ = l_Lean_mkIdentFrom(v___x_2589_, v___x_2590_, v___x_2586_);
lean_dec(v___x_2589_);
v___x_2592_ = lean_array_push(v_b_2580_, v___x_2591_);
v___x_2593_ = ((size_t)1ULL);
v___x_2594_ = lean_usize_add(v_i_2579_, v___x_2593_);
v_i_2579_ = v___x_2594_;
v_b_2580_ = v___x_2592_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_getAliasSyntax___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__2_spec__2___redArg___boxed(lean_object* v___x_2599_, lean_object* v___x_2600_, lean_object* v_as_2601_, lean_object* v_sz_2602_, lean_object* v_i_2603_, lean_object* v_b_2604_, lean_object* v___y_2605_){
_start:
{
uint8_t v___x_4542__boxed_2606_; size_t v_sz_boxed_2607_; size_t v_i_boxed_2608_; lean_object* v_res_2609_; 
v___x_4542__boxed_2606_ = lean_unbox(v___x_2599_);
v_sz_boxed_2607_ = lean_unbox_usize(v_sz_2602_);
lean_dec(v_sz_2602_);
v_i_boxed_2608_ = lean_unbox_usize(v_i_2603_);
lean_dec(v_i_2603_);
v_res_2609_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_getAliasSyntax___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__2_spec__2___redArg(v___x_4542__boxed_2606_, v___x_2600_, v_as_2601_, v_sz_boxed_2607_, v_i_boxed_2608_, v_b_2604_);
lean_dec_ref(v_as_2601_);
return v_res_2609_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_getAliasSyntax___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__2(lean_object* v_stx_2610_, lean_object* v___y_2611_, lean_object* v___y_2612_){
_start:
{
lean_object* v_aliases_2614_; lean_object* v___x_2615_; uint8_t v___x_2616_; 
v_aliases_2614_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_findCDot___closed__1));
v___x_2615_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___lam__1___closed__1));
lean_inc(v_stx_2610_);
v___x_2616_ = l_Lean_Syntax_isOfKind(v_stx_2610_, v___x_2615_);
if (v___x_2616_ == 0)
{
lean_object* v___x_2617_; 
lean_dec(v_stx_2610_);
v___x_2617_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2617_, 0, v_aliases_2614_);
return v___x_2617_;
}
else
{
lean_object* v___x_2618_; 
v___x_2618_ = l_Lean_Elab_Command_getScope___redArg(v___y_2612_);
if (lean_obj_tag(v___x_2618_) == 0)
{
lean_object* v_a_2619_; lean_object* v_currNamespace_2620_; lean_object* v___x_2621_; lean_object* v___x_2622_; lean_object* v_ids_2623_; size_t v_sz_2624_; size_t v___x_2625_; lean_object* v___x_2626_; 
v_a_2619_ = lean_ctor_get(v___x_2618_, 0);
lean_inc(v_a_2619_);
lean_dec_ref_known(v___x_2618_, 1);
v_currNamespace_2620_ = lean_ctor_get(v_a_2619_, 2);
lean_inc(v_currNamespace_2620_);
lean_dec(v_a_2619_);
v___x_2621_ = lean_unsigned_to_nat(3u);
v___x_2622_ = l_Lean_Syntax_getArg(v_stx_2610_, v___x_2621_);
lean_dec(v_stx_2610_);
v_ids_2623_ = l_Lean_Syntax_getArgs(v___x_2622_);
lean_dec(v___x_2622_);
v_sz_2624_ = lean_array_size(v_ids_2623_);
v___x_2625_ = ((size_t)0ULL);
v___x_2626_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_getAliasSyntax___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__2_spec__2___redArg(v___x_2616_, v_currNamespace_2620_, v_ids_2623_, v_sz_2624_, v___x_2625_, v_aliases_2614_);
lean_dec_ref(v_ids_2623_);
if (lean_obj_tag(v___x_2626_) == 0)
{
lean_object* v_a_2627_; lean_object* v___x_2629_; uint8_t v_isShared_2630_; uint8_t v_isSharedCheck_2634_; 
v_a_2627_ = lean_ctor_get(v___x_2626_, 0);
v_isSharedCheck_2634_ = !lean_is_exclusive(v___x_2626_);
if (v_isSharedCheck_2634_ == 0)
{
v___x_2629_ = v___x_2626_;
v_isShared_2630_ = v_isSharedCheck_2634_;
goto v_resetjp_2628_;
}
else
{
lean_inc(v_a_2627_);
lean_dec(v___x_2626_);
v___x_2629_ = lean_box(0);
v_isShared_2630_ = v_isSharedCheck_2634_;
goto v_resetjp_2628_;
}
v_resetjp_2628_:
{
lean_object* v___x_2632_; 
if (v_isShared_2630_ == 0)
{
v___x_2632_ = v___x_2629_;
goto v_reusejp_2631_;
}
else
{
lean_object* v_reuseFailAlloc_2633_; 
v_reuseFailAlloc_2633_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2633_, 0, v_a_2627_);
v___x_2632_ = v_reuseFailAlloc_2633_;
goto v_reusejp_2631_;
}
v_reusejp_2631_:
{
return v___x_2632_;
}
}
}
else
{
lean_object* v_a_2635_; lean_object* v___x_2637_; uint8_t v_isShared_2638_; uint8_t v_isSharedCheck_2642_; 
v_a_2635_ = lean_ctor_get(v___x_2626_, 0);
v_isSharedCheck_2642_ = !lean_is_exclusive(v___x_2626_);
if (v_isSharedCheck_2642_ == 0)
{
v___x_2637_ = v___x_2626_;
v_isShared_2638_ = v_isSharedCheck_2642_;
goto v_resetjp_2636_;
}
else
{
lean_inc(v_a_2635_);
lean_dec(v___x_2626_);
v___x_2637_ = lean_box(0);
v_isShared_2638_ = v_isSharedCheck_2642_;
goto v_resetjp_2636_;
}
v_resetjp_2636_:
{
lean_object* v___x_2640_; 
if (v_isShared_2638_ == 0)
{
v___x_2640_ = v___x_2637_;
goto v_reusejp_2639_;
}
else
{
lean_object* v_reuseFailAlloc_2641_; 
v_reuseFailAlloc_2641_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2641_, 0, v_a_2635_);
v___x_2640_ = v_reuseFailAlloc_2641_;
goto v_reusejp_2639_;
}
v_reusejp_2639_:
{
return v___x_2640_;
}
}
}
}
else
{
lean_object* v_a_2643_; lean_object* v___x_2645_; uint8_t v_isShared_2646_; uint8_t v_isSharedCheck_2650_; 
lean_dec(v_stx_2610_);
v_a_2643_ = lean_ctor_get(v___x_2618_, 0);
v_isSharedCheck_2650_ = !lean_is_exclusive(v___x_2618_);
if (v_isSharedCheck_2650_ == 0)
{
v___x_2645_ = v___x_2618_;
v_isShared_2646_ = v_isSharedCheck_2650_;
goto v_resetjp_2644_;
}
else
{
lean_inc(v_a_2643_);
lean_dec(v___x_2618_);
v___x_2645_ = lean_box(0);
v_isShared_2646_ = v_isSharedCheck_2650_;
goto v_resetjp_2644_;
}
v_resetjp_2644_:
{
lean_object* v___x_2648_; 
if (v_isShared_2646_ == 0)
{
v___x_2648_ = v___x_2645_;
goto v_reusejp_2647_;
}
else
{
lean_object* v_reuseFailAlloc_2649_; 
v_reuseFailAlloc_2649_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2649_, 0, v_a_2643_);
v___x_2648_ = v_reuseFailAlloc_2649_;
goto v_reusejp_2647_;
}
v_reusejp_2647_:
{
return v___x_2648_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_getAliasSyntax___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__2___boxed(lean_object* v_stx_2651_, lean_object* v___y_2652_, lean_object* v___y_2653_, lean_object* v___y_2654_){
_start:
{
lean_object* v_res_2655_; 
v_res_2655_ = lp_mathlib_Mathlib_Linter_getAliasSyntax___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__2(v_stx_2651_, v___y_2652_, v___y_2653_);
lean_dec(v___y_2653_);
lean_dec_ref(v___y_2652_);
return v_res_2655_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___lam__2(lean_object* v___f_2656_, lean_object* v___f_2657_, lean_object* v_stx_2658_, lean_object* v___y_2659_, lean_object* v___y_2660_){
_start:
{
lean_object* v___x_2662_; lean_object* v_a_2663_; lean_object* v___x_2665_; uint8_t v_isShared_2666_; uint8_t v_isSharedCheck_2720_; 
v___x_2662_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__0(v___y_2659_, v___y_2660_);
v_a_2663_ = lean_ctor_get(v___x_2662_, 0);
v_isSharedCheck_2720_ = !lean_is_exclusive(v___x_2662_);
if (v_isSharedCheck_2720_ == 0)
{
v___x_2665_ = v___x_2662_;
v_isShared_2666_ = v_isSharedCheck_2720_;
goto v_resetjp_2664_;
}
else
{
lean_inc(v_a_2663_);
lean_dec(v___x_2662_);
v___x_2665_ = lean_box(0);
v_isShared_2666_ = v_isSharedCheck_2720_;
goto v_resetjp_2664_;
}
v_resetjp_2664_:
{
lean_object* v___x_2667_; uint8_t v___x_2668_; 
v___x_2667_ = lp_mathlib_Mathlib_Linter_linter_style_nameCheck;
v___x_2668_ = l_Lean_Linter_getLinterValue(v___x_2667_, v_a_2663_);
lean_dec(v_a_2663_);
if (v___x_2668_ == 0)
{
lean_object* v___x_2669_; lean_object* v___x_2671_; 
lean_dec(v_stx_2658_);
lean_dec_ref(v___f_2657_);
lean_dec_ref(v___f_2656_);
v___x_2669_ = lean_box(0);
if (v_isShared_2666_ == 0)
{
lean_ctor_set(v___x_2665_, 0, v___x_2669_);
v___x_2671_ = v___x_2665_;
goto v_reusejp_2670_;
}
else
{
lean_object* v_reuseFailAlloc_2672_; 
v_reuseFailAlloc_2672_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2672_, 0, v___x_2669_);
v___x_2671_ = v_reuseFailAlloc_2672_;
goto v_reusejp_2670_;
}
v_reusejp_2670_:
{
return v___x_2671_;
}
}
else
{
lean_object* v___x_2673_; lean_object* v_messages_2674_; uint8_t v___x_2675_; lean_object* v___y_2677_; lean_object* v___y_2678_; lean_object* v___y_2679_; lean_object* v___y_2680_; lean_object* v_aliases_2697_; lean_object* v___y_2698_; lean_object* v___y_2699_; 
v___x_2673_ = lean_st_ref_get(v___y_2660_);
v_messages_2674_ = lean_ctor_get(v___x_2673_, 1);
lean_inc_ref(v_messages_2674_);
lean_dec(v___x_2673_);
v___x_2675_ = l_Lean_MessageLog_hasErrors(v_messages_2674_);
lean_dec_ref(v_messages_2674_);
if (v___x_2675_ == 0)
{
lean_object* v___x_2703_; 
lean_del_object(v___x_2665_);
lean_inc(v_stx_2658_);
v___x_2703_ = l_Lean_Syntax_find_x3f(v_stx_2658_, v___f_2657_);
if (lean_obj_tag(v___x_2703_) == 1)
{
lean_object* v_val_2704_; lean_object* v___x_2705_; 
v_val_2704_ = lean_ctor_get(v___x_2703_, 0);
lean_inc(v_val_2704_);
lean_dec_ref_known(v___x_2703_, 1);
v___x_2705_ = lp_mathlib_Mathlib_Linter_getAliasSyntax___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__2(v_val_2704_, v___y_2659_, v___y_2660_);
if (lean_obj_tag(v___x_2705_) == 0)
{
lean_object* v_a_2706_; 
v_a_2706_ = lean_ctor_get(v___x_2705_, 0);
lean_inc(v_a_2706_);
lean_dec_ref_known(v___x_2705_, 1);
v_aliases_2697_ = v_a_2706_;
v___y_2698_ = v___y_2659_;
v___y_2699_ = v___y_2660_;
goto v___jp_2696_;
}
else
{
lean_object* v_a_2707_; lean_object* v___x_2709_; uint8_t v_isShared_2710_; uint8_t v_isSharedCheck_2714_; 
lean_dec(v_stx_2658_);
lean_dec_ref(v___f_2656_);
v_a_2707_ = lean_ctor_get(v___x_2705_, 0);
v_isSharedCheck_2714_ = !lean_is_exclusive(v___x_2705_);
if (v_isSharedCheck_2714_ == 0)
{
v___x_2709_ = v___x_2705_;
v_isShared_2710_ = v_isSharedCheck_2714_;
goto v_resetjp_2708_;
}
else
{
lean_inc(v_a_2707_);
lean_dec(v___x_2705_);
v___x_2709_ = lean_box(0);
v_isShared_2710_ = v_isSharedCheck_2714_;
goto v_resetjp_2708_;
}
v_resetjp_2708_:
{
lean_object* v___x_2712_; 
if (v_isShared_2710_ == 0)
{
v___x_2712_ = v___x_2709_;
goto v_reusejp_2711_;
}
else
{
lean_object* v_reuseFailAlloc_2713_; 
v_reuseFailAlloc_2713_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2713_, 0, v_a_2707_);
v___x_2712_ = v_reuseFailAlloc_2713_;
goto v_reusejp_2711_;
}
v_reusejp_2711_:
{
return v___x_2712_;
}
}
}
}
else
{
lean_object* v___x_2715_; 
lean_dec(v___x_2703_);
v___x_2715_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_findCDot___closed__1));
v_aliases_2697_ = v___x_2715_;
v___y_2698_ = v___y_2659_;
v___y_2699_ = v___y_2660_;
goto v___jp_2696_;
}
}
else
{
lean_object* v___x_2716_; lean_object* v___x_2718_; 
lean_dec(v_stx_2658_);
lean_dec_ref(v___f_2657_);
lean_dec_ref(v___f_2656_);
v___x_2716_ = lean_box(0);
if (v_isShared_2666_ == 0)
{
lean_ctor_set(v___x_2665_, 0, v___x_2716_);
v___x_2718_ = v___x_2665_;
goto v_reusejp_2717_;
}
else
{
lean_object* v_reuseFailAlloc_2719_; 
v_reuseFailAlloc_2719_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2719_, 0, v___x_2716_);
v___x_2718_ = v_reuseFailAlloc_2719_;
goto v_reusejp_2717_;
}
v_reusejp_2717_:
{
return v___x_2718_;
}
}
v___jp_2676_:
{
lean_object* v___x_2681_; lean_object* v___x_2682_; lean_object* v___x_2683_; lean_object* v___x_2684_; size_t v_sz_2685_; size_t v___x_2686_; lean_object* v___x_2687_; 
v___x_2681_ = lean_unsigned_to_nat(0u);
v___x_2682_ = l_Lean_Syntax_getArg(v___y_2680_, v___x_2681_);
lean_dec(v___y_2680_);
v___x_2683_ = lean_array_push(v___y_2679_, v___x_2682_);
v___x_2684_ = lean_box(0);
v_sz_2685_ = lean_array_size(v___x_2683_);
v___x_2686_ = ((size_t)0ULL);
v___x_2687_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__1(v___x_2675_, v___x_2668_, v___x_2683_, v_sz_2685_, v___x_2686_, v___x_2684_, v___y_2678_, v___y_2677_);
lean_dec_ref(v___x_2683_);
if (lean_obj_tag(v___x_2687_) == 0)
{
lean_object* v___x_2689_; uint8_t v_isShared_2690_; uint8_t v_isSharedCheck_2694_; 
v_isSharedCheck_2694_ = !lean_is_exclusive(v___x_2687_);
if (v_isSharedCheck_2694_ == 0)
{
lean_object* v_unused_2695_; 
v_unused_2695_ = lean_ctor_get(v___x_2687_, 0);
lean_dec(v_unused_2695_);
v___x_2689_ = v___x_2687_;
v_isShared_2690_ = v_isSharedCheck_2694_;
goto v_resetjp_2688_;
}
else
{
lean_dec(v___x_2687_);
v___x_2689_ = lean_box(0);
v_isShared_2690_ = v_isSharedCheck_2694_;
goto v_resetjp_2688_;
}
v_resetjp_2688_:
{
lean_object* v___x_2692_; 
if (v_isShared_2690_ == 0)
{
lean_ctor_set(v___x_2689_, 0, v___x_2684_);
v___x_2692_ = v___x_2689_;
goto v_reusejp_2691_;
}
else
{
lean_object* v_reuseFailAlloc_2693_; 
v_reuseFailAlloc_2693_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2693_, 0, v___x_2684_);
v___x_2692_ = v_reuseFailAlloc_2693_;
goto v_reusejp_2691_;
}
v_reusejp_2691_:
{
return v___x_2692_;
}
}
}
else
{
return v___x_2687_;
}
}
v___jp_2696_:
{
lean_object* v___x_2700_; 
v___x_2700_ = l_Lean_Syntax_find_x3f(v_stx_2658_, v___f_2656_);
if (lean_obj_tag(v___x_2700_) == 0)
{
lean_object* v___x_2701_; 
v___x_2701_ = lean_box(0);
v___y_2677_ = v___y_2699_;
v___y_2678_ = v___y_2698_;
v___y_2679_ = v_aliases_2697_;
v___y_2680_ = v___x_2701_;
goto v___jp_2676_;
}
else
{
lean_object* v_val_2702_; 
v_val_2702_ = lean_ctor_get(v___x_2700_, 0);
lean_inc(v_val_2702_);
lean_dec_ref_known(v___x_2700_, 1);
v___y_2677_ = v___y_2699_;
v___y_2678_ = v___y_2698_;
v___y_2679_ = v_aliases_2697_;
v___y_2680_ = v_val_2702_;
goto v___jp_2676_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___lam__2___boxed(lean_object* v___f_2721_, lean_object* v___f_2722_, lean_object* v_stx_2723_, lean_object* v___y_2724_, lean_object* v___y_2725_, lean_object* v___y_2726_){
_start:
{
lean_object* v_res_2727_; 
v_res_2727_ = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore___lam__2(v___f_2721_, v___f_2722_, v_stx_2723_, v___y_2724_, v___y_2725_);
lean_dec(v___y_2725_);
lean_dec_ref(v___y_2724_);
return v_res_2727_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_getAliasSyntax___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__2_spec__2(uint8_t v___x_2746_, lean_object* v___x_2747_, lean_object* v_as_2748_, size_t v_sz_2749_, size_t v_i_2750_, lean_object* v_b_2751_, lean_object* v___y_2752_, lean_object* v___y_2753_){
_start:
{
lean_object* v___x_2755_; 
v___x_2755_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_getAliasSyntax___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__2_spec__2___redArg(v___x_2746_, v___x_2747_, v_as_2748_, v_sz_2749_, v_i_2750_, v_b_2751_);
return v___x_2755_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_getAliasSyntax___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__2_spec__2___boxed(lean_object* v___x_2756_, lean_object* v___x_2757_, lean_object* v_as_2758_, lean_object* v_sz_2759_, lean_object* v_i_2760_, lean_object* v_b_2761_, lean_object* v___y_2762_, lean_object* v___y_2763_, lean_object* v___y_2764_){
_start:
{
uint8_t v___x_4892__boxed_2765_; size_t v_sz_boxed_2766_; size_t v_i_boxed_2767_; lean_object* v_res_2768_; 
v___x_4892__boxed_2765_ = lean_unbox(v___x_2756_);
v_sz_boxed_2766_ = lean_unbox_usize(v_sz_2759_);
lean_dec(v_sz_2759_);
v_i_boxed_2767_ = lean_unbox_usize(v_i_2760_);
lean_dec(v_i_2760_);
v_res_2768_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_getAliasSyntax___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore_spec__2_spec__2(v___x_4892__boxed_2765_, v___x_2757_, v_as_2758_, v_sz_boxed_2766_, v_i_boxed_2767_, v_b_2761_, v___y_2762_, v___y_2763_);
lean_dec(v___y_2763_);
lean_dec_ref(v___y_2762_);
lean_dec_ref(v_as_2758_);
return v_res_2768_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_initFn_00___x40_Mathlib_Tactic_Linter_Style_1285535146____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_2770_; lean_object* v___x_2771_; 
v___x_2770_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_doubleUnderscore));
v___x_2771_ = l_Lean_Elab_Command_addLinter(v___x_2770_);
return v___x_2771_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_initFn_00___x40_Mathlib_Tactic_Linter_Style_1285535146____hygCtx___hyg_2____boxed(lean_object* v_a_2772_){
_start:
{
lean_object* v_res_2773_; 
v_res_2773_ = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_initFn_00___x40_Mathlib_Tactic_Linter_Style_1285535146____hygCtx___hyg_2_();
return v_res_2773_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_any___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__0(uint8_t v___y_2774_, lean_object* v_x_2775_){
_start:
{
if (lean_obj_tag(v_x_2775_) == 0)
{
uint8_t v___x_2776_; 
v___x_2776_ = 0;
return v___x_2776_;
}
else
{
lean_object* v_head_2777_; lean_object* v_tail_2778_; uint8_t v___y_2780_; uint8_t v___x_2782_; lean_object* v___x_2783_; lean_object* v___x_2784_; lean_object* v___x_2785_; uint8_t v___x_2786_; 
v_head_2777_ = lean_ctor_get(v_x_2775_, 0);
lean_inc(v_head_2777_);
v_tail_2778_ = lean_ctor_get(v_x_2775_, 1);
lean_inc(v_tail_2778_);
lean_dec_ref_known(v_x_2775_, 2);
v___x_2782_ = 1;
v___x_2783_ = l_Lean_Name_toString(v_head_2777_, v___x_2782_);
v___x_2784_ = lean_unsigned_to_nat(0u);
v___x_2785_ = lean_string_utf8_byte_size(v___x_2783_);
v___x_2786_ = lean_nat_dec_eq(v___x_2785_, v___x_2784_);
if (v___x_2786_ == 0)
{
lean_object* v___x_2787_; uint32_t v___x_2788_; lean_object* v___x_2789_; lean_object* v___x_2790_; lean_object* v___x_2791_; uint32_t v___x_2792_; uint8_t v___x_2793_; 
lean_inc_ref(v___x_2783_);
v___x_2787_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2787_, 0, v___x_2783_);
lean_ctor_set(v___x_2787_, 1, v___x_2784_);
lean_ctor_set(v___x_2787_, 2, v___x_2785_);
v___x_2788_ = 95;
v___x_2789_ = lean_unsigned_to_nat(1u);
v___x_2790_ = lean_nat_sub(v___x_2785_, v___x_2789_);
v___x_2791_ = l_String_Slice_posLE(v___x_2787_, v___x_2790_);
lean_dec_ref_known(v___x_2787_, 3);
v___x_2792_ = lean_string_utf8_get_fast(v___x_2783_, v___x_2791_);
lean_dec(v___x_2791_);
lean_dec_ref(v___x_2783_);
v___x_2793_ = lean_uint32_dec_eq(v___x_2792_, v___x_2788_);
v___y_2780_ = v___x_2793_;
goto v___jp_2779_;
}
else
{
lean_dec_ref(v___x_2783_);
v___y_2780_ = v___y_2774_;
goto v___jp_2779_;
}
v___jp_2779_:
{
if (v___y_2780_ == 0)
{
v_x_2775_ = v_tail_2778_;
goto _start;
}
else
{
lean_dec(v_tail_2778_);
return v___y_2780_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_any___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__0___boxed(lean_object* v___y_2794_, lean_object* v_x_2795_){
_start:
{
uint8_t v___y_2757__boxed_2796_; uint8_t v_res_2797_; lean_object* v_r_2798_; 
v___y_2757__boxed_2796_ = lean_unbox(v___y_2794_);
v_res_2797_ = lp_mathlib_List_any___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__0(v___y_2757__boxed_2796_, v_x_2795_);
v_r_2798_ = lean_box(v_res_2797_);
return v_r_2798_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1_spec__1___redArg(lean_object* v_s_2799_, lean_object* v_a_2800_, uint8_t v_b_2801_){
_start:
{
uint8_t v___x_2802_; 
v___x_2802_ = 0;
switch(lean_obj_tag(v_a_2800_))
{
case 0:
{
uint8_t v___x_2803_; 
lean_dec_ref_known(v_a_2800_, 1);
v___x_2803_ = 1;
return v___x_2803_;
}
case 1:
{
lean_object* v_pos_2804_; lean_object* v___x_2806_; uint8_t v_isShared_2807_; uint8_t v_isSharedCheck_2817_; 
v_pos_2804_ = lean_ctor_get(v_a_2800_, 0);
v_isSharedCheck_2817_ = !lean_is_exclusive(v_a_2800_);
if (v_isSharedCheck_2817_ == 0)
{
v___x_2806_ = v_a_2800_;
v_isShared_2807_ = v_isSharedCheck_2817_;
goto v_resetjp_2805_;
}
else
{
lean_inc(v_pos_2804_);
lean_dec(v_a_2800_);
v___x_2806_ = lean_box(0);
v_isShared_2807_ = v_isSharedCheck_2817_;
goto v_resetjp_2805_;
}
v_resetjp_2805_:
{
lean_object* v_str_2808_; lean_object* v_startInclusive_2809_; lean_object* v___x_2810_; lean_object* v___x_2811_; lean_object* v___x_2812_; lean_object* v___x_2814_; 
v_str_2808_ = lean_ctor_get(v_s_2799_, 0);
v_startInclusive_2809_ = lean_ctor_get(v_s_2799_, 1);
v___x_2810_ = lean_nat_add(v_startInclusive_2809_, v_pos_2804_);
lean_dec(v_pos_2804_);
v___x_2811_ = lean_string_utf8_next_fast(v_str_2808_, v___x_2810_);
lean_dec(v___x_2810_);
v___x_2812_ = lean_nat_sub(v___x_2811_, v_startInclusive_2809_);
if (v_isShared_2807_ == 0)
{
lean_ctor_set_tag(v___x_2806_, 0);
lean_ctor_set(v___x_2806_, 0, v___x_2812_);
v___x_2814_ = v___x_2806_;
goto v_reusejp_2813_;
}
else
{
lean_object* v_reuseFailAlloc_2816_; 
v_reuseFailAlloc_2816_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2816_, 0, v___x_2812_);
v___x_2814_ = v_reuseFailAlloc_2816_;
goto v_reusejp_2813_;
}
v_reusejp_2813_:
{
v_a_2800_ = v___x_2814_;
v_b_2801_ = v___x_2802_;
goto _start;
}
}
}
case 2:
{
lean_object* v_needle_2818_; lean_object* v_table_2819_; lean_object* v_stackPos_2820_; lean_object* v_needlePos_2821_; lean_object* v___x_2823_; uint8_t v_isShared_2824_; uint8_t v_isSharedCheck_2874_; 
v_needle_2818_ = lean_ctor_get(v_a_2800_, 0);
v_table_2819_ = lean_ctor_get(v_a_2800_, 1);
v_stackPos_2820_ = lean_ctor_get(v_a_2800_, 2);
v_needlePos_2821_ = lean_ctor_get(v_a_2800_, 3);
v_isSharedCheck_2874_ = !lean_is_exclusive(v_a_2800_);
if (v_isSharedCheck_2874_ == 0)
{
v___x_2823_ = v_a_2800_;
v_isShared_2824_ = v_isSharedCheck_2874_;
goto v_resetjp_2822_;
}
else
{
lean_inc(v_needlePos_2821_);
lean_inc(v_stackPos_2820_);
lean_inc(v_table_2819_);
lean_inc(v_needle_2818_);
lean_dec(v_a_2800_);
v___x_2823_ = lean_box(0);
v_isShared_2824_ = v_isSharedCheck_2874_;
goto v_resetjp_2822_;
}
v_resetjp_2822_:
{
lean_object* v_str_2825_; lean_object* v_startInclusive_2826_; lean_object* v_endExclusive_2827_; lean_object* v_str_2828_; lean_object* v_startInclusive_2829_; lean_object* v_endExclusive_2830_; lean_object* v_basePos_2831_; lean_object* v___x_2832_; lean_object* v___x_2833_; lean_object* v___x_2834_; uint8_t v___x_2835_; 
v_str_2825_ = lean_ctor_get(v_needle_2818_, 0);
v_startInclusive_2826_ = lean_ctor_get(v_needle_2818_, 1);
v_endExclusive_2827_ = lean_ctor_get(v_needle_2818_, 2);
v_str_2828_ = lean_ctor_get(v_s_2799_, 0);
v_startInclusive_2829_ = lean_ctor_get(v_s_2799_, 1);
v_endExclusive_2830_ = lean_ctor_get(v_s_2799_, 2);
v_basePos_2831_ = lean_nat_sub(v_stackPos_2820_, v_needlePos_2821_);
v___x_2832_ = lean_nat_sub(v_endExclusive_2827_, v_startInclusive_2826_);
v___x_2833_ = lean_nat_add(v_basePos_2831_, v___x_2832_);
v___x_2834_ = lean_nat_sub(v_endExclusive_2830_, v_startInclusive_2829_);
v___x_2835_ = lean_nat_dec_le(v___x_2833_, v___x_2834_);
lean_dec(v___x_2833_);
if (v___x_2835_ == 0)
{
uint8_t v___x_2836_; 
lean_dec(v___x_2832_);
lean_del_object(v___x_2823_);
lean_dec(v_needlePos_2821_);
lean_dec(v_stackPos_2820_);
lean_dec_ref(v_table_2819_);
lean_dec_ref(v_needle_2818_);
v___x_2836_ = lean_nat_dec_lt(v_basePos_2831_, v___x_2834_);
lean_dec(v___x_2834_);
lean_dec(v_basePos_2831_);
if (v___x_2836_ == 0)
{
return v_b_2801_;
}
else
{
lean_object* v___x_2837_; 
v___x_2837_ = lean_box(3);
v_a_2800_ = v___x_2837_;
v_b_2801_ = v___x_2802_;
goto _start;
}
}
else
{
lean_object* v___x_2839_; uint8_t v_stackByte_2840_; lean_object* v___x_2841_; uint8_t v_patByte_2842_; uint8_t v___x_2843_; 
lean_dec(v___x_2834_);
lean_dec(v_basePos_2831_);
v___x_2839_ = lean_nat_add(v_startInclusive_2829_, v_stackPos_2820_);
v_stackByte_2840_ = lean_string_get_byte_fast(v_str_2828_, v___x_2839_);
v___x_2841_ = lean_nat_add(v_startInclusive_2826_, v_needlePos_2821_);
v_patByte_2842_ = lean_string_get_byte_fast(v_str_2825_, v___x_2841_);
v___x_2843_ = lean_uint8_dec_eq(v_stackByte_2840_, v_patByte_2842_);
if (v___x_2843_ == 0)
{
lean_object* v___x_2844_; uint8_t v___x_2845_; 
lean_dec(v___x_2832_);
v___x_2844_ = lean_unsigned_to_nat(0u);
v___x_2845_ = lean_nat_dec_eq(v_needlePos_2821_, v___x_2844_);
if (v___x_2845_ == 0)
{
lean_object* v___x_2846_; lean_object* v___x_2847_; lean_object* v_newNeedlePos_2848_; uint8_t v___x_2849_; 
v___x_2846_ = lean_unsigned_to_nat(1u);
v___x_2847_ = lean_nat_sub(v_needlePos_2821_, v___x_2846_);
lean_dec(v_needlePos_2821_);
v_newNeedlePos_2848_ = lean_array_fget_borrowed(v_table_2819_, v___x_2847_);
lean_dec(v___x_2847_);
v___x_2849_ = lean_nat_dec_eq(v_newNeedlePos_2848_, v___x_2844_);
if (v___x_2849_ == 0)
{
lean_object* v___x_2851_; 
lean_inc(v_newNeedlePos_2848_);
if (v_isShared_2824_ == 0)
{
lean_ctor_set(v___x_2823_, 3, v_newNeedlePos_2848_);
v___x_2851_ = v___x_2823_;
goto v_reusejp_2850_;
}
else
{
lean_object* v_reuseFailAlloc_2853_; 
v_reuseFailAlloc_2853_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v_reuseFailAlloc_2853_, 0, v_needle_2818_);
lean_ctor_set(v_reuseFailAlloc_2853_, 1, v_table_2819_);
lean_ctor_set(v_reuseFailAlloc_2853_, 2, v_stackPos_2820_);
lean_ctor_set(v_reuseFailAlloc_2853_, 3, v_newNeedlePos_2848_);
v___x_2851_ = v_reuseFailAlloc_2853_;
goto v_reusejp_2850_;
}
v_reusejp_2850_:
{
v_a_2800_ = v___x_2851_;
v_b_2801_ = v___x_2802_;
goto _start;
}
}
else
{
lean_object* v_nextStackPos_2854_; lean_object* v___x_2856_; 
v_nextStackPos_2854_ = l_String_Slice_posGE___redArg(v_s_2799_, v_stackPos_2820_);
if (v_isShared_2824_ == 0)
{
lean_ctor_set(v___x_2823_, 3, v___x_2844_);
lean_ctor_set(v___x_2823_, 2, v_nextStackPos_2854_);
v___x_2856_ = v___x_2823_;
goto v_reusejp_2855_;
}
else
{
lean_object* v_reuseFailAlloc_2858_; 
v_reuseFailAlloc_2858_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v_reuseFailAlloc_2858_, 0, v_needle_2818_);
lean_ctor_set(v_reuseFailAlloc_2858_, 1, v_table_2819_);
lean_ctor_set(v_reuseFailAlloc_2858_, 2, v_nextStackPos_2854_);
lean_ctor_set(v_reuseFailAlloc_2858_, 3, v___x_2844_);
v___x_2856_ = v_reuseFailAlloc_2858_;
goto v_reusejp_2855_;
}
v_reusejp_2855_:
{
v_a_2800_ = v___x_2856_;
v_b_2801_ = v___x_2802_;
goto _start;
}
}
}
else
{
lean_object* v___x_2859_; lean_object* v___x_2860_; lean_object* v_nextStackPos_2861_; lean_object* v___x_2863_; 
lean_dec(v_needlePos_2821_);
v___x_2859_ = lean_unsigned_to_nat(1u);
v___x_2860_ = lean_nat_add(v_stackPos_2820_, v___x_2859_);
lean_dec(v_stackPos_2820_);
v_nextStackPos_2861_ = l_String_Slice_posGE___redArg(v_s_2799_, v___x_2860_);
if (v_isShared_2824_ == 0)
{
lean_ctor_set(v___x_2823_, 3, v___x_2844_);
lean_ctor_set(v___x_2823_, 2, v_nextStackPos_2861_);
v___x_2863_ = v___x_2823_;
goto v_reusejp_2862_;
}
else
{
lean_object* v_reuseFailAlloc_2865_; 
v_reuseFailAlloc_2865_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v_reuseFailAlloc_2865_, 0, v_needle_2818_);
lean_ctor_set(v_reuseFailAlloc_2865_, 1, v_table_2819_);
lean_ctor_set(v_reuseFailAlloc_2865_, 2, v_nextStackPos_2861_);
lean_ctor_set(v_reuseFailAlloc_2865_, 3, v___x_2844_);
v___x_2863_ = v_reuseFailAlloc_2865_;
goto v_reusejp_2862_;
}
v_reusejp_2862_:
{
v_a_2800_ = v___x_2863_;
v_b_2801_ = v___x_2802_;
goto _start;
}
}
}
else
{
lean_object* v___x_2866_; lean_object* v_nextNeedlePos_2867_; uint8_t v___x_2868_; 
v___x_2866_ = lean_unsigned_to_nat(1u);
v_nextNeedlePos_2867_ = lean_nat_add(v_needlePos_2821_, v___x_2866_);
lean_dec(v_needlePos_2821_);
v___x_2868_ = lean_nat_dec_eq(v_nextNeedlePos_2867_, v___x_2832_);
lean_dec(v___x_2832_);
if (v___x_2868_ == 0)
{
lean_object* v_nextStackPos_2869_; lean_object* v___x_2871_; 
v_nextStackPos_2869_ = lean_nat_add(v_stackPos_2820_, v___x_2866_);
lean_dec(v_stackPos_2820_);
if (v_isShared_2824_ == 0)
{
lean_ctor_set(v___x_2823_, 3, v_nextNeedlePos_2867_);
lean_ctor_set(v___x_2823_, 2, v_nextStackPos_2869_);
v___x_2871_ = v___x_2823_;
goto v_reusejp_2870_;
}
else
{
lean_object* v_reuseFailAlloc_2873_; 
v_reuseFailAlloc_2873_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v_reuseFailAlloc_2873_, 0, v_needle_2818_);
lean_ctor_set(v_reuseFailAlloc_2873_, 1, v_table_2819_);
lean_ctor_set(v_reuseFailAlloc_2873_, 2, v_nextStackPos_2869_);
lean_ctor_set(v_reuseFailAlloc_2873_, 3, v_nextNeedlePos_2867_);
v___x_2871_ = v_reuseFailAlloc_2873_;
goto v_reusejp_2870_;
}
v_reusejp_2870_:
{
v_a_2800_ = v___x_2871_;
goto _start;
}
}
else
{
lean_dec(v_nextNeedlePos_2867_);
lean_del_object(v___x_2823_);
lean_dec(v_stackPos_2820_);
lean_dec_ref(v_table_2819_);
lean_dec_ref(v_needle_2818_);
return v___x_2868_;
}
}
}
}
}
default: 
{
return v_b_2801_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1_spec__1___redArg___boxed(lean_object* v_s_2875_, lean_object* v_a_2876_, lean_object* v_b_2877_){
_start:
{
uint8_t v_b_boxed_2878_; uint8_t v_res_2879_; lean_object* v_r_2880_; 
v_b_boxed_2878_ = lean_unbox(v_b_2877_);
v_res_2879_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1_spec__1___redArg(v_s_2875_, v_a_2876_, v_b_boxed_2878_);
lean_dec_ref(v_s_2875_);
v_r_2880_ = lean_box(v_res_2879_);
return v_r_2880_;
}
}
static lean_object* _init_lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1___closed__1(void){
_start:
{
lean_object* v___x_2882_; lean_object* v___x_2883_; 
v___x_2882_ = ((lean_object*)(lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1___closed__0));
v___x_2883_ = lean_string_utf8_byte_size(v___x_2882_);
return v___x_2883_;
}
}
static uint8_t _init_lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1___closed__2(void){
_start:
{
lean_object* v___x_2884_; lean_object* v___x_2885_; uint8_t v___x_2886_; 
v___x_2884_ = lean_unsigned_to_nat(0u);
v___x_2885_ = lean_obj_once(&lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1___closed__1, &lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1___closed__1_once, _init_lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1___closed__1);
v___x_2886_ = lean_nat_dec_eq(v___x_2885_, v___x_2884_);
return v___x_2886_;
}
}
static lean_object* _init_lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1___closed__3(void){
_start:
{
lean_object* v___x_2887_; lean_object* v___x_2888_; lean_object* v___x_2889_; lean_object* v___x_2890_; 
v___x_2887_ = lean_obj_once(&lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1___closed__1, &lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1___closed__1_once, _init_lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1___closed__1);
v___x_2888_ = lean_unsigned_to_nat(0u);
v___x_2889_ = ((lean_object*)(lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1___closed__0));
v___x_2890_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2890_, 0, v___x_2889_);
lean_ctor_set(v___x_2890_, 1, v___x_2888_);
lean_ctor_set(v___x_2890_, 2, v___x_2887_);
return v___x_2890_;
}
}
static lean_object* _init_lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1___closed__4(void){
_start:
{
lean_object* v___x_2891_; lean_object* v___x_2892_; 
v___x_2891_ = lean_obj_once(&lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1___closed__3, &lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1___closed__3_once, _init_lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1___closed__3);
v___x_2892_ = l_String_Slice_Pattern_ForwardSliceSearcher_buildTable(v___x_2891_);
return v___x_2892_;
}
}
static lean_object* _init_lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1___closed__5(void){
_start:
{
lean_object* v___x_2893_; lean_object* v___x_2894_; lean_object* v___x_2895_; lean_object* v___x_2896_; 
v___x_2893_ = lean_unsigned_to_nat(0u);
v___x_2894_ = lean_obj_once(&lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1___closed__4, &lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1___closed__4_once, _init_lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1___closed__4);
v___x_2895_ = lean_obj_once(&lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1___closed__3, &lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1___closed__3_once, _init_lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1___closed__3);
v___x_2896_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v___x_2896_, 0, v___x_2895_);
lean_ctor_set(v___x_2896_, 1, v___x_2894_);
lean_ctor_set(v___x_2896_, 2, v___x_2893_);
lean_ctor_set(v___x_2896_, 3, v___x_2893_);
return v___x_2896_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1(lean_object* v_s_2899_){
_start:
{
lean_object* v___y_2901_; uint8_t v___x_2904_; 
v___x_2904_ = lean_uint8_once(&lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1___closed__2, &lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1___closed__2_once, _init_lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1___closed__2);
if (v___x_2904_ == 0)
{
lean_object* v___x_2905_; 
v___x_2905_ = lean_obj_once(&lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1___closed__5, &lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1___closed__5_once, _init_lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1___closed__5);
v___y_2901_ = v___x_2905_;
goto v___jp_2900_;
}
else
{
lean_object* v___x_2906_; 
v___x_2906_ = ((lean_object*)(lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1___closed__6));
v___y_2901_ = v___x_2906_;
goto v___jp_2900_;
}
v___jp_2900_:
{
uint8_t v___x_2902_; uint8_t v___x_2903_; 
v___x_2902_ = 0;
lean_inc(v___y_2901_);
v___x_2903_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1_spec__1___redArg(v_s_2899_, v___y_2901_, v___x_2902_);
return v___x_2903_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1___boxed(lean_object* v_s_2907_){
_start:
{
uint8_t v_res_2908_; lean_object* v_r_2909_; 
v_res_2908_ = lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1(v_s_2907_);
lean_dec_ref(v_s_2907_);
v_r_2909_ = lean_box(v_res_2908_);
return v_r_2909_;
}
}
static lean_object* _init_lp_mathlib_List_any___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__2___closed__1(void){
_start:
{
lean_object* v___x_2911_; lean_object* v___x_2912_; 
v___x_2911_ = ((lean_object*)(lp_mathlib_List_any___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__2___closed__0));
v___x_2912_ = lean_string_utf8_byte_size(v___x_2911_);
return v___x_2912_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_any___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__2(uint8_t v___y_2913_, lean_object* v_x_2914_){
_start:
{
if (lean_obj_tag(v_x_2914_) == 0)
{
uint8_t v___x_2915_; 
v___x_2915_ = 0;
return v___x_2915_;
}
else
{
lean_object* v_head_2916_; lean_object* v_tail_2917_; uint8_t v___y_2919_; uint8_t v___x_2921_; lean_object* v___x_2922_; lean_object* v___x_2923_; lean_object* v___x_2924_; lean_object* v___x_2925_; uint8_t v___x_2926_; 
v_head_2916_ = lean_ctor_get(v_x_2914_, 0);
lean_inc(v_head_2916_);
v_tail_2917_ = lean_ctor_get(v_x_2914_, 1);
lean_inc(v_tail_2917_);
lean_dec_ref_known(v_x_2914_, 2);
v___x_2921_ = 1;
v___x_2922_ = l_Lean_Name_toString(v_head_2916_, v___x_2921_);
v___x_2923_ = ((lean_object*)(lp_mathlib_List_any___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__2___closed__0));
v___x_2924_ = lean_string_utf8_byte_size(v___x_2922_);
v___x_2925_ = lean_obj_once(&lp_mathlib_List_any___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__2___closed__1, &lp_mathlib_List_any___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__2___closed__1_once, _init_lp_mathlib_List_any___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__2___closed__1);
v___x_2926_ = lean_nat_dec_le(v___x_2925_, v___x_2924_);
if (v___x_2926_ == 0)
{
lean_dec_ref(v___x_2922_);
v___y_2919_ = v___y_2913_;
goto v___jp_2918_;
}
else
{
lean_object* v___x_2927_; uint8_t v___x_2928_; 
v___x_2927_ = lean_unsigned_to_nat(0u);
v___x_2928_ = lean_string_memcmp(v___x_2922_, v___x_2923_, v___x_2927_, v___x_2927_, v___x_2925_);
lean_dec_ref(v___x_2922_);
v___y_2919_ = v___x_2928_;
goto v___jp_2918_;
}
v___jp_2918_:
{
if (v___y_2919_ == 0)
{
v_x_2914_ = v_tail_2917_;
goto _start;
}
else
{
lean_dec(v_tail_2917_);
return v___y_2919_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_any___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__2___boxed(lean_object* v___y_2929_, lean_object* v_x_2930_){
_start:
{
uint8_t v___y_2990__boxed_2931_; uint8_t v_res_2932_; lean_object* v_r_2933_; 
v___y_2990__boxed_2931_ = lean_unbox(v___y_2929_);
v_res_2932_ = lp_mathlib_List_any___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__2(v___y_2990__boxed_2931_, v_x_2930_);
v_r_2933_ = lean_box(v_res_2932_);
return v_r_2933_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_any___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__3(lean_object* v_x_2937_){
_start:
{
if (lean_obj_tag(v_x_2937_) == 0)
{
uint8_t v___x_2938_; 
v___x_2938_ = 0;
return v___x_2938_;
}
else
{
lean_object* v_head_2939_; lean_object* v_tail_2940_; lean_object* v___x_2941_; uint8_t v___x_2942_; 
v_head_2939_ = lean_ctor_get(v_x_2937_, 0);
v_tail_2940_ = lean_ctor_get(v_x_2937_, 1);
v___x_2941_ = ((lean_object*)(lp_mathlib_List_any___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__3___closed__1));
v___x_2942_ = lean_name_eq(v_head_2939_, v___x_2941_);
if (v___x_2942_ == 0)
{
v_x_2937_ = v_tail_2940_;
goto _start;
}
else
{
return v___x_2942_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_any___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__3___boxed(lean_object* v_x_2944_){
_start:
{
uint8_t v_res_2945_; lean_object* v_r_2946_; 
v_res_2945_ = lp_mathlib_List_any___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__3(v_x_2944_);
lean_dec(v_x_2944_);
v_r_2946_ = lean_box(v_res_2945_);
return v_r_2946_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__4_spec__5___redArg(lean_object* v_s_2947_, lean_object* v_a_2948_, uint8_t v_b_2949_){
_start:
{
lean_object* v_str_2950_; lean_object* v_startInclusive_2951_; lean_object* v_endExclusive_2952_; lean_object* v___x_2953_; uint8_t v___x_2954_; 
v_str_2950_ = lean_ctor_get(v_s_2947_, 0);
v_startInclusive_2951_ = lean_ctor_get(v_s_2947_, 1);
v_endExclusive_2952_ = lean_ctor_get(v_s_2947_, 2);
v___x_2953_ = lean_nat_sub(v_endExclusive_2952_, v_startInclusive_2951_);
v___x_2954_ = lean_nat_dec_eq(v_a_2948_, v___x_2953_);
lean_dec(v___x_2953_);
if (v___x_2954_ == 0)
{
lean_object* v___x_2955_; uint32_t v___x_2956_; uint32_t v___x_2957_; uint8_t v___x_2958_; 
v___x_2955_ = lean_nat_add(v_startInclusive_2951_, v_a_2948_);
lean_dec(v_a_2948_);
v___x_2956_ = lean_string_utf8_get_fast(v_str_2950_, v___x_2955_);
v___x_2957_ = 171;
v___x_2958_ = lean_uint32_dec_eq(v___x_2956_, v___x_2957_);
if (v___x_2958_ == 0)
{
lean_object* v___x_2959_; lean_object* v___x_2960_; 
v___x_2959_ = lean_string_utf8_next_fast(v_str_2950_, v___x_2955_);
lean_dec(v___x_2955_);
v___x_2960_ = lean_nat_sub(v___x_2959_, v_startInclusive_2951_);
v_a_2948_ = v___x_2960_;
v_b_2949_ = v___x_2958_;
goto _start;
}
else
{
lean_dec(v___x_2955_);
return v___x_2958_;
}
}
else
{
lean_dec(v_a_2948_);
return v_b_2949_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__4_spec__5___redArg___boxed(lean_object* v_s_2962_, lean_object* v_a_2963_, lean_object* v_b_2964_){
_start:
{
uint8_t v_b_boxed_2965_; uint8_t v_res_2966_; lean_object* v_r_2967_; 
v_b_boxed_2965_ = lean_unbox(v_b_2964_);
v_res_2966_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__4_spec__5___redArg(v_s_2962_, v_a_2963_, v_b_boxed_2965_);
lean_dec_ref(v_s_2962_);
v_r_2967_ = lean_box(v_res_2966_);
return v_r_2967_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__4(lean_object* v_s_2968_){
_start:
{
lean_object* v_searcher_2969_; uint8_t v___x_2970_; uint8_t v___x_2971_; 
v_searcher_2969_ = lean_unsigned_to_nat(0u);
v___x_2970_ = 0;
v___x_2971_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__4_spec__5___redArg(v_s_2968_, v_searcher_2969_, v___x_2970_);
return v___x_2971_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__4___boxed(lean_object* v_s_2972_){
_start:
{
uint8_t v_res_2973_; lean_object* v_r_2974_; 
v_res_2973_ = lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__4(v_s_2972_);
lean_dec_ref(v_s_2972_);
v_r_2974_ = lean_box(v_res_2973_);
return v_r_2974_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__1(void){
_start:
{
lean_object* v___x_2976_; lean_object* v___x_2977_; 
v___x_2976_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__0));
v___x_2977_ = lean_string_utf8_byte_size(v___x_2976_);
return v___x_2977_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__3(void){
_start:
{
lean_object* v___x_2979_; lean_object* v___x_2980_; 
v___x_2979_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__2));
v___x_2980_ = lean_string_utf8_byte_size(v___x_2979_);
return v___x_2980_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__9(void){
_start:
{
lean_object* v___x_2990_; lean_object* v___x_2991_; 
v___x_2990_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__8));
v___x_2991_ = lean_string_utf8_byte_size(v___x_2990_);
return v___x_2991_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore(lean_object* v_name_2995_){
_start:
{
uint8_t v___x_2996_; lean_object* v_s_2997_; lean_object* v___x_2998_; uint8_t v___y_3000_; lean_object* v___y_3007_; uint8_t v___y_3008_; lean_object* v___y_3017_; uint8_t v___y_3018_; lean_object* v___y_3028_; lean_object* v___y_3049_; lean_object* v___x_3061_; 
v___x_2996_ = 1;
lean_inc_n(v_name_2995_, 2);
v_s_2997_ = l_Lean_Name_toString(v_name_2995_, v___x_2996_);
v___x_2998_ = l_Lean_Name_components(v_name_2995_);
v___x_3061_ = l_List_getLast_x3f___redArg(v___x_2998_);
if (lean_obj_tag(v___x_3061_) == 0)
{
lean_object* v___x_3062_; 
v___x_3062_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__11));
v___y_3049_ = v___x_3062_;
goto v___jp_3048_;
}
else
{
lean_object* v_val_3063_; 
v_val_3063_ = lean_ctor_get(v___x_3061_, 0);
lean_inc(v_val_3063_);
lean_dec_ref_known(v___x_3061_, 1);
v___y_3049_ = v_val_3063_;
goto v___jp_3048_;
}
v___jp_2999_:
{
uint8_t v___x_3001_; 
v___x_3001_ = lp_mathlib_List_any___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__0(v___y_3000_, v___x_2998_);
if (v___x_3001_ == 0)
{
lean_object* v___x_3002_; lean_object* v___x_3003_; lean_object* v___x_3004_; uint8_t v___x_3005_; 
v___x_3002_ = lean_unsigned_to_nat(0u);
v___x_3003_ = lean_string_utf8_byte_size(v_s_2997_);
v___x_3004_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_3004_, 0, v_s_2997_);
lean_ctor_set(v___x_3004_, 1, v___x_3002_);
lean_ctor_set(v___x_3004_, 2, v___x_3003_);
v___x_3005_ = lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1(v___x_3004_);
lean_dec_ref_known(v___x_3004_, 3);
if (v___x_3005_ == 0)
{
return v___x_3005_;
}
else
{
return v___x_2996_;
}
}
else
{
lean_dec_ref(v_s_2997_);
return v___y_3000_;
}
}
v___jp_3006_:
{
lean_object* v___x_3009_; lean_object* v___x_3010_; lean_object* v___x_3011_; uint8_t v___x_3012_; 
v___x_3009_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__0));
v___x_3010_ = lean_string_utf8_byte_size(v___y_3007_);
v___x_3011_ = lean_obj_once(&lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__1, &lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__1_once, _init_lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__1);
v___x_3012_ = lean_nat_dec_le(v___x_3011_, v___x_3010_);
if (v___x_3012_ == 0)
{
lean_dec_ref(v___y_3007_);
v___y_3000_ = v___y_3008_;
goto v___jp_2999_;
}
else
{
lean_object* v___x_3013_; lean_object* v___x_3014_; uint8_t v___x_3015_; 
v___x_3013_ = lean_unsigned_to_nat(0u);
v___x_3014_ = lean_nat_sub(v___x_3010_, v___x_3011_);
v___x_3015_ = lean_string_memcmp(v___y_3007_, v___x_3009_, v___x_3014_, v___x_3013_, v___x_3011_);
lean_dec(v___x_3014_);
lean_dec_ref(v___y_3007_);
if (v___x_3015_ == 0)
{
v___y_3000_ = v___x_3015_;
goto v___jp_2999_;
}
else
{
lean_dec(v___x_2998_);
lean_dec_ref(v_s_2997_);
return v___y_3008_;
}
}
}
v___jp_3016_:
{
lean_object* v___x_3019_; lean_object* v___x_3020_; lean_object* v___x_3021_; uint8_t v___x_3022_; 
v___x_3019_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__2));
v___x_3020_ = lean_string_utf8_byte_size(v___y_3017_);
v___x_3021_ = lean_obj_once(&lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__3, &lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__3_once, _init_lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__3);
v___x_3022_ = lean_nat_dec_le(v___x_3021_, v___x_3020_);
if (v___x_3022_ == 0)
{
v___y_3007_ = v___y_3017_;
v___y_3008_ = v___y_3018_;
goto v___jp_3006_;
}
else
{
lean_object* v___x_3023_; lean_object* v___x_3024_; uint8_t v___x_3025_; 
v___x_3023_ = lean_unsigned_to_nat(0u);
v___x_3024_ = lean_nat_sub(v___x_3020_, v___x_3021_);
v___x_3025_ = lean_string_memcmp(v___y_3017_, v___x_3019_, v___x_3024_, v___x_3023_, v___x_3021_);
lean_dec(v___x_3024_);
if (v___x_3025_ == 0)
{
v___y_3007_ = v___y_3017_;
v___y_3008_ = v___x_3025_;
goto v___jp_3006_;
}
else
{
uint8_t v___x_3026_; 
lean_dec_ref(v___y_3017_);
lean_dec(v___x_2998_);
lean_dec_ref(v_s_2997_);
v___x_3026_ = 0;
return v___x_3026_;
}
}
}
v___jp_3027_:
{
lean_object* v___x_3029_; lean_object* v___x_3030_; lean_object* v___x_3031_; uint8_t v___x_3032_; 
v___x_3029_ = lean_unsigned_to_nat(0u);
v___x_3030_ = lean_string_utf8_byte_size(v_s_2997_);
lean_inc_ref(v_s_2997_);
v___x_3031_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_3031_, 0, v_s_2997_);
lean_ctor_set(v___x_3031_, 1, v___x_3029_);
lean_ctor_set(v___x_3031_, 2, v___x_3030_);
v___x_3032_ = lp_mathlib_String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__4(v___x_3031_);
lean_dec_ref_known(v___x_3031_, 3);
if (v___x_3032_ == 0)
{
uint8_t v___x_3033_; 
lean_inc(v___x_2998_);
v___x_3033_ = lp_mathlib_List_any___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__2(v___x_3032_, v___x_2998_);
if (v___x_3033_ == 0)
{
lean_object* v___x_3034_; uint8_t v___x_3035_; 
v___x_3034_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__5));
v___x_3035_ = l_Lean_Name_isPrefixOf(v___x_3034_, v_name_2995_);
if (v___x_3035_ == 0)
{
lean_object* v___x_3036_; uint8_t v___x_3037_; 
v___x_3036_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__6));
v___x_3037_ = l_Lean_Name_isPrefixOf(v___x_3036_, v_name_2995_);
if (v___x_3037_ == 0)
{
lean_object* v___x_3038_; uint8_t v___x_3039_; 
v___x_3038_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__7));
v___x_3039_ = l_Lean_Name_isPrefixOf(v___x_3038_, v_name_2995_);
lean_dec(v_name_2995_);
if (v___x_3039_ == 0)
{
uint8_t v___x_3040_; 
v___x_3040_ = lp_mathlib_List_any___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__3(v___x_2998_);
if (v___x_3040_ == 0)
{
lean_object* v___x_3041_; lean_object* v___x_3042_; lean_object* v___x_3043_; uint8_t v___x_3044_; 
v___x_3041_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__8));
v___x_3042_ = lean_string_utf8_byte_size(v___y_3028_);
v___x_3043_ = lean_obj_once(&lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__9, &lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__9_once, _init_lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___closed__9);
v___x_3044_ = lean_nat_dec_le(v___x_3043_, v___x_3042_);
if (v___x_3044_ == 0)
{
v___y_3017_ = v___y_3028_;
v___y_3018_ = v___x_3040_;
goto v___jp_3016_;
}
else
{
lean_object* v___x_3045_; uint8_t v___x_3046_; 
v___x_3045_ = lean_nat_sub(v___x_3042_, v___x_3043_);
v___x_3046_ = lean_string_memcmp(v___y_3028_, v___x_3041_, v___x_3045_, v___x_3029_, v___x_3043_);
lean_dec(v___x_3045_);
if (v___x_3046_ == 0)
{
v___y_3017_ = v___y_3028_;
v___y_3018_ = v___x_3046_;
goto v___jp_3016_;
}
else
{
lean_dec_ref(v___y_3028_);
lean_dec(v___x_2998_);
lean_dec_ref(v_s_2997_);
return v___x_3040_;
}
}
}
else
{
lean_dec_ref(v___y_3028_);
lean_dec(v___x_2998_);
lean_dec_ref(v_s_2997_);
return v___x_3039_;
}
}
else
{
lean_dec_ref(v___y_3028_);
lean_dec(v___x_2998_);
lean_dec_ref(v_s_2997_);
return v___x_3037_;
}
}
else
{
lean_dec_ref(v___y_3028_);
lean_dec(v___x_2998_);
lean_dec_ref(v_s_2997_);
lean_dec(v_name_2995_);
return v___x_3035_;
}
}
else
{
lean_dec_ref(v___y_3028_);
lean_dec(v___x_2998_);
lean_dec_ref(v_s_2997_);
lean_dec(v_name_2995_);
return v___x_3033_;
}
}
else
{
lean_dec_ref(v___y_3028_);
lean_dec(v___x_2998_);
lean_dec_ref(v_s_2997_);
lean_dec(v_name_2995_);
return v___x_3032_;
}
}
else
{
uint8_t v___x_3047_; 
lean_dec_ref(v___y_3028_);
lean_dec(v___x_2998_);
lean_dec_ref(v_s_2997_);
lean_dec(v_name_2995_);
v___x_3047_ = 0;
return v___x_3047_;
}
}
v___jp_3048_:
{
lean_object* v_last_3050_; lean_object* v___x_3051_; lean_object* v___x_3052_; uint8_t v___x_3053_; 
v_last_3050_ = l_Lean_Name_toString(v___y_3049_, v___x_2996_);
v___x_3051_ = lean_unsigned_to_nat(0u);
v___x_3052_ = lean_string_utf8_byte_size(v_last_3050_);
v___x_3053_ = lean_nat_dec_eq(v___x_3052_, v___x_3051_);
if (v___x_3053_ == 0)
{
lean_object* v___x_3054_; uint32_t v___x_3055_; lean_object* v___x_3056_; lean_object* v___x_3057_; lean_object* v___x_3058_; uint32_t v___x_3059_; uint8_t v___x_3060_; 
lean_inc_ref(v_last_3050_);
v___x_3054_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_3054_, 0, v_last_3050_);
lean_ctor_set(v___x_3054_, 1, v___x_3051_);
lean_ctor_set(v___x_3054_, 2, v___x_3052_);
v___x_3055_ = 95;
v___x_3056_ = lean_unsigned_to_nat(1u);
v___x_3057_ = lean_nat_sub(v___x_3052_, v___x_3056_);
v___x_3058_ = l_String_Slice_posLE(v___x_3054_, v___x_3057_);
lean_dec_ref_known(v___x_3054_, 3);
v___x_3059_ = lean_string_utf8_get_fast(v_last_3050_, v___x_3058_);
lean_dec(v___x_3058_);
v___x_3060_ = lean_uint32_dec_eq(v___x_3059_, v___x_3055_);
if (v___x_3060_ == 0)
{
v___y_3028_ = v_last_3050_;
goto v___jp_3027_;
}
else
{
lean_dec_ref(v_last_3050_);
lean_dec(v___x_2998_);
lean_dec_ref(v_s_2997_);
lean_dec(v_name_2995_);
return v___x_3053_;
}
}
else
{
v___y_3028_ = v_last_3050_;
goto v___jp_3027_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore___boxed(lean_object* v_name_3064_){
_start:
{
uint8_t v_res_3065_; lean_object* v_r_3066_; 
v_res_3065_ = lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore(v_name_3064_);
v_r_3066_ = lean_box(v_res_3065_);
return v_r_3066_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1_spec__1(lean_object* v_s_3067_, lean_object* v_inst_3068_, lean_object* v_R_3069_, lean_object* v_a_3070_, uint8_t v_b_3071_, lean_object* v_c_3072_){
_start:
{
uint8_t v___x_3073_; 
v___x_3073_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1_spec__1___redArg(v_s_3067_, v_a_3070_, v_b_3071_);
return v___x_3073_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1_spec__1___boxed(lean_object* v_s_3074_, lean_object* v_inst_3075_, lean_object* v_R_3076_, lean_object* v_a_3077_, lean_object* v_b_3078_, lean_object* v_c_3079_){
_start:
{
uint8_t v_b_boxed_3080_; uint8_t v_res_3081_; lean_object* v_r_3082_; 
v_b_boxed_3080_ = lean_unbox(v_b_3078_);
v_res_3081_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__1_spec__1(v_s_3074_, v_inst_3075_, v_R_3076_, v_a_3077_, v_b_boxed_3080_, v_c_3079_);
lean_dec_ref(v_s_3074_);
v_r_3082_ = lean_box(v_res_3081_);
return v_r_3082_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__4_spec__5(lean_object* v_s_3083_, lean_object* v_inst_3084_, lean_object* v_R_3085_, lean_object* v_a_3086_, uint8_t v_b_3087_, lean_object* v_c_3088_){
_start:
{
uint8_t v___x_3089_; 
v___x_3089_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__4_spec__5___redArg(v_s_3083_, v_a_3086_, v_b_3087_);
return v___x_3089_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__4_spec__5___boxed(lean_object* v_s_3090_, lean_object* v_inst_3091_, lean_object* v_R_3092_, lean_object* v_a_3093_, lean_object* v_b_3094_, lean_object* v_c_3095_){
_start:
{
uint8_t v_b_boxed_3096_; uint8_t v_res_3097_; lean_object* v_r_3098_; 
v_b_boxed_3096_ = lean_unbox(v_b_3094_);
v_res_3097_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore_spec__4_spec__5(v_s_3090_, v_inst_3091_, v_R_3092_, v_a_3093_, v_b_boxed_3096_, v_c_3095_);
lean_dec_ref(v_s_3090_);
v_r_3098_ = lean_box(v_res_3097_);
return v_r_3098_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Linter_Style_nameCheck_defsWithUnderscore_spec__0(lean_object* v_msg_3099_){
_start:
{
lean_object* v___x_3100_; lean_object* v___x_3101_; 
v___x_3100_ = l_Lean_instInhabitedConstantInfo_default;
v___x_3101_ = lean_panic_fn_borrowed(v___x_3100_, v_msg_3099_);
return v___x_3101_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___lam__0___closed__5(void){
_start:
{
lean_object* v___x_3111_; lean_object* v___x_3112_; 
v___x_3111_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___lam__0___closed__4));
v___x_3112_ = l_Lean_stringToMessageData(v___x_3111_);
return v___x_3112_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___lam__0___closed__7(void){
_start:
{
lean_object* v___x_3114_; lean_object* v___x_3115_; 
v___x_3114_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___lam__0___closed__6));
v___x_3115_ = l_Lean_stringToMessageData(v___x_3114_);
return v___x_3115_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___lam__0(lean_object* v_declName_3116_, lean_object* v___y_3117_, lean_object* v___y_3118_, lean_object* v___y_3119_, lean_object* v___y_3120_){
_start:
{
uint8_t v___y_3123_; lean_object* v___y_3124_; uint8_t v___y_3144_; uint8_t v___y_3145_; uint8_t v___y_3153_; uint8_t v___y_3154_; lean_object* v___y_3155_; lean_object* v___x_3157_; lean_object* v___x_3158_; lean_object* v_env_3159_; uint8_t v_a_3161_; uint8_t v___x_3167_; 
v___x_3157_ = lean_st_ref_get(v___y_3120_);
v___x_3158_ = lean_st_ref_get(v___y_3120_);
v_env_3159_ = lean_ctor_get(v___x_3157_, 0);
lean_inc_ref(v_env_3159_);
lean_dec(v___x_3157_);
v___x_3167_ = l_Lean_isPrivateName(v_declName_3116_);
if (v___x_3167_ == 0)
{
lean_object* v_env_3168_; uint8_t v___x_3169_; 
v_env_3168_ = lean_ctor_get(v___x_3158_, 0);
lean_inc_ref(v_env_3168_);
lean_dec(v___x_3158_);
lean_inc(v_declName_3116_);
v___x_3169_ = lp_batteries_Lean_Environment_isAutoDecl(v_env_3168_, v_declName_3116_);
v_a_3161_ = v___x_3169_;
goto v___jp_3160_;
}
else
{
lean_dec(v___x_3158_);
v_a_3161_ = v___x_3167_;
goto v___jp_3160_;
}
v___jp_3122_:
{
lean_object* v___x_3125_; lean_object* v___x_3126_; uint8_t v___x_3127_; 
v___x_3125_ = l_Lean_ConstantInfo_type(v___y_3124_);
lean_dec_ref(v___y_3124_);
v___x_3126_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___lam__0___closed__3));
v___x_3127_ = l_Lean_Expr_isConstOf(v___x_3125_, v___x_3126_);
lean_dec_ref(v___x_3125_);
if (v___x_3127_ == 0)
{
uint8_t v___x_3128_; 
lean_inc(v_declName_3116_);
v___x_3128_ = lp_mathlib_Mathlib_Linter_Style_nameCheck_isBadNameWithUnderscore(v_declName_3116_);
if (v___x_3128_ == 0)
{
lean_object* v___x_3129_; lean_object* v___x_3130_; 
lean_dec(v_declName_3116_);
v___x_3129_ = lean_box(0);
v___x_3130_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3130_, 0, v___x_3129_);
return v___x_3130_;
}
else
{
lean_object* v___x_3131_; lean_object* v___x_3132_; lean_object* v___x_3133_; lean_object* v___x_3134_; lean_object* v___x_3135_; lean_object* v___x_3136_; lean_object* v___x_3137_; 
v___x_3131_ = lean_obj_once(&lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___lam__0___closed__5, &lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___lam__0___closed__5_once, _init_lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___lam__0___closed__5);
v___x_3132_ = l_Lean_MessageData_ofConstName(v_declName_3116_, v___y_3123_);
v___x_3133_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3133_, 0, v___x_3131_);
lean_ctor_set(v___x_3133_, 1, v___x_3132_);
v___x_3134_ = lean_obj_once(&lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___lam__0___closed__7, &lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___lam__0___closed__7_once, _init_lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___lam__0___closed__7);
v___x_3135_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3135_, 0, v___x_3133_);
lean_ctor_set(v___x_3135_, 1, v___x_3134_);
v___x_3136_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3136_, 0, v___x_3135_);
v___x_3137_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3137_, 0, v___x_3136_);
return v___x_3137_;
}
}
else
{
lean_object* v___x_3138_; lean_object* v___x_3139_; 
lean_dec(v_declName_3116_);
v___x_3138_ = lean_box(0);
v___x_3139_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3139_, 0, v___x_3138_);
return v___x_3139_;
}
}
v___jp_3140_:
{
lean_object* v___x_3141_; lean_object* v___x_3142_; 
v___x_3141_ = lean_box(0);
v___x_3142_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3142_, 0, v___x_3141_);
return v___x_3142_;
}
v___jp_3143_:
{
if (v___y_3145_ == 0)
{
lean_dec(v_declName_3116_);
goto v___jp_3140_;
}
else
{
lean_object* v___x_3146_; lean_object* v_env_3147_; lean_object* v___x_3148_; 
v___x_3146_ = lean_st_ref_get(v___y_3120_);
v_env_3147_ = lean_ctor_get(v___x_3146_, 0);
lean_inc_ref(v_env_3147_);
lean_dec(v___x_3146_);
lean_inc(v_declName_3116_);
v___x_3148_ = l_Lean_Environment_find_x3f(v_env_3147_, v_declName_3116_, v___y_3144_);
if (lean_obj_tag(v___x_3148_) == 0)
{
lean_object* v___x_3149_; lean_object* v___x_3150_; 
v___x_3149_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___closed__7, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___closed__7_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___closed__7);
v___x_3150_ = lp_mathlib_panic___at___00Mathlib_Linter_Style_nameCheck_defsWithUnderscore_spec__0(v___x_3149_);
v___y_3123_ = v___y_3145_;
v___y_3124_ = v___x_3150_;
goto v___jp_3122_;
}
else
{
lean_object* v_val_3151_; 
v_val_3151_ = lean_ctor_get(v___x_3148_, 0);
lean_inc(v_val_3151_);
lean_dec_ref_known(v___x_3148_, 1);
v___y_3123_ = v___y_3145_;
v___y_3124_ = v_val_3151_;
goto v___jp_3122_;
}
}
}
v___jp_3152_:
{
uint8_t v___x_3156_; 
v___x_3156_ = l_Lean_ConstantInfo_isDefinition(v___y_3155_);
lean_dec_ref(v___y_3155_);
if (v___x_3156_ == 0)
{
v___y_3144_ = v___y_3153_;
v___y_3145_ = v___x_3156_;
goto v___jp_3143_;
}
else
{
if (v___y_3154_ == 0)
{
v___y_3144_ = v___y_3153_;
v___y_3145_ = v___x_3156_;
goto v___jp_3143_;
}
else
{
lean_dec(v_declName_3116_);
goto v___jp_3140_;
}
}
}
v___jp_3160_:
{
uint8_t v___x_3162_; lean_object* v___x_3163_; 
v___x_3162_ = 0;
lean_inc(v_declName_3116_);
v___x_3163_ = l_Lean_Environment_find_x3f(v_env_3159_, v_declName_3116_, v___x_3162_);
if (lean_obj_tag(v___x_3163_) == 0)
{
lean_object* v___x_3164_; lean_object* v___x_3165_; 
v___x_3164_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___closed__7, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___closed__7_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_cdotLinter_spec__4___closed__7);
v___x_3165_ = lp_mathlib_panic___at___00Mathlib_Linter_Style_nameCheck_defsWithUnderscore_spec__0(v___x_3164_);
v___y_3153_ = v___x_3162_;
v___y_3154_ = v_a_3161_;
v___y_3155_ = v___x_3165_;
goto v___jp_3152_;
}
else
{
lean_object* v_val_3166_; 
v_val_3166_ = lean_ctor_get(v___x_3163_, 0);
lean_inc(v_val_3166_);
lean_dec_ref_known(v___x_3163_, 1);
v___y_3153_ = v___x_3162_;
v___y_3154_ = v_a_3161_;
v___y_3155_ = v_val_3166_;
goto v___jp_3152_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___lam__0___boxed(lean_object* v_declName_3170_, lean_object* v___y_3171_, lean_object* v___y_3172_, lean_object* v___y_3173_, lean_object* v___y_3174_, lean_object* v___y_3175_){
_start:
{
lean_object* v_res_3176_; 
v_res_3176_ = lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___lam__0(v_declName_3170_, v___y_3171_, v___y_3172_, v___y_3173_, v___y_3174_);
lean_dec(v___y_3174_);
lean_dec_ref(v___y_3173_);
lean_dec(v___y_3172_);
lean_dec_ref(v___y_3171_);
return v_res_3176_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___closed__3(void){
_start:
{
lean_object* v___x_3181_; lean_object* v___x_3182_; 
v___x_3181_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___closed__2));
v___x_3182_ = l_Lean_MessageData_ofFormat(v___x_3181_);
return v___x_3182_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___closed__6(void){
_start:
{
lean_object* v___x_3186_; lean_object* v___x_3187_; 
v___x_3186_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___closed__5));
v___x_3187_ = l_Lean_MessageData_ofFormat(v___x_3186_);
return v___x_3187_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___closed__7(void){
_start:
{
uint8_t v___x_3188_; lean_object* v___x_3189_; lean_object* v___x_3190_; lean_object* v___f_3191_; lean_object* v___x_3192_; 
v___x_3188_ = 1;
v___x_3189_ = lean_obj_once(&lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___closed__6, &lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___closed__6_once, _init_lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___closed__6);
v___x_3190_ = lean_obj_once(&lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___closed__3, &lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___closed__3_once, _init_lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___closed__3);
v___f_3191_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___closed__0));
v___x_3192_ = lean_alloc_ctor(0, 3, 1);
lean_ctor_set(v___x_3192_, 0, v___f_3191_);
lean_ctor_set(v___x_3192_, 1, v___x_3190_);
lean_ctor_set(v___x_3192_, 2, v___x_3189_);
lean_ctor_set_uint8(v___x_3192_, sizeof(void*)*3, v___x_3188_);
return v___x_3192_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore(void){
_start:
{
lean_object* v___x_3193_; 
v___x_3193_ = lean_obj_once(&lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___closed__7, &lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___closed__7_once, _init_lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore___closed__7);
return v___x_3193_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_856385564____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_3212_; lean_object* v___x_3213_; lean_object* v___x_3214_; lean_object* v___x_3215_; 
v___x_3212_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_856385564____hygCtx___hyg_4_));
v___x_3213_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Style_856385564____hygCtx___hyg_4_));
v___x_3214_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_856385564____hygCtx___hyg_4_));
v___x_3215_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__spec__0(v___x_3212_, v___x_3213_, v___x_3214_);
return v___x_3215_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_856385564____hygCtx___hyg_4____boxed(lean_object* v_a_3216_){
_start:
{
lean_object* v_res_3217_; 
v_res_3217_ = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_856385564____hygCtx___hyg_4_();
return v_res_3217_;
}
}
static lean_object* _init_lp_mathlib_panic___at___00Mathlib_Linter_Style_openClassical_extractOpenNames_spec__0___closed__0(void){
_start:
{
lean_object* v___x_3218_; 
v___x_3218_ = l_Array_instInhabited(lean_box(0));
return v___x_3218_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Linter_Style_openClassical_extractOpenNames_spec__0(lean_object* v_msg_3219_){
_start:
{
lean_object* v___x_3220_; lean_object* v___x_3221_; 
v___x_3220_ = lean_obj_once(&lp_mathlib_panic___at___00Mathlib_Linter_Style_openClassical_extractOpenNames_spec__0___closed__0, &lp_mathlib_panic___at___00Mathlib_Linter_Style_openClassical_extractOpenNames_spec__0___closed__0_once, _init_lp_mathlib_panic___at___00Mathlib_Linter_Style_openClassical_extractOpenNames_spec__0___closed__0);
v___x_3221_ = lean_panic_fn_borrowed(v___x_3220_, v_msg_3219_);
return v___x_3221_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__17(void){
_start:
{
lean_object* v___x_3267_; lean_object* v___x_3268_; lean_object* v___x_3269_; lean_object* v___x_3270_; lean_object* v___x_3271_; lean_object* v___x_3272_; 
v___x_3267_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__16));
v___x_3268_ = lean_unsigned_to_nat(11u);
v___x_3269_ = lean_unsigned_to_nat(600u);
v___x_3270_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__15));
v___x_3271_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__14));
v___x_3272_ = l_mkPanicMessageWithDecl(v___x_3271_, v___x_3270_, v___x_3269_, v___x_3268_, v___x_3267_);
return v___x_3272_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames(lean_object* v_x_3273_){
_start:
{
lean_object* v___x_3274_; uint8_t v___x_3275_; 
v___x_3274_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__1));
lean_inc(v_x_3273_);
v___x_3275_ = l_Lean_Syntax_isOfKind(v_x_3273_, v___x_3274_);
if (v___x_3275_ == 0)
{
lean_object* v___x_3276_; uint8_t v___x_3277_; 
v___x_3276_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__3));
lean_inc(v_x_3273_);
v___x_3277_ = l_Lean_Syntax_isOfKind(v_x_3273_, v___x_3276_);
if (v___x_3277_ == 0)
{
lean_object* v___x_3278_; 
lean_dec(v_x_3273_);
v___x_3278_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_findCDot___closed__1));
return v___x_3278_;
}
else
{
lean_object* v___x_3279_; lean_object* v___x_3280_; lean_object* v___x_3281_; lean_object* v___x_3282_; uint8_t v___x_3283_; 
v___x_3279_ = lean_unsigned_to_nat(0u);
v___x_3280_ = lean_unsigned_to_nat(1u);
v___x_3281_ = l_Lean_Syntax_getArg(v_x_3273_, v___x_3280_);
lean_dec(v_x_3273_);
v___x_3282_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__5));
lean_inc(v___x_3281_);
v___x_3283_ = l_Lean_Syntax_isOfKind(v___x_3281_, v___x_3282_);
if (v___x_3283_ == 0)
{
lean_object* v___x_3284_; uint8_t v___x_3285_; 
v___x_3284_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__7));
lean_inc(v___x_3281_);
v___x_3285_ = l_Lean_Syntax_isOfKind(v___x_3281_, v___x_3284_);
if (v___x_3285_ == 0)
{
lean_object* v___x_3286_; uint8_t v___x_3287_; 
v___x_3286_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__9));
lean_inc(v___x_3281_);
v___x_3287_ = l_Lean_Syntax_isOfKind(v___x_3281_, v___x_3286_);
if (v___x_3287_ == 0)
{
lean_object* v___x_3288_; uint8_t v___x_3289_; 
v___x_3288_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__11));
lean_inc(v___x_3281_);
v___x_3289_ = l_Lean_Syntax_isOfKind(v___x_3281_, v___x_3288_);
if (v___x_3289_ == 0)
{
lean_object* v___x_3290_; uint8_t v___x_3291_; 
v___x_3290_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__13));
lean_inc(v___x_3281_);
v___x_3291_ = l_Lean_Syntax_isOfKind(v___x_3281_, v___x_3290_);
if (v___x_3291_ == 0)
{
lean_object* v___x_3292_; lean_object* v___x_3293_; 
lean_dec(v___x_3281_);
v___x_3292_ = lean_obj_once(&lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__17, &lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__17_once, _init_lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames___closed__17);
v___x_3293_ = lp_mathlib_panic___at___00Mathlib_Linter_Style_openClassical_extractOpenNames_spec__0(v___x_3292_);
return v___x_3293_;
}
else
{
lean_object* v___x_3294_; lean_object* v___x_3295_; 
v___x_3294_ = l_Lean_Syntax_getArg(v___x_3281_, v___x_3280_);
lean_dec(v___x_3281_);
v___x_3295_ = l_Lean_Syntax_getArgs(v___x_3294_);
lean_dec(v___x_3294_);
return v___x_3295_;
}
}
else
{
lean_object* v___x_3296_; lean_object* v___x_3297_; 
v___x_3296_ = l_Lean_Syntax_getArg(v___x_3281_, v___x_3279_);
lean_dec(v___x_3281_);
v___x_3297_ = l_Lean_Syntax_getArgs(v___x_3296_);
lean_dec(v___x_3296_);
return v___x_3297_;
}
}
else
{
lean_object* v_arg_3298_; lean_object* v___x_3299_; lean_object* v___x_3300_; 
v_arg_3298_ = l_Lean_Syntax_getArg(v___x_3281_, v___x_3279_);
lean_dec(v___x_3281_);
v___x_3299_ = lean_mk_empty_array_with_capacity(v___x_3280_);
v___x_3300_ = lean_array_push(v___x_3299_, v_arg_3298_);
return v___x_3300_;
}
}
else
{
lean_object* v_arg_3301_; lean_object* v___x_3302_; lean_object* v___x_3303_; 
v_arg_3301_ = l_Lean_Syntax_getArg(v___x_3281_, v___x_3279_);
lean_dec(v___x_3281_);
v___x_3302_ = lean_mk_empty_array_with_capacity(v___x_3280_);
v___x_3303_ = lean_array_push(v___x_3302_, v_arg_3301_);
return v___x_3303_;
}
}
else
{
lean_object* v_arg_3304_; lean_object* v___x_3305_; lean_object* v___x_3306_; 
v_arg_3304_ = l_Lean_Syntax_getArg(v___x_3281_, v___x_3279_);
lean_dec(v___x_3281_);
v___x_3305_ = lean_mk_empty_array_with_capacity(v___x_3280_);
v___x_3306_ = lean_array_push(v___x_3305_, v_arg_3304_);
return v___x_3306_;
}
}
}
else
{
lean_object* v___x_3307_; 
lean_dec(v_x_3273_);
v___x_3307_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_findCDot___closed__1));
return v___x_3307_;
}
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter_spec__0___closed__2(void){
_start:
{
lean_object* v___x_3311_; lean_object* v___x_3312_; 
v___x_3311_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter_spec__0___closed__1));
v___x_3312_ = l_Lean_MessageData_ofFormat(v___x_3311_);
return v___x_3312_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter_spec__0(lean_object* v_as_3313_, size_t v_sz_3314_, size_t v_i_3315_, lean_object* v_b_3316_, lean_object* v___y_3317_, lean_object* v___y_3318_){
_start:
{
uint8_t v___x_3320_; 
v___x_3320_ = lean_usize_dec_lt(v_i_3315_, v_sz_3314_);
if (v___x_3320_ == 0)
{
lean_object* v___x_3321_; 
v___x_3321_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3321_, 0, v_b_3316_);
return v___x_3321_;
}
else
{
lean_object* v___x_3322_; lean_object* v_a_3323_; lean_object* v___x_3324_; lean_object* v___x_3325_; 
v___x_3322_ = lp_mathlib_Mathlib_Linter_linter_style_openClassical;
v_a_3323_ = lean_array_uget_borrowed(v_as_3313_, v_i_3315_);
v___x_3324_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter_spec__0___closed__2, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter_spec__0___closed__2_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter_spec__0___closed__2);
lean_inc(v_a_3323_);
v___x_3325_ = lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__1(v___x_3322_, v_a_3323_, v___x_3324_, v___y_3317_, v___y_3318_);
if (lean_obj_tag(v___x_3325_) == 0)
{
lean_object* v___x_3326_; size_t v___x_3327_; size_t v___x_3328_; 
lean_dec_ref_known(v___x_3325_, 1);
v___x_3326_ = lean_box(0);
v___x_3327_ = ((size_t)1ULL);
v___x_3328_ = lean_usize_add(v_i_3315_, v___x_3327_);
v_i_3315_ = v___x_3328_;
v_b_3316_ = v___x_3326_;
goto _start;
}
else
{
return v___x_3325_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter_spec__0___boxed(lean_object* v_as_3330_, lean_object* v_sz_3331_, lean_object* v_i_3332_, lean_object* v_b_3333_, lean_object* v___y_3334_, lean_object* v___y_3335_, lean_object* v___y_3336_){
_start:
{
size_t v_sz_boxed_3337_; size_t v_i_boxed_3338_; lean_object* v_res_3339_; 
v_sz_boxed_3337_ = lean_unbox_usize(v_sz_3331_);
lean_dec(v_sz_3331_);
v_i_boxed_3338_ = lean_unbox_usize(v_i_3332_);
lean_dec(v_i_3332_);
v_res_3339_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter_spec__0(v_as_3330_, v_sz_boxed_3337_, v_i_boxed_3338_, v_b_3333_, v___y_3334_, v___y_3335_);
lean_dec(v___y_3335_);
lean_dec_ref(v___y_3334_);
lean_dec_ref(v_as_3330_);
return v_res_3339_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter_spec__1(lean_object* v_as_3343_, size_t v_i_3344_, size_t v_stop_3345_, lean_object* v_b_3346_){
_start:
{
lean_object* v___y_3348_; uint8_t v___x_3352_; 
v___x_3352_ = lean_usize_dec_eq(v_i_3344_, v_stop_3345_);
if (v___x_3352_ == 0)
{
lean_object* v___x_3353_; lean_object* v___x_3354_; lean_object* v___x_3355_; uint8_t v___x_3356_; 
v___x_3353_ = lean_array_uget_borrowed(v_as_3343_, v_i_3344_);
v___x_3354_ = l_Lean_TSyntax_getId(v___x_3353_);
v___x_3355_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter_spec__1___closed__1));
v___x_3356_ = lean_name_eq(v___x_3354_, v___x_3355_);
lean_dec(v___x_3354_);
if (v___x_3356_ == 0)
{
v___y_3348_ = v_b_3346_;
goto v___jp_3347_;
}
else
{
lean_object* v___x_3357_; 
lean_inc(v___x_3353_);
v___x_3357_ = lean_array_push(v_b_3346_, v___x_3353_);
v___y_3348_ = v___x_3357_;
goto v___jp_3347_;
}
}
else
{
return v_b_3346_;
}
v___jp_3347_:
{
size_t v___x_3349_; size_t v___x_3350_; 
v___x_3349_ = ((size_t)1ULL);
v___x_3350_ = lean_usize_add(v_i_3344_, v___x_3349_);
v_i_3344_ = v___x_3350_;
v_b_3346_ = v___y_3348_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter_spec__1___boxed(lean_object* v_as_3358_, lean_object* v_i_3359_, lean_object* v_stop_3360_, lean_object* v_b_3361_){
_start:
{
size_t v_i_boxed_3362_; size_t v_stop_boxed_3363_; lean_object* v_res_3364_; 
v_i_boxed_3362_ = lean_unbox_usize(v_i_3359_);
lean_dec(v_i_3359_);
v_stop_boxed_3363_ = lean_unbox_usize(v_stop_3360_);
lean_dec(v_stop_3360_);
v_res_3364_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter_spec__1(v_as_3358_, v_i_boxed_3362_, v_stop_boxed_3363_, v_b_3361_);
lean_dec_ref(v_as_3358_);
return v_res_3364_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter___lam__0(lean_object* v_stx_3365_, lean_object* v___y_3366_, lean_object* v___y_3367_){
_start:
{
lean_object* v___x_3369_; lean_object* v_a_3370_; lean_object* v___x_3372_; uint8_t v_isShared_3373_; uint8_t v_isSharedCheck_3413_; 
v___x_3369_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__0(v___y_3366_, v___y_3367_);
v_a_3370_ = lean_ctor_get(v___x_3369_, 0);
v_isSharedCheck_3413_ = !lean_is_exclusive(v___x_3369_);
if (v_isSharedCheck_3413_ == 0)
{
v___x_3372_ = v___x_3369_;
v_isShared_3373_ = v_isSharedCheck_3413_;
goto v_resetjp_3371_;
}
else
{
lean_inc(v_a_3370_);
lean_dec(v___x_3369_);
v___x_3372_ = lean_box(0);
v_isShared_3373_ = v_isSharedCheck_3413_;
goto v_resetjp_3371_;
}
v_resetjp_3371_:
{
lean_object* v___x_3374_; uint8_t v___x_3375_; 
v___x_3374_ = lp_mathlib_Mathlib_Linter_linter_style_openClassical;
v___x_3375_ = l_Lean_Linter_getLinterValue(v___x_3374_, v_a_3370_);
lean_dec(v_a_3370_);
if (v___x_3375_ == 0)
{
lean_object* v___x_3376_; lean_object* v___x_3378_; 
lean_dec(v_stx_3365_);
v___x_3376_ = lean_box(0);
if (v_isShared_3373_ == 0)
{
lean_ctor_set(v___x_3372_, 0, v___x_3376_);
v___x_3378_ = v___x_3372_;
goto v_reusejp_3377_;
}
else
{
lean_object* v_reuseFailAlloc_3379_; 
v_reuseFailAlloc_3379_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3379_, 0, v___x_3376_);
v___x_3378_ = v_reuseFailAlloc_3379_;
goto v_reusejp_3377_;
}
v_reusejp_3377_:
{
return v___x_3378_;
}
}
else
{
lean_object* v___x_3380_; lean_object* v___y_3382_; lean_object* v_messages_3395_; uint8_t v___x_3396_; 
v___x_3380_ = lean_st_ref_get(v___y_3367_);
v_messages_3395_ = lean_ctor_get(v___x_3380_, 1);
lean_inc_ref(v_messages_3395_);
lean_dec(v___x_3380_);
v___x_3396_ = l_Lean_MessageLog_hasErrors(v_messages_3395_);
lean_dec_ref(v_messages_3395_);
if (v___x_3396_ == 0)
{
lean_object* v___x_3397_; lean_object* v___x_3398_; lean_object* v___x_3399_; lean_object* v___x_3400_; uint8_t v___x_3401_; 
lean_del_object(v___x_3372_);
v___x_3397_ = lp_mathlib_Mathlib_Linter_Style_openClassical_extractOpenNames(v_stx_3365_);
v___x_3398_ = lean_unsigned_to_nat(0u);
v___x_3399_ = lean_array_get_size(v___x_3397_);
v___x_3400_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_findCDot___closed__1));
v___x_3401_ = lean_nat_dec_lt(v___x_3398_, v___x_3399_);
if (v___x_3401_ == 0)
{
lean_dec_ref(v___x_3397_);
v___y_3382_ = v___x_3400_;
goto v___jp_3381_;
}
else
{
uint8_t v___x_3402_; 
v___x_3402_ = lean_nat_dec_le(v___x_3399_, v___x_3399_);
if (v___x_3402_ == 0)
{
if (v___x_3401_ == 0)
{
lean_dec_ref(v___x_3397_);
v___y_3382_ = v___x_3400_;
goto v___jp_3381_;
}
else
{
size_t v___x_3403_; size_t v___x_3404_; lean_object* v___x_3405_; 
v___x_3403_ = ((size_t)0ULL);
v___x_3404_ = lean_usize_of_nat(v___x_3399_);
v___x_3405_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter_spec__1(v___x_3397_, v___x_3403_, v___x_3404_, v___x_3400_);
lean_dec_ref(v___x_3397_);
v___y_3382_ = v___x_3405_;
goto v___jp_3381_;
}
}
else
{
size_t v___x_3406_; size_t v___x_3407_; lean_object* v___x_3408_; 
v___x_3406_ = ((size_t)0ULL);
v___x_3407_ = lean_usize_of_nat(v___x_3399_);
v___x_3408_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter_spec__1(v___x_3397_, v___x_3406_, v___x_3407_, v___x_3400_);
lean_dec_ref(v___x_3397_);
v___y_3382_ = v___x_3408_;
goto v___jp_3381_;
}
}
}
else
{
lean_object* v___x_3409_; lean_object* v___x_3411_; 
lean_dec(v_stx_3365_);
v___x_3409_ = lean_box(0);
if (v_isShared_3373_ == 0)
{
lean_ctor_set(v___x_3372_, 0, v___x_3409_);
v___x_3411_ = v___x_3372_;
goto v_reusejp_3410_;
}
else
{
lean_object* v_reuseFailAlloc_3412_; 
v_reuseFailAlloc_3412_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3412_, 0, v___x_3409_);
v___x_3411_ = v_reuseFailAlloc_3412_;
goto v_reusejp_3410_;
}
v_reusejp_3410_:
{
return v___x_3411_;
}
}
v___jp_3381_:
{
lean_object* v___x_3383_; size_t v_sz_3384_; size_t v___x_3385_; lean_object* v___x_3386_; 
v___x_3383_ = lean_box(0);
v_sz_3384_ = lean_array_size(v___y_3382_);
v___x_3385_ = ((size_t)0ULL);
v___x_3386_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter_spec__0(v___y_3382_, v_sz_3384_, v___x_3385_, v___x_3383_, v___y_3366_, v___y_3367_);
lean_dec_ref(v___y_3382_);
if (lean_obj_tag(v___x_3386_) == 0)
{
lean_object* v___x_3388_; uint8_t v_isShared_3389_; uint8_t v_isSharedCheck_3393_; 
v_isSharedCheck_3393_ = !lean_is_exclusive(v___x_3386_);
if (v_isSharedCheck_3393_ == 0)
{
lean_object* v_unused_3394_; 
v_unused_3394_ = lean_ctor_get(v___x_3386_, 0);
lean_dec(v_unused_3394_);
v___x_3388_ = v___x_3386_;
v_isShared_3389_ = v_isSharedCheck_3393_;
goto v_resetjp_3387_;
}
else
{
lean_dec(v___x_3386_);
v___x_3388_ = lean_box(0);
v_isShared_3389_ = v_isSharedCheck_3393_;
goto v_resetjp_3387_;
}
v_resetjp_3387_:
{
lean_object* v___x_3391_; 
if (v_isShared_3389_ == 0)
{
lean_ctor_set(v___x_3388_, 0, v___x_3383_);
v___x_3391_ = v___x_3388_;
goto v_reusejp_3390_;
}
else
{
lean_object* v_reuseFailAlloc_3392_; 
v_reuseFailAlloc_3392_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3392_, 0, v___x_3383_);
v___x_3391_ = v_reuseFailAlloc_3392_;
goto v_reusejp_3390_;
}
v_reusejp_3390_:
{
return v___x_3391_;
}
}
}
else
{
return v___x_3386_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter___lam__0___boxed(lean_object* v_stx_3414_, lean_object* v___y_3415_, lean_object* v___y_3416_, lean_object* v___y_3417_){
_start:
{
lean_object* v_res_3418_; 
v_res_3418_ = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter___lam__0(v_stx_3414_, v___y_3415_, v___y_3416_);
lean_dec(v___y_3416_);
lean_dec_ref(v___y_3415_);
return v_res_3418_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_initFn_00___x40_Mathlib_Tactic_Linter_Style_273924139____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_3432_; lean_object* v___x_3433_; 
v___x_3432_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_openClassicalLinter));
v___x_3433_ = l_Lean_Elab_Command_addLinter(v___x_3432_);
return v___x_3433_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_initFn_00___x40_Mathlib_Tactic_Linter_Style_273924139____hygCtx___hyg_2____boxed(lean_object* v_a_3434_){
_start:
{
lean_object* v_res_3435_; 
v_res_3435_ = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_initFn_00___x40_Mathlib_Tactic_Linter_Style_273924139____hygCtx___hyg_2_();
return v_res_3435_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_4166288182____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_3454_; lean_object* v___x_3455_; lean_object* v___x_3456_; lean_object* v___x_3457_; 
v___x_3454_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Style_4166288182____hygCtx___hyg_4_));
v___x_3455_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Style_4166288182____hygCtx___hyg_4_));
v___x_3456_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Style_4166288182____hygCtx___hyg_4_));
v___x_3457_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4__spec__0(v___x_3454_, v___x_3455_, v___x_3456_);
return v___x_3457_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_4166288182____hygCtx___hyg_4____boxed(lean_object* v_a_3458_){
_start:
{
lean_object* v_res_3459_; 
v_res_3459_ = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_4166288182____hygCtx___hyg_4_();
return v_res_3459_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__0___redArg(lean_object* v_e_3460_, lean_object* v___y_3461_){
_start:
{
uint8_t v___x_3463_; 
v___x_3463_ = l_Lean_Expr_hasMVar(v_e_3460_);
if (v___x_3463_ == 0)
{
lean_object* v___x_3464_; 
v___x_3464_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3464_, 0, v_e_3460_);
return v___x_3464_;
}
else
{
lean_object* v___x_3465_; lean_object* v_mctx_3466_; lean_object* v___x_3467_; lean_object* v_fst_3468_; lean_object* v_snd_3469_; lean_object* v___x_3470_; lean_object* v_cache_3471_; lean_object* v_zetaDeltaFVarIds_3472_; lean_object* v_postponed_3473_; lean_object* v_diag_3474_; lean_object* v___x_3476_; uint8_t v_isShared_3477_; uint8_t v_isSharedCheck_3483_; 
v___x_3465_ = lean_st_ref_get(v___y_3461_);
v_mctx_3466_ = lean_ctor_get(v___x_3465_, 0);
lean_inc_ref(v_mctx_3466_);
lean_dec(v___x_3465_);
v___x_3467_ = l_Lean_instantiateMVarsCore(v_mctx_3466_, v_e_3460_);
v_fst_3468_ = lean_ctor_get(v___x_3467_, 0);
lean_inc(v_fst_3468_);
v_snd_3469_ = lean_ctor_get(v___x_3467_, 1);
lean_inc(v_snd_3469_);
lean_dec_ref(v___x_3467_);
v___x_3470_ = lean_st_ref_take(v___y_3461_);
v_cache_3471_ = lean_ctor_get(v___x_3470_, 1);
v_zetaDeltaFVarIds_3472_ = lean_ctor_get(v___x_3470_, 2);
v_postponed_3473_ = lean_ctor_get(v___x_3470_, 3);
v_diag_3474_ = lean_ctor_get(v___x_3470_, 4);
v_isSharedCheck_3483_ = !lean_is_exclusive(v___x_3470_);
if (v_isSharedCheck_3483_ == 0)
{
lean_object* v_unused_3484_; 
v_unused_3484_ = lean_ctor_get(v___x_3470_, 0);
lean_dec(v_unused_3484_);
v___x_3476_ = v___x_3470_;
v_isShared_3477_ = v_isSharedCheck_3483_;
goto v_resetjp_3475_;
}
else
{
lean_inc(v_diag_3474_);
lean_inc(v_postponed_3473_);
lean_inc(v_zetaDeltaFVarIds_3472_);
lean_inc(v_cache_3471_);
lean_dec(v___x_3470_);
v___x_3476_ = lean_box(0);
v_isShared_3477_ = v_isSharedCheck_3483_;
goto v_resetjp_3475_;
}
v_resetjp_3475_:
{
lean_object* v___x_3479_; 
if (v_isShared_3477_ == 0)
{
lean_ctor_set(v___x_3476_, 0, v_snd_3469_);
v___x_3479_ = v___x_3476_;
goto v_reusejp_3478_;
}
else
{
lean_object* v_reuseFailAlloc_3482_; 
v_reuseFailAlloc_3482_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3482_, 0, v_snd_3469_);
lean_ctor_set(v_reuseFailAlloc_3482_, 1, v_cache_3471_);
lean_ctor_set(v_reuseFailAlloc_3482_, 2, v_zetaDeltaFVarIds_3472_);
lean_ctor_set(v_reuseFailAlloc_3482_, 3, v_postponed_3473_);
lean_ctor_set(v_reuseFailAlloc_3482_, 4, v_diag_3474_);
v___x_3479_ = v_reuseFailAlloc_3482_;
goto v_reusejp_3478_;
}
v_reusejp_3478_:
{
lean_object* v___x_3480_; lean_object* v___x_3481_; 
v___x_3480_ = lean_st_ref_set(v___y_3461_, v___x_3479_);
v___x_3481_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3481_, 0, v_fst_3468_);
return v___x_3481_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__0___redArg___boxed(lean_object* v_e_3485_, lean_object* v___y_3486_, lean_object* v___y_3487_){
_start:
{
lean_object* v_res_3488_; 
v_res_3488_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__0___redArg(v_e_3485_, v___y_3486_);
lean_dec(v___y_3486_);
return v_res_3488_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__0(lean_object* v_e_3489_, lean_object* v___y_3490_, lean_object* v___y_3491_, lean_object* v___y_3492_, lean_object* v___y_3493_, lean_object* v___y_3494_, lean_object* v___y_3495_, lean_object* v___y_3496_, lean_object* v___y_3497_){
_start:
{
lean_object* v___x_3499_; 
v___x_3499_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__0___redArg(v_e_3489_, v___y_3495_);
return v___x_3499_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__0___boxed(lean_object* v_e_3500_, lean_object* v___y_3501_, lean_object* v___y_3502_, lean_object* v___y_3503_, lean_object* v___y_3504_, lean_object* v___y_3505_, lean_object* v___y_3506_, lean_object* v___y_3507_, lean_object* v___y_3508_, lean_object* v___y_3509_){
_start:
{
lean_object* v_res_3510_; 
v_res_3510_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__0(v_e_3500_, v___y_3501_, v___y_3502_, v___y_3503_, v___y_3504_, v___y_3505_, v___y_3506_, v___y_3507_, v___y_3508_);
lean_dec(v___y_3508_);
lean_dec_ref(v___y_3507_);
lean_dec(v___y_3506_);
lean_dec_ref(v___y_3505_);
lean_dec(v___y_3504_);
lean_dec_ref(v___y_3503_);
lean_dec(v___y_3502_);
lean_dec_ref(v___y_3501_);
return v_res_3510_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2_spec__3_spec__4_spec__6(lean_object* v_msgData_3511_, lean_object* v___y_3512_, lean_object* v___y_3513_, lean_object* v___y_3514_, lean_object* v___y_3515_){
_start:
{
lean_object* v___x_3517_; lean_object* v_env_3518_; lean_object* v___x_3519_; lean_object* v_mctx_3520_; lean_object* v_lctx_3521_; lean_object* v_options_3522_; lean_object* v___x_3523_; lean_object* v___x_3524_; lean_object* v___x_3525_; 
v___x_3517_ = lean_st_ref_get(v___y_3515_);
v_env_3518_ = lean_ctor_get(v___x_3517_, 0);
lean_inc_ref(v_env_3518_);
lean_dec(v___x_3517_);
v___x_3519_ = lean_st_ref_get(v___y_3513_);
v_mctx_3520_ = lean_ctor_get(v___x_3519_, 0);
lean_inc_ref(v_mctx_3520_);
lean_dec(v___x_3519_);
v_lctx_3521_ = lean_ctor_get(v___y_3512_, 2);
v_options_3522_ = lean_ctor_get(v___y_3514_, 2);
lean_inc_ref(v_options_3522_);
lean_inc_ref(v_lctx_3521_);
v___x_3523_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_3523_, 0, v_env_3518_);
lean_ctor_set(v___x_3523_, 1, v_mctx_3520_);
lean_ctor_set(v___x_3523_, 2, v_lctx_3521_);
lean_ctor_set(v___x_3523_, 3, v_options_3522_);
v___x_3524_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_3524_, 0, v___x_3523_);
lean_ctor_set(v___x_3524_, 1, v_msgData_3511_);
v___x_3525_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3525_, 0, v___x_3524_);
return v___x_3525_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2_spec__3_spec__4_spec__6___boxed(lean_object* v_msgData_3526_, lean_object* v___y_3527_, lean_object* v___y_3528_, lean_object* v___y_3529_, lean_object* v___y_3530_, lean_object* v___y_3531_){
_start:
{
lean_object* v_res_3532_; 
v_res_3532_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2_spec__3_spec__4_spec__6(v_msgData_3526_, v___y_3527_, v___y_3528_, v___y_3529_, v___y_3530_);
lean_dec(v___y_3530_);
lean_dec_ref(v___y_3529_);
lean_dec(v___y_3528_);
lean_dec_ref(v___y_3527_);
return v_res_3532_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2_spec__3_spec__4___redArg___lam__0(uint8_t v___y_3539_, uint8_t v_suppressElabErrors_3540_, lean_object* v_x_3541_){
_start:
{
if (lean_obj_tag(v_x_3541_) == 1)
{
lean_object* v_pre_3542_; 
v_pre_3542_ = lean_ctor_get(v_x_3541_, 0);
switch(lean_obj_tag(v_pre_3542_))
{
case 1:
{
lean_object* v_pre_3543_; 
v_pre_3543_ = lean_ctor_get(v_pre_3542_, 0);
switch(lean_obj_tag(v_pre_3543_))
{
case 0:
{
lean_object* v_str_3544_; lean_object* v_str_3545_; lean_object* v___x_3546_; uint8_t v___x_3547_; 
v_str_3544_ = lean_ctor_get(v_x_3541_, 1);
v_str_3545_ = lean_ctor_get(v_pre_3542_, 1);
v___x_3546_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2_spec__3_spec__4___redArg___lam__0___closed__0));
v___x_3547_ = lean_string_dec_eq(v_str_3545_, v___x_3546_);
if (v___x_3547_ == 0)
{
lean_object* v___x_3548_; uint8_t v___x_3549_; 
v___x_3548_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_parseSetOption___closed__7));
v___x_3549_ = lean_string_dec_eq(v_str_3545_, v___x_3548_);
if (v___x_3549_ == 0)
{
return v___y_3539_;
}
else
{
lean_object* v___x_3550_; uint8_t v___x_3551_; 
v___x_3550_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2_spec__3_spec__4___redArg___lam__0___closed__1));
v___x_3551_ = lean_string_dec_eq(v_str_3544_, v___x_3550_);
if (v___x_3551_ == 0)
{
return v___y_3539_;
}
else
{
return v_suppressElabErrors_3540_;
}
}
}
else
{
lean_object* v___x_3552_; uint8_t v___x_3553_; 
v___x_3552_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2_spec__3_spec__4___redArg___lam__0___closed__2));
v___x_3553_ = lean_string_dec_eq(v_str_3544_, v___x_3552_);
if (v___x_3553_ == 0)
{
return v___y_3539_;
}
else
{
return v_suppressElabErrors_3540_;
}
}
}
case 1:
{
lean_object* v_pre_3554_; 
v_pre_3554_ = lean_ctor_get(v_pre_3543_, 0);
if (lean_obj_tag(v_pre_3554_) == 0)
{
lean_object* v_str_3555_; lean_object* v_str_3556_; lean_object* v_str_3557_; lean_object* v___x_3558_; uint8_t v___x_3559_; 
v_str_3555_ = lean_ctor_get(v_x_3541_, 1);
v_str_3556_ = lean_ctor_get(v_pre_3542_, 1);
v_str_3557_ = lean_ctor_get(v_pre_3543_, 1);
v___x_3558_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2_spec__3_spec__4___redArg___lam__0___closed__3));
v___x_3559_ = lean_string_dec_eq(v_str_3557_, v___x_3558_);
if (v___x_3559_ == 0)
{
return v___y_3539_;
}
else
{
lean_object* v___x_3560_; uint8_t v___x_3561_; 
v___x_3560_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2_spec__3_spec__4___redArg___lam__0___closed__4));
v___x_3561_ = lean_string_dec_eq(v_str_3556_, v___x_3560_);
if (v___x_3561_ == 0)
{
return v___y_3539_;
}
else
{
lean_object* v___x_3562_; uint8_t v___x_3563_; 
v___x_3562_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2_spec__3_spec__4___redArg___lam__0___closed__5));
v___x_3563_ = lean_string_dec_eq(v_str_3555_, v___x_3562_);
if (v___x_3563_ == 0)
{
return v___y_3539_;
}
else
{
return v_suppressElabErrors_3540_;
}
}
}
}
else
{
return v___y_3539_;
}
}
default: 
{
return v___y_3539_;
}
}
}
case 0:
{
lean_object* v_str_3564_; lean_object* v___x_3565_; uint8_t v___x_3566_; 
v_str_3564_ = lean_ctor_get(v_x_3541_, 1);
v___x_3565_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4___lam__0___closed__0));
v___x_3566_ = lean_string_dec_eq(v_str_3564_, v___x_3565_);
if (v___x_3566_ == 0)
{
return v___y_3539_;
}
else
{
return v_suppressElabErrors_3540_;
}
}
default: 
{
return v___y_3539_;
}
}
}
else
{
return v___y_3539_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2_spec__3_spec__4___redArg___lam__0___boxed(lean_object* v___y_3567_, lean_object* v_suppressElabErrors_3568_, lean_object* v_x_3569_){
_start:
{
uint8_t v___y_11197__boxed_3570_; uint8_t v_suppressElabErrors_boxed_3571_; uint8_t v_res_3572_; lean_object* v_r_3573_; 
v___y_11197__boxed_3570_ = lean_unbox(v___y_3567_);
v_suppressElabErrors_boxed_3571_ = lean_unbox(v_suppressElabErrors_3568_);
v_res_3572_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2_spec__3_spec__4___redArg___lam__0(v___y_11197__boxed_3570_, v_suppressElabErrors_boxed_3571_, v_x_3569_);
lean_dec(v_x_3569_);
v_r_3573_ = lean_box(v_res_3572_);
return v_r_3573_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2_spec__3_spec__4___redArg(lean_object* v_ref_3574_, lean_object* v_msgData_3575_, uint8_t v_severity_3576_, uint8_t v_isSilent_3577_, lean_object* v___y_3578_, lean_object* v___y_3579_, lean_object* v___y_3580_, lean_object* v___y_3581_){
_start:
{
lean_object* v___y_3584_; lean_object* v___y_3585_; lean_object* v___y_3586_; lean_object* v___y_3587_; uint8_t v___y_3588_; lean_object* v___y_3589_; uint8_t v___y_3590_; lean_object* v___y_3591_; lean_object* v___y_3592_; lean_object* v___y_3620_; uint8_t v___y_3621_; lean_object* v___y_3622_; lean_object* v___y_3623_; uint8_t v___y_3624_; lean_object* v___y_3625_; uint8_t v___y_3626_; lean_object* v___y_3627_; lean_object* v___y_3645_; uint8_t v___y_3646_; lean_object* v___y_3647_; lean_object* v___y_3648_; lean_object* v___y_3649_; uint8_t v___y_3650_; uint8_t v___y_3651_; lean_object* v___y_3652_; lean_object* v___y_3656_; uint8_t v___y_3657_; lean_object* v___y_3658_; lean_object* v___y_3659_; lean_object* v___y_3660_; uint8_t v___y_3661_; uint8_t v___y_3662_; uint8_t v___x_3667_; uint8_t v___y_3669_; lean_object* v___y_3670_; lean_object* v___y_3671_; lean_object* v___y_3672_; lean_object* v___y_3673_; uint8_t v___y_3674_; uint8_t v___y_3675_; uint8_t v___y_3677_; uint8_t v___x_3692_; 
v___x_3667_ = 2;
v___x_3692_ = l_Lean_instBEqMessageSeverity_beq(v_severity_3576_, v___x_3667_);
if (v___x_3692_ == 0)
{
v___y_3677_ = v___x_3692_;
goto v___jp_3676_;
}
else
{
uint8_t v___x_3693_; 
lean_inc_ref(v_msgData_3575_);
v___x_3693_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_3575_);
v___y_3677_ = v___x_3693_;
goto v___jp_3676_;
}
v___jp_3583_:
{
lean_object* v___x_3593_; lean_object* v_currNamespace_3594_; lean_object* v_openDecls_3595_; lean_object* v_env_3596_; lean_object* v_nextMacroScope_3597_; lean_object* v_ngen_3598_; lean_object* v_auxDeclNGen_3599_; lean_object* v_traceState_3600_; lean_object* v_cache_3601_; lean_object* v_messages_3602_; lean_object* v_infoState_3603_; lean_object* v_snapshotTasks_3604_; lean_object* v___x_3606_; uint8_t v_isShared_3607_; uint8_t v_isSharedCheck_3618_; 
v___x_3593_ = lean_st_ref_take(v___y_3592_);
v_currNamespace_3594_ = lean_ctor_get(v___y_3591_, 6);
v_openDecls_3595_ = lean_ctor_get(v___y_3591_, 7);
v_env_3596_ = lean_ctor_get(v___x_3593_, 0);
v_nextMacroScope_3597_ = lean_ctor_get(v___x_3593_, 1);
v_ngen_3598_ = lean_ctor_get(v___x_3593_, 2);
v_auxDeclNGen_3599_ = lean_ctor_get(v___x_3593_, 3);
v_traceState_3600_ = lean_ctor_get(v___x_3593_, 4);
v_cache_3601_ = lean_ctor_get(v___x_3593_, 5);
v_messages_3602_ = lean_ctor_get(v___x_3593_, 6);
v_infoState_3603_ = lean_ctor_get(v___x_3593_, 7);
v_snapshotTasks_3604_ = lean_ctor_get(v___x_3593_, 8);
v_isSharedCheck_3618_ = !lean_is_exclusive(v___x_3593_);
if (v_isSharedCheck_3618_ == 0)
{
v___x_3606_ = v___x_3593_;
v_isShared_3607_ = v_isSharedCheck_3618_;
goto v_resetjp_3605_;
}
else
{
lean_inc(v_snapshotTasks_3604_);
lean_inc(v_infoState_3603_);
lean_inc(v_messages_3602_);
lean_inc(v_cache_3601_);
lean_inc(v_traceState_3600_);
lean_inc(v_auxDeclNGen_3599_);
lean_inc(v_ngen_3598_);
lean_inc(v_nextMacroScope_3597_);
lean_inc(v_env_3596_);
lean_dec(v___x_3593_);
v___x_3606_ = lean_box(0);
v_isShared_3607_ = v_isSharedCheck_3618_;
goto v_resetjp_3605_;
}
v_resetjp_3605_:
{
lean_object* v___x_3608_; lean_object* v___x_3609_; lean_object* v___x_3610_; lean_object* v___x_3611_; lean_object* v___x_3613_; 
lean_inc(v_openDecls_3595_);
lean_inc(v_currNamespace_3594_);
v___x_3608_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3608_, 0, v_currNamespace_3594_);
lean_ctor_set(v___x_3608_, 1, v_openDecls_3595_);
v___x_3609_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_3609_, 0, v___x_3608_);
lean_ctor_set(v___x_3609_, 1, v___y_3585_);
lean_inc_ref(v___y_3587_);
lean_inc_ref(v___y_3584_);
v___x_3610_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_3610_, 0, v___y_3584_);
lean_ctor_set(v___x_3610_, 1, v___y_3586_);
lean_ctor_set(v___x_3610_, 2, v___y_3589_);
lean_ctor_set(v___x_3610_, 3, v___y_3587_);
lean_ctor_set(v___x_3610_, 4, v___x_3609_);
lean_ctor_set_uint8(v___x_3610_, sizeof(void*)*5, v___y_3588_);
lean_ctor_set_uint8(v___x_3610_, sizeof(void*)*5 + 1, v___y_3590_);
lean_ctor_set_uint8(v___x_3610_, sizeof(void*)*5 + 2, v_isSilent_3577_);
v___x_3611_ = l_Lean_MessageLog_add(v___x_3610_, v_messages_3602_);
if (v_isShared_3607_ == 0)
{
lean_ctor_set(v___x_3606_, 6, v___x_3611_);
v___x_3613_ = v___x_3606_;
goto v_reusejp_3612_;
}
else
{
lean_object* v_reuseFailAlloc_3617_; 
v_reuseFailAlloc_3617_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_3617_, 0, v_env_3596_);
lean_ctor_set(v_reuseFailAlloc_3617_, 1, v_nextMacroScope_3597_);
lean_ctor_set(v_reuseFailAlloc_3617_, 2, v_ngen_3598_);
lean_ctor_set(v_reuseFailAlloc_3617_, 3, v_auxDeclNGen_3599_);
lean_ctor_set(v_reuseFailAlloc_3617_, 4, v_traceState_3600_);
lean_ctor_set(v_reuseFailAlloc_3617_, 5, v_cache_3601_);
lean_ctor_set(v_reuseFailAlloc_3617_, 6, v___x_3611_);
lean_ctor_set(v_reuseFailAlloc_3617_, 7, v_infoState_3603_);
lean_ctor_set(v_reuseFailAlloc_3617_, 8, v_snapshotTasks_3604_);
v___x_3613_ = v_reuseFailAlloc_3617_;
goto v_reusejp_3612_;
}
v_reusejp_3612_:
{
lean_object* v___x_3614_; lean_object* v___x_3615_; lean_object* v___x_3616_; 
v___x_3614_ = lean_st_ref_set(v___y_3592_, v___x_3613_);
v___x_3615_ = lean_box(0);
v___x_3616_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3616_, 0, v___x_3615_);
return v___x_3616_;
}
}
}
v___jp_3619_:
{
lean_object* v___x_3628_; lean_object* v___x_3629_; lean_object* v_a_3630_; lean_object* v___x_3632_; uint8_t v_isShared_3633_; uint8_t v_isSharedCheck_3643_; 
v___x_3628_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_3575_);
v___x_3629_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2_spec__3_spec__4_spec__6(v___x_3628_, v___y_3578_, v___y_3579_, v___y_3580_, v___y_3581_);
v_a_3630_ = lean_ctor_get(v___x_3629_, 0);
v_isSharedCheck_3643_ = !lean_is_exclusive(v___x_3629_);
if (v_isSharedCheck_3643_ == 0)
{
v___x_3632_ = v___x_3629_;
v_isShared_3633_ = v_isSharedCheck_3643_;
goto v_resetjp_3631_;
}
else
{
lean_inc(v_a_3630_);
lean_dec(v___x_3629_);
v___x_3632_ = lean_box(0);
v_isShared_3633_ = v_isSharedCheck_3643_;
goto v_resetjp_3631_;
}
v_resetjp_3631_:
{
lean_object* v___x_3634_; lean_object* v___x_3635_; lean_object* v___x_3636_; lean_object* v___x_3637_; 
lean_inc_ref_n(v___y_3623_, 2);
v___x_3634_ = l_Lean_FileMap_toPosition(v___y_3623_, v___y_3625_);
lean_dec(v___y_3625_);
v___x_3635_ = l_Lean_FileMap_toPosition(v___y_3623_, v___y_3627_);
lean_dec(v___y_3627_);
v___x_3636_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3636_, 0, v___x_3635_);
v___x_3637_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4___closed__0));
if (v___y_3621_ == 0)
{
lean_del_object(v___x_3632_);
lean_dec_ref(v___y_3620_);
v___y_3584_ = v___y_3622_;
v___y_3585_ = v_a_3630_;
v___y_3586_ = v___x_3634_;
v___y_3587_ = v___x_3637_;
v___y_3588_ = v___y_3624_;
v___y_3589_ = v___x_3636_;
v___y_3590_ = v___y_3626_;
v___y_3591_ = v___y_3580_;
v___y_3592_ = v___y_3581_;
goto v___jp_3583_;
}
else
{
uint8_t v___x_3638_; 
lean_inc(v_a_3630_);
v___x_3638_ = l_Lean_MessageData_hasTag(v___y_3620_, v_a_3630_);
if (v___x_3638_ == 0)
{
lean_object* v___x_3639_; lean_object* v___x_3641_; 
lean_dec_ref_known(v___x_3636_, 1);
lean_dec_ref(v___x_3634_);
lean_dec(v_a_3630_);
v___x_3639_ = lean_box(0);
if (v_isShared_3633_ == 0)
{
lean_ctor_set(v___x_3632_, 0, v___x_3639_);
v___x_3641_ = v___x_3632_;
goto v_reusejp_3640_;
}
else
{
lean_object* v_reuseFailAlloc_3642_; 
v_reuseFailAlloc_3642_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3642_, 0, v___x_3639_);
v___x_3641_ = v_reuseFailAlloc_3642_;
goto v_reusejp_3640_;
}
v_reusejp_3640_:
{
return v___x_3641_;
}
}
else
{
lean_del_object(v___x_3632_);
v___y_3584_ = v___y_3622_;
v___y_3585_ = v_a_3630_;
v___y_3586_ = v___x_3634_;
v___y_3587_ = v___x_3637_;
v___y_3588_ = v___y_3624_;
v___y_3589_ = v___x_3636_;
v___y_3590_ = v___y_3626_;
v___y_3591_ = v___y_3580_;
v___y_3592_ = v___y_3581_;
goto v___jp_3583_;
}
}
}
}
v___jp_3644_:
{
lean_object* v___x_3653_; 
v___x_3653_ = l_Lean_Syntax_getTailPos_x3f(v___y_3648_, v___y_3650_);
lean_dec(v___y_3648_);
if (lean_obj_tag(v___x_3653_) == 0)
{
lean_inc(v___y_3652_);
v___y_3620_ = v___y_3645_;
v___y_3621_ = v___y_3646_;
v___y_3622_ = v___y_3647_;
v___y_3623_ = v___y_3649_;
v___y_3624_ = v___y_3650_;
v___y_3625_ = v___y_3652_;
v___y_3626_ = v___y_3651_;
v___y_3627_ = v___y_3652_;
goto v___jp_3619_;
}
else
{
lean_object* v_val_3654_; 
v_val_3654_ = lean_ctor_get(v___x_3653_, 0);
lean_inc(v_val_3654_);
lean_dec_ref_known(v___x_3653_, 1);
v___y_3620_ = v___y_3645_;
v___y_3621_ = v___y_3646_;
v___y_3622_ = v___y_3647_;
v___y_3623_ = v___y_3649_;
v___y_3624_ = v___y_3650_;
v___y_3625_ = v___y_3652_;
v___y_3626_ = v___y_3651_;
v___y_3627_ = v_val_3654_;
goto v___jp_3619_;
}
}
v___jp_3655_:
{
lean_object* v_ref_3663_; lean_object* v___x_3664_; 
v_ref_3663_ = l_Lean_replaceRef(v_ref_3574_, v___y_3659_);
v___x_3664_ = l_Lean_Syntax_getPos_x3f(v_ref_3663_, v___y_3661_);
if (lean_obj_tag(v___x_3664_) == 0)
{
lean_object* v___x_3665_; 
v___x_3665_ = lean_unsigned_to_nat(0u);
v___y_3645_ = v___y_3656_;
v___y_3646_ = v___y_3657_;
v___y_3647_ = v___y_3658_;
v___y_3648_ = v_ref_3663_;
v___y_3649_ = v___y_3660_;
v___y_3650_ = v___y_3661_;
v___y_3651_ = v___y_3662_;
v___y_3652_ = v___x_3665_;
goto v___jp_3644_;
}
else
{
lean_object* v_val_3666_; 
v_val_3666_ = lean_ctor_get(v___x_3664_, 0);
lean_inc(v_val_3666_);
lean_dec_ref_known(v___x_3664_, 1);
v___y_3645_ = v___y_3656_;
v___y_3646_ = v___y_3657_;
v___y_3647_ = v___y_3658_;
v___y_3648_ = v_ref_3663_;
v___y_3649_ = v___y_3660_;
v___y_3650_ = v___y_3661_;
v___y_3651_ = v___y_3662_;
v___y_3652_ = v_val_3666_;
goto v___jp_3644_;
}
}
v___jp_3668_:
{
if (v___y_3675_ == 0)
{
v___y_3656_ = v___y_3673_;
v___y_3657_ = v___y_3669_;
v___y_3658_ = v___y_3670_;
v___y_3659_ = v___y_3671_;
v___y_3660_ = v___y_3672_;
v___y_3661_ = v___y_3674_;
v___y_3662_ = v_severity_3576_;
goto v___jp_3655_;
}
else
{
v___y_3656_ = v___y_3673_;
v___y_3657_ = v___y_3669_;
v___y_3658_ = v___y_3670_;
v___y_3659_ = v___y_3671_;
v___y_3660_ = v___y_3672_;
v___y_3661_ = v___y_3674_;
v___y_3662_ = v___x_3667_;
goto v___jp_3655_;
}
}
v___jp_3676_:
{
if (v___y_3677_ == 0)
{
lean_object* v_fileName_3678_; lean_object* v_fileMap_3679_; lean_object* v_options_3680_; lean_object* v_ref_3681_; uint8_t v_suppressElabErrors_3682_; lean_object* v___x_3683_; lean_object* v___x_3684_; lean_object* v___f_3685_; uint8_t v___x_3686_; uint8_t v___x_3687_; 
v_fileName_3678_ = lean_ctor_get(v___y_3580_, 0);
v_fileMap_3679_ = lean_ctor_get(v___y_3580_, 1);
v_options_3680_ = lean_ctor_get(v___y_3580_, 2);
v_ref_3681_ = lean_ctor_get(v___y_3580_, 5);
v_suppressElabErrors_3682_ = lean_ctor_get_uint8(v___y_3580_, sizeof(void*)*14 + 1);
v___x_3683_ = lean_box(v___y_3677_);
v___x_3684_ = lean_box(v_suppressElabErrors_3682_);
v___f_3685_ = lean_alloc_closure((void*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2_spec__3_spec__4___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_3685_, 0, v___x_3683_);
lean_closure_set(v___f_3685_, 1, v___x_3684_);
v___x_3686_ = 1;
v___x_3687_ = l_Lean_instBEqMessageSeverity_beq(v_severity_3576_, v___x_3686_);
if (v___x_3687_ == 0)
{
v___y_3669_ = v_suppressElabErrors_3682_;
v___y_3670_ = v_fileName_3678_;
v___y_3671_ = v_ref_3681_;
v___y_3672_ = v_fileMap_3679_;
v___y_3673_ = v___f_3685_;
v___y_3674_ = v___y_3677_;
v___y_3675_ = v___x_3687_;
goto v___jp_3668_;
}
else
{
lean_object* v___x_3688_; uint8_t v___x_3689_; 
v___x_3688_ = l_Lean_warningAsError;
v___x_3689_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__3_spec__4_spec__6(v_options_3680_, v___x_3688_);
v___y_3669_ = v_suppressElabErrors_3682_;
v___y_3670_ = v_fileName_3678_;
v___y_3671_ = v_ref_3681_;
v___y_3672_ = v_fileMap_3679_;
v___y_3673_ = v___f_3685_;
v___y_3674_ = v___y_3677_;
v___y_3675_ = v___x_3689_;
goto v___jp_3668_;
}
}
else
{
lean_object* v___x_3690_; lean_object* v___x_3691_; 
lean_dec_ref(v_msgData_3575_);
v___x_3690_ = lean_box(0);
v___x_3691_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3691_, 0, v___x_3690_);
return v___x_3691_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2_spec__3_spec__4___redArg___boxed(lean_object* v_ref_3694_, lean_object* v_msgData_3695_, lean_object* v_severity_3696_, lean_object* v_isSilent_3697_, lean_object* v___y_3698_, lean_object* v___y_3699_, lean_object* v___y_3700_, lean_object* v___y_3701_, lean_object* v___y_3702_){
_start:
{
uint8_t v_severity_boxed_3703_; uint8_t v_isSilent_boxed_3704_; lean_object* v_res_3705_; 
v_severity_boxed_3703_ = lean_unbox(v_severity_3696_);
v_isSilent_boxed_3704_ = lean_unbox(v_isSilent_3697_);
v_res_3705_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2_spec__3_spec__4___redArg(v_ref_3694_, v_msgData_3695_, v_severity_boxed_3703_, v_isSilent_boxed_3704_, v___y_3698_, v___y_3699_, v___y_3700_, v___y_3701_);
lean_dec(v___y_3701_);
lean_dec_ref(v___y_3700_);
lean_dec(v___y_3699_);
lean_dec_ref(v___y_3698_);
lean_dec(v_ref_3694_);
return v_res_3705_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2_spec__3(lean_object* v_ref_3706_, lean_object* v_msgData_3707_, lean_object* v___y_3708_, lean_object* v___y_3709_, lean_object* v___y_3710_, lean_object* v___y_3711_, lean_object* v___y_3712_, lean_object* v___y_3713_, lean_object* v___y_3714_, lean_object* v___y_3715_){
_start:
{
uint8_t v___x_3717_; uint8_t v___x_3718_; lean_object* v___x_3719_; 
v___x_3717_ = 1;
v___x_3718_ = 0;
v___x_3719_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2_spec__3_spec__4___redArg(v_ref_3706_, v_msgData_3707_, v___x_3717_, v___x_3718_, v___y_3712_, v___y_3713_, v___y_3714_, v___y_3715_);
return v___x_3719_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2_spec__3___boxed(lean_object* v_ref_3720_, lean_object* v_msgData_3721_, lean_object* v___y_3722_, lean_object* v___y_3723_, lean_object* v___y_3724_, lean_object* v___y_3725_, lean_object* v___y_3726_, lean_object* v___y_3727_, lean_object* v___y_3728_, lean_object* v___y_3729_, lean_object* v___y_3730_){
_start:
{
lean_object* v_res_3731_; 
v_res_3731_ = lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2_spec__3(v_ref_3720_, v_msgData_3721_, v___y_3722_, v___y_3723_, v___y_3724_, v___y_3725_, v___y_3726_, v___y_3727_, v___y_3728_, v___y_3729_);
lean_dec(v___y_3729_);
lean_dec_ref(v___y_3728_);
lean_dec(v___y_3727_);
lean_dec_ref(v___y_3726_);
lean_dec(v___y_3725_);
lean_dec_ref(v___y_3724_);
lean_dec(v___y_3723_);
lean_dec_ref(v___y_3722_);
lean_dec(v_ref_3720_);
return v_res_3731_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2(lean_object* v_linterOption_3732_, lean_object* v_stx_3733_, lean_object* v_msg_3734_, lean_object* v___y_3735_, lean_object* v___y_3736_, lean_object* v___y_3737_, lean_object* v___y_3738_, lean_object* v___y_3739_, lean_object* v___y_3740_, lean_object* v___y_3741_, lean_object* v___y_3742_){
_start:
{
lean_object* v_name_3744_; lean_object* v___x_3746_; uint8_t v_isShared_3747_; uint8_t v_isSharedCheck_3762_; 
v_name_3744_ = lean_ctor_get(v_linterOption_3732_, 0);
v_isSharedCheck_3762_ = !lean_is_exclusive(v_linterOption_3732_);
if (v_isSharedCheck_3762_ == 0)
{
lean_object* v_unused_3763_; 
v_unused_3763_ = lean_ctor_get(v_linterOption_3732_, 1);
lean_dec(v_unused_3763_);
v___x_3746_ = v_linterOption_3732_;
v_isShared_3747_ = v_isSharedCheck_3762_;
goto v_resetjp_3745_;
}
else
{
lean_inc(v_name_3744_);
lean_dec(v_linterOption_3732_);
v___x_3746_ = lean_box(0);
v_isShared_3747_ = v_isSharedCheck_3762_;
goto v_resetjp_3745_;
}
v_resetjp_3745_:
{
lean_object* v___x_3748_; lean_object* v___x_3749_; lean_object* v___x_3751_; 
v___x_3748_ = lean_obj_once(&lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__1___closed__1, &lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__1___closed__1_once, _init_lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__1___closed__1);
lean_inc(v_name_3744_);
v___x_3749_ = l_Lean_MessageData_ofName(v_name_3744_);
if (v_isShared_3747_ == 0)
{
lean_ctor_set_tag(v___x_3746_, 7);
lean_ctor_set(v___x_3746_, 1, v___x_3749_);
lean_ctor_set(v___x_3746_, 0, v___x_3748_);
v___x_3751_ = v___x_3746_;
goto v_reusejp_3750_;
}
else
{
lean_object* v_reuseFailAlloc_3761_; 
v_reuseFailAlloc_3761_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3761_, 0, v___x_3748_);
lean_ctor_set(v_reuseFailAlloc_3761_, 1, v___x_3749_);
v___x_3751_ = v_reuseFailAlloc_3761_;
goto v_reusejp_3750_;
}
v_reusejp_3750_:
{
lean_object* v___x_3752_; lean_object* v___x_3753_; lean_object* v_disable_3754_; lean_object* v___x_3755_; lean_object* v___x_3756_; lean_object* v___x_3757_; lean_object* v___x_3758_; lean_object* v___x_3759_; lean_object* v___x_3760_; 
v___x_3752_ = lean_obj_once(&lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__1___closed__3, &lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__1___closed__3_once, _init_lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_setOptionLinter_spec__1___closed__3);
v___x_3753_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3753_, 0, v___x_3751_);
lean_ctor_set(v___x_3753_, 1, v___x_3752_);
v_disable_3754_ = l_Lean_MessageData_note(v___x_3753_);
v___x_3755_ = l_Lean_Linter_linterMessageTag;
v___x_3756_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3756_, 0, v_msg_3734_);
lean_ctor_set(v___x_3756_, 1, v_disable_3754_);
v___x_3757_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_3757_, 0, v___x_3755_);
lean_ctor_set(v___x_3757_, 1, v___x_3756_);
v___x_3758_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_3758_, 0, v_name_3744_);
lean_ctor_set(v___x_3758_, 1, v___x_3757_);
lean_inc(v_stx_3733_);
v___x_3759_ = lean_alloc_ctor(11, 2, 0);
lean_ctor_set(v___x_3759_, 0, v_stx_3733_);
lean_ctor_set(v___x_3759_, 1, v___x_3758_);
v___x_3760_ = lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2_spec__3(v_stx_3733_, v___x_3759_, v___y_3735_, v___y_3736_, v___y_3737_, v___y_3738_, v___y_3739_, v___y_3740_, v___y_3741_, v___y_3742_);
lean_dec(v_stx_3733_);
return v___x_3760_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2___boxed(lean_object* v_linterOption_3764_, lean_object* v_stx_3765_, lean_object* v_msg_3766_, lean_object* v___y_3767_, lean_object* v___y_3768_, lean_object* v___y_3769_, lean_object* v___y_3770_, lean_object* v___y_3771_, lean_object* v___y_3772_, lean_object* v___y_3773_, lean_object* v___y_3774_, lean_object* v___y_3775_){
_start:
{
lean_object* v_res_3776_; 
v_res_3776_ = lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2(v_linterOption_3764_, v_stx_3765_, v_msg_3766_, v___y_3767_, v___y_3768_, v___y_3769_, v___y_3770_, v___y_3771_, v___y_3772_, v___y_3773_, v___y_3774_);
lean_dec(v___y_3774_);
lean_dec_ref(v___y_3773_);
lean_dec(v___y_3772_);
lean_dec_ref(v___y_3771_);
lean_dec(v___y_3770_);
lean_dec_ref(v___y_3769_);
lean_dec(v___y_3768_);
lean_dec_ref(v___y_3767_);
return v_res_3776_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__1_spec__1___redArg(lean_object* v_o_3777_, lean_object* v___y_3778_){
_start:
{
lean_object* v___x_3780_; lean_object* v_env_3781_; lean_object* v___x_3782_; lean_object* v_toEnvExtension_3783_; lean_object* v_asyncMode_3784_; lean_object* v___x_3785_; lean_object* v___x_3786_; lean_object* v___x_3787_; lean_object* v_merged_3788_; lean_object* v___x_3790_; uint8_t v_isShared_3791_; uint8_t v_isSharedCheck_3796_; 
v___x_3780_ = lean_st_ref_get(v___y_3778_);
v_env_3781_ = lean_ctor_get(v___x_3780_, 0);
lean_inc_ref(v_env_3781_);
lean_dec(v___x_3780_);
v___x_3782_ = l_Lean_Linter_linterSetsExt;
v_toEnvExtension_3783_ = lean_ctor_get(v___x_3782_, 0);
v_asyncMode_3784_ = lean_ctor_get(v_toEnvExtension_3783_, 2);
v___x_3785_ = l_Lean_Linter_instInhabitedLinterSetsState_default;
v___x_3786_ = lean_box(0);
v___x_3787_ = l_Lean_PersistentEnvExtension_getState___redArg(v___x_3785_, v___x_3782_, v_env_3781_, v_asyncMode_3784_, v___x_3786_);
v_merged_3788_ = lean_ctor_get(v___x_3787_, 0);
v_isSharedCheck_3796_ = !lean_is_exclusive(v___x_3787_);
if (v_isSharedCheck_3796_ == 0)
{
lean_object* v_unused_3797_; 
v_unused_3797_ = lean_ctor_get(v___x_3787_, 1);
lean_dec(v_unused_3797_);
v___x_3790_ = v___x_3787_;
v_isShared_3791_ = v_isSharedCheck_3796_;
goto v_resetjp_3789_;
}
else
{
lean_inc(v_merged_3788_);
lean_dec(v___x_3787_);
v___x_3790_ = lean_box(0);
v_isShared_3791_ = v_isSharedCheck_3796_;
goto v_resetjp_3789_;
}
v_resetjp_3789_:
{
lean_object* v___x_3793_; 
if (v_isShared_3791_ == 0)
{
lean_ctor_set(v___x_3790_, 1, v_merged_3788_);
lean_ctor_set(v___x_3790_, 0, v_o_3777_);
v___x_3793_ = v___x_3790_;
goto v_reusejp_3792_;
}
else
{
lean_object* v_reuseFailAlloc_3795_; 
v_reuseFailAlloc_3795_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3795_, 0, v_o_3777_);
lean_ctor_set(v_reuseFailAlloc_3795_, 1, v_merged_3788_);
v___x_3793_ = v_reuseFailAlloc_3795_;
goto v_reusejp_3792_;
}
v_reusejp_3792_:
{
lean_object* v___x_3794_; 
v___x_3794_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3794_, 0, v___x_3793_);
return v___x_3794_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__1_spec__1___redArg___boxed(lean_object* v_o_3798_, lean_object* v___y_3799_, lean_object* v___y_3800_){
_start:
{
lean_object* v_res_3801_; 
v_res_3801_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__1_spec__1___redArg(v_o_3798_, v___y_3799_);
lean_dec(v___y_3799_);
return v_res_3801_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__1(lean_object* v___y_3802_, lean_object* v___y_3803_, lean_object* v___y_3804_, lean_object* v___y_3805_, lean_object* v___y_3806_, lean_object* v___y_3807_, lean_object* v___y_3808_, lean_object* v___y_3809_){
_start:
{
lean_object* v_options_3811_; lean_object* v___x_3812_; 
v_options_3811_ = lean_ctor_get(v___y_3808_, 2);
lean_inc_ref(v_options_3811_);
v___x_3812_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__1_spec__1___redArg(v_options_3811_, v___y_3809_);
return v___x_3812_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__1___boxed(lean_object* v___y_3813_, lean_object* v___y_3814_, lean_object* v___y_3815_, lean_object* v___y_3816_, lean_object* v___y_3817_, lean_object* v___y_3818_, lean_object* v___y_3819_, lean_object* v___y_3820_, lean_object* v___y_3821_){
_start:
{
lean_object* v_res_3822_; 
v_res_3822_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__1(v___y_3813_, v___y_3814_, v___y_3815_, v___y_3816_, v___y_3817_, v___y_3818_, v___y_3819_, v___y_3820_);
lean_dec(v___y_3820_);
lean_dec_ref(v___y_3819_);
lean_dec(v___y_3818_);
lean_dec_ref(v___y_3817_);
lean_dec(v___y_3816_);
lean_dec_ref(v___y_3815_);
lean_dec(v___y_3814_);
lean_dec_ref(v___y_3813_);
return v_res_3822_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_beq___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__3(lean_object* v_x_3823_, lean_object* v_x_3824_){
_start:
{
if (lean_obj_tag(v_x_3823_) == 0)
{
if (lean_obj_tag(v_x_3824_) == 0)
{
uint8_t v___x_3825_; 
v___x_3825_ = 1;
return v___x_3825_;
}
else
{
uint8_t v___x_3826_; 
v___x_3826_ = 0;
return v___x_3826_;
}
}
else
{
if (lean_obj_tag(v_x_3824_) == 0)
{
uint8_t v___x_3827_; 
v___x_3827_ = 0;
return v___x_3827_;
}
else
{
lean_object* v_head_3828_; lean_object* v_tail_3829_; lean_object* v_head_3830_; lean_object* v_tail_3831_; uint8_t v___x_3832_; 
v_head_3828_ = lean_ctor_get(v_x_3823_, 0);
v_tail_3829_ = lean_ctor_get(v_x_3823_, 1);
v_head_3830_ = lean_ctor_get(v_x_3824_, 0);
v_tail_3831_ = lean_ctor_get(v_x_3824_, 1);
v___x_3832_ = l_Lean_instBEqMVarId_beq(v_head_3828_, v_head_3830_);
if (v___x_3832_ == 0)
{
return v___x_3832_;
}
else
{
v_x_3823_ = v_tail_3829_;
v_x_3824_ = v_tail_3831_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_beq___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__3___boxed(lean_object* v_x_3834_, lean_object* v_x_3835_){
_start:
{
uint8_t v_res_3836_; lean_object* v_r_3837_; 
v_res_3836_ = lp_mathlib_List_beq___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__3(v_x_3834_, v_x_3835_);
lean_dec(v_x_3835_);
lean_dec(v_x_3834_);
v_r_3837_ = lean_box(v_res_3836_);
return v_r_3837_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow___closed__2(void){
_start:
{
lean_object* v___x_3844_; lean_object* v___x_3845_; 
v___x_3844_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow___closed__1));
v___x_3845_ = l_Lean_stringToMessageData(v___x_3844_);
return v___x_3845_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow(lean_object* v_newType_3846_, lean_object* v_a_3847_, lean_object* v_a_3848_, lean_object* v_a_3849_, lean_object* v_a_3850_, lean_object* v_a_3851_, lean_object* v_a_3852_, lean_object* v_a_3853_, lean_object* v_a_3854_){
_start:
{
lean_object* v___x_3859_; 
v___x_3859_ = l_Lean_Elab_Tactic_getGoals___redArg(v_a_3848_);
if (lean_obj_tag(v___x_3859_) == 0)
{
lean_object* v_a_3860_; 
v_a_3860_ = lean_ctor_get(v___x_3859_, 0);
lean_inc(v_a_3860_);
lean_dec_ref_known(v___x_3859_, 1);
if (lean_obj_tag(v_a_3860_) == 1)
{
lean_object* v_head_3861_; lean_object* v_tail_3862_; lean_object* v___x_3864_; uint8_t v_isShared_3865_; uint8_t v_isSharedCheck_3941_; 
v_head_3861_ = lean_ctor_get(v_a_3860_, 0);
v_tail_3862_ = lean_ctor_get(v_a_3860_, 1);
v_isSharedCheck_3941_ = !lean_is_exclusive(v_a_3860_);
if (v_isSharedCheck_3941_ == 0)
{
v___x_3864_ = v_a_3860_;
v_isShared_3865_ = v_isSharedCheck_3941_;
goto v_resetjp_3863_;
}
else
{
lean_inc(v_tail_3862_);
lean_inc(v_head_3861_);
lean_dec(v_a_3860_);
v___x_3864_ = lean_box(0);
v_isShared_3865_ = v_isSharedCheck_3941_;
goto v_resetjp_3863_;
}
v_resetjp_3863_:
{
lean_object* v___x_3866_; 
v___x_3866_ = l_Lean_MVarId_getType(v_head_3861_, v_a_3851_, v_a_3852_, v_a_3853_, v_a_3854_);
if (lean_obj_tag(v___x_3866_) == 0)
{
lean_object* v_a_3867_; lean_object* v___x_3868_; lean_object* v_a_3869_; lean_object* v_ref_3870_; uint8_t v___x_3871_; lean_object* v___x_3872_; lean_object* v___x_3873_; lean_object* v___x_3874_; lean_object* v___x_3876_; 
v_a_3867_ = lean_ctor_get(v___x_3866_, 0);
lean_inc(v_a_3867_);
lean_dec_ref_known(v___x_3866_, 1);
v___x_3868_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__0___redArg(v_a_3867_, v_a_3852_);
v_a_3869_ = lean_ctor_get(v___x_3868_, 0);
lean_inc(v_a_3869_);
lean_dec_ref(v___x_3868_);
v_ref_3870_ = lean_ctor_get(v_a_3853_, 5);
v___x_3871_ = 0;
v___x_3872_ = l_Lean_SourceInfo_fromRef(v_ref_3870_, v___x_3871_);
v___x_3873_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Style_4166288182____hygCtx___hyg_4_));
v___x_3874_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow___closed__0));
lean_inc(v___x_3872_);
if (v_isShared_3865_ == 0)
{
lean_ctor_set_tag(v___x_3864_, 2);
lean_ctor_set(v___x_3864_, 1, v___x_3873_);
lean_ctor_set(v___x_3864_, 0, v___x_3872_);
v___x_3876_ = v___x_3864_;
goto v_reusejp_3875_;
}
else
{
lean_object* v_reuseFailAlloc_3932_; 
v_reuseFailAlloc_3932_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3932_, 0, v___x_3872_);
lean_ctor_set(v_reuseFailAlloc_3932_, 1, v___x_3873_);
v___x_3876_ = v_reuseFailAlloc_3932_;
goto v_reusejp_3875_;
}
v_reusejp_3875_:
{
lean_object* v___x_3877_; lean_object* v___x_3878_; 
v___x_3877_ = l_Lean_Syntax_node2(v___x_3872_, v___x_3874_, v___x_3876_, v_newType_3846_);
v___x_3878_ = l_Lean_Elab_Tactic_evalTactic(v___x_3877_, v_a_3847_, v_a_3848_, v_a_3849_, v_a_3850_, v_a_3851_, v_a_3852_, v_a_3853_, v_a_3854_);
if (lean_obj_tag(v___x_3878_) == 0)
{
lean_object* v___x_3879_; lean_object* v_a_3880_; lean_object* v___x_3882_; uint8_t v_isShared_3883_; uint8_t v_isSharedCheck_3931_; 
lean_dec_ref_known(v___x_3878_, 1);
v___x_3879_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__1(v_a_3847_, v_a_3848_, v_a_3849_, v_a_3850_, v_a_3851_, v_a_3852_, v_a_3853_, v_a_3854_);
v_a_3880_ = lean_ctor_get(v___x_3879_, 0);
v_isSharedCheck_3931_ = !lean_is_exclusive(v___x_3879_);
if (v_isSharedCheck_3931_ == 0)
{
v___x_3882_ = v___x_3879_;
v_isShared_3883_ = v_isSharedCheck_3931_;
goto v_resetjp_3881_;
}
else
{
lean_inc(v_a_3880_);
lean_dec(v___x_3879_);
v___x_3882_ = lean_box(0);
v_isShared_3883_ = v_isSharedCheck_3931_;
goto v_resetjp_3881_;
}
v_resetjp_3881_:
{
lean_object* v___x_3884_; uint8_t v___x_3885_; 
v___x_3884_ = lp_mathlib_Mathlib_Linter_linter_style_show;
v___x_3885_ = l_Lean_Linter_getLinterValue(v___x_3884_, v_a_3880_);
lean_dec(v_a_3880_);
if (v___x_3885_ == 0)
{
lean_object* v___x_3886_; lean_object* v___x_3888_; 
lean_dec(v_a_3869_);
lean_dec(v_tail_3862_);
v___x_3886_ = lean_box(0);
if (v_isShared_3883_ == 0)
{
lean_ctor_set(v___x_3882_, 0, v___x_3886_);
v___x_3888_ = v___x_3882_;
goto v_reusejp_3887_;
}
else
{
lean_object* v_reuseFailAlloc_3889_; 
v_reuseFailAlloc_3889_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3889_, 0, v___x_3886_);
v___x_3888_ = v_reuseFailAlloc_3889_;
goto v_reusejp_3887_;
}
v_reusejp_3887_:
{
return v___x_3888_;
}
}
else
{
lean_object* v___x_3890_; 
lean_del_object(v___x_3882_);
v___x_3890_ = l_Lean_Elab_Tactic_getGoals___redArg(v_a_3848_);
if (lean_obj_tag(v___x_3890_) == 0)
{
lean_object* v_a_3891_; lean_object* v___x_3893_; uint8_t v_isShared_3894_; uint8_t v_isSharedCheck_3922_; 
v_a_3891_ = lean_ctor_get(v___x_3890_, 0);
v_isSharedCheck_3922_ = !lean_is_exclusive(v___x_3890_);
if (v_isSharedCheck_3922_ == 0)
{
v___x_3893_ = v___x_3890_;
v_isShared_3894_ = v_isSharedCheck_3922_;
goto v_resetjp_3892_;
}
else
{
lean_inc(v_a_3891_);
lean_dec(v___x_3890_);
v___x_3893_ = lean_box(0);
v_isShared_3894_ = v_isSharedCheck_3922_;
goto v_resetjp_3892_;
}
v_resetjp_3892_:
{
if (lean_obj_tag(v_a_3891_) == 1)
{
lean_object* v_head_3895_; lean_object* v_tail_3896_; uint8_t v___x_3913_; 
v_head_3895_ = lean_ctor_get(v_a_3891_, 0);
lean_inc(v_head_3895_);
v_tail_3896_ = lean_ctor_get(v_a_3891_, 1);
lean_inc(v_tail_3896_);
lean_dec_ref_known(v_a_3891_, 2);
v___x_3913_ = lp_mathlib_List_beq___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__3(v_tail_3862_, v_tail_3896_);
lean_dec(v_tail_3896_);
lean_dec(v_tail_3862_);
if (v___x_3913_ == 0)
{
if (v___x_3885_ == 0)
{
lean_del_object(v___x_3893_);
goto v___jp_3897_;
}
else
{
lean_object* v___x_3914_; lean_object* v___x_3916_; 
lean_dec(v_head_3895_);
lean_dec(v_a_3869_);
v___x_3914_ = lean_box(0);
if (v_isShared_3894_ == 0)
{
lean_ctor_set(v___x_3893_, 0, v___x_3914_);
v___x_3916_ = v___x_3893_;
goto v_reusejp_3915_;
}
else
{
lean_object* v_reuseFailAlloc_3917_; 
v_reuseFailAlloc_3917_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3917_, 0, v___x_3914_);
v___x_3916_ = v_reuseFailAlloc_3917_;
goto v_reusejp_3915_;
}
v_reusejp_3915_:
{
return v___x_3916_;
}
}
}
else
{
lean_del_object(v___x_3893_);
goto v___jp_3897_;
}
v___jp_3897_:
{
lean_object* v___x_3898_; 
v___x_3898_ = l_Lean_MVarId_getType(v_head_3895_, v_a_3851_, v_a_3852_, v_a_3853_, v_a_3854_);
if (lean_obj_tag(v___x_3898_) == 0)
{
lean_object* v_a_3899_; lean_object* v___x_3900_; lean_object* v_a_3901_; uint8_t v___x_3902_; 
v_a_3899_ = lean_ctor_get(v___x_3898_, 0);
lean_inc(v_a_3899_);
lean_dec_ref_known(v___x_3898_, 1);
v___x_3900_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__0___redArg(v_a_3899_, v_a_3852_);
v_a_3901_ = lean_ctor_get(v___x_3900_, 0);
lean_inc(v_a_3901_);
lean_dec_ref(v___x_3900_);
v___x_3902_ = lean_expr_eqv(v_a_3869_, v_a_3901_);
lean_dec(v_a_3901_);
lean_dec(v_a_3869_);
if (v___x_3902_ == 0)
{
if (v___x_3885_ == 0)
{
goto v___jp_3856_;
}
else
{
lean_object* v___x_3903_; lean_object* v___x_3904_; 
v___x_3903_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow___closed__2, &lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow___closed__2);
lean_inc(v_ref_3870_);
v___x_3904_ = lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2(v___x_3884_, v_ref_3870_, v___x_3903_, v_a_3847_, v_a_3848_, v_a_3849_, v_a_3850_, v_a_3851_, v_a_3852_, v_a_3853_, v_a_3854_);
return v___x_3904_;
}
}
else
{
goto v___jp_3856_;
}
}
else
{
lean_object* v_a_3905_; lean_object* v___x_3907_; uint8_t v_isShared_3908_; uint8_t v_isSharedCheck_3912_; 
lean_dec(v_a_3869_);
v_a_3905_ = lean_ctor_get(v___x_3898_, 0);
v_isSharedCheck_3912_ = !lean_is_exclusive(v___x_3898_);
if (v_isSharedCheck_3912_ == 0)
{
v___x_3907_ = v___x_3898_;
v_isShared_3908_ = v_isSharedCheck_3912_;
goto v_resetjp_3906_;
}
else
{
lean_inc(v_a_3905_);
lean_dec(v___x_3898_);
v___x_3907_ = lean_box(0);
v_isShared_3908_ = v_isSharedCheck_3912_;
goto v_resetjp_3906_;
}
v_resetjp_3906_:
{
lean_object* v___x_3910_; 
if (v_isShared_3908_ == 0)
{
v___x_3910_ = v___x_3907_;
goto v_reusejp_3909_;
}
else
{
lean_object* v_reuseFailAlloc_3911_; 
v_reuseFailAlloc_3911_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3911_, 0, v_a_3905_);
v___x_3910_ = v_reuseFailAlloc_3911_;
goto v_reusejp_3909_;
}
v_reusejp_3909_:
{
return v___x_3910_;
}
}
}
}
}
else
{
lean_object* v___x_3918_; lean_object* v___x_3920_; 
lean_dec(v_a_3891_);
lean_dec(v_a_3869_);
lean_dec(v_tail_3862_);
v___x_3918_ = lean_box(0);
if (v_isShared_3894_ == 0)
{
lean_ctor_set(v___x_3893_, 0, v___x_3918_);
v___x_3920_ = v___x_3893_;
goto v_reusejp_3919_;
}
else
{
lean_object* v_reuseFailAlloc_3921_; 
v_reuseFailAlloc_3921_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3921_, 0, v___x_3918_);
v___x_3920_ = v_reuseFailAlloc_3921_;
goto v_reusejp_3919_;
}
v_reusejp_3919_:
{
return v___x_3920_;
}
}
}
}
else
{
lean_object* v_a_3923_; lean_object* v___x_3925_; uint8_t v_isShared_3926_; uint8_t v_isSharedCheck_3930_; 
lean_dec(v_a_3869_);
lean_dec(v_tail_3862_);
v_a_3923_ = lean_ctor_get(v___x_3890_, 0);
v_isSharedCheck_3930_ = !lean_is_exclusive(v___x_3890_);
if (v_isSharedCheck_3930_ == 0)
{
v___x_3925_ = v___x_3890_;
v_isShared_3926_ = v_isSharedCheck_3930_;
goto v_resetjp_3924_;
}
else
{
lean_inc(v_a_3923_);
lean_dec(v___x_3890_);
v___x_3925_ = lean_box(0);
v_isShared_3926_ = v_isSharedCheck_3930_;
goto v_resetjp_3924_;
}
v_resetjp_3924_:
{
lean_object* v___x_3928_; 
if (v_isShared_3926_ == 0)
{
v___x_3928_ = v___x_3925_;
goto v_reusejp_3927_;
}
else
{
lean_object* v_reuseFailAlloc_3929_; 
v_reuseFailAlloc_3929_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3929_, 0, v_a_3923_);
v___x_3928_ = v_reuseFailAlloc_3929_;
goto v_reusejp_3927_;
}
v_reusejp_3927_:
{
return v___x_3928_;
}
}
}
}
}
}
else
{
lean_dec(v_a_3869_);
lean_dec(v_tail_3862_);
return v___x_3878_;
}
}
}
else
{
lean_object* v_a_3933_; lean_object* v___x_3935_; uint8_t v_isShared_3936_; uint8_t v_isSharedCheck_3940_; 
lean_del_object(v___x_3864_);
lean_dec(v_tail_3862_);
lean_dec(v_newType_3846_);
v_a_3933_ = lean_ctor_get(v___x_3866_, 0);
v_isSharedCheck_3940_ = !lean_is_exclusive(v___x_3866_);
if (v_isSharedCheck_3940_ == 0)
{
v___x_3935_ = v___x_3866_;
v_isShared_3936_ = v_isSharedCheck_3940_;
goto v_resetjp_3934_;
}
else
{
lean_inc(v_a_3933_);
lean_dec(v___x_3866_);
v___x_3935_ = lean_box(0);
v_isShared_3936_ = v_isSharedCheck_3940_;
goto v_resetjp_3934_;
}
v_resetjp_3934_:
{
lean_object* v___x_3938_; 
if (v_isShared_3936_ == 0)
{
v___x_3938_ = v___x_3935_;
goto v_reusejp_3937_;
}
else
{
lean_object* v_reuseFailAlloc_3939_; 
v_reuseFailAlloc_3939_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3939_, 0, v_a_3933_);
v___x_3938_ = v_reuseFailAlloc_3939_;
goto v_reusejp_3937_;
}
v_reusejp_3937_:
{
return v___x_3938_;
}
}
}
}
}
else
{
lean_object* v___x_3942_; 
lean_dec(v_a_3860_);
lean_dec(v_newType_3846_);
v___x_3942_ = l_Lean_Elab_Tactic_throwNoGoalsToBeSolved___redArg(v_a_3851_, v_a_3852_, v_a_3853_, v_a_3854_);
return v___x_3942_;
}
}
else
{
lean_object* v_a_3943_; lean_object* v___x_3945_; uint8_t v_isShared_3946_; uint8_t v_isSharedCheck_3950_; 
lean_dec(v_newType_3846_);
v_a_3943_ = lean_ctor_get(v___x_3859_, 0);
v_isSharedCheck_3950_ = !lean_is_exclusive(v___x_3859_);
if (v_isSharedCheck_3950_ == 0)
{
v___x_3945_ = v___x_3859_;
v_isShared_3946_ = v_isSharedCheck_3950_;
goto v_resetjp_3944_;
}
else
{
lean_inc(v_a_3943_);
lean_dec(v___x_3859_);
v___x_3945_ = lean_box(0);
v_isShared_3946_ = v_isSharedCheck_3950_;
goto v_resetjp_3944_;
}
v_resetjp_3944_:
{
lean_object* v___x_3948_; 
if (v_isShared_3946_ == 0)
{
v___x_3948_ = v___x_3945_;
goto v_reusejp_3947_;
}
else
{
lean_object* v_reuseFailAlloc_3949_; 
v_reuseFailAlloc_3949_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3949_, 0, v_a_3943_);
v___x_3948_ = v_reuseFailAlloc_3949_;
goto v_reusejp_3947_;
}
v_reusejp_3947_:
{
return v___x_3948_;
}
}
}
v___jp_3856_:
{
lean_object* v___x_3857_; lean_object* v___x_3858_; 
v___x_3857_ = lean_box(0);
v___x_3858_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3858_, 0, v___x_3857_);
return v___x_3858_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow___boxed(lean_object* v_newType_3951_, lean_object* v_a_3952_, lean_object* v_a_3953_, lean_object* v_a_3954_, lean_object* v_a_3955_, lean_object* v_a_3956_, lean_object* v_a_3957_, lean_object* v_a_3958_, lean_object* v_a_3959_, lean_object* v_a_3960_){
_start:
{
lean_object* v_res_3961_; 
v_res_3961_ = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow(v_newType_3951_, v_a_3952_, v_a_3953_, v_a_3954_, v_a_3955_, v_a_3956_, v_a_3957_, v_a_3958_, v_a_3959_);
lean_dec(v_a_3959_);
lean_dec_ref(v_a_3958_);
lean_dec(v_a_3957_);
lean_dec_ref(v_a_3956_);
lean_dec(v_a_3955_);
lean_dec_ref(v_a_3954_);
lean_dec(v_a_3953_);
lean_dec_ref(v_a_3952_);
return v_res_3961_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__1_spec__1(lean_object* v_o_3962_, lean_object* v___y_3963_, lean_object* v___y_3964_, lean_object* v___y_3965_, lean_object* v___y_3966_, lean_object* v___y_3967_, lean_object* v___y_3968_, lean_object* v___y_3969_, lean_object* v___y_3970_){
_start:
{
lean_object* v___x_3972_; 
v___x_3972_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__1_spec__1___redArg(v_o_3962_, v___y_3970_);
return v___x_3972_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__1_spec__1___boxed(lean_object* v_o_3973_, lean_object* v___y_3974_, lean_object* v___y_3975_, lean_object* v___y_3976_, lean_object* v___y_3977_, lean_object* v___y_3978_, lean_object* v___y_3979_, lean_object* v___y_3980_, lean_object* v___y_3981_, lean_object* v___y_3982_){
_start:
{
lean_object* v_res_3983_; 
v_res_3983_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__1_spec__1(v_o_3973_, v___y_3974_, v___y_3975_, v___y_3976_, v___y_3977_, v___y_3978_, v___y_3979_, v___y_3980_, v___y_3981_);
lean_dec(v___y_3981_);
lean_dec_ref(v___y_3980_);
lean_dec(v___y_3979_);
lean_dec_ref(v___y_3978_);
lean_dec(v___y_3977_);
lean_dec_ref(v___y_3976_);
lean_dec(v___y_3975_);
lean_dec_ref(v___y_3974_);
return v_res_3983_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2_spec__3_spec__4(lean_object* v_ref_3984_, lean_object* v_msgData_3985_, uint8_t v_severity_3986_, uint8_t v_isSilent_3987_, lean_object* v___y_3988_, lean_object* v___y_3989_, lean_object* v___y_3990_, lean_object* v___y_3991_, lean_object* v___y_3992_, lean_object* v___y_3993_, lean_object* v___y_3994_, lean_object* v___y_3995_){
_start:
{
lean_object* v___x_3997_; 
v___x_3997_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2_spec__3_spec__4___redArg(v_ref_3984_, v_msgData_3985_, v_severity_3986_, v_isSilent_3987_, v___y_3992_, v___y_3993_, v___y_3994_, v___y_3995_);
return v___x_3997_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2_spec__3_spec__4___boxed(lean_object* v_ref_3998_, lean_object* v_msgData_3999_, lean_object* v_severity_4000_, lean_object* v_isSilent_4001_, lean_object* v___y_4002_, lean_object* v___y_4003_, lean_object* v___y_4004_, lean_object* v___y_4005_, lean_object* v___y_4006_, lean_object* v___y_4007_, lean_object* v___y_4008_, lean_object* v___y_4009_, lean_object* v___y_4010_){
_start:
{
uint8_t v_severity_boxed_4011_; uint8_t v_isSilent_boxed_4012_; lean_object* v_res_4013_; 
v_severity_boxed_4011_ = lean_unbox(v_severity_4000_);
v_isSilent_boxed_4012_ = lean_unbox(v_isSilent_4001_);
v_res_4013_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow_spec__2_spec__3_spec__4(v_ref_3998_, v_msgData_3999_, v_severity_boxed_4011_, v_isSilent_boxed_4012_, v___y_4002_, v___y_4003_, v___y_4004_, v___y_4005_, v___y_4006_, v___y_4007_, v___y_4008_, v___y_4009_);
lean_dec(v___y_4009_);
lean_dec_ref(v___y_4008_);
lean_dec(v___y_4007_);
lean_dec_ref(v___y_4006_);
lean_dec(v___y_4005_);
lean_dec_ref(v___y_4004_);
lean_dec(v___y_4003_);
lean_dec_ref(v___y_4002_);
lean_dec(v_ref_3998_);
return v_res_4013_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Linter_Style___aux__Mathlib__Tactic__Linter__Style______elabRules__Mathlib__Linter__Style__show__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_4040_; lean_object* v___x_4041_; lean_object* v___x_4042_; 
v___x_4040_ = lean_box(0);
v___x_4041_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_4042_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4042_, 0, v___x_4041_);
lean_ctor_set(v___x_4042_, 1, v___x_4040_);
return v___x_4042_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Linter_Style___aux__Mathlib__Tactic__Linter__Style______elabRules__Mathlib__Linter__Style__show__1_spec__0___redArg(){
_start:
{
lean_object* v___x_4044_; lean_object* v___x_4045_; 
v___x_4044_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Linter_Style___aux__Mathlib__Tactic__Linter__Style______elabRules__Mathlib__Linter__Style__show__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Linter_Style___aux__Mathlib__Tactic__Linter__Style______elabRules__Mathlib__Linter__Style__show__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Linter_Style___aux__Mathlib__Tactic__Linter__Style______elabRules__Mathlib__Linter__Style__show__1_spec__0___redArg___closed__0);
v___x_4045_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4045_, 0, v___x_4044_);
return v___x_4045_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Linter_Style___aux__Mathlib__Tactic__Linter__Style______elabRules__Mathlib__Linter__Style__show__1_spec__0___redArg___boxed(lean_object* v___y_4046_){
_start:
{
lean_object* v_res_4047_; 
v_res_4047_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Linter_Style___aux__Mathlib__Tactic__Linter__Style______elabRules__Mathlib__Linter__Style__show__1_spec__0___redArg();
return v_res_4047_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Linter_Style___aux__Mathlib__Tactic__Linter__Style______elabRules__Mathlib__Linter__Style__show__1_spec__0(lean_object* v_00_u03b1_4048_, lean_object* v___y_4049_, lean_object* v___y_4050_, lean_object* v___y_4051_, lean_object* v___y_4052_, lean_object* v___y_4053_, lean_object* v___y_4054_, lean_object* v___y_4055_, lean_object* v___y_4056_){
_start:
{
lean_object* v___x_4058_; 
v___x_4058_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Linter_Style___aux__Mathlib__Tactic__Linter__Style______elabRules__Mathlib__Linter__Style__show__1_spec__0___redArg();
return v___x_4058_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Linter_Style___aux__Mathlib__Tactic__Linter__Style______elabRules__Mathlib__Linter__Style__show__1_spec__0___boxed(lean_object* v_00_u03b1_4059_, lean_object* v___y_4060_, lean_object* v___y_4061_, lean_object* v___y_4062_, lean_object* v___y_4063_, lean_object* v___y_4064_, lean_object* v___y_4065_, lean_object* v___y_4066_, lean_object* v___y_4067_, lean_object* v___y_4068_){
_start:
{
lean_object* v_res_4069_; 
v_res_4069_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Linter_Style___aux__Mathlib__Tactic__Linter__Style______elabRules__Mathlib__Linter__Style__show__1_spec__0(v_00_u03b1_4059_, v___y_4060_, v___y_4061_, v___y_4062_, v___y_4063_, v___y_4064_, v___y_4065_, v___y_4066_, v___y_4067_);
lean_dec(v___y_4067_);
lean_dec_ref(v___y_4066_);
lean_dec(v___y_4065_);
lean_dec_ref(v___y_4064_);
lean_dec(v___y_4063_);
lean_dec_ref(v___y_4062_);
lean_dec(v___y_4061_);
lean_dec_ref(v___y_4060_);
return v_res_4069_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_Style___aux__Mathlib__Tactic__Linter__Style______elabRules__Mathlib__Linter__Style__show__1(lean_object* v_x_4070_, lean_object* v_a_4071_, lean_object* v_a_4072_, lean_object* v_a_4073_, lean_object* v_a_4074_, lean_object* v_a_4075_, lean_object* v_a_4076_, lean_object* v_a_4077_, lean_object* v_a_4078_){
_start:
{
lean_object* v___x_4080_; uint8_t v___x_4081_; 
v___x_4080_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_Style_show___closed__0));
lean_inc(v_x_4070_);
v___x_4081_ = l_Lean_Syntax_isOfKind(v_x_4070_, v___x_4080_);
if (v___x_4081_ == 0)
{
lean_object* v___x_4082_; 
lean_dec(v_x_4070_);
v___x_4082_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Linter_Style___aux__Mathlib__Tactic__Linter__Style______elabRules__Mathlib__Linter__Style__show__1_spec__0___redArg();
return v___x_4082_;
}
else
{
lean_object* v___x_4083_; lean_object* v_newType_4084_; lean_object* v___x_4085_; 
v___x_4083_ = lean_unsigned_to_nat(1u);
v_newType_4084_ = l_Lean_Syntax_getArg(v_x_4070_, v___x_4083_);
lean_dec(v_x_4070_);
v___x_4085_ = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_elabShow(v_newType_4084_, v_a_4071_, v_a_4072_, v_a_4073_, v_a_4074_, v_a_4075_, v_a_4076_, v_a_4077_, v_a_4078_);
return v___x_4085_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_Style___aux__Mathlib__Tactic__Linter__Style______elabRules__Mathlib__Linter__Style__show__1___boxed(lean_object* v_x_4086_, lean_object* v_a_4087_, lean_object* v_a_4088_, lean_object* v_a_4089_, lean_object* v_a_4090_, lean_object* v_a_4091_, lean_object* v_a_4092_, lean_object* v_a_4093_, lean_object* v_a_4094_, lean_object* v_a_4095_){
_start:
{
lean_object* v_res_4096_; 
v_res_4096_ = lp_mathlib_Mathlib_Linter_Style___aux__Mathlib__Tactic__Linter__Style______elabRules__Mathlib__Linter__Style__show__1(v_x_4086_, v_a_4087_, v_a_4088_, v_a_4089_, v_a_4090_, v_a_4091_, v_a_4092_, v_a_4093_, v_a_4094_);
lean_dec(v_a_4094_);
lean_dec_ref(v_a_4093_);
lean_dec(v_a_4092_);
lean_dec_ref(v_a_4091_);
lean_dec(v_a_4090_);
lean_dec_ref(v_a_4089_);
lean_dec(v_a_4088_);
lean_dec_ref(v_a_4087_);
return v_res_4096_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Parser_Command(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_DeclarationNames(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Tactic_Lint_Basic(uint8_t builtin);
lean_object* runtime_initialize_Lean_Parser_Module(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_Style(uint8_t builtin) {
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
res = runtime_initialize_mathlib_Mathlib_Tactic_DeclarationNames(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Tactic_Lint_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Parser_Module(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Command(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_Header(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Linter_Style(uint8_t builtin) {
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
res = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_1494121775____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Linter_linter_style_setOption = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Linter_linter_style_setOption);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_setOption_initFn_00___x40_Mathlib_Tactic_Linter_Style_3512398344____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_3409032198____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Linter_linter_style_missingEnd = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Linter_linter_style_missingEnd);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_missingEnd_initFn_00___x40_Mathlib_Tactic_Linter_Style_3360231377____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_3789867222____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Linter_linter_style_cdot = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Linter_linter_style_cdot);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_Style_1823831825____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_830885783____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Linter_linter_style_dollarSyntax = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Linter_linter_style_dollarSyntax);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_dollarSyntax_initFn_00___x40_Mathlib_Tactic_Linter_Style_1303717341____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_450967313____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Linter_linter_style_lambdaSyntax = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Linter_linter_style_lambdaSyntax);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_lambdaSyntax_initFn_00___x40_Mathlib_Tactic_Linter_Style_1166937461____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_695976056____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Linter_linter_style_longFile = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Linter_linter_style_longFile);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_583422302____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Linter_linter_style_longFileDefValue = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Linter_linter_style_longFileDefValue);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longFile_initFn_00___x40_Mathlib_Tactic_Linter_Style_3199531762____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_1043171623____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Linter_linter_style_longLine = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Linter_linter_style_longLine);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_690386724____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Linter_linter_style_longLine_maxLineLength = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Linter_linter_style_longLine_maxLineLength);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_longLine_initFn_00___x40_Mathlib_Tactic_Linter_Style_724867545____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_3276061806____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Linter_linter_style_nameCheck = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Linter_linter_style_nameCheck);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_nameCheck_initFn_00___x40_Mathlib_Tactic_Linter_Style_1285535146____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore = _init_lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore();
lean_mark_persistent(lp_mathlib_Mathlib_Linter_Style_nameCheck_defsWithUnderscore);
res = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_856385564____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Linter_linter_style_openClassical = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Linter_linter_style_openClassical);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_Style_openClassical_initFn_00___x40_Mathlib_Tactic_Linter_Style_273924139____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linter_Style_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Style_4166288182____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Linter_linter_style_show = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Linter_linter_style_show);
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Command(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_Header(uint8_t builtin);
lean_object* initialize_Lean_Parser_Command(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_DeclarationNames(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Tactic_Lint_Basic(uint8_t builtin);
lean_object* initialize_Lean_Parser_Module(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Linter_Style(uint8_t builtin) {
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
res = initialize_mathlib_Mathlib_Tactic_DeclarationNames(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Tactic_Lint_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Parser_Module(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_Style(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Linter_Style(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Linter_Style(builtin);
}
#ifdef __cplusplus
}
#endif
