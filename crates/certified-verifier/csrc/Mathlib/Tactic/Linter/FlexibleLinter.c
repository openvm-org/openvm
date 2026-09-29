// Lean compiler output
// Module: Mathlib.Tactic.Linter.FlexibleLinter
// Imports: public import Init public meta import Init public meta import Lean.Elab.Command public meta import Lean.Elab.Tactic.Simp public meta import Lean.Meta.Tactic.TryThis public meta import Lean.Server.InfoUtils public meta import Mathlib.Tactic.Linter.Header public import Lean.Parser.Term
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
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_array_get_size(lean_object*);
uint64_t lean_uint64_mix_hash(uint64_t, uint64_t);
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_land(size_t, size_t);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
size_t lean_array_size(lean_object*);
lean_object* lean_array_uget(lean_object*, size_t);
uint8_t lean_usize_dec_eq(size_t, size_t);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_beq___boxed(lean_object*, lean_object*);
uint64_t l_Lean_instHashableFVarId_hash(lean_object*);
uint64_t l_Lean_instHashableMVarId_hash(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
extern lean_object* l_Lean_Elab_Command_instInhabitedScope_default;
lean_object* l_List_head_x21___redArg(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_usize_to_nat(size_t);
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
uint8_t l_Lean_instBEqFVarId_beq(lean_object*, lean_object*);
extern lean_object* l_Lean_Linter_linterSetsExt;
extern lean_object* l_Lean_Linter_instInhabitedLinterSetsState_default;
lean_object* l_Lean_PersistentEnvExtension_getState___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_LocalContext_findFromUserName_x3f(lean_object*, lean_object*);
lean_object* l_Lean_LocalDecl_fvarId(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_LocalContext_getFVarIds(lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
uint8_t l_List_elem___redArg(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_MessageLog_hasErrors(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_instInhabitedPersistentArrayNode_default(lean_object*);
size_t lean_usize_shift_left(size_t, size_t);
lean_object* l_Lean_Syntax_getRange_x3f(lean_object*, uint8_t);
lean_object* l_Lean_Elab_InfoTree_foldInfo___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getKind(lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_formatStx(lean_object*, lean_object*, uint8_t);
extern lean_object* l_Std_Format_defWidth;
lean_object* l_Std_Format_pretty(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* lean_dbg_trace(lean_object*, lean_object*);
lean_object* l_Lean_LocalDecl_userName(lean_object*);
lean_object* lean_local_ctx_find(lean_object*, lean_object*);
extern lean_object* l_Lean_instInhabitedLocalDecl_default;
lean_object* l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(lean_object*, uint8_t);
lean_object* l_Lean_Name_toString(lean_object*, uint8_t);
extern lean_object* l_Lean_instInhabitedMetavarDecl_default;
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getAtomVal(lean_object*);
lean_object* l_List_lengthTR___redArg(lean_object*);
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
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* lean_register_option(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* l_Lean_MessageData_note(lean_object*);
extern lean_object* l_Lean_Linter_linterMessageTag;
extern lean_object* l_Lean_MessageData_nil;
lean_object* l_Lean_Meta_Tactic_TryThis_addSuggestion___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_liftCoreM___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_List_get_x3fInternal___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Meta_Simp_Context_mkDefault___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_Simp_getSimprocs___redArg(lean_object*);
lean_object* l_Lean_Meta_simpGoal(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_PersistentHashMap_Node_isEmpty___redArg(lean_object*);
lean_object* l_Lean_Elab_Tactic_mkSimpOnly(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMCtxImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_ContextInfo_runMetaM___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_io_error_to_string(lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* lean_string_utf8_byte_size(lean_object*);
lean_object* l_String_Slice_trimAscii(lean_object*);
lean_object* lean_string_utf8_extract_fast(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_reprint(lean_object*);
uint8_t l_Lean_Linter_getLinterValue(lean_object*, lean_object*);
lean_object* lean_nat_to_int(lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* l_Lean_withSetOptionIn___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_addLinter(lean_object*);
lean_object* l_Repr_addAppParen(lean_object*, lean_object*);
lean_object* l_Lean_Name_reprPrec(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "flexible"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(186, 218, 113, 226, 101, 176, 32, 79)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(87, 87, 124, 175, 51, 209, 186, 58)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "enable the flexible linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(120, 131, 127, 204, 79, 169, 80, 92)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(101, 237, 90, 120, 51, 59, 46, 172)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(180, 89, 7, 245, 117, 99, 245, 34)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_linter_flexible;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "simp"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "simpAll"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "only"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__5_value;
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Lean_Elab_TacticInfo_goalsTargetedBy_spec__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_List_filterTR_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Lean_Elab_TacticInfo_goalsTargetedBy_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Name_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_List_filterTR_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Lean_Elab_TacticInfo_goalsTargetedBy_spec__1___closed__0 = (const lean_object*)&lp_mathlib_List_filterTR_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Lean_Elab_TacticInfo_goalsTargetedBy_spec__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_List_filterTR_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Lean_Elab_TacticInfo_goalsTargetedBy_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Lean_Elab_TacticInfo_goalsTargetedBy(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_filterTR_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Lean_Elab_TacticInfo_goalsCreatedBy_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Lean_Elab_TacticInfo_goalsCreatedBy(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_extractTacticData___lam__0(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_extractTacticData___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_extractTacticData___lam__0, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_extractTacticData___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_extractTacticData___closed__0_value;
static const lean_array_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_extractTacticData___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_extractTacticData___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_extractTacticData___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_extractTacticData(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_Stained_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_Stained_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_Stained_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_Stained_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_Stained_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_Stained_name_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_Stained_name_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_Stained_goal_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_Stained_goal_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_Stained_wildcard_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_Stained_wildcard_elim(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 85, .m_capacity = 85, .m_length = 84, .m_data = "_private.Mathlib.Tactic.Linter.FlexibleLinter.0.Mathlib.Linter.Flexible.Stained.goal"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr___closed__0_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 89, .m_capacity = 89, .m_length = 88, .m_data = "_private.Mathlib.Tactic.Linter.FlexibleLinter.0.Mathlib.Linter.Flexible.Stained.wildcard"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr___closed__2_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr___closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 85, .m_capacity = 85, .m_length = 84, .m_data = "_private.Mathlib.Tactic.Linter.FlexibleLinter.0.Mathlib.Linter.Flexible.Stained.name"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr___closed__4_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr___closed__5_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr___closed__6_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr___closed__7;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr___closed__8;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_Flexible_instInhabitedStained_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Linter_Flexible_instInhabitedStained_default___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Linter_Flexible_instInhabitedStained_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Linter_Flexible_instInhabitedStained_default = (const lean_object*)&lp_mathlib_Mathlib_Linter_Flexible_instInhabitedStained_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instInhabitedStained = (const lean_object*)&lp_mathlib_Mathlib_Linter_Flexible_instInhabitedStained_default___closed__0_value;
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instDecidableEqStained_decEq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instDecidableEqStained_decEq___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instDecidableEqStained(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instDecidableEqStained___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instHashableStained_hash___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static uint64_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instHashableStained_hash___closed__0;
LEAN_EXPORT uint64_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instHashableStained_hash(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instHashableStained_hash___boxed(lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instHashableStained___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instHashableStained_hash___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instHashableStained___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instHashableStained___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instHashableStained = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instHashableStained___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instToStringStained___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "⊢"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instToStringStained___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instToStringStained___lam__0___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instToStringStained___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "*"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instToStringStained___lam__0___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instToStringStained___lam__0___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instToStringStained___lam__0(lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instToStringStained___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instToStringStained___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instToStringStained___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instToStringStained___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instToStringStained = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instToStringStained___closed__0_value;
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__0_spec__1_spec__2_spec__8___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__0_spec__1_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__0_spec__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__1_spec__3_spec__5___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__1_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__1_spec__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__1_spec__5(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__1_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__3(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__5(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__0;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__1;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__2;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "|"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__3_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__4;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__4(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__4___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__0_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__1_spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__0_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__1_spec__3_spec__5(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__0_spec__1_spec__2_spec__8(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_getStained_spec__1(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_getStained_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_getStained___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "location"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_getStained___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_getStained___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_getStained(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_getStained_spec__0(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_getStained_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_getStained_x21(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_Stained_toFMVarId_spec__0(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_Stained_toFMVarId_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_Stained_toFMVarId___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_Stained_toFMVarId___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_Stained_toFMVarId___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_Stained_toFMVarId(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_Stained_toFMVarId___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0_spec__1_spec__2_spec__3___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0_spec__1_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0_spec__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "tacticSorry"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__1_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__0_value),LEAN_SCALAR_PTR_LITERAL(254, 186, 126, 140, 105, 148, 113, 102)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "tacticRepeat_"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__3_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__3_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__3_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__2_value),LEAN_SCALAR_PTR_LITERAL(149, 101, 42, 245, 144, 172, 68, 230)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "tacticStop_"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__5_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__5_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__5_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__5_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__4_value),LEAN_SCALAR_PTR_LITERAL(186, 187, 217, 116, 133, 153, 2, 108)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Abel"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__6_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "abelNF"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__8_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__8_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__6_value),LEAN_SCALAR_PTR_LITERAL(127, 220, 84, 140, 79, 41, 205, 100)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__8_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__7_value),LEAN_SCALAR_PTR_LITERAL(92, 237, 112, 223, 6, 29, 62, 9)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__8_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "tacticAbel_nf!__"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__10_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__10_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__10_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__10_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__6_value),LEAN_SCALAR_PTR_LITERAL(127, 220, 84, 140, 79, 41, 205, 100)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__10_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__9_value),LEAN_SCALAR_PTR_LITERAL(98, 32, 72, 61, 180, 21, 232, 68)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__10_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "RingNF"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__11_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "ringNF"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__12_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__13_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__13_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__13_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__13_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__13_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__11_value),LEAN_SCALAR_PTR_LITERAL(132, 186, 29, 238, 234, 106, 109, 235)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__13_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__12_value),LEAN_SCALAR_PTR_LITERAL(169, 58, 229, 242, 41, 102, 20, 168)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__13_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "tacticRing_nf!__"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__14_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__15_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__15_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__15_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__15_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__15_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__11_value),LEAN_SCALAR_PTR_LITERAL(132, 186, 29, 238, 234, 106, 109, 235)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__15_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__14_value),LEAN_SCALAR_PTR_LITERAL(51, 23, 172, 43, 132, 39, 215, 40)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__15_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Group"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__16_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "group"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__17_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__18_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__18_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__18_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__18_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__18_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__16_value),LEAN_SCALAR_PTR_LITERAL(105, 28, 189, 37, 66, 9, 55, 97)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__18_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__17_value),LEAN_SCALAR_PTR_LITERAL(20, 3, 115, 244, 56, 190, 13, 255)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__18 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__18_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "FieldSimp"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__19 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__19_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "fieldSimp"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__20 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__20_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__21_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__21_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__21_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__21_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__21_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__19_value),LEAN_SCALAR_PTR_LITERAL(187, 95, 53, 248, 72, 241, 7, 199)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__21_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__20_value),LEAN_SCALAR_PTR_LITERAL(114, 169, 79, 101, 45, 193, 170, 105)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__21 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__21_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "field"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__22 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__22_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__23_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__23_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__23_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__23_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__23_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__19_value),LEAN_SCALAR_PTR_LITERAL(187, 95, 53, 248, 72, 241, 7, 199)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__23_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__22_value),LEAN_SCALAR_PTR_LITERAL(160, 11, 124, 52, 224, 100, 58, 138)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__23 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__23_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "finiteness_nonterminal"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__24 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__24_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__24_value),LEAN_SCALAR_PTR_LITERAL(57, 247, 170, 189, 159, 128, 58, 15)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__25 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__25_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__26 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__26_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__27_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__27_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__27_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__27_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__27_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__27_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__26_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__27 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__27_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__28 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__28_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__29_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__29_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__29_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__29_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__29_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__29_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__28_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__29 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__29_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__30 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__30_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "byTactic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__31 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__31_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__32_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__32_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__32_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__32_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__32_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__30_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__32_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__31_value),LEAN_SCALAR_PTR_LITERAL(187, 150, 238, 148, 228, 221, 116, 224)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__32 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__32_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "by"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__33 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__33_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__33_value),LEAN_SCALAR_PTR_LITERAL(33, 100, 221, 244, 231, 185, 222, 214)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__34 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__34_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "tacticTry_"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__35 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__35_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__36_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__36_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__36_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__36_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__36_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__36_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__35_value),LEAN_SCALAR_PTR_LITERAL(34, 109, 187, 155, 23, 130, 33, 152)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__36 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__36_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "choice"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__37 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__37_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__37_value),LEAN_SCALAR_PTR_LITERAL(59, 66, 148, 42, 181, 100, 85, 166)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__38 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__38_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "allGoals"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__39 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__39_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__40_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__40_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__40_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__40_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__40_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__40_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__39_value),LEAN_SCALAR_PTR_LITERAL(105, 66, 138, 83, 251, 171, 29, 196)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__40 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__40_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Std"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__41 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__41_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticOn_goal-_=>_"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__42 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__42_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__43_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__41_value),LEAN_SCALAR_PTR_LITERAL(48, 144, 193, 124, 159, 137, 91, 218)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__43_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__43_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(77, 161, 28, 104, 237, 118, 82, 71)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__43_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__42_value),LEAN_SCALAR_PTR_LITERAL(69, 40, 9, 163, 155, 106, 117, 233)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__43 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__43_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__44_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "tactic_<;>_"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__44 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__44_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__45_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__45_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__45_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__45_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__45_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__45_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__45_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__44_value),LEAN_SCALAR_PTR_LITERAL(31, 118, 44, 159, 195, 11, 47, 176)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__45 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__45_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "cdotTk"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__46 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__46_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__47_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__47_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__47_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__46_value),LEAN_SCALAR_PTR_LITERAL(117, 126, 44, 217, 38, 3, 69, 145)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__47 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__47_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__48_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "cdot"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__48 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__48_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__49_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__49_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__49_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__48_value),LEAN_SCALAR_PTR_LITERAL(238, 151, 138, 49, 249, 18, 254, 242)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__49 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__49_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__50_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__50;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__51_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__51;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__52_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__52;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__53_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__53;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__54_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__54;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__55_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__55;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__56_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__56;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__57_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__57;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__58_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__58;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__59_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__59;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__60_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__60;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__61_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__61;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__62_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__62;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__63_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__63;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__64_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__64;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__65_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__65;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__66_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__66;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__67_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__67;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__68_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__68;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__69_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__69;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__70_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__70;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__71_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__71;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__72_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__72;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__73_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__73;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers;
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0_spec__1_spec__2_spec__3(lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__0_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__0_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__0_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__0_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__3_value),LEAN_SCALAR_PTR_LITERAL(50, 13, 241, 145, 67, 153, 105, 177)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__1_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__4_value),LEAN_SCALAR_PTR_LITERAL(5, 49, 55, 92, 153, 191, 153, 249)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "simpa"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__3_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__3_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__3_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__2_value),LEAN_SCALAR_PTR_LITERAL(197, 186, 141, 63, 66, 208, 56, 113)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "simpaUsingBang"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__5_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__5_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__5_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__5_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__4_value),LEAN_SCALAR_PTR_LITERAL(207, 241, 251, 37, 131, 174, 231, 55)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "dsimp"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__7_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__7_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__7_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__6_value),LEAN_SCALAR_PTR_LITERAL(246, 53, 215, 155, 171, 182, 123, 76)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__7_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "constructor"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__9_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__9_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__9_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__9_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__9_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__8_value),LEAN_SCALAR_PTR_LITERAL(144, 188, 57, 91, 27, 124, 155, 13)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__9_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "congr"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__11_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__11_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__11_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__11_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__10_value),LEAN_SCALAR_PTR_LITERAL(41, 88, 242, 177, 210, 111, 166, 107)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__11_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "done"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__12_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__13_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__13_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__13_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__13_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__13_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__13_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__12_value),LEAN_SCALAR_PTR_LITERAL(113, 161, 179, 82, 204, 87, 48, 123)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__13_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticRfl"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__14_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__15_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__15_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__15_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__15_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__15_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__15_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__14_value),LEAN_SCALAR_PTR_LITERAL(201, 188, 173, 198, 169, 252, 183, 45)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__15_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "acRfl"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__16_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__17_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__17_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__17_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__17_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__17_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__17_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__16_value),LEAN_SCALAR_PTR_LITERAL(251, 10, 210, 32, 196, 152, 20, 107)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__17_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "omega"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__18 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__18_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__19_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__19_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__19_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__19_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__19_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__19_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__18_value),LEAN_SCALAR_PTR_LITERAL(138, 49, 229, 237, 137, 52, 176, 206)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__19 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__19_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "abel"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__20 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__20_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__21_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__21_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__21_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__21_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__21_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__6_value),LEAN_SCALAR_PTR_LITERAL(127, 220, 84, 140, 79, 41, 205, 100)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__21_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__20_value),LEAN_SCALAR_PTR_LITERAL(55, 207, 94, 79, 28, 196, 87, 177)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__21 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__21_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "tacticAbel!"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__22 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__22_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__23_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__23_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__23_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__23_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__23_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__6_value),LEAN_SCALAR_PTR_LITERAL(127, 220, 84, 140, 79, 41, 205, 100)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__23_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__22_value),LEAN_SCALAR_PTR_LITERAL(80, 107, 56, 4, 211, 100, 136, 75)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__23 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__23_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "ring"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__24 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__24_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__25_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__25_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__25_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__25_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__25_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__11_value),LEAN_SCALAR_PTR_LITERAL(132, 186, 29, 238, 234, 106, 109, 235)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__25_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__24_value),LEAN_SCALAR_PTR_LITERAL(142, 86, 114, 51, 188, 197, 85, 98)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__25 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__25_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "tacticRing!"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__26 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__26_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__27_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__27_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__27_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__27_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__27_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__11_value),LEAN_SCALAR_PTR_LITERAL(132, 186, 29, 238, 234, 106, 109, 235)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__27_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__26_value),LEAN_SCALAR_PTR_LITERAL(14, 30, 68, 192, 235, 62, 193, 146)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__27 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__27_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Ring"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__28 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__28_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ring1"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__29 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__29_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__30_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__30_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__30_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__30_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__30_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__28_value),LEAN_SCALAR_PTR_LITERAL(167, 213, 149, 120, 143, 153, 69, 12)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__30_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__29_value),LEAN_SCALAR_PTR_LITERAL(221, 141, 62, 226, 100, 80, 9, 0)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__30 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__30_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "tacticRing1!"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__31 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__31_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__32_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__32_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__32_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__32_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__32_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__28_value),LEAN_SCALAR_PTR_LITERAL(167, 213, 149, 120, 143, 153, 69, 12)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__32_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__31_value),LEAN_SCALAR_PTR_LITERAL(181, 132, 48, 112, 76, 186, 197, 170)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__32 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__32_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ring1NF"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__33 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__33_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__34_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__34_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__34_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__34_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__34_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__11_value),LEAN_SCALAR_PTR_LITERAL(132, 186, 29, 238, 234, 106, 109, 235)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__34_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__33_value),LEAN_SCALAR_PTR_LITERAL(32, 222, 8, 155, 240, 149, 242, 50)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__34 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__34_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "tacticRing1_nf!_"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__35 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__35_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__36_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__36_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__36_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__36_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__36_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__11_value),LEAN_SCALAR_PTR_LITERAL(132, 186, 29, 238, 234, 106, 109, 235)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__36_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__35_value),LEAN_SCALAR_PTR_LITERAL(226, 161, 146, 145, 83, 28, 29, 216)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__36 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__36_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "ring1NF!"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__37 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__37_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__38_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__38_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__38_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__38_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__38_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__11_value),LEAN_SCALAR_PTR_LITERAL(132, 186, 29, 238, 234, 106, 109, 235)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__38_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__37_value),LEAN_SCALAR_PTR_LITERAL(175, 0, 47, 237, 253, 232, 168, 49)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__38 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__38_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Module"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__39 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__39_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "tacticModule"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__40 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__40_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__41_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__41_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__41_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__41_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__41_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__39_value),LEAN_SCALAR_PTR_LITERAL(139, 132, 11, 52, 40, 76, 136, 114)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__41_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__40_value),LEAN_SCALAR_PTR_LITERAL(108, 48, 198, 208, 220, 135, 205, 181)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__41 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__41_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "grind"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__42 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__42_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__43_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__43_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__43_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__43_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__43_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__43_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__42_value),LEAN_SCALAR_PTR_LITERAL(150, 98, 0, 78, 28, 79, 28, 100)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__43 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__43_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__44_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "grobner"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__44 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__44_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__45_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__45_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__45_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__45_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__45_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__45_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__45_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__44_value),LEAN_SCALAR_PTR_LITERAL(241, 188, 100, 211, 106, 33, 144, 166)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__45 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__45_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "lia"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__46 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__46_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__47_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__47_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__47_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__47_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__47_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__47_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__47_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__46_value),LEAN_SCALAR_PTR_LITERAL(85, 74, 44, 139, 89, 35, 186, 8)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__47 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__47_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__48_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "normNum"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__48 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__48_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__49_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__49_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__49_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__49_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__49_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__48_value),LEAN_SCALAR_PTR_LITERAL(235, 202, 36, 226, 215, 147, 189, 233)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__49 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__49_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__50_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "linarith"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__50 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__50_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__51_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__51_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__51_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__51_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__51_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__50_value),LEAN_SCALAR_PTR_LITERAL(220, 64, 194, 165, 28, 0, 228, 77)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__51 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__51_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__52_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "nlinarith"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__52 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__52_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__53_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__53_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__53_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__53_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__53_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__52_value),LEAN_SCALAR_PTR_LITERAL(164, 27, 203, 154, 194, 136, 213, 127)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__53 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__53_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__54_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "tacticNlinarith!_"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__54 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__54_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__55_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__55_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__55_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__55_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__55_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__54_value),LEAN_SCALAR_PTR_LITERAL(245, 86, 150, 6, 80, 44, 0, 181)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__55 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__55_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__56_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "LinearCombination"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__56 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__56_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__57_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "linearCombination"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__57 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__57_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__58_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__58_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__58_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__58_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__58_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__56_value),LEAN_SCALAR_PTR_LITERAL(35, 149, 180, 189, 70, 21, 83, 76)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__58_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__58_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__57_value),LEAN_SCALAR_PTR_LITERAL(157, 213, 151, 22, 157, 163, 3, 66)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__58 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__58_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__59_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "tacticNorm_cast__"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__59 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__59_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__60_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__60_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__60_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__60_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__60_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__60_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__60_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__59_value),LEAN_SCALAR_PTR_LITERAL(236, 48, 221, 117, 176, 58, 9, 174)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__60 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__60_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__61_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Aesop"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__61 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__61_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__62_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Frontend"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__62 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__62_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__63_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "aesopTactic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__63 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__63_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__64_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__61_value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__64_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__64_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__62_value),LEAN_SCALAR_PTR_LITERAL(212, 61, 86, 210, 90, 71, 197, 0)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__64_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__64_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__1_value),LEAN_SCALAR_PTR_LITERAL(13, 69, 189, 207, 58, 198, 186, 160)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__64_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__64_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__63_value),LEAN_SCALAR_PTR_LITERAL(54, 142, 162, 195, 161, 101, 248, 175)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__64 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__64_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__65_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "cfcTac"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__65 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__65_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__66_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__65_value),LEAN_SCALAR_PTR_LITERAL(206, 59, 147, 37, 244, 159, 100, 179)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__66 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__66_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__67_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "cfcZeroTac"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__67 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__67_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__68_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__67_value),LEAN_SCALAR_PTR_LITERAL(188, 136, 137, 154, 178, 76, 249, 30)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__68 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__68_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__69_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "cfcContTac"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__69 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__69_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__70_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__69_value),LEAN_SCALAR_PTR_LITERAL(246, 224, 136, 68, 190, 205, 90, 207)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__70 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__70_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__71_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "tacticContinuity"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__71 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__71_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__72_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__71_value),LEAN_SCALAR_PTR_LITERAL(98, 22, 208, 161, 25, 106, 10, 121)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__72 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__72_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__73_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "measurability"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__73 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__73_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__74_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__74_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__74_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__74_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__74_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__73_value),LEAN_SCALAR_PTR_LITERAL(39, 97, 48, 123, 0, 88, 207, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__74 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__74_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__75_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "finiteness"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__75 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__75_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__76_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__75_value),LEAN_SCALAR_PTR_LITERAL(34, 223, 203, 63, 194, 247, 241, 99)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__76 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__76_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__77_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "finiteness\?"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__77 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__77_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__78_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__77_value),LEAN_SCALAR_PTR_LITERAL(0, 254, 206, 121, 244, 237, 22, 232)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__78 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__78_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__79_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Tauto"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__79 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__79_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__80_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "tauto"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__80 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__80_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__81_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__81_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__81_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__81_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__81_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__79_value),LEAN_SCALAR_PTR_LITERAL(243, 83, 25, 243, 21, 136, 46, 247)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__81_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__81_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__80_value),LEAN_SCALAR_PTR_LITERAL(40, 188, 227, 116, 33, 242, 140, 204)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__81 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__81_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__82_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "split"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__82 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__82_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__83_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__83_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__83_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__83_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__83_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__83_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__83_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__82_value),LEAN_SCALAR_PTR_LITERAL(104, 58, 38, 157, 113, 69, 9, 24)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__83 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__83_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__84_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "splitIfs"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__84 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__84_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__85_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__85_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__85_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__85_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__85_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__84_value),LEAN_SCALAR_PTR_LITERAL(109, 48, 181, 174, 145, 245, 228, 97)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__85 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__85_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__86_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__86;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__87_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__87;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__88_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__88;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__89_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__89;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__90_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__90;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__91_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__91;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__92_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__92;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__93_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__93;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__94_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__94;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__95_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__95;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__96_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__96;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__97_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__97;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__98_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__98;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__99_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__99;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__100_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__100;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__101_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__101;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__102_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__102;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__103_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__103;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__104_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__104;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__105_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__105;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__106_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__106;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__107_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__107;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__108_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__108;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__109_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__109;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__110_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__110;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__111_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__111;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__112_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__112;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__113_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__113;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__114_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__114;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__115_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__115;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__116_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__116;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__117_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__117;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__118_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__118;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__119_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__119;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__120_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__120;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__121_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__121;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__122_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__122;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__123_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__123;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__124_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__124;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__125_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__125;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__126_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__126;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__127_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__127;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__128_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__128;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__129_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__129;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_usesGoal_x3f___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "cases"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_usesGoal_x3f___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_usesGoal_x3f___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_usesGoal_x3f___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "obtain"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_usesGoal_x3f___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_usesGoal_x3f___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_usesGoal_x3f___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "tacticHave__"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_usesGoal_x3f___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_usesGoal_x3f___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_usesGoal_x3f___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "rcases"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_usesGoal_x3f___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_usesGoal_x3f___closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_usesGoal_x3f___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "specialize"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_usesGoal_x3f___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_usesGoal_x3f___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_usesGoal_x3f___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "subst"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_usesGoal_x3f___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_usesGoal_x3f___closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_usesGoal_x3f___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "induction"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_usesGoal_x3f___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_usesGoal_x3f___closed__6_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_usesGoal_x3f___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "cases'"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_usesGoal_x3f___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_usesGoal_x3f___closed__7_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_usesGoal_x3f___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "tacticBy_cases_:_"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_usesGoal_x3f___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_usesGoal_x3f___closed__8_value;
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_usesGoal_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_usesGoal_x3f___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_getFVarIdCandidates_spec__1(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_getFVarIdCandidates_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_getFVarIdCandidates_spec__0_spec__0(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_getFVarIdCandidates_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_getFVarIdCandidates_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_getFVarIdCandidates_spec__0___closed__0 = (const lean_object*)&lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_getFVarIdCandidates_spec__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_getFVarIdCandidates_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_getFVarIdCandidates_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_getFVarIdCandidates(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_getFVarIdCandidates___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_persistFVars(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_elem___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_elem___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__3(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__1_spec__1_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__1_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__1_spec__1___redArg(lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__2___redArg___lam__0(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__2___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 36, .m_capacity = 36, .m_length = 35, .m_data = "'really_persist' could this happen\?"};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__2___redArg___closed__0 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__2___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_Stained_toFMVarId___closed__0_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_Stained_toFMVarId___closed__0_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__1_spec__1(lean_object*, lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__1_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__1_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_isEmpty___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_isEmpty___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_isEmpty___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_isEmpty___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withMCtx___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withMCtx___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withMCtx___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withMCtx___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion___redArg___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion___redArg___lam__0___closed__0;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion___redArg___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion___redArg___lam__0___closed__1;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion___redArg___lam__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion___redArg___lam__0___closed__2;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion___redArg___lam__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getInfoTrees___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__19___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getInfoTrees___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__19___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getInfoTrees___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__19(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getInfoTrees___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__19___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__20_spec__25(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__20_spec__25___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__20_spec__26(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__20_spec__24_spec__26(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__20_spec__24_spec__26___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__20_spec__26___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__20_spec__24___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__20_spec__24___closed__0;
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__20_spec__24(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__20_spec__24___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__20(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__20___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__8_spec__8___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__8_spec__8___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__8___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__8___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__9_spec__10_spec__12_spec__28___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__9_spec__10_spec__12___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__9_spec__10___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__9___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__10___redArg(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__10___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__11(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__12(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__12___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__6___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__6___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__6___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__6(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__7___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__13(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__13___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__14___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__14___redArg___closed__0;
static lean_once_cell_t lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__14___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__14___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__14___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__14___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__5(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__15___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__15___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_beq___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_beq___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__2___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__4___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__4___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Array_isEqvAux___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_isEqvAux___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__1___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__4___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__4___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "lost "};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__4___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__4___closed__0_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__4___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__4___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__4___closed__1_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__4___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ", "};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__4___closed__2 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__4___closed__2_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__4___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__4___closed__3 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__4___closed__3_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__4___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__4___closed__4;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__16_spec__18___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__16_spec__18___closed__0 = (const lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__16_spec__18___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__16_spec__18(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__16_spec__18___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__16(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__16___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__17(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__17___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__21___lam__0(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__21___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__21(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__21___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "`"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__1;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 35, .m_capacity = 35, .m_length = 34, .m_data = "` is a flexible tactic modifying `"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__2 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__2_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__3;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "`."};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__4 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__4_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__5;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0(lean_object*, lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29___lam__0___closed__0_value;
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29_spec__33(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29_spec__33___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29_spec__32___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29_spec__32___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29_spec__32___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29_spec__32___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29_spec__32___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29_spec__32___redArg___closed__2;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29_spec__32___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29_spec__32___redArg___closed__3;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29_spec__32___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29_spec__32___redArg___closed__4;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29_spec__32___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29_spec__32___redArg___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29_spec__32___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29_spec__32___redArg___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__23_spec__31(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__23_spec__31___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__23___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 46, .m_capacity = 46, .m_length = 45, .m_data = "This linter can be disabled with `set_option "};
static const lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__23___closed__0 = (const lean_object*)&lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__23___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__23___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__23___closed__1;
static const lean_string_object lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__23___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = " false`"};
static const lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__23___closed__2 = (const lean_object*)&lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__23___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__23___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__23___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__23(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__23___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "`\nuses `"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__1;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "`, which was modified by "};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__2 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__2_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__3;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "!"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__4 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__4_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__5;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 52, .m_capacity = 52, .m_length = 51, .m_data = "`\nmodifies the current goal, which was modified by "};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__6 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__6_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__7;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 36, .m_capacity = 36, .m_length = 35, .m_data = "`\nuses a rigid tactic. Previously, "};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__8 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__8_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__9;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 87, .m_capacity = 87, .m_length = 86, .m_data = ", which potentially modified all hypotheses and the goal with a wildcard `*`, was used"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__10 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__10_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__11;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "."};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__12 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__12_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__13;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "the flexible tactic "};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__14 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__14_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__14_value)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__15 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__15_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__16;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "a flexible tactic"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__17 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__17_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__17_value)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__18 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__18_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__19;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29___closed__0_value)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__20 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__20_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__21;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = " on line "};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__22 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__22_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "tactic"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__23 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__23_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__23_value),LEAN_SCALAR_PTR_LITERAL(99, 76, 33, 121, 85, 143, 17, 224)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__24 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__24_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Try this:"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__25 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__25_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 45, .m_capacity = 45, .m_length = 44, .m_data = "`. Try `aesop\?` and use the suggested proof."};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__26 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__26_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__27_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__27;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 135, .m_capacity = 135, .m_length = 134, .m_data = "`. Try `simp_all\?` and use the suggested `simp_all only [...]`. Alternatively, use `suffices` to explicitly state the simplified form."};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__28 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__28_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__29_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__29;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 127, .m_capacity = 127, .m_length = 126, .m_data = "`. Try `simp\?` and use the suggested `simp only [...]`. Alternatively, use `suffices` to explicitly state the simplified form."};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__30 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__30_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__31_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__31;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 230, .m_capacity = 230, .m_length = 229, .m_data = "` is a flexible tactic that potentially modifies all hypotheses and the current goal with a wildcard `*`. Try `simp\?` and use the suggested `simp only [...]`. Alternatively, use `suffices` to explicitly state the simplified form."};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__32 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__32_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__33_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__33;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24(uint8_t, uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__18_spec__21___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__18_spec__21___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__18(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__18___boxed(lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_extractTacticData___closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_extractTacticData___closed__1_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___lam__0___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__0_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_withSetOptionIn___boxed, .m_arity = 6, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__0_value)} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__2_value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__4_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(191, 70, 156, 159, 11, 54, 216, 94)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__5_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(37, 204, 154, 235, 250, 222, 148, 114)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__6_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "FlexibleLinter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__6_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__7_value),LEAN_SCALAR_PTR_LITERAL(71, 187, 75, 58, 213, 155, 124, 105)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(58, 86, 165, 135, 208, 58, 11, 148)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__9_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(139, 121, 107, 144, 127, 66, 159, 27)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__10_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(201, 110, 250, 43, 82, 226, 79, 49)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__11_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Flexible"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__12_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__11_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__12_value),LEAN_SCALAR_PTR_LITERAL(23, 136, 187, 237, 127, 248, 187, 4)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__13_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "flexibleLinter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__14_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__13_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__14_value),LEAN_SCALAR_PTR_LITERAL(166, 150, 186, 239, 183, 68, 180, 105)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__15_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__15_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__16_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___closed__16_value;
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__1(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Array_isEqvAux___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_isEqvAux___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__8(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__8___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__9(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__10(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__14(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__14___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__15(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__15___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__18_spec__21(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__18_spec__21___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__8_spec__8(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__8_spec__8___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__9_spec__10(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29_spec__32(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29_spec__32___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__9_spec__10_spec__12(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__9_spec__10_spec__12_spec__28(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_initFn_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_242819373____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_initFn_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_242819373____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__spec__0(lean_object* v_name_1_, lean_object* v_decl_2_, lean_object* v_ref_3_){
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__spec__0___boxed(lean_object* v_name_29_, lean_object* v_decl_30_, lean_object* v_ref_31_, lean_object* v_a_32_){
_start:
{
lean_object* v_res_33_; 
v_res_33_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__spec__0(v_name_29_, v_decl_30_, v_ref_31_);
lean_dec_ref(v_decl_30_);
return v_res_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_53_; lean_object* v___x_54_; lean_object* v___x_55_; lean_object* v___x_56_; 
v___x_53_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4_));
v___x_54_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4_));
v___x_55_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4_));
v___x_56_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4__spec__0(v___x_53_, v___x_54_, v___x_55_);
return v___x_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4____boxed(lean_object* v_a_57_){
_start:
{
lean_object* v_res_58_; 
v_res_58_ = lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4_();
return v_res_58_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f(lean_object* v_x_65_){
_start:
{
if (lean_obj_tag(v_x_65_) == 1)
{
lean_object* v_kind_66_; 
v_kind_66_ = lean_ctor_get(v_x_65_, 1);
if (lean_obj_tag(v_kind_66_) == 1)
{
lean_object* v_pre_67_; 
v_pre_67_ = lean_ctor_get(v_kind_66_, 0);
if (lean_obj_tag(v_pre_67_) == 1)
{
lean_object* v_pre_68_; 
v_pre_68_ = lean_ctor_get(v_pre_67_, 0);
if (lean_obj_tag(v_pre_68_) == 1)
{
lean_object* v_pre_69_; 
v_pre_69_ = lean_ctor_get(v_pre_68_, 0);
if (lean_obj_tag(v_pre_69_) == 1)
{
lean_object* v_pre_70_; 
v_pre_70_ = lean_ctor_get(v_pre_69_, 0);
if (lean_obj_tag(v_pre_70_) == 0)
{
lean_object* v_args_71_; lean_object* v_str_72_; lean_object* v_str_73_; lean_object* v_str_74_; lean_object* v_str_75_; lean_object* v___x_76_; uint8_t v___x_77_; 
v_args_71_ = lean_ctor_get(v_x_65_, 2);
v_str_72_ = lean_ctor_get(v_kind_66_, 1);
v_str_73_ = lean_ctor_get(v_pre_67_, 1);
v_str_74_ = lean_ctor_get(v_pre_68_, 1);
v_str_75_ = lean_ctor_get(v_pre_69_, 1);
v___x_76_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__0));
v___x_77_ = lean_string_dec_eq(v_str_75_, v___x_76_);
if (v___x_77_ == 0)
{
return v___x_77_;
}
else
{
lean_object* v___x_78_; uint8_t v___x_79_; 
v___x_78_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__1));
v___x_79_ = lean_string_dec_eq(v_str_74_, v___x_78_);
if (v___x_79_ == 0)
{
return v___x_79_;
}
else
{
lean_object* v___x_80_; uint8_t v___x_81_; 
v___x_80_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2));
v___x_81_ = lean_string_dec_eq(v_str_73_, v___x_80_);
if (v___x_81_ == 0)
{
return v___x_81_;
}
else
{
lean_object* v___x_82_; uint8_t v___x_83_; 
v___x_82_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__3));
v___x_83_ = lean_string_dec_eq(v_str_72_, v___x_82_);
if (v___x_83_ == 0)
{
lean_object* v___x_84_; uint8_t v___x_85_; 
v___x_84_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__4));
v___x_85_ = lean_string_dec_eq(v_str_72_, v___x_84_);
if (v___x_85_ == 0)
{
return v___x_85_;
}
else
{
lean_object* v___x_86_; lean_object* v___x_87_; uint8_t v___x_88_; 
v___x_86_ = lean_array_get_size(v_args_71_);
v___x_87_ = lean_unsigned_to_nat(5u);
v___x_88_ = lean_nat_dec_eq(v___x_86_, v___x_87_);
if (v___x_88_ == 0)
{
return v___x_88_;
}
else
{
lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; uint8_t v___x_95_; 
v___x_89_ = lean_unsigned_to_nat(3u);
v___x_90_ = lean_array_fget_borrowed(v_args_71_, v___x_89_);
v___x_91_ = lean_unsigned_to_nat(0u);
v___x_92_ = l_Lean_Syntax_getArg(v___x_90_, v___x_91_);
v___x_93_ = l_Lean_Syntax_getAtomVal(v___x_92_);
lean_dec(v___x_92_);
v___x_94_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__5));
v___x_95_ = lean_string_dec_eq(v___x_93_, v___x_94_);
lean_dec_ref(v___x_93_);
if (v___x_95_ == 0)
{
return v___x_88_;
}
else
{
return v___x_83_;
}
}
}
}
else
{
lean_object* v___x_96_; lean_object* v___x_97_; uint8_t v___x_98_; 
v___x_96_ = lean_array_get_size(v_args_71_);
v___x_97_ = lean_unsigned_to_nat(6u);
v___x_98_ = lean_nat_dec_eq(v___x_96_, v___x_97_);
if (v___x_98_ == 0)
{
return v___x_98_;
}
else
{
lean_object* v___x_99_; lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; lean_object* v___x_103_; lean_object* v___x_104_; uint8_t v___x_105_; 
v___x_99_ = lean_unsigned_to_nat(3u);
v___x_100_ = lean_array_fget_borrowed(v_args_71_, v___x_99_);
v___x_101_ = lean_unsigned_to_nat(0u);
v___x_102_ = l_Lean_Syntax_getArg(v___x_100_, v___x_101_);
v___x_103_ = l_Lean_Syntax_getAtomVal(v___x_102_);
lean_dec(v___x_102_);
v___x_104_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__5));
v___x_105_ = lean_string_dec_eq(v___x_103_, v___x_104_);
lean_dec_ref(v___x_103_);
if (v___x_105_ == 0)
{
return v___x_98_;
}
else
{
uint8_t v___x_106_; 
v___x_106_ = 0;
return v___x_106_;
}
}
}
}
}
}
}
else
{
uint8_t v___x_107_; 
v___x_107_ = 0;
return v___x_107_;
}
}
else
{
uint8_t v___x_108_; 
v___x_108_ = 0;
return v___x_108_;
}
}
else
{
uint8_t v___x_109_; 
v___x_109_ = 0;
return v___x_109_;
}
}
else
{
uint8_t v___x_110_; 
v___x_110_ = 0;
return v___x_110_;
}
}
else
{
uint8_t v___x_111_; 
v___x_111_ = 0;
return v___x_111_;
}
}
else
{
uint8_t v___x_112_; 
v___x_112_ = 0;
return v___x_112_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___boxed(lean_object* v_x_113_){
_start:
{
uint8_t v_res_114_; lean_object* v_r_115_; 
v_res_114_ = lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f(v_x_113_);
lean_dec(v_x_113_);
v_r_115_ = lean_box(v_res_114_);
return v_r_115_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Lean_Elab_TacticInfo_goalsTargetedBy_spec__0(lean_object* v_a_116_, lean_object* v_a_117_){
_start:
{
if (lean_obj_tag(v_a_116_) == 0)
{
lean_object* v___x_118_; 
v___x_118_ = l_List_reverse___redArg(v_a_117_);
return v___x_118_;
}
else
{
lean_object* v_head_119_; lean_object* v_tail_120_; lean_object* v___x_122_; uint8_t v_isShared_123_; uint8_t v_isSharedCheck_128_; 
v_head_119_ = lean_ctor_get(v_a_116_, 0);
v_tail_120_ = lean_ctor_get(v_a_116_, 1);
v_isSharedCheck_128_ = !lean_is_exclusive(v_a_116_);
if (v_isSharedCheck_128_ == 0)
{
v___x_122_ = v_a_116_;
v_isShared_123_ = v_isSharedCheck_128_;
goto v_resetjp_121_;
}
else
{
lean_inc(v_tail_120_);
lean_inc(v_head_119_);
lean_dec(v_a_116_);
v___x_122_ = lean_box(0);
v_isShared_123_ = v_isSharedCheck_128_;
goto v_resetjp_121_;
}
v_resetjp_121_:
{
lean_object* v___x_125_; 
if (v_isShared_123_ == 0)
{
lean_ctor_set(v___x_122_, 1, v_a_117_);
v___x_125_ = v___x_122_;
goto v_reusejp_124_;
}
else
{
lean_object* v_reuseFailAlloc_127_; 
v_reuseFailAlloc_127_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_127_, 0, v_head_119_);
lean_ctor_set(v_reuseFailAlloc_127_, 1, v_a_117_);
v___x_125_ = v_reuseFailAlloc_127_;
goto v_reusejp_124_;
}
v_reusejp_124_:
{
v_a_116_ = v_tail_120_;
v_a_117_ = v___x_125_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_filterTR_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Lean_Elab_TacticInfo_goalsTargetedBy_spec__1(lean_object* v_t_130_, lean_object* v_a_131_, lean_object* v_a_132_){
_start:
{
if (lean_obj_tag(v_a_131_) == 0)
{
lean_object* v___x_133_; 
lean_dec_ref(v_t_130_);
v___x_133_ = l_List_reverse___redArg(v_a_132_);
return v___x_133_;
}
else
{
lean_object* v_head_134_; lean_object* v_tail_135_; lean_object* v___x_137_; uint8_t v_isShared_138_; uint8_t v_isSharedCheck_149_; 
v_head_134_ = lean_ctor_get(v_a_131_, 0);
v_tail_135_ = lean_ctor_get(v_a_131_, 1);
v_isSharedCheck_149_ = !lean_is_exclusive(v_a_131_);
if (v_isSharedCheck_149_ == 0)
{
v___x_137_ = v_a_131_;
v_isShared_138_ = v_isSharedCheck_149_;
goto v_resetjp_136_;
}
else
{
lean_inc(v_tail_135_);
lean_inc(v_head_134_);
lean_dec(v_a_131_);
v___x_137_ = lean_box(0);
v_isShared_138_ = v_isSharedCheck_149_;
goto v_resetjp_136_;
}
v_resetjp_136_:
{
lean_object* v_goalsAfter_139_; lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; uint8_t v___x_143_; 
v_goalsAfter_139_ = lean_ctor_get(v_t_130_, 4);
v___x_140_ = ((lean_object*)(lp_mathlib_List_filterTR_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Lean_Elab_TacticInfo_goalsTargetedBy_spec__1___closed__0));
v___x_141_ = lean_box(0);
lean_inc(v_goalsAfter_139_);
v___x_142_ = lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Lean_Elab_TacticInfo_goalsTargetedBy_spec__0(v_goalsAfter_139_, v___x_141_);
lean_inc(v_head_134_);
v___x_143_ = l_List_elem___redArg(v___x_140_, v_head_134_, v___x_142_);
if (v___x_143_ == 0)
{
lean_object* v___x_145_; 
if (v_isShared_138_ == 0)
{
lean_ctor_set(v___x_137_, 1, v_a_132_);
v___x_145_ = v___x_137_;
goto v_reusejp_144_;
}
else
{
lean_object* v_reuseFailAlloc_147_; 
v_reuseFailAlloc_147_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_147_, 0, v_head_134_);
lean_ctor_set(v_reuseFailAlloc_147_, 1, v_a_132_);
v___x_145_ = v_reuseFailAlloc_147_;
goto v_reusejp_144_;
}
v_reusejp_144_:
{
v_a_131_ = v_tail_135_;
v_a_132_ = v___x_145_;
goto _start;
}
}
else
{
lean_del_object(v___x_137_);
lean_dec(v_head_134_);
v_a_131_ = v_tail_135_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Lean_Elab_TacticInfo_goalsTargetedBy(lean_object* v_t_150_){
_start:
{
lean_object* v_goalsBefore_151_; lean_object* v___x_152_; lean_object* v___x_153_; 
v_goalsBefore_151_ = lean_ctor_get(v_t_150_, 2);
lean_inc(v_goalsBefore_151_);
v___x_152_ = lean_box(0);
v___x_153_ = lp_mathlib_List_filterTR_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Lean_Elab_TacticInfo_goalsTargetedBy_spec__1(v_t_150_, v_goalsBefore_151_, v___x_152_);
return v___x_153_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_filterTR_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Lean_Elab_TacticInfo_goalsCreatedBy_spec__0(lean_object* v_t_154_, lean_object* v_a_155_, lean_object* v_a_156_){
_start:
{
if (lean_obj_tag(v_a_155_) == 0)
{
lean_object* v___x_157_; 
lean_dec_ref(v_t_154_);
v___x_157_ = l_List_reverse___redArg(v_a_156_);
return v___x_157_;
}
else
{
lean_object* v_head_158_; lean_object* v_tail_159_; lean_object* v___x_161_; uint8_t v_isShared_162_; uint8_t v_isSharedCheck_173_; 
v_head_158_ = lean_ctor_get(v_a_155_, 0);
v_tail_159_ = lean_ctor_get(v_a_155_, 1);
v_isSharedCheck_173_ = !lean_is_exclusive(v_a_155_);
if (v_isSharedCheck_173_ == 0)
{
v___x_161_ = v_a_155_;
v_isShared_162_ = v_isSharedCheck_173_;
goto v_resetjp_160_;
}
else
{
lean_inc(v_tail_159_);
lean_inc(v_head_158_);
lean_dec(v_a_155_);
v___x_161_ = lean_box(0);
v_isShared_162_ = v_isSharedCheck_173_;
goto v_resetjp_160_;
}
v_resetjp_160_:
{
lean_object* v_goalsBefore_163_; lean_object* v___x_164_; lean_object* v___x_165_; lean_object* v___x_166_; uint8_t v___x_167_; 
v_goalsBefore_163_ = lean_ctor_get(v_t_154_, 2);
v___x_164_ = ((lean_object*)(lp_mathlib_List_filterTR_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Lean_Elab_TacticInfo_goalsTargetedBy_spec__1___closed__0));
v___x_165_ = lean_box(0);
lean_inc(v_goalsBefore_163_);
v___x_166_ = lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Lean_Elab_TacticInfo_goalsTargetedBy_spec__0(v_goalsBefore_163_, v___x_165_);
lean_inc(v_head_158_);
v___x_167_ = l_List_elem___redArg(v___x_164_, v_head_158_, v___x_166_);
if (v___x_167_ == 0)
{
lean_object* v___x_169_; 
if (v_isShared_162_ == 0)
{
lean_ctor_set(v___x_161_, 1, v_a_156_);
v___x_169_ = v___x_161_;
goto v_reusejp_168_;
}
else
{
lean_object* v_reuseFailAlloc_171_; 
v_reuseFailAlloc_171_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_171_, 0, v_head_158_);
lean_ctor_set(v_reuseFailAlloc_171_, 1, v_a_156_);
v___x_169_ = v_reuseFailAlloc_171_;
goto v_reusejp_168_;
}
v_reusejp_168_:
{
v_a_155_ = v_tail_159_;
v_a_156_ = v___x_169_;
goto _start;
}
}
else
{
lean_del_object(v___x_161_);
lean_dec(v_head_158_);
v_a_155_ = v_tail_159_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Lean_Elab_TacticInfo_goalsCreatedBy(lean_object* v_t_174_){
_start:
{
lean_object* v_goalsAfter_175_; lean_object* v___x_176_; lean_object* v___x_177_; 
v_goalsAfter_175_ = lean_ctor_get(v_t_174_, 4);
lean_inc(v_goalsAfter_175_);
v___x_176_ = lean_box(0);
v___x_177_ = lp_mathlib_List_filterTR_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Lean_Elab_TacticInfo_goalsCreatedBy_spec__0(v_t_174_, v_goalsAfter_175_, v___x_176_);
return v___x_177_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_extractTacticData___lam__0(lean_object* v_ci_178_, lean_object* v_info_179_, lean_object* v_acc_180_){
_start:
{
if (lean_obj_tag(v_info_179_) == 0)
{
lean_object* v_i_181_; lean_object* v_toElabInfo_182_; lean_object* v_mctxBefore_183_; lean_object* v_mctxAfter_184_; lean_object* v_stx_185_; uint8_t v___x_186_; lean_object* v___x_187_; 
v_i_181_ = lean_ctor_get(v_info_179_, 0);
lean_inc_ref(v_i_181_);
lean_dec_ref_known(v_info_179_, 1);
v_toElabInfo_182_ = lean_ctor_get(v_i_181_, 0);
v_mctxBefore_183_ = lean_ctor_get(v_i_181_, 1);
lean_inc_ref(v_mctxBefore_183_);
v_mctxAfter_184_ = lean_ctor_get(v_i_181_, 3);
lean_inc_ref(v_mctxAfter_184_);
v_stx_185_ = lean_ctor_get(v_toElabInfo_182_, 1);
lean_inc(v_stx_185_);
v___x_186_ = 1;
v___x_187_ = l_Lean_Syntax_getRange_x3f(v_stx_185_, v___x_186_);
if (lean_obj_tag(v___x_187_) == 0)
{
lean_dec(v_stx_185_);
lean_dec_ref(v_mctxAfter_184_);
lean_dec_ref(v_mctxBefore_183_);
lean_dec_ref(v_i_181_);
lean_dec_ref(v_ci_178_);
return v_acc_180_;
}
else
{
lean_object* v___x_188_; lean_object* v___x_189_; lean_object* v___x_190_; lean_object* v___x_191_; 
lean_dec_ref_known(v___x_187_, 1);
lean_inc_ref(v_i_181_);
v___x_188_ = lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Lean_Elab_TacticInfo_goalsTargetedBy(v_i_181_);
v___x_189_ = lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Lean_Elab_TacticInfo_goalsCreatedBy(v_i_181_);
v___x_190_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_190_, 0, v_stx_185_);
lean_ctor_set(v___x_190_, 1, v_ci_178_);
lean_ctor_set(v___x_190_, 2, v_mctxBefore_183_);
lean_ctor_set(v___x_190_, 3, v_mctxAfter_184_);
lean_ctor_set(v___x_190_, 4, v___x_188_);
lean_ctor_set(v___x_190_, 5, v___x_189_);
v___x_191_ = lean_array_push(v_acc_180_, v___x_190_);
return v___x_191_;
}
}
else
{
lean_dec_ref(v_info_179_);
lean_dec_ref(v_ci_178_);
return v_acc_180_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_extractTacticData(lean_object* v_tree_195_){
_start:
{
lean_object* v___f_196_; lean_object* v___x_197_; lean_object* v___x_198_; 
v___f_196_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_extractTacticData___closed__0));
v___x_197_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_extractTacticData___closed__1));
v___x_198_ = l_Lean_Elab_InfoTree_foldInfo___redArg(v___f_196_, v___x_197_, v_tree_195_);
return v___x_198_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_Stained_ctorIdx(lean_object* v_x_199_){
_start:
{
switch(lean_obj_tag(v_x_199_))
{
case 0:
{
lean_object* v___x_200_; 
v___x_200_ = lean_unsigned_to_nat(0u);
return v___x_200_;
}
case 1:
{
lean_object* v___x_201_; 
v___x_201_ = lean_unsigned_to_nat(1u);
return v___x_201_;
}
default: 
{
lean_object* v___x_202_; 
v___x_202_ = lean_unsigned_to_nat(2u);
return v___x_202_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_Stained_ctorIdx___boxed(lean_object* v_x_203_){
_start:
{
lean_object* v_res_204_; 
v_res_204_ = lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_Stained_ctorIdx(v_x_203_);
lean_dec(v_x_203_);
return v_res_204_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_Stained_ctorElim___redArg(lean_object* v_t_205_, lean_object* v_k_206_){
_start:
{
if (lean_obj_tag(v_t_205_) == 0)
{
lean_object* v_a_207_; lean_object* v___x_208_; 
v_a_207_ = lean_ctor_get(v_t_205_, 0);
lean_inc(v_a_207_);
lean_dec_ref_known(v_t_205_, 1);
v___x_208_ = lean_apply_1(v_k_206_, v_a_207_);
return v___x_208_;
}
else
{
lean_dec(v_t_205_);
return v_k_206_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_Stained_ctorElim(lean_object* v_motive_209_, lean_object* v_ctorIdx_210_, lean_object* v_t_211_, lean_object* v_h_212_, lean_object* v_k_213_){
_start:
{
lean_object* v___x_214_; 
v___x_214_ = lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_Stained_ctorElim___redArg(v_t_211_, v_k_213_);
return v___x_214_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_Stained_ctorElim___boxed(lean_object* v_motive_215_, lean_object* v_ctorIdx_216_, lean_object* v_t_217_, lean_object* v_h_218_, lean_object* v_k_219_){
_start:
{
lean_object* v_res_220_; 
v_res_220_ = lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_Stained_ctorElim(v_motive_215_, v_ctorIdx_216_, v_t_217_, v_h_218_, v_k_219_);
lean_dec(v_ctorIdx_216_);
return v_res_220_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_Stained_name_elim___redArg(lean_object* v_t_221_, lean_object* v_name_222_){
_start:
{
lean_object* v___x_223_; 
v___x_223_ = lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_Stained_ctorElim___redArg(v_t_221_, v_name_222_);
return v___x_223_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_Stained_name_elim(lean_object* v_motive_224_, lean_object* v_t_225_, lean_object* v_h_226_, lean_object* v_name_227_){
_start:
{
lean_object* v___x_228_; 
v___x_228_ = lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_Stained_ctorElim___redArg(v_t_225_, v_name_227_);
return v___x_228_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_Stained_goal_elim___redArg(lean_object* v_t_229_, lean_object* v_goal_230_){
_start:
{
lean_object* v___x_231_; 
v___x_231_ = lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_Stained_ctorElim___redArg(v_t_229_, v_goal_230_);
return v___x_231_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_Stained_goal_elim(lean_object* v_motive_232_, lean_object* v_t_233_, lean_object* v_h_234_, lean_object* v_goal_235_){
_start:
{
lean_object* v___x_236_; 
v___x_236_ = lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_Stained_ctorElim___redArg(v_t_233_, v_goal_235_);
return v___x_236_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_Stained_wildcard_elim___redArg(lean_object* v_t_237_, lean_object* v_wildcard_238_){
_start:
{
lean_object* v___x_239_; 
v___x_239_ = lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_Stained_ctorElim___redArg(v_t_237_, v_wildcard_238_);
return v___x_239_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_Stained_wildcard_elim(lean_object* v_motive_240_, lean_object* v_t_241_, lean_object* v_h_242_, lean_object* v_wildcard_243_){
_start:
{
lean_object* v___x_244_; 
v___x_244_ = lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_Stained_ctorElim___redArg(v_t_241_, v_wildcard_243_);
return v___x_244_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr___closed__7(void){
_start:
{
lean_object* v___x_257_; lean_object* v___x_258_; 
v___x_257_ = lean_unsigned_to_nat(2u);
v___x_258_ = lean_nat_to_int(v___x_257_);
return v___x_258_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr___closed__8(void){
_start:
{
lean_object* v___x_259_; lean_object* v___x_260_; 
v___x_259_ = lean_unsigned_to_nat(1u);
v___x_260_ = lean_nat_to_int(v___x_259_);
return v___x_260_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr(lean_object* v_x_261_, lean_object* v_prec_262_){
_start:
{
lean_object* v___y_264_; lean_object* v___y_271_; 
switch(lean_obj_tag(v_x_261_))
{
case 0:
{
lean_object* v_a_277_; lean_object* v___y_279_; lean_object* v___x_288_; uint8_t v___x_289_; 
v_a_277_ = lean_ctor_get(v_x_261_, 0);
lean_inc(v_a_277_);
lean_dec_ref_known(v_x_261_, 1);
v___x_288_ = lean_unsigned_to_nat(1024u);
v___x_289_ = lean_nat_dec_le(v___x_288_, v_prec_262_);
if (v___x_289_ == 0)
{
lean_object* v___x_290_; 
v___x_290_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr___closed__7, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr___closed__7_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr___closed__7);
v___y_279_ = v___x_290_;
goto v___jp_278_;
}
else
{
lean_object* v___x_291_; 
v___x_291_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr___closed__8, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr___closed__8_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr___closed__8);
v___y_279_ = v___x_291_;
goto v___jp_278_;
}
v___jp_278_:
{
lean_object* v___x_280_; lean_object* v___x_281_; lean_object* v___x_282_; lean_object* v___x_283_; lean_object* v___x_284_; uint8_t v___x_285_; lean_object* v___x_286_; lean_object* v___x_287_; 
v___x_280_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr___closed__6));
v___x_281_ = lean_unsigned_to_nat(1024u);
v___x_282_ = l_Lean_Name_reprPrec(v_a_277_, v___x_281_);
v___x_283_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_283_, 0, v___x_280_);
lean_ctor_set(v___x_283_, 1, v___x_282_);
lean_inc(v___y_279_);
v___x_284_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_284_, 0, v___y_279_);
lean_ctor_set(v___x_284_, 1, v___x_283_);
v___x_285_ = 0;
v___x_286_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_286_, 0, v___x_284_);
lean_ctor_set_uint8(v___x_286_, sizeof(void*)*1, v___x_285_);
v___x_287_ = l_Repr_addAppParen(v___x_286_, v_prec_262_);
return v___x_287_;
}
}
case 1:
{
lean_object* v___x_292_; uint8_t v___x_293_; 
v___x_292_ = lean_unsigned_to_nat(1024u);
v___x_293_ = lean_nat_dec_le(v___x_292_, v_prec_262_);
if (v___x_293_ == 0)
{
lean_object* v___x_294_; 
v___x_294_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr___closed__7, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr___closed__7_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr___closed__7);
v___y_264_ = v___x_294_;
goto v___jp_263_;
}
else
{
lean_object* v___x_295_; 
v___x_295_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr___closed__8, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr___closed__8_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr___closed__8);
v___y_264_ = v___x_295_;
goto v___jp_263_;
}
}
default: 
{
lean_object* v___x_296_; uint8_t v___x_297_; 
v___x_296_ = lean_unsigned_to_nat(1024u);
v___x_297_ = lean_nat_dec_le(v___x_296_, v_prec_262_);
if (v___x_297_ == 0)
{
lean_object* v___x_298_; 
v___x_298_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr___closed__7, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr___closed__7_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr___closed__7);
v___y_271_ = v___x_298_;
goto v___jp_270_;
}
else
{
lean_object* v___x_299_; 
v___x_299_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr___closed__8, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr___closed__8_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr___closed__8);
v___y_271_ = v___x_299_;
goto v___jp_270_;
}
}
}
v___jp_263_:
{
lean_object* v___x_265_; lean_object* v___x_266_; uint8_t v___x_267_; lean_object* v___x_268_; lean_object* v___x_269_; 
v___x_265_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr___closed__1));
lean_inc(v___y_264_);
v___x_266_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_266_, 0, v___y_264_);
lean_ctor_set(v___x_266_, 1, v___x_265_);
v___x_267_ = 0;
v___x_268_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_268_, 0, v___x_266_);
lean_ctor_set_uint8(v___x_268_, sizeof(void*)*1, v___x_267_);
v___x_269_ = l_Repr_addAppParen(v___x_268_, v_prec_262_);
return v___x_269_;
}
v___jp_270_:
{
lean_object* v___x_272_; lean_object* v___x_273_; uint8_t v___x_274_; lean_object* v___x_275_; lean_object* v___x_276_; 
v___x_272_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr___closed__3));
lean_inc(v___y_271_);
v___x_273_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_273_, 0, v___y_271_);
lean_ctor_set(v___x_273_, 1, v___x_272_);
v___x_274_ = 0;
v___x_275_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_275_, 0, v___x_273_);
lean_ctor_set_uint8(v___x_275_, sizeof(void*)*1, v___x_274_);
v___x_276_ = l_Repr_addAppParen(v___x_275_, v_prec_262_);
return v___x_276_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr___boxed(lean_object* v_x_300_, lean_object* v_prec_301_){
_start:
{
lean_object* v_res_302_; 
v_res_302_ = lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instReprStained_repr(v_x_300_, v_prec_301_);
lean_dec(v_prec_301_);
return v_res_302_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instDecidableEqStained_decEq(lean_object* v_x_309_, lean_object* v_x_310_){
_start:
{
switch(lean_obj_tag(v_x_309_))
{
case 0:
{
lean_object* v_a_311_; uint8_t v___x_312_; 
v_a_311_ = lean_ctor_get(v_x_309_, 0);
v___x_312_ = 0;
if (lean_obj_tag(v_x_310_) == 0)
{
lean_object* v_a_313_; uint8_t v___x_314_; 
v_a_313_ = lean_ctor_get(v_x_310_, 0);
v___x_314_ = lean_name_eq(v_a_311_, v_a_313_);
if (v___x_314_ == 0)
{
return v___x_312_;
}
else
{
return v___x_314_;
}
}
else
{
return v___x_312_;
}
}
case 1:
{
if (lean_obj_tag(v_x_310_) == 1)
{
uint8_t v___x_315_; 
v___x_315_ = 1;
return v___x_315_;
}
else
{
uint8_t v___x_316_; 
v___x_316_ = 0;
return v___x_316_;
}
}
default: 
{
if (lean_obj_tag(v_x_310_) == 2)
{
uint8_t v___x_317_; 
v___x_317_ = 1;
return v___x_317_;
}
else
{
uint8_t v___x_318_; 
v___x_318_ = 0;
return v___x_318_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instDecidableEqStained_decEq___boxed(lean_object* v_x_319_, lean_object* v_x_320_){
_start:
{
uint8_t v_res_321_; lean_object* v_r_322_; 
v_res_321_ = lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instDecidableEqStained_decEq(v_x_319_, v_x_320_);
lean_dec(v_x_320_);
lean_dec(v_x_319_);
v_r_322_ = lean_box(v_res_321_);
return v_r_322_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instDecidableEqStained(lean_object* v_x_323_, lean_object* v_x_324_){
_start:
{
uint8_t v___x_325_; 
v___x_325_ = lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instDecidableEqStained_decEq(v_x_323_, v_x_324_);
return v___x_325_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instDecidableEqStained___boxed(lean_object* v_x_326_, lean_object* v_x_327_){
_start:
{
uint8_t v_res_328_; lean_object* v_r_329_; 
v_res_328_ = lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instDecidableEqStained(v_x_326_, v_x_327_);
lean_dec(v_x_327_);
lean_dec(v_x_326_);
v_r_329_ = lean_box(v_res_328_);
return v_r_329_;
}
}
static uint64_t _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instHashableStained_hash___closed__0(void){
_start:
{
uint64_t v___x_330_; uint64_t v___x_331_; uint64_t v___x_332_; 
v___x_330_ = 1723ULL;
v___x_331_ = 0ULL;
v___x_332_ = lean_uint64_mix_hash(v___x_331_, v___x_330_);
return v___x_332_;
}
}
LEAN_EXPORT uint64_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instHashableStained_hash(lean_object* v_x_333_){
_start:
{
switch(lean_obj_tag(v_x_333_))
{
case 0:
{
lean_object* v_a_334_; uint64_t v___x_335_; 
v_a_334_ = lean_ctor_get(v_x_333_, 0);
v___x_335_ = 0ULL;
if (lean_obj_tag(v_a_334_) == 0)
{
uint64_t v___x_336_; 
v___x_336_ = lean_uint64_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instHashableStained_hash___closed__0, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instHashableStained_hash___closed__0_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instHashableStained_hash___closed__0);
return v___x_336_;
}
else
{
uint64_t v_hash_337_; uint64_t v___x_338_; 
v_hash_337_ = lean_ctor_get_uint64(v_a_334_, sizeof(void*)*2);
v___x_338_ = lean_uint64_mix_hash(v___x_335_, v_hash_337_);
return v___x_338_;
}
}
case 1:
{
uint64_t v___x_339_; 
v___x_339_ = 1ULL;
return v___x_339_;
}
default: 
{
uint64_t v___x_340_; 
v___x_340_ = 2ULL;
return v___x_340_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instHashableStained_hash___boxed(lean_object* v_x_341_){
_start:
{
uint64_t v_res_342_; lean_object* v_r_343_; 
v_res_342_ = lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instHashableStained_hash(v_x_341_);
lean_dec(v_x_341_);
v_r_343_ = lean_box_uint64(v_res_342_);
return v_r_343_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instToStringStained___lam__0(lean_object* v_x_348_){
_start:
{
switch(lean_obj_tag(v_x_348_))
{
case 0:
{
lean_object* v_a_349_; uint8_t v___x_350_; lean_object* v___x_351_; 
v_a_349_ = lean_ctor_get(v_x_348_, 0);
lean_inc(v_a_349_);
lean_dec_ref_known(v_x_348_, 1);
v___x_350_ = 1;
v___x_351_ = l_Lean_Name_toString(v_a_349_, v___x_350_);
return v___x_351_;
}
case 1:
{
lean_object* v___x_352_; 
v___x_352_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instToStringStained___lam__0___closed__0));
return v___x_352_;
}
default: 
{
lean_object* v___x_353_; 
v___x_353_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instToStringStained___lam__0___closed__1));
return v___x_353_;
}
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__0_spec__0___redArg(lean_object* v_a_356_, lean_object* v_x_357_){
_start:
{
if (lean_obj_tag(v_x_357_) == 0)
{
uint8_t v___x_358_; 
v___x_358_ = 0;
return v___x_358_;
}
else
{
lean_object* v_key_359_; lean_object* v_tail_360_; uint8_t v___x_361_; 
v_key_359_ = lean_ctor_get(v_x_357_, 0);
v_tail_360_ = lean_ctor_get(v_x_357_, 2);
v___x_361_ = lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instDecidableEqStained_decEq(v_key_359_, v_a_356_);
if (v___x_361_ == 0)
{
v_x_357_ = v_tail_360_;
goto _start;
}
else
{
return v___x_361_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__0_spec__0___redArg___boxed(lean_object* v_a_363_, lean_object* v_x_364_){
_start:
{
uint8_t v_res_365_; lean_object* v_r_366_; 
v_res_365_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__0_spec__0___redArg(v_a_363_, v_x_364_);
lean_dec(v_x_364_);
lean_dec(v_a_363_);
v_r_366_ = lean_box(v_res_365_);
return v_r_366_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__0_spec__1_spec__2_spec__8___redArg(lean_object* v_x_367_, lean_object* v_x_368_){
_start:
{
if (lean_obj_tag(v_x_368_) == 0)
{
return v_x_367_;
}
else
{
lean_object* v_key_369_; lean_object* v_value_370_; lean_object* v_tail_371_; lean_object* v___x_373_; uint8_t v_isShared_374_; uint8_t v_isSharedCheck_394_; 
v_key_369_ = lean_ctor_get(v_x_368_, 0);
v_value_370_ = lean_ctor_get(v_x_368_, 1);
v_tail_371_ = lean_ctor_get(v_x_368_, 2);
v_isSharedCheck_394_ = !lean_is_exclusive(v_x_368_);
if (v_isSharedCheck_394_ == 0)
{
v___x_373_ = v_x_368_;
v_isShared_374_ = v_isSharedCheck_394_;
goto v_resetjp_372_;
}
else
{
lean_inc(v_tail_371_);
lean_inc(v_value_370_);
lean_inc(v_key_369_);
lean_dec(v_x_368_);
v___x_373_ = lean_box(0);
v_isShared_374_ = v_isSharedCheck_394_;
goto v_resetjp_372_;
}
v_resetjp_372_:
{
lean_object* v___x_375_; uint64_t v___x_376_; uint64_t v___x_377_; uint64_t v___x_378_; uint64_t v_fold_379_; uint64_t v___x_380_; uint64_t v___x_381_; uint64_t v___x_382_; size_t v___x_383_; size_t v___x_384_; size_t v___x_385_; size_t v___x_386_; size_t v___x_387_; lean_object* v___x_388_; lean_object* v___x_390_; 
v___x_375_ = lean_array_get_size(v_x_367_);
v___x_376_ = lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instHashableStained_hash(v_key_369_);
v___x_377_ = 32ULL;
v___x_378_ = lean_uint64_shift_right(v___x_376_, v___x_377_);
v_fold_379_ = lean_uint64_xor(v___x_376_, v___x_378_);
v___x_380_ = 16ULL;
v___x_381_ = lean_uint64_shift_right(v_fold_379_, v___x_380_);
v___x_382_ = lean_uint64_xor(v_fold_379_, v___x_381_);
v___x_383_ = lean_uint64_to_usize(v___x_382_);
v___x_384_ = lean_usize_of_nat(v___x_375_);
v___x_385_ = ((size_t)1ULL);
v___x_386_ = lean_usize_sub(v___x_384_, v___x_385_);
v___x_387_ = lean_usize_land(v___x_383_, v___x_386_);
v___x_388_ = lean_array_uget_borrowed(v_x_367_, v___x_387_);
lean_inc(v___x_388_);
if (v_isShared_374_ == 0)
{
lean_ctor_set(v___x_373_, 2, v___x_388_);
v___x_390_ = v___x_373_;
goto v_reusejp_389_;
}
else
{
lean_object* v_reuseFailAlloc_393_; 
v_reuseFailAlloc_393_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_393_, 0, v_key_369_);
lean_ctor_set(v_reuseFailAlloc_393_, 1, v_value_370_);
lean_ctor_set(v_reuseFailAlloc_393_, 2, v___x_388_);
v___x_390_ = v_reuseFailAlloc_393_;
goto v_reusejp_389_;
}
v_reusejp_389_:
{
lean_object* v___x_391_; 
v___x_391_ = lean_array_uset(v_x_367_, v___x_387_, v___x_390_);
v_x_367_ = v___x_391_;
v_x_368_ = v_tail_371_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__0_spec__1_spec__2___redArg(lean_object* v_i_395_, lean_object* v_source_396_, lean_object* v_target_397_){
_start:
{
lean_object* v___x_398_; uint8_t v___x_399_; 
v___x_398_ = lean_array_get_size(v_source_396_);
v___x_399_ = lean_nat_dec_lt(v_i_395_, v___x_398_);
if (v___x_399_ == 0)
{
lean_dec_ref(v_source_396_);
lean_dec(v_i_395_);
return v_target_397_;
}
else
{
lean_object* v_es_400_; lean_object* v___x_401_; lean_object* v_source_402_; lean_object* v_target_403_; lean_object* v___x_404_; lean_object* v___x_405_; 
v_es_400_ = lean_array_fget(v_source_396_, v_i_395_);
v___x_401_ = lean_box(0);
v_source_402_ = lean_array_fset(v_source_396_, v_i_395_, v___x_401_);
v_target_403_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__0_spec__1_spec__2_spec__8___redArg(v_target_397_, v_es_400_);
v___x_404_ = lean_unsigned_to_nat(1u);
v___x_405_ = lean_nat_add(v_i_395_, v___x_404_);
lean_dec(v_i_395_);
v_i_395_ = v___x_405_;
v_source_396_ = v_source_402_;
v_target_397_ = v_target_403_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__0_spec__1___redArg(lean_object* v_data_407_){
_start:
{
lean_object* v___x_408_; lean_object* v___x_409_; lean_object* v_nbuckets_410_; lean_object* v___x_411_; lean_object* v___x_412_; lean_object* v___x_413_; lean_object* v___x_414_; 
v___x_408_ = lean_array_get_size(v_data_407_);
v___x_409_ = lean_unsigned_to_nat(2u);
v_nbuckets_410_ = lean_nat_mul(v___x_408_, v___x_409_);
v___x_411_ = lean_unsigned_to_nat(0u);
v___x_412_ = lean_box(0);
v___x_413_ = lean_mk_array(v_nbuckets_410_, v___x_412_);
v___x_414_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__0_spec__1_spec__2___redArg(v___x_411_, v_data_407_, v___x_413_);
return v___x_414_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__0___redArg(lean_object* v_m_415_, lean_object* v_a_416_, lean_object* v_b_417_){
_start:
{
lean_object* v_size_418_; lean_object* v_buckets_419_; lean_object* v___x_420_; uint64_t v___x_421_; uint64_t v___x_422_; uint64_t v___x_423_; uint64_t v_fold_424_; uint64_t v___x_425_; uint64_t v___x_426_; uint64_t v___x_427_; size_t v___x_428_; size_t v___x_429_; size_t v___x_430_; size_t v___x_431_; size_t v___x_432_; lean_object* v_bkt_433_; uint8_t v___x_434_; 
v_size_418_ = lean_ctor_get(v_m_415_, 0);
v_buckets_419_ = lean_ctor_get(v_m_415_, 1);
v___x_420_ = lean_array_get_size(v_buckets_419_);
v___x_421_ = lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instHashableStained_hash(v_a_416_);
v___x_422_ = 32ULL;
v___x_423_ = lean_uint64_shift_right(v___x_421_, v___x_422_);
v_fold_424_ = lean_uint64_xor(v___x_421_, v___x_423_);
v___x_425_ = 16ULL;
v___x_426_ = lean_uint64_shift_right(v_fold_424_, v___x_425_);
v___x_427_ = lean_uint64_xor(v_fold_424_, v___x_426_);
v___x_428_ = lean_uint64_to_usize(v___x_427_);
v___x_429_ = lean_usize_of_nat(v___x_420_);
v___x_430_ = ((size_t)1ULL);
v___x_431_ = lean_usize_sub(v___x_429_, v___x_430_);
v___x_432_ = lean_usize_land(v___x_428_, v___x_431_);
v_bkt_433_ = lean_array_uget_borrowed(v_buckets_419_, v___x_432_);
v___x_434_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__0_spec__0___redArg(v_a_416_, v_bkt_433_);
if (v___x_434_ == 0)
{
lean_object* v___x_436_; uint8_t v_isShared_437_; uint8_t v_isSharedCheck_455_; 
lean_inc_ref(v_buckets_419_);
lean_inc(v_size_418_);
v_isSharedCheck_455_ = !lean_is_exclusive(v_m_415_);
if (v_isSharedCheck_455_ == 0)
{
lean_object* v_unused_456_; lean_object* v_unused_457_; 
v_unused_456_ = lean_ctor_get(v_m_415_, 1);
lean_dec(v_unused_456_);
v_unused_457_ = lean_ctor_get(v_m_415_, 0);
lean_dec(v_unused_457_);
v___x_436_ = v_m_415_;
v_isShared_437_ = v_isSharedCheck_455_;
goto v_resetjp_435_;
}
else
{
lean_dec(v_m_415_);
v___x_436_ = lean_box(0);
v_isShared_437_ = v_isSharedCheck_455_;
goto v_resetjp_435_;
}
v_resetjp_435_:
{
lean_object* v___x_438_; lean_object* v_size_x27_439_; lean_object* v___x_440_; lean_object* v_buckets_x27_441_; lean_object* v___x_442_; lean_object* v___x_443_; lean_object* v___x_444_; lean_object* v___x_445_; lean_object* v___x_446_; uint8_t v___x_447_; 
v___x_438_ = lean_unsigned_to_nat(1u);
v_size_x27_439_ = lean_nat_add(v_size_418_, v___x_438_);
lean_dec(v_size_418_);
lean_inc(v_bkt_433_);
v___x_440_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_440_, 0, v_a_416_);
lean_ctor_set(v___x_440_, 1, v_b_417_);
lean_ctor_set(v___x_440_, 2, v_bkt_433_);
v_buckets_x27_441_ = lean_array_uset(v_buckets_419_, v___x_432_, v___x_440_);
v___x_442_ = lean_unsigned_to_nat(4u);
v___x_443_ = lean_nat_mul(v_size_x27_439_, v___x_442_);
v___x_444_ = lean_unsigned_to_nat(3u);
v___x_445_ = lean_nat_div(v___x_443_, v___x_444_);
lean_dec(v___x_443_);
v___x_446_ = lean_array_get_size(v_buckets_x27_441_);
v___x_447_ = lean_nat_dec_le(v___x_445_, v___x_446_);
lean_dec(v___x_445_);
if (v___x_447_ == 0)
{
lean_object* v_val_448_; lean_object* v___x_450_; 
v_val_448_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__0_spec__1___redArg(v_buckets_x27_441_);
if (v_isShared_437_ == 0)
{
lean_ctor_set(v___x_436_, 1, v_val_448_);
lean_ctor_set(v___x_436_, 0, v_size_x27_439_);
v___x_450_ = v___x_436_;
goto v_reusejp_449_;
}
else
{
lean_object* v_reuseFailAlloc_451_; 
v_reuseFailAlloc_451_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_451_, 0, v_size_x27_439_);
lean_ctor_set(v_reuseFailAlloc_451_, 1, v_val_448_);
v___x_450_ = v_reuseFailAlloc_451_;
goto v_reusejp_449_;
}
v_reusejp_449_:
{
return v___x_450_;
}
}
else
{
lean_object* v___x_453_; 
if (v_isShared_437_ == 0)
{
lean_ctor_set(v___x_436_, 1, v_buckets_x27_441_);
lean_ctor_set(v___x_436_, 0, v_size_x27_439_);
v___x_453_ = v___x_436_;
goto v_reusejp_452_;
}
else
{
lean_object* v_reuseFailAlloc_454_; 
v_reuseFailAlloc_454_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_454_, 0, v_size_x27_439_);
lean_ctor_set(v_reuseFailAlloc_454_, 1, v_buckets_x27_441_);
v___x_453_ = v_reuseFailAlloc_454_;
goto v_reusejp_452_;
}
v_reusejp_452_:
{
return v___x_453_;
}
}
}
}
else
{
lean_dec(v_b_417_);
lean_dec(v_a_416_);
return v_m_415_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__1_spec__3_spec__5___redArg(lean_object* v_a_458_, lean_object* v_b_459_, lean_object* v_x_460_){
_start:
{
if (lean_obj_tag(v_x_460_) == 0)
{
lean_dec(v_b_459_);
lean_dec(v_a_458_);
return v_x_460_;
}
else
{
lean_object* v_key_461_; lean_object* v_value_462_; lean_object* v_tail_463_; lean_object* v___x_465_; uint8_t v_isShared_466_; uint8_t v_isSharedCheck_475_; 
v_key_461_ = lean_ctor_get(v_x_460_, 0);
v_value_462_ = lean_ctor_get(v_x_460_, 1);
v_tail_463_ = lean_ctor_get(v_x_460_, 2);
v_isSharedCheck_475_ = !lean_is_exclusive(v_x_460_);
if (v_isSharedCheck_475_ == 0)
{
v___x_465_ = v_x_460_;
v_isShared_466_ = v_isSharedCheck_475_;
goto v_resetjp_464_;
}
else
{
lean_inc(v_tail_463_);
lean_inc(v_value_462_);
lean_inc(v_key_461_);
lean_dec(v_x_460_);
v___x_465_ = lean_box(0);
v_isShared_466_ = v_isSharedCheck_475_;
goto v_resetjp_464_;
}
v_resetjp_464_:
{
uint8_t v___x_467_; 
v___x_467_ = lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instDecidableEqStained_decEq(v_key_461_, v_a_458_);
if (v___x_467_ == 0)
{
lean_object* v___x_468_; lean_object* v___x_470_; 
v___x_468_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__1_spec__3_spec__5___redArg(v_a_458_, v_b_459_, v_tail_463_);
if (v_isShared_466_ == 0)
{
lean_ctor_set(v___x_465_, 2, v___x_468_);
v___x_470_ = v___x_465_;
goto v_reusejp_469_;
}
else
{
lean_object* v_reuseFailAlloc_471_; 
v_reuseFailAlloc_471_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_471_, 0, v_key_461_);
lean_ctor_set(v_reuseFailAlloc_471_, 1, v_value_462_);
lean_ctor_set(v_reuseFailAlloc_471_, 2, v___x_468_);
v___x_470_ = v_reuseFailAlloc_471_;
goto v_reusejp_469_;
}
v_reusejp_469_:
{
return v___x_470_;
}
}
else
{
lean_object* v___x_473_; 
lean_dec(v_value_462_);
lean_dec(v_key_461_);
if (v_isShared_466_ == 0)
{
lean_ctor_set(v___x_465_, 1, v_b_459_);
lean_ctor_set(v___x_465_, 0, v_a_458_);
v___x_473_ = v___x_465_;
goto v_reusejp_472_;
}
else
{
lean_object* v_reuseFailAlloc_474_; 
v_reuseFailAlloc_474_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_474_, 0, v_a_458_);
lean_ctor_set(v_reuseFailAlloc_474_, 1, v_b_459_);
lean_ctor_set(v_reuseFailAlloc_474_, 2, v_tail_463_);
v___x_473_ = v_reuseFailAlloc_474_;
goto v_reusejp_472_;
}
v_reusejp_472_:
{
return v___x_473_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__1_spec__3___redArg(lean_object* v_m_476_, lean_object* v_a_477_, lean_object* v_b_478_){
_start:
{
lean_object* v_size_479_; lean_object* v_buckets_480_; lean_object* v___x_482_; uint8_t v_isShared_483_; uint8_t v_isSharedCheck_523_; 
v_size_479_ = lean_ctor_get(v_m_476_, 0);
v_buckets_480_ = lean_ctor_get(v_m_476_, 1);
v_isSharedCheck_523_ = !lean_is_exclusive(v_m_476_);
if (v_isSharedCheck_523_ == 0)
{
v___x_482_ = v_m_476_;
v_isShared_483_ = v_isSharedCheck_523_;
goto v_resetjp_481_;
}
else
{
lean_inc(v_buckets_480_);
lean_inc(v_size_479_);
lean_dec(v_m_476_);
v___x_482_ = lean_box(0);
v_isShared_483_ = v_isSharedCheck_523_;
goto v_resetjp_481_;
}
v_resetjp_481_:
{
lean_object* v___x_484_; uint64_t v___x_485_; uint64_t v___x_486_; uint64_t v___x_487_; uint64_t v_fold_488_; uint64_t v___x_489_; uint64_t v___x_490_; uint64_t v___x_491_; size_t v___x_492_; size_t v___x_493_; size_t v___x_494_; size_t v___x_495_; size_t v___x_496_; lean_object* v_bkt_497_; uint8_t v___x_498_; 
v___x_484_ = lean_array_get_size(v_buckets_480_);
v___x_485_ = lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instHashableStained_hash(v_a_477_);
v___x_486_ = 32ULL;
v___x_487_ = lean_uint64_shift_right(v___x_485_, v___x_486_);
v_fold_488_ = lean_uint64_xor(v___x_485_, v___x_487_);
v___x_489_ = 16ULL;
v___x_490_ = lean_uint64_shift_right(v_fold_488_, v___x_489_);
v___x_491_ = lean_uint64_xor(v_fold_488_, v___x_490_);
v___x_492_ = lean_uint64_to_usize(v___x_491_);
v___x_493_ = lean_usize_of_nat(v___x_484_);
v___x_494_ = ((size_t)1ULL);
v___x_495_ = lean_usize_sub(v___x_493_, v___x_494_);
v___x_496_ = lean_usize_land(v___x_492_, v___x_495_);
v_bkt_497_ = lean_array_uget_borrowed(v_buckets_480_, v___x_496_);
v___x_498_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__0_spec__0___redArg(v_a_477_, v_bkt_497_);
if (v___x_498_ == 0)
{
lean_object* v___x_499_; lean_object* v_size_x27_500_; lean_object* v___x_501_; lean_object* v_buckets_x27_502_; lean_object* v___x_503_; lean_object* v___x_504_; lean_object* v___x_505_; lean_object* v___x_506_; lean_object* v___x_507_; uint8_t v___x_508_; 
v___x_499_ = lean_unsigned_to_nat(1u);
v_size_x27_500_ = lean_nat_add(v_size_479_, v___x_499_);
lean_dec(v_size_479_);
lean_inc(v_bkt_497_);
v___x_501_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_501_, 0, v_a_477_);
lean_ctor_set(v___x_501_, 1, v_b_478_);
lean_ctor_set(v___x_501_, 2, v_bkt_497_);
v_buckets_x27_502_ = lean_array_uset(v_buckets_480_, v___x_496_, v___x_501_);
v___x_503_ = lean_unsigned_to_nat(4u);
v___x_504_ = lean_nat_mul(v_size_x27_500_, v___x_503_);
v___x_505_ = lean_unsigned_to_nat(3u);
v___x_506_ = lean_nat_div(v___x_504_, v___x_505_);
lean_dec(v___x_504_);
v___x_507_ = lean_array_get_size(v_buckets_x27_502_);
v___x_508_ = lean_nat_dec_le(v___x_506_, v___x_507_);
lean_dec(v___x_506_);
if (v___x_508_ == 0)
{
lean_object* v_val_509_; lean_object* v___x_511_; 
v_val_509_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__0_spec__1___redArg(v_buckets_x27_502_);
if (v_isShared_483_ == 0)
{
lean_ctor_set(v___x_482_, 1, v_val_509_);
lean_ctor_set(v___x_482_, 0, v_size_x27_500_);
v___x_511_ = v___x_482_;
goto v_reusejp_510_;
}
else
{
lean_object* v_reuseFailAlloc_512_; 
v_reuseFailAlloc_512_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_512_, 0, v_size_x27_500_);
lean_ctor_set(v_reuseFailAlloc_512_, 1, v_val_509_);
v___x_511_ = v_reuseFailAlloc_512_;
goto v_reusejp_510_;
}
v_reusejp_510_:
{
return v___x_511_;
}
}
else
{
lean_object* v___x_514_; 
if (v_isShared_483_ == 0)
{
lean_ctor_set(v___x_482_, 1, v_buckets_x27_502_);
lean_ctor_set(v___x_482_, 0, v_size_x27_500_);
v___x_514_ = v___x_482_;
goto v_reusejp_513_;
}
else
{
lean_object* v_reuseFailAlloc_515_; 
v_reuseFailAlloc_515_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_515_, 0, v_size_x27_500_);
lean_ctor_set(v_reuseFailAlloc_515_, 1, v_buckets_x27_502_);
v___x_514_ = v_reuseFailAlloc_515_;
goto v_reusejp_513_;
}
v_reusejp_513_:
{
return v___x_514_;
}
}
}
else
{
lean_object* v___x_516_; lean_object* v_buckets_x27_517_; lean_object* v___x_518_; lean_object* v___x_519_; lean_object* v___x_521_; 
lean_inc(v_bkt_497_);
v___x_516_ = lean_box(0);
v_buckets_x27_517_ = lean_array_uset(v_buckets_480_, v___x_496_, v___x_516_);
v___x_518_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__1_spec__3_spec__5___redArg(v_a_477_, v_b_478_, v_bkt_497_);
v___x_519_ = lean_array_uset(v_buckets_x27_517_, v___x_496_, v___x_518_);
if (v_isShared_483_ == 0)
{
lean_ctor_set(v___x_482_, 1, v___x_519_);
v___x_521_ = v___x_482_;
goto v_reusejp_520_;
}
else
{
lean_object* v_reuseFailAlloc_522_; 
v_reuseFailAlloc_522_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_522_, 0, v_size_479_);
lean_ctor_set(v_reuseFailAlloc_522_, 1, v___x_519_);
v___x_521_ = v_reuseFailAlloc_522_;
goto v_reusejp_520_;
}
v_reusejp_520_:
{
return v___x_521_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__1_spec__4(lean_object* v_a_524_, lean_object* v_a_525_){
_start:
{
if (lean_obj_tag(v_a_524_) == 0)
{
lean_object* v___x_526_; 
v___x_526_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_526_, 0, v_a_525_);
return v___x_526_;
}
else
{
lean_object* v_key_527_; lean_object* v_value_528_; lean_object* v_tail_529_; lean_object* v_r_530_; 
v_key_527_ = lean_ctor_get(v_a_524_, 0);
lean_inc(v_key_527_);
v_value_528_ = lean_ctor_get(v_a_524_, 1);
lean_inc(v_value_528_);
v_tail_529_ = lean_ctor_get(v_a_524_, 2);
lean_inc(v_tail_529_);
lean_dec_ref_known(v_a_524_, 3);
v_r_530_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__1_spec__3___redArg(v_a_525_, v_key_527_, v_value_528_);
v_a_524_ = v_tail_529_;
v_a_525_ = v_r_530_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__1_spec__5(lean_object* v_as_532_, size_t v_sz_533_, size_t v_i_534_, lean_object* v_b_535_){
_start:
{
uint8_t v___x_536_; 
v___x_536_ = lean_usize_dec_lt(v_i_534_, v_sz_533_);
if (v___x_536_ == 0)
{
return v_b_535_;
}
else
{
lean_object* v_a_537_; lean_object* v___x_538_; 
v_a_537_ = lean_array_uget_borrowed(v_as_532_, v_i_534_);
lean_inc(v_a_537_);
v___x_538_ = lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__1_spec__4(v_a_537_, v_b_535_);
if (lean_obj_tag(v___x_538_) == 0)
{
lean_object* v_a_539_; 
v_a_539_ = lean_ctor_get(v___x_538_, 0);
lean_inc(v_a_539_);
lean_dec_ref_known(v___x_538_, 1);
return v_a_539_;
}
else
{
lean_object* v_a_540_; size_t v___x_541_; size_t v___x_542_; 
v_a_540_ = lean_ctor_get(v___x_538_, 0);
lean_inc(v_a_540_);
lean_dec_ref_known(v___x_538_, 1);
v___x_541_ = ((size_t)1ULL);
v___x_542_ = lean_usize_add(v_i_534_, v___x_541_);
v_i_534_ = v___x_542_;
v_b_535_ = v_a_540_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__1_spec__5___boxed(lean_object* v_as_544_, lean_object* v_sz_545_, lean_object* v_i_546_, lean_object* v_b_547_){
_start:
{
size_t v_sz_boxed_548_; size_t v_i_boxed_549_; lean_object* v_res_550_; 
v_sz_boxed_548_ = lean_unbox_usize(v_sz_545_);
lean_dec(v_sz_545_);
v_i_boxed_549_ = lean_unbox_usize(v_i_546_);
lean_dec(v_i_546_);
v_res_550_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__1_spec__5(v_as_544_, v_sz_boxed_548_, v_i_boxed_549_, v_b_547_);
lean_dec_ref(v_as_544_);
return v_res_550_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__1(lean_object* v_m_551_, lean_object* v_l_552_){
_start:
{
lean_object* v_buckets_553_; size_t v_sz_554_; size_t v___x_555_; lean_object* v___x_556_; 
v_buckets_553_ = lean_ctor_get(v_l_552_, 1);
v_sz_554_ = lean_array_size(v_buckets_553_);
v___x_555_ = ((size_t)0ULL);
v___x_556_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__1_spec__5(v_buckets_553_, v_sz_554_, v___x_555_, v_m_551_);
return v___x_556_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__1___boxed(lean_object* v_m_557_, lean_object* v_l_558_){
_start:
{
lean_object* v_res_559_; 
v_res_559_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__1(v_m_557_, v_l_558_);
lean_dec_ref(v_l_558_);
return v_res_559_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__2(lean_object* v_a_560_, lean_object* v_a_561_){
_start:
{
if (lean_obj_tag(v_a_560_) == 0)
{
lean_object* v___x_562_; 
v___x_562_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_562_, 0, v_a_561_);
return v___x_562_;
}
else
{
lean_object* v_key_563_; lean_object* v_value_564_; lean_object* v_tail_565_; lean_object* v_r_566_; 
v_key_563_ = lean_ctor_get(v_a_560_, 0);
lean_inc(v_key_563_);
v_value_564_ = lean_ctor_get(v_a_560_, 1);
lean_inc(v_value_564_);
v_tail_565_ = lean_ctor_get(v_a_560_, 2);
lean_inc(v_tail_565_);
lean_dec_ref_known(v_a_560_, 3);
v_r_566_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__0___redArg(v_a_561_, v_key_563_, v_value_564_);
v_a_560_ = v_tail_565_;
v_a_561_ = v_r_566_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__3(lean_object* v_as_568_, size_t v_sz_569_, size_t v_i_570_, lean_object* v_b_571_){
_start:
{
uint8_t v___x_572_; 
v___x_572_ = lean_usize_dec_lt(v_i_570_, v_sz_569_);
if (v___x_572_ == 0)
{
return v_b_571_;
}
else
{
lean_object* v_a_573_; lean_object* v___x_574_; 
v_a_573_ = lean_array_uget_borrowed(v_as_568_, v_i_570_);
lean_inc(v_a_573_);
v___x_574_ = lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__2(v_a_573_, v_b_571_);
if (lean_obj_tag(v___x_574_) == 0)
{
lean_object* v_a_575_; 
v_a_575_ = lean_ctor_get(v___x_574_, 0);
lean_inc(v_a_575_);
lean_dec_ref_known(v___x_574_, 1);
return v_a_575_;
}
else
{
lean_object* v_a_576_; size_t v___x_577_; size_t v___x_578_; 
v_a_576_ = lean_ctor_get(v___x_574_, 0);
lean_inc(v_a_576_);
lean_dec_ref_known(v___x_574_, 1);
v___x_577_ = ((size_t)1ULL);
v___x_578_ = lean_usize_add(v_i_570_, v___x_577_);
v_i_570_ = v___x_578_;
v_b_571_ = v_a_576_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__3___boxed(lean_object* v_as_580_, lean_object* v_sz_581_, lean_object* v_i_582_, lean_object* v_b_583_){
_start:
{
size_t v_sz_boxed_584_; size_t v_i_boxed_585_; lean_object* v_res_586_; 
v_sz_boxed_584_ = lean_unbox_usize(v_sz_581_);
lean_dec(v_sz_581_);
v_i_boxed_585_ = lean_unbox_usize(v_i_582_);
lean_dec(v_i_582_);
v_res_586_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__3(v_as_580_, v_sz_boxed_584_, v_i_boxed_585_, v_b_583_);
lean_dec_ref(v_as_580_);
return v_res_586_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__5(lean_object* v_as_587_, size_t v_i_588_, size_t v_stop_589_, lean_object* v_b_590_){
_start:
{
lean_object* v___y_592_; uint8_t v___x_596_; 
v___x_596_ = lean_usize_dec_eq(v_i_588_, v_stop_589_);
if (v___x_596_ == 0)
{
lean_object* v_size_597_; lean_object* v_buckets_598_; lean_object* v_r_599_; lean_object* v_size_600_; uint8_t v___x_601_; 
v_size_597_ = lean_ctor_get(v_b_590_, 0);
v_buckets_598_ = lean_ctor_get(v_b_590_, 1);
v_r_599_ = lean_array_uget_borrowed(v_as_587_, v_i_588_);
v_size_600_ = lean_ctor_get(v_r_599_, 0);
v___x_601_ = lean_nat_dec_le(v_size_597_, v_size_600_);
if (v___x_601_ == 0)
{
lean_object* v___x_602_; 
v___x_602_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__1(v_b_590_, v_r_599_);
v___y_592_ = v___x_602_;
goto v___jp_591_;
}
else
{
size_t v_sz_603_; size_t v___x_604_; lean_object* v___x_605_; 
lean_inc_ref(v_buckets_598_);
lean_dec_ref(v_b_590_);
v_sz_603_ = lean_array_size(v_buckets_598_);
v___x_604_ = ((size_t)0ULL);
lean_inc(v_r_599_);
v___x_605_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__3(v_buckets_598_, v_sz_603_, v___x_604_, v_r_599_);
lean_dec_ref(v_buckets_598_);
v___y_592_ = v___x_605_;
goto v___jp_591_;
}
}
else
{
return v_b_590_;
}
v___jp_591_:
{
size_t v___x_593_; size_t v___x_594_; 
v___x_593_ = ((size_t)1ULL);
v___x_594_ = lean_usize_add(v_i_588_, v___x_593_);
v_i_588_ = v___x_594_;
v_b_590_ = v___y_592_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__5___boxed(lean_object* v_as_606_, lean_object* v_i_607_, lean_object* v_stop_608_, lean_object* v_b_609_){
_start:
{
size_t v_i_boxed_610_; size_t v_stop_boxed_611_; lean_object* v_res_612_; 
v_i_boxed_610_ = lean_unbox_usize(v_i_607_);
lean_dec(v_i_607_);
v_stop_boxed_611_ = lean_unbox_usize(v_stop_608_);
lean_dec(v_stop_608_);
v_res_612_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__5(v_as_606_, v_i_boxed_610_, v_stop_boxed_611_, v_b_609_);
lean_dec_ref(v_as_606_);
return v_res_612_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__0(void){
_start:
{
lean_object* v___x_613_; lean_object* v___x_614_; lean_object* v___x_615_; 
v___x_613_ = lean_box(0);
v___x_614_ = lean_unsigned_to_nat(16u);
v___x_615_ = lean_mk_array(v___x_614_, v___x_613_);
return v___x_615_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__1(void){
_start:
{
lean_object* v___x_616_; lean_object* v___x_617_; lean_object* v___x_618_; 
v___x_616_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__0, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__0_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__0);
v___x_617_ = lean_unsigned_to_nat(0u);
v___x_618_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_618_, 0, v___x_617_);
lean_ctor_set(v___x_618_, 1, v___x_616_);
return v___x_618_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__2(void){
_start:
{
lean_object* v___x_619_; lean_object* v___x_620_; lean_object* v___x_621_; lean_object* v___x_622_; 
v___x_619_ = lean_box(0);
v___x_620_ = lean_box(1);
v___x_621_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__1, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__1);
v___x_622_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__0___redArg(v___x_621_, v___x_620_, v___x_619_);
return v___x_622_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__4(void){
_start:
{
lean_object* v___x_624_; lean_object* v___x_625_; lean_object* v___x_626_; lean_object* v___x_627_; 
v___x_624_ = lean_box(0);
v___x_625_ = lean_box(2);
v___x_626_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__1, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__1);
v___x_627_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__0___redArg(v___x_626_, v___x_625_, v___x_624_);
return v___x_627_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained(lean_object* v_x_628_){
_start:
{
switch(lean_obj_tag(v_x_628_))
{
case 1:
{
lean_object* v_args_631_; lean_object* v___x_632_; lean_object* v___x_633_; size_t v_sz_634_; size_t v___x_635_; lean_object* v___x_636_; lean_object* v___x_637_; uint8_t v___x_638_; 
v_args_631_ = lean_ctor_get(v_x_628_, 2);
lean_inc_ref(v_args_631_);
lean_dec_ref_known(v_x_628_, 3);
v___x_632_ = lean_unsigned_to_nat(0u);
v___x_633_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__1, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__1);
v_sz_634_ = lean_array_size(v_args_631_);
v___x_635_ = ((size_t)0ULL);
v___x_636_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__4(v_sz_634_, v___x_635_, v_args_631_);
v___x_637_ = lean_array_get_size(v___x_636_);
v___x_638_ = lean_nat_dec_lt(v___x_632_, v___x_637_);
if (v___x_638_ == 0)
{
lean_dec_ref(v___x_636_);
return v___x_633_;
}
else
{
uint8_t v___x_639_; 
v___x_639_ = lean_nat_dec_le(v___x_637_, v___x_637_);
if (v___x_639_ == 0)
{
if (v___x_638_ == 0)
{
lean_dec_ref(v___x_636_);
return v___x_633_;
}
else
{
size_t v___x_640_; lean_object* v___x_641_; 
v___x_640_ = lean_usize_of_nat(v___x_637_);
v___x_641_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__5(v___x_636_, v___x_635_, v___x_640_, v___x_633_);
lean_dec_ref(v___x_636_);
return v___x_641_;
}
}
else
{
size_t v___x_642_; lean_object* v___x_643_; 
v___x_642_ = lean_usize_of_nat(v___x_637_);
v___x_643_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__5(v___x_636_, v___x_635_, v___x_642_, v___x_633_);
lean_dec_ref(v___x_636_);
return v___x_643_;
}
}
}
case 3:
{
lean_object* v_val_644_; lean_object* v___x_645_; lean_object* v___x_646_; lean_object* v___x_647_; lean_object* v___x_648_; 
v_val_644_ = lean_ctor_get(v_x_628_, 2);
lean_inc(v_val_644_);
lean_dec_ref_known(v_x_628_, 4);
v___x_645_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_645_, 0, v_val_644_);
v___x_646_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__1, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__1);
v___x_647_ = lean_box(0);
v___x_648_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__0___redArg(v___x_646_, v___x_645_, v___x_647_);
return v___x_648_;
}
case 2:
{
lean_object* v_val_649_; lean_object* v___x_650_; uint8_t v___x_651_; 
v_val_649_ = lean_ctor_get(v_x_628_, 1);
lean_inc_ref(v_val_649_);
lean_dec_ref_known(v_x_628_, 2);
v___x_650_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instToStringStained___lam__0___closed__1));
v___x_651_ = lean_string_dec_eq(v_val_649_, v___x_650_);
if (v___x_651_ == 0)
{
lean_object* v___x_652_; uint8_t v___x_653_; 
v___x_652_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instToStringStained___lam__0___closed__0));
v___x_653_ = lean_string_dec_eq(v_val_649_, v___x_652_);
if (v___x_653_ == 0)
{
lean_object* v___x_654_; uint8_t v___x_655_; 
v___x_654_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__3));
v___x_655_ = lean_string_dec_eq(v_val_649_, v___x_654_);
lean_dec_ref(v_val_649_);
if (v___x_655_ == 0)
{
lean_object* v___x_656_; 
v___x_656_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__1, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__1);
return v___x_656_;
}
else
{
goto v___jp_629_;
}
}
else
{
lean_dec_ref(v_val_649_);
goto v___jp_629_;
}
}
else
{
lean_object* v___x_657_; 
lean_dec_ref(v_val_649_);
v___x_657_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__4, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__4_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__4);
return v___x_657_;
}
}
default: 
{
lean_object* v___x_658_; 
lean_dec(v_x_628_);
v___x_658_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__1, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__1);
return v___x_658_;
}
}
v___jp_629_:
{
lean_object* v___x_630_; 
v___x_630_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__2, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__2);
return v___x_630_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__4(size_t v_sz_659_, size_t v_i_660_, lean_object* v_bs_661_){
_start:
{
uint8_t v___x_662_; 
v___x_662_ = lean_usize_dec_lt(v_i_660_, v_sz_659_);
if (v___x_662_ == 0)
{
return v_bs_661_;
}
else
{
lean_object* v_v_663_; lean_object* v___x_664_; lean_object* v_bs_x27_665_; lean_object* v___x_666_; size_t v___x_667_; size_t v___x_668_; lean_object* v___x_669_; 
v_v_663_ = lean_array_uget(v_bs_661_, v_i_660_);
v___x_664_ = lean_unsigned_to_nat(0u);
v_bs_x27_665_ = lean_array_uset(v_bs_661_, v_i_660_, v___x_664_);
v___x_666_ = lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained(v_v_663_);
v___x_667_ = ((size_t)1ULL);
v___x_668_ = lean_usize_add(v_i_660_, v___x_667_);
v___x_669_ = lean_array_uset(v_bs_x27_665_, v_i_660_, v___x_666_);
v_i_660_ = v___x_668_;
v_bs_661_ = v___x_669_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__4___boxed(lean_object* v_sz_671_, lean_object* v_i_672_, lean_object* v_bs_673_){
_start:
{
size_t v_sz_boxed_674_; size_t v_i_boxed_675_; lean_object* v_res_676_; 
v_sz_boxed_674_ = lean_unbox_usize(v_sz_671_);
lean_dec(v_sz_671_);
v_i_boxed_675_ = lean_unbox_usize(v_i_672_);
lean_dec(v_i_672_);
v_res_676_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__4(v_sz_boxed_674_, v_i_boxed_675_, v_bs_673_);
return v_res_676_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__0(lean_object* v_00_u03b2_677_, lean_object* v_m_678_, lean_object* v_a_679_, lean_object* v_b_680_){
_start:
{
lean_object* v___x_681_; 
v___x_681_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__0___redArg(v_m_678_, v_a_679_, v_b_680_);
return v___x_681_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__0_spec__0(lean_object* v_00_u03b2_682_, lean_object* v_a_683_, lean_object* v_x_684_){
_start:
{
uint8_t v___x_685_; 
v___x_685_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__0_spec__0___redArg(v_a_683_, v_x_684_);
return v___x_685_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__0_spec__0___boxed(lean_object* v_00_u03b2_686_, lean_object* v_a_687_, lean_object* v_x_688_){
_start:
{
uint8_t v_res_689_; lean_object* v_r_690_; 
v_res_689_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__0_spec__0(v_00_u03b2_686_, v_a_687_, v_x_688_);
lean_dec(v_x_688_);
lean_dec(v_a_687_);
v_r_690_ = lean_box(v_res_689_);
return v_r_690_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__0_spec__1(lean_object* v_00_u03b2_691_, lean_object* v_data_692_){
_start:
{
lean_object* v___x_693_; 
v___x_693_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__0_spec__1___redArg(v_data_692_);
return v___x_693_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__1_spec__3(lean_object* v_00_u03b2_694_, lean_object* v_m_695_, lean_object* v_a_696_, lean_object* v_b_697_){
_start:
{
lean_object* v___x_698_; 
v___x_698_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__1_spec__3___redArg(v_m_695_, v_a_696_, v_b_697_);
return v___x_698_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__0_spec__1_spec__2(lean_object* v_00_u03b2_699_, lean_object* v_i_700_, lean_object* v_source_701_, lean_object* v_target_702_){
_start:
{
lean_object* v___x_703_; 
v___x_703_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__0_spec__1_spec__2___redArg(v_i_700_, v_source_701_, v_target_702_);
return v___x_703_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__1_spec__3_spec__5(lean_object* v_00_u03b2_704_, lean_object* v_a_705_, lean_object* v_b_706_, lean_object* v_x_707_){
_start:
{
lean_object* v___x_708_; 
v___x_708_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__1_spec__3_spec__5___redArg(v_a_705_, v_b_706_, v_x_707_);
return v___x_708_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__0_spec__1_spec__2_spec__8(lean_object* v_00_u03b2_709_, lean_object* v_x_710_, lean_object* v_x_711_){
_start:
{
lean_object* v___x_712_; 
v___x_712_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__0_spec__1_spec__2_spec__8___redArg(v_x_710_, v_x_711_);
return v___x_712_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_getStained_spec__1(lean_object* v_as_713_, size_t v_i_714_, size_t v_stop_715_, lean_object* v_b_716_){
_start:
{
lean_object* v___y_718_; uint8_t v___x_722_; 
v___x_722_ = lean_usize_dec_eq(v_i_714_, v_stop_715_);
if (v___x_722_ == 0)
{
lean_object* v_size_723_; lean_object* v_buckets_724_; lean_object* v_r_725_; lean_object* v_size_726_; uint8_t v___x_727_; 
v_size_723_ = lean_ctor_get(v_b_716_, 0);
v_buckets_724_ = lean_ctor_get(v_b_716_, 1);
v_r_725_ = lean_array_uget_borrowed(v_as_713_, v_i_714_);
v_size_726_ = lean_ctor_get(v_r_725_, 0);
v___x_727_ = lean_nat_dec_le(v_size_723_, v_size_726_);
if (v___x_727_ == 0)
{
lean_object* v___x_728_; 
v___x_728_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__1(v_b_716_, v_r_725_);
v___y_718_ = v___x_728_;
goto v___jp_717_;
}
else
{
size_t v_sz_729_; size_t v___x_730_; lean_object* v___x_731_; 
lean_inc_ref(v_buckets_724_);
lean_dec_ref(v_b_716_);
v_sz_729_ = lean_array_size(v_buckets_724_);
v___x_730_ = ((size_t)0ULL);
lean_inc(v_r_725_);
v___x_731_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__3(v_buckets_724_, v_sz_729_, v___x_730_, v_r_725_);
lean_dec_ref(v_buckets_724_);
v___y_718_ = v___x_731_;
goto v___jp_717_;
}
}
else
{
return v_b_716_;
}
v___jp_717_:
{
size_t v___x_719_; size_t v___x_720_; 
v___x_719_ = ((size_t)1ULL);
v___x_720_ = lean_usize_add(v_i_714_, v___x_719_);
v_i_714_ = v___x_720_;
v_b_716_ = v___y_718_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_getStained_spec__1___boxed(lean_object* v_as_732_, lean_object* v_i_733_, lean_object* v_stop_734_, lean_object* v_b_735_){
_start:
{
size_t v_i_boxed_736_; size_t v_stop_boxed_737_; lean_object* v_res_738_; 
v_i_boxed_736_ = lean_unbox_usize(v_i_733_);
lean_dec(v_i_733_);
v_stop_boxed_737_ = lean_unbox_usize(v_stop_734_);
lean_dec(v_stop_734_);
v_res_738_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_getStained_spec__1(v_as_732_, v_i_boxed_736_, v_stop_boxed_737_, v_b_735_);
lean_dec_ref(v_as_732_);
return v_res_738_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_getStained(lean_object* v_stx_740_, lean_object* v_all_x3f_741_){
_start:
{
if (lean_obj_tag(v_stx_740_) == 1)
{
lean_object* v_info_742_; lean_object* v_kind_743_; lean_object* v_args_744_; lean_object* v___x_746_; uint8_t v_isShared_747_; uint8_t v_isSharedCheck_799_; 
v_info_742_ = lean_ctor_get(v_stx_740_, 0);
v_kind_743_ = lean_ctor_get(v_stx_740_, 1);
v_args_744_ = lean_ctor_get(v_stx_740_, 2);
v_isSharedCheck_799_ = !lean_is_exclusive(v_stx_740_);
if (v_isSharedCheck_799_ == 0)
{
v___x_746_ = v_stx_740_;
v_isShared_747_ = v_isSharedCheck_799_;
goto v_resetjp_745_;
}
else
{
lean_inc(v_args_744_);
lean_inc(v_kind_743_);
lean_inc(v_info_742_);
lean_dec(v_stx_740_);
v___x_746_ = lean_box(0);
v_isShared_747_ = v_isSharedCheck_799_;
goto v_resetjp_745_;
}
v_resetjp_745_:
{
if (lean_obj_tag(v_kind_743_) == 1)
{
lean_object* v_pre_761_; 
v_pre_761_ = lean_ctor_get(v_kind_743_, 0);
lean_inc(v_pre_761_);
if (lean_obj_tag(v_pre_761_) == 1)
{
lean_object* v_pre_762_; 
v_pre_762_ = lean_ctor_get(v_pre_761_, 0);
lean_inc(v_pre_762_);
if (lean_obj_tag(v_pre_762_) == 1)
{
lean_object* v_pre_763_; 
v_pre_763_ = lean_ctor_get(v_pre_762_, 0);
lean_inc(v_pre_763_);
if (lean_obj_tag(v_pre_763_) == 1)
{
lean_object* v_pre_764_; 
v_pre_764_ = lean_ctor_get(v_pre_763_, 0);
lean_inc(v_pre_764_);
if (lean_obj_tag(v_pre_764_) == 0)
{
lean_object* v_str_765_; lean_object* v_str_766_; lean_object* v_str_767_; lean_object* v_str_768_; lean_object* v___x_769_; uint8_t v___x_770_; 
v_str_765_ = lean_ctor_get(v_kind_743_, 1);
lean_inc_ref(v_str_765_);
lean_dec_ref_known(v_kind_743_, 2);
v_str_766_ = lean_ctor_get(v_pre_761_, 1);
lean_inc_ref(v_str_766_);
lean_dec_ref_known(v_pre_761_, 2);
v_str_767_ = lean_ctor_get(v_pre_762_, 1);
lean_inc_ref(v_str_767_);
lean_dec_ref_known(v_pre_762_, 2);
v_str_768_ = lean_ctor_get(v_pre_763_, 1);
lean_inc_ref(v_str_768_);
lean_dec_ref_known(v_pre_763_, 2);
v___x_769_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__0));
v___x_770_ = lean_string_dec_eq(v_str_768_, v___x_769_);
lean_dec_ref(v_str_768_);
if (v___x_770_ == 0)
{
lean_dec_ref(v_str_767_);
lean_dec_ref(v_str_766_);
lean_dec_ref(v_str_765_);
lean_del_object(v___x_746_);
lean_dec(v_info_742_);
goto v___jp_748_;
}
else
{
lean_object* v___x_771_; uint8_t v___x_772_; 
v___x_771_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__1));
v___x_772_ = lean_string_dec_eq(v_str_767_, v___x_771_);
lean_dec_ref(v_str_767_);
if (v___x_772_ == 0)
{
lean_dec_ref(v_str_766_);
lean_dec_ref(v_str_765_);
lean_del_object(v___x_746_);
lean_dec(v_info_742_);
goto v___jp_748_;
}
else
{
lean_object* v___x_773_; uint8_t v___x_774_; 
v___x_773_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2));
v___x_774_ = lean_string_dec_eq(v_str_766_, v___x_773_);
lean_dec_ref(v_str_766_);
if (v___x_774_ == 0)
{
lean_dec_ref(v_str_765_);
lean_del_object(v___x_746_);
lean_dec(v_info_742_);
goto v___jp_748_;
}
else
{
lean_object* v___x_775_; uint8_t v___x_776_; 
v___x_775_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_getStained___closed__0));
v___x_776_ = lean_string_dec_eq(v_str_765_, v___x_775_);
lean_dec_ref(v_str_765_);
if (v___x_776_ == 0)
{
lean_del_object(v___x_746_);
lean_dec(v_info_742_);
goto v___jp_748_;
}
else
{
lean_object* v___x_777_; lean_object* v___x_778_; lean_object* v___x_779_; lean_object* v___x_780_; lean_object* v___x_782_; 
v___x_777_ = l_Lean_Name_str___override(v_pre_764_, v___x_769_);
v___x_778_ = l_Lean_Name_str___override(v___x_777_, v___x_771_);
v___x_779_ = l_Lean_Name_str___override(v___x_778_, v___x_773_);
v___x_780_ = l_Lean_Name_str___override(v___x_779_, v___x_775_);
lean_inc_ref(v_args_744_);
if (v_isShared_747_ == 0)
{
lean_ctor_set(v___x_746_, 1, v___x_780_);
v___x_782_ = v___x_746_;
goto v_reusejp_781_;
}
else
{
lean_object* v_reuseFailAlloc_798_; 
v_reuseFailAlloc_798_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_798_, 0, v_info_742_);
lean_ctor_set(v_reuseFailAlloc_798_, 1, v___x_780_);
lean_ctor_set(v_reuseFailAlloc_798_, 2, v_args_744_);
v___x_782_ = v_reuseFailAlloc_798_;
goto v_reusejp_781_;
}
v_reusejp_781_:
{
lean_object* v___x_783_; uint8_t v___x_784_; 
v___x_783_ = lean_apply_1(v_all_x3f_741_, v___x_782_);
v___x_784_ = lean_unbox(v___x_783_);
if (v___x_784_ == 0)
{
lean_object* v___x_785_; lean_object* v___x_786_; size_t v_sz_787_; size_t v___x_788_; lean_object* v___x_789_; lean_object* v___x_790_; uint8_t v___x_791_; 
v___x_785_ = lean_unsigned_to_nat(0u);
v___x_786_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__1, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__1);
v_sz_787_ = lean_array_size(v_args_744_);
v___x_788_ = ((size_t)0ULL);
v___x_789_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__4(v_sz_787_, v___x_788_, v_args_744_);
v___x_790_ = lean_array_get_size(v___x_789_);
v___x_791_ = lean_nat_dec_lt(v___x_785_, v___x_790_);
if (v___x_791_ == 0)
{
lean_dec_ref(v___x_789_);
return v___x_786_;
}
else
{
uint8_t v___x_792_; 
v___x_792_ = lean_nat_dec_le(v___x_790_, v___x_790_);
if (v___x_792_ == 0)
{
if (v___x_791_ == 0)
{
lean_dec_ref(v___x_789_);
return v___x_786_;
}
else
{
size_t v___x_793_; lean_object* v___x_794_; 
v___x_793_ = lean_usize_of_nat(v___x_790_);
v___x_794_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_getStained_spec__1(v___x_789_, v___x_788_, v___x_793_, v___x_786_);
lean_dec_ref(v___x_789_);
return v___x_794_;
}
}
else
{
size_t v___x_795_; lean_object* v___x_796_; 
v___x_795_ = lean_usize_of_nat(v___x_790_);
v___x_796_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_getStained_spec__1(v___x_789_, v___x_788_, v___x_795_, v___x_786_);
lean_dec_ref(v___x_789_);
return v___x_796_;
}
}
}
else
{
lean_object* v___x_797_; 
lean_dec_ref(v_args_744_);
v___x_797_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__1, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__1);
return v___x_797_;
}
}
}
}
}
}
}
else
{
lean_dec_ref_known(v_pre_763_, 2);
lean_dec(v_pre_764_);
lean_dec_ref_known(v_pre_762_, 2);
lean_dec_ref_known(v_pre_761_, 2);
lean_dec_ref_known(v_kind_743_, 2);
lean_del_object(v___x_746_);
lean_dec(v_info_742_);
goto v___jp_748_;
}
}
else
{
lean_dec(v_pre_763_);
lean_dec_ref_known(v_pre_762_, 2);
lean_dec_ref_known(v_pre_761_, 2);
lean_dec_ref_known(v_kind_743_, 2);
lean_del_object(v___x_746_);
lean_dec(v_info_742_);
goto v___jp_748_;
}
}
else
{
lean_dec_ref_known(v_pre_761_, 2);
lean_dec(v_pre_762_);
lean_dec_ref_known(v_kind_743_, 2);
lean_del_object(v___x_746_);
lean_dec(v_info_742_);
goto v___jp_748_;
}
}
else
{
lean_dec_ref_known(v_kind_743_, 2);
lean_dec(v_pre_761_);
lean_del_object(v___x_746_);
lean_dec(v_info_742_);
goto v___jp_748_;
}
}
else
{
lean_del_object(v___x_746_);
lean_dec(v_kind_743_);
lean_dec(v_info_742_);
goto v___jp_748_;
}
v___jp_748_:
{
lean_object* v___x_749_; lean_object* v___x_750_; size_t v_sz_751_; size_t v___x_752_; lean_object* v___x_753_; lean_object* v___x_754_; uint8_t v___x_755_; 
v___x_749_ = lean_unsigned_to_nat(0u);
v___x_750_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__1, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__1);
v_sz_751_ = lean_array_size(v_args_744_);
v___x_752_ = ((size_t)0ULL);
v___x_753_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_getStained_spec__0(v_all_x3f_741_, v_sz_751_, v___x_752_, v_args_744_);
v___x_754_ = lean_array_get_size(v___x_753_);
v___x_755_ = lean_nat_dec_lt(v___x_749_, v___x_754_);
if (v___x_755_ == 0)
{
lean_dec_ref(v___x_753_);
return v___x_750_;
}
else
{
uint8_t v___x_756_; 
v___x_756_ = lean_nat_dec_le(v___x_754_, v___x_754_);
if (v___x_756_ == 0)
{
if (v___x_755_ == 0)
{
lean_dec_ref(v___x_753_);
return v___x_750_;
}
else
{
size_t v___x_757_; lean_object* v___x_758_; 
v___x_757_ = lean_usize_of_nat(v___x_754_);
v___x_758_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_getStained_spec__1(v___x_753_, v___x_752_, v___x_757_, v___x_750_);
lean_dec_ref(v___x_753_);
return v___x_758_;
}
}
else
{
size_t v___x_759_; lean_object* v___x_760_; 
v___x_759_ = lean_usize_of_nat(v___x_754_);
v___x_760_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_getStained_spec__1(v___x_753_, v___x_752_, v___x_759_, v___x_750_);
lean_dec_ref(v___x_753_);
return v___x_760_;
}
}
}
}
}
else
{
lean_object* v___x_800_; 
lean_dec_ref(v_all_x3f_741_);
lean_dec(v_stx_740_);
v___x_800_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__1, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__1);
return v___x_800_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_getStained_spec__0(lean_object* v_all_x3f_801_, size_t v_sz_802_, size_t v_i_803_, lean_object* v_bs_804_){
_start:
{
uint8_t v___x_805_; 
v___x_805_ = lean_usize_dec_lt(v_i_803_, v_sz_802_);
if (v___x_805_ == 0)
{
lean_dec_ref(v_all_x3f_801_);
return v_bs_804_;
}
else
{
lean_object* v_v_806_; lean_object* v___x_807_; lean_object* v_bs_x27_808_; lean_object* v___x_809_; size_t v___x_810_; size_t v___x_811_; lean_object* v___x_812_; 
v_v_806_ = lean_array_uget(v_bs_804_, v_i_803_);
v___x_807_ = lean_unsigned_to_nat(0u);
v_bs_x27_808_ = lean_array_uset(v_bs_804_, v_i_803_, v___x_807_);
lean_inc_ref(v_all_x3f_801_);
v___x_809_ = lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_getStained(v_v_806_, v_all_x3f_801_);
v___x_810_ = ((size_t)1ULL);
v___x_811_ = lean_usize_add(v_i_803_, v___x_810_);
v___x_812_ = lean_array_uset(v_bs_x27_808_, v_i_803_, v___x_809_);
v_i_803_ = v___x_811_;
v_bs_804_ = v___x_812_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_getStained_spec__0___boxed(lean_object* v_all_x3f_814_, lean_object* v_sz_815_, lean_object* v_i_816_, lean_object* v_bs_817_){
_start:
{
size_t v_sz_boxed_818_; size_t v_i_boxed_819_; lean_object* v_res_820_; 
v_sz_boxed_818_ = lean_unbox_usize(v_sz_815_);
lean_dec(v_sz_815_);
v_i_boxed_819_ = lean_unbox_usize(v_i_816_);
lean_dec(v_i_816_);
v_res_820_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_getStained_spec__0(v_all_x3f_814_, v_sz_boxed_818_, v_i_boxed_819_, v_bs_817_);
return v_res_820_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_getStained_x21(lean_object* v_stx_821_, lean_object* v_all_x3f_822_){
_start:
{
lean_object* v_out_823_; lean_object* v_size_824_; lean_object* v___x_825_; uint8_t v___x_826_; 
v_out_823_ = lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_getStained(v_stx_821_, v_all_x3f_822_);
v_size_824_ = lean_ctor_get(v_out_823_, 0);
lean_inc(v_size_824_);
v___x_825_ = lean_unsigned_to_nat(0u);
v___x_826_ = lean_nat_dec_eq(v_size_824_, v___x_825_);
lean_dec(v_size_824_);
if (v___x_826_ == 0)
{
return v_out_823_;
}
else
{
lean_object* v___x_827_; 
lean_dec_ref(v_out_823_);
v___x_827_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__2, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained___closed__2);
return v___x_827_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_Stained_toFMVarId_spec__0(lean_object* v_mv_828_, size_t v_sz_829_, size_t v_i_830_, lean_object* v_bs_831_){
_start:
{
uint8_t v___x_832_; 
v___x_832_ = lean_usize_dec_lt(v_i_830_, v_sz_829_);
if (v___x_832_ == 0)
{
lean_dec(v_mv_828_);
return v_bs_831_;
}
else
{
lean_object* v_v_833_; lean_object* v___x_834_; lean_object* v_bs_x27_835_; lean_object* v___x_836_; size_t v___x_837_; size_t v___x_838_; lean_object* v___x_839_; 
v_v_833_ = lean_array_uget(v_bs_831_, v_i_830_);
v___x_834_ = lean_unsigned_to_nat(0u);
v_bs_x27_835_ = lean_array_uset(v_bs_831_, v_i_830_, v___x_834_);
lean_inc(v_mv_828_);
v___x_836_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_836_, 0, v_v_833_);
lean_ctor_set(v___x_836_, 1, v_mv_828_);
v___x_837_ = ((size_t)1ULL);
v___x_838_ = lean_usize_add(v_i_830_, v___x_837_);
v___x_839_ = lean_array_uset(v_bs_x27_835_, v_i_830_, v___x_836_);
v_i_830_ = v___x_838_;
v_bs_831_ = v___x_839_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_Stained_toFMVarId_spec__0___boxed(lean_object* v_mv_841_, lean_object* v_sz_842_, lean_object* v_i_843_, lean_object* v_bs_844_){
_start:
{
size_t v_sz_boxed_845_; size_t v_i_boxed_846_; lean_object* v_res_847_; 
v_sz_boxed_845_ = lean_unbox_usize(v_sz_842_);
lean_dec(v_sz_842_);
v_i_boxed_846_ = lean_unbox_usize(v_i_843_);
lean_dec(v_i_843_);
v_res_847_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_Stained_toFMVarId_spec__0(v_mv_841_, v_sz_boxed_845_, v_i_boxed_846_, v_bs_844_);
return v_res_847_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_Stained_toFMVarId(lean_object* v_mv_850_, lean_object* v_lctx_851_, lean_object* v_x_852_){
_start:
{
switch(lean_obj_tag(v_x_852_))
{
case 0:
{
lean_object* v_a_853_; lean_object* v___x_854_; 
v_a_853_ = lean_ctor_get(v_x_852_, 0);
v___x_854_ = l_Lean_LocalContext_findFromUserName_x3f(v_lctx_851_, v_a_853_);
if (lean_obj_tag(v___x_854_) == 0)
{
lean_object* v___x_855_; 
lean_dec(v_mv_850_);
v___x_855_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_Stained_toFMVarId___closed__0));
return v___x_855_;
}
else
{
lean_object* v_val_856_; lean_object* v___x_857_; lean_object* v___x_858_; lean_object* v___x_859_; lean_object* v___x_860_; lean_object* v___x_861_; 
v_val_856_ = lean_ctor_get(v___x_854_, 0);
lean_inc(v_val_856_);
lean_dec_ref_known(v___x_854_, 1);
v___x_857_ = l_Lean_LocalDecl_fvarId(v_val_856_);
lean_dec(v_val_856_);
v___x_858_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_858_, 0, v___x_857_);
lean_ctor_set(v___x_858_, 1, v_mv_850_);
v___x_859_ = lean_unsigned_to_nat(1u);
v___x_860_ = lean_mk_empty_array_with_capacity(v___x_859_);
v___x_861_ = lean_array_push(v___x_860_, v___x_858_);
return v___x_861_;
}
}
case 1:
{
lean_object* v___x_862_; lean_object* v___x_863_; lean_object* v___x_864_; lean_object* v___x_865_; lean_object* v___x_866_; 
v___x_862_ = lean_box(0);
v___x_863_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_863_, 0, v___x_862_);
lean_ctor_set(v___x_863_, 1, v_mv_850_);
v___x_864_ = lean_unsigned_to_nat(1u);
v___x_865_ = lean_mk_empty_array_with_capacity(v___x_864_);
v___x_866_ = lean_array_push(v___x_865_, v___x_863_);
return v___x_866_;
}
default: 
{
lean_object* v___x_867_; lean_object* v___x_868_; lean_object* v___x_869_; size_t v_sz_870_; size_t v___x_871_; lean_object* v___x_872_; 
v___x_867_ = l_Lean_LocalContext_getFVarIds(v_lctx_851_);
v___x_868_ = lean_box(0);
v___x_869_ = lean_array_push(v___x_867_, v___x_868_);
v_sz_870_ = lean_array_size(v___x_869_);
v___x_871_ = ((size_t)0ULL);
v___x_872_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_Stained_toFMVarId_spec__0(v_mv_850_, v_sz_870_, v___x_871_, v___x_869_);
return v___x_872_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_Stained_toFMVarId___boxed(lean_object* v_mv_873_, lean_object* v_lctx_874_, lean_object* v_x_875_){
_start:
{
lean_object* v_res_876_; 
v_res_876_ = lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_Stained_toFMVarId(v_mv_873_, v_lctx_874_, v_x_875_);
lean_dec(v_x_875_);
lean_dec_ref(v_lctx_874_);
return v_res_876_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0_spec__0___redArg(lean_object* v_a_877_, lean_object* v_x_878_){
_start:
{
if (lean_obj_tag(v_x_878_) == 0)
{
uint8_t v___x_879_; 
v___x_879_ = 0;
return v___x_879_;
}
else
{
lean_object* v_key_880_; lean_object* v_tail_881_; uint8_t v___x_882_; 
v_key_880_ = lean_ctor_get(v_x_878_, 0);
v_tail_881_ = lean_ctor_get(v_x_878_, 2);
v___x_882_ = lean_name_eq(v_key_880_, v_a_877_);
if (v___x_882_ == 0)
{
v_x_878_ = v_tail_881_;
goto _start;
}
else
{
return v___x_882_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0_spec__0___redArg___boxed(lean_object* v_a_884_, lean_object* v_x_885_){
_start:
{
uint8_t v_res_886_; lean_object* v_r_887_; 
v_res_886_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0_spec__0___redArg(v_a_884_, v_x_885_);
lean_dec(v_x_885_);
lean_dec(v_a_884_);
v_r_887_ = lean_box(v_res_886_);
return v_r_887_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0_spec__1_spec__2_spec__3___redArg(lean_object* v_x_888_, lean_object* v_x_889_){
_start:
{
if (lean_obj_tag(v_x_889_) == 0)
{
return v_x_888_;
}
else
{
lean_object* v_key_890_; lean_object* v_value_891_; lean_object* v_tail_892_; lean_object* v___x_894_; uint8_t v_isShared_895_; uint8_t v_isSharedCheck_918_; 
v_key_890_ = lean_ctor_get(v_x_889_, 0);
v_value_891_ = lean_ctor_get(v_x_889_, 1);
v_tail_892_ = lean_ctor_get(v_x_889_, 2);
v_isSharedCheck_918_ = !lean_is_exclusive(v_x_889_);
if (v_isSharedCheck_918_ == 0)
{
v___x_894_ = v_x_889_;
v_isShared_895_ = v_isSharedCheck_918_;
goto v_resetjp_893_;
}
else
{
lean_inc(v_tail_892_);
lean_inc(v_value_891_);
lean_inc(v_key_890_);
lean_dec(v_x_889_);
v___x_894_ = lean_box(0);
v_isShared_895_ = v_isSharedCheck_918_;
goto v_resetjp_893_;
}
v_resetjp_893_:
{
lean_object* v___x_896_; uint64_t v___y_898_; 
v___x_896_ = lean_array_get_size(v_x_888_);
if (lean_obj_tag(v_key_890_) == 0)
{
uint64_t v___x_916_; 
v___x_916_ = 1723ULL;
v___y_898_ = v___x_916_;
goto v___jp_897_;
}
else
{
uint64_t v_hash_917_; 
v_hash_917_ = lean_ctor_get_uint64(v_key_890_, sizeof(void*)*2);
v___y_898_ = v_hash_917_;
goto v___jp_897_;
}
v___jp_897_:
{
uint64_t v___x_899_; uint64_t v___x_900_; uint64_t v_fold_901_; uint64_t v___x_902_; uint64_t v___x_903_; uint64_t v___x_904_; size_t v___x_905_; size_t v___x_906_; size_t v___x_907_; size_t v___x_908_; size_t v___x_909_; lean_object* v___x_910_; lean_object* v___x_912_; 
v___x_899_ = 32ULL;
v___x_900_ = lean_uint64_shift_right(v___y_898_, v___x_899_);
v_fold_901_ = lean_uint64_xor(v___y_898_, v___x_900_);
v___x_902_ = 16ULL;
v___x_903_ = lean_uint64_shift_right(v_fold_901_, v___x_902_);
v___x_904_ = lean_uint64_xor(v_fold_901_, v___x_903_);
v___x_905_ = lean_uint64_to_usize(v___x_904_);
v___x_906_ = lean_usize_of_nat(v___x_896_);
v___x_907_ = ((size_t)1ULL);
v___x_908_ = lean_usize_sub(v___x_906_, v___x_907_);
v___x_909_ = lean_usize_land(v___x_905_, v___x_908_);
v___x_910_ = lean_array_uget_borrowed(v_x_888_, v___x_909_);
lean_inc(v___x_910_);
if (v_isShared_895_ == 0)
{
lean_ctor_set(v___x_894_, 2, v___x_910_);
v___x_912_ = v___x_894_;
goto v_reusejp_911_;
}
else
{
lean_object* v_reuseFailAlloc_915_; 
v_reuseFailAlloc_915_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_915_, 0, v_key_890_);
lean_ctor_set(v_reuseFailAlloc_915_, 1, v_value_891_);
lean_ctor_set(v_reuseFailAlloc_915_, 2, v___x_910_);
v___x_912_ = v_reuseFailAlloc_915_;
goto v_reusejp_911_;
}
v_reusejp_911_:
{
lean_object* v___x_913_; 
v___x_913_ = lean_array_uset(v_x_888_, v___x_909_, v___x_912_);
v_x_888_ = v___x_913_;
v_x_889_ = v_tail_892_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0_spec__1_spec__2___redArg(lean_object* v_i_919_, lean_object* v_source_920_, lean_object* v_target_921_){
_start:
{
lean_object* v___x_922_; uint8_t v___x_923_; 
v___x_922_ = lean_array_get_size(v_source_920_);
v___x_923_ = lean_nat_dec_lt(v_i_919_, v___x_922_);
if (v___x_923_ == 0)
{
lean_dec_ref(v_source_920_);
lean_dec(v_i_919_);
return v_target_921_;
}
else
{
lean_object* v_es_924_; lean_object* v___x_925_; lean_object* v_source_926_; lean_object* v_target_927_; lean_object* v___x_928_; lean_object* v___x_929_; 
v_es_924_ = lean_array_fget(v_source_920_, v_i_919_);
v___x_925_ = lean_box(0);
v_source_926_ = lean_array_fset(v_source_920_, v_i_919_, v___x_925_);
v_target_927_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0_spec__1_spec__2_spec__3___redArg(v_target_921_, v_es_924_);
v___x_928_ = lean_unsigned_to_nat(1u);
v___x_929_ = lean_nat_add(v_i_919_, v___x_928_);
lean_dec(v_i_919_);
v_i_919_ = v___x_929_;
v_source_920_ = v_source_926_;
v_target_921_ = v_target_927_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0_spec__1___redArg(lean_object* v_data_931_){
_start:
{
lean_object* v___x_932_; lean_object* v___x_933_; lean_object* v_nbuckets_934_; lean_object* v___x_935_; lean_object* v___x_936_; lean_object* v___x_937_; lean_object* v___x_938_; 
v___x_932_ = lean_array_get_size(v_data_931_);
v___x_933_ = lean_unsigned_to_nat(2u);
v_nbuckets_934_ = lean_nat_mul(v___x_932_, v___x_933_);
v___x_935_ = lean_unsigned_to_nat(0u);
v___x_936_ = lean_box(0);
v___x_937_ = lean_mk_array(v_nbuckets_934_, v___x_936_);
v___x_938_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0_spec__1_spec__2___redArg(v___x_935_, v_data_931_, v___x_937_);
return v___x_938_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(lean_object* v_m_939_, lean_object* v_a_940_, lean_object* v_b_941_){
_start:
{
lean_object* v_size_942_; lean_object* v_buckets_943_; lean_object* v___x_944_; uint64_t v___y_946_; 
v_size_942_ = lean_ctor_get(v_m_939_, 0);
v_buckets_943_ = lean_ctor_get(v_m_939_, 1);
v___x_944_ = lean_array_get_size(v_buckets_943_);
if (lean_obj_tag(v_a_940_) == 0)
{
uint64_t v___x_983_; 
v___x_983_ = 1723ULL;
v___y_946_ = v___x_983_;
goto v___jp_945_;
}
else
{
uint64_t v_hash_984_; 
v_hash_984_ = lean_ctor_get_uint64(v_a_940_, sizeof(void*)*2);
v___y_946_ = v_hash_984_;
goto v___jp_945_;
}
v___jp_945_:
{
uint64_t v___x_947_; uint64_t v___x_948_; uint64_t v_fold_949_; uint64_t v___x_950_; uint64_t v___x_951_; uint64_t v___x_952_; size_t v___x_953_; size_t v___x_954_; size_t v___x_955_; size_t v___x_956_; size_t v___x_957_; lean_object* v_bkt_958_; uint8_t v___x_959_; 
v___x_947_ = 32ULL;
v___x_948_ = lean_uint64_shift_right(v___y_946_, v___x_947_);
v_fold_949_ = lean_uint64_xor(v___y_946_, v___x_948_);
v___x_950_ = 16ULL;
v___x_951_ = lean_uint64_shift_right(v_fold_949_, v___x_950_);
v___x_952_ = lean_uint64_xor(v_fold_949_, v___x_951_);
v___x_953_ = lean_uint64_to_usize(v___x_952_);
v___x_954_ = lean_usize_of_nat(v___x_944_);
v___x_955_ = ((size_t)1ULL);
v___x_956_ = lean_usize_sub(v___x_954_, v___x_955_);
v___x_957_ = lean_usize_land(v___x_953_, v___x_956_);
v_bkt_958_ = lean_array_uget_borrowed(v_buckets_943_, v___x_957_);
v___x_959_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0_spec__0___redArg(v_a_940_, v_bkt_958_);
if (v___x_959_ == 0)
{
lean_object* v___x_961_; uint8_t v_isShared_962_; uint8_t v_isSharedCheck_980_; 
lean_inc_ref(v_buckets_943_);
lean_inc(v_size_942_);
v_isSharedCheck_980_ = !lean_is_exclusive(v_m_939_);
if (v_isSharedCheck_980_ == 0)
{
lean_object* v_unused_981_; lean_object* v_unused_982_; 
v_unused_981_ = lean_ctor_get(v_m_939_, 1);
lean_dec(v_unused_981_);
v_unused_982_ = lean_ctor_get(v_m_939_, 0);
lean_dec(v_unused_982_);
v___x_961_ = v_m_939_;
v_isShared_962_ = v_isSharedCheck_980_;
goto v_resetjp_960_;
}
else
{
lean_dec(v_m_939_);
v___x_961_ = lean_box(0);
v_isShared_962_ = v_isSharedCheck_980_;
goto v_resetjp_960_;
}
v_resetjp_960_:
{
lean_object* v___x_963_; lean_object* v_size_x27_964_; lean_object* v___x_965_; lean_object* v_buckets_x27_966_; lean_object* v___x_967_; lean_object* v___x_968_; lean_object* v___x_969_; lean_object* v___x_970_; lean_object* v___x_971_; uint8_t v___x_972_; 
v___x_963_ = lean_unsigned_to_nat(1u);
v_size_x27_964_ = lean_nat_add(v_size_942_, v___x_963_);
lean_dec(v_size_942_);
lean_inc(v_bkt_958_);
v___x_965_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_965_, 0, v_a_940_);
lean_ctor_set(v___x_965_, 1, v_b_941_);
lean_ctor_set(v___x_965_, 2, v_bkt_958_);
v_buckets_x27_966_ = lean_array_uset(v_buckets_943_, v___x_957_, v___x_965_);
v___x_967_ = lean_unsigned_to_nat(4u);
v___x_968_ = lean_nat_mul(v_size_x27_964_, v___x_967_);
v___x_969_ = lean_unsigned_to_nat(3u);
v___x_970_ = lean_nat_div(v___x_968_, v___x_969_);
lean_dec(v___x_968_);
v___x_971_ = lean_array_get_size(v_buckets_x27_966_);
v___x_972_ = lean_nat_dec_le(v___x_970_, v___x_971_);
lean_dec(v___x_970_);
if (v___x_972_ == 0)
{
lean_object* v_val_973_; lean_object* v___x_975_; 
v_val_973_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0_spec__1___redArg(v_buckets_x27_966_);
if (v_isShared_962_ == 0)
{
lean_ctor_set(v___x_961_, 1, v_val_973_);
lean_ctor_set(v___x_961_, 0, v_size_x27_964_);
v___x_975_ = v___x_961_;
goto v_reusejp_974_;
}
else
{
lean_object* v_reuseFailAlloc_976_; 
v_reuseFailAlloc_976_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_976_, 0, v_size_x27_964_);
lean_ctor_set(v_reuseFailAlloc_976_, 1, v_val_973_);
v___x_975_ = v_reuseFailAlloc_976_;
goto v_reusejp_974_;
}
v_reusejp_974_:
{
return v___x_975_;
}
}
else
{
lean_object* v___x_978_; 
if (v_isShared_962_ == 0)
{
lean_ctor_set(v___x_961_, 1, v_buckets_x27_966_);
lean_ctor_set(v___x_961_, 0, v_size_x27_964_);
v___x_978_ = v___x_961_;
goto v_reusejp_977_;
}
else
{
lean_object* v_reuseFailAlloc_979_; 
v_reuseFailAlloc_979_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_979_, 0, v_size_x27_964_);
lean_ctor_set(v_reuseFailAlloc_979_, 1, v_buckets_x27_966_);
v___x_978_ = v_reuseFailAlloc_979_;
goto v_reusejp_977_;
}
v_reusejp_977_:
{
return v___x_978_;
}
}
}
}
else
{
lean_dec(v_b_941_);
lean_dec(v_a_940_);
return v_m_939_;
}
}
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__50(void){
_start:
{
lean_object* v___x_1109_; lean_object* v___x_1110_; lean_object* v___x_1111_; 
v___x_1109_ = lean_box(0);
v___x_1110_ = lean_unsigned_to_nat(16u);
v___x_1111_ = lean_mk_array(v___x_1110_, v___x_1109_);
return v___x_1111_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__51(void){
_start:
{
lean_object* v___x_1112_; lean_object* v___x_1113_; lean_object* v___x_1114_; 
v___x_1112_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__50, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__50_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__50);
v___x_1113_ = lean_unsigned_to_nat(0u);
v___x_1114_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1114_, 0, v___x_1113_);
lean_ctor_set(v___x_1114_, 1, v___x_1112_);
return v___x_1114_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__52(void){
_start:
{
lean_object* v___x_1115_; lean_object* v___x_1116_; lean_object* v___x_1117_; lean_object* v___x_1118_; 
v___x_1115_ = lean_box(0);
v___x_1116_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__49));
v___x_1117_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__51, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__51_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__51);
v___x_1118_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1117_, v___x_1116_, v___x_1115_);
return v___x_1118_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__53(void){
_start:
{
lean_object* v___x_1119_; lean_object* v___x_1120_; lean_object* v___x_1121_; lean_object* v___x_1122_; 
v___x_1119_ = lean_box(0);
v___x_1120_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__47));
v___x_1121_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__52, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__52_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__52);
v___x_1122_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1121_, v___x_1120_, v___x_1119_);
return v___x_1122_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__54(void){
_start:
{
lean_object* v___x_1123_; lean_object* v___x_1124_; lean_object* v___x_1125_; lean_object* v___x_1126_; 
v___x_1123_ = lean_box(0);
v___x_1124_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__45));
v___x_1125_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__53, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__53_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__53);
v___x_1126_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1125_, v___x_1124_, v___x_1123_);
return v___x_1126_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__55(void){
_start:
{
lean_object* v___x_1127_; lean_object* v___x_1128_; lean_object* v___x_1129_; lean_object* v___x_1130_; 
v___x_1127_ = lean_box(0);
v___x_1128_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__43));
v___x_1129_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__54, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__54_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__54);
v___x_1130_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1129_, v___x_1128_, v___x_1127_);
return v___x_1130_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__56(void){
_start:
{
lean_object* v___x_1131_; lean_object* v___x_1132_; lean_object* v___x_1133_; lean_object* v___x_1134_; 
v___x_1131_ = lean_box(0);
v___x_1132_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__40));
v___x_1133_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__55, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__55_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__55);
v___x_1134_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1133_, v___x_1132_, v___x_1131_);
return v___x_1134_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__57(void){
_start:
{
lean_object* v___x_1135_; lean_object* v___x_1136_; lean_object* v___x_1137_; lean_object* v___x_1138_; 
v___x_1135_ = lean_box(0);
v___x_1136_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__38));
v___x_1137_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__56, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__56_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__56);
v___x_1138_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1137_, v___x_1136_, v___x_1135_);
return v___x_1138_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__58(void){
_start:
{
lean_object* v___x_1139_; lean_object* v___x_1140_; lean_object* v___x_1141_; lean_object* v___x_1142_; 
v___x_1139_ = lean_box(0);
v___x_1140_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__36));
v___x_1141_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__57, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__57_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__57);
v___x_1142_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1141_, v___x_1140_, v___x_1139_);
return v___x_1142_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__59(void){
_start:
{
lean_object* v___x_1143_; lean_object* v___x_1144_; lean_object* v___x_1145_; lean_object* v___x_1146_; 
v___x_1143_ = lean_box(0);
v___x_1144_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__34));
v___x_1145_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__58, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__58_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__58);
v___x_1146_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1145_, v___x_1144_, v___x_1143_);
return v___x_1146_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__60(void){
_start:
{
lean_object* v___x_1147_; lean_object* v___x_1148_; lean_object* v___x_1149_; lean_object* v___x_1150_; 
v___x_1147_ = lean_box(0);
v___x_1148_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__32));
v___x_1149_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__59, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__59_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__59);
v___x_1150_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1149_, v___x_1148_, v___x_1147_);
return v___x_1150_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__61(void){
_start:
{
lean_object* v___x_1151_; lean_object* v___x_1152_; lean_object* v___x_1153_; lean_object* v___x_1154_; 
v___x_1151_ = lean_box(0);
v___x_1152_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__29));
v___x_1153_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__60, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__60_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__60);
v___x_1154_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1153_, v___x_1152_, v___x_1151_);
return v___x_1154_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__62(void){
_start:
{
lean_object* v___x_1155_; lean_object* v___x_1156_; lean_object* v___x_1157_; lean_object* v___x_1158_; 
v___x_1155_ = lean_box(0);
v___x_1156_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__27));
v___x_1157_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__61, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__61_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__61);
v___x_1158_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1157_, v___x_1156_, v___x_1155_);
return v___x_1158_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__63(void){
_start:
{
lean_object* v___x_1159_; lean_object* v___x_1160_; lean_object* v___x_1161_; lean_object* v___x_1162_; 
v___x_1159_ = lean_box(0);
v___x_1160_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__25));
v___x_1161_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__62, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__62_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__62);
v___x_1162_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1161_, v___x_1160_, v___x_1159_);
return v___x_1162_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__64(void){
_start:
{
lean_object* v___x_1163_; lean_object* v___x_1164_; lean_object* v___x_1165_; lean_object* v___x_1166_; 
v___x_1163_ = lean_box(0);
v___x_1164_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__23));
v___x_1165_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__63, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__63_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__63);
v___x_1166_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1165_, v___x_1164_, v___x_1163_);
return v___x_1166_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__65(void){
_start:
{
lean_object* v___x_1167_; lean_object* v___x_1168_; lean_object* v___x_1169_; lean_object* v___x_1170_; 
v___x_1167_ = lean_box(0);
v___x_1168_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__21));
v___x_1169_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__64, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__64_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__64);
v___x_1170_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1169_, v___x_1168_, v___x_1167_);
return v___x_1170_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__66(void){
_start:
{
lean_object* v___x_1171_; lean_object* v___x_1172_; lean_object* v___x_1173_; lean_object* v___x_1174_; 
v___x_1171_ = lean_box(0);
v___x_1172_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__18));
v___x_1173_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__65, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__65_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__65);
v___x_1174_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1173_, v___x_1172_, v___x_1171_);
return v___x_1174_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__67(void){
_start:
{
lean_object* v___x_1175_; lean_object* v___x_1176_; lean_object* v___x_1177_; lean_object* v___x_1178_; 
v___x_1175_ = lean_box(0);
v___x_1176_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__15));
v___x_1177_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__66, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__66_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__66);
v___x_1178_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1177_, v___x_1176_, v___x_1175_);
return v___x_1178_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__68(void){
_start:
{
lean_object* v___x_1179_; lean_object* v___x_1180_; lean_object* v___x_1181_; lean_object* v___x_1182_; 
v___x_1179_ = lean_box(0);
v___x_1180_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__13));
v___x_1181_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__67, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__67_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__67);
v___x_1182_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1181_, v___x_1180_, v___x_1179_);
return v___x_1182_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__69(void){
_start:
{
lean_object* v___x_1183_; lean_object* v___x_1184_; lean_object* v___x_1185_; lean_object* v___x_1186_; 
v___x_1183_ = lean_box(0);
v___x_1184_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__10));
v___x_1185_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__68, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__68_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__68);
v___x_1186_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1185_, v___x_1184_, v___x_1183_);
return v___x_1186_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__70(void){
_start:
{
lean_object* v___x_1187_; lean_object* v___x_1188_; lean_object* v___x_1189_; lean_object* v___x_1190_; 
v___x_1187_ = lean_box(0);
v___x_1188_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__8));
v___x_1189_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__69, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__69_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__69);
v___x_1190_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1189_, v___x_1188_, v___x_1187_);
return v___x_1190_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__71(void){
_start:
{
lean_object* v___x_1191_; lean_object* v___x_1192_; lean_object* v___x_1193_; lean_object* v___x_1194_; 
v___x_1191_ = lean_box(0);
v___x_1192_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__5));
v___x_1193_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__70, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__70_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__70);
v___x_1194_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1193_, v___x_1192_, v___x_1191_);
return v___x_1194_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__72(void){
_start:
{
lean_object* v___x_1195_; lean_object* v___x_1196_; lean_object* v___x_1197_; lean_object* v___x_1198_; 
v___x_1195_ = lean_box(0);
v___x_1196_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__3));
v___x_1197_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__71, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__71_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__71);
v___x_1198_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1197_, v___x_1196_, v___x_1195_);
return v___x_1198_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__73(void){
_start:
{
lean_object* v___x_1199_; lean_object* v___x_1200_; lean_object* v___x_1201_; lean_object* v___x_1202_; 
v___x_1199_ = lean_box(0);
v___x_1200_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__1));
v___x_1201_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__72, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__72_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__72);
v___x_1202_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1201_, v___x_1200_, v___x_1199_);
return v___x_1202_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers(void){
_start:
{
lean_object* v___x_1203_; 
v___x_1203_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__73, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__73_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__73);
return v___x_1203_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0(lean_object* v_00_u03b2_1204_, lean_object* v_m_1205_, lean_object* v_a_1206_, lean_object* v_b_1207_){
_start:
{
lean_object* v___x_1208_; 
v___x_1208_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v_m_1205_, v_a_1206_, v_b_1207_);
return v___x_1208_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0_spec__0(lean_object* v_00_u03b2_1209_, lean_object* v_a_1210_, lean_object* v_x_1211_){
_start:
{
uint8_t v___x_1212_; 
v___x_1212_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0_spec__0___redArg(v_a_1210_, v_x_1211_);
return v___x_1212_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0_spec__0___boxed(lean_object* v_00_u03b2_1213_, lean_object* v_a_1214_, lean_object* v_x_1215_){
_start:
{
uint8_t v_res_1216_; lean_object* v_r_1217_; 
v_res_1216_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0_spec__0(v_00_u03b2_1213_, v_a_1214_, v_x_1215_);
lean_dec(v_x_1215_);
lean_dec(v_a_1214_);
v_r_1217_ = lean_box(v_res_1216_);
return v_r_1217_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0_spec__1(lean_object* v_00_u03b2_1218_, lean_object* v_data_1219_){
_start:
{
lean_object* v___x_1220_; 
v___x_1220_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0_spec__1___redArg(v_data_1219_);
return v___x_1220_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0_spec__1_spec__2(lean_object* v_00_u03b2_1221_, lean_object* v_i_1222_, lean_object* v_source_1223_, lean_object* v_target_1224_){
_start:
{
lean_object* v___x_1225_; 
v___x_1225_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0_spec__1_spec__2___redArg(v_i_1222_, v_source_1223_, v_target_1224_);
return v___x_1225_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0_spec__1_spec__2_spec__3(lean_object* v_00_u03b2_1226_, lean_object* v_x_1227_, lean_object* v_x_1228_){
_start:
{
lean_object* v___x_1229_; 
v___x_1229_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0_spec__1_spec__2_spec__3___redArg(v_x_1227_, v_x_1228_);
return v___x_1229_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__86(void){
_start:
{
lean_object* v___x_1456_; lean_object* v___x_1457_; lean_object* v___x_1458_; lean_object* v___x_1459_; 
v___x_1456_ = lean_box(0);
v___x_1457_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__85));
v___x_1458_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__51, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__51_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__51);
v___x_1459_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1458_, v___x_1457_, v___x_1456_);
return v___x_1459_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__87(void){
_start:
{
lean_object* v___x_1460_; lean_object* v___x_1461_; lean_object* v___x_1462_; lean_object* v___x_1463_; 
v___x_1460_ = lean_box(0);
v___x_1461_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__83));
v___x_1462_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__86, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__86_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__86);
v___x_1463_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1462_, v___x_1461_, v___x_1460_);
return v___x_1463_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__88(void){
_start:
{
lean_object* v___x_1464_; lean_object* v___x_1465_; lean_object* v___x_1466_; lean_object* v___x_1467_; 
v___x_1464_ = lean_box(0);
v___x_1465_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__81));
v___x_1466_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__87, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__87_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__87);
v___x_1467_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1466_, v___x_1465_, v___x_1464_);
return v___x_1467_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__89(void){
_start:
{
lean_object* v___x_1468_; lean_object* v___x_1469_; lean_object* v___x_1470_; lean_object* v___x_1471_; 
v___x_1468_ = lean_box(0);
v___x_1469_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__78));
v___x_1470_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__88, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__88_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__88);
v___x_1471_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1470_, v___x_1469_, v___x_1468_);
return v___x_1471_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__90(void){
_start:
{
lean_object* v___x_1472_; lean_object* v___x_1473_; lean_object* v___x_1474_; lean_object* v___x_1475_; 
v___x_1472_ = lean_box(0);
v___x_1473_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__76));
v___x_1474_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__89, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__89_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__89);
v___x_1475_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1474_, v___x_1473_, v___x_1472_);
return v___x_1475_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__91(void){
_start:
{
lean_object* v___x_1476_; lean_object* v___x_1477_; lean_object* v___x_1478_; lean_object* v___x_1479_; 
v___x_1476_ = lean_box(0);
v___x_1477_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__74));
v___x_1478_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__90, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__90_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__90);
v___x_1479_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1478_, v___x_1477_, v___x_1476_);
return v___x_1479_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__92(void){
_start:
{
lean_object* v___x_1480_; lean_object* v___x_1481_; lean_object* v___x_1482_; lean_object* v___x_1483_; 
v___x_1480_ = lean_box(0);
v___x_1481_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__72));
v___x_1482_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__91, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__91_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__91);
v___x_1483_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1482_, v___x_1481_, v___x_1480_);
return v___x_1483_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__93(void){
_start:
{
lean_object* v___x_1484_; lean_object* v___x_1485_; lean_object* v___x_1486_; lean_object* v___x_1487_; 
v___x_1484_ = lean_box(0);
v___x_1485_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__70));
v___x_1486_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__92, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__92_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__92);
v___x_1487_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1486_, v___x_1485_, v___x_1484_);
return v___x_1487_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__94(void){
_start:
{
lean_object* v___x_1488_; lean_object* v___x_1489_; lean_object* v___x_1490_; lean_object* v___x_1491_; 
v___x_1488_ = lean_box(0);
v___x_1489_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__68));
v___x_1490_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__93, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__93_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__93);
v___x_1491_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1490_, v___x_1489_, v___x_1488_);
return v___x_1491_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__95(void){
_start:
{
lean_object* v___x_1492_; lean_object* v___x_1493_; lean_object* v___x_1494_; lean_object* v___x_1495_; 
v___x_1492_ = lean_box(0);
v___x_1493_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__66));
v___x_1494_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__94, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__94_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__94);
v___x_1495_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1494_, v___x_1493_, v___x_1492_);
return v___x_1495_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__96(void){
_start:
{
lean_object* v___x_1496_; lean_object* v___x_1497_; lean_object* v___x_1498_; lean_object* v___x_1499_; 
v___x_1496_ = lean_box(0);
v___x_1497_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__64));
v___x_1498_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__95, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__95_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__95);
v___x_1499_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1498_, v___x_1497_, v___x_1496_);
return v___x_1499_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__97(void){
_start:
{
lean_object* v___x_1500_; lean_object* v___x_1501_; lean_object* v___x_1502_; lean_object* v___x_1503_; 
v___x_1500_ = lean_box(0);
v___x_1501_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__60));
v___x_1502_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__96, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__96_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__96);
v___x_1503_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1502_, v___x_1501_, v___x_1500_);
return v___x_1503_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__98(void){
_start:
{
lean_object* v___x_1504_; lean_object* v___x_1505_; lean_object* v___x_1506_; lean_object* v___x_1507_; 
v___x_1504_ = lean_box(0);
v___x_1505_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__58));
v___x_1506_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__97, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__97_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__97);
v___x_1507_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1506_, v___x_1505_, v___x_1504_);
return v___x_1507_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__99(void){
_start:
{
lean_object* v___x_1508_; lean_object* v___x_1509_; lean_object* v___x_1510_; lean_object* v___x_1511_; 
v___x_1508_ = lean_box(0);
v___x_1509_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__55));
v___x_1510_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__98, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__98_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__98);
v___x_1511_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1510_, v___x_1509_, v___x_1508_);
return v___x_1511_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__100(void){
_start:
{
lean_object* v___x_1512_; lean_object* v___x_1513_; lean_object* v___x_1514_; lean_object* v___x_1515_; 
v___x_1512_ = lean_box(0);
v___x_1513_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__53));
v___x_1514_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__99, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__99_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__99);
v___x_1515_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1514_, v___x_1513_, v___x_1512_);
return v___x_1515_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__101(void){
_start:
{
lean_object* v___x_1516_; lean_object* v___x_1517_; lean_object* v___x_1518_; lean_object* v___x_1519_; 
v___x_1516_ = lean_box(0);
v___x_1517_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__51));
v___x_1518_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__100, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__100_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__100);
v___x_1519_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1518_, v___x_1517_, v___x_1516_);
return v___x_1519_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__102(void){
_start:
{
lean_object* v___x_1520_; lean_object* v___x_1521_; lean_object* v___x_1522_; lean_object* v___x_1523_; 
v___x_1520_ = lean_box(0);
v___x_1521_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__49));
v___x_1522_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__101, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__101_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__101);
v___x_1523_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1522_, v___x_1521_, v___x_1520_);
return v___x_1523_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__103(void){
_start:
{
lean_object* v___x_1524_; lean_object* v___x_1525_; lean_object* v___x_1526_; lean_object* v___x_1527_; 
v___x_1524_ = lean_box(0);
v___x_1525_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__47));
v___x_1526_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__102, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__102_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__102);
v___x_1527_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1526_, v___x_1525_, v___x_1524_);
return v___x_1527_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__104(void){
_start:
{
lean_object* v___x_1528_; lean_object* v___x_1529_; lean_object* v___x_1530_; lean_object* v___x_1531_; 
v___x_1528_ = lean_box(0);
v___x_1529_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__45));
v___x_1530_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__103, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__103_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__103);
v___x_1531_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1530_, v___x_1529_, v___x_1528_);
return v___x_1531_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__105(void){
_start:
{
lean_object* v___x_1532_; lean_object* v___x_1533_; lean_object* v___x_1534_; lean_object* v___x_1535_; 
v___x_1532_ = lean_box(0);
v___x_1533_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__43));
v___x_1534_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__104, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__104_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__104);
v___x_1535_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1534_, v___x_1533_, v___x_1532_);
return v___x_1535_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__106(void){
_start:
{
lean_object* v___x_1536_; lean_object* v___x_1537_; lean_object* v___x_1538_; lean_object* v___x_1539_; 
v___x_1536_ = lean_box(0);
v___x_1537_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__23));
v___x_1538_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__105, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__105_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__105);
v___x_1539_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1538_, v___x_1537_, v___x_1536_);
return v___x_1539_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__107(void){
_start:
{
lean_object* v___x_1540_; lean_object* v___x_1541_; lean_object* v___x_1542_; lean_object* v___x_1543_; 
v___x_1540_ = lean_box(0);
v___x_1541_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__21));
v___x_1542_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__106, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__106_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__106);
v___x_1543_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1542_, v___x_1541_, v___x_1540_);
return v___x_1543_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__108(void){
_start:
{
lean_object* v___x_1544_; lean_object* v___x_1545_; lean_object* v___x_1546_; lean_object* v___x_1547_; 
v___x_1544_ = lean_box(0);
v___x_1545_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__41));
v___x_1546_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__107, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__107_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__107);
v___x_1547_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1546_, v___x_1545_, v___x_1544_);
return v___x_1547_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__109(void){
_start:
{
lean_object* v___x_1548_; lean_object* v___x_1549_; lean_object* v___x_1550_; lean_object* v___x_1551_; 
v___x_1548_ = lean_box(0);
v___x_1549_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__38));
v___x_1550_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__108, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__108_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__108);
v___x_1551_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1550_, v___x_1549_, v___x_1548_);
return v___x_1551_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__110(void){
_start:
{
lean_object* v___x_1552_; lean_object* v___x_1553_; lean_object* v___x_1554_; lean_object* v___x_1555_; 
v___x_1552_ = lean_box(0);
v___x_1553_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__36));
v___x_1554_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__109, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__109_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__109);
v___x_1555_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1554_, v___x_1553_, v___x_1552_);
return v___x_1555_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__111(void){
_start:
{
lean_object* v___x_1556_; lean_object* v___x_1557_; lean_object* v___x_1558_; lean_object* v___x_1559_; 
v___x_1556_ = lean_box(0);
v___x_1557_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__34));
v___x_1558_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__110, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__110_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__110);
v___x_1559_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1558_, v___x_1557_, v___x_1556_);
return v___x_1559_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__112(void){
_start:
{
lean_object* v___x_1560_; lean_object* v___x_1561_; lean_object* v___x_1562_; lean_object* v___x_1563_; 
v___x_1560_ = lean_box(0);
v___x_1561_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__32));
v___x_1562_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__111, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__111_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__111);
v___x_1563_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1562_, v___x_1561_, v___x_1560_);
return v___x_1563_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__113(void){
_start:
{
lean_object* v___x_1564_; lean_object* v___x_1565_; lean_object* v___x_1566_; lean_object* v___x_1567_; 
v___x_1564_ = lean_box(0);
v___x_1565_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__30));
v___x_1566_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__112, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__112_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__112);
v___x_1567_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1566_, v___x_1565_, v___x_1564_);
return v___x_1567_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__114(void){
_start:
{
lean_object* v___x_1568_; lean_object* v___x_1569_; lean_object* v___x_1570_; lean_object* v___x_1571_; 
v___x_1568_ = lean_box(0);
v___x_1569_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__27));
v___x_1570_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__113, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__113_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__113);
v___x_1571_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1570_, v___x_1569_, v___x_1568_);
return v___x_1571_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__115(void){
_start:
{
lean_object* v___x_1572_; lean_object* v___x_1573_; lean_object* v___x_1574_; lean_object* v___x_1575_; 
v___x_1572_ = lean_box(0);
v___x_1573_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__25));
v___x_1574_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__114, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__114_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__114);
v___x_1575_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1574_, v___x_1573_, v___x_1572_);
return v___x_1575_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__116(void){
_start:
{
lean_object* v___x_1576_; lean_object* v___x_1577_; lean_object* v___x_1578_; lean_object* v___x_1579_; 
v___x_1576_ = lean_box(0);
v___x_1577_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers___closed__18));
v___x_1578_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__115, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__115_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__115);
v___x_1579_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1578_, v___x_1577_, v___x_1576_);
return v___x_1579_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__117(void){
_start:
{
lean_object* v___x_1580_; lean_object* v___x_1581_; lean_object* v___x_1582_; lean_object* v___x_1583_; 
v___x_1580_ = lean_box(0);
v___x_1581_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__23));
v___x_1582_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__116, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__116_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__116);
v___x_1583_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1582_, v___x_1581_, v___x_1580_);
return v___x_1583_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__118(void){
_start:
{
lean_object* v___x_1584_; lean_object* v___x_1585_; lean_object* v___x_1586_; lean_object* v___x_1587_; 
v___x_1584_ = lean_box(0);
v___x_1585_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__21));
v___x_1586_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__117, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__117_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__117);
v___x_1587_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1586_, v___x_1585_, v___x_1584_);
return v___x_1587_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__119(void){
_start:
{
lean_object* v___x_1588_; lean_object* v___x_1589_; lean_object* v___x_1590_; lean_object* v___x_1591_; 
v___x_1588_ = lean_box(0);
v___x_1589_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__19));
v___x_1590_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__118, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__118_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__118);
v___x_1591_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1590_, v___x_1589_, v___x_1588_);
return v___x_1591_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__120(void){
_start:
{
lean_object* v___x_1592_; lean_object* v___x_1593_; lean_object* v___x_1594_; lean_object* v___x_1595_; 
v___x_1592_ = lean_box(0);
v___x_1593_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__17));
v___x_1594_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__119, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__119_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__119);
v___x_1595_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1594_, v___x_1593_, v___x_1592_);
return v___x_1595_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__121(void){
_start:
{
lean_object* v___x_1596_; lean_object* v___x_1597_; lean_object* v___x_1598_; lean_object* v___x_1599_; 
v___x_1596_ = lean_box(0);
v___x_1597_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__15));
v___x_1598_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__120, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__120_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__120);
v___x_1599_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1598_, v___x_1597_, v___x_1596_);
return v___x_1599_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__122(void){
_start:
{
lean_object* v___x_1600_; lean_object* v___x_1601_; lean_object* v___x_1602_; lean_object* v___x_1603_; 
v___x_1600_ = lean_box(0);
v___x_1601_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__13));
v___x_1602_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__121, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__121_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__121);
v___x_1603_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1602_, v___x_1601_, v___x_1600_);
return v___x_1603_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__123(void){
_start:
{
lean_object* v___x_1604_; lean_object* v___x_1605_; lean_object* v___x_1606_; lean_object* v___x_1607_; 
v___x_1604_ = lean_box(0);
v___x_1605_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__11));
v___x_1606_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__122, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__122_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__122);
v___x_1607_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1606_, v___x_1605_, v___x_1604_);
return v___x_1607_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__124(void){
_start:
{
lean_object* v___x_1608_; lean_object* v___x_1609_; lean_object* v___x_1610_; lean_object* v___x_1611_; 
v___x_1608_ = lean_box(0);
v___x_1609_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__9));
v___x_1610_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__123, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__123_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__123);
v___x_1611_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1610_, v___x_1609_, v___x_1608_);
return v___x_1611_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__125(void){
_start:
{
lean_object* v___x_1612_; lean_object* v___x_1613_; lean_object* v___x_1614_; lean_object* v___x_1615_; 
v___x_1612_ = lean_box(0);
v___x_1613_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__7));
v___x_1614_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__124, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__124_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__124);
v___x_1615_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1614_, v___x_1613_, v___x_1612_);
return v___x_1615_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__126(void){
_start:
{
lean_object* v___x_1616_; lean_object* v___x_1617_; lean_object* v___x_1618_; lean_object* v___x_1619_; 
v___x_1616_ = lean_box(0);
v___x_1617_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__5));
v___x_1618_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__125, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__125_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__125);
v___x_1619_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1618_, v___x_1617_, v___x_1616_);
return v___x_1619_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__127(void){
_start:
{
lean_object* v___x_1620_; lean_object* v___x_1621_; lean_object* v___x_1622_; lean_object* v___x_1623_; 
v___x_1620_ = lean_box(0);
v___x_1621_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__3));
v___x_1622_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__126, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__126_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__126);
v___x_1623_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1622_, v___x_1621_, v___x_1620_);
return v___x_1623_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__128(void){
_start:
{
lean_object* v___x_1624_; lean_object* v___x_1625_; lean_object* v___x_1626_; lean_object* v___x_1627_; 
v___x_1624_ = lean_box(0);
v___x_1625_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__1));
v___x_1626_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__127, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__127_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__127);
v___x_1627_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1626_, v___x_1625_, v___x_1624_);
return v___x_1627_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__129(void){
_start:
{
lean_object* v___x_1628_; lean_object* v___x_1629_; lean_object* v___x_1630_; lean_object* v___x_1631_; 
v___x_1628_ = lean_box(0);
v___x_1629_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__0));
v___x_1630_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__128, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__128_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__128);
v___x_1631_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0___redArg(v___x_1630_, v___x_1629_, v___x_1628_);
return v___x_1631_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible(void){
_start:
{
lean_object* v___x_1632_; 
v___x_1632_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__129, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__129_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__129);
return v___x_1632_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_usesGoal_x3f(lean_object* v_x_1642_){
_start:
{
if (lean_obj_tag(v_x_1642_) == 1)
{
lean_object* v_pre_1643_; 
v_pre_1643_ = lean_ctor_get(v_x_1642_, 0);
switch(lean_obj_tag(v_pre_1643_))
{
case 1:
{
lean_object* v_pre_1644_; 
v_pre_1644_ = lean_ctor_get(v_pre_1643_, 0);
if (lean_obj_tag(v_pre_1644_) == 1)
{
lean_object* v_pre_1645_; 
v_pre_1645_ = lean_ctor_get(v_pre_1644_, 0);
switch(lean_obj_tag(v_pre_1645_))
{
case 1:
{
lean_object* v_pre_1646_; 
v_pre_1646_ = lean_ctor_get(v_pre_1645_, 0);
if (lean_obj_tag(v_pre_1646_) == 0)
{
lean_object* v_str_1647_; lean_object* v_str_1648_; lean_object* v_str_1649_; lean_object* v_str_1650_; lean_object* v___x_1651_; uint8_t v___x_1652_; 
v_str_1647_ = lean_ctor_get(v_x_1642_, 1);
v_str_1648_ = lean_ctor_get(v_pre_1643_, 1);
v_str_1649_ = lean_ctor_get(v_pre_1644_, 1);
v_str_1650_ = lean_ctor_get(v_pre_1645_, 1);
v___x_1651_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__0));
v___x_1652_ = lean_string_dec_eq(v_str_1650_, v___x_1651_);
if (v___x_1652_ == 0)
{
uint8_t v___x_1653_; 
v___x_1653_ = 1;
return v___x_1653_;
}
else
{
lean_object* v___x_1654_; uint8_t v___x_1655_; 
v___x_1654_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__1));
v___x_1655_ = lean_string_dec_eq(v_str_1649_, v___x_1654_);
if (v___x_1655_ == 0)
{
return v___x_1652_;
}
else
{
lean_object* v___x_1656_; uint8_t v___x_1657_; 
v___x_1656_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2));
v___x_1657_ = lean_string_dec_eq(v_str_1648_, v___x_1656_);
if (v___x_1657_ == 0)
{
return v___x_1655_;
}
else
{
lean_object* v___x_1658_; uint8_t v___x_1659_; 
v___x_1658_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_usesGoal_x3f___closed__0));
v___x_1659_ = lean_string_dec_eq(v_str_1647_, v___x_1658_);
if (v___x_1659_ == 0)
{
lean_object* v___x_1660_; uint8_t v___x_1661_; 
v___x_1660_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_usesGoal_x3f___closed__1));
v___x_1661_ = lean_string_dec_eq(v_str_1647_, v___x_1660_);
if (v___x_1661_ == 0)
{
lean_object* v___x_1662_; uint8_t v___x_1663_; 
v___x_1662_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_usesGoal_x3f___closed__2));
v___x_1663_ = lean_string_dec_eq(v_str_1647_, v___x_1662_);
if (v___x_1663_ == 0)
{
lean_object* v___x_1664_; uint8_t v___x_1665_; 
v___x_1664_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_usesGoal_x3f___closed__3));
v___x_1665_ = lean_string_dec_eq(v_str_1647_, v___x_1664_);
if (v___x_1665_ == 0)
{
lean_object* v___x_1666_; uint8_t v___x_1667_; 
v___x_1666_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_usesGoal_x3f___closed__4));
v___x_1667_ = lean_string_dec_eq(v_str_1647_, v___x_1666_);
if (v___x_1667_ == 0)
{
lean_object* v___x_1668_; uint8_t v___x_1669_; 
v___x_1668_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_usesGoal_x3f___closed__5));
v___x_1669_ = lean_string_dec_eq(v_str_1647_, v___x_1668_);
if (v___x_1669_ == 0)
{
lean_object* v___x_1670_; uint8_t v___x_1671_; 
v___x_1670_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_usesGoal_x3f___closed__6));
v___x_1671_ = lean_string_dec_eq(v_str_1647_, v___x_1670_);
if (v___x_1671_ == 0)
{
return v___x_1657_;
}
else
{
return v___x_1669_;
}
}
else
{
return v___x_1667_;
}
}
else
{
return v___x_1665_;
}
}
else
{
return v___x_1663_;
}
}
else
{
return v___x_1661_;
}
}
else
{
return v___x_1659_;
}
}
else
{
uint8_t v___x_1672_; 
v___x_1672_ = 0;
return v___x_1672_;
}
}
}
}
}
else
{
uint8_t v___x_1673_; 
v___x_1673_ = 1;
return v___x_1673_;
}
}
case 0:
{
lean_object* v_str_1674_; lean_object* v_str_1675_; lean_object* v_str_1676_; lean_object* v___x_1677_; uint8_t v___x_1678_; 
v_str_1674_ = lean_ctor_get(v_x_1642_, 1);
v_str_1675_ = lean_ctor_get(v_pre_1643_, 1);
v_str_1676_ = lean_ctor_get(v_pre_1644_, 1);
v___x_1677_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4_));
v___x_1678_ = lean_string_dec_eq(v_str_1676_, v___x_1677_);
if (v___x_1678_ == 0)
{
uint8_t v___x_1679_; 
v___x_1679_ = 1;
return v___x_1679_;
}
else
{
lean_object* v___x_1680_; uint8_t v___x_1681_; 
v___x_1680_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2));
v___x_1681_ = lean_string_dec_eq(v_str_1675_, v___x_1680_);
if (v___x_1681_ == 0)
{
return v___x_1678_;
}
else
{
lean_object* v___x_1682_; uint8_t v___x_1683_; 
v___x_1682_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_usesGoal_x3f___closed__7));
v___x_1683_ = lean_string_dec_eq(v_str_1674_, v___x_1682_);
if (v___x_1683_ == 0)
{
return v___x_1681_;
}
else
{
uint8_t v___x_1684_; 
v___x_1684_ = 0;
return v___x_1684_;
}
}
}
}
default: 
{
uint8_t v___x_1685_; 
v___x_1685_ = 1;
return v___x_1685_;
}
}
}
else
{
uint8_t v___x_1686_; 
v___x_1686_ = 1;
return v___x_1686_;
}
}
case 0:
{
lean_object* v_str_1687_; lean_object* v___x_1688_; uint8_t v___x_1689_; 
v_str_1687_ = lean_ctor_get(v_x_1642_, 1);
v___x_1688_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_usesGoal_x3f___closed__8));
v___x_1689_ = lean_string_dec_eq(v_str_1687_, v___x_1688_);
if (v___x_1689_ == 0)
{
uint8_t v___x_1690_; 
v___x_1690_ = 1;
return v___x_1690_;
}
else
{
uint8_t v___x_1691_; 
v___x_1691_ = 0;
return v___x_1691_;
}
}
default: 
{
uint8_t v___x_1692_; 
v___x_1692_ = 1;
return v___x_1692_;
}
}
}
else
{
uint8_t v___x_1693_; 
v___x_1693_ = 1;
return v___x_1693_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_usesGoal_x3f___boxed(lean_object* v_x_1694_){
_start:
{
uint8_t v_res_1695_; lean_object* v_r_1696_; 
v_res_1695_ = lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_usesGoal_x3f(v_x_1694_);
lean_dec(v_x_1694_);
v_r_1696_ = lean_box(v_res_1695_);
return v_r_1696_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_getFVarIdCandidates_spec__1(size_t v_sz_1697_, size_t v_i_1698_, lean_object* v_bs_1699_){
_start:
{
uint8_t v___x_1700_; 
v___x_1700_ = lean_usize_dec_lt(v_i_1698_, v_sz_1697_);
if (v___x_1700_ == 0)
{
return v_bs_1699_;
}
else
{
lean_object* v_v_1701_; lean_object* v___x_1702_; lean_object* v_bs_x27_1703_; lean_object* v___x_1704_; size_t v___x_1705_; size_t v___x_1706_; lean_object* v___x_1707_; 
v_v_1701_ = lean_array_uget(v_bs_1699_, v_i_1698_);
v___x_1702_ = lean_unsigned_to_nat(0u);
v_bs_x27_1703_ = lean_array_uset(v_bs_1699_, v_i_1698_, v___x_1702_);
v___x_1704_ = l_Lean_LocalDecl_fvarId(v_v_1701_);
lean_dec(v_v_1701_);
v___x_1705_ = ((size_t)1ULL);
v___x_1706_ = lean_usize_add(v_i_1698_, v___x_1705_);
v___x_1707_ = lean_array_uset(v_bs_x27_1703_, v_i_1698_, v___x_1704_);
v_i_1698_ = v___x_1706_;
v_bs_1699_ = v___x_1707_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_getFVarIdCandidates_spec__1___boxed(lean_object* v_sz_1709_, lean_object* v_i_1710_, lean_object* v_bs_1711_){
_start:
{
size_t v_sz_boxed_1712_; size_t v_i_boxed_1713_; lean_object* v_res_1714_; 
v_sz_boxed_1712_ = lean_unbox_usize(v_sz_1709_);
lean_dec(v_sz_1709_);
v_i_boxed_1713_ = lean_unbox_usize(v_i_1710_);
lean_dec(v_i_1710_);
v_res_1714_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_getFVarIdCandidates_spec__1(v_sz_boxed_1712_, v_i_boxed_1713_, v_bs_1711_);
return v_res_1714_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_getFVarIdCandidates_spec__0_spec__0(lean_object* v_as_1715_, size_t v_i_1716_, size_t v_stop_1717_, lean_object* v_b_1718_){
_start:
{
lean_object* v___y_1720_; uint8_t v___x_1724_; 
v___x_1724_ = lean_usize_dec_eq(v_i_1716_, v_stop_1717_);
if (v___x_1724_ == 0)
{
lean_object* v___x_1725_; 
v___x_1725_ = lean_array_uget_borrowed(v_as_1715_, v_i_1716_);
if (lean_obj_tag(v___x_1725_) == 0)
{
v___y_1720_ = v_b_1718_;
goto v___jp_1719_;
}
else
{
lean_object* v_val_1726_; lean_object* v___x_1727_; 
v_val_1726_ = lean_ctor_get(v___x_1725_, 0);
lean_inc(v_val_1726_);
v___x_1727_ = lean_array_push(v_b_1718_, v_val_1726_);
v___y_1720_ = v___x_1727_;
goto v___jp_1719_;
}
}
else
{
return v_b_1718_;
}
v___jp_1719_:
{
size_t v___x_1721_; size_t v___x_1722_; 
v___x_1721_ = ((size_t)1ULL);
v___x_1722_ = lean_usize_add(v_i_1716_, v___x_1721_);
v_i_1716_ = v___x_1722_;
v_b_1718_ = v___y_1720_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_getFVarIdCandidates_spec__0_spec__0___boxed(lean_object* v_as_1728_, lean_object* v_i_1729_, lean_object* v_stop_1730_, lean_object* v_b_1731_){
_start:
{
size_t v_i_boxed_1732_; size_t v_stop_boxed_1733_; lean_object* v_res_1734_; 
v_i_boxed_1732_ = lean_unbox_usize(v_i_1729_);
lean_dec(v_i_1729_);
v_stop_boxed_1733_ = lean_unbox_usize(v_stop_1730_);
lean_dec(v_stop_1730_);
v_res_1734_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_getFVarIdCandidates_spec__0_spec__0(v_as_1728_, v_i_boxed_1732_, v_stop_boxed_1733_, v_b_1731_);
lean_dec_ref(v_as_1728_);
return v_res_1734_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_getFVarIdCandidates_spec__0(lean_object* v_as_1737_, lean_object* v_start_1738_, lean_object* v_stop_1739_){
_start:
{
lean_object* v___x_1740_; uint8_t v___x_1741_; 
v___x_1740_ = ((lean_object*)(lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_getFVarIdCandidates_spec__0___closed__0));
v___x_1741_ = lean_nat_dec_lt(v_start_1738_, v_stop_1739_);
if (v___x_1741_ == 0)
{
return v___x_1740_;
}
else
{
lean_object* v___x_1742_; uint8_t v___x_1743_; 
v___x_1742_ = lean_array_get_size(v_as_1737_);
v___x_1743_ = lean_nat_dec_le(v_stop_1739_, v___x_1742_);
if (v___x_1743_ == 0)
{
uint8_t v___x_1744_; 
v___x_1744_ = lean_nat_dec_lt(v_start_1738_, v___x_1742_);
if (v___x_1744_ == 0)
{
return v___x_1740_;
}
else
{
size_t v___x_1745_; size_t v___x_1746_; lean_object* v___x_1747_; 
v___x_1745_ = lean_usize_of_nat(v_start_1738_);
v___x_1746_ = lean_usize_of_nat(v___x_1742_);
v___x_1747_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_getFVarIdCandidates_spec__0_spec__0(v_as_1737_, v___x_1745_, v___x_1746_, v___x_1740_);
return v___x_1747_;
}
}
else
{
size_t v___x_1748_; size_t v___x_1749_; lean_object* v___x_1750_; 
v___x_1748_ = lean_usize_of_nat(v_start_1738_);
v___x_1749_ = lean_usize_of_nat(v_stop_1739_);
v___x_1750_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_getFVarIdCandidates_spec__0_spec__0(v_as_1737_, v___x_1748_, v___x_1749_, v___x_1740_);
return v___x_1750_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_getFVarIdCandidates_spec__0___boxed(lean_object* v_as_1751_, lean_object* v_start_1752_, lean_object* v_stop_1753_){
_start:
{
lean_object* v_res_1754_; 
v_res_1754_ = lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_getFVarIdCandidates_spec__0(v_as_1751_, v_start_1752_, v_stop_1753_);
lean_dec(v_stop_1753_);
lean_dec(v_start_1752_);
lean_dec_ref(v_as_1751_);
return v_res_1754_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_getFVarIdCandidates(lean_object* v_fv_1755_, lean_object* v_name_1756_, lean_object* v_lctx_1757_){
_start:
{
lean_object* v___x_1758_; lean_object* v___x_1759_; lean_object* v___x_1760_; lean_object* v___x_1761_; lean_object* v___x_1762_; lean_object* v___x_1763_; lean_object* v___x_1764_; lean_object* v___x_1765_; lean_object* v___x_1766_; size_t v_sz_1767_; size_t v___x_1768_; lean_object* v___x_1769_; 
lean_inc_ref(v_lctx_1757_);
v___x_1758_ = lean_local_ctx_find(v_lctx_1757_, v_fv_1755_);
v___x_1759_ = l_Lean_LocalContext_findFromUserName_x3f(v_lctx_1757_, v_name_1756_);
lean_dec_ref(v_lctx_1757_);
v___x_1760_ = lean_unsigned_to_nat(2u);
v___x_1761_ = lean_mk_empty_array_with_capacity(v___x_1760_);
v___x_1762_ = lean_array_push(v___x_1761_, v___x_1758_);
v___x_1763_ = lean_array_push(v___x_1762_, v___x_1759_);
v___x_1764_ = lean_unsigned_to_nat(0u);
v___x_1765_ = lean_array_get_size(v___x_1763_);
v___x_1766_ = lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_getFVarIdCandidates_spec__0(v___x_1763_, v___x_1764_, v___x_1765_);
lean_dec_ref(v___x_1763_);
v_sz_1767_ = lean_array_size(v___x_1766_);
v___x_1768_ = ((size_t)0ULL);
v___x_1769_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_getFVarIdCandidates_spec__1(v_sz_1767_, v___x_1768_, v___x_1766_);
return v___x_1769_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_getFVarIdCandidates___boxed(lean_object* v_fv_1770_, lean_object* v_name_1771_, lean_object* v_lctx_1772_){
_start:
{
lean_object* v_res_1773_; 
v_res_1773_ = lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_getFVarIdCandidates(v_fv_1770_, v_name_1771_, v_lctx_1772_);
lean_dec(v_name_1771_);
return v_res_1773_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_persistFVars(lean_object* v_fv_1774_, lean_object* v_before_1775_, lean_object* v_after_1776_){
_start:
{
lean_object* v___y_1778_; lean_object* v___x_1786_; 
lean_inc(v_fv_1774_);
v___x_1786_ = lean_local_ctx_find(v_before_1775_, v_fv_1774_);
if (lean_obj_tag(v___x_1786_) == 0)
{
lean_object* v___x_1787_; 
v___x_1787_ = l_Lean_instInhabitedLocalDecl_default;
v___y_1778_ = v___x_1787_;
goto v___jp_1777_;
}
else
{
lean_object* v_val_1788_; 
v_val_1788_ = lean_ctor_get(v___x_1786_, 0);
lean_inc(v_val_1788_);
lean_dec_ref_known(v___x_1786_, 1);
v___y_1778_ = v_val_1788_;
goto v___jp_1777_;
}
v___jp_1777_:
{
lean_object* v___x_1779_; lean_object* v___x_1780_; lean_object* v___x_1781_; lean_object* v___x_1782_; uint8_t v___x_1783_; 
v___x_1779_ = l_Lean_LocalDecl_userName(v___y_1778_);
lean_dec_ref(v___y_1778_);
v___x_1780_ = lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_getFVarIdCandidates(v_fv_1774_, v___x_1779_, v_after_1776_);
lean_dec(v___x_1779_);
v___x_1781_ = lean_unsigned_to_nat(0u);
v___x_1782_ = lean_array_get_size(v___x_1780_);
v___x_1783_ = lean_nat_dec_lt(v___x_1781_, v___x_1782_);
if (v___x_1783_ == 0)
{
lean_object* v___x_1784_; 
lean_dec_ref(v___x_1780_);
v___x_1784_ = lean_box(0);
return v___x_1784_;
}
else
{
lean_object* v___x_1785_; 
v___x_1785_ = lean_array_fget(v___x_1780_, v___x_1781_);
lean_dec_ref(v___x_1780_);
return v___x_1785_;
}
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_elem___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__0(lean_object* v_a_1789_, lean_object* v_x_1790_){
_start:
{
if (lean_obj_tag(v_x_1790_) == 0)
{
uint8_t v___x_1791_; 
v___x_1791_ = 0;
return v___x_1791_;
}
else
{
lean_object* v_head_1792_; lean_object* v_tail_1793_; uint8_t v___x_1794_; 
v_head_1792_ = lean_ctor_get(v_x_1790_, 0);
v_tail_1793_ = lean_ctor_get(v_x_1790_, 1);
v___x_1794_ = l_Lean_instBEqMVarId_beq(v_a_1789_, v_head_1792_);
if (v___x_1794_ == 0)
{
v_x_1790_ = v_tail_1793_;
goto _start;
}
else
{
return v___x_1794_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_elem___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__0___boxed(lean_object* v_a_1796_, lean_object* v_x_1797_){
_start:
{
uint8_t v_res_1798_; lean_object* v_r_1799_; 
v_res_1798_ = lp_mathlib_List_elem___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__0(v_a_1796_, v_x_1797_);
lean_dec(v_x_1797_);
lean_dec(v_a_1796_);
v_r_1799_ = lean_box(v_res_1798_);
return v_r_1799_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__3(lean_object* v_mvs0_1800_, lean_object* v_as_1801_, size_t v_sz_1802_, size_t v_i_1803_, lean_object* v_b_1804_){
_start:
{
lean_object* v_a_1806_; uint8_t v___x_1810_; 
v___x_1810_ = lean_usize_dec_lt(v_i_1803_, v_sz_1802_);
if (v___x_1810_ == 0)
{
return v_b_1804_;
}
else
{
lean_object* v_a_1811_; lean_object* v_snd_1812_; lean_object* v_fst_1813_; lean_object* v_snd_1814_; lean_object* v___x_1816_; uint8_t v_isShared_1817_; uint8_t v_isSharedCheck_1827_; 
v_a_1811_ = lean_array_uget_borrowed(v_as_1801_, v_i_1803_);
v_snd_1812_ = lean_ctor_get(v_a_1811_, 1);
v_fst_1813_ = lean_ctor_get(v_b_1804_, 0);
v_snd_1814_ = lean_ctor_get(v_b_1804_, 1);
v_isSharedCheck_1827_ = !lean_is_exclusive(v_b_1804_);
if (v_isSharedCheck_1827_ == 0)
{
v___x_1816_ = v_b_1804_;
v_isShared_1817_ = v_isSharedCheck_1827_;
goto v_resetjp_1815_;
}
else
{
lean_inc(v_snd_1814_);
lean_inc(v_fst_1813_);
lean_dec(v_b_1804_);
v___x_1816_ = lean_box(0);
v_isShared_1817_ = v_isSharedCheck_1827_;
goto v_resetjp_1815_;
}
v_resetjp_1815_:
{
uint8_t v___x_1818_; 
v___x_1818_ = lp_mathlib_List_elem___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__0(v_snd_1812_, v_mvs0_1800_);
if (v___x_1818_ == 0)
{
lean_object* v___x_1819_; lean_object* v___x_1821_; 
lean_inc(v_a_1811_);
v___x_1819_ = lean_array_push(v_snd_1814_, v_a_1811_);
if (v_isShared_1817_ == 0)
{
lean_ctor_set(v___x_1816_, 1, v___x_1819_);
v___x_1821_ = v___x_1816_;
goto v_reusejp_1820_;
}
else
{
lean_object* v_reuseFailAlloc_1822_; 
v_reuseFailAlloc_1822_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1822_, 0, v_fst_1813_);
lean_ctor_set(v_reuseFailAlloc_1822_, 1, v___x_1819_);
v___x_1821_ = v_reuseFailAlloc_1822_;
goto v_reusejp_1820_;
}
v_reusejp_1820_:
{
v_a_1806_ = v___x_1821_;
goto v___jp_1805_;
}
}
else
{
lean_object* v___x_1823_; lean_object* v___x_1825_; 
lean_inc(v_a_1811_);
v___x_1823_ = lean_array_push(v_fst_1813_, v_a_1811_);
if (v_isShared_1817_ == 0)
{
lean_ctor_set(v___x_1816_, 0, v___x_1823_);
v___x_1825_ = v___x_1816_;
goto v_reusejp_1824_;
}
else
{
lean_object* v_reuseFailAlloc_1826_; 
v_reuseFailAlloc_1826_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1826_, 0, v___x_1823_);
lean_ctor_set(v_reuseFailAlloc_1826_, 1, v_snd_1814_);
v___x_1825_ = v_reuseFailAlloc_1826_;
goto v_reusejp_1824_;
}
v_reusejp_1824_:
{
v_a_1806_ = v___x_1825_;
goto v___jp_1805_;
}
}
}
}
v___jp_1805_:
{
size_t v___x_1807_; size_t v___x_1808_; 
v___x_1807_ = ((size_t)1ULL);
v___x_1808_ = lean_usize_add(v_i_1803_, v___x_1807_);
v_i_1803_ = v___x_1808_;
v_b_1804_ = v_a_1806_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__3___boxed(lean_object* v_mvs0_1828_, lean_object* v_as_1829_, lean_object* v_sz_1830_, lean_object* v_i_1831_, lean_object* v_b_1832_){
_start:
{
size_t v_sz_boxed_1833_; size_t v_i_boxed_1834_; lean_object* v_res_1835_; 
v_sz_boxed_1833_ = lean_unbox_usize(v_sz_1830_);
lean_dec(v_sz_1830_);
v_i_boxed_1834_ = lean_unbox_usize(v_i_1831_);
lean_dec(v_i_1831_);
v_res_1835_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__3(v_mvs0_1828_, v_as_1829_, v_sz_boxed_1833_, v_i_boxed_1834_, v_b_1832_);
lean_dec_ref(v_as_1829_);
lean_dec(v_mvs0_1828_);
return v_res_1835_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__1_spec__1_spec__2___redArg(lean_object* v_keys_1836_, lean_object* v_vals_1837_, lean_object* v_i_1838_, lean_object* v_k_1839_){
_start:
{
lean_object* v___x_1840_; uint8_t v___x_1841_; 
v___x_1840_ = lean_array_get_size(v_keys_1836_);
v___x_1841_ = lean_nat_dec_lt(v_i_1838_, v___x_1840_);
if (v___x_1841_ == 0)
{
lean_object* v___x_1842_; 
lean_dec(v_i_1838_);
v___x_1842_ = lean_box(0);
return v___x_1842_;
}
else
{
lean_object* v_k_x27_1843_; uint8_t v___x_1844_; 
v_k_x27_1843_ = lean_array_fget_borrowed(v_keys_1836_, v_i_1838_);
v___x_1844_ = l_Lean_instBEqMVarId_beq(v_k_1839_, v_k_x27_1843_);
if (v___x_1844_ == 0)
{
lean_object* v___x_1845_; lean_object* v___x_1846_; 
v___x_1845_ = lean_unsigned_to_nat(1u);
v___x_1846_ = lean_nat_add(v_i_1838_, v___x_1845_);
lean_dec(v_i_1838_);
v_i_1838_ = v___x_1846_;
goto _start;
}
else
{
lean_object* v___x_1848_; lean_object* v___x_1849_; 
v___x_1848_ = lean_array_fget_borrowed(v_vals_1837_, v_i_1838_);
lean_dec(v_i_1838_);
lean_inc(v___x_1848_);
v___x_1849_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1849_, 0, v___x_1848_);
return v___x_1849_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__1_spec__1_spec__2___redArg___boxed(lean_object* v_keys_1850_, lean_object* v_vals_1851_, lean_object* v_i_1852_, lean_object* v_k_1853_){
_start:
{
lean_object* v_res_1854_; 
v_res_1854_ = lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__1_spec__1_spec__2___redArg(v_keys_1850_, v_vals_1851_, v_i_1852_, v_k_1853_);
lean_dec(v_k_1853_);
lean_dec_ref(v_vals_1851_);
lean_dec_ref(v_keys_1850_);
return v_res_1854_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__1_spec__1___redArg(lean_object* v_x_1855_, size_t v_x_1856_, lean_object* v_x_1857_){
_start:
{
if (lean_obj_tag(v_x_1855_) == 0)
{
lean_object* v_es_1858_; lean_object* v___x_1859_; size_t v___x_1860_; size_t v___x_1861_; lean_object* v_j_1862_; lean_object* v___x_1863_; 
v_es_1858_ = lean_ctor_get(v_x_1855_, 0);
v___x_1859_ = lean_box(2);
v___x_1860_ = ((size_t)31ULL);
v___x_1861_ = lean_usize_land(v_x_1856_, v___x_1860_);
v_j_1862_ = lean_usize_to_nat(v___x_1861_);
v___x_1863_ = lean_array_get_borrowed(v___x_1859_, v_es_1858_, v_j_1862_);
lean_dec(v_j_1862_);
switch(lean_obj_tag(v___x_1863_))
{
case 0:
{
lean_object* v_key_1864_; lean_object* v_val_1865_; uint8_t v___x_1866_; 
v_key_1864_ = lean_ctor_get(v___x_1863_, 0);
v_val_1865_ = lean_ctor_get(v___x_1863_, 1);
v___x_1866_ = l_Lean_instBEqMVarId_beq(v_x_1857_, v_key_1864_);
if (v___x_1866_ == 0)
{
lean_object* v___x_1867_; 
v___x_1867_ = lean_box(0);
return v___x_1867_;
}
else
{
lean_object* v___x_1868_; 
lean_inc(v_val_1865_);
v___x_1868_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1868_, 0, v_val_1865_);
return v___x_1868_;
}
}
case 1:
{
lean_object* v_node_1869_; size_t v___x_1870_; size_t v___x_1871_; 
v_node_1869_ = lean_ctor_get(v___x_1863_, 0);
v___x_1870_ = ((size_t)5ULL);
v___x_1871_ = lean_usize_shift_right(v_x_1856_, v___x_1870_);
v_x_1855_ = v_node_1869_;
v_x_1856_ = v___x_1871_;
goto _start;
}
default: 
{
lean_object* v___x_1873_; 
v___x_1873_ = lean_box(0);
return v___x_1873_;
}
}
}
else
{
lean_object* v_ks_1874_; lean_object* v_vs_1875_; lean_object* v___x_1876_; lean_object* v___x_1877_; 
v_ks_1874_ = lean_ctor_get(v_x_1855_, 0);
v_vs_1875_ = lean_ctor_get(v_x_1855_, 1);
v___x_1876_ = lean_unsigned_to_nat(0u);
v___x_1877_ = lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__1_spec__1_spec__2___redArg(v_ks_1874_, v_vs_1875_, v___x_1876_, v_x_1857_);
return v___x_1877_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__1_spec__1___redArg___boxed(lean_object* v_x_1878_, lean_object* v_x_1879_, lean_object* v_x_1880_){
_start:
{
size_t v_x_1413__boxed_1881_; lean_object* v_res_1882_; 
v_x_1413__boxed_1881_ = lean_unbox_usize(v_x_1879_);
lean_dec(v_x_1879_);
v_res_1882_ = lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__1_spec__1___redArg(v_x_1878_, v_x_1413__boxed_1881_, v_x_1880_);
lean_dec(v_x_1880_);
lean_dec_ref(v_x_1878_);
return v_res_1882_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__1___redArg(lean_object* v_x_1883_, lean_object* v_x_1884_){
_start:
{
uint64_t v___x_1885_; size_t v___x_1886_; lean_object* v___x_1887_; 
v___x_1885_ = l_Lean_instHashableMVarId_hash(v_x_1884_);
v___x_1886_ = lean_uint64_to_usize(v___x_1885_);
v___x_1887_ = lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__1_spec__1___redArg(v_x_1883_, v___x_1886_, v_x_1884_);
return v___x_1887_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__1___redArg___boxed(lean_object* v_x_1888_, lean_object* v_x_1889_){
_start:
{
lean_object* v_res_1890_; 
v_res_1890_ = lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__1___redArg(v_x_1888_, v_x_1889_);
lean_dec(v_x_1889_);
lean_dec_ref(v_x_1888_);
return v_res_1890_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__2___redArg___lam__0(lean_object* v_b_1891_, lean_object* v_x_1892_){
_start:
{
lean_object* v___x_1893_; 
v___x_1893_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1893_, 0, v_b_1891_);
return v___x_1893_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__2___redArg(lean_object* v_ctx1_1895_, lean_object* v_val_1896_, lean_object* v_fst_1897_, lean_object* v_as_x27_1898_, lean_object* v_b_1899_){
_start:
{
if (lean_obj_tag(v_as_x27_1898_) == 0)
{
lean_dec(v_fst_1897_);
lean_dec_ref(v_val_1896_);
return v_b_1899_;
}
else
{
lean_object* v_head_1900_; lean_object* v_tail_1901_; lean_object* v_decls_1902_; lean_object* v___x_1903_; 
v_head_1900_ = lean_ctor_get(v_as_x27_1898_, 0);
v_tail_1901_ = lean_ctor_get(v_as_x27_1898_, 1);
v_decls_1902_ = lean_ctor_get(v_ctx1_1895_, 5);
v___x_1903_ = lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__1___redArg(v_decls_1902_, v_head_1900_);
if (lean_obj_tag(v___x_1903_) == 0)
{
lean_object* v___f_1904_; lean_object* v___x_1905_; lean_object* v___x_1906_; 
v___f_1904_ = lean_alloc_closure((void*)(lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__2___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1904_, 0, v_b_1899_);
v___x_1905_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__2___redArg___closed__0));
v___x_1906_ = lean_dbg_trace(v___x_1905_, v___f_1904_);
if (lean_obj_tag(v___x_1906_) == 0)
{
lean_object* v_a_1907_; 
lean_dec(v_fst_1897_);
lean_dec_ref(v_val_1896_);
v_a_1907_ = lean_ctor_get(v___x_1906_, 0);
lean_inc(v_a_1907_);
lean_dec_ref_known(v___x_1906_, 1);
return v_a_1907_;
}
else
{
lean_object* v_a_1908_; 
v_a_1908_ = lean_ctor_get(v___x_1906_, 0);
lean_inc(v_a_1908_);
lean_dec_ref_known(v___x_1906_, 1);
v_as_x27_1898_ = v_tail_1901_;
v_b_1899_ = v_a_1908_;
goto _start;
}
}
else
{
lean_object* v_val_1910_; lean_object* v_lctx_1911_; lean_object* v_lctx_1912_; lean_object* v___x_1913_; lean_object* v___x_1914_; lean_object* v___x_1915_; 
v_val_1910_ = lean_ctor_get(v___x_1903_, 0);
lean_inc(v_val_1910_);
lean_dec_ref_known(v___x_1903_, 1);
v_lctx_1911_ = lean_ctor_get(v_val_1896_, 1);
v_lctx_1912_ = lean_ctor_get(v_val_1910_, 1);
lean_inc_ref(v_lctx_1912_);
lean_dec(v_val_1910_);
lean_inc_ref(v_lctx_1911_);
lean_inc(v_fst_1897_);
v___x_1913_ = lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_persistFVars(v_fst_1897_, v_lctx_1911_, v_lctx_1912_);
lean_inc(v_head_1900_);
v___x_1914_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1914_, 0, v___x_1913_);
lean_ctor_set(v___x_1914_, 1, v_head_1900_);
v___x_1915_ = lean_array_push(v_b_1899_, v___x_1914_);
v_as_x27_1898_ = v_tail_1901_;
v_b_1899_ = v___x_1915_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__2___redArg___boxed(lean_object* v_ctx1_1917_, lean_object* v_val_1918_, lean_object* v_fst_1919_, lean_object* v_as_x27_1920_, lean_object* v_b_1921_){
_start:
{
lean_object* v_res_1922_; 
v_res_1922_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__2___redArg(v_ctx1_1917_, v_val_1918_, v_fst_1919_, v_as_x27_1920_, v_b_1921_);
lean_dec(v_as_x27_1920_);
lean_dec_ref(v_ctx1_1917_);
return v_res_1922_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__4(lean_object* v_ctx0_1923_, lean_object* v_ctx1_1924_, lean_object* v_mvs1_1925_, lean_object* v_as_1926_, size_t v_sz_1927_, size_t v_i_1928_, lean_object* v_b_1929_){
_start:
{
lean_object* v_a_1931_; uint8_t v___x_1935_; 
v___x_1935_ = lean_usize_dec_lt(v_i_1928_, v_sz_1927_);
if (v___x_1935_ == 0)
{
return v_b_1929_;
}
else
{
lean_object* v_a_1936_; lean_object* v_fst_1937_; lean_object* v_snd_1938_; lean_object* v_decls_1939_; lean_object* v___x_1940_; 
v_a_1936_ = lean_array_uget_borrowed(v_as_1926_, v_i_1928_);
v_fst_1937_ = lean_ctor_get(v_a_1936_, 0);
v_snd_1938_ = lean_ctor_get(v_a_1936_, 1);
v_decls_1939_ = lean_ctor_get(v_ctx0_1923_, 5);
v___x_1940_ = lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__1___redArg(v_decls_1939_, v_snd_1938_);
if (lean_obj_tag(v___x_1940_) == 0)
{
lean_object* v___x_1941_; 
lean_inc(v_a_1936_);
v___x_1941_ = lean_array_push(v_b_1929_, v_a_1936_);
v_a_1931_ = v___x_1941_;
goto v___jp_1930_;
}
else
{
lean_object* v_val_1942_; lean_object* v___x_1943_; 
v_val_1942_ = lean_ctor_get(v___x_1940_, 0);
lean_inc(v_val_1942_);
lean_dec_ref_known(v___x_1940_, 1);
lean_inc(v_fst_1937_);
v___x_1943_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__2___redArg(v_ctx1_1924_, v_val_1942_, v_fst_1937_, v_mvs1_1925_, v_b_1929_);
v_a_1931_ = v___x_1943_;
goto v___jp_1930_;
}
}
v___jp_1930_:
{
size_t v___x_1932_; size_t v___x_1933_; 
v___x_1932_ = ((size_t)1ULL);
v___x_1933_ = lean_usize_add(v_i_1928_, v___x_1932_);
v_i_1928_ = v___x_1933_;
v_b_1929_ = v_a_1931_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__4___boxed(lean_object* v_ctx0_1944_, lean_object* v_ctx1_1945_, lean_object* v_mvs1_1946_, lean_object* v_as_1947_, lean_object* v_sz_1948_, lean_object* v_i_1949_, lean_object* v_b_1950_){
_start:
{
size_t v_sz_boxed_1951_; size_t v_i_boxed_1952_; lean_object* v_res_1953_; 
v_sz_boxed_1951_ = lean_unbox_usize(v_sz_1948_);
lean_dec(v_sz_1948_);
v_i_boxed_1952_ = lean_unbox_usize(v_i_1949_);
lean_dec(v_i_1949_);
v_res_1953_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__4(v_ctx0_1944_, v_ctx1_1945_, v_mvs1_1946_, v_as_1947_, v_sz_boxed_1951_, v_i_boxed_1952_, v_b_1950_);
lean_dec_ref(v_as_1947_);
lean_dec(v_mvs1_1946_);
lean_dec_ref(v_ctx1_1945_);
lean_dec_ref(v_ctx0_1944_);
return v_res_1953_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist(lean_object* v_fmvars_1956_, lean_object* v_mvs0_1957_, lean_object* v_mvs1_1958_, lean_object* v_ctx0_1959_, lean_object* v_ctx1_1960_){
_start:
{
lean_object* v_bs_1961_; lean_object* v___x_1962_; size_t v_sz_1963_; size_t v___x_1964_; lean_object* v___x_1965_; lean_object* v_fst_1966_; lean_object* v_snd_1967_; size_t v_sz_1968_; lean_object* v___x_1969_; lean_object* v___x_1970_; 
v_bs_1961_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_Stained_toFMVarId___closed__0));
v___x_1962_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist___closed__0));
v_sz_1963_ = lean_array_size(v_fmvars_1956_);
v___x_1964_ = ((size_t)0ULL);
v___x_1965_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__3(v_mvs0_1957_, v_fmvars_1956_, v_sz_1963_, v___x_1964_, v___x_1962_);
v_fst_1966_ = lean_ctor_get(v___x_1965_, 0);
lean_inc(v_fst_1966_);
v_snd_1967_ = lean_ctor_get(v___x_1965_, 1);
lean_inc(v_snd_1967_);
lean_dec_ref(v___x_1965_);
v_sz_1968_ = lean_array_size(v_fst_1966_);
v___x_1969_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__4(v_ctx0_1959_, v_ctx1_1960_, v_mvs1_1958_, v_fst_1966_, v_sz_1968_, v___x_1964_, v_bs_1961_);
lean_dec(v_fst_1966_);
v___x_1970_ = l_Array_append___redArg(v_snd_1967_, v___x_1969_);
lean_dec_ref(v___x_1969_);
return v___x_1970_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist___boxed(lean_object* v_fmvars_1971_, lean_object* v_mvs0_1972_, lean_object* v_mvs1_1973_, lean_object* v_ctx0_1974_, lean_object* v_ctx1_1975_){
_start:
{
lean_object* v_res_1976_; 
v_res_1976_ = lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist(v_fmvars_1971_, v_mvs0_1972_, v_mvs1_1973_, v_ctx0_1974_, v_ctx1_1975_);
lean_dec_ref(v_ctx1_1975_);
lean_dec_ref(v_ctx0_1974_);
lean_dec(v_mvs1_1973_);
lean_dec(v_mvs0_1972_);
lean_dec_ref(v_fmvars_1971_);
return v_res_1976_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__1(lean_object* v_00_u03b2_1977_, lean_object* v_x_1978_, lean_object* v_x_1979_){
_start:
{
lean_object* v___x_1980_; 
v___x_1980_ = lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__1___redArg(v_x_1978_, v_x_1979_);
return v___x_1980_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__1___boxed(lean_object* v_00_u03b2_1981_, lean_object* v_x_1982_, lean_object* v_x_1983_){
_start:
{
lean_object* v_res_1984_; 
v_res_1984_ = lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__1(v_00_u03b2_1981_, v_x_1982_, v_x_1983_);
lean_dec(v_x_1983_);
lean_dec_ref(v_x_1982_);
return v_res_1984_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__2(lean_object* v_ctx1_1985_, lean_object* v_val_1986_, lean_object* v_fst_1987_, lean_object* v_as_1988_, lean_object* v_as_x27_1989_, lean_object* v_b_1990_, lean_object* v_a_1991_){
_start:
{
lean_object* v___x_1992_; 
v___x_1992_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__2___redArg(v_ctx1_1985_, v_val_1986_, v_fst_1987_, v_as_x27_1989_, v_b_1990_);
return v___x_1992_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__2___boxed(lean_object* v_ctx1_1993_, lean_object* v_val_1994_, lean_object* v_fst_1995_, lean_object* v_as_1996_, lean_object* v_as_x27_1997_, lean_object* v_b_1998_, lean_object* v_a_1999_){
_start:
{
lean_object* v_res_2000_; 
v_res_2000_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__2(v_ctx1_1993_, v_val_1994_, v_fst_1995_, v_as_1996_, v_as_x27_1997_, v_b_1998_, v_a_1999_);
lean_dec(v_as_x27_1997_);
lean_dec(v_as_1996_);
lean_dec_ref(v_ctx1_1993_);
return v_res_2000_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__1_spec__1(lean_object* v_00_u03b2_2001_, lean_object* v_x_2002_, size_t v_x_2003_, lean_object* v_x_2004_){
_start:
{
lean_object* v___x_2005_; 
v___x_2005_ = lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__1_spec__1___redArg(v_x_2002_, v_x_2003_, v_x_2004_);
return v___x_2005_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__1_spec__1___boxed(lean_object* v_00_u03b2_2006_, lean_object* v_x_2007_, lean_object* v_x_2008_, lean_object* v_x_2009_){
_start:
{
size_t v_x_1570__boxed_2010_; lean_object* v_res_2011_; 
v_x_1570__boxed_2010_ = lean_unbox_usize(v_x_2008_);
lean_dec(v_x_2008_);
v_res_2011_ = lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__1_spec__1(v_00_u03b2_2006_, v_x_2007_, v_x_1570__boxed_2010_, v_x_2009_);
lean_dec(v_x_2009_);
lean_dec_ref(v_x_2007_);
return v_res_2011_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__1_spec__1_spec__2(lean_object* v_00_u03b2_2012_, lean_object* v_keys_2013_, lean_object* v_vals_2014_, lean_object* v_heq_2015_, lean_object* v_i_2016_, lean_object* v_k_2017_){
_start:
{
lean_object* v___x_2018_; 
v___x_2018_ = lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__1_spec__1_spec__2___redArg(v_keys_2013_, v_vals_2014_, v_i_2016_, v_k_2017_);
return v___x_2018_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__1_spec__1_spec__2___boxed(lean_object* v_00_u03b2_2019_, lean_object* v_keys_2020_, lean_object* v_vals_2021_, lean_object* v_heq_2022_, lean_object* v_i_2023_, lean_object* v_k_2024_){
_start:
{
lean_object* v_res_2025_; 
v_res_2025_ = lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__1_spec__1_spec__2(v_00_u03b2_2019_, v_keys_2020_, v_vals_2021_, v_heq_2022_, v_i_2023_, v_k_2024_);
lean_dec(v_k_2024_);
lean_dec_ref(v_vals_2021_);
lean_dec_ref(v_keys_2020_);
return v_res_2025_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_isEmpty___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion_spec__0___redArg(lean_object* v_x_2026_){
_start:
{
uint8_t v___x_2027_; 
v___x_2027_ = l_Lean_PersistentHashMap_Node_isEmpty___redArg(v_x_2026_);
return v___x_2027_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_isEmpty___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion_spec__0___redArg___boxed(lean_object* v_x_2028_){
_start:
{
uint8_t v_res_2029_; lean_object* v_r_2030_; 
v_res_2029_ = lp_mathlib_Lean_PersistentHashMap_isEmpty___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion_spec__0___redArg(v_x_2028_);
lean_dec_ref(v_x_2028_);
v_r_2030_ = lean_box(v_res_2029_);
return v_r_2030_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_isEmpty___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion_spec__0(lean_object* v_00_u03b2_2031_, lean_object* v_x_2032_){
_start:
{
uint8_t v___x_2033_; 
v___x_2033_ = l_Lean_PersistentHashMap_Node_isEmpty___redArg(v_x_2032_);
return v___x_2033_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_isEmpty___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion_spec__0___boxed(lean_object* v_00_u03b2_2034_, lean_object* v_x_2035_){
_start:
{
uint8_t v_res_2036_; lean_object* v_r_2037_; 
v_res_2036_ = lp_mathlib_Lean_PersistentHashMap_isEmpty___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion_spec__0(v_00_u03b2_2034_, v_x_2035_);
lean_dec_ref(v_x_2035_);
v_r_2037_ = lean_box(v_res_2036_);
return v_r_2037_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withMCtx___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion_spec__1___redArg(lean_object* v_mctx_2038_, lean_object* v_x_2039_, lean_object* v___y_2040_, lean_object* v___y_2041_, lean_object* v___y_2042_, lean_object* v___y_2043_){
_start:
{
lean_object* v___x_2045_; 
v___x_2045_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMCtxImp(lean_box(0), v_mctx_2038_, v_x_2039_, v___y_2040_, v___y_2041_, v___y_2042_, v___y_2043_);
if (lean_obj_tag(v___x_2045_) == 0)
{
lean_object* v_a_2046_; lean_object* v___x_2048_; uint8_t v_isShared_2049_; uint8_t v_isSharedCheck_2053_; 
v_a_2046_ = lean_ctor_get(v___x_2045_, 0);
v_isSharedCheck_2053_ = !lean_is_exclusive(v___x_2045_);
if (v_isSharedCheck_2053_ == 0)
{
v___x_2048_ = v___x_2045_;
v_isShared_2049_ = v_isSharedCheck_2053_;
goto v_resetjp_2047_;
}
else
{
lean_inc(v_a_2046_);
lean_dec(v___x_2045_);
v___x_2048_ = lean_box(0);
v_isShared_2049_ = v_isSharedCheck_2053_;
goto v_resetjp_2047_;
}
v_resetjp_2047_:
{
lean_object* v___x_2051_; 
if (v_isShared_2049_ == 0)
{
v___x_2051_ = v___x_2048_;
goto v_reusejp_2050_;
}
else
{
lean_object* v_reuseFailAlloc_2052_; 
v_reuseFailAlloc_2052_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2052_, 0, v_a_2046_);
v___x_2051_ = v_reuseFailAlloc_2052_;
goto v_reusejp_2050_;
}
v_reusejp_2050_:
{
return v___x_2051_;
}
}
}
else
{
lean_object* v_a_2054_; lean_object* v___x_2056_; uint8_t v_isShared_2057_; uint8_t v_isSharedCheck_2061_; 
v_a_2054_ = lean_ctor_get(v___x_2045_, 0);
v_isSharedCheck_2061_ = !lean_is_exclusive(v___x_2045_);
if (v_isSharedCheck_2061_ == 0)
{
v___x_2056_ = v___x_2045_;
v_isShared_2057_ = v_isSharedCheck_2061_;
goto v_resetjp_2055_;
}
else
{
lean_inc(v_a_2054_);
lean_dec(v___x_2045_);
v___x_2056_ = lean_box(0);
v_isShared_2057_ = v_isSharedCheck_2061_;
goto v_resetjp_2055_;
}
v_resetjp_2055_:
{
lean_object* v___x_2059_; 
if (v_isShared_2057_ == 0)
{
v___x_2059_ = v___x_2056_;
goto v_reusejp_2058_;
}
else
{
lean_object* v_reuseFailAlloc_2060_; 
v_reuseFailAlloc_2060_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2060_, 0, v_a_2054_);
v___x_2059_ = v_reuseFailAlloc_2060_;
goto v_reusejp_2058_;
}
v_reusejp_2058_:
{
return v___x_2059_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withMCtx___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion_spec__1___redArg___boxed(lean_object* v_mctx_2062_, lean_object* v_x_2063_, lean_object* v___y_2064_, lean_object* v___y_2065_, lean_object* v___y_2066_, lean_object* v___y_2067_, lean_object* v___y_2068_){
_start:
{
lean_object* v_res_2069_; 
v_res_2069_ = lp_mathlib_Lean_Meta_withMCtx___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion_spec__1___redArg(v_mctx_2062_, v_x_2063_, v___y_2064_, v___y_2065_, v___y_2066_, v___y_2067_);
lean_dec(v___y_2067_);
lean_dec_ref(v___y_2066_);
lean_dec(v___y_2065_);
lean_dec_ref(v___y_2064_);
return v_res_2069_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withMCtx___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion_spec__1(lean_object* v_00_u03b1_2070_, lean_object* v_mctx_2071_, lean_object* v_x_2072_, lean_object* v___y_2073_, lean_object* v___y_2074_, lean_object* v___y_2075_, lean_object* v___y_2076_){
_start:
{
lean_object* v___x_2078_; 
v___x_2078_ = lp_mathlib_Lean_Meta_withMCtx___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion_spec__1___redArg(v_mctx_2071_, v_x_2072_, v___y_2073_, v___y_2074_, v___y_2075_, v___y_2076_);
return v___x_2078_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withMCtx___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion_spec__1___boxed(lean_object* v_00_u03b1_2079_, lean_object* v_mctx_2080_, lean_object* v_x_2081_, lean_object* v___y_2082_, lean_object* v___y_2083_, lean_object* v___y_2084_, lean_object* v___y_2085_, lean_object* v___y_2086_){
_start:
{
lean_object* v_res_2087_; 
v_res_2087_ = lp_mathlib_Lean_Meta_withMCtx___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion_spec__1(v_00_u03b1_2079_, v_mctx_2080_, v_x_2081_, v___y_2082_, v___y_2083_, v___y_2084_, v___y_2085_);
lean_dec(v___y_2085_);
lean_dec_ref(v___y_2084_);
lean_dec(v___y_2083_);
lean_dec_ref(v___y_2082_);
return v_res_2087_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion___redArg___lam__0___closed__0(void){
_start:
{
lean_object* v___x_2088_; 
v___x_2088_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_2088_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion___redArg___lam__0___closed__1(void){
_start:
{
lean_object* v___x_2089_; lean_object* v___x_2090_; 
v___x_2089_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion___redArg___lam__0___closed__0, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion___redArg___lam__0___closed__0_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion___redArg___lam__0___closed__0);
v___x_2090_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2090_, 0, v___x_2089_);
return v___x_2090_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion___redArg___lam__0___closed__2(void){
_start:
{
lean_object* v___x_2091_; lean_object* v___x_2092_; lean_object* v___x_2093_; 
v___x_2091_ = lean_unsigned_to_nat(32u);
v___x_2092_ = lean_mk_empty_array_with_capacity(v___x_2091_);
v___x_2093_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2093_, 0, v___x_2092_);
return v___x_2093_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion___redArg___lam__0(lean_object* v___x_2094_, lean_object* v_val_2095_, uint8_t v___x_2096_, lean_object* v_stainStx_2097_, lean_object* v___y_2098_, lean_object* v___y_2099_, lean_object* v___y_2100_, lean_object* v___y_2101_){
_start:
{
lean_object* v___x_2103_; 
v___x_2103_ = l_Lean_Meta_Simp_Context_mkDefault___redArg(v___y_2098_, v___y_2100_, v___y_2101_);
if (lean_obj_tag(v___x_2103_) == 0)
{
lean_object* v_a_2104_; lean_object* v___x_2105_; 
v_a_2104_ = lean_ctor_get(v___x_2103_, 0);
lean_inc(v_a_2104_);
lean_dec_ref_known(v___x_2103_, 1);
v___x_2105_ = l_Lean_Meta_Simp_getSimprocs___redArg(v___y_2101_);
if (lean_obj_tag(v___x_2105_) == 0)
{
lean_object* v_a_2106_; lean_object* v___x_2107_; lean_object* v___x_2108_; lean_object* v___x_2109_; lean_object* v___x_2110_; lean_object* v___x_2111_; lean_object* v___x_2112_; lean_object* v___x_2113_; lean_object* v___x_2114_; lean_object* v___x_2115_; lean_object* v___x_2116_; size_t v___x_2117_; lean_object* v___x_2118_; lean_object* v___x_2119_; lean_object* v___x_2120_; lean_object* v___x_2121_; 
v_a_2106_ = lean_ctor_get(v___x_2105_, 0);
lean_inc(v_a_2106_);
lean_dec_ref_known(v___x_2105_, 1);
v___x_2107_ = lean_unsigned_to_nat(1u);
v___x_2108_ = lean_mk_empty_array_with_capacity(v___x_2107_);
v___x_2109_ = lean_array_push(v___x_2108_, v_a_2106_);
v___x_2110_ = lean_box(0);
v___x_2111_ = lean_mk_empty_array_with_capacity(v___x_2094_);
v___x_2112_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion___redArg___lam__0___closed__1, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion___redArg___lam__0___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion___redArg___lam__0___closed__1);
lean_inc_n(v___x_2094_, 2);
v___x_2113_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2113_, 0, v___x_2112_);
lean_ctor_set(v___x_2113_, 1, v___x_2094_);
v___x_2114_ = lean_unsigned_to_nat(32u);
v___x_2115_ = lean_mk_empty_array_with_capacity(v___x_2114_);
v___x_2116_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion___redArg___lam__0___closed__2, &lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion___redArg___lam__0___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion___redArg___lam__0___closed__2);
v___x_2117_ = ((size_t)5ULL);
v___x_2118_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_2118_, 0, v___x_2116_);
lean_ctor_set(v___x_2118_, 1, v___x_2115_);
lean_ctor_set(v___x_2118_, 2, v___x_2094_);
lean_ctor_set(v___x_2118_, 3, v___x_2094_);
lean_ctor_set_usize(v___x_2118_, 4, v___x_2117_);
v___x_2119_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_2119_, 0, v___x_2112_);
lean_ctor_set(v___x_2119_, 1, v___x_2112_);
lean_ctor_set(v___x_2119_, 2, v___x_2112_);
lean_ctor_set(v___x_2119_, 3, v___x_2118_);
v___x_2120_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2120_, 0, v___x_2113_);
lean_ctor_set(v___x_2120_, 1, v___x_2119_);
v___x_2121_ = l_Lean_Meta_simpGoal(v_val_2095_, v_a_2104_, v___x_2109_, v___x_2110_, v___x_2096_, v___x_2111_, v___x_2120_, v___y_2098_, v___y_2099_, v___y_2100_, v___y_2101_);
if (lean_obj_tag(v___x_2121_) == 0)
{
lean_object* v_a_2122_; lean_object* v___x_2124_; uint8_t v_isShared_2125_; uint8_t v_isSharedCheck_2151_; 
v_a_2122_ = lean_ctor_get(v___x_2121_, 0);
v_isSharedCheck_2151_ = !lean_is_exclusive(v___x_2121_);
if (v_isSharedCheck_2151_ == 0)
{
v___x_2124_ = v___x_2121_;
v_isShared_2125_ = v_isSharedCheck_2151_;
goto v_resetjp_2123_;
}
else
{
lean_inc(v_a_2122_);
lean_dec(v___x_2121_);
v___x_2124_ = lean_box(0);
v_isShared_2125_ = v_isSharedCheck_2151_;
goto v_resetjp_2123_;
}
v_resetjp_2123_:
{
lean_object* v_snd_2126_; lean_object* v_usedTheorems_2127_; lean_object* v_map_2128_; uint8_t v___x_2129_; 
v_snd_2126_ = lean_ctor_get(v_a_2122_, 1);
lean_inc(v_snd_2126_);
lean_dec(v_a_2122_);
v_usedTheorems_2127_ = lean_ctor_get(v_snd_2126_, 0);
lean_inc_ref(v_usedTheorems_2127_);
lean_dec(v_snd_2126_);
v_map_2128_ = lean_ctor_get(v_usedTheorems_2127_, 0);
v___x_2129_ = l_Lean_PersistentHashMap_Node_isEmpty___redArg(v_map_2128_);
if (v___x_2129_ == 0)
{
lean_object* v___x_2130_; 
lean_del_object(v___x_2124_);
v___x_2130_ = l_Lean_Elab_Tactic_mkSimpOnly(v_stainStx_2097_, v_usedTheorems_2127_, v___y_2098_, v___y_2099_, v___y_2100_, v___y_2101_);
lean_dec_ref(v_usedTheorems_2127_);
if (lean_obj_tag(v___x_2130_) == 0)
{
lean_object* v_a_2131_; lean_object* v___x_2133_; uint8_t v_isShared_2134_; uint8_t v_isSharedCheck_2139_; 
v_a_2131_ = lean_ctor_get(v___x_2130_, 0);
v_isSharedCheck_2139_ = !lean_is_exclusive(v___x_2130_);
if (v_isSharedCheck_2139_ == 0)
{
v___x_2133_ = v___x_2130_;
v_isShared_2134_ = v_isSharedCheck_2139_;
goto v_resetjp_2132_;
}
else
{
lean_inc(v_a_2131_);
lean_dec(v___x_2130_);
v___x_2133_ = lean_box(0);
v_isShared_2134_ = v_isSharedCheck_2139_;
goto v_resetjp_2132_;
}
v_resetjp_2132_:
{
lean_object* v___x_2135_; lean_object* v___x_2137_; 
v___x_2135_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2135_, 0, v_a_2131_);
if (v_isShared_2134_ == 0)
{
lean_ctor_set(v___x_2133_, 0, v___x_2135_);
v___x_2137_ = v___x_2133_;
goto v_reusejp_2136_;
}
else
{
lean_object* v_reuseFailAlloc_2138_; 
v_reuseFailAlloc_2138_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2138_, 0, v___x_2135_);
v___x_2137_ = v_reuseFailAlloc_2138_;
goto v_reusejp_2136_;
}
v_reusejp_2136_:
{
return v___x_2137_;
}
}
}
else
{
lean_object* v_a_2140_; lean_object* v___x_2142_; uint8_t v_isShared_2143_; uint8_t v_isSharedCheck_2147_; 
v_a_2140_ = lean_ctor_get(v___x_2130_, 0);
v_isSharedCheck_2147_ = !lean_is_exclusive(v___x_2130_);
if (v_isSharedCheck_2147_ == 0)
{
v___x_2142_ = v___x_2130_;
v_isShared_2143_ = v_isSharedCheck_2147_;
goto v_resetjp_2141_;
}
else
{
lean_inc(v_a_2140_);
lean_dec(v___x_2130_);
v___x_2142_ = lean_box(0);
v_isShared_2143_ = v_isSharedCheck_2147_;
goto v_resetjp_2141_;
}
v_resetjp_2141_:
{
lean_object* v___x_2145_; 
if (v_isShared_2143_ == 0)
{
v___x_2145_ = v___x_2142_;
goto v_reusejp_2144_;
}
else
{
lean_object* v_reuseFailAlloc_2146_; 
v_reuseFailAlloc_2146_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2146_, 0, v_a_2140_);
v___x_2145_ = v_reuseFailAlloc_2146_;
goto v_reusejp_2144_;
}
v_reusejp_2144_:
{
return v___x_2145_;
}
}
}
}
else
{
lean_object* v___x_2149_; 
lean_dec_ref(v_usedTheorems_2127_);
lean_dec(v_stainStx_2097_);
if (v_isShared_2125_ == 0)
{
lean_ctor_set(v___x_2124_, 0, v___x_2110_);
v___x_2149_ = v___x_2124_;
goto v_reusejp_2148_;
}
else
{
lean_object* v_reuseFailAlloc_2150_; 
v_reuseFailAlloc_2150_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2150_, 0, v___x_2110_);
v___x_2149_ = v_reuseFailAlloc_2150_;
goto v_reusejp_2148_;
}
v_reusejp_2148_:
{
return v___x_2149_;
}
}
}
}
else
{
lean_object* v_a_2152_; lean_object* v___x_2154_; uint8_t v_isShared_2155_; uint8_t v_isSharedCheck_2159_; 
lean_dec(v_stainStx_2097_);
v_a_2152_ = lean_ctor_get(v___x_2121_, 0);
v_isSharedCheck_2159_ = !lean_is_exclusive(v___x_2121_);
if (v_isSharedCheck_2159_ == 0)
{
v___x_2154_ = v___x_2121_;
v_isShared_2155_ = v_isSharedCheck_2159_;
goto v_resetjp_2153_;
}
else
{
lean_inc(v_a_2152_);
lean_dec(v___x_2121_);
v___x_2154_ = lean_box(0);
v_isShared_2155_ = v_isSharedCheck_2159_;
goto v_resetjp_2153_;
}
v_resetjp_2153_:
{
lean_object* v___x_2157_; 
if (v_isShared_2155_ == 0)
{
v___x_2157_ = v___x_2154_;
goto v_reusejp_2156_;
}
else
{
lean_object* v_reuseFailAlloc_2158_; 
v_reuseFailAlloc_2158_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2158_, 0, v_a_2152_);
v___x_2157_ = v_reuseFailAlloc_2158_;
goto v_reusejp_2156_;
}
v_reusejp_2156_:
{
return v___x_2157_;
}
}
}
}
else
{
lean_object* v_a_2160_; lean_object* v___x_2162_; uint8_t v_isShared_2163_; uint8_t v_isSharedCheck_2167_; 
lean_dec(v_a_2104_);
lean_dec(v_stainStx_2097_);
lean_dec(v_val_2095_);
lean_dec(v___x_2094_);
v_a_2160_ = lean_ctor_get(v___x_2105_, 0);
v_isSharedCheck_2167_ = !lean_is_exclusive(v___x_2105_);
if (v_isSharedCheck_2167_ == 0)
{
v___x_2162_ = v___x_2105_;
v_isShared_2163_ = v_isSharedCheck_2167_;
goto v_resetjp_2161_;
}
else
{
lean_inc(v_a_2160_);
lean_dec(v___x_2105_);
v___x_2162_ = lean_box(0);
v_isShared_2163_ = v_isSharedCheck_2167_;
goto v_resetjp_2161_;
}
v_resetjp_2161_:
{
lean_object* v___x_2165_; 
if (v_isShared_2163_ == 0)
{
v___x_2165_ = v___x_2162_;
goto v_reusejp_2164_;
}
else
{
lean_object* v_reuseFailAlloc_2166_; 
v_reuseFailAlloc_2166_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2166_, 0, v_a_2160_);
v___x_2165_ = v_reuseFailAlloc_2166_;
goto v_reusejp_2164_;
}
v_reusejp_2164_:
{
return v___x_2165_;
}
}
}
}
else
{
lean_object* v_a_2168_; lean_object* v___x_2170_; uint8_t v_isShared_2171_; uint8_t v_isSharedCheck_2175_; 
lean_dec(v_stainStx_2097_);
lean_dec(v_val_2095_);
lean_dec(v___x_2094_);
v_a_2168_ = lean_ctor_get(v___x_2103_, 0);
v_isSharedCheck_2175_ = !lean_is_exclusive(v___x_2103_);
if (v_isSharedCheck_2175_ == 0)
{
v___x_2170_ = v___x_2103_;
v_isShared_2171_ = v_isSharedCheck_2175_;
goto v_resetjp_2169_;
}
else
{
lean_inc(v_a_2168_);
lean_dec(v___x_2103_);
v___x_2170_ = lean_box(0);
v_isShared_2171_ = v_isSharedCheck_2175_;
goto v_resetjp_2169_;
}
v_resetjp_2169_:
{
lean_object* v___x_2173_; 
if (v_isShared_2171_ == 0)
{
v___x_2173_ = v___x_2170_;
goto v_reusejp_2172_;
}
else
{
lean_object* v_reuseFailAlloc_2174_; 
v_reuseFailAlloc_2174_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2174_, 0, v_a_2168_);
v___x_2173_ = v_reuseFailAlloc_2174_;
goto v_reusejp_2172_;
}
v_reusejp_2172_:
{
return v___x_2173_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion___redArg___lam__0___boxed(lean_object* v___x_2176_, lean_object* v_val_2177_, lean_object* v___x_2178_, lean_object* v_stainStx_2179_, lean_object* v___y_2180_, lean_object* v___y_2181_, lean_object* v___y_2182_, lean_object* v___y_2183_, lean_object* v___y_2184_){
_start:
{
uint8_t v___x_2787__boxed_2185_; lean_object* v_res_2186_; 
v___x_2787__boxed_2185_ = lean_unbox(v___x_2178_);
v_res_2186_ = lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion___redArg___lam__0(v___x_2176_, v_val_2177_, v___x_2787__boxed_2185_, v_stainStx_2179_, v___y_2180_, v___y_2181_, v___y_2182_, v___y_2183_);
lean_dec(v___y_2183_);
lean_dec_ref(v___y_2182_);
lean_dec(v___y_2181_);
lean_dec_ref(v___y_2180_);
return v_res_2186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion___redArg(lean_object* v_stainData_2187_, lean_object* v_stainStx_2188_, lean_object* v_a_2189_){
_start:
{
lean_object* v___y_2192_; uint8_t v___y_2193_; lean_object* v___x_2200_; 
lean_inc(v_stainStx_2188_);
v___x_2200_ = l_Lean_Syntax_getKind(v_stainStx_2188_);
if (lean_obj_tag(v___x_2200_) == 1)
{
lean_object* v_pre_2201_; 
v_pre_2201_ = lean_ctor_get(v___x_2200_, 0);
lean_inc(v_pre_2201_);
if (lean_obj_tag(v_pre_2201_) == 1)
{
lean_object* v_pre_2202_; 
v_pre_2202_ = lean_ctor_get(v_pre_2201_, 0);
lean_inc(v_pre_2202_);
if (lean_obj_tag(v_pre_2202_) == 1)
{
lean_object* v_pre_2203_; 
v_pre_2203_ = lean_ctor_get(v_pre_2202_, 0);
lean_inc(v_pre_2203_);
if (lean_obj_tag(v_pre_2203_) == 1)
{
lean_object* v_pre_2204_; 
v_pre_2204_ = lean_ctor_get(v_pre_2203_, 0);
if (lean_obj_tag(v_pre_2204_) == 0)
{
lean_object* v_str_2205_; lean_object* v_str_2206_; lean_object* v_str_2207_; lean_object* v_str_2208_; lean_object* v___x_2209_; uint8_t v___x_2210_; 
v_str_2205_ = lean_ctor_get(v___x_2200_, 1);
lean_inc_ref(v_str_2205_);
lean_dec_ref_known(v___x_2200_, 2);
v_str_2206_ = lean_ctor_get(v_pre_2201_, 1);
lean_inc_ref(v_str_2206_);
lean_dec_ref_known(v_pre_2201_, 2);
v_str_2207_ = lean_ctor_get(v_pre_2202_, 1);
lean_inc_ref(v_str_2207_);
lean_dec_ref_known(v_pre_2202_, 2);
v_str_2208_ = lean_ctor_get(v_pre_2203_, 1);
lean_inc_ref(v_str_2208_);
lean_dec_ref_known(v_pre_2203_, 2);
v___x_2209_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__0));
v___x_2210_ = lean_string_dec_eq(v_str_2208_, v___x_2209_);
lean_dec_ref(v_str_2208_);
if (v___x_2210_ == 0)
{
lean_dec_ref(v_str_2207_);
lean_dec_ref(v_str_2206_);
lean_dec_ref(v_str_2205_);
lean_dec(v_stainStx_2188_);
lean_dec_ref(v_stainData_2187_);
goto v___jp_2197_;
}
else
{
lean_object* v___x_2211_; uint8_t v___x_2212_; 
v___x_2211_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__1));
v___x_2212_ = lean_string_dec_eq(v_str_2207_, v___x_2211_);
lean_dec_ref(v_str_2207_);
if (v___x_2212_ == 0)
{
lean_dec_ref(v_str_2206_);
lean_dec_ref(v_str_2205_);
lean_dec(v_stainStx_2188_);
lean_dec_ref(v_stainData_2187_);
goto v___jp_2197_;
}
else
{
lean_object* v___x_2213_; uint8_t v___x_2214_; lean_object* v___y_2216_; 
v___x_2213_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2));
v___x_2214_ = lean_string_dec_eq(v_str_2206_, v___x_2213_);
lean_dec_ref(v_str_2206_);
if (v___x_2214_ == 0)
{
lean_dec_ref(v_str_2205_);
lean_dec(v_stainStx_2188_);
lean_dec_ref(v_stainData_2187_);
goto v___jp_2197_;
}
else
{
lean_object* v___x_2263_; uint8_t v___x_2264_; 
v___x_2263_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__3));
v___x_2264_ = lean_string_dec_eq(v_str_2205_, v___x_2263_);
if (v___x_2264_ == 0)
{
lean_object* v___x_2265_; uint8_t v___x_2266_; 
v___x_2265_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__4));
v___x_2266_ = lean_string_dec_eq(v_str_2205_, v___x_2265_);
lean_dec_ref(v_str_2205_);
if (v___x_2266_ == 0)
{
lean_dec(v_stainStx_2188_);
lean_dec_ref(v_stainData_2187_);
goto v___jp_2197_;
}
else
{
v___y_2216_ = v_a_2189_;
goto v___jp_2215_;
}
}
else
{
lean_dec_ref(v_str_2205_);
v___y_2216_ = v_a_2189_;
goto v___jp_2215_;
}
}
v___jp_2215_:
{
lean_object* v_ci_2217_; lean_object* v_mctx_2218_; lean_object* v_goals_2219_; lean_object* v___x_2220_; lean_object* v___x_2221_; 
v_ci_2217_ = lean_ctor_get(v_stainData_2187_, 2);
lean_inc_ref(v_ci_2217_);
v_mctx_2218_ = lean_ctor_get(v_stainData_2187_, 3);
lean_inc_ref(v_mctx_2218_);
v_goals_2219_ = lean_ctor_get(v_stainData_2187_, 4);
lean_inc(v_goals_2219_);
lean_dec_ref(v_stainData_2187_);
v___x_2220_ = lean_unsigned_to_nat(0u);
v___x_2221_ = l_List_get_x3fInternal___redArg(v_goals_2219_, v___x_2220_);
lean_dec(v_goals_2219_);
if (lean_obj_tag(v___x_2221_) == 1)
{
lean_object* v_val_2222_; lean_object* v___x_2224_; uint8_t v_isShared_2225_; uint8_t v_isSharedCheck_2260_; 
v_val_2222_ = lean_ctor_get(v___x_2221_, 0);
v_isSharedCheck_2260_ = !lean_is_exclusive(v___x_2221_);
if (v_isSharedCheck_2260_ == 0)
{
v___x_2224_ = v___x_2221_;
v_isShared_2225_ = v_isSharedCheck_2260_;
goto v_resetjp_2223_;
}
else
{
lean_inc(v_val_2222_);
lean_dec(v___x_2221_);
v___x_2224_ = lean_box(0);
v_isShared_2225_ = v_isSharedCheck_2260_;
goto v_resetjp_2223_;
}
v_resetjp_2223_:
{
lean_object* v_decls_2226_; lean_object* v___x_2227_; 
v_decls_2226_ = lean_ctor_get(v_mctx_2218_, 5);
v___x_2227_ = lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__1___redArg(v_decls_2226_, v_val_2222_);
if (lean_obj_tag(v___x_2227_) == 1)
{
lean_object* v_val_2228_; lean_object* v___x_2230_; uint8_t v_isShared_2231_; uint8_t v_isSharedCheck_2255_; 
lean_del_object(v___x_2224_);
v_val_2228_ = lean_ctor_get(v___x_2227_, 0);
v_isSharedCheck_2255_ = !lean_is_exclusive(v___x_2227_);
if (v_isSharedCheck_2255_ == 0)
{
v___x_2230_ = v___x_2227_;
v_isShared_2231_ = v_isSharedCheck_2255_;
goto v_resetjp_2229_;
}
else
{
lean_inc(v_val_2228_);
lean_dec(v___x_2227_);
v___x_2230_ = lean_box(0);
v_isShared_2231_ = v_isSharedCheck_2255_;
goto v_resetjp_2229_;
}
v_resetjp_2229_:
{
lean_object* v_lctx_2232_; lean_object* v___x_2233_; lean_object* v___f_2234_; lean_object* v___x_2235_; lean_object* v___x_2236_; 
v_lctx_2232_ = lean_ctor_get(v_val_2228_, 1);
lean_inc_ref(v_lctx_2232_);
lean_dec(v_val_2228_);
v___x_2233_ = lean_box(v___x_2214_);
v___f_2234_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion___redArg___lam__0___boxed), 9, 4);
lean_closure_set(v___f_2234_, 0, v___x_2220_);
lean_closure_set(v___f_2234_, 1, v_val_2222_);
lean_closure_set(v___f_2234_, 2, v___x_2233_);
lean_closure_set(v___f_2234_, 3, v_stainStx_2188_);
v___x_2235_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withMCtx___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion_spec__1___boxed), 8, 3);
lean_closure_set(v___x_2235_, 0, lean_box(0));
lean_closure_set(v___x_2235_, 1, v_mctx_2218_);
lean_closure_set(v___x_2235_, 2, v___f_2234_);
v___x_2236_ = l_Lean_Elab_ContextInfo_runMetaM___redArg(v_ci_2217_, v_lctx_2232_, v___x_2235_);
if (lean_obj_tag(v___x_2236_) == 0)
{
lean_object* v_a_2237_; lean_object* v___x_2239_; uint8_t v_isShared_2240_; uint8_t v_isSharedCheck_2244_; 
lean_del_object(v___x_2230_);
v_a_2237_ = lean_ctor_get(v___x_2236_, 0);
v_isSharedCheck_2244_ = !lean_is_exclusive(v___x_2236_);
if (v_isSharedCheck_2244_ == 0)
{
v___x_2239_ = v___x_2236_;
v_isShared_2240_ = v_isSharedCheck_2244_;
goto v_resetjp_2238_;
}
else
{
lean_inc(v_a_2237_);
lean_dec(v___x_2236_);
v___x_2239_ = lean_box(0);
v_isShared_2240_ = v_isSharedCheck_2244_;
goto v_resetjp_2238_;
}
v_resetjp_2238_:
{
lean_object* v___x_2242_; 
if (v_isShared_2240_ == 0)
{
v___x_2242_ = v___x_2239_;
goto v_reusejp_2241_;
}
else
{
lean_object* v_reuseFailAlloc_2243_; 
v_reuseFailAlloc_2243_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2243_, 0, v_a_2237_);
v___x_2242_ = v_reuseFailAlloc_2243_;
goto v_reusejp_2241_;
}
v_reusejp_2241_:
{
return v___x_2242_;
}
}
}
else
{
lean_object* v_a_2245_; lean_object* v_ref_2246_; lean_object* v___x_2247_; lean_object* v___x_2249_; 
v_a_2245_ = lean_ctor_get(v___x_2236_, 0);
lean_inc(v_a_2245_);
lean_dec_ref_known(v___x_2236_, 1);
v_ref_2246_ = lean_ctor_get(v___y_2216_, 5);
v___x_2247_ = lean_io_error_to_string(v_a_2245_);
if (v_isShared_2231_ == 0)
{
lean_ctor_set_tag(v___x_2230_, 3);
lean_ctor_set(v___x_2230_, 0, v___x_2247_);
v___x_2249_ = v___x_2230_;
goto v_reusejp_2248_;
}
else
{
lean_object* v_reuseFailAlloc_2254_; 
v_reuseFailAlloc_2254_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2254_, 0, v___x_2247_);
v___x_2249_ = v_reuseFailAlloc_2254_;
goto v_reusejp_2248_;
}
v_reusejp_2248_:
{
lean_object* v___x_2250_; lean_object* v___x_2251_; uint8_t v___x_2252_; 
v___x_2250_ = l_Lean_MessageData_ofFormat(v___x_2249_);
lean_inc(v_ref_2246_);
v___x_2251_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2251_, 0, v_ref_2246_);
lean_ctor_set(v___x_2251_, 1, v___x_2250_);
v___x_2252_ = l_Lean_Exception_isInterrupt(v___x_2251_);
if (v___x_2252_ == 0)
{
uint8_t v___x_2253_; 
lean_inc_ref(v___x_2251_);
v___x_2253_ = l_Lean_Exception_isRuntime(v___x_2251_);
v___y_2192_ = v___x_2251_;
v___y_2193_ = v___x_2253_;
goto v___jp_2191_;
}
else
{
v___y_2192_ = v___x_2251_;
v___y_2193_ = v___x_2252_;
goto v___jp_2191_;
}
}
}
}
}
else
{
lean_object* v___x_2256_; lean_object* v___x_2258_; 
lean_dec(v___x_2227_);
lean_dec(v_val_2222_);
lean_dec_ref(v_mctx_2218_);
lean_dec_ref(v_ci_2217_);
lean_dec(v_stainStx_2188_);
v___x_2256_ = lean_box(0);
if (v_isShared_2225_ == 0)
{
lean_ctor_set_tag(v___x_2224_, 0);
lean_ctor_set(v___x_2224_, 0, v___x_2256_);
v___x_2258_ = v___x_2224_;
goto v_reusejp_2257_;
}
else
{
lean_object* v_reuseFailAlloc_2259_; 
v_reuseFailAlloc_2259_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2259_, 0, v___x_2256_);
v___x_2258_ = v_reuseFailAlloc_2259_;
goto v_reusejp_2257_;
}
v_reusejp_2257_:
{
return v___x_2258_;
}
}
}
}
else
{
lean_object* v___x_2261_; lean_object* v___x_2262_; 
lean_dec(v___x_2221_);
lean_dec_ref(v_mctx_2218_);
lean_dec_ref(v_ci_2217_);
lean_dec(v_stainStx_2188_);
v___x_2261_ = lean_box(0);
v___x_2262_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2262_, 0, v___x_2261_);
return v___x_2262_;
}
}
}
}
}
else
{
lean_dec_ref_known(v_pre_2203_, 2);
lean_dec_ref_known(v_pre_2202_, 2);
lean_dec_ref_known(v_pre_2201_, 2);
lean_dec_ref_known(v___x_2200_, 2);
lean_dec(v_stainStx_2188_);
lean_dec_ref(v_stainData_2187_);
goto v___jp_2197_;
}
}
else
{
lean_dec(v_pre_2203_);
lean_dec_ref_known(v_pre_2202_, 2);
lean_dec_ref_known(v_pre_2201_, 2);
lean_dec_ref_known(v___x_2200_, 2);
lean_dec(v_stainStx_2188_);
lean_dec_ref(v_stainData_2187_);
goto v___jp_2197_;
}
}
else
{
lean_dec(v_pre_2202_);
lean_dec_ref_known(v_pre_2201_, 2);
lean_dec_ref_known(v___x_2200_, 2);
lean_dec(v_stainStx_2188_);
lean_dec_ref(v_stainData_2187_);
goto v___jp_2197_;
}
}
else
{
lean_dec(v_pre_2201_);
lean_dec_ref_known(v___x_2200_, 2);
lean_dec(v_stainStx_2188_);
lean_dec_ref(v_stainData_2187_);
goto v___jp_2197_;
}
}
else
{
lean_dec(v___x_2200_);
lean_dec(v_stainStx_2188_);
lean_dec_ref(v_stainData_2187_);
goto v___jp_2197_;
}
v___jp_2191_:
{
if (v___y_2193_ == 0)
{
lean_object* v___x_2194_; lean_object* v___x_2195_; 
lean_dec_ref(v___y_2192_);
v___x_2194_ = lean_box(0);
v___x_2195_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2195_, 0, v___x_2194_);
return v___x_2195_;
}
else
{
lean_object* v___x_2196_; 
v___x_2196_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2196_, 0, v___y_2192_);
return v___x_2196_;
}
}
v___jp_2197_:
{
lean_object* v___x_2198_; lean_object* v___x_2199_; 
v___x_2198_ = lean_box(0);
v___x_2199_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2199_, 0, v___x_2198_);
return v___x_2199_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion___redArg___boxed(lean_object* v_stainData_2267_, lean_object* v_stainStx_2268_, lean_object* v_a_2269_, lean_object* v_a_2270_){
_start:
{
lean_object* v_res_2271_; 
v_res_2271_ = lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion___redArg(v_stainData_2267_, v_stainStx_2268_, v_a_2269_);
lean_dec_ref(v_a_2269_);
return v_res_2271_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion(lean_object* v_stainData_2272_, lean_object* v_stainStx_2273_, lean_object* v_a_2274_, lean_object* v_a_2275_){
_start:
{
lean_object* v___x_2277_; 
v___x_2277_ = lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion___redArg(v_stainData_2272_, v_stainStx_2273_, v_a_2274_);
return v___x_2277_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion___boxed(lean_object* v_stainData_2278_, lean_object* v_stainStx_2279_, lean_object* v_a_2280_, lean_object* v_a_2281_, lean_object* v_a_2282_){
_start:
{
lean_object* v_res_2283_; 
v_res_2283_ = lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion(v_stainData_2278_, v_stainStx_2279_, v_a_2280_, v_a_2281_);
lean_dec(v_a_2281_);
lean_dec_ref(v_a_2280_);
return v_res_2283_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getInfoTrees___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__19___redArg(lean_object* v___y_2284_){
_start:
{
lean_object* v___x_2286_; lean_object* v_infoState_2287_; lean_object* v_trees_2288_; lean_object* v___x_2289_; 
v___x_2286_ = lean_st_ref_get(v___y_2284_);
v_infoState_2287_ = lean_ctor_get(v___x_2286_, 8);
lean_inc_ref(v_infoState_2287_);
lean_dec(v___x_2286_);
v_trees_2288_ = lean_ctor_get(v_infoState_2287_, 2);
lean_inc_ref(v_trees_2288_);
lean_dec_ref(v_infoState_2287_);
v___x_2289_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2289_, 0, v_trees_2288_);
return v___x_2289_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getInfoTrees___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__19___redArg___boxed(lean_object* v___y_2290_, lean_object* v___y_2291_){
_start:
{
lean_object* v_res_2292_; 
v_res_2292_ = lp_mathlib_Lean_Elab_getInfoTrees___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__19___redArg(v___y_2290_);
lean_dec(v___y_2290_);
return v_res_2292_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getInfoTrees___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__19(lean_object* v___y_2293_, lean_object* v___y_2294_){
_start:
{
lean_object* v___x_2296_; 
v___x_2296_ = lp_mathlib_Lean_Elab_getInfoTrees___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__19___redArg(v___y_2294_);
return v___x_2296_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getInfoTrees___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__19___boxed(lean_object* v___y_2297_, lean_object* v___y_2298_, lean_object* v___y_2299_){
_start:
{
lean_object* v_res_2300_; 
v_res_2300_ = lp_mathlib_Lean_Elab_getInfoTrees___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__19(v___y_2297_, v___y_2298_);
lean_dec(v___y_2298_);
lean_dec_ref(v___y_2297_);
return v_res_2300_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__20_spec__25(lean_object* v_as_2301_, size_t v_i_2302_, size_t v_stop_2303_, lean_object* v_b_2304_){
_start:
{
uint8_t v___x_2305_; 
v___x_2305_ = lean_usize_dec_eq(v_i_2302_, v_stop_2303_);
if (v___x_2305_ == 0)
{
lean_object* v___x_2306_; lean_object* v___x_2307_; lean_object* v___x_2308_; size_t v___x_2309_; size_t v___x_2310_; 
v___x_2306_ = lean_array_uget_borrowed(v_as_2301_, v_i_2302_);
lean_inc(v___x_2306_);
v___x_2307_ = lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_extractTacticData(v___x_2306_);
v___x_2308_ = l_Array_append___redArg(v_b_2304_, v___x_2307_);
lean_dec_ref(v___x_2307_);
v___x_2309_ = ((size_t)1ULL);
v___x_2310_ = lean_usize_add(v_i_2302_, v___x_2309_);
v_i_2302_ = v___x_2310_;
v_b_2304_ = v___x_2308_;
goto _start;
}
else
{
return v_b_2304_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__20_spec__25___boxed(lean_object* v_as_2312_, lean_object* v_i_2313_, lean_object* v_stop_2314_, lean_object* v_b_2315_){
_start:
{
size_t v_i_boxed_2316_; size_t v_stop_boxed_2317_; lean_object* v_res_2318_; 
v_i_boxed_2316_ = lean_unbox_usize(v_i_2313_);
lean_dec(v_i_2313_);
v_stop_boxed_2317_ = lean_unbox_usize(v_stop_2314_);
lean_dec(v_stop_2314_);
v_res_2318_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__20_spec__25(v_as_2312_, v_i_boxed_2316_, v_stop_boxed_2317_, v_b_2315_);
lean_dec_ref(v_as_2312_);
return v_res_2318_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__20_spec__26(lean_object* v_x_2319_, lean_object* v_x_2320_){
_start:
{
if (lean_obj_tag(v_x_2319_) == 0)
{
lean_object* v_cs_2321_; lean_object* v___x_2322_; lean_object* v___x_2323_; uint8_t v___x_2324_; 
v_cs_2321_ = lean_ctor_get(v_x_2319_, 0);
v___x_2322_ = lean_unsigned_to_nat(0u);
v___x_2323_ = lean_array_get_size(v_cs_2321_);
v___x_2324_ = lean_nat_dec_lt(v___x_2322_, v___x_2323_);
if (v___x_2324_ == 0)
{
return v_x_2320_;
}
else
{
uint8_t v___x_2325_; 
v___x_2325_ = lean_nat_dec_le(v___x_2323_, v___x_2323_);
if (v___x_2325_ == 0)
{
if (v___x_2324_ == 0)
{
return v_x_2320_;
}
else
{
size_t v___x_2326_; size_t v___x_2327_; lean_object* v___x_2328_; 
v___x_2326_ = ((size_t)0ULL);
v___x_2327_ = lean_usize_of_nat(v___x_2323_);
v___x_2328_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__20_spec__24_spec__26(v_cs_2321_, v___x_2326_, v___x_2327_, v_x_2320_);
return v___x_2328_;
}
}
else
{
size_t v___x_2329_; size_t v___x_2330_; lean_object* v___x_2331_; 
v___x_2329_ = ((size_t)0ULL);
v___x_2330_ = lean_usize_of_nat(v___x_2323_);
v___x_2331_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__20_spec__24_spec__26(v_cs_2321_, v___x_2329_, v___x_2330_, v_x_2320_);
return v___x_2331_;
}
}
}
else
{
lean_object* v_vs_2332_; lean_object* v___x_2333_; lean_object* v___x_2334_; uint8_t v___x_2335_; 
v_vs_2332_ = lean_ctor_get(v_x_2319_, 0);
v___x_2333_ = lean_unsigned_to_nat(0u);
v___x_2334_ = lean_array_get_size(v_vs_2332_);
v___x_2335_ = lean_nat_dec_lt(v___x_2333_, v___x_2334_);
if (v___x_2335_ == 0)
{
return v_x_2320_;
}
else
{
uint8_t v___x_2336_; 
v___x_2336_ = lean_nat_dec_le(v___x_2334_, v___x_2334_);
if (v___x_2336_ == 0)
{
if (v___x_2335_ == 0)
{
return v_x_2320_;
}
else
{
size_t v___x_2337_; size_t v___x_2338_; lean_object* v___x_2339_; 
v___x_2337_ = ((size_t)0ULL);
v___x_2338_ = lean_usize_of_nat(v___x_2334_);
v___x_2339_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__20_spec__25(v_vs_2332_, v___x_2337_, v___x_2338_, v_x_2320_);
return v___x_2339_;
}
}
else
{
size_t v___x_2340_; size_t v___x_2341_; lean_object* v___x_2342_; 
v___x_2340_ = ((size_t)0ULL);
v___x_2341_ = lean_usize_of_nat(v___x_2334_);
v___x_2342_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__20_spec__25(v_vs_2332_, v___x_2340_, v___x_2341_, v_x_2320_);
return v___x_2342_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__20_spec__24_spec__26(lean_object* v_as_2343_, size_t v_i_2344_, size_t v_stop_2345_, lean_object* v_b_2346_){
_start:
{
uint8_t v___x_2347_; 
v___x_2347_ = lean_usize_dec_eq(v_i_2344_, v_stop_2345_);
if (v___x_2347_ == 0)
{
lean_object* v___x_2348_; lean_object* v___x_2349_; size_t v___x_2350_; size_t v___x_2351_; 
v___x_2348_ = lean_array_uget_borrowed(v_as_2343_, v_i_2344_);
v___x_2349_ = lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__20_spec__26(v___x_2348_, v_b_2346_);
v___x_2350_ = ((size_t)1ULL);
v___x_2351_ = lean_usize_add(v_i_2344_, v___x_2350_);
v_i_2344_ = v___x_2351_;
v_b_2346_ = v___x_2349_;
goto _start;
}
else
{
return v_b_2346_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__20_spec__24_spec__26___boxed(lean_object* v_as_2353_, lean_object* v_i_2354_, lean_object* v_stop_2355_, lean_object* v_b_2356_){
_start:
{
size_t v_i_boxed_2357_; size_t v_stop_boxed_2358_; lean_object* v_res_2359_; 
v_i_boxed_2357_ = lean_unbox_usize(v_i_2354_);
lean_dec(v_i_2354_);
v_stop_boxed_2358_ = lean_unbox_usize(v_stop_2355_);
lean_dec(v_stop_2355_);
v_res_2359_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__20_spec__24_spec__26(v_as_2353_, v_i_boxed_2357_, v_stop_boxed_2358_, v_b_2356_);
lean_dec_ref(v_as_2353_);
return v_res_2359_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__20_spec__26___boxed(lean_object* v_x_2360_, lean_object* v_x_2361_){
_start:
{
lean_object* v_res_2362_; 
v_res_2362_ = lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__20_spec__26(v_x_2360_, v_x_2361_);
lean_dec_ref(v_x_2360_);
return v_res_2362_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__20_spec__24___closed__0(void){
_start:
{
lean_object* v___x_2363_; 
v___x_2363_ = l_Lean_instInhabitedPersistentArrayNode_default(lean_box(0));
return v___x_2363_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__20_spec__24(lean_object* v_x_2364_, size_t v_x_2365_, size_t v_x_2366_, lean_object* v_x_2367_){
_start:
{
if (lean_obj_tag(v_x_2364_) == 0)
{
lean_object* v_cs_2368_; lean_object* v___x_2369_; size_t v___x_2370_; lean_object* v_j_2371_; lean_object* v___x_2372_; size_t v___x_2373_; size_t v___x_2374_; size_t v___x_2375_; size_t v___x_2376_; size_t v___x_2377_; size_t v___x_2378_; lean_object* v___x_2379_; lean_object* v___x_2380_; lean_object* v___x_2381_; lean_object* v___x_2382_; uint8_t v___x_2383_; 
v_cs_2368_ = lean_ctor_get(v_x_2364_, 0);
v___x_2369_ = lean_obj_once(&lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__20_spec__24___closed__0, &lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__20_spec__24___closed__0_once, _init_lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__20_spec__24___closed__0);
v___x_2370_ = lean_usize_shift_right(v_x_2365_, v_x_2366_);
v_j_2371_ = lean_usize_to_nat(v___x_2370_);
v___x_2372_ = lean_array_get_borrowed(v___x_2369_, v_cs_2368_, v_j_2371_);
v___x_2373_ = ((size_t)1ULL);
v___x_2374_ = lean_usize_shift_left(v___x_2373_, v_x_2366_);
v___x_2375_ = lean_usize_sub(v___x_2374_, v___x_2373_);
v___x_2376_ = lean_usize_land(v_x_2365_, v___x_2375_);
v___x_2377_ = ((size_t)5ULL);
v___x_2378_ = lean_usize_sub(v_x_2366_, v___x_2377_);
v___x_2379_ = lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__20_spec__24(v___x_2372_, v___x_2376_, v___x_2378_, v_x_2367_);
v___x_2380_ = lean_unsigned_to_nat(1u);
v___x_2381_ = lean_nat_add(v_j_2371_, v___x_2380_);
lean_dec(v_j_2371_);
v___x_2382_ = lean_array_get_size(v_cs_2368_);
v___x_2383_ = lean_nat_dec_lt(v___x_2381_, v___x_2382_);
if (v___x_2383_ == 0)
{
lean_dec(v___x_2381_);
return v___x_2379_;
}
else
{
uint8_t v___x_2384_; 
v___x_2384_ = lean_nat_dec_le(v___x_2382_, v___x_2382_);
if (v___x_2384_ == 0)
{
if (v___x_2383_ == 0)
{
lean_dec(v___x_2381_);
return v___x_2379_;
}
else
{
size_t v___x_2385_; size_t v___x_2386_; lean_object* v___x_2387_; 
v___x_2385_ = lean_usize_of_nat(v___x_2381_);
lean_dec(v___x_2381_);
v___x_2386_ = lean_usize_of_nat(v___x_2382_);
v___x_2387_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__20_spec__24_spec__26(v_cs_2368_, v___x_2385_, v___x_2386_, v___x_2379_);
return v___x_2387_;
}
}
else
{
size_t v___x_2388_; size_t v___x_2389_; lean_object* v___x_2390_; 
v___x_2388_ = lean_usize_of_nat(v___x_2381_);
lean_dec(v___x_2381_);
v___x_2389_ = lean_usize_of_nat(v___x_2382_);
v___x_2390_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__20_spec__24_spec__26(v_cs_2368_, v___x_2388_, v___x_2389_, v___x_2379_);
return v___x_2390_;
}
}
}
else
{
lean_object* v_vs_2391_; lean_object* v___x_2392_; lean_object* v___x_2393_; uint8_t v___x_2394_; 
v_vs_2391_ = lean_ctor_get(v_x_2364_, 0);
v___x_2392_ = lean_usize_to_nat(v_x_2365_);
v___x_2393_ = lean_array_get_size(v_vs_2391_);
v___x_2394_ = lean_nat_dec_lt(v___x_2392_, v___x_2393_);
if (v___x_2394_ == 0)
{
lean_dec(v___x_2392_);
return v_x_2367_;
}
else
{
uint8_t v___x_2395_; 
v___x_2395_ = lean_nat_dec_le(v___x_2393_, v___x_2393_);
if (v___x_2395_ == 0)
{
if (v___x_2394_ == 0)
{
lean_dec(v___x_2392_);
return v_x_2367_;
}
else
{
size_t v___x_2396_; size_t v___x_2397_; lean_object* v___x_2398_; 
v___x_2396_ = lean_usize_of_nat(v___x_2392_);
lean_dec(v___x_2392_);
v___x_2397_ = lean_usize_of_nat(v___x_2393_);
v___x_2398_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__20_spec__25(v_vs_2391_, v___x_2396_, v___x_2397_, v_x_2367_);
return v___x_2398_;
}
}
else
{
size_t v___x_2399_; size_t v___x_2400_; lean_object* v___x_2401_; 
v___x_2399_ = lean_usize_of_nat(v___x_2392_);
lean_dec(v___x_2392_);
v___x_2400_ = lean_usize_of_nat(v___x_2393_);
v___x_2401_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__20_spec__25(v_vs_2391_, v___x_2399_, v___x_2400_, v_x_2367_);
return v___x_2401_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__20_spec__24___boxed(lean_object* v_x_2402_, lean_object* v_x_2403_, lean_object* v_x_2404_, lean_object* v_x_2405_){
_start:
{
size_t v_x_33877__boxed_2406_; size_t v_x_33878__boxed_2407_; lean_object* v_res_2408_; 
v_x_33877__boxed_2406_ = lean_unbox_usize(v_x_2403_);
lean_dec(v_x_2403_);
v_x_33878__boxed_2407_ = lean_unbox_usize(v_x_2404_);
lean_dec(v_x_2404_);
v_res_2408_ = lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__20_spec__24(v_x_2402_, v_x_33877__boxed_2406_, v_x_33878__boxed_2407_, v_x_2405_);
lean_dec_ref(v_x_2402_);
return v_res_2408_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__20(lean_object* v_t_2409_, lean_object* v_init_2410_, lean_object* v_start_2411_){
_start:
{
lean_object* v___x_2412_; uint8_t v___x_2413_; 
v___x_2412_ = lean_unsigned_to_nat(0u);
v___x_2413_ = lean_nat_dec_eq(v_start_2411_, v___x_2412_);
if (v___x_2413_ == 0)
{
lean_object* v_root_2414_; lean_object* v_tail_2415_; size_t v_shift_2416_; lean_object* v_tailOff_2417_; uint8_t v___x_2418_; 
v_root_2414_ = lean_ctor_get(v_t_2409_, 0);
v_tail_2415_ = lean_ctor_get(v_t_2409_, 1);
v_shift_2416_ = lean_ctor_get_usize(v_t_2409_, 4);
v_tailOff_2417_ = lean_ctor_get(v_t_2409_, 3);
v___x_2418_ = lean_nat_dec_le(v_tailOff_2417_, v_start_2411_);
if (v___x_2418_ == 0)
{
size_t v___x_2419_; lean_object* v___x_2420_; lean_object* v___x_2421_; uint8_t v___x_2422_; 
v___x_2419_ = lean_usize_of_nat(v_start_2411_);
v___x_2420_ = lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__20_spec__24(v_root_2414_, v___x_2419_, v_shift_2416_, v_init_2410_);
v___x_2421_ = lean_array_get_size(v_tail_2415_);
v___x_2422_ = lean_nat_dec_lt(v___x_2412_, v___x_2421_);
if (v___x_2422_ == 0)
{
return v___x_2420_;
}
else
{
uint8_t v___x_2423_; 
v___x_2423_ = lean_nat_dec_le(v___x_2421_, v___x_2421_);
if (v___x_2423_ == 0)
{
if (v___x_2422_ == 0)
{
return v___x_2420_;
}
else
{
size_t v___x_2424_; size_t v___x_2425_; lean_object* v___x_2426_; 
v___x_2424_ = ((size_t)0ULL);
v___x_2425_ = lean_usize_of_nat(v___x_2421_);
v___x_2426_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__20_spec__25(v_tail_2415_, v___x_2424_, v___x_2425_, v___x_2420_);
return v___x_2426_;
}
}
else
{
size_t v___x_2427_; size_t v___x_2428_; lean_object* v___x_2429_; 
v___x_2427_ = ((size_t)0ULL);
v___x_2428_ = lean_usize_of_nat(v___x_2421_);
v___x_2429_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__20_spec__25(v_tail_2415_, v___x_2427_, v___x_2428_, v___x_2420_);
return v___x_2429_;
}
}
}
else
{
lean_object* v___x_2430_; lean_object* v___x_2431_; uint8_t v___x_2432_; 
v___x_2430_ = lean_nat_sub(v_start_2411_, v_tailOff_2417_);
v___x_2431_ = lean_array_get_size(v_tail_2415_);
v___x_2432_ = lean_nat_dec_lt(v___x_2430_, v___x_2431_);
if (v___x_2432_ == 0)
{
lean_dec(v___x_2430_);
return v_init_2410_;
}
else
{
uint8_t v___x_2433_; 
v___x_2433_ = lean_nat_dec_le(v___x_2431_, v___x_2431_);
if (v___x_2433_ == 0)
{
if (v___x_2432_ == 0)
{
lean_dec(v___x_2430_);
return v_init_2410_;
}
else
{
size_t v___x_2434_; size_t v___x_2435_; lean_object* v___x_2436_; 
v___x_2434_ = lean_usize_of_nat(v___x_2430_);
lean_dec(v___x_2430_);
v___x_2435_ = lean_usize_of_nat(v___x_2431_);
v___x_2436_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__20_spec__25(v_tail_2415_, v___x_2434_, v___x_2435_, v_init_2410_);
return v___x_2436_;
}
}
else
{
size_t v___x_2437_; size_t v___x_2438_; lean_object* v___x_2439_; 
v___x_2437_ = lean_usize_of_nat(v___x_2430_);
lean_dec(v___x_2430_);
v___x_2438_ = lean_usize_of_nat(v___x_2431_);
v___x_2439_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__20_spec__25(v_tail_2415_, v___x_2437_, v___x_2438_, v_init_2410_);
return v___x_2439_;
}
}
}
}
else
{
lean_object* v_root_2440_; lean_object* v_tail_2441_; lean_object* v___x_2442_; lean_object* v___x_2443_; uint8_t v___x_2444_; 
v_root_2440_ = lean_ctor_get(v_t_2409_, 0);
v_tail_2441_ = lean_ctor_get(v_t_2409_, 1);
v___x_2442_ = lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__20_spec__26(v_root_2440_, v_init_2410_);
v___x_2443_ = lean_array_get_size(v_tail_2441_);
v___x_2444_ = lean_nat_dec_lt(v___x_2412_, v___x_2443_);
if (v___x_2444_ == 0)
{
return v___x_2442_;
}
else
{
uint8_t v___x_2445_; 
v___x_2445_ = lean_nat_dec_le(v___x_2443_, v___x_2443_);
if (v___x_2445_ == 0)
{
if (v___x_2444_ == 0)
{
return v___x_2442_;
}
else
{
size_t v___x_2446_; size_t v___x_2447_; lean_object* v___x_2448_; 
v___x_2446_ = ((size_t)0ULL);
v___x_2447_ = lean_usize_of_nat(v___x_2443_);
v___x_2448_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__20_spec__25(v_tail_2441_, v___x_2446_, v___x_2447_, v___x_2442_);
return v___x_2448_;
}
}
else
{
size_t v___x_2449_; size_t v___x_2450_; lean_object* v___x_2451_; 
v___x_2449_ = ((size_t)0ULL);
v___x_2450_ = lean_usize_of_nat(v___x_2443_);
v___x_2451_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__20_spec__25(v_tail_2441_, v___x_2449_, v___x_2450_, v___x_2442_);
return v___x_2451_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__20___boxed(lean_object* v_t_2452_, lean_object* v_init_2453_, lean_object* v_start_2454_){
_start:
{
lean_object* v_res_2455_; 
v_res_2455_ = lp_mathlib_Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__20(v_t_2452_, v_init_2453_, v_start_2454_);
lean_dec(v_start_2454_);
lean_dec_ref(v_t_2452_);
return v_res_2455_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__0___redArg(lean_object* v_m_2456_, lean_object* v_a_2457_){
_start:
{
lean_object* v_buckets_2458_; lean_object* v___x_2459_; uint64_t v___y_2461_; 
v_buckets_2458_ = lean_ctor_get(v_m_2456_, 1);
v___x_2459_ = lean_array_get_size(v_buckets_2458_);
if (lean_obj_tag(v_a_2457_) == 0)
{
uint64_t v___x_2475_; 
v___x_2475_ = 1723ULL;
v___y_2461_ = v___x_2475_;
goto v___jp_2460_;
}
else
{
uint64_t v_hash_2476_; 
v_hash_2476_ = lean_ctor_get_uint64(v_a_2457_, sizeof(void*)*2);
v___y_2461_ = v_hash_2476_;
goto v___jp_2460_;
}
v___jp_2460_:
{
uint64_t v___x_2462_; uint64_t v___x_2463_; uint64_t v_fold_2464_; uint64_t v___x_2465_; uint64_t v___x_2466_; uint64_t v___x_2467_; size_t v___x_2468_; size_t v___x_2469_; size_t v___x_2470_; size_t v___x_2471_; size_t v___x_2472_; lean_object* v___x_2473_; uint8_t v___x_2474_; 
v___x_2462_ = 32ULL;
v___x_2463_ = lean_uint64_shift_right(v___y_2461_, v___x_2462_);
v_fold_2464_ = lean_uint64_xor(v___y_2461_, v___x_2463_);
v___x_2465_ = 16ULL;
v___x_2466_ = lean_uint64_shift_right(v_fold_2464_, v___x_2465_);
v___x_2467_ = lean_uint64_xor(v_fold_2464_, v___x_2466_);
v___x_2468_ = lean_uint64_to_usize(v___x_2467_);
v___x_2469_ = lean_usize_of_nat(v___x_2459_);
v___x_2470_ = ((size_t)1ULL);
v___x_2471_ = lean_usize_sub(v___x_2469_, v___x_2470_);
v___x_2472_ = lean_usize_land(v___x_2468_, v___x_2471_);
v___x_2473_ = lean_array_uget_borrowed(v_buckets_2458_, v___x_2472_);
v___x_2474_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers_spec__0_spec__0___redArg(v_a_2457_, v___x_2473_);
return v___x_2474_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__0___redArg___boxed(lean_object* v_m_2477_, lean_object* v_a_2478_){
_start:
{
uint8_t v_res_2479_; lean_object* v_r_2480_; 
v_res_2479_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__0___redArg(v_m_2477_, v_a_2478_);
lean_dec(v_a_2478_);
lean_dec_ref(v_m_2477_);
v_r_2480_ = lean_box(v_res_2479_);
return v_r_2480_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__8_spec__8___redArg(lean_object* v_a_2481_, lean_object* v_x_2482_){
_start:
{
if (lean_obj_tag(v_x_2482_) == 0)
{
uint8_t v___x_2483_; 
v___x_2483_ = 0;
return v___x_2483_;
}
else
{
lean_object* v_key_2484_; lean_object* v_tail_2485_; uint8_t v___y_2487_; lean_object* v_fst_2489_; lean_object* v_snd_2490_; lean_object* v_fst_2491_; lean_object* v_snd_2492_; uint8_t v___x_2493_; 
v_key_2484_ = lean_ctor_get(v_x_2482_, 0);
v_tail_2485_ = lean_ctor_get(v_x_2482_, 2);
v_fst_2489_ = lean_ctor_get(v_key_2484_, 0);
v_snd_2490_ = lean_ctor_get(v_key_2484_, 1);
v_fst_2491_ = lean_ctor_get(v_a_2481_, 0);
v_snd_2492_ = lean_ctor_get(v_a_2481_, 1);
v___x_2493_ = l_Lean_instBEqFVarId_beq(v_fst_2489_, v_fst_2491_);
if (v___x_2493_ == 0)
{
v___y_2487_ = v___x_2493_;
goto v___jp_2486_;
}
else
{
uint8_t v___x_2494_; 
v___x_2494_ = l_Lean_instBEqMVarId_beq(v_snd_2490_, v_snd_2492_);
v___y_2487_ = v___x_2494_;
goto v___jp_2486_;
}
v___jp_2486_:
{
if (v___y_2487_ == 0)
{
v_x_2482_ = v_tail_2485_;
goto _start;
}
else
{
return v___y_2487_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__8_spec__8___redArg___boxed(lean_object* v_a_2495_, lean_object* v_x_2496_){
_start:
{
uint8_t v_res_2497_; lean_object* v_r_2498_; 
v_res_2497_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__8_spec__8___redArg(v_a_2495_, v_x_2496_);
lean_dec(v_x_2496_);
lean_dec_ref(v_a_2495_);
v_r_2498_ = lean_box(v_res_2497_);
return v_r_2498_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__8___redArg(lean_object* v_m_2499_, lean_object* v_a_2500_){
_start:
{
lean_object* v_buckets_2501_; lean_object* v_fst_2502_; lean_object* v_snd_2503_; lean_object* v___x_2504_; uint64_t v___x_2505_; uint64_t v___x_2506_; uint64_t v___x_2507_; uint64_t v___x_2508_; uint64_t v___x_2509_; uint64_t v_fold_2510_; uint64_t v___x_2511_; uint64_t v___x_2512_; uint64_t v___x_2513_; size_t v___x_2514_; size_t v___x_2515_; size_t v___x_2516_; size_t v___x_2517_; size_t v___x_2518_; lean_object* v___x_2519_; uint8_t v___x_2520_; 
v_buckets_2501_ = lean_ctor_get(v_m_2499_, 1);
v_fst_2502_ = lean_ctor_get(v_a_2500_, 0);
v_snd_2503_ = lean_ctor_get(v_a_2500_, 1);
v___x_2504_ = lean_array_get_size(v_buckets_2501_);
v___x_2505_ = l_Lean_instHashableFVarId_hash(v_fst_2502_);
v___x_2506_ = l_Lean_instHashableMVarId_hash(v_snd_2503_);
v___x_2507_ = lean_uint64_mix_hash(v___x_2505_, v___x_2506_);
v___x_2508_ = 32ULL;
v___x_2509_ = lean_uint64_shift_right(v___x_2507_, v___x_2508_);
v_fold_2510_ = lean_uint64_xor(v___x_2507_, v___x_2509_);
v___x_2511_ = 16ULL;
v___x_2512_ = lean_uint64_shift_right(v_fold_2510_, v___x_2511_);
v___x_2513_ = lean_uint64_xor(v_fold_2510_, v___x_2512_);
v___x_2514_ = lean_uint64_to_usize(v___x_2513_);
v___x_2515_ = lean_usize_of_nat(v___x_2504_);
v___x_2516_ = ((size_t)1ULL);
v___x_2517_ = lean_usize_sub(v___x_2515_, v___x_2516_);
v___x_2518_ = lean_usize_land(v___x_2514_, v___x_2517_);
v___x_2519_ = lean_array_uget_borrowed(v_buckets_2501_, v___x_2518_);
v___x_2520_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__8_spec__8___redArg(v_a_2500_, v___x_2519_);
return v___x_2520_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__8___redArg___boxed(lean_object* v_m_2521_, lean_object* v_a_2522_){
_start:
{
uint8_t v_res_2523_; lean_object* v_r_2524_; 
v_res_2523_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__8___redArg(v_m_2521_, v_a_2522_);
lean_dec_ref(v_a_2522_);
lean_dec_ref(v_m_2521_);
v_r_2524_ = lean_box(v_res_2523_);
return v_r_2524_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__9_spec__10_spec__12_spec__28___redArg(lean_object* v_x_2525_, lean_object* v_x_2526_){
_start:
{
if (lean_obj_tag(v_x_2526_) == 0)
{
return v_x_2525_;
}
else
{
lean_object* v_key_2527_; lean_object* v_value_2528_; lean_object* v_tail_2529_; lean_object* v___x_2531_; uint8_t v_isShared_2532_; uint8_t v_isSharedCheck_2556_; 
v_key_2527_ = lean_ctor_get(v_x_2526_, 0);
v_value_2528_ = lean_ctor_get(v_x_2526_, 1);
v_tail_2529_ = lean_ctor_get(v_x_2526_, 2);
v_isSharedCheck_2556_ = !lean_is_exclusive(v_x_2526_);
if (v_isSharedCheck_2556_ == 0)
{
v___x_2531_ = v_x_2526_;
v_isShared_2532_ = v_isSharedCheck_2556_;
goto v_resetjp_2530_;
}
else
{
lean_inc(v_tail_2529_);
lean_inc(v_value_2528_);
lean_inc(v_key_2527_);
lean_dec(v_x_2526_);
v___x_2531_ = lean_box(0);
v_isShared_2532_ = v_isSharedCheck_2556_;
goto v_resetjp_2530_;
}
v_resetjp_2530_:
{
lean_object* v_fst_2533_; lean_object* v_snd_2534_; lean_object* v___x_2535_; uint64_t v___x_2536_; uint64_t v___x_2537_; uint64_t v___x_2538_; uint64_t v___x_2539_; uint64_t v___x_2540_; uint64_t v_fold_2541_; uint64_t v___x_2542_; uint64_t v___x_2543_; uint64_t v___x_2544_; size_t v___x_2545_; size_t v___x_2546_; size_t v___x_2547_; size_t v___x_2548_; size_t v___x_2549_; lean_object* v___x_2550_; lean_object* v___x_2552_; 
v_fst_2533_ = lean_ctor_get(v_key_2527_, 0);
v_snd_2534_ = lean_ctor_get(v_key_2527_, 1);
v___x_2535_ = lean_array_get_size(v_x_2525_);
v___x_2536_ = l_Lean_instHashableFVarId_hash(v_fst_2533_);
v___x_2537_ = l_Lean_instHashableMVarId_hash(v_snd_2534_);
v___x_2538_ = lean_uint64_mix_hash(v___x_2536_, v___x_2537_);
v___x_2539_ = 32ULL;
v___x_2540_ = lean_uint64_shift_right(v___x_2538_, v___x_2539_);
v_fold_2541_ = lean_uint64_xor(v___x_2538_, v___x_2540_);
v___x_2542_ = 16ULL;
v___x_2543_ = lean_uint64_shift_right(v_fold_2541_, v___x_2542_);
v___x_2544_ = lean_uint64_xor(v_fold_2541_, v___x_2543_);
v___x_2545_ = lean_uint64_to_usize(v___x_2544_);
v___x_2546_ = lean_usize_of_nat(v___x_2535_);
v___x_2547_ = ((size_t)1ULL);
v___x_2548_ = lean_usize_sub(v___x_2546_, v___x_2547_);
v___x_2549_ = lean_usize_land(v___x_2545_, v___x_2548_);
v___x_2550_ = lean_array_uget_borrowed(v_x_2525_, v___x_2549_);
lean_inc(v___x_2550_);
if (v_isShared_2532_ == 0)
{
lean_ctor_set(v___x_2531_, 2, v___x_2550_);
v___x_2552_ = v___x_2531_;
goto v_reusejp_2551_;
}
else
{
lean_object* v_reuseFailAlloc_2555_; 
v_reuseFailAlloc_2555_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2555_, 0, v_key_2527_);
lean_ctor_set(v_reuseFailAlloc_2555_, 1, v_value_2528_);
lean_ctor_set(v_reuseFailAlloc_2555_, 2, v___x_2550_);
v___x_2552_ = v_reuseFailAlloc_2555_;
goto v_reusejp_2551_;
}
v_reusejp_2551_:
{
lean_object* v___x_2553_; 
v___x_2553_ = lean_array_uset(v_x_2525_, v___x_2549_, v___x_2552_);
v_x_2525_ = v___x_2553_;
v_x_2526_ = v_tail_2529_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__9_spec__10_spec__12___redArg(lean_object* v_i_2557_, lean_object* v_source_2558_, lean_object* v_target_2559_){
_start:
{
lean_object* v___x_2560_; uint8_t v___x_2561_; 
v___x_2560_ = lean_array_get_size(v_source_2558_);
v___x_2561_ = lean_nat_dec_lt(v_i_2557_, v___x_2560_);
if (v___x_2561_ == 0)
{
lean_dec_ref(v_source_2558_);
lean_dec(v_i_2557_);
return v_target_2559_;
}
else
{
lean_object* v_es_2562_; lean_object* v___x_2563_; lean_object* v_source_2564_; lean_object* v_target_2565_; lean_object* v___x_2566_; lean_object* v___x_2567_; 
v_es_2562_ = lean_array_fget(v_source_2558_, v_i_2557_);
v___x_2563_ = lean_box(0);
v_source_2564_ = lean_array_fset(v_source_2558_, v_i_2557_, v___x_2563_);
v_target_2565_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__9_spec__10_spec__12_spec__28___redArg(v_target_2559_, v_es_2562_);
v___x_2566_ = lean_unsigned_to_nat(1u);
v___x_2567_ = lean_nat_add(v_i_2557_, v___x_2566_);
lean_dec(v_i_2557_);
v_i_2557_ = v___x_2567_;
v_source_2558_ = v_source_2564_;
v_target_2559_ = v_target_2565_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__9_spec__10___redArg(lean_object* v_data_2569_){
_start:
{
lean_object* v___x_2570_; lean_object* v___x_2571_; lean_object* v_nbuckets_2572_; lean_object* v___x_2573_; lean_object* v___x_2574_; lean_object* v___x_2575_; lean_object* v___x_2576_; 
v___x_2570_ = lean_array_get_size(v_data_2569_);
v___x_2571_ = lean_unsigned_to_nat(2u);
v_nbuckets_2572_ = lean_nat_mul(v___x_2570_, v___x_2571_);
v___x_2573_ = lean_unsigned_to_nat(0u);
v___x_2574_ = lean_box(0);
v___x_2575_ = lean_mk_array(v_nbuckets_2572_, v___x_2574_);
v___x_2576_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__9_spec__10_spec__12___redArg(v___x_2573_, v_data_2569_, v___x_2575_);
return v___x_2576_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__9___redArg(lean_object* v_m_2577_, lean_object* v_a_2578_, lean_object* v_b_2579_){
_start:
{
lean_object* v_size_2580_; lean_object* v_buckets_2581_; lean_object* v_fst_2582_; lean_object* v_snd_2583_; lean_object* v___x_2584_; uint64_t v___x_2585_; uint64_t v___x_2586_; uint64_t v___x_2587_; uint64_t v___x_2588_; uint64_t v___x_2589_; uint64_t v_fold_2590_; uint64_t v___x_2591_; uint64_t v___x_2592_; uint64_t v___x_2593_; size_t v___x_2594_; size_t v___x_2595_; size_t v___x_2596_; size_t v___x_2597_; size_t v___x_2598_; lean_object* v_bkt_2599_; uint8_t v___x_2600_; 
v_size_2580_ = lean_ctor_get(v_m_2577_, 0);
v_buckets_2581_ = lean_ctor_get(v_m_2577_, 1);
v_fst_2582_ = lean_ctor_get(v_a_2578_, 0);
v_snd_2583_ = lean_ctor_get(v_a_2578_, 1);
v___x_2584_ = lean_array_get_size(v_buckets_2581_);
v___x_2585_ = l_Lean_instHashableFVarId_hash(v_fst_2582_);
v___x_2586_ = l_Lean_instHashableMVarId_hash(v_snd_2583_);
v___x_2587_ = lean_uint64_mix_hash(v___x_2585_, v___x_2586_);
v___x_2588_ = 32ULL;
v___x_2589_ = lean_uint64_shift_right(v___x_2587_, v___x_2588_);
v_fold_2590_ = lean_uint64_xor(v___x_2587_, v___x_2589_);
v___x_2591_ = 16ULL;
v___x_2592_ = lean_uint64_shift_right(v_fold_2590_, v___x_2591_);
v___x_2593_ = lean_uint64_xor(v_fold_2590_, v___x_2592_);
v___x_2594_ = lean_uint64_to_usize(v___x_2593_);
v___x_2595_ = lean_usize_of_nat(v___x_2584_);
v___x_2596_ = ((size_t)1ULL);
v___x_2597_ = lean_usize_sub(v___x_2595_, v___x_2596_);
v___x_2598_ = lean_usize_land(v___x_2594_, v___x_2597_);
v_bkt_2599_ = lean_array_uget_borrowed(v_buckets_2581_, v___x_2598_);
v___x_2600_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__8_spec__8___redArg(v_a_2578_, v_bkt_2599_);
if (v___x_2600_ == 0)
{
lean_object* v___x_2602_; uint8_t v_isShared_2603_; uint8_t v_isSharedCheck_2621_; 
lean_inc_ref(v_buckets_2581_);
lean_inc(v_size_2580_);
v_isSharedCheck_2621_ = !lean_is_exclusive(v_m_2577_);
if (v_isSharedCheck_2621_ == 0)
{
lean_object* v_unused_2622_; lean_object* v_unused_2623_; 
v_unused_2622_ = lean_ctor_get(v_m_2577_, 1);
lean_dec(v_unused_2622_);
v_unused_2623_ = lean_ctor_get(v_m_2577_, 0);
lean_dec(v_unused_2623_);
v___x_2602_ = v_m_2577_;
v_isShared_2603_ = v_isSharedCheck_2621_;
goto v_resetjp_2601_;
}
else
{
lean_dec(v_m_2577_);
v___x_2602_ = lean_box(0);
v_isShared_2603_ = v_isSharedCheck_2621_;
goto v_resetjp_2601_;
}
v_resetjp_2601_:
{
lean_object* v___x_2604_; lean_object* v_size_x27_2605_; lean_object* v___x_2606_; lean_object* v_buckets_x27_2607_; lean_object* v___x_2608_; lean_object* v___x_2609_; lean_object* v___x_2610_; lean_object* v___x_2611_; lean_object* v___x_2612_; uint8_t v___x_2613_; 
v___x_2604_ = lean_unsigned_to_nat(1u);
v_size_x27_2605_ = lean_nat_add(v_size_2580_, v___x_2604_);
lean_dec(v_size_2580_);
lean_inc(v_bkt_2599_);
v___x_2606_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2606_, 0, v_a_2578_);
lean_ctor_set(v___x_2606_, 1, v_b_2579_);
lean_ctor_set(v___x_2606_, 2, v_bkt_2599_);
v_buckets_x27_2607_ = lean_array_uset(v_buckets_2581_, v___x_2598_, v___x_2606_);
v___x_2608_ = lean_unsigned_to_nat(4u);
v___x_2609_ = lean_nat_mul(v_size_x27_2605_, v___x_2608_);
v___x_2610_ = lean_unsigned_to_nat(3u);
v___x_2611_ = lean_nat_div(v___x_2609_, v___x_2610_);
lean_dec(v___x_2609_);
v___x_2612_ = lean_array_get_size(v_buckets_x27_2607_);
v___x_2613_ = lean_nat_dec_le(v___x_2611_, v___x_2612_);
lean_dec(v___x_2611_);
if (v___x_2613_ == 0)
{
lean_object* v_val_2614_; lean_object* v___x_2616_; 
v_val_2614_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__9_spec__10___redArg(v_buckets_x27_2607_);
if (v_isShared_2603_ == 0)
{
lean_ctor_set(v___x_2602_, 1, v_val_2614_);
lean_ctor_set(v___x_2602_, 0, v_size_x27_2605_);
v___x_2616_ = v___x_2602_;
goto v_reusejp_2615_;
}
else
{
lean_object* v_reuseFailAlloc_2617_; 
v_reuseFailAlloc_2617_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2617_, 0, v_size_x27_2605_);
lean_ctor_set(v_reuseFailAlloc_2617_, 1, v_val_2614_);
v___x_2616_ = v_reuseFailAlloc_2617_;
goto v_reusejp_2615_;
}
v_reusejp_2615_:
{
return v___x_2616_;
}
}
else
{
lean_object* v___x_2619_; 
if (v_isShared_2603_ == 0)
{
lean_ctor_set(v___x_2602_, 1, v_buckets_x27_2607_);
lean_ctor_set(v___x_2602_, 0, v_size_x27_2605_);
v___x_2619_ = v___x_2602_;
goto v_reusejp_2618_;
}
else
{
lean_object* v_reuseFailAlloc_2620_; 
v_reuseFailAlloc_2620_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2620_, 0, v_size_x27_2605_);
lean_ctor_set(v_reuseFailAlloc_2620_, 1, v_buckets_x27_2607_);
v___x_2619_ = v_reuseFailAlloc_2620_;
goto v_reusejp_2618_;
}
v_reusejp_2618_:
{
return v___x_2619_;
}
}
}
}
else
{
lean_dec(v_b_2579_);
lean_dec_ref(v_a_2578_);
return v_m_2577_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__10___redArg(lean_object* v_as_2624_, size_t v_sz_2625_, size_t v_i_2626_, lean_object* v_b_2627_){
_start:
{
lean_object* v_a_2630_; uint8_t v___x_2634_; 
v___x_2634_ = lean_usize_dec_lt(v_i_2626_, v_sz_2625_);
if (v___x_2634_ == 0)
{
lean_object* v___x_2635_; 
v___x_2635_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2635_, 0, v_b_2627_);
return v___x_2635_;
}
else
{
lean_object* v_a_2636_; uint8_t v___x_2637_; 
v_a_2636_ = lean_array_uget_borrowed(v_as_2624_, v_i_2626_);
v___x_2637_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__8___redArg(v_b_2627_, v_a_2636_);
if (v___x_2637_ == 0)
{
lean_object* v___x_2638_; lean_object* v___x_2639_; 
v___x_2638_ = lean_box(0);
lean_inc(v_a_2636_);
v___x_2639_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__9___redArg(v_b_2627_, v_a_2636_, v___x_2638_);
v_a_2630_ = v___x_2639_;
goto v___jp_2629_;
}
else
{
v_a_2630_ = v_b_2627_;
goto v___jp_2629_;
}
}
v___jp_2629_:
{
size_t v___x_2631_; size_t v___x_2632_; 
v___x_2631_ = ((size_t)1ULL);
v___x_2632_ = lean_usize_add(v_i_2626_, v___x_2631_);
v_i_2626_ = v___x_2632_;
v_b_2627_ = v_a_2630_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__10___redArg___boxed(lean_object* v_as_2640_, lean_object* v_sz_2641_, lean_object* v_i_2642_, lean_object* v_b_2643_, lean_object* v___y_2644_){
_start:
{
size_t v_sz_boxed_2645_; size_t v_i_boxed_2646_; lean_object* v_res_2647_; 
v_sz_boxed_2645_ = lean_unbox_usize(v_sz_2641_);
lean_dec(v_sz_2641_);
v_i_boxed_2646_ = lean_unbox_usize(v_i_2642_);
lean_dec(v_i_2642_);
v_res_2647_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__10___redArg(v_as_2640_, v_sz_boxed_2645_, v_i_boxed_2646_, v_b_2643_);
lean_dec_ref(v_as_2640_);
return v_res_2647_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__11(lean_object* v_a_2648_, lean_object* v___x_2649_, lean_object* v_a_2650_, lean_object* v_a_2651_, lean_object* v___y_2652_, lean_object* v___y_2653_){
_start:
{
if (lean_obj_tag(v_a_2650_) == 0)
{
lean_object* v___x_2655_; lean_object* v___x_2656_; 
lean_dec(v_a_2648_);
v___x_2655_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2655_, 0, v_a_2651_);
v___x_2656_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2656_, 0, v___x_2655_);
return v___x_2656_;
}
else
{
lean_object* v_key_2657_; lean_object* v_tail_2658_; lean_object* v___x_2659_; size_t v_sz_2660_; size_t v___x_2661_; lean_object* v___x_2662_; 
v_key_2657_ = lean_ctor_get(v_a_2650_, 0);
v_tail_2658_ = lean_ctor_get(v_a_2650_, 2);
lean_inc(v_a_2648_);
v___x_2659_ = lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_Stained_toFMVarId(v_a_2648_, v___x_2649_, v_key_2657_);
v_sz_2660_ = lean_array_size(v___x_2659_);
v___x_2661_ = ((size_t)0ULL);
v___x_2662_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__10___redArg(v___x_2659_, v_sz_2660_, v___x_2661_, v_a_2651_);
lean_dec_ref(v___x_2659_);
if (lean_obj_tag(v___x_2662_) == 0)
{
lean_object* v_a_2663_; 
v_a_2663_ = lean_ctor_get(v___x_2662_, 0);
lean_inc(v_a_2663_);
lean_dec_ref_known(v___x_2662_, 1);
v_a_2650_ = v_tail_2658_;
v_a_2651_ = v_a_2663_;
goto _start;
}
else
{
lean_object* v_a_2665_; lean_object* v___x_2667_; uint8_t v_isShared_2668_; uint8_t v_isSharedCheck_2672_; 
lean_dec(v_a_2648_);
v_a_2665_ = lean_ctor_get(v___x_2662_, 0);
v_isSharedCheck_2672_ = !lean_is_exclusive(v___x_2662_);
if (v_isSharedCheck_2672_ == 0)
{
v___x_2667_ = v___x_2662_;
v_isShared_2668_ = v_isSharedCheck_2672_;
goto v_resetjp_2666_;
}
else
{
lean_inc(v_a_2665_);
lean_dec(v___x_2662_);
v___x_2667_ = lean_box(0);
v_isShared_2668_ = v_isSharedCheck_2672_;
goto v_resetjp_2666_;
}
v_resetjp_2666_:
{
lean_object* v___x_2670_; 
if (v_isShared_2668_ == 0)
{
v___x_2670_ = v___x_2667_;
goto v_reusejp_2669_;
}
else
{
lean_object* v_reuseFailAlloc_2671_; 
v_reuseFailAlloc_2671_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2671_, 0, v_a_2665_);
v___x_2670_ = v_reuseFailAlloc_2671_;
goto v_reusejp_2669_;
}
v_reusejp_2669_:
{
return v___x_2670_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__11___boxed(lean_object* v_a_2673_, lean_object* v___x_2674_, lean_object* v_a_2675_, lean_object* v_a_2676_, lean_object* v___y_2677_, lean_object* v___y_2678_, lean_object* v___y_2679_){
_start:
{
lean_object* v_res_2680_; 
v_res_2680_ = lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__11(v_a_2673_, v___x_2674_, v_a_2675_, v_a_2676_, v___y_2677_, v___y_2678_);
lean_dec(v___y_2678_);
lean_dec_ref(v___y_2677_);
lean_dec(v_a_2675_);
lean_dec_ref(v___x_2674_);
return v_res_2680_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__12(lean_object* v_a_2681_, lean_object* v___x_2682_, lean_object* v_as_2683_, size_t v_sz_2684_, size_t v_i_2685_, lean_object* v_b_2686_, lean_object* v___y_2687_, lean_object* v___y_2688_){
_start:
{
uint8_t v___x_2690_; 
v___x_2690_ = lean_usize_dec_lt(v_i_2685_, v_sz_2684_);
if (v___x_2690_ == 0)
{
lean_object* v___x_2691_; 
lean_dec(v_a_2681_);
v___x_2691_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2691_, 0, v_b_2686_);
return v___x_2691_;
}
else
{
lean_object* v_a_2692_; lean_object* v___x_2693_; 
v_a_2692_ = lean_array_uget_borrowed(v_as_2683_, v_i_2685_);
lean_inc(v_a_2681_);
v___x_2693_ = lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__11(v_a_2681_, v___x_2682_, v_a_2692_, v_b_2686_, v___y_2687_, v___y_2688_);
if (lean_obj_tag(v___x_2693_) == 0)
{
lean_object* v_a_2694_; lean_object* v___x_2696_; uint8_t v_isShared_2697_; uint8_t v_isSharedCheck_2706_; 
v_a_2694_ = lean_ctor_get(v___x_2693_, 0);
v_isSharedCheck_2706_ = !lean_is_exclusive(v___x_2693_);
if (v_isSharedCheck_2706_ == 0)
{
v___x_2696_ = v___x_2693_;
v_isShared_2697_ = v_isSharedCheck_2706_;
goto v_resetjp_2695_;
}
else
{
lean_inc(v_a_2694_);
lean_dec(v___x_2693_);
v___x_2696_ = lean_box(0);
v_isShared_2697_ = v_isSharedCheck_2706_;
goto v_resetjp_2695_;
}
v_resetjp_2695_:
{
if (lean_obj_tag(v_a_2694_) == 0)
{
lean_object* v_a_2698_; lean_object* v___x_2700_; 
lean_dec(v_a_2681_);
v_a_2698_ = lean_ctor_get(v_a_2694_, 0);
lean_inc(v_a_2698_);
lean_dec_ref_known(v_a_2694_, 1);
if (v_isShared_2697_ == 0)
{
lean_ctor_set(v___x_2696_, 0, v_a_2698_);
v___x_2700_ = v___x_2696_;
goto v_reusejp_2699_;
}
else
{
lean_object* v_reuseFailAlloc_2701_; 
v_reuseFailAlloc_2701_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2701_, 0, v_a_2698_);
v___x_2700_ = v_reuseFailAlloc_2701_;
goto v_reusejp_2699_;
}
v_reusejp_2699_:
{
return v___x_2700_;
}
}
else
{
lean_object* v_a_2702_; size_t v___x_2703_; size_t v___x_2704_; 
lean_del_object(v___x_2696_);
v_a_2702_ = lean_ctor_get(v_a_2694_, 0);
lean_inc(v_a_2702_);
lean_dec_ref_known(v_a_2694_, 1);
v___x_2703_ = ((size_t)1ULL);
v___x_2704_ = lean_usize_add(v_i_2685_, v___x_2703_);
v_i_2685_ = v___x_2704_;
v_b_2686_ = v_a_2702_;
goto _start;
}
}
}
else
{
lean_object* v_a_2707_; lean_object* v___x_2709_; uint8_t v_isShared_2710_; uint8_t v_isSharedCheck_2714_; 
lean_dec(v_a_2681_);
v_a_2707_ = lean_ctor_get(v___x_2693_, 0);
v_isSharedCheck_2714_ = !lean_is_exclusive(v___x_2693_);
if (v_isSharedCheck_2714_ == 0)
{
v___x_2709_ = v___x_2693_;
v_isShared_2710_ = v_isSharedCheck_2714_;
goto v_resetjp_2708_;
}
else
{
lean_inc(v_a_2707_);
lean_dec(v___x_2693_);
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
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__12___boxed(lean_object* v_a_2715_, lean_object* v___x_2716_, lean_object* v_as_2717_, lean_object* v_sz_2718_, lean_object* v_i_2719_, lean_object* v_b_2720_, lean_object* v___y_2721_, lean_object* v___y_2722_, lean_object* v___y_2723_){
_start:
{
size_t v_sz_boxed_2724_; size_t v_i_boxed_2725_; lean_object* v_res_2726_; 
v_sz_boxed_2724_ = lean_unbox_usize(v_sz_2718_);
lean_dec(v_sz_2718_);
v_i_boxed_2725_ = lean_unbox_usize(v_i_2719_);
lean_dec(v_i_2719_);
v_res_2726_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__12(v_a_2715_, v___x_2716_, v_as_2717_, v_sz_boxed_2724_, v_i_boxed_2725_, v_b_2720_, v___y_2721_, v___y_2722_);
lean_dec(v___y_2722_);
lean_dec_ref(v___y_2721_);
lean_dec_ref(v_as_2717_);
lean_dec_ref(v___x_2716_);
return v_res_2726_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__6(lean_object* v_a_2730_, lean_object* v_as_2731_, size_t v_sz_2732_, size_t v_i_2733_, lean_object* v_b_2734_){
_start:
{
uint8_t v___x_2735_; 
v___x_2735_ = lean_usize_dec_lt(v_i_2733_, v_sz_2732_);
if (v___x_2735_ == 0)
{
lean_dec_ref(v_a_2730_);
lean_inc_ref(v_b_2734_);
return v_b_2734_;
}
else
{
lean_object* v_a_2736_; lean_object* v_fst_2737_; lean_object* v_fst_2738_; lean_object* v_snd_2739_; lean_object* v_fst_2740_; lean_object* v_snd_2741_; lean_object* v___x_2742_; lean_object* v___x_2743_; uint8_t v___y_2745_; uint8_t v___x_2760_; 
v_a_2736_ = lean_array_uget_borrowed(v_as_2731_, v_i_2733_);
v_fst_2737_ = lean_ctor_get(v_a_2736_, 0);
v_fst_2738_ = lean_ctor_get(v_fst_2737_, 0);
v_snd_2739_ = lean_ctor_get(v_fst_2737_, 1);
v_fst_2740_ = lean_ctor_get(v_a_2730_, 0);
v_snd_2741_ = lean_ctor_get(v_a_2730_, 1);
v___x_2742_ = lean_box(0);
v___x_2743_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__6___closed__0));
v___x_2760_ = l_Lean_instBEqFVarId_beq(v_fst_2738_, v_fst_2740_);
if (v___x_2760_ == 0)
{
v___y_2745_ = v___x_2760_;
goto v___jp_2744_;
}
else
{
uint8_t v___x_2761_; 
v___x_2761_ = l_Lean_instBEqMVarId_beq(v_snd_2739_, v_snd_2741_);
v___y_2745_ = v___x_2761_;
goto v___jp_2744_;
}
v___jp_2744_:
{
if (v___y_2745_ == 0)
{
size_t v___x_2746_; size_t v___x_2747_; 
v___x_2746_ = ((size_t)1ULL);
v___x_2747_ = lean_usize_add(v_i_2733_, v___x_2746_);
v_i_2733_ = v___x_2747_;
v_b_2734_ = v___x_2743_;
goto _start;
}
else
{
lean_object* v___x_2750_; uint8_t v_isShared_2751_; uint8_t v_isSharedCheck_2757_; 
v_isSharedCheck_2757_ = !lean_is_exclusive(v_a_2730_);
if (v_isSharedCheck_2757_ == 0)
{
lean_object* v_unused_2758_; lean_object* v_unused_2759_; 
v_unused_2758_ = lean_ctor_get(v_a_2730_, 1);
lean_dec(v_unused_2758_);
v_unused_2759_ = lean_ctor_get(v_a_2730_, 0);
lean_dec(v_unused_2759_);
v___x_2750_ = v_a_2730_;
v_isShared_2751_ = v_isSharedCheck_2757_;
goto v_resetjp_2749_;
}
else
{
lean_dec(v_a_2730_);
v___x_2750_ = lean_box(0);
v_isShared_2751_ = v_isSharedCheck_2757_;
goto v_resetjp_2749_;
}
v_resetjp_2749_:
{
lean_object* v___x_2752_; lean_object* v___x_2753_; lean_object* v___x_2755_; 
lean_inc(v_a_2736_);
v___x_2752_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2752_, 0, v_a_2736_);
v___x_2753_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2753_, 0, v___x_2752_);
if (v_isShared_2751_ == 0)
{
lean_ctor_set(v___x_2750_, 1, v___x_2742_);
lean_ctor_set(v___x_2750_, 0, v___x_2753_);
v___x_2755_ = v___x_2750_;
goto v_reusejp_2754_;
}
else
{
lean_object* v_reuseFailAlloc_2756_; 
v_reuseFailAlloc_2756_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2756_, 0, v___x_2753_);
lean_ctor_set(v_reuseFailAlloc_2756_, 1, v___x_2742_);
v___x_2755_ = v_reuseFailAlloc_2756_;
goto v_reusejp_2754_;
}
v_reusejp_2754_:
{
return v___x_2755_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__6___boxed(lean_object* v_a_2762_, lean_object* v_as_2763_, lean_object* v_sz_2764_, lean_object* v_i_2765_, lean_object* v_b_2766_){
_start:
{
size_t v_sz_boxed_2767_; size_t v_i_boxed_2768_; lean_object* v_res_2769_; 
v_sz_boxed_2767_ = lean_unbox_usize(v_sz_2764_);
lean_dec(v_sz_2764_);
v_i_boxed_2768_ = lean_unbox_usize(v_i_2765_);
lean_dec(v_i_2765_);
v_res_2769_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__6(v_a_2762_, v_as_2763_, v_sz_boxed_2767_, v_i_boxed_2768_, v_b_2766_);
lean_dec_ref(v_b_2766_);
lean_dec_ref(v_as_2763_);
return v_res_2769_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__7___redArg(lean_object* v___x_2770_, lean_object* v___x_2771_, lean_object* v_a_2772_, lean_object* v_a_2773_){
_start:
{
if (lean_obj_tag(v_a_2772_) == 0)
{
lean_object* v___x_2775_; lean_object* v___x_2776_; 
lean_dec(v___x_2771_);
v___x_2775_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2775_, 0, v_a_2773_);
v___x_2776_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2776_, 0, v___x_2775_);
return v___x_2776_;
}
else
{
lean_object* v_key_2777_; lean_object* v_tail_2778_; lean_object* v___x_2779_; size_t v_sz_2780_; size_t v___x_2781_; lean_object* v___x_2782_; lean_object* v_fst_2783_; 
v_key_2777_ = lean_ctor_get(v_a_2772_, 0);
lean_inc(v_key_2777_);
v_tail_2778_ = lean_ctor_get(v_a_2772_, 2);
lean_inc(v_tail_2778_);
lean_dec_ref_known(v_a_2772_, 3);
v___x_2779_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__6___closed__0));
v_sz_2780_ = lean_array_size(v___x_2770_);
v___x_2781_ = ((size_t)0ULL);
v___x_2782_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__6(v_key_2777_, v___x_2770_, v_sz_2780_, v___x_2781_, v___x_2779_);
v_fst_2783_ = lean_ctor_get(v___x_2782_, 0);
lean_inc(v_fst_2783_);
lean_dec_ref(v___x_2782_);
if (lean_obj_tag(v_fst_2783_) == 0)
{
v_a_2772_ = v_tail_2778_;
goto _start;
}
else
{
lean_object* v_val_2785_; 
v_val_2785_ = lean_ctor_get(v_fst_2783_, 0);
lean_inc(v_val_2785_);
lean_dec_ref_known(v_fst_2783_, 1);
if (lean_obj_tag(v_val_2785_) == 1)
{
lean_object* v_val_2786_; lean_object* v_snd_2787_; lean_object* v___x_2789_; uint8_t v_isShared_2790_; uint8_t v_isSharedCheck_2796_; 
v_val_2786_ = lean_ctor_get(v_val_2785_, 0);
lean_inc(v_val_2786_);
lean_dec_ref_known(v_val_2785_, 1);
v_snd_2787_ = lean_ctor_get(v_val_2786_, 1);
v_isSharedCheck_2796_ = !lean_is_exclusive(v_val_2786_);
if (v_isSharedCheck_2796_ == 0)
{
lean_object* v_unused_2797_; 
v_unused_2797_ = lean_ctor_get(v_val_2786_, 0);
lean_dec(v_unused_2797_);
v___x_2789_ = v_val_2786_;
v_isShared_2790_ = v_isSharedCheck_2796_;
goto v_resetjp_2788_;
}
else
{
lean_inc(v_snd_2787_);
lean_dec(v_val_2786_);
v___x_2789_ = lean_box(0);
v_isShared_2790_ = v_isSharedCheck_2796_;
goto v_resetjp_2788_;
}
v_resetjp_2788_:
{
lean_object* v___x_2792_; 
lean_inc(v___x_2771_);
if (v_isShared_2790_ == 0)
{
lean_ctor_set(v___x_2789_, 0, v___x_2771_);
v___x_2792_ = v___x_2789_;
goto v_reusejp_2791_;
}
else
{
lean_object* v_reuseFailAlloc_2795_; 
v_reuseFailAlloc_2795_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2795_, 0, v___x_2771_);
lean_ctor_set(v_reuseFailAlloc_2795_, 1, v_snd_2787_);
v___x_2792_ = v_reuseFailAlloc_2795_;
goto v_reusejp_2791_;
}
v_reusejp_2791_:
{
lean_object* v___x_2793_; 
v___x_2793_ = lean_array_push(v_a_2773_, v___x_2792_);
v_a_2772_ = v_tail_2778_;
v_a_2773_ = v___x_2793_;
goto _start;
}
}
}
else
{
lean_dec(v_val_2785_);
v_a_2772_ = v_tail_2778_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__7___redArg___boxed(lean_object* v___x_2799_, lean_object* v___x_2800_, lean_object* v_a_2801_, lean_object* v_a_2802_, lean_object* v___y_2803_){
_start:
{
lean_object* v_res_2804_; 
v_res_2804_ = lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__7___redArg(v___x_2799_, v___x_2800_, v_a_2801_, v_a_2802_);
lean_dec_ref(v___x_2799_);
return v_res_2804_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__13(lean_object* v___x_2805_, lean_object* v___x_2806_, lean_object* v_as_2807_, size_t v_sz_2808_, size_t v_i_2809_, lean_object* v_b_2810_, lean_object* v___y_2811_, lean_object* v___y_2812_){
_start:
{
uint8_t v___x_2814_; 
v___x_2814_ = lean_usize_dec_lt(v_i_2809_, v_sz_2808_);
if (v___x_2814_ == 0)
{
lean_object* v___x_2815_; 
lean_dec(v___x_2806_);
v___x_2815_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2815_, 0, v_b_2810_);
return v___x_2815_;
}
else
{
lean_object* v_a_2816_; lean_object* v___x_2817_; 
v_a_2816_ = lean_array_uget_borrowed(v_as_2807_, v_i_2809_);
lean_inc(v_a_2816_);
lean_inc(v___x_2806_);
v___x_2817_ = lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__7___redArg(v___x_2805_, v___x_2806_, v_a_2816_, v_b_2810_);
if (lean_obj_tag(v___x_2817_) == 0)
{
lean_object* v_a_2818_; lean_object* v___x_2820_; uint8_t v_isShared_2821_; uint8_t v_isSharedCheck_2830_; 
v_a_2818_ = lean_ctor_get(v___x_2817_, 0);
v_isSharedCheck_2830_ = !lean_is_exclusive(v___x_2817_);
if (v_isSharedCheck_2830_ == 0)
{
v___x_2820_ = v___x_2817_;
v_isShared_2821_ = v_isSharedCheck_2830_;
goto v_resetjp_2819_;
}
else
{
lean_inc(v_a_2818_);
lean_dec(v___x_2817_);
v___x_2820_ = lean_box(0);
v_isShared_2821_ = v_isSharedCheck_2830_;
goto v_resetjp_2819_;
}
v_resetjp_2819_:
{
if (lean_obj_tag(v_a_2818_) == 0)
{
lean_object* v_a_2822_; lean_object* v___x_2824_; 
lean_dec(v___x_2806_);
v_a_2822_ = lean_ctor_get(v_a_2818_, 0);
lean_inc(v_a_2822_);
lean_dec_ref_known(v_a_2818_, 1);
if (v_isShared_2821_ == 0)
{
lean_ctor_set(v___x_2820_, 0, v_a_2822_);
v___x_2824_ = v___x_2820_;
goto v_reusejp_2823_;
}
else
{
lean_object* v_reuseFailAlloc_2825_; 
v_reuseFailAlloc_2825_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2825_, 0, v_a_2822_);
v___x_2824_ = v_reuseFailAlloc_2825_;
goto v_reusejp_2823_;
}
v_reusejp_2823_:
{
return v___x_2824_;
}
}
else
{
lean_object* v_a_2826_; size_t v___x_2827_; size_t v___x_2828_; 
lean_del_object(v___x_2820_);
v_a_2826_ = lean_ctor_get(v_a_2818_, 0);
lean_inc(v_a_2826_);
lean_dec_ref_known(v_a_2818_, 1);
v___x_2827_ = ((size_t)1ULL);
v___x_2828_ = lean_usize_add(v_i_2809_, v___x_2827_);
v_i_2809_ = v___x_2828_;
v_b_2810_ = v_a_2826_;
goto _start;
}
}
}
else
{
lean_object* v_a_2831_; lean_object* v___x_2833_; uint8_t v_isShared_2834_; uint8_t v_isSharedCheck_2838_; 
lean_dec(v___x_2806_);
v_a_2831_ = lean_ctor_get(v___x_2817_, 0);
v_isSharedCheck_2838_ = !lean_is_exclusive(v___x_2817_);
if (v_isSharedCheck_2838_ == 0)
{
v___x_2833_ = v___x_2817_;
v_isShared_2834_ = v_isSharedCheck_2838_;
goto v_resetjp_2832_;
}
else
{
lean_inc(v_a_2831_);
lean_dec(v___x_2817_);
v___x_2833_ = lean_box(0);
v_isShared_2834_ = v_isSharedCheck_2838_;
goto v_resetjp_2832_;
}
v_resetjp_2832_:
{
lean_object* v___x_2836_; 
if (v_isShared_2834_ == 0)
{
v___x_2836_ = v___x_2833_;
goto v_reusejp_2835_;
}
else
{
lean_object* v_reuseFailAlloc_2837_; 
v_reuseFailAlloc_2837_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2837_, 0, v_a_2831_);
v___x_2836_ = v_reuseFailAlloc_2837_;
goto v_reusejp_2835_;
}
v_reusejp_2835_:
{
return v___x_2836_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__13___boxed(lean_object* v___x_2839_, lean_object* v___x_2840_, lean_object* v_as_2841_, lean_object* v_sz_2842_, lean_object* v_i_2843_, lean_object* v_b_2844_, lean_object* v___y_2845_, lean_object* v___y_2846_, lean_object* v___y_2847_){
_start:
{
size_t v_sz_boxed_2848_; size_t v_i_boxed_2849_; lean_object* v_res_2850_; 
v_sz_boxed_2848_ = lean_unbox_usize(v_sz_2842_);
lean_dec(v_sz_2842_);
v_i_boxed_2849_ = lean_unbox_usize(v_i_2843_);
lean_dec(v_i_2843_);
v_res_2850_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__13(v___x_2839_, v___x_2840_, v_as_2841_, v_sz_boxed_2848_, v_i_boxed_2849_, v_b_2844_, v___y_2845_, v___y_2846_);
lean_dec(v___y_2846_);
lean_dec_ref(v___y_2845_);
lean_dec_ref(v_as_2841_);
lean_dec_ref(v___x_2839_);
return v_res_2850_;
}
}
static lean_object* _init_lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__14___redArg___closed__0(void){
_start:
{
lean_object* v___x_2851_; lean_object* v___x_2852_; lean_object* v___x_2853_; 
v___x_2851_ = lean_box(0);
v___x_2852_ = lean_unsigned_to_nat(16u);
v___x_2853_ = lean_mk_array(v___x_2852_, v___x_2851_);
return v___x_2853_;
}
}
static lean_object* _init_lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__14___redArg___closed__1(void){
_start:
{
lean_object* v___x_2854_; lean_object* v___x_2855_; lean_object* v___x_2856_; 
v___x_2854_ = lean_obj_once(&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__14___redArg___closed__0, &lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__14___redArg___closed__0_once, _init_lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__14___redArg___closed__0);
v___x_2855_ = lean_unsigned_to_nat(0u);
v___x_2856_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2856_, 0, v___x_2855_);
lean_ctor_set(v___x_2856_, 1, v___x_2854_);
return v___x_2856_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__14___redArg(lean_object* v___y_2857_, lean_object* v___x_2858_, lean_object* v___x_2859_, lean_object* v___x_2860_, lean_object* v_as_x27_2861_, lean_object* v_b_2862_, lean_object* v___y_2863_, lean_object* v___y_2864_){
_start:
{
if (lean_obj_tag(v_as_x27_2861_) == 0)
{
lean_object* v___x_2866_; 
lean_dec(v___x_2859_);
v___x_2866_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2866_, 0, v_b_2862_);
return v___x_2866_;
}
else
{
lean_object* v_head_2867_; lean_object* v_tail_2868_; lean_object* v___y_2870_; lean_object* v_decls_2891_; lean_object* v___x_2892_; 
v_head_2867_ = lean_ctor_get(v_as_x27_2861_, 0);
v_tail_2868_ = lean_ctor_get(v_as_x27_2861_, 1);
v_decls_2891_ = lean_ctor_get(v___x_2860_, 5);
v___x_2892_ = lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__1___redArg(v_decls_2891_, v_head_2867_);
if (lean_obj_tag(v___x_2892_) == 0)
{
lean_object* v___x_2893_; 
v___x_2893_ = l_Lean_instInhabitedMetavarDecl_default;
v___y_2870_ = v___x_2893_;
goto v___jp_2869_;
}
else
{
lean_object* v_val_2894_; 
v_val_2894_ = lean_ctor_get(v___x_2892_, 0);
lean_inc(v_val_2894_);
lean_dec_ref_known(v___x_2892_, 1);
v___y_2870_ = v_val_2894_;
goto v___jp_2869_;
}
v___jp_2869_:
{
lean_object* v_lctx_2871_; lean_object* v_buckets_2872_; lean_object* v___x_2873_; size_t v_sz_2874_; size_t v___x_2875_; lean_object* v___x_2876_; 
v_lctx_2871_ = lean_ctor_get(v___y_2870_, 1);
lean_inc_ref(v_lctx_2871_);
lean_dec_ref(v___y_2870_);
v_buckets_2872_ = lean_ctor_get(v___y_2857_, 1);
v___x_2873_ = lean_obj_once(&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__14___redArg___closed__1, &lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__14___redArg___closed__1_once, _init_lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__14___redArg___closed__1);
v_sz_2874_ = lean_array_size(v_buckets_2872_);
v___x_2875_ = ((size_t)0ULL);
lean_inc(v_head_2867_);
v___x_2876_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__12(v_head_2867_, v_lctx_2871_, v_buckets_2872_, v_sz_2874_, v___x_2875_, v___x_2873_, v___y_2863_, v___y_2864_);
lean_dec_ref(v_lctx_2871_);
if (lean_obj_tag(v___x_2876_) == 0)
{
lean_object* v_a_2877_; lean_object* v_buckets_2878_; size_t v_sz_2879_; lean_object* v___x_2880_; 
v_a_2877_ = lean_ctor_get(v___x_2876_, 0);
lean_inc(v_a_2877_);
lean_dec_ref_known(v___x_2876_, 1);
v_buckets_2878_ = lean_ctor_get(v_a_2877_, 1);
lean_inc_ref(v_buckets_2878_);
lean_dec(v_a_2877_);
v_sz_2879_ = lean_array_size(v_buckets_2878_);
lean_inc(v___x_2859_);
v___x_2880_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__13(v___x_2858_, v___x_2859_, v_buckets_2878_, v_sz_2879_, v___x_2875_, v_b_2862_, v___y_2863_, v___y_2864_);
lean_dec_ref(v_buckets_2878_);
if (lean_obj_tag(v___x_2880_) == 0)
{
lean_object* v_a_2881_; 
v_a_2881_ = lean_ctor_get(v___x_2880_, 0);
lean_inc(v_a_2881_);
lean_dec_ref_known(v___x_2880_, 1);
v_as_x27_2861_ = v_tail_2868_;
v_b_2862_ = v_a_2881_;
goto _start;
}
else
{
lean_dec(v___x_2859_);
return v___x_2880_;
}
}
else
{
lean_object* v_a_2883_; lean_object* v___x_2885_; uint8_t v_isShared_2886_; uint8_t v_isSharedCheck_2890_; 
lean_dec_ref(v_b_2862_);
lean_dec(v___x_2859_);
v_a_2883_ = lean_ctor_get(v___x_2876_, 0);
v_isSharedCheck_2890_ = !lean_is_exclusive(v___x_2876_);
if (v_isSharedCheck_2890_ == 0)
{
v___x_2885_ = v___x_2876_;
v_isShared_2886_ = v_isSharedCheck_2890_;
goto v_resetjp_2884_;
}
else
{
lean_inc(v_a_2883_);
lean_dec(v___x_2876_);
v___x_2885_ = lean_box(0);
v_isShared_2886_ = v_isSharedCheck_2890_;
goto v_resetjp_2884_;
}
v_resetjp_2884_:
{
lean_object* v___x_2888_; 
if (v_isShared_2886_ == 0)
{
v___x_2888_ = v___x_2885_;
goto v_reusejp_2887_;
}
else
{
lean_object* v_reuseFailAlloc_2889_; 
v_reuseFailAlloc_2889_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2889_, 0, v_a_2883_);
v___x_2888_ = v_reuseFailAlloc_2889_;
goto v_reusejp_2887_;
}
v_reusejp_2887_:
{
return v___x_2888_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__14___redArg___boxed(lean_object* v___y_2895_, lean_object* v___x_2896_, lean_object* v___x_2897_, lean_object* v___x_2898_, lean_object* v_as_x27_2899_, lean_object* v_b_2900_, lean_object* v___y_2901_, lean_object* v___y_2902_, lean_object* v___y_2903_){
_start:
{
lean_object* v_res_2904_; 
v_res_2904_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__14___redArg(v___y_2895_, v___x_2896_, v___x_2897_, v___x_2898_, v_as_x27_2899_, v_b_2900_, v___y_2901_, v___y_2902_);
lean_dec(v___y_2902_);
lean_dec_ref(v___y_2901_);
lean_dec(v_as_x27_2899_);
lean_dec_ref(v___x_2898_);
lean_dec_ref(v___x_2896_);
lean_dec_ref(v___y_2895_);
return v_res_2904_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__5(lean_object* v___x_2905_, size_t v_sz_2906_, size_t v_i_2907_, lean_object* v_bs_2908_){
_start:
{
uint8_t v___x_2909_; 
v___x_2909_ = lean_usize_dec_lt(v_i_2907_, v_sz_2906_);
if (v___x_2909_ == 0)
{
lean_dec_ref(v___x_2905_);
return v_bs_2908_;
}
else
{
lean_object* v_v_2910_; lean_object* v___x_2911_; lean_object* v_bs_x27_2912_; lean_object* v___x_2913_; size_t v___x_2914_; size_t v___x_2915_; lean_object* v___x_2916_; 
v_v_2910_ = lean_array_uget(v_bs_2908_, v_i_2907_);
v___x_2911_ = lean_unsigned_to_nat(0u);
v_bs_x27_2912_ = lean_array_uset(v_bs_2908_, v_i_2907_, v___x_2911_);
lean_inc_ref(v___x_2905_);
v___x_2913_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2913_, 0, v_v_2910_);
lean_ctor_set(v___x_2913_, 1, v___x_2905_);
v___x_2914_ = ((size_t)1ULL);
v___x_2915_ = lean_usize_add(v_i_2907_, v___x_2914_);
v___x_2916_ = lean_array_uset(v_bs_x27_2912_, v_i_2907_, v___x_2913_);
v_i_2907_ = v___x_2915_;
v_bs_2908_ = v___x_2916_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__5___boxed(lean_object* v___x_2918_, lean_object* v_sz_2919_, lean_object* v_i_2920_, lean_object* v_bs_2921_){
_start:
{
size_t v_sz_boxed_2922_; size_t v_i_boxed_2923_; lean_object* v_res_2924_; 
v_sz_boxed_2922_ = lean_unbox_usize(v_sz_2919_);
lean_dec(v_sz_2919_);
v_i_boxed_2923_ = lean_unbox_usize(v_i_2920_);
lean_dec(v_i_2920_);
v_res_2924_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__5(v___x_2918_, v_sz_boxed_2922_, v_i_boxed_2923_, v_bs_2921_);
return v_res_2924_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__15___redArg(lean_object* v_a_2925_, lean_object* v_a_2926_, lean_object* v___x_2927_, lean_object* v___x_2928_, lean_object* v___x_2929_, lean_object* v___x_2930_, lean_object* v_as_x27_2931_, lean_object* v_b_2932_){
_start:
{
if (lean_obj_tag(v_as_x27_2931_) == 0)
{
lean_object* v___x_2934_; 
lean_dec(v___x_2929_);
lean_dec_ref(v___x_2928_);
lean_dec(v___x_2927_);
lean_dec(v_a_2925_);
v___x_2934_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2934_, 0, v_b_2932_);
return v___x_2934_;
}
else
{
lean_object* v_head_2935_; lean_object* v_tail_2936_; lean_object* v___y_2938_; lean_object* v_decls_2948_; lean_object* v___x_2949_; 
v_head_2935_ = lean_ctor_get(v_as_x27_2931_, 0);
v_tail_2936_ = lean_ctor_get(v_as_x27_2931_, 1);
v_decls_2948_ = lean_ctor_get(v___x_2930_, 5);
v___x_2949_ = lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist_spec__1___redArg(v_decls_2948_, v_head_2935_);
if (lean_obj_tag(v___x_2949_) == 0)
{
lean_object* v___x_2950_; 
v___x_2950_ = l_Lean_instInhabitedMetavarDecl_default;
v___y_2938_ = v___x_2950_;
goto v___jp_2937_;
}
else
{
lean_object* v_val_2951_; 
v_val_2951_ = lean_ctor_get(v___x_2949_, 0);
lean_inc(v_val_2951_);
lean_dec_ref_known(v___x_2949_, 1);
v___y_2938_ = v_val_2951_;
goto v___jp_2937_;
}
v___jp_2937_:
{
lean_object* v_lctx_2939_; lean_object* v_ci_2940_; lean_object* v___x_2941_; lean_object* v___x_2942_; size_t v_sz_2943_; size_t v___x_2944_; lean_object* v___x_2945_; lean_object* v___x_2946_; 
v_lctx_2939_ = lean_ctor_get(v___y_2938_, 1);
lean_inc_ref(v_lctx_2939_);
lean_dec_ref(v___y_2938_);
v_ci_2940_ = lean_ctor_get(v_a_2926_, 1);
lean_inc(v_head_2935_);
v___x_2941_ = lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_Stained_toFMVarId(v_head_2935_, v_lctx_2939_, v_a_2925_);
lean_dec_ref(v_lctx_2939_);
lean_inc(v___x_2929_);
lean_inc_ref(v___x_2928_);
lean_inc_ref(v_ci_2940_);
lean_inc(v___x_2927_);
lean_inc(v_a_2925_);
v___x_2942_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_2942_, 0, v_a_2925_);
lean_ctor_set(v___x_2942_, 1, v___x_2927_);
lean_ctor_set(v___x_2942_, 2, v_ci_2940_);
lean_ctor_set(v___x_2942_, 3, v___x_2928_);
lean_ctor_set(v___x_2942_, 4, v___x_2929_);
v_sz_2943_ = lean_array_size(v___x_2941_);
v___x_2944_ = ((size_t)0ULL);
v___x_2945_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__5(v___x_2942_, v_sz_2943_, v___x_2944_, v___x_2941_);
v___x_2946_ = l_Array_append___redArg(v_b_2932_, v___x_2945_);
lean_dec_ref(v___x_2945_);
v_as_x27_2931_ = v_tail_2936_;
v_b_2932_ = v___x_2946_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__15___redArg___boxed(lean_object* v_a_2952_, lean_object* v_a_2953_, lean_object* v___x_2954_, lean_object* v___x_2955_, lean_object* v___x_2956_, lean_object* v___x_2957_, lean_object* v_as_x27_2958_, lean_object* v_b_2959_, lean_object* v___y_2960_){
_start:
{
lean_object* v_res_2961_; 
v_res_2961_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__15___redArg(v_a_2952_, v_a_2953_, v___x_2954_, v___x_2955_, v___x_2956_, v___x_2957_, v_as_x27_2958_, v_b_2959_);
lean_dec(v_as_x27_2958_);
lean_dec_ref(v___x_2957_);
lean_dec_ref(v_a_2953_);
return v_res_2961_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_beq___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__2(lean_object* v_x_2962_, lean_object* v_x_2963_){
_start:
{
if (lean_obj_tag(v_x_2962_) == 0)
{
if (lean_obj_tag(v_x_2963_) == 0)
{
uint8_t v___x_2964_; 
v___x_2964_ = 1;
return v___x_2964_;
}
else
{
uint8_t v___x_2965_; 
v___x_2965_ = 0;
return v___x_2965_;
}
}
else
{
if (lean_obj_tag(v_x_2963_) == 0)
{
uint8_t v___x_2966_; 
v___x_2966_ = 0;
return v___x_2966_;
}
else
{
lean_object* v_head_2967_; lean_object* v_tail_2968_; lean_object* v_head_2969_; lean_object* v_tail_2970_; uint8_t v___x_2971_; 
v_head_2967_ = lean_ctor_get(v_x_2962_, 0);
v_tail_2968_ = lean_ctor_get(v_x_2962_, 1);
v_head_2969_ = lean_ctor_get(v_x_2963_, 0);
v_tail_2970_ = lean_ctor_get(v_x_2963_, 1);
v___x_2971_ = l_Lean_instBEqMVarId_beq(v_head_2967_, v_head_2969_);
if (v___x_2971_ == 0)
{
return v___x_2971_;
}
else
{
v_x_2962_ = v_tail_2968_;
v_x_2963_ = v_tail_2970_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_beq___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__2___boxed(lean_object* v_x_2973_, lean_object* v_x_2974_){
_start:
{
uint8_t v_res_2975_; lean_object* v_r_2976_; 
v_res_2975_ = lp_mathlib_List_beq___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__2(v_x_2973_, v_x_2974_);
lean_dec(v_x_2974_);
lean_dec(v_x_2973_);
v_r_2976_ = lean_box(v_res_2975_);
return v_r_2976_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__4___lam__1(lean_object* v___f_2977_, lean_object* v___x_2978_, lean_object* v_x_2979_, lean_object* v___y_2980_, lean_object* v___y_2981_){
_start:
{
lean_object* v___x_2983_; lean_object* v___x_2984_; 
v___x_2983_ = lean_box(0);
lean_inc(v___y_2981_);
lean_inc_ref(v___y_2980_);
v___x_2984_ = lean_apply_5(v___f_2977_, v___x_2983_, v___x_2978_, v___y_2980_, v___y_2981_, lean_box(0));
return v___x_2984_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__4___lam__1___boxed(lean_object* v___f_2985_, lean_object* v___x_2986_, lean_object* v_x_2987_, lean_object* v___y_2988_, lean_object* v___y_2989_, lean_object* v___y_2990_){
_start:
{
lean_object* v_res_2991_; 
v_res_2991_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__4___lam__1(v___f_2985_, v___x_2986_, v_x_2987_, v___y_2988_, v___y_2989_);
lean_dec(v___y_2989_);
lean_dec_ref(v___y_2988_);
return v_res_2991_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Array_isEqvAux___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__3___redArg(lean_object* v_xs_2992_, lean_object* v_ys_2993_, lean_object* v_x_2994_){
_start:
{
lean_object* v_zero_2995_; uint8_t v_isZero_2996_; 
v_zero_2995_ = lean_unsigned_to_nat(0u);
v_isZero_2996_ = lean_nat_dec_eq(v_x_2994_, v_zero_2995_);
if (v_isZero_2996_ == 1)
{
lean_dec(v_x_2994_);
return v_isZero_2996_;
}
else
{
lean_object* v_one_2997_; lean_object* v_n_2998_; uint8_t v___y_3000_; lean_object* v___x_3002_; lean_object* v_fst_3003_; lean_object* v_snd_3004_; lean_object* v___x_3005_; lean_object* v_fst_3006_; lean_object* v_snd_3007_; uint8_t v___x_3008_; 
v_one_2997_ = lean_unsigned_to_nat(1u);
v_n_2998_ = lean_nat_sub(v_x_2994_, v_one_2997_);
lean_dec(v_x_2994_);
v___x_3002_ = lean_array_fget_borrowed(v_xs_2992_, v_n_2998_);
v_fst_3003_ = lean_ctor_get(v___x_3002_, 0);
v_snd_3004_ = lean_ctor_get(v___x_3002_, 1);
v___x_3005_ = lean_array_fget_borrowed(v_ys_2993_, v_n_2998_);
v_fst_3006_ = lean_ctor_get(v___x_3005_, 0);
v_snd_3007_ = lean_ctor_get(v___x_3005_, 1);
v___x_3008_ = l_Lean_instBEqFVarId_beq(v_fst_3003_, v_fst_3006_);
if (v___x_3008_ == 0)
{
v___y_3000_ = v___x_3008_;
goto v___jp_2999_;
}
else
{
uint8_t v___x_3009_; 
v___x_3009_ = l_Lean_instBEqMVarId_beq(v_snd_3004_, v_snd_3007_);
v___y_3000_ = v___x_3009_;
goto v___jp_2999_;
}
v___jp_2999_:
{
if (v___y_3000_ == 0)
{
lean_dec(v_n_2998_);
return v___y_3000_;
}
else
{
v_x_2994_ = v_n_2998_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_isEqvAux___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__3___redArg___boxed(lean_object* v_xs_3010_, lean_object* v_ys_3011_, lean_object* v_x_3012_){
_start:
{
uint8_t v_res_3013_; lean_object* v_r_3014_; 
v_res_3013_ = lp_mathlib_Array_isEqvAux___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__3___redArg(v_xs_3010_, v_ys_3011_, v_x_3012_);
lean_dec_ref(v_ys_3011_);
lean_dec_ref(v_xs_3010_);
v_r_3014_ = lean_box(v_res_3013_);
return v_r_3014_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__1___redArg(lean_object* v_snd_3015_, lean_object* v_as_3016_, size_t v_sz_3017_, size_t v_i_3018_, lean_object* v_b_3019_){
_start:
{
uint8_t v___x_3021_; 
v___x_3021_ = lean_usize_dec_lt(v_i_3018_, v_sz_3017_);
if (v___x_3021_ == 0)
{
lean_object* v___x_3022_; 
lean_dec_ref(v_snd_3015_);
v___x_3022_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3022_, 0, v_b_3019_);
return v___x_3022_;
}
else
{
lean_object* v_a_3023_; lean_object* v___x_3024_; lean_object* v___x_3025_; size_t v___x_3026_; size_t v___x_3027_; 
v_a_3023_ = lean_array_uget_borrowed(v_as_3016_, v_i_3018_);
lean_inc_ref(v_snd_3015_);
lean_inc(v_a_3023_);
v___x_3024_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3024_, 0, v_a_3023_);
lean_ctor_set(v___x_3024_, 1, v_snd_3015_);
v___x_3025_ = lean_array_push(v_b_3019_, v___x_3024_);
v___x_3026_ = ((size_t)1ULL);
v___x_3027_ = lean_usize_add(v_i_3018_, v___x_3026_);
v_i_3018_ = v___x_3027_;
v_b_3019_ = v___x_3025_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__1___redArg___boxed(lean_object* v_snd_3029_, lean_object* v_as_3030_, lean_object* v_sz_3031_, lean_object* v_i_3032_, lean_object* v_b_3033_, lean_object* v___y_3034_){
_start:
{
size_t v_sz_boxed_3035_; size_t v_i_boxed_3036_; lean_object* v_res_3037_; 
v_sz_boxed_3035_ = lean_unbox_usize(v_sz_3031_);
lean_dec(v_sz_3031_);
v_i_boxed_3036_ = lean_unbox_usize(v_i_3032_);
lean_dec(v_i_3032_);
v_res_3037_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__1___redArg(v_snd_3029_, v_as_3030_, v_sz_boxed_3035_, v_i_boxed_3036_, v_b_3033_);
lean_dec_ref(v_as_3030_);
return v_res_3037_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__4___lam__0(lean_object* v___x_3038_, lean_object* v_snd_3039_, lean_object* v_____r_3040_, lean_object* v_new_3041_, lean_object* v___y_3042_, lean_object* v___y_3043_){
_start:
{
size_t v_sz_3045_; size_t v___x_3046_; lean_object* v___x_3047_; 
v_sz_3045_ = lean_array_size(v___x_3038_);
v___x_3046_ = ((size_t)0ULL);
v___x_3047_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__1___redArg(v_snd_3039_, v___x_3038_, v_sz_3045_, v___x_3046_, v_new_3041_);
if (lean_obj_tag(v___x_3047_) == 0)
{
lean_object* v_a_3048_; lean_object* v___x_3050_; uint8_t v_isShared_3051_; uint8_t v_isSharedCheck_3056_; 
v_a_3048_ = lean_ctor_get(v___x_3047_, 0);
v_isSharedCheck_3056_ = !lean_is_exclusive(v___x_3047_);
if (v_isSharedCheck_3056_ == 0)
{
v___x_3050_ = v___x_3047_;
v_isShared_3051_ = v_isSharedCheck_3056_;
goto v_resetjp_3049_;
}
else
{
lean_inc(v_a_3048_);
lean_dec(v___x_3047_);
v___x_3050_ = lean_box(0);
v_isShared_3051_ = v_isSharedCheck_3056_;
goto v_resetjp_3049_;
}
v_resetjp_3049_:
{
lean_object* v___x_3052_; lean_object* v___x_3054_; 
v___x_3052_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3052_, 0, v_a_3048_);
if (v_isShared_3051_ == 0)
{
lean_ctor_set(v___x_3050_, 0, v___x_3052_);
v___x_3054_ = v___x_3050_;
goto v_reusejp_3053_;
}
else
{
lean_object* v_reuseFailAlloc_3055_; 
v_reuseFailAlloc_3055_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3055_, 0, v___x_3052_);
v___x_3054_ = v_reuseFailAlloc_3055_;
goto v_reusejp_3053_;
}
v_reusejp_3053_:
{
return v___x_3054_;
}
}
}
else
{
lean_object* v_a_3057_; lean_object* v___x_3059_; uint8_t v_isShared_3060_; uint8_t v_isSharedCheck_3064_; 
v_a_3057_ = lean_ctor_get(v___x_3047_, 0);
v_isSharedCheck_3064_ = !lean_is_exclusive(v___x_3047_);
if (v_isSharedCheck_3064_ == 0)
{
v___x_3059_ = v___x_3047_;
v_isShared_3060_ = v_isSharedCheck_3064_;
goto v_resetjp_3058_;
}
else
{
lean_inc(v_a_3057_);
lean_dec(v___x_3047_);
v___x_3059_ = lean_box(0);
v_isShared_3060_ = v_isSharedCheck_3064_;
goto v_resetjp_3058_;
}
v_resetjp_3058_:
{
lean_object* v___x_3062_; 
if (v_isShared_3060_ == 0)
{
v___x_3062_ = v___x_3059_;
goto v_reusejp_3061_;
}
else
{
lean_object* v_reuseFailAlloc_3063_; 
v_reuseFailAlloc_3063_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3063_, 0, v_a_3057_);
v___x_3062_ = v_reuseFailAlloc_3063_;
goto v_reusejp_3061_;
}
v_reusejp_3061_:
{
return v___x_3062_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__4___lam__0___boxed(lean_object* v___x_3065_, lean_object* v_snd_3066_, lean_object* v_____r_3067_, lean_object* v_new_3068_, lean_object* v___y_3069_, lean_object* v___y_3070_, lean_object* v___y_3071_){
_start:
{
lean_object* v_res_3072_; 
v_res_3072_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__4___lam__0(v___x_3065_, v_snd_3066_, v_____r_3067_, v_new_3068_, v___y_3069_, v___y_3070_);
lean_dec(v___y_3070_);
lean_dec_ref(v___y_3069_);
lean_dec_ref(v___x_3065_);
return v_res_3072_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__4___closed__4(void){
_start:
{
lean_object* v___x_3077_; lean_object* v___x_3078_; 
v___x_3077_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_Stained_toFMVarId___closed__0));
v___x_3078_ = lean_array_get_size(v___x_3077_);
return v___x_3078_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__4(lean_object* v___x_3079_, lean_object* v___x_3080_, lean_object* v___x_3081_, lean_object* v___x_3082_, uint8_t v___x_3083_, lean_object* v_as_3084_, size_t v_sz_3085_, size_t v_i_3086_, lean_object* v_b_3087_, lean_object* v___y_3088_, lean_object* v___y_3089_){
_start:
{
lean_object* v___y_3092_; lean_object* v___y_3115_; lean_object* v___y_3116_; lean_object* v___y_3117_; lean_object* v___y_3118_; lean_object* v___y_3119_; lean_object* v___y_3120_; lean_object* v___y_3121_; lean_object* v___y_3122_; uint8_t v___x_3137_; 
v___x_3137_ = lean_usize_dec_lt(v_i_3086_, v_sz_3085_);
if (v___x_3137_ == 0)
{
lean_object* v___x_3138_; 
v___x_3138_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3138_, 0, v_b_3087_);
return v___x_3138_;
}
else
{
lean_object* v_a_3139_; lean_object* v_fst_3140_; lean_object* v_snd_3141_; lean_object* v___x_3142_; lean_object* v___x_3143_; lean_object* v___x_3144_; lean_object* v___x_3145_; lean_object* v___f_3146_; uint8_t v___y_3148_; uint8_t v___y_3175_; lean_object* v___x_3178_; lean_object* v___x_3179_; lean_object* v___x_3180_; uint8_t v___x_3181_; 
v_a_3139_ = lean_array_uget_borrowed(v_as_3084_, v_i_3086_);
v_fst_3140_ = lean_ctor_get(v_a_3139_, 0);
v_snd_3141_ = lean_ctor_get(v_a_3139_, 1);
v___x_3142_ = lean_unsigned_to_nat(1u);
v___x_3143_ = lean_mk_empty_array_with_capacity(v___x_3142_);
lean_inc(v_fst_3140_);
v___x_3144_ = lean_array_push(v___x_3143_, v_fst_3140_);
v___x_3145_ = lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_reallyPersist(v___x_3144_, v___x_3079_, v___x_3080_, v___x_3081_, v___x_3082_);
lean_dec_ref(v___x_3144_);
lean_inc(v_snd_3141_);
lean_inc_ref(v___x_3145_);
v___f_3146_ = lean_alloc_closure((void*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__4___lam__0___boxed), 7, 2);
lean_closure_set(v___f_3146_, 0, v___x_3145_);
lean_closure_set(v___f_3146_, 1, v_snd_3141_);
v___x_3178_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_Stained_toFMVarId___closed__0));
v___x_3179_ = lean_array_get_size(v___x_3145_);
v___x_3180_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__4___closed__4, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__4___closed__4_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__4___closed__4);
v___x_3181_ = lean_nat_dec_eq(v___x_3179_, v___x_3180_);
if (v___x_3181_ == 0)
{
v___y_3175_ = v___x_3083_;
goto v___jp_3174_;
}
else
{
uint8_t v___x_3182_; 
v___x_3182_ = lp_mathlib_Array_isEqvAux___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__3___redArg(v___x_3145_, v___x_3178_, v___x_3179_);
v___y_3175_ = v___x_3182_;
goto v___jp_3174_;
}
v___jp_3147_:
{
lean_object* v_fst_3149_; lean_object* v_snd_3150_; lean_object* v_stained_3151_; lean_object* v_stx_3152_; lean_object* v___x_3153_; lean_object* v___f_3154_; lean_object* v___x_3155_; lean_object* v___x_3156_; lean_object* v___x_3157_; lean_object* v___x_3158_; lean_object* v___x_3159_; lean_object* v___x_3160_; lean_object* v___x_3161_; lean_object* v___x_3162_; lean_object* v___x_3163_; lean_object* v___x_3164_; lean_object* v___x_3165_; lean_object* v___x_3166_; 
v_fst_3149_ = lean_ctor_get(v_fst_3140_, 0);
v_snd_3150_ = lean_ctor_get(v_fst_3140_, 1);
v_stained_3151_ = lean_ctor_get(v_snd_3141_, 0);
v_stx_3152_ = lean_ctor_get(v_snd_3141_, 1);
lean_inc(v_a_3139_);
v___x_3153_ = lean_array_push(v_b_3087_, v_a_3139_);
v___f_3154_ = lean_alloc_closure((void*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__4___lam__1___boxed), 6, 2);
lean_closure_set(v___f_3154_, 0, v___f_3146_);
lean_closure_set(v___f_3154_, 1, v___x_3153_);
v___x_3155_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__4___closed__0));
v___x_3156_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__4___closed__1));
lean_inc(v_fst_3149_);
v___x_3157_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_fst_3149_, v___y_3148_);
v___x_3158_ = lean_string_append(v___x_3156_, v___x_3157_);
lean_dec_ref(v___x_3157_);
v___x_3159_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__4___closed__2));
v___x_3160_ = lean_string_append(v___x_3158_, v___x_3159_);
lean_inc(v_snd_3150_);
v___x_3161_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_snd_3150_, v___y_3148_);
v___x_3162_ = lean_string_append(v___x_3160_, v___x_3161_);
lean_dec_ref(v___x_3161_);
v___x_3163_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__4___closed__3));
v___x_3164_ = lean_string_append(v___x_3162_, v___x_3163_);
v___x_3165_ = lean_string_append(v___x_3156_, v___x_3164_);
lean_dec_ref(v___x_3164_);
v___x_3166_ = lean_string_append(v___x_3165_, v___x_3159_);
switch(lean_obj_tag(v_stained_3151_))
{
case 0:
{
lean_object* v_a_3167_; lean_object* v___x_3168_; 
v_a_3167_ = lean_ctor_get(v_stained_3151_, 0);
lean_inc(v_a_3167_);
v___x_3168_ = l_Lean_Name_toString(v_a_3167_, v___y_3148_);
lean_inc(v_stx_3152_);
v___y_3115_ = v___f_3154_;
v___y_3116_ = v___x_3155_;
v___y_3117_ = v_stx_3152_;
v___y_3118_ = v___x_3156_;
v___y_3119_ = v___x_3159_;
v___y_3120_ = v___x_3163_;
v___y_3121_ = v___x_3166_;
v___y_3122_ = v___x_3168_;
goto v___jp_3114_;
}
case 1:
{
lean_object* v___x_3169_; 
v___x_3169_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instToStringStained___lam__0___closed__0));
lean_inc(v_stx_3152_);
v___y_3115_ = v___f_3154_;
v___y_3116_ = v___x_3155_;
v___y_3117_ = v_stx_3152_;
v___y_3118_ = v___x_3156_;
v___y_3119_ = v___x_3159_;
v___y_3120_ = v___x_3163_;
v___y_3121_ = v___x_3166_;
v___y_3122_ = v___x_3169_;
goto v___jp_3114_;
}
default: 
{
lean_object* v___x_3170_; 
v___x_3170_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instToStringStained___lam__0___closed__1));
lean_inc(v_stx_3152_);
v___y_3115_ = v___f_3154_;
v___y_3116_ = v___x_3155_;
v___y_3117_ = v_stx_3152_;
v___y_3118_ = v___x_3156_;
v___y_3119_ = v___x_3159_;
v___y_3120_ = v___x_3163_;
v___y_3121_ = v___x_3166_;
v___y_3122_ = v___x_3170_;
goto v___jp_3114_;
}
}
}
v___jp_3171_:
{
lean_object* v___x_3172_; lean_object* v___x_3173_; 
v___x_3172_ = lean_box(0);
lean_inc(v_snd_3141_);
v___x_3173_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__4___lam__0(v___x_3145_, v_snd_3141_, v___x_3172_, v_b_3087_, v___y_3088_, v___y_3089_);
lean_dec_ref(v___x_3145_);
v___y_3092_ = v___x_3173_;
goto v___jp_3091_;
}
v___jp_3174_:
{
if (v___y_3175_ == 0)
{
lean_dec_ref(v___f_3146_);
goto v___jp_3171_;
}
else
{
lean_object* v___x_3176_; uint8_t v___x_3177_; 
v___x_3176_ = lean_box(0);
v___x_3177_ = lp_mathlib_List_beq___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__2(v___x_3080_, v___x_3176_);
if (v___x_3177_ == 0)
{
lean_dec_ref(v___x_3145_);
v___y_3148_ = v___y_3175_;
goto v___jp_3147_;
}
else
{
if (v___x_3083_ == 0)
{
lean_dec_ref(v___f_3146_);
goto v___jp_3171_;
}
else
{
lean_dec_ref(v___x_3145_);
v___y_3148_ = v___x_3083_;
goto v___jp_3147_;
}
}
}
}
}
v___jp_3091_:
{
if (lean_obj_tag(v___y_3092_) == 0)
{
lean_object* v_a_3093_; lean_object* v___x_3095_; uint8_t v_isShared_3096_; uint8_t v_isSharedCheck_3105_; 
v_a_3093_ = lean_ctor_get(v___y_3092_, 0);
v_isSharedCheck_3105_ = !lean_is_exclusive(v___y_3092_);
if (v_isSharedCheck_3105_ == 0)
{
v___x_3095_ = v___y_3092_;
v_isShared_3096_ = v_isSharedCheck_3105_;
goto v_resetjp_3094_;
}
else
{
lean_inc(v_a_3093_);
lean_dec(v___y_3092_);
v___x_3095_ = lean_box(0);
v_isShared_3096_ = v_isSharedCheck_3105_;
goto v_resetjp_3094_;
}
v_resetjp_3094_:
{
if (lean_obj_tag(v_a_3093_) == 0)
{
lean_object* v_a_3097_; lean_object* v___x_3099_; 
v_a_3097_ = lean_ctor_get(v_a_3093_, 0);
lean_inc(v_a_3097_);
lean_dec_ref_known(v_a_3093_, 1);
if (v_isShared_3096_ == 0)
{
lean_ctor_set(v___x_3095_, 0, v_a_3097_);
v___x_3099_ = v___x_3095_;
goto v_reusejp_3098_;
}
else
{
lean_object* v_reuseFailAlloc_3100_; 
v_reuseFailAlloc_3100_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3100_, 0, v_a_3097_);
v___x_3099_ = v_reuseFailAlloc_3100_;
goto v_reusejp_3098_;
}
v_reusejp_3098_:
{
return v___x_3099_;
}
}
else
{
lean_object* v_a_3101_; size_t v___x_3102_; size_t v___x_3103_; 
lean_del_object(v___x_3095_);
v_a_3101_ = lean_ctor_get(v_a_3093_, 0);
lean_inc(v_a_3101_);
lean_dec_ref_known(v_a_3093_, 1);
v___x_3102_ = ((size_t)1ULL);
v___x_3103_ = lean_usize_add(v_i_3086_, v___x_3102_);
v_i_3086_ = v___x_3103_;
v_b_3087_ = v_a_3101_;
goto _start;
}
}
}
else
{
lean_object* v_a_3106_; lean_object* v___x_3108_; uint8_t v_isShared_3109_; uint8_t v_isSharedCheck_3113_; 
v_a_3106_ = lean_ctor_get(v___y_3092_, 0);
v_isSharedCheck_3113_ = !lean_is_exclusive(v___y_3092_);
if (v_isSharedCheck_3113_ == 0)
{
v___x_3108_ = v___y_3092_;
v_isShared_3109_ = v_isSharedCheck_3113_;
goto v_resetjp_3107_;
}
else
{
lean_inc(v_a_3106_);
lean_dec(v___y_3092_);
v___x_3108_ = lean_box(0);
v_isShared_3109_ = v_isSharedCheck_3113_;
goto v_resetjp_3107_;
}
v_resetjp_3107_:
{
lean_object* v___x_3111_; 
if (v_isShared_3109_ == 0)
{
v___x_3111_ = v___x_3108_;
goto v_reusejp_3110_;
}
else
{
lean_object* v_reuseFailAlloc_3112_; 
v_reuseFailAlloc_3112_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3112_, 0, v_a_3106_);
v___x_3111_ = v_reuseFailAlloc_3112_;
goto v_reusejp_3110_;
}
v_reusejp_3110_:
{
return v___x_3111_;
}
}
}
}
v___jp_3114_:
{
lean_object* v___x_3123_; lean_object* v___x_3124_; lean_object* v___x_3125_; lean_object* v___x_3126_; lean_object* v___x_3127_; lean_object* v___x_3128_; lean_object* v___x_3129_; lean_object* v___x_3130_; lean_object* v___x_3131_; lean_object* v___x_3132_; lean_object* v___x_3133_; lean_object* v___x_3134_; lean_object* v___x_28993__overap_3135_; lean_object* v___x_3136_; 
v___x_3123_ = lean_string_append(v___y_3118_, v___y_3122_);
lean_dec_ref(v___y_3122_);
v___x_3124_ = lean_string_append(v___x_3123_, v___y_3119_);
v___x_3125_ = lean_box(0);
v___x_3126_ = l_Lean_Syntax_formatStx(v___y_3117_, v___x_3125_, v___x_3083_);
v___x_3127_ = l_Std_Format_defWidth;
v___x_3128_ = lean_unsigned_to_nat(0u);
v___x_3129_ = l_Std_Format_pretty(v___x_3126_, v___x_3127_, v___x_3128_, v___x_3128_);
v___x_3130_ = lean_string_append(v___x_3124_, v___x_3129_);
lean_dec_ref(v___x_3129_);
v___x_3131_ = lean_string_append(v___x_3130_, v___y_3120_);
v___x_3132_ = lean_string_append(v___y_3121_, v___x_3131_);
lean_dec_ref(v___x_3131_);
v___x_3133_ = lean_string_append(v___x_3132_, v___y_3120_);
lean_inc_ref(v___y_3116_);
v___x_3134_ = lean_string_append(v___y_3116_, v___x_3133_);
lean_dec_ref(v___x_3133_);
v___x_28993__overap_3135_ = lean_dbg_trace(v___x_3134_, v___y_3115_);
lean_inc(v___y_3089_);
lean_inc_ref(v___y_3088_);
v___x_3136_ = lean_apply_3(v___x_28993__overap_3135_, v___y_3088_, v___y_3089_, lean_box(0));
v___y_3092_ = v___x_3136_;
goto v___jp_3091_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__4___boxed(lean_object* v___x_3183_, lean_object* v___x_3184_, lean_object* v___x_3185_, lean_object* v___x_3186_, lean_object* v___x_3187_, lean_object* v_as_3188_, lean_object* v_sz_3189_, lean_object* v_i_3190_, lean_object* v_b_3191_, lean_object* v___y_3192_, lean_object* v___y_3193_, lean_object* v___y_3194_){
_start:
{
uint8_t v___x_34931__boxed_3195_; size_t v_sz_boxed_3196_; size_t v_i_boxed_3197_; lean_object* v_res_3198_; 
v___x_34931__boxed_3195_ = lean_unbox(v___x_3187_);
v_sz_boxed_3196_ = lean_unbox_usize(v_sz_3189_);
lean_dec(v_sz_3189_);
v_i_boxed_3197_ = lean_unbox_usize(v_i_3190_);
lean_dec(v_i_3190_);
v_res_3198_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__4(v___x_3183_, v___x_3184_, v___x_3185_, v___x_3186_, v___x_34931__boxed_3195_, v_as_3188_, v_sz_boxed_3196_, v_i_boxed_3197_, v_b_3191_, v___y_3192_, v___y_3193_);
lean_dec(v___y_3193_);
lean_dec_ref(v___y_3192_);
lean_dec_ref(v_as_3188_);
lean_dec_ref(v___x_3186_);
lean_dec_ref(v___x_3185_);
lean_dec(v___x_3184_);
lean_dec(v___x_3183_);
return v_res_3198_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__16_spec__18(lean_object* v___x_3201_, lean_object* v___x_3202_, lean_object* v___x_3203_, lean_object* v___x_3204_, uint8_t v___x_3205_, lean_object* v___x_3206_, lean_object* v___x_3207_, uint8_t v___y_3208_, lean_object* v_a_3209_, lean_object* v_a_3210_, lean_object* v_a_3211_, lean_object* v___y_3212_, lean_object* v___y_3213_){
_start:
{
if (lean_obj_tag(v_a_3210_) == 0)
{
lean_object* v___x_3215_; lean_object* v___x_3216_; 
lean_dec(v___x_3207_);
lean_dec_ref(v___x_3203_);
lean_dec(v___x_3201_);
v___x_3215_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3215_, 0, v_a_3211_);
v___x_3216_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3216_, 0, v___x_3215_);
return v___x_3216_;
}
else
{
lean_object* v_key_3217_; lean_object* v_tail_3218_; lean_object* v_stains_3220_; lean_object* v_msgs_3221_; lean_object* v___y_3222_; lean_object* v___y_3223_; lean_object* v_fst_3239_; lean_object* v_snd_3240_; lean_object* v___y_3242_; 
v_key_3217_ = lean_ctor_get(v_a_3210_, 0);
lean_inc(v_key_3217_);
v_tail_3218_ = lean_ctor_get(v_a_3210_, 2);
lean_inc(v_tail_3218_);
lean_dec_ref_known(v_a_3210_, 3);
v_fst_3239_ = lean_ctor_get(v_a_3211_, 0);
lean_inc(v_fst_3239_);
v_snd_3240_ = lean_ctor_get(v_a_3211_, 1);
lean_inc(v_snd_3240_);
lean_dec_ref(v_a_3211_);
if (v___y_3208_ == 0)
{
uint8_t v___x_3255_; 
v___x_3255_ = lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_usesGoal_x3f(v___x_3206_);
if (v___x_3255_ == 0)
{
lean_object* v___x_3256_; 
lean_dec(v_key_3217_);
lean_inc(v___x_3207_);
v___x_3256_ = lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained(v___x_3207_);
v___y_3242_ = v___x_3256_;
goto v___jp_3241_;
}
else
{
lean_object* v___x_3257_; lean_object* v___x_3258_; lean_object* v___x_3259_; 
lean_inc(v___x_3207_);
v___x_3257_ = lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained(v___x_3207_);
v___x_3258_ = lean_box(0);
v___x_3259_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__0___redArg(v___x_3257_, v_key_3217_, v___x_3258_);
v___y_3242_ = v___x_3259_;
goto v___jp_3241_;
}
}
else
{
lean_object* v___x_3260_; lean_object* v_a_3261_; 
lean_inc(v___x_3201_);
lean_inc_ref(v___x_3203_);
lean_inc(v___x_3207_);
v___x_3260_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__15___redArg(v_key_3217_, v_a_3209_, v___x_3207_, v___x_3203_, v___x_3201_, v___x_3204_, v___x_3202_, v_fst_3239_);
v_a_3261_ = lean_ctor_get(v___x_3260_, 0);
lean_inc(v_a_3261_);
lean_dec_ref(v___x_3260_);
v_stains_3220_ = v_a_3261_;
v_msgs_3221_ = v_snd_3240_;
v___y_3222_ = v___y_3212_;
v___y_3223_ = v___y_3213_;
goto v___jp_3219_;
}
v___jp_3219_:
{
lean_object* v___x_3224_; size_t v_sz_3225_; size_t v___x_3226_; lean_object* v___x_3227_; 
v___x_3224_ = ((lean_object*)(lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__16_spec__18___closed__0));
v_sz_3225_ = lean_array_size(v_stains_3220_);
v___x_3226_ = ((size_t)0ULL);
v___x_3227_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__4(v___x_3201_, v___x_3202_, v___x_3203_, v___x_3204_, v___x_3205_, v_stains_3220_, v_sz_3225_, v___x_3226_, v___x_3224_, v___y_3222_, v___y_3223_);
lean_dec_ref(v_stains_3220_);
if (lean_obj_tag(v___x_3227_) == 0)
{
lean_object* v_a_3228_; lean_object* v___x_3229_; 
v_a_3228_ = lean_ctor_get(v___x_3227_, 0);
lean_inc(v_a_3228_);
lean_dec_ref_known(v___x_3227_, 1);
v___x_3229_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3229_, 0, v_a_3228_);
lean_ctor_set(v___x_3229_, 1, v_msgs_3221_);
v_a_3210_ = v_tail_3218_;
v_a_3211_ = v___x_3229_;
goto _start;
}
else
{
lean_object* v_a_3231_; lean_object* v___x_3233_; uint8_t v_isShared_3234_; uint8_t v_isSharedCheck_3238_; 
lean_dec_ref(v_msgs_3221_);
lean_dec(v_tail_3218_);
lean_dec(v___x_3207_);
lean_dec_ref(v___x_3203_);
lean_dec(v___x_3201_);
v_a_3231_ = lean_ctor_get(v___x_3227_, 0);
v_isSharedCheck_3238_ = !lean_is_exclusive(v___x_3227_);
if (v_isSharedCheck_3238_ == 0)
{
v___x_3233_ = v___x_3227_;
v_isShared_3234_ = v_isSharedCheck_3238_;
goto v_resetjp_3232_;
}
else
{
lean_inc(v_a_3231_);
lean_dec(v___x_3227_);
v___x_3233_ = lean_box(0);
v_isShared_3234_ = v_isSharedCheck_3238_;
goto v_resetjp_3232_;
}
v_resetjp_3232_:
{
lean_object* v___x_3236_; 
if (v_isShared_3234_ == 0)
{
v___x_3236_ = v___x_3233_;
goto v_reusejp_3235_;
}
else
{
lean_object* v_reuseFailAlloc_3237_; 
v_reuseFailAlloc_3237_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3237_, 0, v_a_3231_);
v___x_3236_ = v_reuseFailAlloc_3237_;
goto v_reusejp_3235_;
}
v_reusejp_3235_:
{
return v___x_3236_;
}
}
}
}
v___jp_3241_:
{
lean_object* v___x_3243_; uint8_t v___x_3244_; 
v___x_3243_ = lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible;
v___x_3244_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__0___redArg(v___x_3243_, v___x_3206_);
if (v___x_3244_ == 0)
{
lean_object* v___x_3245_; 
lean_inc(v___x_3207_);
v___x_3245_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__14___redArg(v___y_3242_, v_fst_3239_, v___x_3207_, v___x_3203_, v___x_3201_, v_snd_3240_, v___y_3212_, v___y_3213_);
lean_dec_ref(v___y_3242_);
if (lean_obj_tag(v___x_3245_) == 0)
{
lean_object* v_a_3246_; 
v_a_3246_ = lean_ctor_get(v___x_3245_, 0);
lean_inc(v_a_3246_);
lean_dec_ref_known(v___x_3245_, 1);
v_stains_3220_ = v_fst_3239_;
v_msgs_3221_ = v_a_3246_;
v___y_3222_ = v___y_3212_;
v___y_3223_ = v___y_3213_;
goto v___jp_3219_;
}
else
{
lean_object* v_a_3247_; lean_object* v___x_3249_; uint8_t v_isShared_3250_; uint8_t v_isSharedCheck_3254_; 
lean_dec(v_fst_3239_);
lean_dec(v_tail_3218_);
lean_dec(v___x_3207_);
lean_dec_ref(v___x_3203_);
lean_dec(v___x_3201_);
v_a_3247_ = lean_ctor_get(v___x_3245_, 0);
v_isSharedCheck_3254_ = !lean_is_exclusive(v___x_3245_);
if (v_isSharedCheck_3254_ == 0)
{
v___x_3249_ = v___x_3245_;
v_isShared_3250_ = v_isSharedCheck_3254_;
goto v_resetjp_3248_;
}
else
{
lean_inc(v_a_3247_);
lean_dec(v___x_3245_);
v___x_3249_ = lean_box(0);
v_isShared_3250_ = v_isSharedCheck_3254_;
goto v_resetjp_3248_;
}
v_resetjp_3248_:
{
lean_object* v___x_3252_; 
if (v_isShared_3250_ == 0)
{
v___x_3252_ = v___x_3249_;
goto v_reusejp_3251_;
}
else
{
lean_object* v_reuseFailAlloc_3253_; 
v_reuseFailAlloc_3253_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3253_, 0, v_a_3247_);
v___x_3252_ = v_reuseFailAlloc_3253_;
goto v_reusejp_3251_;
}
v_reusejp_3251_:
{
return v___x_3252_;
}
}
}
}
else
{
lean_dec_ref(v___y_3242_);
v_stains_3220_ = v_fst_3239_;
v_msgs_3221_ = v_snd_3240_;
v___y_3222_ = v___y_3212_;
v___y_3223_ = v___y_3213_;
goto v___jp_3219_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__16_spec__18___boxed(lean_object* v___x_3262_, lean_object* v___x_3263_, lean_object* v___x_3264_, lean_object* v___x_3265_, lean_object* v___x_3266_, lean_object* v___x_3267_, lean_object* v___x_3268_, lean_object* v___y_3269_, lean_object* v_a_3270_, lean_object* v_a_3271_, lean_object* v_a_3272_, lean_object* v___y_3273_, lean_object* v___y_3274_, lean_object* v___y_3275_){
_start:
{
uint8_t v___x_35137__boxed_3276_; uint8_t v___y_35140__boxed_3277_; lean_object* v_res_3278_; 
v___x_35137__boxed_3276_ = lean_unbox(v___x_3266_);
v___y_35140__boxed_3277_ = lean_unbox(v___y_3269_);
v_res_3278_ = lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__16_spec__18(v___x_3262_, v___x_3263_, v___x_3264_, v___x_3265_, v___x_35137__boxed_3276_, v___x_3267_, v___x_3268_, v___y_35140__boxed_3277_, v_a_3270_, v_a_3271_, v_a_3272_, v___y_3273_, v___y_3274_);
lean_dec(v___y_3274_);
lean_dec_ref(v___y_3273_);
lean_dec_ref(v_a_3270_);
lean_dec(v___x_3267_);
lean_dec_ref(v___x_3265_);
lean_dec(v___x_3263_);
return v_res_3278_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__16(lean_object* v___x_3279_, lean_object* v___x_3280_, lean_object* v___x_3281_, lean_object* v___x_3282_, uint8_t v___x_3283_, lean_object* v_a_3284_, lean_object* v___x_3285_, lean_object* v___x_3286_, uint8_t v___y_3287_, lean_object* v_a_3288_, lean_object* v_a_3289_, lean_object* v___y_3290_, lean_object* v___y_3291_){
_start:
{
if (lean_obj_tag(v_a_3288_) == 0)
{
lean_object* v___x_3293_; lean_object* v___x_3294_; 
lean_dec(v___x_3285_);
lean_dec_ref(v___x_3281_);
lean_dec(v___x_3279_);
v___x_3293_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3293_, 0, v_a_3289_);
v___x_3294_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3294_, 0, v___x_3293_);
return v___x_3294_;
}
else
{
lean_object* v_key_3295_; lean_object* v_tail_3296_; lean_object* v_stains_3298_; lean_object* v_msgs_3299_; lean_object* v___y_3300_; lean_object* v___y_3301_; lean_object* v_fst_3317_; lean_object* v_snd_3318_; lean_object* v___y_3320_; 
v_key_3295_ = lean_ctor_get(v_a_3288_, 0);
lean_inc(v_key_3295_);
v_tail_3296_ = lean_ctor_get(v_a_3288_, 2);
lean_inc(v_tail_3296_);
lean_dec_ref_known(v_a_3288_, 3);
v_fst_3317_ = lean_ctor_get(v_a_3289_, 0);
lean_inc(v_fst_3317_);
v_snd_3318_ = lean_ctor_get(v_a_3289_, 1);
lean_inc(v_snd_3318_);
lean_dec_ref(v_a_3289_);
if (v___y_3287_ == 0)
{
uint8_t v___x_3333_; 
v___x_3333_ = lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_usesGoal_x3f(v___x_3286_);
if (v___x_3333_ == 0)
{
lean_object* v___x_3334_; 
lean_dec(v_key_3295_);
lean_inc(v___x_3285_);
v___x_3334_ = lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained(v___x_3285_);
v___y_3320_ = v___x_3334_;
goto v___jp_3319_;
}
else
{
lean_object* v___x_3335_; lean_object* v___x_3336_; lean_object* v___x_3337_; 
lean_inc(v___x_3285_);
v___x_3335_ = lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained(v___x_3285_);
v___x_3336_ = lean_box(0);
v___x_3337_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_toStained_spec__0___redArg(v___x_3335_, v_key_3295_, v___x_3336_);
v___y_3320_ = v___x_3337_;
goto v___jp_3319_;
}
}
else
{
lean_object* v___x_3338_; lean_object* v_a_3339_; 
lean_inc(v___x_3279_);
lean_inc_ref(v___x_3281_);
lean_inc(v___x_3285_);
v___x_3338_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__15___redArg(v_key_3295_, v_a_3284_, v___x_3285_, v___x_3281_, v___x_3279_, v___x_3282_, v___x_3280_, v_fst_3317_);
v_a_3339_ = lean_ctor_get(v___x_3338_, 0);
lean_inc(v_a_3339_);
lean_dec_ref(v___x_3338_);
v_stains_3298_ = v_a_3339_;
v_msgs_3299_ = v_snd_3318_;
v___y_3300_ = v___y_3290_;
v___y_3301_ = v___y_3291_;
goto v___jp_3297_;
}
v___jp_3297_:
{
lean_object* v___x_3302_; size_t v_sz_3303_; size_t v___x_3304_; lean_object* v___x_3305_; 
v___x_3302_ = ((lean_object*)(lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__16_spec__18___closed__0));
v_sz_3303_ = lean_array_size(v_stains_3298_);
v___x_3304_ = ((size_t)0ULL);
v___x_3305_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__4(v___x_3279_, v___x_3280_, v___x_3281_, v___x_3282_, v___x_3283_, v_stains_3298_, v_sz_3303_, v___x_3304_, v___x_3302_, v___y_3300_, v___y_3301_);
lean_dec_ref(v_stains_3298_);
if (lean_obj_tag(v___x_3305_) == 0)
{
lean_object* v_a_3306_; lean_object* v___x_3307_; lean_object* v___x_3308_; 
v_a_3306_ = lean_ctor_get(v___x_3305_, 0);
lean_inc(v_a_3306_);
lean_dec_ref_known(v___x_3305_, 1);
v___x_3307_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3307_, 0, v_a_3306_);
lean_ctor_set(v___x_3307_, 1, v_msgs_3299_);
v___x_3308_ = lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__16_spec__18(v___x_3279_, v___x_3280_, v___x_3281_, v___x_3282_, v___x_3283_, v___x_3286_, v___x_3285_, v___y_3287_, v_a_3284_, v_tail_3296_, v___x_3307_, v___y_3290_, v___y_3291_);
return v___x_3308_;
}
else
{
lean_object* v_a_3309_; lean_object* v___x_3311_; uint8_t v_isShared_3312_; uint8_t v_isSharedCheck_3316_; 
lean_dec_ref(v_msgs_3299_);
lean_dec(v_tail_3296_);
lean_dec(v___x_3285_);
lean_dec_ref(v___x_3281_);
lean_dec(v___x_3279_);
v_a_3309_ = lean_ctor_get(v___x_3305_, 0);
v_isSharedCheck_3316_ = !lean_is_exclusive(v___x_3305_);
if (v_isSharedCheck_3316_ == 0)
{
v___x_3311_ = v___x_3305_;
v_isShared_3312_ = v_isSharedCheck_3316_;
goto v_resetjp_3310_;
}
else
{
lean_inc(v_a_3309_);
lean_dec(v___x_3305_);
v___x_3311_ = lean_box(0);
v_isShared_3312_ = v_isSharedCheck_3316_;
goto v_resetjp_3310_;
}
v_resetjp_3310_:
{
lean_object* v___x_3314_; 
if (v_isShared_3312_ == 0)
{
v___x_3314_ = v___x_3311_;
goto v_reusejp_3313_;
}
else
{
lean_object* v_reuseFailAlloc_3315_; 
v_reuseFailAlloc_3315_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3315_, 0, v_a_3309_);
v___x_3314_ = v_reuseFailAlloc_3315_;
goto v_reusejp_3313_;
}
v_reusejp_3313_:
{
return v___x_3314_;
}
}
}
}
v___jp_3319_:
{
lean_object* v___x_3321_; uint8_t v___x_3322_; 
v___x_3321_ = lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible;
v___x_3322_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__0___redArg(v___x_3321_, v___x_3286_);
if (v___x_3322_ == 0)
{
lean_object* v___x_3323_; 
lean_inc(v___x_3285_);
v___x_3323_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__14___redArg(v___y_3320_, v_fst_3317_, v___x_3285_, v___x_3281_, v___x_3279_, v_snd_3318_, v___y_3290_, v___y_3291_);
lean_dec_ref(v___y_3320_);
if (lean_obj_tag(v___x_3323_) == 0)
{
lean_object* v_a_3324_; 
v_a_3324_ = lean_ctor_get(v___x_3323_, 0);
lean_inc(v_a_3324_);
lean_dec_ref_known(v___x_3323_, 1);
v_stains_3298_ = v_fst_3317_;
v_msgs_3299_ = v_a_3324_;
v___y_3300_ = v___y_3290_;
v___y_3301_ = v___y_3291_;
goto v___jp_3297_;
}
else
{
lean_object* v_a_3325_; lean_object* v___x_3327_; uint8_t v_isShared_3328_; uint8_t v_isSharedCheck_3332_; 
lean_dec(v_fst_3317_);
lean_dec(v_tail_3296_);
lean_dec(v___x_3285_);
lean_dec_ref(v___x_3281_);
lean_dec(v___x_3279_);
v_a_3325_ = lean_ctor_get(v___x_3323_, 0);
v_isSharedCheck_3332_ = !lean_is_exclusive(v___x_3323_);
if (v_isSharedCheck_3332_ == 0)
{
v___x_3327_ = v___x_3323_;
v_isShared_3328_ = v_isSharedCheck_3332_;
goto v_resetjp_3326_;
}
else
{
lean_inc(v_a_3325_);
lean_dec(v___x_3323_);
v___x_3327_ = lean_box(0);
v_isShared_3328_ = v_isSharedCheck_3332_;
goto v_resetjp_3326_;
}
v_resetjp_3326_:
{
lean_object* v___x_3330_; 
if (v_isShared_3328_ == 0)
{
v___x_3330_ = v___x_3327_;
goto v_reusejp_3329_;
}
else
{
lean_object* v_reuseFailAlloc_3331_; 
v_reuseFailAlloc_3331_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3331_, 0, v_a_3325_);
v___x_3330_ = v_reuseFailAlloc_3331_;
goto v_reusejp_3329_;
}
v_reusejp_3329_:
{
return v___x_3330_;
}
}
}
}
else
{
lean_dec_ref(v___y_3320_);
v_stains_3298_ = v_fst_3317_;
v_msgs_3299_ = v_snd_3318_;
v___y_3300_ = v___y_3290_;
v___y_3301_ = v___y_3291_;
goto v___jp_3297_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__16___boxed(lean_object* v___x_3340_, lean_object* v___x_3341_, lean_object* v___x_3342_, lean_object* v___x_3343_, lean_object* v___x_3344_, lean_object* v_a_3345_, lean_object* v___x_3346_, lean_object* v___x_3347_, lean_object* v___y_3348_, lean_object* v_a_3349_, lean_object* v_a_3350_, lean_object* v___y_3351_, lean_object* v___y_3352_, lean_object* v___y_3353_){
_start:
{
uint8_t v___x_35266__boxed_3354_; uint8_t v___y_35269__boxed_3355_; lean_object* v_res_3356_; 
v___x_35266__boxed_3354_ = lean_unbox(v___x_3344_);
v___y_35269__boxed_3355_ = lean_unbox(v___y_3348_);
v_res_3356_ = lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__16(v___x_3340_, v___x_3341_, v___x_3342_, v___x_3343_, v___x_35266__boxed_3354_, v_a_3345_, v___x_3346_, v___x_3347_, v___y_35269__boxed_3355_, v_a_3349_, v_a_3350_, v___y_3351_, v___y_3352_);
lean_dec(v___y_3352_);
lean_dec_ref(v___y_3351_);
lean_dec(v___x_3347_);
lean_dec_ref(v_a_3345_);
lean_dec_ref(v___x_3343_);
lean_dec(v___x_3341_);
return v_res_3356_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__17(lean_object* v___x_3357_, lean_object* v___x_3358_, lean_object* v___x_3359_, lean_object* v___x_3360_, uint8_t v___x_3361_, lean_object* v_a_3362_, lean_object* v___x_3363_, lean_object* v___x_3364_, uint8_t v___y_3365_, lean_object* v_as_3366_, size_t v_sz_3367_, size_t v_i_3368_, lean_object* v_b_3369_, lean_object* v___y_3370_, lean_object* v___y_3371_){
_start:
{
uint8_t v___x_3373_; 
v___x_3373_ = lean_usize_dec_lt(v_i_3368_, v_sz_3367_);
if (v___x_3373_ == 0)
{
lean_object* v___x_3374_; 
lean_dec(v___x_3363_);
lean_dec_ref(v___x_3359_);
lean_dec(v___x_3357_);
v___x_3374_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3374_, 0, v_b_3369_);
return v___x_3374_;
}
else
{
lean_object* v_a_3375_; lean_object* v___x_3376_; 
v_a_3375_ = lean_array_uget_borrowed(v_as_3366_, v_i_3368_);
lean_inc(v_a_3375_);
lean_inc(v___x_3363_);
lean_inc_ref(v___x_3359_);
lean_inc(v___x_3357_);
v___x_3376_ = lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__16(v___x_3357_, v___x_3358_, v___x_3359_, v___x_3360_, v___x_3361_, v_a_3362_, v___x_3363_, v___x_3364_, v___y_3365_, v_a_3375_, v_b_3369_, v___y_3370_, v___y_3371_);
if (lean_obj_tag(v___x_3376_) == 0)
{
lean_object* v_a_3377_; lean_object* v___x_3379_; uint8_t v_isShared_3380_; uint8_t v_isSharedCheck_3389_; 
v_a_3377_ = lean_ctor_get(v___x_3376_, 0);
v_isSharedCheck_3389_ = !lean_is_exclusive(v___x_3376_);
if (v_isSharedCheck_3389_ == 0)
{
v___x_3379_ = v___x_3376_;
v_isShared_3380_ = v_isSharedCheck_3389_;
goto v_resetjp_3378_;
}
else
{
lean_inc(v_a_3377_);
lean_dec(v___x_3376_);
v___x_3379_ = lean_box(0);
v_isShared_3380_ = v_isSharedCheck_3389_;
goto v_resetjp_3378_;
}
v_resetjp_3378_:
{
if (lean_obj_tag(v_a_3377_) == 0)
{
lean_object* v_a_3381_; lean_object* v___x_3383_; 
lean_dec(v___x_3363_);
lean_dec_ref(v___x_3359_);
lean_dec(v___x_3357_);
v_a_3381_ = lean_ctor_get(v_a_3377_, 0);
lean_inc(v_a_3381_);
lean_dec_ref_known(v_a_3377_, 1);
if (v_isShared_3380_ == 0)
{
lean_ctor_set(v___x_3379_, 0, v_a_3381_);
v___x_3383_ = v___x_3379_;
goto v_reusejp_3382_;
}
else
{
lean_object* v_reuseFailAlloc_3384_; 
v_reuseFailAlloc_3384_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3384_, 0, v_a_3381_);
v___x_3383_ = v_reuseFailAlloc_3384_;
goto v_reusejp_3382_;
}
v_reusejp_3382_:
{
return v___x_3383_;
}
}
else
{
lean_object* v_a_3385_; size_t v___x_3386_; size_t v___x_3387_; 
lean_del_object(v___x_3379_);
v_a_3385_ = lean_ctor_get(v_a_3377_, 0);
lean_inc(v_a_3385_);
lean_dec_ref_known(v_a_3377_, 1);
v___x_3386_ = ((size_t)1ULL);
v___x_3387_ = lean_usize_add(v_i_3368_, v___x_3386_);
v_i_3368_ = v___x_3387_;
v_b_3369_ = v_a_3385_;
goto _start;
}
}
}
else
{
lean_object* v_a_3390_; lean_object* v___x_3392_; uint8_t v_isShared_3393_; uint8_t v_isSharedCheck_3397_; 
lean_dec(v___x_3363_);
lean_dec_ref(v___x_3359_);
lean_dec(v___x_3357_);
v_a_3390_ = lean_ctor_get(v___x_3376_, 0);
v_isSharedCheck_3397_ = !lean_is_exclusive(v___x_3376_);
if (v_isSharedCheck_3397_ == 0)
{
v___x_3392_ = v___x_3376_;
v_isShared_3393_ = v_isSharedCheck_3397_;
goto v_resetjp_3391_;
}
else
{
lean_inc(v_a_3390_);
lean_dec(v___x_3376_);
v___x_3392_ = lean_box(0);
v_isShared_3393_ = v_isSharedCheck_3397_;
goto v_resetjp_3391_;
}
v_resetjp_3391_:
{
lean_object* v___x_3395_; 
if (v_isShared_3393_ == 0)
{
v___x_3395_ = v___x_3392_;
goto v_reusejp_3394_;
}
else
{
lean_object* v_reuseFailAlloc_3396_; 
v_reuseFailAlloc_3396_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3396_, 0, v_a_3390_);
v___x_3395_ = v_reuseFailAlloc_3396_;
goto v_reusejp_3394_;
}
v_reusejp_3394_:
{
return v___x_3395_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__17___boxed(lean_object* v___x_3398_, lean_object* v___x_3399_, lean_object* v___x_3400_, lean_object* v___x_3401_, lean_object* v___x_3402_, lean_object* v_a_3403_, lean_object* v___x_3404_, lean_object* v___x_3405_, lean_object* v___y_3406_, lean_object* v_as_3407_, lean_object* v_sz_3408_, lean_object* v_i_3409_, lean_object* v_b_3410_, lean_object* v___y_3411_, lean_object* v___y_3412_, lean_object* v___y_3413_){
_start:
{
uint8_t v___x_35392__boxed_3414_; uint8_t v___y_35395__boxed_3415_; size_t v_sz_boxed_3416_; size_t v_i_boxed_3417_; lean_object* v_res_3418_; 
v___x_35392__boxed_3414_ = lean_unbox(v___x_3402_);
v___y_35395__boxed_3415_ = lean_unbox(v___y_3406_);
v_sz_boxed_3416_ = lean_unbox_usize(v_sz_3408_);
lean_dec(v_sz_3408_);
v_i_boxed_3417_ = lean_unbox_usize(v_i_3409_);
lean_dec(v_i_3409_);
v_res_3418_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__17(v___x_3398_, v___x_3399_, v___x_3400_, v___x_3401_, v___x_35392__boxed_3414_, v_a_3403_, v___x_3404_, v___x_3405_, v___y_35395__boxed_3415_, v_as_3407_, v_sz_boxed_3416_, v_i_boxed_3417_, v_b_3410_, v___y_3411_, v___y_3412_);
lean_dec(v___y_3412_);
lean_dec_ref(v___y_3411_);
lean_dec_ref(v_as_3407_);
lean_dec(v___x_3405_);
lean_dec_ref(v_a_3403_);
lean_dec_ref(v___x_3401_);
lean_dec(v___x_3399_);
return v_res_3418_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__21___lam__0(uint8_t v___x_3419_, lean_object* v_x_3420_){
_start:
{
return v___x_3419_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__21___lam__0___boxed(lean_object* v___x_3421_, lean_object* v_x_3422_){
_start:
{
uint8_t v___x_35469__boxed_3423_; uint8_t v_res_3424_; lean_object* v_r_3425_; 
v___x_35469__boxed_3423_ = lean_unbox(v___x_3421_);
v_res_3424_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__21___lam__0(v___x_35469__boxed_3423_, v_x_3422_);
lean_dec(v_x_3422_);
v_r_3425_ = lean_box(v_res_3424_);
return v_r_3425_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__21(lean_object* v_as_3426_, size_t v_sz_3427_, size_t v_i_3428_, lean_object* v_b_3429_, lean_object* v___y_3430_, lean_object* v___y_3431_){
_start:
{
lean_object* v_a_3434_; uint8_t v___x_3438_; 
v___x_3438_ = lean_usize_dec_lt(v_i_3428_, v_sz_3427_);
if (v___x_3438_ == 0)
{
lean_object* v___x_3439_; 
v___x_3439_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3439_, 0, v_b_3429_);
return v___x_3439_;
}
else
{
lean_object* v_fst_3440_; lean_object* v_snd_3441_; lean_object* v___x_3443_; uint8_t v_isShared_3444_; uint8_t v_isSharedCheck_3483_; 
v_fst_3440_ = lean_ctor_get(v_b_3429_, 0);
v_snd_3441_ = lean_ctor_get(v_b_3429_, 1);
v_isSharedCheck_3483_ = !lean_is_exclusive(v_b_3429_);
if (v_isSharedCheck_3483_ == 0)
{
v___x_3443_ = v_b_3429_;
v_isShared_3444_ = v_isSharedCheck_3483_;
goto v_resetjp_3442_;
}
else
{
lean_inc(v_snd_3441_);
lean_inc(v_fst_3440_);
lean_dec(v_b_3429_);
v___x_3443_ = lean_box(0);
v_isShared_3444_ = v_isSharedCheck_3483_;
goto v_resetjp_3442_;
}
v_resetjp_3442_:
{
lean_object* v_a_3445_; lean_object* v_stx_3446_; lean_object* v_mctxBefore_3447_; lean_object* v_mctxAfter_3448_; lean_object* v_goalsTargetedBy_3449_; lean_object* v_goalsCreatedBy_3450_; lean_object* v___x_3451_; lean_object* v___x_3452_; uint8_t v___x_3453_; 
v_a_3445_ = lean_array_uget_borrowed(v_as_3426_, v_i_3428_);
v_stx_3446_ = lean_ctor_get(v_a_3445_, 0);
v_mctxBefore_3447_ = lean_ctor_get(v_a_3445_, 2);
v_mctxAfter_3448_ = lean_ctor_get(v_a_3445_, 3);
v_goalsTargetedBy_3449_ = lean_ctor_get(v_a_3445_, 4);
v_goalsCreatedBy_3450_ = lean_ctor_get(v_a_3445_, 5);
lean_inc(v_stx_3446_);
v___x_3451_ = l_Lean_Syntax_getKind(v_stx_3446_);
v___x_3452_ = lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers;
v___x_3453_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__0___redArg(v___x_3452_, v___x_3451_);
if (v___x_3453_ == 0)
{
lean_object* v___x_3454_; lean_object* v___f_3455_; uint8_t v___y_3457_; uint8_t v___x_3476_; 
v___x_3454_ = lean_box(v___x_3453_);
v___f_3455_ = lean_alloc_closure((void*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__21___lam__0___boxed), 2, 1);
lean_closure_set(v___f_3455_, 0, v___x_3454_);
v___x_3476_ = lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f(v_stx_3446_);
if (v___x_3476_ == 0)
{
v___y_3457_ = v___x_3476_;
goto v___jp_3456_;
}
else
{
lean_object* v___x_3477_; lean_object* v___x_3478_; uint8_t v___x_3479_; 
v___x_3477_ = l_List_lengthTR___redArg(v_goalsCreatedBy_3450_);
v___x_3478_ = l_List_lengthTR___redArg(v_goalsTargetedBy_3449_);
v___x_3479_ = lean_nat_dec_eq(v___x_3477_, v___x_3478_);
lean_dec(v___x_3478_);
lean_dec(v___x_3477_);
v___y_3457_ = v___x_3479_;
goto v___jp_3456_;
}
v___jp_3456_:
{
lean_object* v___x_3458_; lean_object* v_buckets_3459_; lean_object* v___x_3461_; 
lean_inc(v_stx_3446_);
v___x_3458_ = lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_getStained_x21(v_stx_3446_, v___f_3455_);
v_buckets_3459_ = lean_ctor_get(v___x_3458_, 1);
lean_inc_ref(v_buckets_3459_);
lean_dec_ref(v___x_3458_);
if (v_isShared_3444_ == 0)
{
v___x_3461_ = v___x_3443_;
goto v_reusejp_3460_;
}
else
{
lean_object* v_reuseFailAlloc_3475_; 
v_reuseFailAlloc_3475_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3475_, 0, v_fst_3440_);
lean_ctor_set(v_reuseFailAlloc_3475_, 1, v_snd_3441_);
v___x_3461_ = v_reuseFailAlloc_3475_;
goto v_reusejp_3460_;
}
v_reusejp_3460_:
{
size_t v_sz_3462_; size_t v___x_3463_; lean_object* v___x_3464_; 
v_sz_3462_ = lean_array_size(v_buckets_3459_);
v___x_3463_ = ((size_t)0ULL);
lean_inc(v_stx_3446_);
lean_inc_ref(v_mctxBefore_3447_);
lean_inc(v_goalsTargetedBy_3449_);
v___x_3464_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__17(v_goalsTargetedBy_3449_, v_goalsCreatedBy_3450_, v_mctxBefore_3447_, v_mctxAfter_3448_, v___x_3453_, v_a_3445_, v_stx_3446_, v___x_3451_, v___y_3457_, v_buckets_3459_, v_sz_3462_, v___x_3463_, v___x_3461_, v___y_3430_, v___y_3431_);
lean_dec_ref(v_buckets_3459_);
lean_dec(v___x_3451_);
if (lean_obj_tag(v___x_3464_) == 0)
{
lean_object* v_a_3465_; lean_object* v_fst_3466_; lean_object* v_snd_3467_; lean_object* v___x_3469_; uint8_t v_isShared_3470_; uint8_t v_isSharedCheck_3474_; 
v_a_3465_ = lean_ctor_get(v___x_3464_, 0);
lean_inc(v_a_3465_);
lean_dec_ref_known(v___x_3464_, 1);
v_fst_3466_ = lean_ctor_get(v_a_3465_, 0);
v_snd_3467_ = lean_ctor_get(v_a_3465_, 1);
v_isSharedCheck_3474_ = !lean_is_exclusive(v_a_3465_);
if (v_isSharedCheck_3474_ == 0)
{
v___x_3469_ = v_a_3465_;
v_isShared_3470_ = v_isSharedCheck_3474_;
goto v_resetjp_3468_;
}
else
{
lean_inc(v_snd_3467_);
lean_inc(v_fst_3466_);
lean_dec(v_a_3465_);
v___x_3469_ = lean_box(0);
v_isShared_3470_ = v_isSharedCheck_3474_;
goto v_resetjp_3468_;
}
v_resetjp_3468_:
{
lean_object* v___x_3472_; 
if (v_isShared_3470_ == 0)
{
v___x_3472_ = v___x_3469_;
goto v_reusejp_3471_;
}
else
{
lean_object* v_reuseFailAlloc_3473_; 
v_reuseFailAlloc_3473_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3473_, 0, v_fst_3466_);
lean_ctor_set(v_reuseFailAlloc_3473_, 1, v_snd_3467_);
v___x_3472_ = v_reuseFailAlloc_3473_;
goto v_reusejp_3471_;
}
v_reusejp_3471_:
{
v_a_3434_ = v___x_3472_;
goto v___jp_3433_;
}
}
}
else
{
return v___x_3464_;
}
}
}
}
else
{
lean_object* v___x_3481_; 
lean_dec(v___x_3451_);
if (v_isShared_3444_ == 0)
{
v___x_3481_ = v___x_3443_;
goto v_reusejp_3480_;
}
else
{
lean_object* v_reuseFailAlloc_3482_; 
v_reuseFailAlloc_3482_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3482_, 0, v_fst_3440_);
lean_ctor_set(v_reuseFailAlloc_3482_, 1, v_snd_3441_);
v___x_3481_ = v_reuseFailAlloc_3482_;
goto v_reusejp_3480_;
}
v_reusejp_3480_:
{
v_a_3434_ = v___x_3481_;
goto v___jp_3433_;
}
}
}
}
v___jp_3433_:
{
size_t v___x_3435_; size_t v___x_3436_; 
v___x_3435_ = ((size_t)1ULL);
v___x_3436_ = lean_usize_add(v_i_3428_, v___x_3435_);
v_i_3428_ = v___x_3436_;
v_b_3429_ = v_a_3434_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__21___boxed(lean_object* v_as_3484_, lean_object* v_sz_3485_, lean_object* v_i_3486_, lean_object* v_b_3487_, lean_object* v___y_3488_, lean_object* v___y_3489_, lean_object* v___y_3490_){
_start:
{
size_t v_sz_boxed_3491_; size_t v_i_boxed_3492_; lean_object* v_res_3493_; 
v_sz_boxed_3491_ = lean_unbox_usize(v_sz_3485_);
lean_dec(v_sz_3485_);
v_i_boxed_3492_ = lean_unbox_usize(v_i_3486_);
lean_dec(v_i_3486_);
v_res_3493_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__21(v_as_3484_, v_sz_boxed_3491_, v_i_boxed_3492_, v_b_3487_, v___y_3488_, v___y_3489_);
lean_dec(v___y_3489_);
lean_dec_ref(v___y_3488_);
lean_dec_ref(v_as_3484_);
return v_res_3493_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__1(void){
_start:
{
lean_object* v___x_3495_; lean_object* v___x_3496_; 
v___x_3495_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__0));
v___x_3496_ = l_Lean_stringToMessageData(v___x_3495_);
return v___x_3496_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__3(void){
_start:
{
lean_object* v___x_3498_; lean_object* v___x_3499_; 
v___x_3498_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__2));
v___x_3499_ = l_Lean_stringToMessageData(v___x_3498_);
return v___x_3499_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__5(void){
_start:
{
lean_object* v___x_3501_; lean_object* v___x_3502_; 
v___x_3501_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__4));
v___x_3502_ = l_Lean_stringToMessageData(v___x_3501_);
return v___x_3502_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0(lean_object* v___x_3503_, lean_object* v_stained_3504_, uint8_t v___y_3505_, lean_object* v_x_3506_){
_start:
{
lean_object* v_str_3507_; lean_object* v_startInclusive_3508_; lean_object* v_endExclusive_3509_; lean_object* v___x_3510_; lean_object* v___x_3511_; lean_object* v___x_3512_; lean_object* v___x_3513_; lean_object* v___x_3514_; lean_object* v___x_3515_; lean_object* v___x_3516_; lean_object* v___y_3518_; 
v_str_3507_ = lean_ctor_get(v___x_3503_, 0);
v_startInclusive_3508_ = lean_ctor_get(v___x_3503_, 1);
v_endExclusive_3509_ = lean_ctor_get(v___x_3503_, 2);
v___x_3510_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__1, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__1_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__1);
v___x_3511_ = lean_string_utf8_extract_fast(v_str_3507_, v_startInclusive_3508_, v_endExclusive_3509_);
v___x_3512_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_3512_, 0, v___x_3511_);
v___x_3513_ = l_Lean_MessageData_ofFormat(v___x_3512_);
v___x_3514_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3514_, 0, v___x_3510_);
lean_ctor_set(v___x_3514_, 1, v___x_3513_);
v___x_3515_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__3, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__3_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__3);
v___x_3516_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3516_, 0, v___x_3514_);
lean_ctor_set(v___x_3516_, 1, v___x_3515_);
switch(lean_obj_tag(v_stained_3504_))
{
case 0:
{
lean_object* v_a_3524_; lean_object* v___x_3525_; 
v_a_3524_ = lean_ctor_get(v_stained_3504_, 0);
lean_inc(v_a_3524_);
lean_dec_ref_known(v_stained_3504_, 1);
v___x_3525_ = l_Lean_Name_toString(v_a_3524_, v___y_3505_);
v___y_3518_ = v___x_3525_;
goto v___jp_3517_;
}
case 1:
{
lean_object* v___x_3526_; 
v___x_3526_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instToStringStained___lam__0___closed__0));
v___y_3518_ = v___x_3526_;
goto v___jp_3517_;
}
default: 
{
lean_object* v___x_3527_; 
v___x_3527_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instToStringStained___lam__0___closed__1));
v___y_3518_ = v___x_3527_;
goto v___jp_3517_;
}
}
v___jp_3517_:
{
lean_object* v___x_3519_; lean_object* v___x_3520_; lean_object* v___x_3521_; lean_object* v___x_3522_; lean_object* v___x_3523_; 
v___x_3519_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_3519_, 0, v___y_3518_);
v___x_3520_ = l_Lean_MessageData_ofFormat(v___x_3519_);
v___x_3521_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3521_, 0, v___x_3516_);
lean_ctor_set(v___x_3521_, 1, v___x_3520_);
v___x_3522_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__5, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__5_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__5);
v___x_3523_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3523_, 0, v___x_3521_);
lean_ctor_set(v___x_3523_, 1, v___x_3522_);
return v___x_3523_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___boxed(lean_object* v___x_3528_, lean_object* v_stained_3529_, lean_object* v___y_3530_, lean_object* v_x_3531_){
_start:
{
uint8_t v___y_35584__boxed_3532_; lean_object* v_res_3533_; 
v___y_35584__boxed_3532_ = lean_unbox(v___y_3530_);
v_res_3533_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0(v___x_3528_, v_stained_3529_, v___y_35584__boxed_3532_, v_x_3531_);
lean_dec(v_x_3531_);
lean_dec_ref(v___x_3528_);
return v_res_3533_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29___lam__0(uint8_t v___y_3535_, uint8_t v_suppressElabErrors_3536_, lean_object* v_x_3537_){
_start:
{
if (lean_obj_tag(v_x_3537_) == 1)
{
lean_object* v_pre_3538_; 
v_pre_3538_ = lean_ctor_get(v_x_3537_, 0);
if (lean_obj_tag(v_pre_3538_) == 0)
{
lean_object* v_str_3539_; lean_object* v___x_3540_; uint8_t v___x_3541_; 
v_str_3539_ = lean_ctor_get(v_x_3537_, 1);
v___x_3540_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29___lam__0___closed__0));
v___x_3541_ = lean_string_dec_eq(v_str_3539_, v___x_3540_);
if (v___x_3541_ == 0)
{
return v___y_3535_;
}
else
{
return v_suppressElabErrors_3536_;
}
}
else
{
return v___y_3535_;
}
}
else
{
return v___y_3535_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29___lam__0___boxed(lean_object* v___y_3542_, lean_object* v_suppressElabErrors_3543_, lean_object* v_x_3544_){
_start:
{
uint8_t v___y_35639__boxed_3545_; uint8_t v_suppressElabErrors_boxed_3546_; uint8_t v_res_3547_; lean_object* v_r_3548_; 
v___y_35639__boxed_3545_ = lean_unbox(v___y_3542_);
v_suppressElabErrors_boxed_3546_ = lean_unbox(v_suppressElabErrors_3543_);
v_res_3547_ = lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29___lam__0(v___y_35639__boxed_3545_, v_suppressElabErrors_boxed_3546_, v_x_3544_);
lean_dec(v_x_3544_);
v_r_3548_ = lean_box(v_res_3547_);
return v_r_3548_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29_spec__33(lean_object* v_opts_3549_, lean_object* v_opt_3550_){
_start:
{
lean_object* v_name_3551_; lean_object* v_defValue_3552_; lean_object* v_map_3553_; lean_object* v___x_3554_; 
v_name_3551_ = lean_ctor_get(v_opt_3550_, 0);
v_defValue_3552_ = lean_ctor_get(v_opt_3550_, 1);
v_map_3553_ = lean_ctor_get(v_opts_3549_, 0);
v___x_3554_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_3553_, v_name_3551_);
if (lean_obj_tag(v___x_3554_) == 0)
{
uint8_t v___x_3555_; 
v___x_3555_ = lean_unbox(v_defValue_3552_);
return v___x_3555_;
}
else
{
lean_object* v_val_3556_; 
v_val_3556_ = lean_ctor_get(v___x_3554_, 0);
lean_inc(v_val_3556_);
lean_dec_ref_known(v___x_3554_, 1);
if (lean_obj_tag(v_val_3556_) == 1)
{
uint8_t v_v_3557_; 
v_v_3557_ = lean_ctor_get_uint8(v_val_3556_, 0);
lean_dec_ref_known(v_val_3556_, 0);
return v_v_3557_;
}
else
{
uint8_t v___x_3558_; 
lean_dec(v_val_3556_);
v___x_3558_ = lean_unbox(v_defValue_3552_);
return v___x_3558_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29_spec__33___boxed(lean_object* v_opts_3559_, lean_object* v_opt_3560_){
_start:
{
uint8_t v_res_3561_; lean_object* v_r_3562_; 
v_res_3561_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29_spec__33(v_opts_3559_, v_opt_3560_);
lean_dec_ref(v_opt_3560_);
lean_dec_ref(v_opts_3559_);
v_r_3562_ = lean_box(v_res_3561_);
return v_r_3562_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29_spec__32___redArg___closed__0(void){
_start:
{
lean_object* v___x_3563_; 
v___x_3563_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_3563_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29_spec__32___redArg___closed__1(void){
_start:
{
lean_object* v___x_3564_; lean_object* v___x_3565_; 
v___x_3564_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29_spec__32___redArg___closed__0, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29_spec__32___redArg___closed__0_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29_spec__32___redArg___closed__0);
v___x_3565_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3565_, 0, v___x_3564_);
return v___x_3565_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29_spec__32___redArg___closed__2(void){
_start:
{
lean_object* v___x_3566_; lean_object* v___x_3567_; lean_object* v___x_3568_; 
v___x_3566_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29_spec__32___redArg___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29_spec__32___redArg___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29_spec__32___redArg___closed__1);
v___x_3567_ = lean_unsigned_to_nat(0u);
v___x_3568_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_3568_, 0, v___x_3567_);
lean_ctor_set(v___x_3568_, 1, v___x_3567_);
lean_ctor_set(v___x_3568_, 2, v___x_3567_);
lean_ctor_set(v___x_3568_, 3, v___x_3567_);
lean_ctor_set(v___x_3568_, 4, v___x_3566_);
lean_ctor_set(v___x_3568_, 5, v___x_3566_);
lean_ctor_set(v___x_3568_, 6, v___x_3566_);
lean_ctor_set(v___x_3568_, 7, v___x_3566_);
lean_ctor_set(v___x_3568_, 8, v___x_3566_);
lean_ctor_set(v___x_3568_, 9, v___x_3566_);
return v___x_3568_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29_spec__32___redArg___closed__3(void){
_start:
{
lean_object* v___x_3569_; lean_object* v___x_3570_; lean_object* v___x_3571_; 
v___x_3569_ = lean_unsigned_to_nat(32u);
v___x_3570_ = lean_mk_empty_array_with_capacity(v___x_3569_);
v___x_3571_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3571_, 0, v___x_3570_);
return v___x_3571_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29_spec__32___redArg___closed__4(void){
_start:
{
size_t v___x_3572_; lean_object* v___x_3573_; lean_object* v___x_3574_; lean_object* v___x_3575_; lean_object* v___x_3576_; lean_object* v___x_3577_; 
v___x_3572_ = ((size_t)5ULL);
v___x_3573_ = lean_unsigned_to_nat(0u);
v___x_3574_ = lean_unsigned_to_nat(32u);
v___x_3575_ = lean_mk_empty_array_with_capacity(v___x_3574_);
v___x_3576_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29_spec__32___redArg___closed__3, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29_spec__32___redArg___closed__3_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29_spec__32___redArg___closed__3);
v___x_3577_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_3577_, 0, v___x_3576_);
lean_ctor_set(v___x_3577_, 1, v___x_3575_);
lean_ctor_set(v___x_3577_, 2, v___x_3573_);
lean_ctor_set(v___x_3577_, 3, v___x_3573_);
lean_ctor_set_usize(v___x_3577_, 4, v___x_3572_);
return v___x_3577_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29_spec__32___redArg___closed__5(void){
_start:
{
lean_object* v___x_3578_; lean_object* v___x_3579_; lean_object* v___x_3580_; lean_object* v___x_3581_; 
v___x_3578_ = lean_box(1);
v___x_3579_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29_spec__32___redArg___closed__4, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29_spec__32___redArg___closed__4_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29_spec__32___redArg___closed__4);
v___x_3580_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29_spec__32___redArg___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29_spec__32___redArg___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29_spec__32___redArg___closed__1);
v___x_3581_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_3581_, 0, v___x_3580_);
lean_ctor_set(v___x_3581_, 1, v___x_3579_);
lean_ctor_set(v___x_3581_, 2, v___x_3578_);
return v___x_3581_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29_spec__32___redArg(lean_object* v_msgData_3582_, lean_object* v___y_3583_){
_start:
{
lean_object* v___x_3585_; lean_object* v_env_3586_; lean_object* v___x_3587_; lean_object* v_scopes_3588_; lean_object* v___x_3589_; lean_object* v___x_3590_; lean_object* v_opts_3591_; lean_object* v___x_3592_; lean_object* v___x_3593_; lean_object* v___x_3594_; lean_object* v___x_3595_; lean_object* v___x_3596_; 
v___x_3585_ = lean_st_ref_get(v___y_3583_);
v_env_3586_ = lean_ctor_get(v___x_3585_, 0);
lean_inc_ref(v_env_3586_);
lean_dec(v___x_3585_);
v___x_3587_ = lean_st_ref_get(v___y_3583_);
v_scopes_3588_ = lean_ctor_get(v___x_3587_, 2);
lean_inc(v_scopes_3588_);
lean_dec(v___x_3587_);
v___x_3589_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_3590_ = l_List_head_x21___redArg(v___x_3589_, v_scopes_3588_);
lean_dec(v_scopes_3588_);
v_opts_3591_ = lean_ctor_get(v___x_3590_, 1);
lean_inc_ref(v_opts_3591_);
lean_dec(v___x_3590_);
v___x_3592_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29_spec__32___redArg___closed__2, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29_spec__32___redArg___closed__2_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29_spec__32___redArg___closed__2);
v___x_3593_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29_spec__32___redArg___closed__5, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29_spec__32___redArg___closed__5_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29_spec__32___redArg___closed__5);
v___x_3594_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_3594_, 0, v_env_3586_);
lean_ctor_set(v___x_3594_, 1, v___x_3592_);
lean_ctor_set(v___x_3594_, 2, v___x_3593_);
lean_ctor_set(v___x_3594_, 3, v_opts_3591_);
v___x_3595_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_3595_, 0, v___x_3594_);
lean_ctor_set(v___x_3595_, 1, v_msgData_3582_);
v___x_3596_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3596_, 0, v___x_3595_);
return v___x_3596_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29_spec__32___redArg___boxed(lean_object* v_msgData_3597_, lean_object* v___y_3598_, lean_object* v___y_3599_){
_start:
{
lean_object* v_res_3600_; 
v_res_3600_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29_spec__32___redArg(v_msgData_3597_, v___y_3598_);
lean_dec(v___y_3598_);
return v_res_3600_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29(lean_object* v_ref_3602_, lean_object* v_msgData_3603_, uint8_t v_severity_3604_, uint8_t v_isSilent_3605_, lean_object* v___y_3606_, lean_object* v___y_3607_){
_start:
{
uint8_t v___y_3610_; lean_object* v___y_3611_; lean_object* v___y_3612_; lean_object* v___y_3613_; lean_object* v___y_3614_; uint8_t v___y_3615_; lean_object* v___y_3616_; lean_object* v___y_3617_; uint8_t v___y_3674_; uint8_t v___y_3675_; lean_object* v___y_3676_; uint8_t v___y_3677_; lean_object* v___y_3678_; uint8_t v___y_3702_; lean_object* v___y_3703_; uint8_t v___y_3704_; uint8_t v___y_3705_; lean_object* v___y_3706_; uint8_t v___y_3710_; uint8_t v___y_3711_; uint8_t v___y_3712_; uint8_t v___x_3727_; uint8_t v___y_3729_; uint8_t v___y_3730_; uint8_t v___y_3731_; uint8_t v___y_3733_; uint8_t v___x_3745_; 
v___x_3727_ = 2;
v___x_3745_ = l_Lean_instBEqMessageSeverity_beq(v_severity_3604_, v___x_3727_);
if (v___x_3745_ == 0)
{
v___y_3733_ = v___x_3745_;
goto v___jp_3732_;
}
else
{
uint8_t v___x_3746_; 
lean_inc_ref(v_msgData_3603_);
v___x_3746_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_3603_);
v___y_3733_ = v___x_3746_;
goto v___jp_3732_;
}
v___jp_3609_:
{
lean_object* v___x_3618_; 
v___x_3618_ = l_Lean_Elab_Command_getScope___redArg(v___y_3617_);
if (lean_obj_tag(v___x_3618_) == 0)
{
lean_object* v_a_3619_; lean_object* v___x_3620_; 
v_a_3619_ = lean_ctor_get(v___x_3618_, 0);
lean_inc(v_a_3619_);
lean_dec_ref_known(v___x_3618_, 1);
v___x_3620_ = l_Lean_Elab_Command_getScope___redArg(v___y_3617_);
if (lean_obj_tag(v___x_3620_) == 0)
{
lean_object* v_a_3621_; lean_object* v___x_3623_; uint8_t v_isShared_3624_; uint8_t v_isSharedCheck_3656_; 
v_a_3621_ = lean_ctor_get(v___x_3620_, 0);
v_isSharedCheck_3656_ = !lean_is_exclusive(v___x_3620_);
if (v_isSharedCheck_3656_ == 0)
{
v___x_3623_ = v___x_3620_;
v_isShared_3624_ = v_isSharedCheck_3656_;
goto v_resetjp_3622_;
}
else
{
lean_inc(v_a_3621_);
lean_dec(v___x_3620_);
v___x_3623_ = lean_box(0);
v_isShared_3624_ = v_isSharedCheck_3656_;
goto v_resetjp_3622_;
}
v_resetjp_3622_:
{
lean_object* v___x_3625_; lean_object* v_currNamespace_3626_; lean_object* v_openDecls_3627_; lean_object* v_env_3628_; lean_object* v_messages_3629_; lean_object* v_scopes_3630_; lean_object* v_usedQuotCtxts_3631_; lean_object* v_nextMacroScope_3632_; lean_object* v_maxRecDepth_3633_; lean_object* v_ngen_3634_; lean_object* v_auxDeclNGen_3635_; lean_object* v_infoState_3636_; lean_object* v_traceState_3637_; lean_object* v_snapshotTasks_3638_; lean_object* v_prevLinterStates_3639_; lean_object* v___x_3641_; uint8_t v_isShared_3642_; uint8_t v_isSharedCheck_3655_; 
v___x_3625_ = lean_st_ref_take(v___y_3617_);
v_currNamespace_3626_ = lean_ctor_get(v_a_3619_, 2);
lean_inc(v_currNamespace_3626_);
lean_dec(v_a_3619_);
v_openDecls_3627_ = lean_ctor_get(v_a_3621_, 3);
lean_inc(v_openDecls_3627_);
lean_dec(v_a_3621_);
v_env_3628_ = lean_ctor_get(v___x_3625_, 0);
v_messages_3629_ = lean_ctor_get(v___x_3625_, 1);
v_scopes_3630_ = lean_ctor_get(v___x_3625_, 2);
v_usedQuotCtxts_3631_ = lean_ctor_get(v___x_3625_, 3);
v_nextMacroScope_3632_ = lean_ctor_get(v___x_3625_, 4);
v_maxRecDepth_3633_ = lean_ctor_get(v___x_3625_, 5);
v_ngen_3634_ = lean_ctor_get(v___x_3625_, 6);
v_auxDeclNGen_3635_ = lean_ctor_get(v___x_3625_, 7);
v_infoState_3636_ = lean_ctor_get(v___x_3625_, 8);
v_traceState_3637_ = lean_ctor_get(v___x_3625_, 9);
v_snapshotTasks_3638_ = lean_ctor_get(v___x_3625_, 10);
v_prevLinterStates_3639_ = lean_ctor_get(v___x_3625_, 11);
v_isSharedCheck_3655_ = !lean_is_exclusive(v___x_3625_);
if (v_isSharedCheck_3655_ == 0)
{
v___x_3641_ = v___x_3625_;
v_isShared_3642_ = v_isSharedCheck_3655_;
goto v_resetjp_3640_;
}
else
{
lean_inc(v_prevLinterStates_3639_);
lean_inc(v_snapshotTasks_3638_);
lean_inc(v_traceState_3637_);
lean_inc(v_infoState_3636_);
lean_inc(v_auxDeclNGen_3635_);
lean_inc(v_ngen_3634_);
lean_inc(v_maxRecDepth_3633_);
lean_inc(v_nextMacroScope_3632_);
lean_inc(v_usedQuotCtxts_3631_);
lean_inc(v_scopes_3630_);
lean_inc(v_messages_3629_);
lean_inc(v_env_3628_);
lean_dec(v___x_3625_);
v___x_3641_ = lean_box(0);
v_isShared_3642_ = v_isSharedCheck_3655_;
goto v_resetjp_3640_;
}
v_resetjp_3640_:
{
lean_object* v___x_3643_; lean_object* v___x_3644_; lean_object* v___x_3645_; lean_object* v___x_3646_; lean_object* v___x_3648_; 
v___x_3643_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3643_, 0, v_currNamespace_3626_);
lean_ctor_set(v___x_3643_, 1, v_openDecls_3627_);
v___x_3644_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_3644_, 0, v___x_3643_);
lean_ctor_set(v___x_3644_, 1, v___y_3613_);
lean_inc_ref(v___y_3616_);
lean_inc_ref(v___y_3611_);
v___x_3645_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_3645_, 0, v___y_3611_);
lean_ctor_set(v___x_3645_, 1, v___y_3612_);
lean_ctor_set(v___x_3645_, 2, v___y_3614_);
lean_ctor_set(v___x_3645_, 3, v___y_3616_);
lean_ctor_set(v___x_3645_, 4, v___x_3644_);
lean_ctor_set_uint8(v___x_3645_, sizeof(void*)*5, v___y_3610_);
lean_ctor_set_uint8(v___x_3645_, sizeof(void*)*5 + 1, v___y_3615_);
lean_ctor_set_uint8(v___x_3645_, sizeof(void*)*5 + 2, v_isSilent_3605_);
v___x_3646_ = l_Lean_MessageLog_add(v___x_3645_, v_messages_3629_);
if (v_isShared_3642_ == 0)
{
lean_ctor_set(v___x_3641_, 1, v___x_3646_);
v___x_3648_ = v___x_3641_;
goto v_reusejp_3647_;
}
else
{
lean_object* v_reuseFailAlloc_3654_; 
v_reuseFailAlloc_3654_ = lean_alloc_ctor(0, 12, 0);
lean_ctor_set(v_reuseFailAlloc_3654_, 0, v_env_3628_);
lean_ctor_set(v_reuseFailAlloc_3654_, 1, v___x_3646_);
lean_ctor_set(v_reuseFailAlloc_3654_, 2, v_scopes_3630_);
lean_ctor_set(v_reuseFailAlloc_3654_, 3, v_usedQuotCtxts_3631_);
lean_ctor_set(v_reuseFailAlloc_3654_, 4, v_nextMacroScope_3632_);
lean_ctor_set(v_reuseFailAlloc_3654_, 5, v_maxRecDepth_3633_);
lean_ctor_set(v_reuseFailAlloc_3654_, 6, v_ngen_3634_);
lean_ctor_set(v_reuseFailAlloc_3654_, 7, v_auxDeclNGen_3635_);
lean_ctor_set(v_reuseFailAlloc_3654_, 8, v_infoState_3636_);
lean_ctor_set(v_reuseFailAlloc_3654_, 9, v_traceState_3637_);
lean_ctor_set(v_reuseFailAlloc_3654_, 10, v_snapshotTasks_3638_);
lean_ctor_set(v_reuseFailAlloc_3654_, 11, v_prevLinterStates_3639_);
v___x_3648_ = v_reuseFailAlloc_3654_;
goto v_reusejp_3647_;
}
v_reusejp_3647_:
{
lean_object* v___x_3649_; lean_object* v___x_3650_; lean_object* v___x_3652_; 
v___x_3649_ = lean_st_ref_set(v___y_3617_, v___x_3648_);
v___x_3650_ = lean_box(0);
if (v_isShared_3624_ == 0)
{
lean_ctor_set(v___x_3623_, 0, v___x_3650_);
v___x_3652_ = v___x_3623_;
goto v_reusejp_3651_;
}
else
{
lean_object* v_reuseFailAlloc_3653_; 
v_reuseFailAlloc_3653_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3653_, 0, v___x_3650_);
v___x_3652_ = v_reuseFailAlloc_3653_;
goto v_reusejp_3651_;
}
v_reusejp_3651_:
{
return v___x_3652_;
}
}
}
}
}
else
{
lean_object* v_a_3657_; lean_object* v___x_3659_; uint8_t v_isShared_3660_; uint8_t v_isSharedCheck_3664_; 
lean_dec(v_a_3619_);
lean_dec(v___y_3614_);
lean_dec_ref(v___y_3613_);
lean_dec_ref(v___y_3612_);
v_a_3657_ = lean_ctor_get(v___x_3620_, 0);
v_isSharedCheck_3664_ = !lean_is_exclusive(v___x_3620_);
if (v_isSharedCheck_3664_ == 0)
{
v___x_3659_ = v___x_3620_;
v_isShared_3660_ = v_isSharedCheck_3664_;
goto v_resetjp_3658_;
}
else
{
lean_inc(v_a_3657_);
lean_dec(v___x_3620_);
v___x_3659_ = lean_box(0);
v_isShared_3660_ = v_isSharedCheck_3664_;
goto v_resetjp_3658_;
}
v_resetjp_3658_:
{
lean_object* v___x_3662_; 
if (v_isShared_3660_ == 0)
{
v___x_3662_ = v___x_3659_;
goto v_reusejp_3661_;
}
else
{
lean_object* v_reuseFailAlloc_3663_; 
v_reuseFailAlloc_3663_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3663_, 0, v_a_3657_);
v___x_3662_ = v_reuseFailAlloc_3663_;
goto v_reusejp_3661_;
}
v_reusejp_3661_:
{
return v___x_3662_;
}
}
}
}
else
{
lean_object* v_a_3665_; lean_object* v___x_3667_; uint8_t v_isShared_3668_; uint8_t v_isSharedCheck_3672_; 
lean_dec(v___y_3614_);
lean_dec_ref(v___y_3613_);
lean_dec_ref(v___y_3612_);
v_a_3665_ = lean_ctor_get(v___x_3618_, 0);
v_isSharedCheck_3672_ = !lean_is_exclusive(v___x_3618_);
if (v_isSharedCheck_3672_ == 0)
{
v___x_3667_ = v___x_3618_;
v_isShared_3668_ = v_isSharedCheck_3672_;
goto v_resetjp_3666_;
}
else
{
lean_inc(v_a_3665_);
lean_dec(v___x_3618_);
v___x_3667_ = lean_box(0);
v_isShared_3668_ = v_isSharedCheck_3672_;
goto v_resetjp_3666_;
}
v_resetjp_3666_:
{
lean_object* v___x_3670_; 
if (v_isShared_3668_ == 0)
{
v___x_3670_ = v___x_3667_;
goto v_reusejp_3669_;
}
else
{
lean_object* v_reuseFailAlloc_3671_; 
v_reuseFailAlloc_3671_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3671_, 0, v_a_3665_);
v___x_3670_ = v_reuseFailAlloc_3671_;
goto v_reusejp_3669_;
}
v_reusejp_3669_:
{
return v___x_3670_;
}
}
}
}
v___jp_3673_:
{
lean_object* v_fileName_3679_; lean_object* v_fileMap_3680_; uint8_t v_suppressElabErrors_3681_; lean_object* v___x_3682_; lean_object* v___x_3683_; lean_object* v_a_3684_; lean_object* v___x_3686_; uint8_t v_isShared_3687_; uint8_t v_isSharedCheck_3700_; 
v_fileName_3679_ = lean_ctor_get(v___y_3606_, 0);
v_fileMap_3680_ = lean_ctor_get(v___y_3606_, 1);
v_suppressElabErrors_3681_ = lean_ctor_get_uint8(v___y_3606_, sizeof(void*)*10);
v___x_3682_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_3603_);
v___x_3683_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29_spec__32___redArg(v___x_3682_, v___y_3607_);
v_a_3684_ = lean_ctor_get(v___x_3683_, 0);
v_isSharedCheck_3700_ = !lean_is_exclusive(v___x_3683_);
if (v_isSharedCheck_3700_ == 0)
{
v___x_3686_ = v___x_3683_;
v_isShared_3687_ = v_isSharedCheck_3700_;
goto v_resetjp_3685_;
}
else
{
lean_inc(v_a_3684_);
lean_dec(v___x_3683_);
v___x_3686_ = lean_box(0);
v_isShared_3687_ = v_isSharedCheck_3700_;
goto v_resetjp_3685_;
}
v_resetjp_3685_:
{
lean_object* v___x_3688_; lean_object* v___x_3689_; lean_object* v___x_3690_; lean_object* v___x_3691_; 
lean_inc_ref_n(v_fileMap_3680_, 2);
v___x_3688_ = l_Lean_FileMap_toPosition(v_fileMap_3680_, v___y_3676_);
lean_dec(v___y_3676_);
v___x_3689_ = l_Lean_FileMap_toPosition(v_fileMap_3680_, v___y_3678_);
lean_dec(v___y_3678_);
v___x_3690_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3690_, 0, v___x_3689_);
v___x_3691_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29___closed__0));
if (v_suppressElabErrors_3681_ == 0)
{
lean_del_object(v___x_3686_);
v___y_3610_ = v___y_3675_;
v___y_3611_ = v_fileName_3679_;
v___y_3612_ = v___x_3688_;
v___y_3613_ = v_a_3684_;
v___y_3614_ = v___x_3690_;
v___y_3615_ = v___y_3677_;
v___y_3616_ = v___x_3691_;
v___y_3617_ = v___y_3607_;
goto v___jp_3609_;
}
else
{
lean_object* v___x_3692_; lean_object* v___x_3693_; lean_object* v___f_3694_; uint8_t v___x_3695_; 
v___x_3692_ = lean_box(v___y_3674_);
v___x_3693_ = lean_box(v_suppressElabErrors_3681_);
v___f_3694_ = lean_alloc_closure((void*)(lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29___lam__0___boxed), 3, 2);
lean_closure_set(v___f_3694_, 0, v___x_3692_);
lean_closure_set(v___f_3694_, 1, v___x_3693_);
lean_inc(v_a_3684_);
v___x_3695_ = l_Lean_MessageData_hasTag(v___f_3694_, v_a_3684_);
if (v___x_3695_ == 0)
{
lean_object* v___x_3696_; lean_object* v___x_3698_; 
lean_dec_ref_known(v___x_3690_, 1);
lean_dec_ref(v___x_3688_);
lean_dec(v_a_3684_);
v___x_3696_ = lean_box(0);
if (v_isShared_3687_ == 0)
{
lean_ctor_set(v___x_3686_, 0, v___x_3696_);
v___x_3698_ = v___x_3686_;
goto v_reusejp_3697_;
}
else
{
lean_object* v_reuseFailAlloc_3699_; 
v_reuseFailAlloc_3699_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3699_, 0, v___x_3696_);
v___x_3698_ = v_reuseFailAlloc_3699_;
goto v_reusejp_3697_;
}
v_reusejp_3697_:
{
return v___x_3698_;
}
}
else
{
lean_del_object(v___x_3686_);
v___y_3610_ = v___y_3675_;
v___y_3611_ = v_fileName_3679_;
v___y_3612_ = v___x_3688_;
v___y_3613_ = v_a_3684_;
v___y_3614_ = v___x_3690_;
v___y_3615_ = v___y_3677_;
v___y_3616_ = v___x_3691_;
v___y_3617_ = v___y_3607_;
goto v___jp_3609_;
}
}
}
}
v___jp_3701_:
{
lean_object* v___x_3707_; 
v___x_3707_ = l_Lean_Syntax_getTailPos_x3f(v___y_3703_, v___y_3704_);
lean_dec(v___y_3703_);
if (lean_obj_tag(v___x_3707_) == 0)
{
lean_inc(v___y_3706_);
v___y_3674_ = v___y_3702_;
v___y_3675_ = v___y_3704_;
v___y_3676_ = v___y_3706_;
v___y_3677_ = v___y_3705_;
v___y_3678_ = v___y_3706_;
goto v___jp_3673_;
}
else
{
lean_object* v_val_3708_; 
v_val_3708_ = lean_ctor_get(v___x_3707_, 0);
lean_inc(v_val_3708_);
lean_dec_ref_known(v___x_3707_, 1);
v___y_3674_ = v___y_3702_;
v___y_3675_ = v___y_3704_;
v___y_3676_ = v___y_3706_;
v___y_3677_ = v___y_3705_;
v___y_3678_ = v_val_3708_;
goto v___jp_3673_;
}
}
v___jp_3709_:
{
lean_object* v___x_3713_; 
v___x_3713_ = l_Lean_Elab_Command_getRef___redArg(v___y_3606_);
if (lean_obj_tag(v___x_3713_) == 0)
{
lean_object* v_a_3714_; lean_object* v_ref_3715_; lean_object* v___x_3716_; 
v_a_3714_ = lean_ctor_get(v___x_3713_, 0);
lean_inc(v_a_3714_);
lean_dec_ref_known(v___x_3713_, 1);
v_ref_3715_ = l_Lean_replaceRef(v_ref_3602_, v_a_3714_);
lean_dec(v_a_3714_);
v___x_3716_ = l_Lean_Syntax_getPos_x3f(v_ref_3715_, v___y_3711_);
if (lean_obj_tag(v___x_3716_) == 0)
{
lean_object* v___x_3717_; 
v___x_3717_ = lean_unsigned_to_nat(0u);
v___y_3702_ = v___y_3710_;
v___y_3703_ = v_ref_3715_;
v___y_3704_ = v___y_3711_;
v___y_3705_ = v___y_3712_;
v___y_3706_ = v___x_3717_;
goto v___jp_3701_;
}
else
{
lean_object* v_val_3718_; 
v_val_3718_ = lean_ctor_get(v___x_3716_, 0);
lean_inc(v_val_3718_);
lean_dec_ref_known(v___x_3716_, 1);
v___y_3702_ = v___y_3710_;
v___y_3703_ = v_ref_3715_;
v___y_3704_ = v___y_3711_;
v___y_3705_ = v___y_3712_;
v___y_3706_ = v_val_3718_;
goto v___jp_3701_;
}
}
else
{
lean_object* v_a_3719_; lean_object* v___x_3721_; uint8_t v_isShared_3722_; uint8_t v_isSharedCheck_3726_; 
lean_dec_ref(v_msgData_3603_);
v_a_3719_ = lean_ctor_get(v___x_3713_, 0);
v_isSharedCheck_3726_ = !lean_is_exclusive(v___x_3713_);
if (v_isSharedCheck_3726_ == 0)
{
v___x_3721_ = v___x_3713_;
v_isShared_3722_ = v_isSharedCheck_3726_;
goto v_resetjp_3720_;
}
else
{
lean_inc(v_a_3719_);
lean_dec(v___x_3713_);
v___x_3721_ = lean_box(0);
v_isShared_3722_ = v_isSharedCheck_3726_;
goto v_resetjp_3720_;
}
v_resetjp_3720_:
{
lean_object* v___x_3724_; 
if (v_isShared_3722_ == 0)
{
v___x_3724_ = v___x_3721_;
goto v_reusejp_3723_;
}
else
{
lean_object* v_reuseFailAlloc_3725_; 
v_reuseFailAlloc_3725_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3725_, 0, v_a_3719_);
v___x_3724_ = v_reuseFailAlloc_3725_;
goto v_reusejp_3723_;
}
v_reusejp_3723_:
{
return v___x_3724_;
}
}
}
}
v___jp_3728_:
{
if (v___y_3731_ == 0)
{
v___y_3710_ = v___y_3729_;
v___y_3711_ = v___y_3730_;
v___y_3712_ = v_severity_3604_;
goto v___jp_3709_;
}
else
{
v___y_3710_ = v___y_3729_;
v___y_3711_ = v___y_3730_;
v___y_3712_ = v___x_3727_;
goto v___jp_3709_;
}
}
v___jp_3732_:
{
if (v___y_3733_ == 0)
{
lean_object* v___x_3734_; lean_object* v_scopes_3735_; lean_object* v___x_3736_; lean_object* v___x_3737_; lean_object* v_opts_3738_; uint8_t v___x_3739_; uint8_t v___x_3740_; 
v___x_3734_ = lean_st_ref_get(v___y_3607_);
v_scopes_3735_ = lean_ctor_get(v___x_3734_, 2);
lean_inc(v_scopes_3735_);
lean_dec(v___x_3734_);
v___x_3736_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_3737_ = l_List_head_x21___redArg(v___x_3736_, v_scopes_3735_);
lean_dec(v_scopes_3735_);
v_opts_3738_ = lean_ctor_get(v___x_3737_, 1);
lean_inc_ref(v_opts_3738_);
lean_dec(v___x_3737_);
v___x_3739_ = 1;
v___x_3740_ = l_Lean_instBEqMessageSeverity_beq(v_severity_3604_, v___x_3739_);
if (v___x_3740_ == 0)
{
lean_dec_ref(v_opts_3738_);
v___y_3729_ = v___y_3733_;
v___y_3730_ = v___y_3733_;
v___y_3731_ = v___x_3740_;
goto v___jp_3728_;
}
else
{
lean_object* v___x_3741_; uint8_t v___x_3742_; 
v___x_3741_ = l_Lean_warningAsError;
v___x_3742_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29_spec__33(v_opts_3738_, v___x_3741_);
lean_dec_ref(v_opts_3738_);
v___y_3729_ = v___y_3733_;
v___y_3730_ = v___y_3733_;
v___y_3731_ = v___x_3742_;
goto v___jp_3728_;
}
}
else
{
lean_object* v___x_3743_; lean_object* v___x_3744_; 
lean_dec_ref(v_msgData_3603_);
v___x_3743_ = lean_box(0);
v___x_3744_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3744_, 0, v___x_3743_);
return v___x_3744_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29___boxed(lean_object* v_ref_3747_, lean_object* v_msgData_3748_, lean_object* v_severity_3749_, lean_object* v_isSilent_3750_, lean_object* v___y_3751_, lean_object* v___y_3752_, lean_object* v___y_3753_){
_start:
{
uint8_t v_severity_boxed_3754_; uint8_t v_isSilent_boxed_3755_; lean_object* v_res_3756_; 
v_severity_boxed_3754_ = lean_unbox(v_severity_3749_);
v_isSilent_boxed_3755_ = lean_unbox(v_isSilent_3750_);
v_res_3756_ = lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29(v_ref_3747_, v_msgData_3748_, v_severity_boxed_3754_, v_isSilent_boxed_3755_, v___y_3751_, v___y_3752_);
lean_dec(v___y_3752_);
lean_dec_ref(v___y_3751_);
lean_dec(v_ref_3747_);
return v_res_3756_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__23_spec__31(lean_object* v_ref_3757_, lean_object* v_msgData_3758_, lean_object* v___y_3759_, lean_object* v___y_3760_){
_start:
{
uint8_t v___x_3762_; uint8_t v___x_3763_; lean_object* v___x_3764_; 
v___x_3762_ = 1;
v___x_3763_ = 0;
v___x_3764_ = lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29(v_ref_3757_, v_msgData_3758_, v___x_3762_, v___x_3763_, v___y_3759_, v___y_3760_);
return v___x_3764_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__23_spec__31___boxed(lean_object* v_ref_3765_, lean_object* v_msgData_3766_, lean_object* v___y_3767_, lean_object* v___y_3768_, lean_object* v___y_3769_){
_start:
{
lean_object* v_res_3770_; 
v_res_3770_ = lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__23_spec__31(v_ref_3765_, v_msgData_3766_, v___y_3767_, v___y_3768_);
lean_dec(v___y_3768_);
lean_dec_ref(v___y_3767_);
lean_dec(v_ref_3765_);
return v_res_3770_;
}
}
static lean_object* _init_lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__23___closed__1(void){
_start:
{
lean_object* v___x_3772_; lean_object* v___x_3773_; 
v___x_3772_ = ((lean_object*)(lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__23___closed__0));
v___x_3773_ = l_Lean_stringToMessageData(v___x_3772_);
return v___x_3773_;
}
}
static lean_object* _init_lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__23___closed__3(void){
_start:
{
lean_object* v___x_3775_; lean_object* v___x_3776_; 
v___x_3775_ = ((lean_object*)(lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__23___closed__2));
v___x_3776_ = l_Lean_stringToMessageData(v___x_3775_);
return v___x_3776_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__23(lean_object* v_linterOption_3777_, lean_object* v_stx_3778_, lean_object* v_msg_3779_, lean_object* v___y_3780_, lean_object* v___y_3781_){
_start:
{
lean_object* v_name_3783_; lean_object* v___x_3785_; uint8_t v_isShared_3786_; uint8_t v_isSharedCheck_3801_; 
v_name_3783_ = lean_ctor_get(v_linterOption_3777_, 0);
v_isSharedCheck_3801_ = !lean_is_exclusive(v_linterOption_3777_);
if (v_isSharedCheck_3801_ == 0)
{
lean_object* v_unused_3802_; 
v_unused_3802_ = lean_ctor_get(v_linterOption_3777_, 1);
lean_dec(v_unused_3802_);
v___x_3785_ = v_linterOption_3777_;
v_isShared_3786_ = v_isSharedCheck_3801_;
goto v_resetjp_3784_;
}
else
{
lean_inc(v_name_3783_);
lean_dec(v_linterOption_3777_);
v___x_3785_ = lean_box(0);
v_isShared_3786_ = v_isSharedCheck_3801_;
goto v_resetjp_3784_;
}
v_resetjp_3784_:
{
lean_object* v___x_3787_; lean_object* v___x_3788_; lean_object* v___x_3790_; 
v___x_3787_ = lean_obj_once(&lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__23___closed__1, &lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__23___closed__1_once, _init_lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__23___closed__1);
lean_inc(v_name_3783_);
v___x_3788_ = l_Lean_MessageData_ofName(v_name_3783_);
if (v_isShared_3786_ == 0)
{
lean_ctor_set_tag(v___x_3785_, 7);
lean_ctor_set(v___x_3785_, 1, v___x_3788_);
lean_ctor_set(v___x_3785_, 0, v___x_3787_);
v___x_3790_ = v___x_3785_;
goto v_reusejp_3789_;
}
else
{
lean_object* v_reuseFailAlloc_3800_; 
v_reuseFailAlloc_3800_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3800_, 0, v___x_3787_);
lean_ctor_set(v_reuseFailAlloc_3800_, 1, v___x_3788_);
v___x_3790_ = v_reuseFailAlloc_3800_;
goto v_reusejp_3789_;
}
v_reusejp_3789_:
{
lean_object* v___x_3791_; lean_object* v___x_3792_; lean_object* v_disable_3793_; lean_object* v___x_3794_; lean_object* v___x_3795_; lean_object* v___x_3796_; lean_object* v___x_3797_; lean_object* v___x_3798_; lean_object* v___x_3799_; 
v___x_3791_ = lean_obj_once(&lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__23___closed__3, &lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__23___closed__3_once, _init_lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__23___closed__3);
v___x_3792_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3792_, 0, v___x_3790_);
lean_ctor_set(v___x_3792_, 1, v___x_3791_);
v_disable_3793_ = l_Lean_MessageData_note(v___x_3792_);
v___x_3794_ = l_Lean_Linter_linterMessageTag;
v___x_3795_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3795_, 0, v_msg_3779_);
lean_ctor_set(v___x_3795_, 1, v_disable_3793_);
v___x_3796_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_3796_, 0, v___x_3794_);
lean_ctor_set(v___x_3796_, 1, v___x_3795_);
v___x_3797_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_3797_, 0, v_name_3783_);
lean_ctor_set(v___x_3797_, 1, v___x_3796_);
lean_inc(v_stx_3778_);
v___x_3798_ = lean_alloc_ctor(11, 2, 0);
lean_ctor_set(v___x_3798_, 0, v_stx_3778_);
lean_ctor_set(v___x_3798_, 1, v___x_3797_);
v___x_3799_ = lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__23_spec__31(v_stx_3778_, v___x_3798_, v___y_3780_, v___y_3781_);
lean_dec(v_stx_3778_);
return v___x_3799_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__23___boxed(lean_object* v_linterOption_3803_, lean_object* v_stx_3804_, lean_object* v_msg_3805_, lean_object* v___y_3806_, lean_object* v___y_3807_, lean_object* v___y_3808_){
_start:
{
lean_object* v_res_3809_; 
v_res_3809_ = lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__23(v_linterOption_3803_, v_stx_3804_, v_msg_3805_, v___y_3806_, v___y_3807_);
lean_dec(v___y_3807_);
lean_dec_ref(v___y_3806_);
return v_res_3809_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22(lean_object* v_ref_3810_, lean_object* v_msgData_3811_, lean_object* v___y_3812_, lean_object* v___y_3813_){
_start:
{
uint8_t v___x_3815_; uint8_t v___x_3816_; lean_object* v___x_3817_; 
v___x_3815_ = 0;
v___x_3816_ = 0;
v___x_3817_ = lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29(v_ref_3810_, v_msgData_3811_, v___x_3815_, v___x_3816_, v___y_3812_, v___y_3813_);
return v___x_3817_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22___boxed(lean_object* v_ref_3818_, lean_object* v_msgData_3819_, lean_object* v___y_3820_, lean_object* v___y_3821_, lean_object* v___y_3822_){
_start:
{
lean_object* v_res_3823_; 
v_res_3823_ = lp_mathlib_Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22(v_ref_3818_, v_msgData_3819_, v___y_3820_, v___y_3821_);
lean_dec(v___y_3821_);
lean_dec_ref(v___y_3820_);
lean_dec(v_ref_3818_);
return v_res_3823_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__1(void){
_start:
{
lean_object* v___x_3825_; lean_object* v___x_3826_; 
v___x_3825_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__0));
v___x_3826_ = l_Lean_stringToMessageData(v___x_3825_);
return v___x_3826_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__3(void){
_start:
{
lean_object* v___x_3828_; lean_object* v___x_3829_; 
v___x_3828_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__2));
v___x_3829_ = l_Lean_stringToMessageData(v___x_3828_);
return v___x_3829_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__5(void){
_start:
{
lean_object* v___x_3831_; lean_object* v___x_3832_; 
v___x_3831_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__4));
v___x_3832_ = l_Lean_stringToMessageData(v___x_3831_);
return v___x_3832_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__7(void){
_start:
{
lean_object* v___x_3834_; lean_object* v___x_3835_; 
v___x_3834_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__6));
v___x_3835_ = l_Lean_stringToMessageData(v___x_3834_);
return v___x_3835_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__9(void){
_start:
{
lean_object* v___x_3837_; lean_object* v___x_3838_; 
v___x_3837_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__8));
v___x_3838_ = l_Lean_stringToMessageData(v___x_3837_);
return v___x_3838_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__11(void){
_start:
{
lean_object* v___x_3840_; lean_object* v___x_3841_; 
v___x_3840_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__10));
v___x_3841_ = l_Lean_stringToMessageData(v___x_3840_);
return v___x_3841_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__13(void){
_start:
{
lean_object* v___x_3843_; lean_object* v___x_3844_; 
v___x_3843_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__12));
v___x_3844_ = l_Lean_stringToMessageData(v___x_3843_);
return v___x_3844_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__16(void){
_start:
{
lean_object* v___x_3848_; lean_object* v___x_3849_; 
v___x_3848_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__15));
v___x_3849_ = l_Lean_MessageData_ofFormat(v___x_3848_);
return v___x_3849_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__19(void){
_start:
{
lean_object* v___x_3853_; lean_object* v___x_3854_; 
v___x_3853_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__18));
v___x_3854_ = l_Lean_MessageData_ofFormat(v___x_3853_);
return v___x_3854_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__21(void){
_start:
{
lean_object* v___x_3857_; lean_object* v___x_3858_; 
v___x_3857_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__20));
v___x_3858_ = l_Lean_MessageData_ofFormat(v___x_3857_);
return v___x_3858_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__27(void){
_start:
{
lean_object* v___x_3865_; lean_object* v___x_3866_; 
v___x_3865_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__26));
v___x_3866_ = l_Lean_stringToMessageData(v___x_3865_);
return v___x_3866_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__29(void){
_start:
{
lean_object* v___x_3868_; lean_object* v___x_3869_; 
v___x_3868_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__28));
v___x_3869_ = l_Lean_stringToMessageData(v___x_3868_);
return v___x_3869_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__31(void){
_start:
{
lean_object* v___x_3871_; lean_object* v___x_3872_; 
v___x_3871_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__30));
v___x_3872_ = l_Lean_stringToMessageData(v___x_3871_);
return v___x_3872_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__33(void){
_start:
{
lean_object* v___x_3874_; lean_object* v___x_3875_; 
v___x_3874_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__32));
v___x_3875_ = l_Lean_stringToMessageData(v___x_3874_);
return v___x_3875_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24(uint8_t v___y_3876_, uint8_t v___x_3877_, lean_object* v_as_3878_, size_t v_sz_3879_, size_t v_i_3880_, lean_object* v_b_3881_, lean_object* v___y_3882_, lean_object* v___y_3883_){
_start:
{
uint8_t v___x_3885_; 
v___x_3885_ = lean_usize_dec_lt(v_i_3880_, v_sz_3879_);
if (v___x_3885_ == 0)
{
lean_object* v___x_3886_; 
v___x_3886_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3886_, 0, v_b_3881_);
return v___x_3886_;
}
else
{
lean_object* v_a_3887_; lean_object* v_snd_3888_; lean_object* v_fst_3889_; lean_object* v___x_3891_; uint8_t v_isShared_3892_; uint8_t v_isSharedCheck_4188_; 
v_a_3887_ = lean_array_uget(v_as_3878_, v_i_3880_);
v_snd_3888_ = lean_ctor_get(v_a_3887_, 1);
v_fst_3889_ = lean_ctor_get(v_a_3887_, 0);
v_isSharedCheck_4188_ = !lean_is_exclusive(v_a_3887_);
if (v_isSharedCheck_4188_ == 0)
{
v___x_3891_ = v_a_3887_;
v_isShared_3892_ = v_isSharedCheck_4188_;
goto v_resetjp_3890_;
}
else
{
lean_inc(v_snd_3888_);
lean_inc(v_fst_3889_);
lean_dec(v_a_3887_);
v___x_3891_ = lean_box(0);
v_isShared_3892_ = v_isSharedCheck_4188_;
goto v_resetjp_3890_;
}
v_resetjp_3890_:
{
lean_object* v_stained_3893_; lean_object* v_stx_3894_; lean_object* v___x_3895_; lean_object* v___y_3897_; lean_object* v___y_3898_; lean_object* v___y_3899_; lean_object* v___y_3905_; lean_object* v___y_3906_; lean_object* v___y_3907_; lean_object* v___y_3908_; lean_object* v___x_3960_; lean_object* v___y_3962_; lean_object* v___y_3963_; lean_object* v___y_3964_; lean_object* v___y_3982_; lean_object* v___y_3983_; lean_object* v___x_4001_; lean_object* v___y_4003_; lean_object* v___y_4004_; lean_object* v___y_4025_; lean_object* v___y_4026_; lean_object* v___y_4027_; lean_object* v___y_4034_; lean_object* v___y_4035_; lean_object* v___y_4036_; lean_object* v___y_4043_; lean_object* v___y_4044_; lean_object* v___y_4045_; lean_object* v___y_4052_; lean_object* v___x_4182_; 
v_stained_3893_ = lean_ctor_get(v_snd_3888_, 0);
lean_inc(v_stained_3893_);
v_stx_3894_ = lean_ctor_get(v_snd_3888_, 1);
lean_inc_n(v_stx_3894_, 2);
v___x_3895_ = lean_box(0);
v___x_3960_ = lean_unsigned_to_nat(0u);
v___x_4001_ = lp_mathlib_Mathlib_Linter_linter_flexible;
v___x_4182_ = l_Lean_Syntax_reprint(v_stx_3894_);
if (lean_obj_tag(v___x_4182_) == 0)
{
lean_object* v___x_4183_; lean_object* v___x_4184_; lean_object* v___x_4185_; lean_object* v___x_4186_; 
v___x_4183_ = lean_box(0);
lean_inc(v_stx_3894_);
v___x_4184_ = l_Lean_Syntax_formatStx(v_stx_3894_, v___x_4183_, v___x_3877_);
v___x_4185_ = l_Std_Format_defWidth;
v___x_4186_ = l_Std_Format_pretty(v___x_4184_, v___x_4185_, v___x_3960_, v___x_3960_);
v___y_4052_ = v___x_4186_;
goto v___jp_4051_;
}
else
{
lean_object* v_val_4187_; 
v_val_4187_ = lean_ctor_get(v___x_4182_, 0);
lean_inc(v_val_4187_);
lean_dec_ref_known(v___x_4182_, 1);
v___y_4052_ = v_val_4187_;
goto v___jp_4051_;
}
v___jp_3896_:
{
lean_object* v___x_3900_; 
v___x_3900_ = lp_mathlib_Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22(v_fst_3889_, v___y_3899_, v___y_3898_, v___y_3897_);
lean_dec(v_fst_3889_);
if (lean_obj_tag(v___x_3900_) == 0)
{
size_t v___x_3901_; size_t v___x_3902_; 
lean_dec_ref_known(v___x_3900_, 1);
v___x_3901_ = ((size_t)1ULL);
v___x_3902_ = lean_usize_add(v_i_3880_, v___x_3901_);
v_i_3880_ = v___x_3902_;
v_b_3881_ = v___x_3895_;
goto _start;
}
else
{
return v___x_3900_;
}
}
v___jp_3904_:
{
switch(lean_obj_tag(v_stained_3893_))
{
case 0:
{
lean_object* v_a_3909_; lean_object* v___x_3911_; uint8_t v_isShared_3912_; uint8_t v_isSharedCheck_3933_; 
v_a_3909_ = lean_ctor_get(v_stained_3893_, 0);
v_isSharedCheck_3933_ = !lean_is_exclusive(v_stained_3893_);
if (v_isSharedCheck_3933_ == 0)
{
v___x_3911_ = v_stained_3893_;
v_isShared_3912_ = v_isSharedCheck_3933_;
goto v_resetjp_3910_;
}
else
{
lean_inc(v_a_3909_);
lean_dec(v_stained_3893_);
v___x_3911_ = lean_box(0);
v_isShared_3912_ = v_isSharedCheck_3933_;
goto v_resetjp_3910_;
}
v_resetjp_3910_:
{
lean_object* v___x_3913_; lean_object* v___x_3914_; lean_object* v___x_3916_; 
v___x_3913_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__1, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__1_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__1);
lean_inc(v_fst_3889_);
v___x_3914_ = l_Lean_MessageData_ofSyntax(v_fst_3889_);
if (v_isShared_3912_ == 0)
{
lean_ctor_set_tag(v___x_3911_, 6);
lean_ctor_set(v___x_3911_, 0, v___x_3914_);
v___x_3916_ = v___x_3911_;
goto v_reusejp_3915_;
}
else
{
lean_object* v_reuseFailAlloc_3932_; 
v_reuseFailAlloc_3932_ = lean_alloc_ctor(6, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3932_, 0, v___x_3914_);
v___x_3916_ = v_reuseFailAlloc_3932_;
goto v_reusejp_3915_;
}
v_reusejp_3915_:
{
lean_object* v___x_3918_; 
if (v_isShared_3892_ == 0)
{
lean_ctor_set_tag(v___x_3891_, 7);
lean_ctor_set(v___x_3891_, 1, v___x_3916_);
lean_ctor_set(v___x_3891_, 0, v___x_3913_);
v___x_3918_ = v___x_3891_;
goto v_reusejp_3917_;
}
else
{
lean_object* v_reuseFailAlloc_3931_; 
v_reuseFailAlloc_3931_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3931_, 0, v___x_3913_);
lean_ctor_set(v_reuseFailAlloc_3931_, 1, v___x_3916_);
v___x_3918_ = v_reuseFailAlloc_3931_;
goto v_reusejp_3917_;
}
v_reusejp_3917_:
{
lean_object* v___x_3919_; lean_object* v___x_3920_; lean_object* v___x_3921_; lean_object* v___x_3922_; lean_object* v___x_3923_; lean_object* v___x_3924_; lean_object* v___x_3925_; lean_object* v___x_3926_; lean_object* v___x_3927_; lean_object* v___x_3928_; lean_object* v___x_3929_; lean_object* v___x_3930_; 
v___x_3919_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__1, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__1_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__1);
v___x_3920_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3920_, 0, v___x_3918_);
lean_ctor_set(v___x_3920_, 1, v___x_3919_);
v___x_3921_ = l_Lean_Name_toString(v_a_3909_, v___y_3876_);
v___x_3922_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_3922_, 0, v___x_3921_);
v___x_3923_ = l_Lean_MessageData_ofFormat(v___x_3922_);
v___x_3924_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3924_, 0, v___x_3920_);
lean_ctor_set(v___x_3924_, 1, v___x_3923_);
v___x_3925_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__3, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__3_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__3);
v___x_3926_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3926_, 0, v___x_3924_);
lean_ctor_set(v___x_3926_, 1, v___x_3925_);
v___x_3927_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3927_, 0, v___x_3926_);
lean_ctor_set(v___x_3927_, 1, v___y_3908_);
v___x_3928_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3928_, 0, v___x_3927_);
lean_ctor_set(v___x_3928_, 1, v___y_3905_);
v___x_3929_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__5, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__5_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__5);
v___x_3930_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3930_, 0, v___x_3928_);
lean_ctor_set(v___x_3930_, 1, v___x_3929_);
v___y_3897_ = v___y_3907_;
v___y_3898_ = v___y_3906_;
v___y_3899_ = v___x_3930_;
goto v___jp_3896_;
}
}
}
}
case 1:
{
lean_object* v___x_3934_; lean_object* v___x_3935_; lean_object* v___x_3936_; lean_object* v___x_3938_; 
v___x_3934_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__1, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__1_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__1);
lean_inc(v_fst_3889_);
v___x_3935_ = l_Lean_MessageData_ofSyntax(v_fst_3889_);
v___x_3936_ = lean_alloc_ctor(6, 1, 0);
lean_ctor_set(v___x_3936_, 0, v___x_3935_);
if (v_isShared_3892_ == 0)
{
lean_ctor_set_tag(v___x_3891_, 7);
lean_ctor_set(v___x_3891_, 1, v___x_3936_);
lean_ctor_set(v___x_3891_, 0, v___x_3934_);
v___x_3938_ = v___x_3891_;
goto v_reusejp_3937_;
}
else
{
lean_object* v_reuseFailAlloc_3945_; 
v_reuseFailAlloc_3945_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3945_, 0, v___x_3934_);
lean_ctor_set(v_reuseFailAlloc_3945_, 1, v___x_3936_);
v___x_3938_ = v_reuseFailAlloc_3945_;
goto v_reusejp_3937_;
}
v_reusejp_3937_:
{
lean_object* v___x_3939_; lean_object* v___x_3940_; lean_object* v___x_3941_; lean_object* v___x_3942_; lean_object* v___x_3943_; lean_object* v___x_3944_; 
v___x_3939_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__7, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__7_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__7);
v___x_3940_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3940_, 0, v___x_3938_);
lean_ctor_set(v___x_3940_, 1, v___x_3939_);
v___x_3941_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3941_, 0, v___x_3940_);
lean_ctor_set(v___x_3941_, 1, v___y_3908_);
v___x_3942_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3942_, 0, v___x_3941_);
lean_ctor_set(v___x_3942_, 1, v___y_3905_);
v___x_3943_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__5, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__5_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__5);
v___x_3944_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3944_, 0, v___x_3942_);
lean_ctor_set(v___x_3944_, 1, v___x_3943_);
v___y_3897_ = v___y_3907_;
v___y_3898_ = v___y_3906_;
v___y_3899_ = v___x_3944_;
goto v___jp_3896_;
}
}
default: 
{
lean_object* v___x_3946_; lean_object* v___x_3947_; lean_object* v___x_3948_; lean_object* v___x_3950_; 
v___x_3946_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__1, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__1_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__1);
lean_inc(v_fst_3889_);
v___x_3947_ = l_Lean_MessageData_ofSyntax(v_fst_3889_);
v___x_3948_ = lean_alloc_ctor(6, 1, 0);
lean_ctor_set(v___x_3948_, 0, v___x_3947_);
if (v_isShared_3892_ == 0)
{
lean_ctor_set_tag(v___x_3891_, 7);
lean_ctor_set(v___x_3891_, 1, v___x_3948_);
lean_ctor_set(v___x_3891_, 0, v___x_3946_);
v___x_3950_ = v___x_3891_;
goto v_reusejp_3949_;
}
else
{
lean_object* v_reuseFailAlloc_3959_; 
v_reuseFailAlloc_3959_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3959_, 0, v___x_3946_);
lean_ctor_set(v_reuseFailAlloc_3959_, 1, v___x_3948_);
v___x_3950_ = v_reuseFailAlloc_3959_;
goto v_reusejp_3949_;
}
v_reusejp_3949_:
{
lean_object* v___x_3951_; lean_object* v___x_3952_; lean_object* v___x_3953_; lean_object* v___x_3954_; lean_object* v___x_3955_; lean_object* v___x_3956_; lean_object* v___x_3957_; lean_object* v___x_3958_; 
v___x_3951_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__9, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__9_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__9);
v___x_3952_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3952_, 0, v___x_3950_);
lean_ctor_set(v___x_3952_, 1, v___x_3951_);
v___x_3953_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3953_, 0, v___x_3952_);
lean_ctor_set(v___x_3953_, 1, v___y_3908_);
v___x_3954_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__11, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__11_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__11);
v___x_3955_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3955_, 0, v___x_3953_);
lean_ctor_set(v___x_3955_, 1, v___x_3954_);
v___x_3956_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3956_, 0, v___x_3955_);
lean_ctor_set(v___x_3956_, 1, v___y_3905_);
v___x_3957_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__13, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__13_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__13);
v___x_3958_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3958_, 0, v___x_3956_);
lean_ctor_set(v___x_3958_, 1, v___x_3957_);
v___y_3897_ = v___y_3907_;
v___y_3898_ = v___y_3906_;
v___y_3899_ = v___x_3958_;
goto v___jp_3896_;
}
}
}
}
v___jp_3961_:
{
lean_object* v___x_3965_; 
v___x_3965_ = l_Lean_Syntax_getArg(v_stx_3894_, v___x_3960_);
lean_dec(v_stx_3894_);
if (lean_obj_tag(v___x_3965_) == 2)
{
lean_object* v_val_3966_; lean_object* v___x_3968_; uint8_t v_isShared_3969_; uint8_t v_isSharedCheck_3978_; 
v_val_3966_ = lean_ctor_get(v___x_3965_, 1);
v_isSharedCheck_3978_ = !lean_is_exclusive(v___x_3965_);
if (v_isSharedCheck_3978_ == 0)
{
lean_object* v_unused_3979_; 
v_unused_3979_ = lean_ctor_get(v___x_3965_, 0);
lean_dec(v_unused_3979_);
v___x_3968_ = v___x_3965_;
v_isShared_3969_ = v_isSharedCheck_3978_;
goto v_resetjp_3967_;
}
else
{
lean_inc(v_val_3966_);
lean_dec(v___x_3965_);
v___x_3968_ = lean_box(0);
v_isShared_3969_ = v_isSharedCheck_3978_;
goto v_resetjp_3967_;
}
v_resetjp_3967_:
{
lean_object* v___x_3970_; lean_object* v___x_3971_; lean_object* v___x_3972_; lean_object* v___x_3974_; 
v___x_3970_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__16, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__16_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__16);
v___x_3971_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__1, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__1_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__1);
v___x_3972_ = l_Lean_stringToMessageData(v_val_3966_);
if (v_isShared_3969_ == 0)
{
lean_ctor_set_tag(v___x_3968_, 7);
lean_ctor_set(v___x_3968_, 1, v___x_3972_);
lean_ctor_set(v___x_3968_, 0, v___x_3971_);
v___x_3974_ = v___x_3968_;
goto v_reusejp_3973_;
}
else
{
lean_object* v_reuseFailAlloc_3977_; 
v_reuseFailAlloc_3977_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3977_, 0, v___x_3971_);
lean_ctor_set(v_reuseFailAlloc_3977_, 1, v___x_3972_);
v___x_3974_ = v_reuseFailAlloc_3977_;
goto v_reusejp_3973_;
}
v_reusejp_3973_:
{
lean_object* v___x_3975_; lean_object* v___x_3976_; 
v___x_3975_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3975_, 0, v___x_3974_);
lean_ctor_set(v___x_3975_, 1, v___x_3971_);
v___x_3976_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3976_, 0, v___x_3970_);
lean_ctor_set(v___x_3976_, 1, v___x_3975_);
v___y_3905_ = v___y_3964_;
v___y_3906_ = v___y_3963_;
v___y_3907_ = v___y_3962_;
v___y_3908_ = v___x_3976_;
goto v___jp_3904_;
}
}
}
else
{
lean_object* v___x_3980_; 
lean_dec(v___x_3965_);
v___x_3980_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__19, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__19_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__19);
v___y_3905_ = v___y_3964_;
v___y_3906_ = v___y_3963_;
v___y_3907_ = v___y_3962_;
v___y_3908_ = v___x_3980_;
goto v___jp_3904_;
}
}
v___jp_3981_:
{
lean_object* v___x_3984_; 
v___x_3984_ = l_Lean_Syntax_getPos_x3f(v_stx_3894_, v___x_3877_);
if (lean_obj_tag(v___x_3984_) == 0)
{
lean_object* v___x_3985_; 
v___x_3985_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__21, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__21_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__21);
v___y_3962_ = v___y_3983_;
v___y_3963_ = v___y_3982_;
v___y_3964_ = v___x_3985_;
goto v___jp_3961_;
}
else
{
lean_object* v_val_3986_; lean_object* v___x_3988_; uint8_t v_isShared_3989_; uint8_t v_isSharedCheck_4000_; 
v_val_3986_ = lean_ctor_get(v___x_3984_, 0);
v_isSharedCheck_4000_ = !lean_is_exclusive(v___x_3984_);
if (v_isSharedCheck_4000_ == 0)
{
v___x_3988_ = v___x_3984_;
v_isShared_3989_ = v_isSharedCheck_4000_;
goto v_resetjp_3987_;
}
else
{
lean_inc(v_val_3986_);
lean_dec(v___x_3984_);
v___x_3988_ = lean_box(0);
v_isShared_3989_ = v_isSharedCheck_4000_;
goto v_resetjp_3987_;
}
v_resetjp_3987_:
{
lean_object* v_fileMap_3990_; lean_object* v___x_3991_; lean_object* v_line_3992_; lean_object* v___x_3993_; lean_object* v___x_3994_; lean_object* v___x_3995_; lean_object* v___x_3997_; 
v_fileMap_3990_ = lean_ctor_get(v___y_3982_, 1);
lean_inc_ref(v_fileMap_3990_);
v___x_3991_ = l_Lean_FileMap_toPosition(v_fileMap_3990_, v_val_3986_);
lean_dec(v_val_3986_);
v_line_3992_ = lean_ctor_get(v___x_3991_, 0);
lean_inc(v_line_3992_);
lean_dec_ref(v___x_3991_);
v___x_3993_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__22));
v___x_3994_ = l_Nat_reprFast(v_line_3992_);
v___x_3995_ = lean_string_append(v___x_3993_, v___x_3994_);
lean_dec_ref(v___x_3994_);
if (v_isShared_3989_ == 0)
{
lean_ctor_set_tag(v___x_3988_, 3);
lean_ctor_set(v___x_3988_, 0, v___x_3995_);
v___x_3997_ = v___x_3988_;
goto v_reusejp_3996_;
}
else
{
lean_object* v_reuseFailAlloc_3999_; 
v_reuseFailAlloc_3999_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3999_, 0, v___x_3995_);
v___x_3997_ = v_reuseFailAlloc_3999_;
goto v_reusejp_3996_;
}
v_reusejp_3996_:
{
lean_object* v___x_3998_; 
v___x_3998_ = l_Lean_MessageData_ofFormat(v___x_3997_);
v___y_3962_ = v___y_3983_;
v___y_3963_ = v___y_3982_;
v___y_3964_ = v___x_3998_;
goto v___jp_3961_;
}
}
}
}
v___jp_4002_:
{
lean_object* v___x_4005_; 
lean_inc(v_stx_3894_);
v___x_4005_ = lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__23(v___x_4001_, v_stx_3894_, v___y_4004_, v___y_3882_, v___y_3883_);
if (lean_obj_tag(v___x_4005_) == 0)
{
lean_dec_ref_known(v___x_4005_, 1);
if (lean_obj_tag(v___y_4003_) == 1)
{
lean_object* v_val_4006_; lean_object* v___x_4008_; uint8_t v_isShared_4009_; uint8_t v_isSharedCheck_4023_; 
v_val_4006_ = lean_ctor_get(v___y_4003_, 0);
v_isSharedCheck_4023_ = !lean_is_exclusive(v___y_4003_);
if (v_isSharedCheck_4023_ == 0)
{
v___x_4008_ = v___y_4003_;
v_isShared_4009_ = v_isSharedCheck_4023_;
goto v_resetjp_4007_;
}
else
{
lean_inc(v_val_4006_);
lean_dec(v___y_4003_);
v___x_4008_ = lean_box(0);
v_isShared_4009_ = v_isSharedCheck_4023_;
goto v_resetjp_4007_;
}
v_resetjp_4007_:
{
lean_object* v___x_4010_; lean_object* v___x_4011_; lean_object* v___x_4012_; lean_object* v___x_4013_; lean_object* v___x_4015_; 
v___x_4010_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__24));
v___x_4011_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4011_, 0, v___x_4010_);
lean_ctor_set(v___x_4011_, 1, v_val_4006_);
v___x_4012_ = lean_box(0);
v___x_4013_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_4013_, 0, v___x_4011_);
lean_ctor_set(v___x_4013_, 1, v___x_4012_);
lean_ctor_set(v___x_4013_, 2, v___x_4012_);
lean_ctor_set(v___x_4013_, 3, v___x_4012_);
lean_ctor_set(v___x_4013_, 4, v___x_4012_);
lean_ctor_set(v___x_4013_, 5, v___x_4012_);
lean_inc(v_stx_3894_);
if (v_isShared_4009_ == 0)
{
lean_ctor_set(v___x_4008_, 0, v_stx_3894_);
v___x_4015_ = v___x_4008_;
goto v_reusejp_4014_;
}
else
{
lean_object* v_reuseFailAlloc_4022_; 
v_reuseFailAlloc_4022_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4022_, 0, v_stx_3894_);
v___x_4015_ = v_reuseFailAlloc_4022_;
goto v_reusejp_4014_;
}
v_reusejp_4014_:
{
lean_object* v___x_4016_; uint8_t v___x_4017_; lean_object* v___x_4018_; lean_object* v___x_4019_; lean_object* v___x_4020_; lean_object* v___x_4021_; 
v___x_4016_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__25));
v___x_4017_ = 4;
v___x_4018_ = l_Lean_MessageData_nil;
v___x_4019_ = lean_box(v___x_4017_);
lean_inc(v_stx_3894_);
v___x_4020_ = lean_alloc_closure((void*)(l_Lean_Meta_Tactic_TryThis_addSuggestion___boxed), 10, 7);
lean_closure_set(v___x_4020_, 0, v_stx_3894_);
lean_closure_set(v___x_4020_, 1, v___x_4013_);
lean_closure_set(v___x_4020_, 2, v___x_4015_);
lean_closure_set(v___x_4020_, 3, v___x_4016_);
lean_closure_set(v___x_4020_, 4, v___x_4012_);
lean_closure_set(v___x_4020_, 5, v___x_4019_);
lean_closure_set(v___x_4020_, 6, v___x_4018_);
v___x_4021_ = l_Lean_Elab_Command_liftCoreM___redArg(v___x_4020_, v___y_3882_, v___y_3883_);
if (lean_obj_tag(v___x_4021_) == 0)
{
lean_dec_ref_known(v___x_4021_, 1);
v___y_3982_ = v___y_3882_;
v___y_3983_ = v___y_3883_;
goto v___jp_3981_;
}
else
{
lean_dec(v_stx_3894_);
lean_dec(v_stained_3893_);
lean_del_object(v___x_3891_);
lean_dec(v_fst_3889_);
return v___x_4021_;
}
}
}
}
else
{
lean_dec(v___y_4003_);
v___y_3982_ = v___y_3882_;
v___y_3983_ = v___y_3883_;
goto v___jp_3981_;
}
}
else
{
lean_dec(v___y_4003_);
lean_dec(v_stx_3894_);
lean_dec(v_stained_3893_);
lean_del_object(v___x_3891_);
lean_dec(v_fst_3889_);
return v___x_4005_;
}
}
v___jp_4024_:
{
lean_object* v___x_4028_; lean_object* v___x_4029_; lean_object* v___x_4030_; lean_object* v___x_4031_; lean_object* v___x_4032_; 
v___x_4028_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_4028_, 0, v___y_4027_);
v___x_4029_ = l_Lean_MessageData_ofFormat(v___x_4028_);
v___x_4030_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4030_, 0, v___y_4026_);
lean_ctor_set(v___x_4030_, 1, v___x_4029_);
v___x_4031_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__27, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__27_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__27);
v___x_4032_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4032_, 0, v___x_4030_);
lean_ctor_set(v___x_4032_, 1, v___x_4031_);
v___y_4003_ = v___y_4025_;
v___y_4004_ = v___x_4032_;
goto v___jp_4002_;
}
v___jp_4033_:
{
lean_object* v___x_4037_; lean_object* v___x_4038_; lean_object* v___x_4039_; lean_object* v___x_4040_; lean_object* v___x_4041_; 
v___x_4037_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_4037_, 0, v___y_4036_);
v___x_4038_ = l_Lean_MessageData_ofFormat(v___x_4037_);
v___x_4039_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4039_, 0, v___y_4034_);
lean_ctor_set(v___x_4039_, 1, v___x_4038_);
v___x_4040_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__29, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__29_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__29);
v___x_4041_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4041_, 0, v___x_4039_);
lean_ctor_set(v___x_4041_, 1, v___x_4040_);
v___y_4003_ = v___y_4035_;
v___y_4004_ = v___x_4041_;
goto v___jp_4002_;
}
v___jp_4042_:
{
lean_object* v___x_4046_; lean_object* v___x_4047_; lean_object* v___x_4048_; lean_object* v___x_4049_; lean_object* v___x_4050_; 
v___x_4046_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_4046_, 0, v___y_4045_);
v___x_4047_ = l_Lean_MessageData_ofFormat(v___x_4046_);
v___x_4048_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4048_, 0, v___y_4043_);
lean_ctor_set(v___x_4048_, 1, v___x_4047_);
v___x_4049_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__31, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__31_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__31);
v___x_4050_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4050_, 0, v___x_4048_);
lean_ctor_set(v___x_4050_, 1, v___x_4049_);
v___y_4003_ = v___y_4044_;
v___y_4004_ = v___x_4050_;
goto v___jp_4002_;
}
v___jp_4051_:
{
lean_object* v___x_4053_; lean_object* v___x_4054_; 
lean_inc(v_stx_3894_);
v___x_4053_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_generateSimpSuggestion___boxed), 5, 2);
lean_closure_set(v___x_4053_, 0, v_snd_3888_);
lean_closure_set(v___x_4053_, 1, v_stx_3894_);
v___x_4054_ = l_Lean_Elab_Command_liftCoreM___redArg(v___x_4053_, v___y_3882_, v___y_3883_);
if (lean_obj_tag(v___x_4054_) == 0)
{
lean_object* v_a_4055_; lean_object* v___x_4056_; lean_object* v___x_4057_; lean_object* v___x_4058_; lean_object* v___x_4059_; 
v_a_4055_ = lean_ctor_get(v___x_4054_, 0);
lean_inc(v_a_4055_);
lean_dec_ref_known(v___x_4054_, 1);
v___x_4056_ = lean_string_utf8_byte_size(v___y_4052_);
v___x_4057_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_4057_, 0, v___y_4052_);
lean_ctor_set(v___x_4057_, 1, v___x_3960_);
lean_ctor_set(v___x_4057_, 2, v___x_4056_);
v___x_4058_ = l_String_Slice_trimAscii(v___x_4057_);
lean_inc(v_stx_3894_);
v___x_4059_ = l_Lean_Syntax_getKind(v_stx_3894_);
if (lean_obj_tag(v___x_4059_) == 1)
{
lean_object* v_pre_4060_; 
v_pre_4060_ = lean_ctor_get(v___x_4059_, 0);
lean_inc(v_pre_4060_);
if (lean_obj_tag(v_pre_4060_) == 1)
{
lean_object* v_pre_4061_; 
v_pre_4061_ = lean_ctor_get(v_pre_4060_, 0);
lean_inc(v_pre_4061_);
if (lean_obj_tag(v_pre_4061_) == 1)
{
lean_object* v_pre_4062_; 
v_pre_4062_ = lean_ctor_get(v_pre_4061_, 0);
lean_inc(v_pre_4062_);
if (lean_obj_tag(v_pre_4062_) == 1)
{
lean_object* v_pre_4063_; 
v_pre_4063_ = lean_ctor_get(v_pre_4062_, 0);
lean_inc(v_pre_4063_);
if (lean_obj_tag(v_pre_4063_) == 0)
{
lean_object* v_str_4064_; lean_object* v_str_4065_; lean_object* v_str_4066_; lean_object* v_str_4067_; lean_object* v___x_4068_; uint8_t v___x_4069_; 
v_str_4064_ = lean_ctor_get(v___x_4059_, 1);
lean_inc_ref(v_str_4064_);
v_str_4065_ = lean_ctor_get(v_pre_4060_, 1);
lean_inc_ref(v_str_4065_);
lean_dec_ref_known(v_pre_4060_, 2);
v_str_4066_ = lean_ctor_get(v_pre_4061_, 1);
lean_inc_ref(v_str_4066_);
lean_dec_ref_known(v_pre_4061_, 2);
v_str_4067_ = lean_ctor_get(v_pre_4062_, 1);
lean_inc_ref(v_str_4067_);
lean_dec_ref_known(v_pre_4062_, 2);
v___x_4068_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__0));
v___x_4069_ = lean_string_dec_eq(v_str_4067_, v___x_4068_);
if (v___x_4069_ == 0)
{
lean_object* v___x_4070_; uint8_t v___x_4071_; 
v___x_4070_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__61));
v___x_4071_ = lean_string_dec_eq(v_str_4067_, v___x_4070_);
lean_dec_ref(v_str_4067_);
if (v___x_4071_ == 0)
{
lean_object* v___x_4072_; 
lean_dec_ref(v_str_4066_);
lean_dec_ref(v_str_4065_);
lean_dec_ref(v_str_4064_);
lean_inc(v_stained_3893_);
v___x_4072_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0(v___x_4058_, v_stained_3893_, v___y_3876_, v___x_4059_);
lean_dec_ref_known(v___x_4059_, 2);
lean_dec_ref(v___x_4058_);
v___y_4003_ = v_a_4055_;
v___y_4004_ = v___x_4072_;
goto v___jp_4002_;
}
else
{
lean_object* v___x_4073_; uint8_t v___x_4074_; 
lean_dec_ref_known(v___x_4059_, 2);
v___x_4073_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__62));
v___x_4074_ = lean_string_dec_eq(v_str_4066_, v___x_4073_);
if (v___x_4074_ == 0)
{
lean_object* v___x_4075_; lean_object* v___x_4076_; lean_object* v___x_4077_; lean_object* v___x_4078_; lean_object* v___x_4079_; 
v___x_4075_ = l_Lean_Name_str___override(v_pre_4063_, v___x_4070_);
v___x_4076_ = l_Lean_Name_str___override(v___x_4075_, v_str_4066_);
v___x_4077_ = l_Lean_Name_str___override(v___x_4076_, v_str_4065_);
v___x_4078_ = l_Lean_Name_str___override(v___x_4077_, v_str_4064_);
lean_inc(v_stained_3893_);
v___x_4079_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0(v___x_4058_, v_stained_3893_, v___y_3876_, v___x_4078_);
lean_dec(v___x_4078_);
lean_dec_ref(v___x_4058_);
v___y_4003_ = v_a_4055_;
v___y_4004_ = v___x_4079_;
goto v___jp_4002_;
}
else
{
lean_object* v___x_4080_; uint8_t v___x_4081_; 
lean_dec_ref(v_str_4066_);
v___x_4080_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__1));
v___x_4081_ = lean_string_dec_eq(v_str_4065_, v___x_4080_);
if (v___x_4081_ == 0)
{
lean_object* v___x_4082_; lean_object* v___x_4083_; lean_object* v___x_4084_; lean_object* v___x_4085_; lean_object* v___x_4086_; 
v___x_4082_ = l_Lean_Name_str___override(v_pre_4063_, v___x_4070_);
v___x_4083_ = l_Lean_Name_str___override(v___x_4082_, v___x_4073_);
v___x_4084_ = l_Lean_Name_str___override(v___x_4083_, v_str_4065_);
v___x_4085_ = l_Lean_Name_str___override(v___x_4084_, v_str_4064_);
lean_inc(v_stained_3893_);
v___x_4086_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0(v___x_4058_, v_stained_3893_, v___y_3876_, v___x_4085_);
lean_dec(v___x_4085_);
lean_dec_ref(v___x_4058_);
v___y_4003_ = v_a_4055_;
v___y_4004_ = v___x_4086_;
goto v___jp_4002_;
}
else
{
lean_object* v___x_4087_; uint8_t v___x_4088_; 
lean_dec_ref(v_str_4065_);
v___x_4087_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible___closed__63));
v___x_4088_ = lean_string_dec_eq(v_str_4064_, v___x_4087_);
if (v___x_4088_ == 0)
{
lean_object* v___x_4089_; lean_object* v___x_4090_; lean_object* v___x_4091_; lean_object* v___x_4092_; lean_object* v___x_4093_; 
v___x_4089_ = l_Lean_Name_str___override(v_pre_4063_, v___x_4070_);
v___x_4090_ = l_Lean_Name_str___override(v___x_4089_, v___x_4073_);
v___x_4091_ = l_Lean_Name_str___override(v___x_4090_, v___x_4080_);
v___x_4092_ = l_Lean_Name_str___override(v___x_4091_, v_str_4064_);
lean_inc(v_stained_3893_);
v___x_4093_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0(v___x_4058_, v_stained_3893_, v___y_3876_, v___x_4092_);
lean_dec(v___x_4092_);
lean_dec_ref(v___x_4058_);
v___y_4003_ = v_a_4055_;
v___y_4004_ = v___x_4093_;
goto v___jp_4002_;
}
else
{
lean_object* v_str_4094_; lean_object* v_startInclusive_4095_; lean_object* v_endExclusive_4096_; lean_object* v___x_4097_; lean_object* v___x_4098_; lean_object* v___x_4099_; lean_object* v___x_4100_; lean_object* v___x_4101_; lean_object* v___x_4102_; lean_object* v___x_4103_; 
lean_dec_ref(v_str_4064_);
v_str_4094_ = lean_ctor_get(v___x_4058_, 0);
lean_inc_ref(v_str_4094_);
v_startInclusive_4095_ = lean_ctor_get(v___x_4058_, 1);
lean_inc(v_startInclusive_4095_);
v_endExclusive_4096_ = lean_ctor_get(v___x_4058_, 2);
lean_inc(v_endExclusive_4096_);
lean_dec_ref(v___x_4058_);
v___x_4097_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__1, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__1_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__1);
v___x_4098_ = lean_string_utf8_extract_fast(v_str_4094_, v_startInclusive_4095_, v_endExclusive_4096_);
lean_dec(v_endExclusive_4096_);
lean_dec(v_startInclusive_4095_);
lean_dec_ref(v_str_4094_);
v___x_4099_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_4099_, 0, v___x_4098_);
v___x_4100_ = l_Lean_MessageData_ofFormat(v___x_4099_);
v___x_4101_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4101_, 0, v___x_4097_);
lean_ctor_set(v___x_4101_, 1, v___x_4100_);
v___x_4102_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__3, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__3_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__3);
v___x_4103_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4103_, 0, v___x_4101_);
lean_ctor_set(v___x_4103_, 1, v___x_4102_);
switch(lean_obj_tag(v_stained_3893_))
{
case 0:
{
lean_object* v_a_4104_; lean_object* v___x_4105_; 
v_a_4104_ = lean_ctor_get(v_stained_3893_, 0);
lean_inc(v_a_4104_);
v___x_4105_ = l_Lean_Name_toString(v_a_4104_, v___y_3876_);
v___y_4025_ = v_a_4055_;
v___y_4026_ = v___x_4103_;
v___y_4027_ = v___x_4105_;
goto v___jp_4024_;
}
case 1:
{
lean_object* v___x_4106_; 
v___x_4106_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instToStringStained___lam__0___closed__0));
v___y_4025_ = v_a_4055_;
v___y_4026_ = v___x_4103_;
v___y_4027_ = v___x_4106_;
goto v___jp_4024_;
}
default: 
{
lean_object* v___x_4107_; 
v___x_4107_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instToStringStained___lam__0___closed__1));
v___y_4025_ = v_a_4055_;
v___y_4026_ = v___x_4103_;
v___y_4027_ = v___x_4107_;
goto v___jp_4024_;
}
}
}
}
}
}
}
else
{
lean_object* v___x_4108_; uint8_t v___x_4109_; 
lean_dec_ref(v_str_4067_);
lean_dec_ref_known(v___x_4059_, 2);
v___x_4108_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__1));
v___x_4109_ = lean_string_dec_eq(v_str_4066_, v___x_4108_);
if (v___x_4109_ == 0)
{
lean_object* v___x_4110_; lean_object* v___x_4111_; lean_object* v___x_4112_; lean_object* v___x_4113_; lean_object* v___x_4114_; 
v___x_4110_ = l_Lean_Name_str___override(v_pre_4063_, v___x_4068_);
v___x_4111_ = l_Lean_Name_str___override(v___x_4110_, v_str_4066_);
v___x_4112_ = l_Lean_Name_str___override(v___x_4111_, v_str_4065_);
v___x_4113_ = l_Lean_Name_str___override(v___x_4112_, v_str_4064_);
lean_inc(v_stained_3893_);
v___x_4114_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0(v___x_4058_, v_stained_3893_, v___y_3876_, v___x_4113_);
lean_dec(v___x_4113_);
lean_dec_ref(v___x_4058_);
v___y_4003_ = v_a_4055_;
v___y_4004_ = v___x_4114_;
goto v___jp_4002_;
}
else
{
lean_object* v___x_4115_; uint8_t v___x_4116_; 
lean_dec_ref(v_str_4066_);
v___x_4115_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__2));
v___x_4116_ = lean_string_dec_eq(v_str_4065_, v___x_4115_);
if (v___x_4116_ == 0)
{
lean_object* v___x_4117_; lean_object* v___x_4118_; lean_object* v___x_4119_; lean_object* v___x_4120_; lean_object* v___x_4121_; 
v___x_4117_ = l_Lean_Name_str___override(v_pre_4063_, v___x_4068_);
v___x_4118_ = l_Lean_Name_str___override(v___x_4117_, v___x_4108_);
v___x_4119_ = l_Lean_Name_str___override(v___x_4118_, v_str_4065_);
v___x_4120_ = l_Lean_Name_str___override(v___x_4119_, v_str_4064_);
lean_inc(v_stained_3893_);
v___x_4121_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0(v___x_4058_, v_stained_3893_, v___y_3876_, v___x_4120_);
lean_dec(v___x_4120_);
lean_dec_ref(v___x_4058_);
v___y_4003_ = v_a_4055_;
v___y_4004_ = v___x_4121_;
goto v___jp_4002_;
}
else
{
lean_object* v___x_4122_; uint8_t v___x_4123_; 
lean_dec_ref(v_str_4065_);
v___x_4122_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__3));
v___x_4123_ = lean_string_dec_eq(v_str_4064_, v___x_4122_);
if (v___x_4123_ == 0)
{
lean_object* v___x_4124_; uint8_t v___x_4125_; 
v___x_4124_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_flexible_x3f___closed__4));
v___x_4125_ = lean_string_dec_eq(v_str_4064_, v___x_4124_);
if (v___x_4125_ == 0)
{
lean_object* v___x_4126_; lean_object* v___x_4127_; lean_object* v___x_4128_; lean_object* v___x_4129_; lean_object* v___x_4130_; 
v___x_4126_ = l_Lean_Name_str___override(v_pre_4063_, v___x_4068_);
v___x_4127_ = l_Lean_Name_str___override(v___x_4126_, v___x_4108_);
v___x_4128_ = l_Lean_Name_str___override(v___x_4127_, v___x_4115_);
v___x_4129_ = l_Lean_Name_str___override(v___x_4128_, v_str_4064_);
lean_inc(v_stained_3893_);
v___x_4130_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0(v___x_4058_, v_stained_3893_, v___y_3876_, v___x_4129_);
lean_dec(v___x_4129_);
lean_dec_ref(v___x_4058_);
v___y_4003_ = v_a_4055_;
v___y_4004_ = v___x_4130_;
goto v___jp_4002_;
}
else
{
lean_object* v_str_4131_; lean_object* v_startInclusive_4132_; lean_object* v_endExclusive_4133_; lean_object* v___x_4134_; lean_object* v___x_4135_; lean_object* v___x_4136_; lean_object* v___x_4137_; lean_object* v___x_4138_; lean_object* v___x_4139_; lean_object* v___x_4140_; 
lean_dec_ref(v_str_4064_);
v_str_4131_ = lean_ctor_get(v___x_4058_, 0);
lean_inc_ref(v_str_4131_);
v_startInclusive_4132_ = lean_ctor_get(v___x_4058_, 1);
lean_inc(v_startInclusive_4132_);
v_endExclusive_4133_ = lean_ctor_get(v___x_4058_, 2);
lean_inc(v_endExclusive_4133_);
lean_dec_ref(v___x_4058_);
v___x_4134_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__1, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__1_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__1);
v___x_4135_ = lean_string_utf8_extract_fast(v_str_4131_, v_startInclusive_4132_, v_endExclusive_4133_);
lean_dec(v_endExclusive_4133_);
lean_dec(v_startInclusive_4132_);
lean_dec_ref(v_str_4131_);
v___x_4136_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_4136_, 0, v___x_4135_);
v___x_4137_ = l_Lean_MessageData_ofFormat(v___x_4136_);
v___x_4138_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4138_, 0, v___x_4134_);
lean_ctor_set(v___x_4138_, 1, v___x_4137_);
v___x_4139_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__3, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__3_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__3);
v___x_4140_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4140_, 0, v___x_4138_);
lean_ctor_set(v___x_4140_, 1, v___x_4139_);
switch(lean_obj_tag(v_stained_3893_))
{
case 0:
{
lean_object* v_a_4141_; lean_object* v___x_4142_; 
v_a_4141_ = lean_ctor_get(v_stained_3893_, 0);
lean_inc(v_a_4141_);
v___x_4142_ = l_Lean_Name_toString(v_a_4141_, v___y_3876_);
v___y_4034_ = v___x_4140_;
v___y_4035_ = v_a_4055_;
v___y_4036_ = v___x_4142_;
goto v___jp_4033_;
}
case 1:
{
lean_object* v___x_4143_; 
v___x_4143_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instToStringStained___lam__0___closed__0));
v___y_4034_ = v___x_4140_;
v___y_4035_ = v_a_4055_;
v___y_4036_ = v___x_4143_;
goto v___jp_4033_;
}
default: 
{
lean_object* v___x_4144_; 
v___x_4144_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instToStringStained___lam__0___closed__1));
v___y_4034_ = v___x_4140_;
v___y_4035_ = v_a_4055_;
v___y_4036_ = v___x_4144_;
goto v___jp_4033_;
}
}
}
}
else
{
lean_dec_ref(v_str_4064_);
if (lean_obj_tag(v_stained_3893_) == 2)
{
lean_object* v_str_4145_; lean_object* v_startInclusive_4146_; lean_object* v_endExclusive_4147_; lean_object* v___x_4148_; lean_object* v___x_4149_; lean_object* v___x_4150_; lean_object* v___x_4151_; lean_object* v___x_4152_; lean_object* v___x_4153_; lean_object* v___x_4154_; 
v_str_4145_ = lean_ctor_get(v___x_4058_, 0);
lean_inc_ref(v_str_4145_);
v_startInclusive_4146_ = lean_ctor_get(v___x_4058_, 1);
lean_inc(v_startInclusive_4146_);
v_endExclusive_4147_ = lean_ctor_get(v___x_4058_, 2);
lean_inc(v_endExclusive_4147_);
lean_dec_ref(v___x_4058_);
v___x_4148_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__1, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__1_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__1);
v___x_4149_ = lean_string_utf8_extract_fast(v_str_4145_, v_startInclusive_4146_, v_endExclusive_4147_);
lean_dec(v_endExclusive_4147_);
lean_dec(v_startInclusive_4146_);
lean_dec_ref(v_str_4145_);
v___x_4150_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_4150_, 0, v___x_4149_);
v___x_4151_ = l_Lean_MessageData_ofFormat(v___x_4150_);
v___x_4152_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4152_, 0, v___x_4148_);
lean_ctor_set(v___x_4152_, 1, v___x_4151_);
v___x_4153_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__33, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__33_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___closed__33);
v___x_4154_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4154_, 0, v___x_4152_);
lean_ctor_set(v___x_4154_, 1, v___x_4153_);
v___y_4003_ = v_a_4055_;
v___y_4004_ = v___x_4154_;
goto v___jp_4002_;
}
else
{
lean_object* v_str_4155_; lean_object* v_startInclusive_4156_; lean_object* v_endExclusive_4157_; lean_object* v___x_4158_; lean_object* v___x_4159_; lean_object* v___x_4160_; lean_object* v___x_4161_; lean_object* v___x_4162_; lean_object* v___x_4163_; lean_object* v___x_4164_; 
v_str_4155_ = lean_ctor_get(v___x_4058_, 0);
lean_inc_ref(v_str_4155_);
v_startInclusive_4156_ = lean_ctor_get(v___x_4058_, 1);
lean_inc(v_startInclusive_4156_);
v_endExclusive_4157_ = lean_ctor_get(v___x_4058_, 2);
lean_inc(v_endExclusive_4157_);
lean_dec_ref(v___x_4058_);
v___x_4158_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__1, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__1_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__1);
v___x_4159_ = lean_string_utf8_extract_fast(v_str_4155_, v_startInclusive_4156_, v_endExclusive_4157_);
lean_dec(v_endExclusive_4157_);
lean_dec(v_startInclusive_4156_);
lean_dec_ref(v_str_4155_);
v___x_4160_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_4160_, 0, v___x_4159_);
v___x_4161_ = l_Lean_MessageData_ofFormat(v___x_4160_);
v___x_4162_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4162_, 0, v___x_4158_);
lean_ctor_set(v___x_4162_, 1, v___x_4161_);
v___x_4163_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__3, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__3_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0___closed__3);
v___x_4164_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4164_, 0, v___x_4162_);
lean_ctor_set(v___x_4164_, 1, v___x_4163_);
switch(lean_obj_tag(v_stained_3893_))
{
case 0:
{
lean_object* v_a_4165_; lean_object* v___x_4166_; 
v_a_4165_ = lean_ctor_get(v_stained_3893_, 0);
lean_inc(v_a_4165_);
v___x_4166_ = l_Lean_Name_toString(v_a_4165_, v___y_3876_);
v___y_4043_ = v___x_4164_;
v___y_4044_ = v_a_4055_;
v___y_4045_ = v___x_4166_;
goto v___jp_4042_;
}
case 1:
{
lean_object* v___x_4167_; 
v___x_4167_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instToStringStained___lam__0___closed__0));
v___y_4043_ = v___x_4164_;
v___y_4044_ = v_a_4055_;
v___y_4045_ = v___x_4167_;
goto v___jp_4042_;
}
default: 
{
lean_object* v___x_4168_; 
v___x_4168_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_instToStringStained___lam__0___closed__1));
v___y_4043_ = v___x_4164_;
v___y_4044_ = v_a_4055_;
v___y_4045_ = v___x_4168_;
goto v___jp_4042_;
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
lean_object* v___x_4169_; 
lean_dec(v_pre_4063_);
lean_dec_ref_known(v_pre_4062_, 2);
lean_dec_ref_known(v_pre_4061_, 2);
lean_dec_ref_known(v_pre_4060_, 2);
lean_inc(v_stained_3893_);
v___x_4169_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0(v___x_4058_, v_stained_3893_, v___y_3876_, v___x_4059_);
lean_dec_ref_known(v___x_4059_, 2);
lean_dec_ref(v___x_4058_);
v___y_4003_ = v_a_4055_;
v___y_4004_ = v___x_4169_;
goto v___jp_4002_;
}
}
else
{
lean_object* v___x_4170_; 
lean_dec_ref_known(v_pre_4061_, 2);
lean_dec(v_pre_4062_);
lean_dec_ref_known(v_pre_4060_, 2);
lean_inc(v_stained_3893_);
v___x_4170_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0(v___x_4058_, v_stained_3893_, v___y_3876_, v___x_4059_);
lean_dec_ref_known(v___x_4059_, 2);
lean_dec_ref(v___x_4058_);
v___y_4003_ = v_a_4055_;
v___y_4004_ = v___x_4170_;
goto v___jp_4002_;
}
}
else
{
lean_object* v___x_4171_; 
lean_dec_ref_known(v_pre_4060_, 2);
lean_dec(v_pre_4061_);
lean_inc(v_stained_3893_);
v___x_4171_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0(v___x_4058_, v_stained_3893_, v___y_3876_, v___x_4059_);
lean_dec_ref_known(v___x_4059_, 2);
lean_dec_ref(v___x_4058_);
v___y_4003_ = v_a_4055_;
v___y_4004_ = v___x_4171_;
goto v___jp_4002_;
}
}
else
{
lean_object* v___x_4172_; 
lean_dec(v_pre_4060_);
lean_inc(v_stained_3893_);
v___x_4172_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0(v___x_4058_, v_stained_3893_, v___y_3876_, v___x_4059_);
lean_dec_ref_known(v___x_4059_, 2);
lean_dec_ref(v___x_4058_);
v___y_4003_ = v_a_4055_;
v___y_4004_ = v___x_4172_;
goto v___jp_4002_;
}
}
else
{
lean_object* v___x_4173_; 
lean_inc(v_stained_3893_);
v___x_4173_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___lam__0(v___x_4058_, v_stained_3893_, v___y_3876_, v___x_4059_);
lean_dec(v___x_4059_);
lean_dec_ref(v___x_4058_);
v___y_4003_ = v_a_4055_;
v___y_4004_ = v___x_4173_;
goto v___jp_4002_;
}
}
else
{
lean_object* v_a_4174_; lean_object* v___x_4176_; uint8_t v_isShared_4177_; uint8_t v_isSharedCheck_4181_; 
lean_dec_ref(v___y_4052_);
lean_dec(v_stx_3894_);
lean_dec(v_stained_3893_);
lean_del_object(v___x_3891_);
lean_dec(v_fst_3889_);
v_a_4174_ = lean_ctor_get(v___x_4054_, 0);
v_isSharedCheck_4181_ = !lean_is_exclusive(v___x_4054_);
if (v_isSharedCheck_4181_ == 0)
{
v___x_4176_ = v___x_4054_;
v_isShared_4177_ = v_isSharedCheck_4181_;
goto v_resetjp_4175_;
}
else
{
lean_inc(v_a_4174_);
lean_dec(v___x_4054_);
v___x_4176_ = lean_box(0);
v_isShared_4177_ = v_isSharedCheck_4181_;
goto v_resetjp_4175_;
}
v_resetjp_4175_:
{
lean_object* v___x_4179_; 
if (v_isShared_4177_ == 0)
{
v___x_4179_ = v___x_4176_;
goto v_reusejp_4178_;
}
else
{
lean_object* v_reuseFailAlloc_4180_; 
v_reuseFailAlloc_4180_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4180_, 0, v_a_4174_);
v___x_4179_ = v_reuseFailAlloc_4180_;
goto v_reusejp_4178_;
}
v_reusejp_4178_:
{
return v___x_4179_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24___boxed(lean_object* v___y_4189_, lean_object* v___x_4190_, lean_object* v_as_4191_, lean_object* v_sz_4192_, lean_object* v_i_4193_, lean_object* v_b_4194_, lean_object* v___y_4195_, lean_object* v___y_4196_, lean_object* v___y_4197_){
_start:
{
uint8_t v___y_36265__boxed_4198_; uint8_t v___x_36266__boxed_4199_; size_t v_sz_boxed_4200_; size_t v_i_boxed_4201_; lean_object* v_res_4202_; 
v___y_36265__boxed_4198_ = lean_unbox(v___y_4189_);
v___x_36266__boxed_4199_ = lean_unbox(v___x_4190_);
v_sz_boxed_4200_ = lean_unbox_usize(v_sz_4192_);
lean_dec(v_sz_4192_);
v_i_boxed_4201_ = lean_unbox_usize(v_i_4193_);
lean_dec(v_i_4193_);
v_res_4202_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24(v___y_36265__boxed_4198_, v___x_36266__boxed_4199_, v_as_4191_, v_sz_boxed_4200_, v_i_boxed_4201_, v_b_4194_, v___y_4195_, v___y_4196_);
lean_dec(v___y_4196_);
lean_dec_ref(v___y_4195_);
lean_dec_ref(v_as_4191_);
return v_res_4202_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__18_spec__21___redArg(lean_object* v_o_4203_, lean_object* v___y_4204_){
_start:
{
lean_object* v___x_4206_; lean_object* v_env_4207_; lean_object* v___x_4208_; lean_object* v_toEnvExtension_4209_; lean_object* v_asyncMode_4210_; lean_object* v___x_4211_; lean_object* v___x_4212_; lean_object* v___x_4213_; lean_object* v_merged_4214_; lean_object* v___x_4216_; uint8_t v_isShared_4217_; uint8_t v_isSharedCheck_4222_; 
v___x_4206_ = lean_st_ref_get(v___y_4204_);
v_env_4207_ = lean_ctor_get(v___x_4206_, 0);
lean_inc_ref(v_env_4207_);
lean_dec(v___x_4206_);
v___x_4208_ = l_Lean_Linter_linterSetsExt;
v_toEnvExtension_4209_ = lean_ctor_get(v___x_4208_, 0);
v_asyncMode_4210_ = lean_ctor_get(v_toEnvExtension_4209_, 2);
v___x_4211_ = l_Lean_Linter_instInhabitedLinterSetsState_default;
v___x_4212_ = lean_box(0);
v___x_4213_ = l_Lean_PersistentEnvExtension_getState___redArg(v___x_4211_, v___x_4208_, v_env_4207_, v_asyncMode_4210_, v___x_4212_);
v_merged_4214_ = lean_ctor_get(v___x_4213_, 0);
v_isSharedCheck_4222_ = !lean_is_exclusive(v___x_4213_);
if (v_isSharedCheck_4222_ == 0)
{
lean_object* v_unused_4223_; 
v_unused_4223_ = lean_ctor_get(v___x_4213_, 1);
lean_dec(v_unused_4223_);
v___x_4216_ = v___x_4213_;
v_isShared_4217_ = v_isSharedCheck_4222_;
goto v_resetjp_4215_;
}
else
{
lean_inc(v_merged_4214_);
lean_dec(v___x_4213_);
v___x_4216_ = lean_box(0);
v_isShared_4217_ = v_isSharedCheck_4222_;
goto v_resetjp_4215_;
}
v_resetjp_4215_:
{
lean_object* v___x_4219_; 
if (v_isShared_4217_ == 0)
{
lean_ctor_set(v___x_4216_, 1, v_merged_4214_);
lean_ctor_set(v___x_4216_, 0, v_o_4203_);
v___x_4219_ = v___x_4216_;
goto v_reusejp_4218_;
}
else
{
lean_object* v_reuseFailAlloc_4221_; 
v_reuseFailAlloc_4221_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4221_, 0, v_o_4203_);
lean_ctor_set(v_reuseFailAlloc_4221_, 1, v_merged_4214_);
v___x_4219_ = v_reuseFailAlloc_4221_;
goto v_reusejp_4218_;
}
v_reusejp_4218_:
{
lean_object* v___x_4220_; 
v___x_4220_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4220_, 0, v___x_4219_);
return v___x_4220_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__18_spec__21___redArg___boxed(lean_object* v_o_4224_, lean_object* v___y_4225_, lean_object* v___y_4226_){
_start:
{
lean_object* v_res_4227_; 
v_res_4227_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__18_spec__21___redArg(v_o_4224_, v___y_4225_);
lean_dec(v___y_4225_);
return v_res_4227_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__18(lean_object* v___y_4228_, lean_object* v___y_4229_){
_start:
{
lean_object* v___x_4231_; lean_object* v_scopes_4232_; lean_object* v___x_4233_; lean_object* v___x_4234_; lean_object* v_opts_4235_; lean_object* v___x_4236_; 
v___x_4231_ = lean_st_ref_get(v___y_4229_);
v_scopes_4232_ = lean_ctor_get(v___x_4231_, 2);
lean_inc(v_scopes_4232_);
lean_dec(v___x_4231_);
v___x_4233_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_4234_ = l_List_head_x21___redArg(v___x_4233_, v_scopes_4232_);
lean_dec(v_scopes_4232_);
v_opts_4235_ = lean_ctor_get(v___x_4234_, 1);
lean_inc_ref(v_opts_4235_);
lean_dec(v___x_4234_);
v___x_4236_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__18_spec__21___redArg(v_opts_4235_, v___y_4229_);
return v___x_4236_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__18___boxed(lean_object* v___y_4237_, lean_object* v___y_4238_, lean_object* v___y_4239_){
_start:
{
lean_object* v_res_4240_; 
v_res_4240_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__18(v___y_4237_, v___y_4238_);
lean_dec(v___y_4238_);
lean_dec_ref(v___y_4237_);
return v_res_4240_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___lam__0(lean_object* v___stx_4243_, lean_object* v___y_4244_, lean_object* v___y_4245_){
_start:
{
lean_object* v___x_4247_; lean_object* v_a_4248_; lean_object* v___x_4250_; uint8_t v_isShared_4251_; uint8_t v_isSharedCheck_4300_; 
v___x_4247_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__18(v___y_4244_, v___y_4245_);
v_a_4248_ = lean_ctor_get(v___x_4247_, 0);
v_isSharedCheck_4300_ = !lean_is_exclusive(v___x_4247_);
if (v_isSharedCheck_4300_ == 0)
{
v___x_4250_ = v___x_4247_;
v_isShared_4251_ = v_isSharedCheck_4300_;
goto v_resetjp_4249_;
}
else
{
lean_inc(v_a_4248_);
lean_dec(v___x_4247_);
v___x_4250_ = lean_box(0);
v_isShared_4251_ = v_isSharedCheck_4300_;
goto v_resetjp_4249_;
}
v_resetjp_4249_:
{
lean_object* v___x_4252_; uint8_t v___y_4254_; lean_object* v___x_4296_; uint8_t v___x_4297_; 
v___x_4252_ = lean_st_ref_get(v___y_4245_);
v___x_4296_ = lp_mathlib_Mathlib_Linter_linter_flexible;
v___x_4297_ = l_Lean_Linter_getLinterValue(v___x_4296_, v_a_4248_);
lean_dec(v_a_4248_);
if (v___x_4297_ == 0)
{
lean_dec(v___x_4252_);
v___y_4254_ = v___x_4297_;
goto v___jp_4253_;
}
else
{
lean_object* v_infoState_4298_; uint8_t v_enabled_4299_; 
v_infoState_4298_ = lean_ctor_get(v___x_4252_, 8);
lean_inc_ref(v_infoState_4298_);
lean_dec(v___x_4252_);
v_enabled_4299_ = lean_ctor_get_uint8(v_infoState_4298_, sizeof(void*)*3);
lean_dec_ref(v_infoState_4298_);
v___y_4254_ = v_enabled_4299_;
goto v___jp_4253_;
}
v___jp_4253_:
{
if (v___y_4254_ == 0)
{
lean_object* v___x_4255_; lean_object* v___x_4257_; 
v___x_4255_ = lean_box(0);
if (v_isShared_4251_ == 0)
{
lean_ctor_set(v___x_4250_, 0, v___x_4255_);
v___x_4257_ = v___x_4250_;
goto v_reusejp_4256_;
}
else
{
lean_object* v_reuseFailAlloc_4258_; 
v_reuseFailAlloc_4258_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4258_, 0, v___x_4255_);
v___x_4257_ = v_reuseFailAlloc_4258_;
goto v_reusejp_4256_;
}
v_reusejp_4256_:
{
return v___x_4257_;
}
}
else
{
lean_object* v___x_4259_; lean_object* v_messages_4260_; uint8_t v___x_4261_; 
v___x_4259_ = lean_st_ref_get(v___y_4245_);
v_messages_4260_ = lean_ctor_get(v___x_4259_, 1);
lean_inc_ref(v_messages_4260_);
lean_dec(v___x_4259_);
v___x_4261_ = l_Lean_MessageLog_hasErrors(v_messages_4260_);
lean_dec_ref(v_messages_4260_);
if (v___x_4261_ == 0)
{
lean_object* v___x_4262_; lean_object* v_a_4263_; lean_object* v___x_4264_; lean_object* v___x_4265_; lean_object* v___x_4266_; lean_object* v___x_4267_; size_t v_sz_4268_; size_t v___x_4269_; lean_object* v___x_4270_; 
lean_del_object(v___x_4250_);
v___x_4262_ = lp_mathlib_Lean_Elab_getInfoTrees___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__19___redArg(v___y_4245_);
v_a_4263_ = lean_ctor_get(v___x_4262_, 0);
lean_inc(v_a_4263_);
lean_dec_ref(v___x_4262_);
v___x_4264_ = lean_unsigned_to_nat(0u);
v___x_4265_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_extractTacticData___closed__1));
v___x_4266_ = lp_mathlib_Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__20(v_a_4263_, v___x_4265_, v___x_4264_);
lean_dec(v_a_4263_);
v___x_4267_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___lam__0___closed__0));
v_sz_4268_ = lean_array_size(v___x_4266_);
v___x_4269_ = ((size_t)0ULL);
v___x_4270_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__21(v___x_4266_, v_sz_4268_, v___x_4269_, v___x_4267_, v___y_4244_, v___y_4245_);
lean_dec_ref(v___x_4266_);
if (lean_obj_tag(v___x_4270_) == 0)
{
lean_object* v_a_4271_; lean_object* v_snd_4272_; lean_object* v___x_4273_; size_t v_sz_4274_; lean_object* v___x_4275_; 
v_a_4271_ = lean_ctor_get(v___x_4270_, 0);
lean_inc(v_a_4271_);
lean_dec_ref_known(v___x_4270_, 1);
v_snd_4272_ = lean_ctor_get(v_a_4271_, 1);
lean_inc(v_snd_4272_);
lean_dec(v_a_4271_);
v___x_4273_ = lean_box(0);
v_sz_4274_ = lean_array_size(v_snd_4272_);
v___x_4275_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__24(v___y_4254_, v___x_4261_, v_snd_4272_, v_sz_4274_, v___x_4269_, v___x_4273_, v___y_4244_, v___y_4245_);
lean_dec(v_snd_4272_);
if (lean_obj_tag(v___x_4275_) == 0)
{
lean_object* v___x_4277_; uint8_t v_isShared_4278_; uint8_t v_isSharedCheck_4282_; 
v_isSharedCheck_4282_ = !lean_is_exclusive(v___x_4275_);
if (v_isSharedCheck_4282_ == 0)
{
lean_object* v_unused_4283_; 
v_unused_4283_ = lean_ctor_get(v___x_4275_, 0);
lean_dec(v_unused_4283_);
v___x_4277_ = v___x_4275_;
v_isShared_4278_ = v_isSharedCheck_4282_;
goto v_resetjp_4276_;
}
else
{
lean_dec(v___x_4275_);
v___x_4277_ = lean_box(0);
v_isShared_4278_ = v_isSharedCheck_4282_;
goto v_resetjp_4276_;
}
v_resetjp_4276_:
{
lean_object* v___x_4280_; 
if (v_isShared_4278_ == 0)
{
lean_ctor_set(v___x_4277_, 0, v___x_4273_);
v___x_4280_ = v___x_4277_;
goto v_reusejp_4279_;
}
else
{
lean_object* v_reuseFailAlloc_4281_; 
v_reuseFailAlloc_4281_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4281_, 0, v___x_4273_);
v___x_4280_ = v_reuseFailAlloc_4281_;
goto v_reusejp_4279_;
}
v_reusejp_4279_:
{
return v___x_4280_;
}
}
}
else
{
return v___x_4275_;
}
}
else
{
lean_object* v_a_4284_; lean_object* v___x_4286_; uint8_t v_isShared_4287_; uint8_t v_isSharedCheck_4291_; 
v_a_4284_ = lean_ctor_get(v___x_4270_, 0);
v_isSharedCheck_4291_ = !lean_is_exclusive(v___x_4270_);
if (v_isSharedCheck_4291_ == 0)
{
v___x_4286_ = v___x_4270_;
v_isShared_4287_ = v_isSharedCheck_4291_;
goto v_resetjp_4285_;
}
else
{
lean_inc(v_a_4284_);
lean_dec(v___x_4270_);
v___x_4286_ = lean_box(0);
v_isShared_4287_ = v_isSharedCheck_4291_;
goto v_resetjp_4285_;
}
v_resetjp_4285_:
{
lean_object* v___x_4289_; 
if (v_isShared_4287_ == 0)
{
v___x_4289_ = v___x_4286_;
goto v_reusejp_4288_;
}
else
{
lean_object* v_reuseFailAlloc_4290_; 
v_reuseFailAlloc_4290_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4290_, 0, v_a_4284_);
v___x_4289_ = v_reuseFailAlloc_4290_;
goto v_reusejp_4288_;
}
v_reusejp_4288_:
{
return v___x_4289_;
}
}
}
}
else
{
lean_object* v___x_4292_; lean_object* v___x_4294_; 
v___x_4292_ = lean_box(0);
if (v_isShared_4251_ == 0)
{
lean_ctor_set(v___x_4250_, 0, v___x_4292_);
v___x_4294_ = v___x_4250_;
goto v_reusejp_4293_;
}
else
{
lean_object* v_reuseFailAlloc_4295_; 
v_reuseFailAlloc_4295_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4295_, 0, v___x_4292_);
v___x_4294_ = v_reuseFailAlloc_4295_;
goto v_reusejp_4293_;
}
v_reusejp_4293_:
{
return v___x_4294_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___lam__0___boxed(lean_object* v___stx_4301_, lean_object* v___y_4302_, lean_object* v___y_4303_, lean_object* v___y_4304_){
_start:
{
lean_object* v_res_4305_; 
v_res_4305_ = lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter___lam__0(v___stx_4301_, v___y_4302_, v___y_4303_);
lean_dec(v___y_4303_);
lean_dec_ref(v___y_4302_);
lean_dec(v___stx_4301_);
return v_res_4305_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__0(lean_object* v_00_u03b2_4347_, lean_object* v_m_4348_, lean_object* v_a_4349_){
_start:
{
uint8_t v___x_4350_; 
v___x_4350_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__0___redArg(v_m_4348_, v_a_4349_);
return v___x_4350_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__0___boxed(lean_object* v_00_u03b2_4351_, lean_object* v_m_4352_, lean_object* v_a_4353_){
_start:
{
uint8_t v_res_4354_; lean_object* v_r_4355_; 
v_res_4354_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__0(v_00_u03b2_4351_, v_m_4352_, v_a_4353_);
lean_dec(v_a_4353_);
lean_dec_ref(v_m_4352_);
v_r_4355_ = lean_box(v_res_4354_);
return v_r_4355_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__1(lean_object* v_snd_4356_, lean_object* v_as_4357_, size_t v_sz_4358_, size_t v_i_4359_, lean_object* v_b_4360_, lean_object* v___y_4361_, lean_object* v___y_4362_){
_start:
{
lean_object* v___x_4364_; 
v___x_4364_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__1___redArg(v_snd_4356_, v_as_4357_, v_sz_4358_, v_i_4359_, v_b_4360_);
return v___x_4364_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__1___boxed(lean_object* v_snd_4365_, lean_object* v_as_4366_, lean_object* v_sz_4367_, lean_object* v_i_4368_, lean_object* v_b_4369_, lean_object* v___y_4370_, lean_object* v___y_4371_, lean_object* v___y_4372_){
_start:
{
size_t v_sz_boxed_4373_; size_t v_i_boxed_4374_; lean_object* v_res_4375_; 
v_sz_boxed_4373_ = lean_unbox_usize(v_sz_4367_);
lean_dec(v_sz_4367_);
v_i_boxed_4374_ = lean_unbox_usize(v_i_4368_);
lean_dec(v_i_4368_);
v_res_4375_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__1(v_snd_4365_, v_as_4366_, v_sz_boxed_4373_, v_i_boxed_4374_, v_b_4369_, v___y_4370_, v___y_4371_);
lean_dec(v___y_4371_);
lean_dec_ref(v___y_4370_);
lean_dec_ref(v_as_4366_);
return v_res_4375_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Array_isEqvAux___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__3(lean_object* v_xs_4376_, lean_object* v_ys_4377_, lean_object* v_hsz_4378_, lean_object* v_x_4379_, lean_object* v_x_4380_){
_start:
{
uint8_t v___x_4381_; 
v___x_4381_ = lp_mathlib_Array_isEqvAux___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__3___redArg(v_xs_4376_, v_ys_4377_, v_x_4379_);
return v___x_4381_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_isEqvAux___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__3___boxed(lean_object* v_xs_4382_, lean_object* v_ys_4383_, lean_object* v_hsz_4384_, lean_object* v_x_4385_, lean_object* v_x_4386_){
_start:
{
uint8_t v_res_4387_; lean_object* v_r_4388_; 
v_res_4387_ = lp_mathlib_Array_isEqvAux___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__3(v_xs_4382_, v_ys_4383_, v_hsz_4384_, v_x_4385_, v_x_4386_);
lean_dec_ref(v_ys_4383_);
lean_dec_ref(v_xs_4382_);
v_r_4388_ = lean_box(v_res_4387_);
return v_r_4388_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__7(lean_object* v___x_4389_, lean_object* v___x_4390_, lean_object* v_a_4391_, lean_object* v_a_4392_, lean_object* v___y_4393_, lean_object* v___y_4394_){
_start:
{
lean_object* v___x_4396_; 
v___x_4396_ = lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__7___redArg(v___x_4389_, v___x_4390_, v_a_4391_, v_a_4392_);
return v___x_4396_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__7___boxed(lean_object* v___x_4397_, lean_object* v___x_4398_, lean_object* v_a_4399_, lean_object* v_a_4400_, lean_object* v___y_4401_, lean_object* v___y_4402_, lean_object* v___y_4403_){
_start:
{
lean_object* v_res_4404_; 
v_res_4404_ = lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__7(v___x_4397_, v___x_4398_, v_a_4399_, v_a_4400_, v___y_4401_, v___y_4402_);
lean_dec(v___y_4402_);
lean_dec_ref(v___y_4401_);
lean_dec_ref(v___x_4397_);
return v_res_4404_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__8(lean_object* v_00_u03b2_4405_, lean_object* v_m_4406_, lean_object* v_a_4407_){
_start:
{
uint8_t v___x_4408_; 
v___x_4408_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__8___redArg(v_m_4406_, v_a_4407_);
return v___x_4408_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__8___boxed(lean_object* v_00_u03b2_4409_, lean_object* v_m_4410_, lean_object* v_a_4411_){
_start:
{
uint8_t v_res_4412_; lean_object* v_r_4413_; 
v_res_4412_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__8(v_00_u03b2_4409_, v_m_4410_, v_a_4411_);
lean_dec_ref(v_a_4411_);
lean_dec_ref(v_m_4410_);
v_r_4413_ = lean_box(v_res_4412_);
return v_r_4413_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__9(lean_object* v_00_u03b2_4414_, lean_object* v_m_4415_, lean_object* v_a_4416_, lean_object* v_b_4417_){
_start:
{
lean_object* v___x_4418_; 
v___x_4418_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__9___redArg(v_m_4415_, v_a_4416_, v_b_4417_);
return v___x_4418_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__10(lean_object* v_as_4419_, size_t v_sz_4420_, size_t v_i_4421_, lean_object* v_b_4422_, lean_object* v___y_4423_, lean_object* v___y_4424_){
_start:
{
lean_object* v___x_4426_; 
v___x_4426_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__10___redArg(v_as_4419_, v_sz_4420_, v_i_4421_, v_b_4422_);
return v___x_4426_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__10___boxed(lean_object* v_as_4427_, lean_object* v_sz_4428_, lean_object* v_i_4429_, lean_object* v_b_4430_, lean_object* v___y_4431_, lean_object* v___y_4432_, lean_object* v___y_4433_){
_start:
{
size_t v_sz_boxed_4434_; size_t v_i_boxed_4435_; lean_object* v_res_4436_; 
v_sz_boxed_4434_ = lean_unbox_usize(v_sz_4428_);
lean_dec(v_sz_4428_);
v_i_boxed_4435_ = lean_unbox_usize(v_i_4429_);
lean_dec(v_i_4429_);
v_res_4436_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__10(v_as_4427_, v_sz_boxed_4434_, v_i_boxed_4435_, v_b_4430_, v___y_4431_, v___y_4432_);
lean_dec(v___y_4432_);
lean_dec_ref(v___y_4431_);
lean_dec_ref(v_as_4427_);
return v_res_4436_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__14(lean_object* v___y_4437_, lean_object* v___x_4438_, lean_object* v___x_4439_, lean_object* v___x_4440_, lean_object* v_as_4441_, lean_object* v_as_x27_4442_, lean_object* v_b_4443_, lean_object* v_a_4444_, lean_object* v___y_4445_, lean_object* v___y_4446_){
_start:
{
lean_object* v___x_4448_; 
v___x_4448_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__14___redArg(v___y_4437_, v___x_4438_, v___x_4439_, v___x_4440_, v_as_x27_4442_, v_b_4443_, v___y_4445_, v___y_4446_);
return v___x_4448_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__14___boxed(lean_object* v___y_4449_, lean_object* v___x_4450_, lean_object* v___x_4451_, lean_object* v___x_4452_, lean_object* v_as_4453_, lean_object* v_as_x27_4454_, lean_object* v_b_4455_, lean_object* v_a_4456_, lean_object* v___y_4457_, lean_object* v___y_4458_, lean_object* v___y_4459_){
_start:
{
lean_object* v_res_4460_; 
v_res_4460_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__14(v___y_4449_, v___x_4450_, v___x_4451_, v___x_4452_, v_as_4453_, v_as_x27_4454_, v_b_4455_, v_a_4456_, v___y_4457_, v___y_4458_);
lean_dec(v___y_4458_);
lean_dec_ref(v___y_4457_);
lean_dec(v_as_x27_4454_);
lean_dec(v_as_4453_);
lean_dec_ref(v___x_4452_);
lean_dec_ref(v___x_4450_);
lean_dec_ref(v___y_4449_);
return v_res_4460_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__15(lean_object* v_a_4461_, lean_object* v_a_4462_, lean_object* v___x_4463_, lean_object* v___x_4464_, lean_object* v___x_4465_, lean_object* v___x_4466_, lean_object* v_as_4467_, lean_object* v_as_x27_4468_, lean_object* v_b_4469_, lean_object* v_a_4470_, lean_object* v___y_4471_, lean_object* v___y_4472_){
_start:
{
lean_object* v___x_4474_; 
v___x_4474_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__15___redArg(v_a_4461_, v_a_4462_, v___x_4463_, v___x_4464_, v___x_4465_, v___x_4466_, v_as_x27_4468_, v_b_4469_);
return v___x_4474_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__15___boxed(lean_object* v_a_4475_, lean_object* v_a_4476_, lean_object* v___x_4477_, lean_object* v___x_4478_, lean_object* v___x_4479_, lean_object* v___x_4480_, lean_object* v_as_4481_, lean_object* v_as_x27_4482_, lean_object* v_b_4483_, lean_object* v_a_4484_, lean_object* v___y_4485_, lean_object* v___y_4486_, lean_object* v___y_4487_){
_start:
{
lean_object* v_res_4488_; 
v_res_4488_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__15(v_a_4475_, v_a_4476_, v___x_4477_, v___x_4478_, v___x_4479_, v___x_4480_, v_as_4481_, v_as_x27_4482_, v_b_4483_, v_a_4484_, v___y_4485_, v___y_4486_);
lean_dec(v___y_4486_);
lean_dec_ref(v___y_4485_);
lean_dec(v_as_x27_4482_);
lean_dec(v_as_4481_);
lean_dec_ref(v___x_4480_);
lean_dec_ref(v_a_4476_);
return v_res_4488_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__18_spec__21(lean_object* v_o_4489_, lean_object* v___y_4490_, lean_object* v___y_4491_){
_start:
{
lean_object* v___x_4493_; 
v___x_4493_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__18_spec__21___redArg(v_o_4489_, v___y_4491_);
return v___x_4493_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__18_spec__21___boxed(lean_object* v_o_4494_, lean_object* v___y_4495_, lean_object* v___y_4496_, lean_object* v___y_4497_){
_start:
{
lean_object* v_res_4498_; 
v_res_4498_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__18_spec__21(v_o_4494_, v___y_4495_, v___y_4496_);
lean_dec(v___y_4496_);
lean_dec_ref(v___y_4495_);
return v_res_4498_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__8_spec__8(lean_object* v_00_u03b2_4499_, lean_object* v_a_4500_, lean_object* v_x_4501_){
_start:
{
uint8_t v___x_4502_; 
v___x_4502_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__8_spec__8___redArg(v_a_4500_, v_x_4501_);
return v___x_4502_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__8_spec__8___boxed(lean_object* v_00_u03b2_4503_, lean_object* v_a_4504_, lean_object* v_x_4505_){
_start:
{
uint8_t v_res_4506_; lean_object* v_r_4507_; 
v_res_4506_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__8_spec__8(v_00_u03b2_4503_, v_a_4504_, v_x_4505_);
lean_dec(v_x_4505_);
lean_dec_ref(v_a_4504_);
v_r_4507_ = lean_box(v_res_4506_);
return v_r_4507_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__9_spec__10(lean_object* v_00_u03b2_4508_, lean_object* v_data_4509_){
_start:
{
lean_object* v___x_4510_; 
v___x_4510_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__9_spec__10___redArg(v_data_4509_);
return v___x_4510_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29_spec__32(lean_object* v_msgData_4511_, lean_object* v___y_4512_, lean_object* v___y_4513_){
_start:
{
lean_object* v___x_4515_; 
v___x_4515_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29_spec__32___redArg(v_msgData_4511_, v___y_4513_);
return v___x_4515_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29_spec__32___boxed(lean_object* v_msgData_4516_, lean_object* v___y_4517_, lean_object* v___y_4518_, lean_object* v___y_4519_){
_start:
{
lean_object* v_res_4520_; 
v_res_4520_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__22_spec__29_spec__32(v_msgData_4516_, v___y_4517_, v___y_4518_);
lean_dec(v___y_4518_);
lean_dec_ref(v___y_4517_);
return v_res_4520_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__9_spec__10_spec__12(lean_object* v_00_u03b2_4521_, lean_object* v_i_4522_, lean_object* v_source_4523_, lean_object* v_target_4524_){
_start:
{
lean_object* v___x_4525_; 
v___x_4525_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__9_spec__10_spec__12___redArg(v_i_4522_, v_source_4523_, v_target_4524_);
return v___x_4525_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__9_spec__10_spec__12_spec__28(lean_object* v_00_u03b2_4526_, lean_object* v_x_4527_, lean_object* v_x_4528_){
_start:
{
lean_object* v___x_4529_; 
v___x_4529_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter_spec__9_spec__10_spec__12_spec__28___redArg(v_x_4527_, v_x_4528_);
return v___x_4529_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_initFn_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_242819373____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_4531_; lean_object* v___x_4532_; 
v___x_4531_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexibleLinter));
v___x_4532_ = l_Lean_Elab_Command_addLinter(v___x_4531_);
return v___x_4532_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_initFn_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_242819373____hygCtx___hyg_2____boxed(lean_object* v_a_4533_){
_start:
{
lean_object* v_res_4534_; 
v_res_4534_ = lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_initFn_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_242819373____hygCtx___hyg_2_();
return v_res_4534_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Parser_Term(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_FlexibleLinter(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Parser_Term(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Command(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Tactic_Simp(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_TryThis(uint8_t builtin);
lean_object* runtime_initialize_Lean_Server_InfoUtils(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_Header(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Linter_FlexibleLinter(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Command(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_Simp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_TryThis(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Server_InfoUtils(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_Header(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_1549988361____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Linter_linter_flexible = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Linter_linter_flexible);
lean_dec_ref(res);
lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers = _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_stoppers);
lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible = _init_lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_flexible);
res = lp_mathlib___private_Mathlib_Tactic_Linter_FlexibleLinter_0__Mathlib_Linter_Flexible_initFn_00___x40_Mathlib_Tactic_Linter_FlexibleLinter_242819373____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Command(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_Simp(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_TryThis(uint8_t builtin);
lean_object* initialize_Lean_Server_InfoUtils(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_Header(uint8_t builtin);
lean_object* initialize_Lean_Parser_Term(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Linter_FlexibleLinter(uint8_t builtin) {
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
res = initialize_Lean_Elab_Tactic_Simp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_TryThis(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Server_InfoUtils(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linter_Header(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Parser_Term(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_FlexibleLinter(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Linter_FlexibleLinter(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Linter_FlexibleLinter(builtin);
}
#ifdef __cplusplus
}
#endif
