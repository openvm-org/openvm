// Lean compiler output
// Module: Mathlib.Tactic.Linter.UnusedTactic
// Imports: public import Init public meta import Init public meta import Lean.Server.InfoUtils public meta import Mathlib.Tactic.Linter.Header public import Batteries.Tactic.Unreachable public import Lean.Parser.Syntax public import Mathlib.Tactic.Linter.UnusedTacticExtension
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
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_instInhabitedPersistentArrayNode_default(lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
lean_object* lean_usize_to_nat(size_t);
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
size_t lean_usize_shift_left(size_t, size_t);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_Lean_Syntax_getRange_x3f(lean_object*, uint8_t);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getKind(lean_object*);
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
size_t lean_uint64_to_usize(uint64_t);
uint8_t lean_name_eq(lean_object*, lean_object*);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_Lean_PersistentArray_toList___redArg(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_List_foldl___at___00Array_appendList_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_LocalDecl_userName(lean_object*);
lean_object* l_Lean_Elab_InfoTree_foldInfo___redArg(lean_object*, lean_object*, lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* l_Lean_Name_beq___boxed(lean_object*, lean_object*);
uint8_t l_List_elem___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_register_option(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* l_Lean_MessageData_note(lean_object*);
extern lean_object* l_Lean_Linter_linterMessageTag;
lean_object* l_Lean_Elab_Command_getScope___redArg(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Lean_MessageLog_add(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
extern lean_object* l_Lean_Elab_Command_instInhabitedScope_default;
lean_object* l_List_head_x21___redArg(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* l_Lean_FileMap_toPosition(lean_object*, lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasTag(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getTailPos_x3f(lean_object*, uint8_t);
lean_object* l_Lean_Elab_Command_getRef___redArg(lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getPos_x3f(lean_object*, uint8_t);
uint8_t l_Lean_instBEqMessageSeverity_beq(uint8_t, uint8_t);
extern lean_object* l_Lean_warningAsError;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasSyntheticSorry(lean_object*);
uint8_t l_Lean_Syntax_instBEqRange_beq(lean_object*, lean_object*);
uint64_t l_Lean_Syntax_instHashableRange_hash(lean_object*);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Linter_linterSetsExt;
extern lean_object* l_Lean_Linter_instInhabitedLinterSetsState_default;
lean_object* l_Lean_PersistentEnvExtension_getState___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
lean_object* lean_array_fswap(lean_object*, lean_object*, lean_object*);
lean_object* l_instOrdNat___lam__0___boxed(lean_object*, lean_object*);
lean_object* l_instOrdInt___lam__0___boxed(lean_object*, lean_object*);
lean_object* lean_nat_to_int(lean_object*);
lean_object* lean_int_neg(lean_object*);
lean_object* l_lexOrd___redArg(lean_object*, lean_object*);
lean_object* lean_nat_shiftr(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_io_error_to_string(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
uint8_t l_Lean_NameHashSet_contains(lean_object*, lean_object*);
uint8_t l_Lean_MessageLog_hasErrors(lean_object*);
extern lean_object* l_Lean_Parser_parserExtension;
extern lean_object* l_Lean_Parser_ParserExtension_instInhabitedState_default;
lean_object* l_Lean_ScopedEnvExtension_getState___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
extern lean_object* lp_mathlib_Mathlib_Linter_UnusedTactic_allowedRef;
extern lean_object* lp_mathlib_Mathlib_Linter_UnusedTactic_allowedUnusedTacticExt;
lean_object* l_Lean_Name_hash___override___boxed(lean_object*);
lean_object* l_Std_HashSet_instInhabited(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_SimplePersistentEnvExtension_getState___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Linter_getLinterValue(lean_object*, lean_object*);
lean_object* l_instMonadEIO(lean_object*);
lean_object* l_StateRefT_x27_instMonad___redArg(lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_instBEqRange_beq___boxed(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_instHashableRange_hash___boxed(lean_object*);
lean_object* l_Std_DHashMap_Internal_Raw_u2080_insert___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
lean_object* l_Lean_NameHashSet_insert(lean_object*, lean_object*);
lean_object* l_Lean_withSetOptionIn___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_addLinter(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4__spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "unusedTactic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(186, 218, 113, 226, 101, 176, 32, 79)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(56, 148, 213, 113, 169, 72, 231, 196)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 32, .m_capacity = 32, .m_length = 31, .m_data = "enable the unused tactic linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(120, 131, 127, 204, 79, 169, 80, 92)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(101, 237, 90, 120, 51, 59, 46, 172)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(147, 103, 66, 148, 167, 234, 238, 192)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_linter_unusedTactic;
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__spec__0_spec__0_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__spec__0_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__3_spec__5___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__spec__0_spec__0_spec__2___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__spec__0_spec__1(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__spec__0___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Says"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "says"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(49, 164, 223, 43, 155, 124, 248, 66)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(143, 165, 67, 238, 10, 199, 6, 65)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "binderTactic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(17, 181, 78, 34, 190, 12, 180, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__9_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "dynamicQuot"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__9_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__9_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__10_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__10_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__10_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__10_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__10_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__10_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__10_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__9_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(116, 123, 139, 164, 173, 191, 116, 242)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__10_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__10_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__11_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "quotSeq"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__11_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__11_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__12_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__12_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__12_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__12_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__12_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__12_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__12_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__11_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(171, 67, 133, 150, 48, 85, 223, 184)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__12_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__12_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__13_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "tacticStop_"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__13_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__13_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__14_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__14_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__14_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__14_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__14_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__14_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__14_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__13_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(186, 187, 217, 116, 133, 153, 2, 108)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__14_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__14_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__15_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Command"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__15_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__15_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__16_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "notation"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__16_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__16_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__17_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__17_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__17_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__17_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__17_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__15_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__17_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__17_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__16_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(13, 34, 53, 7, 182, 20, 8, 182)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__17_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__17_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__18_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "mixfix"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__18_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__18_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__19_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__19_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__19_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__19_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__19_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__15_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__19_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__19_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__18_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(1, 31, 80, 86, 44, 46, 155, 0)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__19_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__19_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__20_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "discharger"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__20_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__20_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__21_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__21_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__21_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__21_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__21_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__21_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__21_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__20_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(233, 186, 255, 143, 150, 72, 152, 71)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__21_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__21_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__22_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "registerTryTactic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__22_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__22_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__23_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__23_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__23_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__23_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__23_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__15_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__23_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__23_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__22_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(64, 133, 180, 171, 152, 84, 222, 30)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__23_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__23_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__24_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Batteries"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__24_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__24_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__25_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "seq_focus"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__25_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__25_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__26_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__24_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__26_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__26_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__26_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__26_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__25_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(2, 25, 59, 65, 212, 252, 96, 203)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__26_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__26_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__27_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Hint"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__27_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__27_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__28_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "registerHintStx"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__28_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__28_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__29_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__29_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__29_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__29_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__29_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__27_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(144, 22, 88, 160, 25, 48, 219, 242)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__29_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__29_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__28_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(54, 47, 26, 100, 196, 8, 163, 146)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__29_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__29_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__30_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "LinearCombination"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__30_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__30_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__31_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "linearCombination"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__31_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__31_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__32_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__32_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__32_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__32_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__32_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__30_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(35, 149, 180, 189, 70, 21, 83, 76)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__32_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__32_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__31_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(157, 213, 151, 22, 157, 163, 3, 66)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__32_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__32_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__33_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "LinearCombinationPrime"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__33_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__33_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__34_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "linearCombination'"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__34_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__34_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__35_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__35_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__35_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__35_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__35_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__33_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(142, 199, 207, 73, 87, 45, 84, 242)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__35_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__35_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__34_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(239, 252, 8, 203, 209, 190, 216, 60)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__35_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__35_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__36_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Aesop"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__36_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__36_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__37_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Frontend"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__37_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__37_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__38_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "addRules"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__38_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__38_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__39_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__36_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__39_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__39_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__37_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(212, 61, 86, 210, 90, 71, 197, 0)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__39_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__39_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(13, 69, 189, 207, 58, 198, 186, 160)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__39_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__39_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__38_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(212, 211, 170, 103, 196, 177, 113, 83)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__39_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__39_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__40_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "aesopTactic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__40_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__40_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__41_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__36_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__41_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__41_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__37_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(212, 61, 86, 210, 90, 71, 197, 0)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__41_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__41_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(13, 69, 189, 207, 58, 198, 186, 160)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__41_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__41_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__40_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(54, 142, 162, 195, 161, 101, 248, 175)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__41_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__41_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__42_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "aesopTactic\?"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__42_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__42_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__43_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__36_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__43_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__43_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__37_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(212, 61, 86, 210, 90, 71, 197, 0)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__43_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__43_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(13, 69, 189, 207, 58, 198, 186, 160)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__43_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__43_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__42_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(24, 245, 87, 84, 72, 165, 203, 79)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__43_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__43_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__44_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "UnusedTactic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__44_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__44_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__45_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "command#show_kind_"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__45_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__45_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__46_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__46_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__46_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(120, 131, 127, 204, 79, 169, 80, 92)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__46_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__46_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__44_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(231, 84, 76, 169, 184, 93, 115, 177)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__46_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__46_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__45_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(95, 123, 92, 55, 201, 62, 152, 82)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__46_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__46_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__47_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "failIfSuccess"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__47_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__47_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__48_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__48_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__48_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__48_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__48_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__48_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__48_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__47_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(227, 159, 155, 237, 20, 68, 221, 246)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__48_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__48_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__49_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "successIfFailWithMsg"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__49_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__49_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__50_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__50_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__50_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__50_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__50_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__49_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(244, 58, 29, 249, 84, 193, 89, 5)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__50_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__50_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__51_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "failIfNoProgress"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__51_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__51_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__52_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__52_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__52_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__52_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__52_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__51_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(238, 120, 52, 11, 174, 48, 92, 172)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__52_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__52_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value;
static const lean_array_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__53_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*20, .m_other = 0, .m_tag = 246}, .m_size = 20, .m_capacity = 20, .m_data = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__10_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__12_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__14_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__17_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__19_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__21_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__23_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__26_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__29_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__32_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__35_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__39_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__41_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__43_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__46_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__48_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__50_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__52_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__53_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__53_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__54_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__54_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__55_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__55_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__56_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__56_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__spec__0_spec__0_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__3_spec__5(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_ignoreTacticKindsRef;
static const lean_string_object lp_mathlib_Mathlib_Linter_UnusedTactic_isIgnoreTacticKind___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "quot"};
static const lean_object* lp_mathlib_Mathlib_Linter_UnusedTactic_isIgnoreTacticKind___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Linter_UnusedTactic_isIgnoreTacticKind___closed__0_value;
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Linter_UnusedTactic_isIgnoreTacticKind(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_UnusedTactic_isIgnoreTacticKind___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_UnusedTactic_addIgnoreTacticKind(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_UnusedTactic_addIgnoreTacticKind___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___closed__0;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___closed__1;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Syntax_instBEqRange_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___closed__2_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Syntax_instHashableRange_hash___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_filterMapTR_go___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0_spec__1_spec__7_spec__9___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0_spec__1_spec__7_spec__9___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0_spec__1_spec__7___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0_spec__1_spec__7_spec__8___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0_spec__1_spec__7_spec__8___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0_spec__1_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0___redArg___lam__0, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0___redArg___boxed(lean_object*);
static const lean_array_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0_spec__1_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0_spec__1_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0_spec__1_spec__7_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0_spec__1_spec__7_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0_spec__1_spec__7_spec__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0_spec__1_spec__7_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__4_spec__8___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__4_spec__8___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__4_spec__7___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__4_spec__7___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__4___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__4___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__5(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_beq___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_beq___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__2___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_beq___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_beq___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__0___redArg___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__4___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSwap_var__,,"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__4___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__4___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__4___lam__0___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__4___lam__0___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__4___lam__0___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__4___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__4___lam__0___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__4___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(78, 115, 87, 127, 138, 34, 160, 62)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__4___lam__0___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__4___lam__0___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__4___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__4___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__4(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__5(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__3_spec__4(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__5___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__3___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__3___closed__0;
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__3(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__4___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__4_spec__7(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__4_spec__7___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__4_spec__8(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__4_spec__8___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getInfoTrees___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__3___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getInfoTrees___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__3___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getInfoTrees___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getInfoTrees___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__7___redArg___lam__0(lean_object*);
static const lean_closure_object lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__7_spec__11___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instOrdNat___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__7_spec__11___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__7_spec__11___redArg___closed__0_value;
static const lean_closure_object lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__7_spec__11___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instOrdInt___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__7_spec__11___redArg___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__7_spec__11___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__7_spec__11___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__7_spec__11___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__7___redArg___lam__1(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__7___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__7___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__7___redArg___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__7___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__7___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__7___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__8(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__8___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__9(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__10_spec__15_spec__19_spec__25_spec__30___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__10_spec__15_spec__19_spec__25___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__10_spec__15_spec__19___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__10_spec__15_spec__20___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__10_spec__15___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__5_spec__8_spec__11___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__5_spec__8_spec__11___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__5_spec__8___redArg(lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__5_spec__8___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__5___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__5___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__10(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__10_spec__16(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__10_spec__16___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__11_spec__18_spec__24___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__11_spec__18___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__11_spec__19(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__11_spec__20(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__11_spec__20___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__11(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__11___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__2_spec__3_spec__5___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__2_spec__3_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__2_spec__3___redArg(lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__2_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__2___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__12(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__13(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__13___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__1_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8___lam__0___closed__0_value;
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8___lam__0___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8_spec__18___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8_spec__18___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8_spec__18___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8_spec__18___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8_spec__18___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8_spec__18___redArg___closed__2;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8_spec__18___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8_spec__18___redArg___closed__3;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8_spec__18___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8_spec__18___redArg___closed__4;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8_spec__18___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8_spec__18___redArg___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8_spec__18___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8_spec__18___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8_spec__19(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8_spec__19___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 46, .m_capacity = 46, .m_length = 45, .m_data = "This linter can be disabled with `set_option "};
static const lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4___closed__0 = (const lean_object*)&lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4___closed__1;
static const lean_string_object lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = " false`"};
static const lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4___closed__2 = (const lean_object*)&lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Name_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___closed__0_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "unreachable"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__24_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___closed__2_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___closed__2_value_aux_1),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___closed__1_value),LEAN_SCALAR_PTR_LITERAL(138, 69, 16, 178, 93, 143, 143, 50)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___closed__2 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___closed__2_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "unreachableConv"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___closed__3 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__24_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___closed__4_value_aux_1),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___closed__3_value),LEAN_SCALAR_PTR_LITERAL(180, 51, 125, 100, 108, 230, 32, 33)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___closed__4 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___closed__5 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___closed__2_value),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___closed__5_value)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___closed__6 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___closed__6_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "Unused tactic linter: `"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___closed__7 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___closed__7_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___closed__8;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "` does nothing"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___closed__9 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___closed__9_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___closed__10;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___lam__0___closed__0;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___lam__0___closed__1;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "tactic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___lam__0___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___lam__0___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(99, 76, 33, 121, 85, 143, 17, 224)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___lam__0___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___lam__0___closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "conv"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___lam__0___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___lam__0___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___lam__0___closed__4_value),LEAN_SCALAR_PTR_LITERAL(232, 67, 39, 189, 45, 247, 54, 81)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___lam__0___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___lam__0___closed__5_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Name_hash___override___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___lam__0___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___lam__0___closed__6_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___lam__0___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___lam__0___closed__7;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___lam__0___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__0_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_withSetOptionIn___boxed, .m_arity = 6, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__0_value)} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__2_value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__4_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(191, 70, 156, 159, 11, 54, 216, 94)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__5_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(37, 204, 154, 235, 250, 222, 148, 114)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__6_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__44_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(142, 156, 112, 45, 32, 80, 172, 174)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(183, 187, 153, 51, 95, 156, 192, 20)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__8_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(122, 75, 8, 38, 140, 133, 173, 236)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__9_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(36, 215, 246, 175, 72, 146, 64, 26)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__10_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__44_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(35, 73, 128, 74, 228, 188, 16, 166)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__11_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "unusedTacticLinter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__12_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__11_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__12_value),LEAN_SCALAR_PTR_LITERAL(251, 135, 255, 227, 66, 191, 48, 134)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__13_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__13_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__14_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___closed__14_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__1_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__5(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__5___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__2_spec__3(lean_object*, lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__5_spec__8(lean_object*, lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__5_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__7_spec__11(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__7_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__10_spec__15(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__11_spec__18(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__2_spec__3_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__2_spec__3_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8_spec__18(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8_spec__18___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__5_spec__8_spec__11(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__5_spec__8_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__10_spec__15_spec__19(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__10_spec__15_spec__20(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__11_spec__18_spec__24(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__10_spec__15_spec__19_spec__25(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__10_spec__15_spec__19_spec__25_spec__30(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_300650473____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_300650473____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4__spec__0(lean_object* v_name_1_, lean_object* v_decl_2_, lean_object* v_ref_3_){
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4__spec__0___boxed(lean_object* v_name_29_, lean_object* v_decl_30_, lean_object* v_ref_31_, lean_object* v_a_32_){
_start:
{
lean_object* v_res_33_; 
v_res_33_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4__spec__0(v_name_29_, v_decl_30_, v_ref_31_);
lean_dec_ref(v_decl_30_);
return v_res_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_53_; lean_object* v___x_54_; lean_object* v___x_55_; lean_object* v___x_56_; 
v___x_53_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4_));
v___x_54_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4_));
v___x_55_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4_));
v___x_56_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4__spec__0(v___x_53_, v___x_54_, v___x_55_);
return v___x_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4____boxed(lean_object* v_a_57_){
_start:
{
lean_object* v_res_58_; 
v_res_58_ = lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4_();
return v_res_58_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__spec__0_spec__0_spec__1___redArg(lean_object* v_a_59_, lean_object* v_x_60_){
_start:
{
if (lean_obj_tag(v_x_60_) == 0)
{
uint8_t v___x_61_; 
v___x_61_ = 0;
return v___x_61_;
}
else
{
lean_object* v_key_62_; lean_object* v_tail_63_; uint8_t v___x_64_; 
v_key_62_ = lean_ctor_get(v_x_60_, 0);
v_tail_63_ = lean_ctor_get(v_x_60_, 2);
v___x_64_ = lean_name_eq(v_key_62_, v_a_59_);
if (v___x_64_ == 0)
{
v_x_60_ = v_tail_63_;
goto _start;
}
else
{
return v___x_64_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__spec__0_spec__0_spec__1___redArg___boxed(lean_object* v_a_66_, lean_object* v_x_67_){
_start:
{
uint8_t v_res_68_; lean_object* v_r_69_; 
v_res_68_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__spec__0_spec__0_spec__1___redArg(v_a_66_, v_x_67_);
lean_dec(v_x_67_);
lean_dec(v_a_66_);
v_r_69_ = lean_box(v_res_68_);
return v_r_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__3_spec__5___redArg(lean_object* v_x_70_, lean_object* v_x_71_){
_start:
{
if (lean_obj_tag(v_x_71_) == 0)
{
return v_x_70_;
}
else
{
lean_object* v_key_72_; lean_object* v_value_73_; lean_object* v_tail_74_; lean_object* v___x_76_; uint8_t v_isShared_77_; uint8_t v_isSharedCheck_100_; 
v_key_72_ = lean_ctor_get(v_x_71_, 0);
v_value_73_ = lean_ctor_get(v_x_71_, 1);
v_tail_74_ = lean_ctor_get(v_x_71_, 2);
v_isSharedCheck_100_ = !lean_is_exclusive(v_x_71_);
if (v_isSharedCheck_100_ == 0)
{
v___x_76_ = v_x_71_;
v_isShared_77_ = v_isSharedCheck_100_;
goto v_resetjp_75_;
}
else
{
lean_inc(v_tail_74_);
lean_inc(v_value_73_);
lean_inc(v_key_72_);
lean_dec(v_x_71_);
v___x_76_ = lean_box(0);
v_isShared_77_ = v_isSharedCheck_100_;
goto v_resetjp_75_;
}
v_resetjp_75_:
{
lean_object* v___x_78_; uint64_t v___y_80_; 
v___x_78_ = lean_array_get_size(v_x_70_);
if (lean_obj_tag(v_key_72_) == 0)
{
uint64_t v___x_98_; 
v___x_98_ = 1723ULL;
v___y_80_ = v___x_98_;
goto v___jp_79_;
}
else
{
uint64_t v_hash_99_; 
v_hash_99_ = lean_ctor_get_uint64(v_key_72_, sizeof(void*)*2);
v___y_80_ = v_hash_99_;
goto v___jp_79_;
}
v___jp_79_:
{
uint64_t v___x_81_; uint64_t v___x_82_; uint64_t v_fold_83_; uint64_t v___x_84_; uint64_t v___x_85_; uint64_t v___x_86_; size_t v___x_87_; size_t v___x_88_; size_t v___x_89_; size_t v___x_90_; size_t v___x_91_; lean_object* v___x_92_; lean_object* v___x_94_; 
v___x_81_ = 32ULL;
v___x_82_ = lean_uint64_shift_right(v___y_80_, v___x_81_);
v_fold_83_ = lean_uint64_xor(v___y_80_, v___x_82_);
v___x_84_ = 16ULL;
v___x_85_ = lean_uint64_shift_right(v_fold_83_, v___x_84_);
v___x_86_ = lean_uint64_xor(v_fold_83_, v___x_85_);
v___x_87_ = lean_uint64_to_usize(v___x_86_);
v___x_88_ = lean_usize_of_nat(v___x_78_);
v___x_89_ = ((size_t)1ULL);
v___x_90_ = lean_usize_sub(v___x_88_, v___x_89_);
v___x_91_ = lean_usize_land(v___x_87_, v___x_90_);
v___x_92_ = lean_array_uget_borrowed(v_x_70_, v___x_91_);
lean_inc(v___x_92_);
if (v_isShared_77_ == 0)
{
lean_ctor_set(v___x_76_, 2, v___x_92_);
v___x_94_ = v___x_76_;
goto v_reusejp_93_;
}
else
{
lean_object* v_reuseFailAlloc_97_; 
v_reuseFailAlloc_97_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_97_, 0, v_key_72_);
lean_ctor_set(v_reuseFailAlloc_97_, 1, v_value_73_);
lean_ctor_set(v_reuseFailAlloc_97_, 2, v___x_92_);
v___x_94_ = v_reuseFailAlloc_97_;
goto v_reusejp_93_;
}
v_reusejp_93_:
{
lean_object* v___x_95_; 
v___x_95_ = lean_array_uset(v_x_70_, v___x_91_, v___x_94_);
v_x_70_ = v___x_95_;
v_x_71_ = v_tail_74_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__3___redArg(lean_object* v_i_101_, lean_object* v_source_102_, lean_object* v_target_103_){
_start:
{
lean_object* v___x_104_; uint8_t v___x_105_; 
v___x_104_ = lean_array_get_size(v_source_102_);
v___x_105_ = lean_nat_dec_lt(v_i_101_, v___x_104_);
if (v___x_105_ == 0)
{
lean_dec_ref(v_source_102_);
lean_dec(v_i_101_);
return v_target_103_;
}
else
{
lean_object* v_es_106_; lean_object* v___x_107_; lean_object* v_source_108_; lean_object* v_target_109_; lean_object* v___x_110_; lean_object* v___x_111_; 
v_es_106_ = lean_array_fget(v_source_102_, v_i_101_);
v___x_107_ = lean_box(0);
v_source_108_ = lean_array_fset(v_source_102_, v_i_101_, v___x_107_);
v_target_109_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__3_spec__5___redArg(v_target_103_, v_es_106_);
v___x_110_ = lean_unsigned_to_nat(1u);
v___x_111_ = lean_nat_add(v_i_101_, v___x_110_);
lean_dec(v_i_101_);
v_i_101_ = v___x_111_;
v_source_102_ = v_source_108_;
v_target_103_ = v_target_109_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__spec__0_spec__0_spec__2___redArg(lean_object* v_data_113_){
_start:
{
lean_object* v___x_114_; lean_object* v___x_115_; lean_object* v_nbuckets_116_; lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; lean_object* v___x_120_; 
v___x_114_ = lean_array_get_size(v_data_113_);
v___x_115_ = lean_unsigned_to_nat(2u);
v_nbuckets_116_ = lean_nat_mul(v___x_114_, v___x_115_);
v___x_117_ = lean_unsigned_to_nat(0u);
v___x_118_ = lean_box(0);
v___x_119_ = lean_mk_array(v_nbuckets_116_, v___x_118_);
v___x_120_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__3___redArg(v___x_117_, v_data_113_, v___x_119_);
return v___x_120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__spec__0_spec__0___redArg(lean_object* v_m_121_, lean_object* v_a_122_, lean_object* v_b_123_){
_start:
{
lean_object* v_size_124_; lean_object* v_buckets_125_; lean_object* v___x_126_; uint64_t v___y_128_; 
v_size_124_ = lean_ctor_get(v_m_121_, 0);
v_buckets_125_ = lean_ctor_get(v_m_121_, 1);
v___x_126_ = lean_array_get_size(v_buckets_125_);
if (lean_obj_tag(v_a_122_) == 0)
{
uint64_t v___x_165_; 
v___x_165_ = 1723ULL;
v___y_128_ = v___x_165_;
goto v___jp_127_;
}
else
{
uint64_t v_hash_166_; 
v_hash_166_ = lean_ctor_get_uint64(v_a_122_, sizeof(void*)*2);
v___y_128_ = v_hash_166_;
goto v___jp_127_;
}
v___jp_127_:
{
uint64_t v___x_129_; uint64_t v___x_130_; uint64_t v_fold_131_; uint64_t v___x_132_; uint64_t v___x_133_; uint64_t v___x_134_; size_t v___x_135_; size_t v___x_136_; size_t v___x_137_; size_t v___x_138_; size_t v___x_139_; lean_object* v_bkt_140_; uint8_t v___x_141_; 
v___x_129_ = 32ULL;
v___x_130_ = lean_uint64_shift_right(v___y_128_, v___x_129_);
v_fold_131_ = lean_uint64_xor(v___y_128_, v___x_130_);
v___x_132_ = 16ULL;
v___x_133_ = lean_uint64_shift_right(v_fold_131_, v___x_132_);
v___x_134_ = lean_uint64_xor(v_fold_131_, v___x_133_);
v___x_135_ = lean_uint64_to_usize(v___x_134_);
v___x_136_ = lean_usize_of_nat(v___x_126_);
v___x_137_ = ((size_t)1ULL);
v___x_138_ = lean_usize_sub(v___x_136_, v___x_137_);
v___x_139_ = lean_usize_land(v___x_135_, v___x_138_);
v_bkt_140_ = lean_array_uget_borrowed(v_buckets_125_, v___x_139_);
v___x_141_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__spec__0_spec__0_spec__1___redArg(v_a_122_, v_bkt_140_);
if (v___x_141_ == 0)
{
lean_object* v___x_143_; uint8_t v_isShared_144_; uint8_t v_isSharedCheck_162_; 
lean_inc_ref(v_buckets_125_);
lean_inc(v_size_124_);
v_isSharedCheck_162_ = !lean_is_exclusive(v_m_121_);
if (v_isSharedCheck_162_ == 0)
{
lean_object* v_unused_163_; lean_object* v_unused_164_; 
v_unused_163_ = lean_ctor_get(v_m_121_, 1);
lean_dec(v_unused_163_);
v_unused_164_ = lean_ctor_get(v_m_121_, 0);
lean_dec(v_unused_164_);
v___x_143_ = v_m_121_;
v_isShared_144_ = v_isSharedCheck_162_;
goto v_resetjp_142_;
}
else
{
lean_dec(v_m_121_);
v___x_143_ = lean_box(0);
v_isShared_144_ = v_isSharedCheck_162_;
goto v_resetjp_142_;
}
v_resetjp_142_:
{
lean_object* v___x_145_; lean_object* v_size_x27_146_; lean_object* v___x_147_; lean_object* v_buckets_x27_148_; lean_object* v___x_149_; lean_object* v___x_150_; lean_object* v___x_151_; lean_object* v___x_152_; lean_object* v___x_153_; uint8_t v___x_154_; 
v___x_145_ = lean_unsigned_to_nat(1u);
v_size_x27_146_ = lean_nat_add(v_size_124_, v___x_145_);
lean_dec(v_size_124_);
lean_inc(v_bkt_140_);
v___x_147_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_147_, 0, v_a_122_);
lean_ctor_set(v___x_147_, 1, v_b_123_);
lean_ctor_set(v___x_147_, 2, v_bkt_140_);
v_buckets_x27_148_ = lean_array_uset(v_buckets_125_, v___x_139_, v___x_147_);
v___x_149_ = lean_unsigned_to_nat(4u);
v___x_150_ = lean_nat_mul(v_size_x27_146_, v___x_149_);
v___x_151_ = lean_unsigned_to_nat(3u);
v___x_152_ = lean_nat_div(v___x_150_, v___x_151_);
lean_dec(v___x_150_);
v___x_153_ = lean_array_get_size(v_buckets_x27_148_);
v___x_154_ = lean_nat_dec_le(v___x_152_, v___x_153_);
lean_dec(v___x_152_);
if (v___x_154_ == 0)
{
lean_object* v_val_155_; lean_object* v___x_157_; 
v_val_155_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__spec__0_spec__0_spec__2___redArg(v_buckets_x27_148_);
if (v_isShared_144_ == 0)
{
lean_ctor_set(v___x_143_, 1, v_val_155_);
lean_ctor_set(v___x_143_, 0, v_size_x27_146_);
v___x_157_ = v___x_143_;
goto v_reusejp_156_;
}
else
{
lean_object* v_reuseFailAlloc_158_; 
v_reuseFailAlloc_158_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_158_, 0, v_size_x27_146_);
lean_ctor_set(v_reuseFailAlloc_158_, 1, v_val_155_);
v___x_157_ = v_reuseFailAlloc_158_;
goto v_reusejp_156_;
}
v_reusejp_156_:
{
return v___x_157_;
}
}
else
{
lean_object* v___x_160_; 
if (v_isShared_144_ == 0)
{
lean_ctor_set(v___x_143_, 1, v_buckets_x27_148_);
lean_ctor_set(v___x_143_, 0, v_size_x27_146_);
v___x_160_ = v___x_143_;
goto v_reusejp_159_;
}
else
{
lean_object* v_reuseFailAlloc_161_; 
v_reuseFailAlloc_161_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_161_, 0, v_size_x27_146_);
lean_ctor_set(v_reuseFailAlloc_161_, 1, v_buckets_x27_148_);
v___x_160_ = v_reuseFailAlloc_161_;
goto v_reusejp_159_;
}
v_reusejp_159_:
{
return v___x_160_;
}
}
}
}
else
{
lean_dec(v_b_123_);
lean_dec(v_a_122_);
return v_m_121_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__spec__0_spec__1(lean_object* v_as_167_, size_t v_sz_168_, size_t v_i_169_, lean_object* v_b_170_){
_start:
{
uint8_t v___x_171_; 
v___x_171_ = lean_usize_dec_lt(v_i_169_, v_sz_168_);
if (v___x_171_ == 0)
{
return v_b_170_;
}
else
{
lean_object* v_a_172_; lean_object* v___x_173_; lean_object* v_r_174_; size_t v___x_175_; size_t v___x_176_; 
v_a_172_ = lean_array_uget_borrowed(v_as_167_, v_i_169_);
v___x_173_ = lean_box(0);
lean_inc(v_a_172_);
v_r_174_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__spec__0_spec__0___redArg(v_b_170_, v_a_172_, v___x_173_);
v___x_175_ = ((size_t)1ULL);
v___x_176_ = lean_usize_add(v_i_169_, v___x_175_);
v_i_169_ = v___x_176_;
v_b_170_ = v_r_174_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__spec__0_spec__1___boxed(lean_object* v_as_178_, lean_object* v_sz_179_, lean_object* v_i_180_, lean_object* v_b_181_){
_start:
{
size_t v_sz_boxed_182_; size_t v_i_boxed_183_; lean_object* v_res_184_; 
v_sz_boxed_182_ = lean_unbox_usize(v_sz_179_);
lean_dec(v_sz_179_);
v_i_boxed_183_ = lean_unbox_usize(v_i_180_);
lean_dec(v_i_180_);
v_res_184_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__spec__0_spec__1(v_as_178_, v_sz_boxed_182_, v_i_boxed_183_, v_b_181_);
lean_dec_ref(v_as_178_);
return v_res_184_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__spec__0(lean_object* v_m_185_, lean_object* v_l_186_){
_start:
{
size_t v_sz_187_; size_t v___x_188_; lean_object* v___x_189_; 
v_sz_187_ = lean_array_size(v_l_186_);
v___x_188_ = ((size_t)0ULL);
v___x_189_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__spec__0_spec__1(v_l_186_, v_sz_187_, v___x_188_, v_m_185_);
return v___x_189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__spec__0___boxed(lean_object* v_m_190_, lean_object* v_l_191_){
_start:
{
lean_object* v_res_192_; 
v_res_192_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__spec__0(v_m_190_, v_l_191_);
lean_dec_ref(v_l_191_);
return v_res_192_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__54_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_365_; lean_object* v___x_366_; lean_object* v___x_367_; 
v___x_365_ = lean_box(0);
v___x_366_ = lean_unsigned_to_nat(16u);
v___x_367_ = lean_mk_array(v___x_366_, v___x_365_);
return v___x_367_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__55_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_368_; lean_object* v___x_369_; lean_object* v___x_370_; 
v___x_368_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__54_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__54_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__54_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_);
v___x_369_ = lean_unsigned_to_nat(0u);
v___x_370_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_370_, 0, v___x_369_);
lean_ctor_set(v___x_370_, 1, v___x_368_);
return v___x_370_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__56_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_371_; lean_object* v___x_372_; lean_object* v___x_373_; 
v___x_371_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__53_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_));
v___x_372_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__55_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__55_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__55_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_);
v___x_373_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__spec__0(v___x_372_, v___x_371_);
return v___x_373_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_375_; lean_object* v___x_376_; lean_object* v___x_377_; 
v___x_375_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__56_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__56_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn___closed__56_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_);
v___x_376_ = lean_st_mk_ref(v___x_375_);
v___x_377_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_377_, 0, v___x_376_);
return v___x_377_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2____boxed(lean_object* v_a_378_){
_start:
{
lean_object* v_res_379_; 
v_res_379_ = lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_();
return v_res_379_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__spec__0_spec__0(lean_object* v_00_u03b2_380_, lean_object* v_m_381_, lean_object* v_a_382_, lean_object* v_b_383_){
_start:
{
lean_object* v___x_384_; 
v___x_384_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__spec__0_spec__0___redArg(v_m_381_, v_a_382_, v_b_383_);
return v___x_384_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__spec__0_spec__0_spec__1(lean_object* v_00_u03b2_385_, lean_object* v_a_386_, lean_object* v_x_387_){
_start:
{
uint8_t v___x_388_; 
v___x_388_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__spec__0_spec__0_spec__1___redArg(v_a_386_, v_x_387_);
return v___x_388_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__spec__0_spec__0_spec__1___boxed(lean_object* v_00_u03b2_389_, lean_object* v_a_390_, lean_object* v_x_391_){
_start:
{
uint8_t v_res_392_; lean_object* v_r_393_; 
v_res_392_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__spec__0_spec__0_spec__1(v_00_u03b2_389_, v_a_390_, v_x_391_);
lean_dec(v_x_391_);
lean_dec(v_a_390_);
v_r_393_ = lean_box(v_res_392_);
return v_r_393_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__spec__0_spec__0_spec__2(lean_object* v_00_u03b2_394_, lean_object* v_data_395_){
_start:
{
lean_object* v___x_396_; 
v___x_396_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__spec__0_spec__0_spec__2___redArg(v_data_395_);
return v___x_396_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__3(lean_object* v_00_u03b2_397_, lean_object* v_i_398_, lean_object* v_source_399_, lean_object* v_target_400_){
_start:
{
lean_object* v___x_401_; 
v___x_401_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__3___redArg(v_i_398_, v_source_399_, v_target_400_);
return v___x_401_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__3_spec__5(lean_object* v_00_u03b2_402_, lean_object* v_x_403_, lean_object* v_x_404_){
_start:
{
lean_object* v___x_405_; 
v___x_405_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__3_spec__5___redArg(v_x_403_, v_x_404_);
return v___x_405_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Linter_UnusedTactic_isIgnoreTacticKind(lean_object* v_ignoreTacticKinds_407_, lean_object* v_k_408_){
_start:
{
if (lean_obj_tag(v_k_408_) == 1)
{
lean_object* v_str_409_; lean_object* v___x_410_; uint8_t v___x_411_; 
v_str_409_ = lean_ctor_get(v_k_408_, 1);
v___x_410_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_UnusedTactic_isIgnoreTacticKind___closed__0));
v___x_411_ = lean_string_dec_eq(v_str_409_, v___x_410_);
if (v___x_411_ == 0)
{
uint8_t v___x_412_; 
v___x_412_ = l_Lean_NameHashSet_contains(v_ignoreTacticKinds_407_, v_k_408_);
return v___x_412_;
}
else
{
return v___x_411_;
}
}
else
{
uint8_t v___x_413_; 
v___x_413_ = l_Lean_NameHashSet_contains(v_ignoreTacticKinds_407_, v_k_408_);
return v___x_413_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_UnusedTactic_isIgnoreTacticKind___boxed(lean_object* v_ignoreTacticKinds_414_, lean_object* v_k_415_){
_start:
{
uint8_t v_res_416_; lean_object* v_r_417_; 
v_res_416_ = lp_mathlib_Mathlib_Linter_UnusedTactic_isIgnoreTacticKind(v_ignoreTacticKinds_414_, v_k_415_);
lean_dec(v_k_415_);
lean_dec_ref(v_ignoreTacticKinds_414_);
v_r_417_ = lean_box(v_res_416_);
return v_r_417_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_UnusedTactic_addIgnoreTacticKind(lean_object* v_kind_418_){
_start:
{
lean_object* v___x_420_; lean_object* v___x_421_; lean_object* v___x_422_; lean_object* v___x_423_; lean_object* v___x_424_; 
v___x_420_ = lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_ignoreTacticKindsRef;
v___x_421_ = lean_st_ref_take(v___x_420_);
v___x_422_ = l_Lean_NameHashSet_insert(v___x_421_, v_kind_418_);
v___x_423_ = lean_st_ref_set(v___x_420_, v___x_422_);
v___x_424_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_424_, 0, v___x_423_);
return v___x_424_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_UnusedTactic_addIgnoreTacticKind___boxed(lean_object* v_kind_425_, lean_object* v_a_426_){
_start:
{
lean_object* v_res_427_; 
v_res_427_ = lp_mathlib_Mathlib_Linter_UnusedTactic_addIgnoreTacticKind(v_kind_425_);
return v_res_427_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___closed__0(void){
_start:
{
lean_object* v___x_428_; 
v___x_428_ = l_instMonadEIO(lean_box(0));
return v___x_428_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___closed__1(void){
_start:
{
lean_object* v___x_429_; lean_object* v___x_430_; 
v___x_429_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___closed__0, &lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___closed__0_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___closed__0);
v___x_430_ = l_StateRefT_x27_instMonad___redArg(v___x_429_);
return v___x_430_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___lam__0___boxed(lean_object* v_ignoreTacticKinds_433_, lean_object* v_isTacKind_434_, lean_object* v_x_435_, lean_object* v___y_436_, lean_object* v___y_437_, lean_object* v___y_438_){
_start:
{
lean_object* v_res_439_; 
v_res_439_ = lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___lam__0(v_ignoreTacticKinds_433_, v_isTacKind_434_, v_x_435_, v___y_436_, v___y_437_);
lean_dec(v___y_437_);
return v_res_439_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics(lean_object* v_ignoreTacticKinds_440_, lean_object* v_isTacKind_441_, lean_object* v_stx_442_, lean_object* v_a_443_){
_start:
{
lean_object* v___x_445_; 
v___x_445_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___closed__1, &lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___closed__1);
if (lean_obj_tag(v_stx_442_) == 1)
{
lean_object* v_kind_446_; lean_object* v_args_447_; lean_object* v___y_449_; lean_object* v___y_473_; uint8_t v___x_474_; 
v_kind_446_ = lean_ctor_get(v_stx_442_, 1);
v_args_447_ = lean_ctor_get(v_stx_442_, 2);
v___x_474_ = lp_mathlib_Mathlib_Linter_UnusedTactic_isIgnoreTacticKind(v_ignoreTacticKinds_440_, v_kind_446_);
if (v___x_474_ == 0)
{
lean_object* v___x_475_; lean_object* v___x_476_; uint8_t v___x_477_; 
v___x_475_ = lean_unsigned_to_nat(0u);
v___x_476_ = lean_array_get_size(v_args_447_);
v___x_477_ = lean_nat_dec_lt(v___x_475_, v___x_476_);
if (v___x_477_ == 0)
{
lean_dec_ref(v_ignoreTacticKinds_440_);
v___y_449_ = v_a_443_;
goto v___jp_448_;
}
else
{
lean_object* v___f_478_; lean_object* v___x_479_; uint8_t v___x_480_; 
lean_inc_ref(v_isTacKind_441_);
v___f_478_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___lam__0___boxed), 6, 2);
lean_closure_set(v___f_478_, 0, v_ignoreTacticKinds_440_);
lean_closure_set(v___f_478_, 1, v_isTacKind_441_);
v___x_479_ = lean_box(0);
v___x_480_ = lean_nat_dec_le(v___x_476_, v___x_476_);
if (v___x_480_ == 0)
{
if (v___x_477_ == 0)
{
lean_dec_ref(v___f_478_);
v___y_449_ = v_a_443_;
goto v___jp_448_;
}
else
{
size_t v___x_481_; size_t v___x_482_; lean_object* v___x_1198__overap_483_; lean_object* v___x_484_; 
v___x_481_ = ((size_t)0ULL);
v___x_482_ = lean_usize_of_nat(v___x_476_);
lean_inc_ref(v_args_447_);
v___x_1198__overap_483_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_445_, v___f_478_, v_args_447_, v___x_481_, v___x_482_, v___x_479_);
lean_inc(v_a_443_);
v___x_484_ = lean_apply_2(v___x_1198__overap_483_, v_a_443_, lean_box(0));
v___y_473_ = v___x_484_;
goto v___jp_472_;
}
}
else
{
size_t v___x_485_; size_t v___x_486_; lean_object* v___x_1202__overap_487_; lean_object* v___x_488_; 
v___x_485_ = ((size_t)0ULL);
v___x_486_ = lean_usize_of_nat(v___x_476_);
lean_inc_ref(v_args_447_);
v___x_1202__overap_487_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_445_, v___f_478_, v_args_447_, v___x_485_, v___x_486_, v___x_479_);
lean_inc(v_a_443_);
v___x_488_ = lean_apply_2(v___x_1202__overap_487_, v_a_443_, lean_box(0));
v___y_473_ = v___x_488_;
goto v___jp_472_;
}
}
}
else
{
lean_dec_ref(v_ignoreTacticKinds_440_);
v___y_449_ = v_a_443_;
goto v___jp_448_;
}
v___jp_448_:
{
lean_object* v___x_450_; uint8_t v___x_451_; 
lean_inc(v_kind_446_);
v___x_450_ = lean_apply_1(v_isTacKind_441_, v_kind_446_);
v___x_451_ = lean_unbox(v___x_450_);
if (v___x_451_ == 0)
{
lean_object* v___x_452_; lean_object* v___x_453_; 
lean_dec_ref_known(v_stx_442_, 3);
v___x_452_ = lean_box(0);
v___x_453_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_453_, 0, v___x_452_);
return v___x_453_;
}
else
{
uint8_t v___x_454_; lean_object* v___x_455_; 
v___x_454_ = lean_unbox(v___x_450_);
v___x_455_ = l_Lean_Syntax_getRange_x3f(v_stx_442_, v___x_454_);
if (lean_obj_tag(v___x_455_) == 1)
{
lean_object* v_val_456_; lean_object* v___x_458_; uint8_t v_isShared_459_; uint8_t v_isSharedCheck_469_; 
v_val_456_ = lean_ctor_get(v___x_455_, 0);
v_isSharedCheck_469_ = !lean_is_exclusive(v___x_455_);
if (v_isSharedCheck_469_ == 0)
{
v___x_458_ = v___x_455_;
v_isShared_459_ = v_isSharedCheck_469_;
goto v_resetjp_457_;
}
else
{
lean_inc(v_val_456_);
lean_dec(v___x_455_);
v___x_458_ = lean_box(0);
v_isShared_459_ = v_isSharedCheck_469_;
goto v_resetjp_457_;
}
v_resetjp_457_:
{
lean_object* v___x_460_; lean_object* v___x_461_; lean_object* v___x_462_; lean_object* v___x_463_; lean_object* v___x_464_; lean_object* v___x_465_; lean_object* v___x_467_; 
v___x_460_ = lean_st_ref_take(v___y_449_);
v___x_461_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___closed__2));
v___x_462_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___closed__3));
v___x_463_ = l_Std_DHashMap_Internal_Raw_u2080_insert___redArg(v___x_461_, v___x_462_, v___x_460_, v_val_456_, v_stx_442_);
v___x_464_ = lean_st_ref_set(v___y_449_, v___x_463_);
v___x_465_ = lean_box(0);
if (v_isShared_459_ == 0)
{
lean_ctor_set_tag(v___x_458_, 0);
lean_ctor_set(v___x_458_, 0, v___x_465_);
v___x_467_ = v___x_458_;
goto v_reusejp_466_;
}
else
{
lean_object* v_reuseFailAlloc_468_; 
v_reuseFailAlloc_468_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_468_, 0, v___x_465_);
v___x_467_ = v_reuseFailAlloc_468_;
goto v_reusejp_466_;
}
v_reusejp_466_:
{
return v___x_467_;
}
}
}
else
{
lean_object* v___x_470_; lean_object* v___x_471_; 
lean_dec(v___x_455_);
lean_dec_ref_known(v_stx_442_, 3);
v___x_470_ = lean_box(0);
v___x_471_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_471_, 0, v___x_470_);
return v___x_471_;
}
}
}
v___jp_472_:
{
if (lean_obj_tag(v___y_473_) == 0)
{
lean_dec_ref_known(v___y_473_, 1);
v___y_449_ = v_a_443_;
goto v___jp_448_;
}
else
{
lean_dec_ref_known(v_stx_442_, 3);
lean_dec_ref(v_isTacKind_441_);
return v___y_473_;
}
}
}
else
{
lean_object* v___x_489_; lean_object* v___x_490_; 
lean_dec(v_stx_442_);
lean_dec_ref(v_isTacKind_441_);
lean_dec_ref(v_ignoreTacticKinds_440_);
v___x_489_ = lean_box(0);
v___x_490_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_490_, 0, v___x_489_);
return v___x_490_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___lam__0(lean_object* v_ignoreTacticKinds_491_, lean_object* v_isTacKind_492_, lean_object* v_x_493_, lean_object* v___y_494_, lean_object* v___y_495_){
_start:
{
lean_object* v___x_497_; 
v___x_497_ = lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics(v_ignoreTacticKinds_491_, v_isTacKind_492_, v___y_494_, v___y_495_);
return v___x_497_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___boxed(lean_object* v_ignoreTacticKinds_498_, lean_object* v_isTacKind_499_, lean_object* v_stx_500_, lean_object* v_a_501_, lean_object* v_a_502_){
_start:
{
lean_object* v_res_503_; 
v_res_503_ = lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics(v_ignoreTacticKinds_498_, v_isTacKind_499_, v_stx_500_, v_a_501_);
lean_dec(v_a_501_);
return v_res_503_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__2(lean_object* v_a_504_, lean_object* v_a_505_){
_start:
{
if (lean_obj_tag(v_a_504_) == 0)
{
lean_object* v___x_506_; 
v___x_506_ = l_List_reverse___redArg(v_a_505_);
return v___x_506_;
}
else
{
lean_object* v_head_507_; lean_object* v_tail_508_; lean_object* v___x_510_; uint8_t v_isShared_511_; uint8_t v_isSharedCheck_518_; 
v_head_507_ = lean_ctor_get(v_a_504_, 0);
v_tail_508_ = lean_ctor_get(v_a_504_, 1);
v_isSharedCheck_518_ = !lean_is_exclusive(v_a_504_);
if (v_isSharedCheck_518_ == 0)
{
v___x_510_ = v_a_504_;
v_isShared_511_ = v_isSharedCheck_518_;
goto v_resetjp_509_;
}
else
{
lean_inc(v_tail_508_);
lean_inc(v_head_507_);
lean_dec(v_a_504_);
v___x_510_ = lean_box(0);
v_isShared_511_ = v_isSharedCheck_518_;
goto v_resetjp_509_;
}
v_resetjp_509_:
{
lean_object* v_decls_512_; lean_object* v___x_513_; lean_object* v___x_515_; 
v_decls_512_ = lean_ctor_get(v_head_507_, 1);
lean_inc_ref(v_decls_512_);
lean_dec(v_head_507_);
v___x_513_ = l_Lean_PersistentArray_toList___redArg(v_decls_512_);
lean_dec_ref(v_decls_512_);
if (v_isShared_511_ == 0)
{
lean_ctor_set(v___x_510_, 1, v_a_505_);
lean_ctor_set(v___x_510_, 0, v___x_513_);
v___x_515_ = v___x_510_;
goto v_reusejp_514_;
}
else
{
lean_object* v_reuseFailAlloc_517_; 
v_reuseFailAlloc_517_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_517_, 0, v___x_513_);
lean_ctor_set(v_reuseFailAlloc_517_, 1, v_a_505_);
v___x_515_ = v_reuseFailAlloc_517_;
goto v_reusejp_514_;
}
v_reusejp_514_:
{
v_a_504_ = v_tail_508_;
v_a_505_ = v___x_515_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__3(lean_object* v_a_519_, lean_object* v_a_520_){
_start:
{
if (lean_obj_tag(v_a_519_) == 0)
{
lean_object* v___x_521_; 
v___x_521_ = lean_array_to_list(v_a_520_);
return v___x_521_;
}
else
{
lean_object* v_head_522_; lean_object* v_tail_523_; lean_object* v___x_524_; 
v_head_522_ = lean_ctor_get(v_a_519_, 0);
lean_inc(v_head_522_);
v_tail_523_ = lean_ctor_get(v_a_519_, 1);
lean_inc(v_tail_523_);
lean_dec_ref_known(v_a_519_, 2);
v___x_524_ = l_List_foldl___at___00Array_appendList_spec__0___redArg(v_a_520_, v_head_522_);
v_a_519_ = v_tail_523_;
v_a_520_ = v___x_524_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__5(lean_object* v_a_526_, lean_object* v_a_527_){
_start:
{
if (lean_obj_tag(v_a_526_) == 0)
{
lean_object* v___x_528_; 
v___x_528_ = l_List_reverse___redArg(v_a_527_);
return v___x_528_;
}
else
{
lean_object* v_head_529_; lean_object* v_tail_530_; lean_object* v___x_532_; uint8_t v_isShared_533_; uint8_t v_isSharedCheck_539_; 
v_head_529_ = lean_ctor_get(v_a_526_, 0);
v_tail_530_ = lean_ctor_get(v_a_526_, 1);
v_isSharedCheck_539_ = !lean_is_exclusive(v_a_526_);
if (v_isSharedCheck_539_ == 0)
{
v___x_532_ = v_a_526_;
v_isShared_533_ = v_isSharedCheck_539_;
goto v_resetjp_531_;
}
else
{
lean_inc(v_tail_530_);
lean_inc(v_head_529_);
lean_dec(v_a_526_);
v___x_532_ = lean_box(0);
v_isShared_533_ = v_isSharedCheck_539_;
goto v_resetjp_531_;
}
v_resetjp_531_:
{
lean_object* v___x_534_; lean_object* v___x_536_; 
v___x_534_ = l_Lean_LocalDecl_userName(v_head_529_);
lean_dec(v_head_529_);
if (v_isShared_533_ == 0)
{
lean_ctor_set(v___x_532_, 1, v_a_527_);
lean_ctor_set(v___x_532_, 0, v___x_534_);
v___x_536_ = v___x_532_;
goto v_reusejp_535_;
}
else
{
lean_object* v_reuseFailAlloc_538_; 
v_reuseFailAlloc_538_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_538_, 0, v___x_534_);
lean_ctor_set(v_reuseFailAlloc_538_, 1, v_a_527_);
v___x_536_ = v_reuseFailAlloc_538_;
goto v_reusejp_535_;
}
v_reusejp_535_:
{
v_a_526_ = v_tail_530_;
v_a_527_ = v___x_536_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__1(lean_object* v_a_540_, lean_object* v_a_541_){
_start:
{
if (lean_obj_tag(v_a_540_) == 0)
{
lean_object* v___x_542_; 
v___x_542_ = l_List_reverse___redArg(v_a_541_);
return v___x_542_;
}
else
{
lean_object* v_head_543_; lean_object* v_snd_544_; lean_object* v_tail_545_; lean_object* v___x_547_; uint8_t v_isShared_548_; uint8_t v_isSharedCheck_554_; 
v_head_543_ = lean_ctor_get(v_a_540_, 0);
v_snd_544_ = lean_ctor_get(v_head_543_, 1);
lean_inc(v_snd_544_);
v_tail_545_ = lean_ctor_get(v_a_540_, 1);
v_isSharedCheck_554_ = !lean_is_exclusive(v_a_540_);
if (v_isSharedCheck_554_ == 0)
{
lean_object* v_unused_555_; 
v_unused_555_ = lean_ctor_get(v_a_540_, 0);
lean_dec(v_unused_555_);
v___x_547_ = v_a_540_;
v_isShared_548_ = v_isSharedCheck_554_;
goto v_resetjp_546_;
}
else
{
lean_inc(v_tail_545_);
lean_dec(v_a_540_);
v___x_547_ = lean_box(0);
v_isShared_548_ = v_isSharedCheck_554_;
goto v_resetjp_546_;
}
v_resetjp_546_:
{
lean_object* v_lctx_549_; lean_object* v___x_551_; 
v_lctx_549_ = lean_ctor_get(v_snd_544_, 1);
lean_inc_ref(v_lctx_549_);
lean_dec(v_snd_544_);
if (v_isShared_548_ == 0)
{
lean_ctor_set(v___x_547_, 1, v_a_541_);
lean_ctor_set(v___x_547_, 0, v_lctx_549_);
v___x_551_ = v___x_547_;
goto v_reusejp_550_;
}
else
{
lean_object* v_reuseFailAlloc_553_; 
v_reuseFailAlloc_553_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_553_, 0, v_lctx_549_);
lean_ctor_set(v_reuseFailAlloc_553_, 1, v_a_541_);
v___x_551_ = v_reuseFailAlloc_553_;
goto v_reusejp_550_;
}
v_reusejp_550_:
{
v_a_540_ = v_tail_545_;
v_a_541_ = v___x_551_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_filterMapTR_go___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__4(lean_object* v_a_556_, lean_object* v_a_557_){
_start:
{
if (lean_obj_tag(v_a_556_) == 0)
{
lean_object* v___x_558_; 
v___x_558_ = lean_array_to_list(v_a_557_);
return v___x_558_;
}
else
{
lean_object* v_head_559_; 
v_head_559_ = lean_ctor_get(v_a_556_, 0);
if (lean_obj_tag(v_head_559_) == 0)
{
lean_object* v_tail_560_; 
v_tail_560_ = lean_ctor_get(v_a_556_, 1);
lean_inc(v_tail_560_);
lean_dec_ref_known(v_a_556_, 2);
v_a_556_ = v_tail_560_;
goto _start;
}
else
{
lean_object* v_tail_562_; lean_object* v_val_563_; lean_object* v___x_564_; 
lean_inc_ref(v_head_559_);
v_tail_562_ = lean_ctor_get(v_a_556_, 1);
lean_inc(v_tail_562_);
lean_dec_ref_known(v_a_556_, 2);
v_val_563_ = lean_ctor_get(v_head_559_, 0);
lean_inc(v_val_563_);
lean_dec_ref_known(v_head_559_, 1);
v___x_564_ = lean_array_push(v_a_557_, v_val_563_);
v_a_556_ = v_tail_562_;
v_a_557_ = v___x_564_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0___redArg___lam__0(lean_object* v_ps_566_, lean_object* v_k_567_, lean_object* v_v_568_){
_start:
{
lean_object* v___x_569_; lean_object* v___x_570_; 
v___x_569_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_569_, 0, v_k_567_);
lean_ctor_set(v___x_569_, 1, v_v_568_);
v___x_570_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_570_, 0, v___x_569_);
lean_ctor_set(v___x_570_, 1, v_ps_566_);
return v___x_570_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0_spec__1_spec__7_spec__9___redArg(lean_object* v_f_571_, lean_object* v_keys_572_, lean_object* v_vals_573_, lean_object* v_i_574_, lean_object* v_acc_575_){
_start:
{
lean_object* v___x_576_; uint8_t v___x_577_; 
v___x_576_ = lean_array_get_size(v_keys_572_);
v___x_577_ = lean_nat_dec_lt(v_i_574_, v___x_576_);
if (v___x_577_ == 0)
{
lean_dec(v_i_574_);
lean_dec(v_f_571_);
return v_acc_575_;
}
else
{
lean_object* v_k_578_; lean_object* v_v_579_; lean_object* v___x_580_; lean_object* v___x_581_; lean_object* v___x_582_; 
v_k_578_ = lean_array_fget_borrowed(v_keys_572_, v_i_574_);
v_v_579_ = lean_array_fget_borrowed(v_vals_573_, v_i_574_);
lean_inc(v_f_571_);
lean_inc(v_v_579_);
lean_inc(v_k_578_);
v___x_580_ = lean_apply_3(v_f_571_, v_acc_575_, v_k_578_, v_v_579_);
v___x_581_ = lean_unsigned_to_nat(1u);
v___x_582_ = lean_nat_add(v_i_574_, v___x_581_);
lean_dec(v_i_574_);
v_i_574_ = v___x_582_;
v_acc_575_ = v___x_580_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0_spec__1_spec__7_spec__9___redArg___boxed(lean_object* v_f_584_, lean_object* v_keys_585_, lean_object* v_vals_586_, lean_object* v_i_587_, lean_object* v_acc_588_){
_start:
{
lean_object* v_res_589_; 
v_res_589_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0_spec__1_spec__7_spec__9___redArg(v_f_584_, v_keys_585_, v_vals_586_, v_i_587_, v_acc_588_);
lean_dec_ref(v_vals_586_);
lean_dec_ref(v_keys_585_);
return v_res_589_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0_spec__1_spec__7___redArg(lean_object* v_f_590_, lean_object* v_x_591_, lean_object* v_x_592_){
_start:
{
if (lean_obj_tag(v_x_591_) == 0)
{
lean_object* v_es_593_; lean_object* v___x_594_; lean_object* v___x_595_; uint8_t v___x_596_; 
v_es_593_ = lean_ctor_get(v_x_591_, 0);
v___x_594_ = lean_unsigned_to_nat(0u);
v___x_595_ = lean_array_get_size(v_es_593_);
v___x_596_ = lean_nat_dec_lt(v___x_594_, v___x_595_);
if (v___x_596_ == 0)
{
lean_dec(v_f_590_);
return v_x_592_;
}
else
{
uint8_t v___x_597_; 
v___x_597_ = lean_nat_dec_le(v___x_595_, v___x_595_);
if (v___x_597_ == 0)
{
if (v___x_596_ == 0)
{
lean_dec(v_f_590_);
return v_x_592_;
}
else
{
size_t v___x_598_; size_t v___x_599_; lean_object* v___x_600_; 
v___x_598_ = ((size_t)0ULL);
v___x_599_ = lean_usize_of_nat(v___x_595_);
v___x_600_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0_spec__1_spec__7_spec__8___redArg(v_f_590_, v_es_593_, v___x_598_, v___x_599_, v_x_592_);
return v___x_600_;
}
}
else
{
size_t v___x_601_; size_t v___x_602_; lean_object* v___x_603_; 
v___x_601_ = ((size_t)0ULL);
v___x_602_ = lean_usize_of_nat(v___x_595_);
v___x_603_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0_spec__1_spec__7_spec__8___redArg(v_f_590_, v_es_593_, v___x_601_, v___x_602_, v_x_592_);
return v___x_603_;
}
}
}
else
{
lean_object* v_ks_604_; lean_object* v_vs_605_; lean_object* v___x_606_; lean_object* v___x_607_; 
v_ks_604_ = lean_ctor_get(v_x_591_, 0);
v_vs_605_ = lean_ctor_get(v_x_591_, 1);
v___x_606_ = lean_unsigned_to_nat(0u);
v___x_607_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0_spec__1_spec__7_spec__9___redArg(v_f_590_, v_ks_604_, v_vs_605_, v___x_606_, v_x_592_);
return v___x_607_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0_spec__1_spec__7_spec__8___redArg(lean_object* v_f_608_, lean_object* v_as_609_, size_t v_i_610_, size_t v_stop_611_, lean_object* v_b_612_){
_start:
{
lean_object* v___y_614_; uint8_t v___x_618_; 
v___x_618_ = lean_usize_dec_eq(v_i_610_, v_stop_611_);
if (v___x_618_ == 0)
{
lean_object* v___x_619_; 
v___x_619_ = lean_array_uget_borrowed(v_as_609_, v_i_610_);
switch(lean_obj_tag(v___x_619_))
{
case 0:
{
lean_object* v_key_620_; lean_object* v_val_621_; lean_object* v___x_622_; 
v_key_620_ = lean_ctor_get(v___x_619_, 0);
v_val_621_ = lean_ctor_get(v___x_619_, 1);
lean_inc(v_f_608_);
lean_inc(v_val_621_);
lean_inc(v_key_620_);
v___x_622_ = lean_apply_3(v_f_608_, v_b_612_, v_key_620_, v_val_621_);
v___y_614_ = v___x_622_;
goto v___jp_613_;
}
case 1:
{
lean_object* v_node_623_; lean_object* v___x_624_; 
v_node_623_ = lean_ctor_get(v___x_619_, 0);
lean_inc(v_f_608_);
v___x_624_ = lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0_spec__1_spec__7___redArg(v_f_608_, v_node_623_, v_b_612_);
v___y_614_ = v___x_624_;
goto v___jp_613_;
}
default: 
{
v___y_614_ = v_b_612_;
goto v___jp_613_;
}
}
}
else
{
lean_dec(v_f_608_);
return v_b_612_;
}
v___jp_613_:
{
size_t v___x_615_; size_t v___x_616_; 
v___x_615_ = ((size_t)1ULL);
v___x_616_ = lean_usize_add(v_i_610_, v___x_615_);
v_i_610_ = v___x_616_;
v_b_612_ = v___y_614_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0_spec__1_spec__7_spec__8___redArg___boxed(lean_object* v_f_625_, lean_object* v_as_626_, lean_object* v_i_627_, lean_object* v_stop_628_, lean_object* v_b_629_){
_start:
{
size_t v_i_boxed_630_; size_t v_stop_boxed_631_; lean_object* v_res_632_; 
v_i_boxed_630_ = lean_unbox_usize(v_i_627_);
lean_dec(v_i_627_);
v_stop_boxed_631_ = lean_unbox_usize(v_stop_628_);
lean_dec(v_stop_628_);
v_res_632_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0_spec__1_spec__7_spec__8___redArg(v_f_625_, v_as_626_, v_i_boxed_630_, v_stop_boxed_631_, v_b_629_);
lean_dec_ref(v_as_626_);
return v_res_632_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0_spec__1_spec__7___redArg___boxed(lean_object* v_f_633_, lean_object* v_x_634_, lean_object* v_x_635_){
_start:
{
lean_object* v_res_636_; 
v_res_636_ = lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0_spec__1_spec__7___redArg(v_f_633_, v_x_634_, v_x_635_);
lean_dec_ref(v_x_634_);
return v_res_636_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0___redArg___lam__0(lean_object* v_f_637_, lean_object* v_x1_638_, lean_object* v_x2_639_, lean_object* v_x3_640_){
_start:
{
lean_object* v___x_641_; 
v___x_641_ = lean_apply_3(v_f_637_, v_x1_638_, v_x2_639_, v_x3_640_);
return v___x_641_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0___redArg(lean_object* v_map_642_, lean_object* v_f_643_, lean_object* v_init_644_){
_start:
{
lean_object* v___f_645_; lean_object* v___x_646_; 
v___f_645_ = lean_alloc_closure((void*)(lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0___redArg___lam__0), 4, 1);
lean_closure_set(v___f_645_, 0, v_f_643_);
v___x_646_ = lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0_spec__1_spec__7___redArg(v___f_645_, v_map_642_, v_init_644_);
return v___x_646_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0___redArg___boxed(lean_object* v_map_647_, lean_object* v_f_648_, lean_object* v_init_649_){
_start:
{
lean_object* v_res_650_; 
v_res_650_ = lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0___redArg(v_map_647_, v_f_648_, v_init_649_);
lean_dec_ref(v_map_647_);
return v_res_650_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0___redArg(lean_object* v_m_652_){
_start:
{
lean_object* v___f_653_; lean_object* v___x_654_; lean_object* v___x_655_; 
v___f_653_ = ((lean_object*)(lp_mathlib_Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0___redArg___closed__0));
v___x_654_ = lean_box(0);
v___x_655_ = lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0___redArg(v_m_652_, v___f_653_, v___x_654_);
return v___x_655_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0___redArg___boxed(lean_object* v_m_656_){
_start:
{
lean_object* v_res_657_; 
v_res_657_ = lp_mathlib_Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0___redArg(v_m_656_);
lean_dec_ref(v_m_656_);
return v_res_657_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames(lean_object* v_mctx_660_){
_start:
{
lean_object* v_decls_661_; lean_object* v___x_662_; lean_object* v___x_663_; lean_object* v_lcts_664_; lean_object* v___x_665_; lean_object* v___x_666_; lean_object* v___x_667_; lean_object* v_locDecls_668_; lean_object* v___x_669_; 
v_decls_661_ = lean_ctor_get(v_mctx_660_, 5);
v___x_662_ = lp_mathlib_Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0___redArg(v_decls_661_);
v___x_663_ = lean_box(0);
v_lcts_664_ = lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__1(v___x_662_, v___x_663_);
v___x_665_ = lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__2(v_lcts_664_, v___x_663_);
v___x_666_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames___closed__0));
v___x_667_ = lp_mathlib___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__3(v___x_665_, v___x_666_);
v_locDecls_668_ = lp_mathlib_List_filterMapTR_go___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__4(v___x_667_, v___x_666_);
v___x_669_ = lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__5(v_locDecls_668_, v___x_663_);
return v___x_669_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames___boxed(lean_object* v_mctx_670_){
_start:
{
lean_object* v_res_671_; 
v_res_671_ = lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames(v_mctx_670_);
lean_dec_ref(v_mctx_670_);
return v_res_671_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0(lean_object* v_00_u03b2_672_, lean_object* v_m_673_){
_start:
{
lean_object* v___x_674_; 
v___x_674_ = lp_mathlib_Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0___redArg(v_m_673_);
return v___x_674_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0___boxed(lean_object* v_00_u03b2_675_, lean_object* v_m_676_){
_start:
{
lean_object* v_res_677_; 
v_res_677_ = lp_mathlib_Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0(v_00_u03b2_675_, v_m_676_);
lean_dec_ref(v_m_676_);
return v_res_677_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0(lean_object* v_00_u03c3_678_, lean_object* v_00_u03b2_679_, lean_object* v_map_680_, lean_object* v_f_681_, lean_object* v_init_682_){
_start:
{
lean_object* v___x_683_; 
v___x_683_ = lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0___redArg(v_map_680_, v_f_681_, v_init_682_);
return v___x_683_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0___boxed(lean_object* v_00_u03c3_684_, lean_object* v_00_u03b2_685_, lean_object* v_map_686_, lean_object* v_f_687_, lean_object* v_init_688_){
_start:
{
lean_object* v_res_689_; 
v_res_689_ = lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0(v_00_u03c3_684_, v_00_u03b2_685_, v_map_686_, v_f_687_, v_init_688_);
lean_dec_ref(v_map_686_);
return v_res_689_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0_spec__1___redArg(lean_object* v_map_690_, lean_object* v_f_691_, lean_object* v_init_692_){
_start:
{
lean_object* v___x_693_; 
v___x_693_ = lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0_spec__1_spec__7___redArg(v_f_691_, v_map_690_, v_init_692_);
return v___x_693_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0_spec__1___redArg___boxed(lean_object* v_map_694_, lean_object* v_f_695_, lean_object* v_init_696_){
_start:
{
lean_object* v_res_697_; 
v_res_697_ = lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0_spec__1___redArg(v_map_694_, v_f_695_, v_init_696_);
lean_dec_ref(v_map_694_);
return v_res_697_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0_spec__1(lean_object* v_00_u03c3_698_, lean_object* v_00_u03b2_699_, lean_object* v_map_700_, lean_object* v_f_701_, lean_object* v_init_702_){
_start:
{
lean_object* v___x_703_; 
v___x_703_ = lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0_spec__1_spec__7___redArg(v_f_701_, v_map_700_, v_init_702_);
return v___x_703_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0_spec__1___boxed(lean_object* v_00_u03c3_704_, lean_object* v_00_u03b2_705_, lean_object* v_map_706_, lean_object* v_f_707_, lean_object* v_init_708_){
_start:
{
lean_object* v_res_709_; 
v_res_709_ = lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0_spec__1(v_00_u03c3_704_, v_00_u03b2_705_, v_map_706_, v_f_707_, v_init_708_);
lean_dec_ref(v_map_706_);
return v_res_709_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0_spec__1_spec__7(lean_object* v_00_u03c3_710_, lean_object* v_00_u03b1_711_, lean_object* v_00_u03b2_712_, lean_object* v_f_713_, lean_object* v_x_714_, lean_object* v_x_715_){
_start:
{
lean_object* v___x_716_; 
v___x_716_ = lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0_spec__1_spec__7___redArg(v_f_713_, v_x_714_, v_x_715_);
return v___x_716_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0_spec__1_spec__7___boxed(lean_object* v_00_u03c3_717_, lean_object* v_00_u03b1_718_, lean_object* v_00_u03b2_719_, lean_object* v_f_720_, lean_object* v_x_721_, lean_object* v_x_722_){
_start:
{
lean_object* v_res_723_; 
v_res_723_ = lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0_spec__1_spec__7(v_00_u03c3_717_, v_00_u03b1_718_, v_00_u03b2_719_, v_f_720_, v_x_721_, v_x_722_);
lean_dec_ref(v_x_721_);
return v_res_723_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0_spec__1_spec__7_spec__8(lean_object* v_00_u03b1_724_, lean_object* v_00_u03b2_725_, lean_object* v_00_u03c3_726_, lean_object* v_f_727_, lean_object* v_as_728_, size_t v_i_729_, size_t v_stop_730_, lean_object* v_b_731_){
_start:
{
lean_object* v___x_732_; 
v___x_732_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0_spec__1_spec__7_spec__8___redArg(v_f_727_, v_as_728_, v_i_729_, v_stop_730_, v_b_731_);
return v___x_732_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0_spec__1_spec__7_spec__8___boxed(lean_object* v_00_u03b1_733_, lean_object* v_00_u03b2_734_, lean_object* v_00_u03c3_735_, lean_object* v_f_736_, lean_object* v_as_737_, lean_object* v_i_738_, lean_object* v_stop_739_, lean_object* v_b_740_){
_start:
{
size_t v_i_boxed_741_; size_t v_stop_boxed_742_; lean_object* v_res_743_; 
v_i_boxed_741_ = lean_unbox_usize(v_i_738_);
lean_dec(v_i_738_);
v_stop_boxed_742_ = lean_unbox_usize(v_stop_739_);
lean_dec(v_stop_739_);
v_res_743_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0_spec__1_spec__7_spec__8(v_00_u03b1_733_, v_00_u03b2_734_, v_00_u03c3_735_, v_f_736_, v_as_737_, v_i_boxed_741_, v_stop_boxed_742_, v_b_740_);
lean_dec_ref(v_as_737_);
return v_res_743_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0_spec__1_spec__7_spec__9(lean_object* v_00_u03c3_744_, lean_object* v_00_u03b1_745_, lean_object* v_00_u03b2_746_, lean_object* v_f_747_, lean_object* v_keys_748_, lean_object* v_vals_749_, lean_object* v_heq_750_, lean_object* v_i_751_, lean_object* v_acc_752_){
_start:
{
lean_object* v___x_753_; 
v___x_753_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0_spec__1_spec__7_spec__9___redArg(v_f_747_, v_keys_748_, v_vals_749_, v_i_751_, v_acc_752_);
return v___x_753_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0_spec__1_spec__7_spec__9___boxed(lean_object* v_00_u03c3_754_, lean_object* v_00_u03b1_755_, lean_object* v_00_u03b2_756_, lean_object* v_f_757_, lean_object* v_keys_758_, lean_object* v_vals_759_, lean_object* v_heq_760_, lean_object* v_i_761_, lean_object* v_acc_762_){
_start:
{
lean_object* v_res_763_; 
v_res_763_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames_spec__0_spec__0_spec__1_spec__7_spec__9(v_00_u03c3_754_, v_00_u03b1_755_, v_00_u03b2_756_, v_f_757_, v_keys_758_, v_vals_759_, v_heq_760_, v_i_761_, v_acc_762_);
lean_dec_ref(v_vals_759_);
lean_dec_ref(v_keys_758_);
return v_res_763_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__4_spec__8___redArg(lean_object* v_a_764_, lean_object* v_x_765_){
_start:
{
if (lean_obj_tag(v_x_765_) == 0)
{
return v_x_765_;
}
else
{
lean_object* v_key_766_; lean_object* v_value_767_; lean_object* v_tail_768_; lean_object* v___x_770_; uint8_t v_isShared_771_; uint8_t v_isSharedCheck_777_; 
v_key_766_ = lean_ctor_get(v_x_765_, 0);
v_value_767_ = lean_ctor_get(v_x_765_, 1);
v_tail_768_ = lean_ctor_get(v_x_765_, 2);
v_isSharedCheck_777_ = !lean_is_exclusive(v_x_765_);
if (v_isSharedCheck_777_ == 0)
{
v___x_770_ = v_x_765_;
v_isShared_771_ = v_isSharedCheck_777_;
goto v_resetjp_769_;
}
else
{
lean_inc(v_tail_768_);
lean_inc(v_value_767_);
lean_inc(v_key_766_);
lean_dec(v_x_765_);
v___x_770_ = lean_box(0);
v_isShared_771_ = v_isSharedCheck_777_;
goto v_resetjp_769_;
}
v_resetjp_769_:
{
uint8_t v___x_772_; 
v___x_772_ = l_Lean_Syntax_instBEqRange_beq(v_key_766_, v_a_764_);
if (v___x_772_ == 0)
{
lean_object* v___x_773_; lean_object* v___x_775_; 
v___x_773_ = lp_mathlib_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__4_spec__8___redArg(v_a_764_, v_tail_768_);
if (v_isShared_771_ == 0)
{
lean_ctor_set(v___x_770_, 2, v___x_773_);
v___x_775_ = v___x_770_;
goto v_reusejp_774_;
}
else
{
lean_object* v_reuseFailAlloc_776_; 
v_reuseFailAlloc_776_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_776_, 0, v_key_766_);
lean_ctor_set(v_reuseFailAlloc_776_, 1, v_value_767_);
lean_ctor_set(v_reuseFailAlloc_776_, 2, v___x_773_);
v___x_775_ = v_reuseFailAlloc_776_;
goto v_reusejp_774_;
}
v_reusejp_774_:
{
return v___x_775_;
}
}
else
{
lean_del_object(v___x_770_);
lean_dec(v_value_767_);
lean_dec(v_key_766_);
return v_tail_768_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__4_spec__8___redArg___boxed(lean_object* v_a_778_, lean_object* v_x_779_){
_start:
{
lean_object* v_res_780_; 
v_res_780_ = lp_mathlib_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__4_spec__8___redArg(v_a_778_, v_x_779_);
lean_dec_ref(v_a_778_);
return v_res_780_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__4_spec__7___redArg(lean_object* v_a_781_, lean_object* v_x_782_){
_start:
{
if (lean_obj_tag(v_x_782_) == 0)
{
uint8_t v___x_783_; 
v___x_783_ = 0;
return v___x_783_;
}
else
{
lean_object* v_key_784_; lean_object* v_tail_785_; uint8_t v___x_786_; 
v_key_784_ = lean_ctor_get(v_x_782_, 0);
v_tail_785_ = lean_ctor_get(v_x_782_, 2);
v___x_786_ = l_Lean_Syntax_instBEqRange_beq(v_key_784_, v_a_781_);
if (v___x_786_ == 0)
{
v_x_782_ = v_tail_785_;
goto _start;
}
else
{
return v___x_786_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__4_spec__7___redArg___boxed(lean_object* v_a_788_, lean_object* v_x_789_){
_start:
{
uint8_t v_res_790_; lean_object* v_r_791_; 
v_res_790_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__4_spec__7___redArg(v_a_788_, v_x_789_);
lean_dec(v_x_789_);
lean_dec_ref(v_a_788_);
v_r_791_ = lean_box(v_res_790_);
return v_r_791_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__4___redArg(lean_object* v_m_792_, lean_object* v_a_793_){
_start:
{
lean_object* v_size_794_; lean_object* v_buckets_795_; lean_object* v___x_796_; uint64_t v___x_797_; uint64_t v___x_798_; uint64_t v___x_799_; uint64_t v_fold_800_; uint64_t v___x_801_; uint64_t v___x_802_; uint64_t v___x_803_; size_t v___x_804_; size_t v___x_805_; size_t v___x_806_; size_t v___x_807_; size_t v___x_808_; lean_object* v_bkt_809_; uint8_t v___x_810_; 
v_size_794_ = lean_ctor_get(v_m_792_, 0);
v_buckets_795_ = lean_ctor_get(v_m_792_, 1);
v___x_796_ = lean_array_get_size(v_buckets_795_);
v___x_797_ = l_Lean_Syntax_instHashableRange_hash(v_a_793_);
v___x_798_ = 32ULL;
v___x_799_ = lean_uint64_shift_right(v___x_797_, v___x_798_);
v_fold_800_ = lean_uint64_xor(v___x_797_, v___x_799_);
v___x_801_ = 16ULL;
v___x_802_ = lean_uint64_shift_right(v_fold_800_, v___x_801_);
v___x_803_ = lean_uint64_xor(v_fold_800_, v___x_802_);
v___x_804_ = lean_uint64_to_usize(v___x_803_);
v___x_805_ = lean_usize_of_nat(v___x_796_);
v___x_806_ = ((size_t)1ULL);
v___x_807_ = lean_usize_sub(v___x_805_, v___x_806_);
v___x_808_ = lean_usize_land(v___x_804_, v___x_807_);
v_bkt_809_ = lean_array_uget_borrowed(v_buckets_795_, v___x_808_);
v___x_810_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__4_spec__7___redArg(v_a_793_, v_bkt_809_);
if (v___x_810_ == 0)
{
return v_m_792_;
}
else
{
lean_object* v___x_812_; uint8_t v_isShared_813_; uint8_t v_isSharedCheck_823_; 
lean_inc(v_bkt_809_);
lean_inc_ref(v_buckets_795_);
lean_inc(v_size_794_);
v_isSharedCheck_823_ = !lean_is_exclusive(v_m_792_);
if (v_isSharedCheck_823_ == 0)
{
lean_object* v_unused_824_; lean_object* v_unused_825_; 
v_unused_824_ = lean_ctor_get(v_m_792_, 1);
lean_dec(v_unused_824_);
v_unused_825_ = lean_ctor_get(v_m_792_, 0);
lean_dec(v_unused_825_);
v___x_812_ = v_m_792_;
v_isShared_813_ = v_isSharedCheck_823_;
goto v_resetjp_811_;
}
else
{
lean_dec(v_m_792_);
v___x_812_ = lean_box(0);
v_isShared_813_ = v_isSharedCheck_823_;
goto v_resetjp_811_;
}
v_resetjp_811_:
{
lean_object* v___x_814_; lean_object* v_buckets_x27_815_; lean_object* v___x_816_; lean_object* v___x_817_; lean_object* v___x_818_; lean_object* v___x_819_; lean_object* v___x_821_; 
v___x_814_ = lean_box(0);
v_buckets_x27_815_ = lean_array_uset(v_buckets_795_, v___x_808_, v___x_814_);
v___x_816_ = lean_unsigned_to_nat(1u);
v___x_817_ = lean_nat_sub(v_size_794_, v___x_816_);
lean_dec(v_size_794_);
v___x_818_ = lp_mathlib_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__4_spec__8___redArg(v_a_793_, v_bkt_809_);
v___x_819_ = lean_array_uset(v_buckets_x27_815_, v___x_808_, v___x_818_);
if (v_isShared_813_ == 0)
{
lean_ctor_set(v___x_812_, 1, v___x_819_);
lean_ctor_set(v___x_812_, 0, v___x_817_);
v___x_821_ = v___x_812_;
goto v_reusejp_820_;
}
else
{
lean_object* v_reuseFailAlloc_822_; 
v_reuseFailAlloc_822_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_822_, 0, v___x_817_);
lean_ctor_set(v_reuseFailAlloc_822_, 1, v___x_819_);
v___x_821_ = v_reuseFailAlloc_822_;
goto v_reusejp_820_;
}
v_reusejp_820_:
{
return v___x_821_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__4___redArg___boxed(lean_object* v_m_826_, lean_object* v_a_827_){
_start:
{
lean_object* v_res_828_; 
v_res_828_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__4___redArg(v_m_826_, v_a_827_);
lean_dec_ref(v_a_827_);
return v_res_828_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__5(lean_object* v_as_829_, size_t v_sz_830_, size_t v_i_831_, lean_object* v_b_832_, lean_object* v___y_833_){
_start:
{
uint8_t v___x_835_; 
v___x_835_ = lean_usize_dec_lt(v_i_831_, v_sz_830_);
if (v___x_835_ == 0)
{
lean_object* v___x_836_; 
v___x_836_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_836_, 0, v_b_832_);
return v___x_836_;
}
else
{
lean_object* v___x_837_; lean_object* v_a_838_; lean_object* v___x_839_; lean_object* v___x_840_; lean_object* v___x_841_; size_t v___x_842_; size_t v___x_843_; 
v___x_837_ = lean_st_ref_take(v___y_833_);
v_a_838_ = lean_array_uget_borrowed(v_as_829_, v_i_831_);
v___x_839_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__4___redArg(v___x_837_, v_a_838_);
v___x_840_ = lean_st_ref_set(v___y_833_, v___x_839_);
v___x_841_ = lean_box(0);
v___x_842_ = ((size_t)1ULL);
v___x_843_ = lean_usize_add(v_i_831_, v___x_842_);
v_i_831_ = v___x_843_;
v_b_832_ = v___x_841_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__5___boxed(lean_object* v_as_845_, lean_object* v_sz_846_, lean_object* v_i_847_, lean_object* v_b_848_, lean_object* v___y_849_, lean_object* v___y_850_){
_start:
{
size_t v_sz_boxed_851_; size_t v_i_boxed_852_; lean_object* v_res_853_; 
v_sz_boxed_851_ = lean_unbox_usize(v_sz_846_);
lean_dec(v_sz_846_);
v_i_boxed_852_ = lean_unbox_usize(v_i_847_);
lean_dec(v_i_847_);
v_res_853_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__5(v_as_845_, v_sz_boxed_851_, v_i_boxed_852_, v_b_848_, v___y_849_);
lean_dec(v___y_849_);
lean_dec_ref(v_as_845_);
return v_res_853_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_beq___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__2(lean_object* v_x_854_, lean_object* v_x_855_){
_start:
{
if (lean_obj_tag(v_x_854_) == 0)
{
if (lean_obj_tag(v_x_855_) == 0)
{
uint8_t v___x_856_; 
v___x_856_ = 1;
return v___x_856_;
}
else
{
uint8_t v___x_857_; 
v___x_857_ = 0;
return v___x_857_;
}
}
else
{
if (lean_obj_tag(v_x_855_) == 0)
{
uint8_t v___x_858_; 
v___x_858_ = 0;
return v___x_858_;
}
else
{
lean_object* v_head_859_; lean_object* v_tail_860_; lean_object* v_head_861_; lean_object* v_tail_862_; uint8_t v___x_863_; 
v_head_859_ = lean_ctor_get(v_x_854_, 0);
v_tail_860_ = lean_ctor_get(v_x_854_, 1);
v_head_861_ = lean_ctor_get(v_x_855_, 0);
v_tail_862_ = lean_ctor_get(v_x_855_, 1);
v___x_863_ = lean_name_eq(v_head_859_, v_head_861_);
if (v___x_863_ == 0)
{
return v___x_863_;
}
else
{
v_x_854_ = v_tail_860_;
v_x_855_ = v_tail_862_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_beq___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__2___boxed(lean_object* v_x_865_, lean_object* v_x_866_){
_start:
{
uint8_t v_res_867_; lean_object* v_r_868_; 
v_res_867_ = lp_mathlib_List_beq___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__2(v_x_865_, v_x_866_);
lean_dec(v_x_866_);
lean_dec(v_x_865_);
v_r_868_ = lean_box(v_res_867_);
return v_r_868_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_beq___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__1(lean_object* v_x_869_, lean_object* v_x_870_){
_start:
{
if (lean_obj_tag(v_x_869_) == 0)
{
if (lean_obj_tag(v_x_870_) == 0)
{
uint8_t v___x_871_; 
v___x_871_ = 1;
return v___x_871_;
}
else
{
uint8_t v___x_872_; 
v___x_872_ = 0;
return v___x_872_;
}
}
else
{
if (lean_obj_tag(v_x_870_) == 0)
{
uint8_t v___x_873_; 
v___x_873_ = 0;
return v___x_873_;
}
else
{
lean_object* v_head_874_; lean_object* v_tail_875_; lean_object* v_head_876_; lean_object* v_tail_877_; uint8_t v___x_878_; 
v_head_874_ = lean_ctor_get(v_x_869_, 0);
v_tail_875_ = lean_ctor_get(v_x_869_, 1);
v_head_876_ = lean_ctor_get(v_x_870_, 0);
v_tail_877_ = lean_ctor_get(v_x_870_, 1);
v___x_878_ = l_Lean_instBEqMVarId_beq(v_head_874_, v_head_876_);
if (v___x_878_ == 0)
{
return v___x_878_;
}
else
{
v_x_869_ = v_tail_875_;
v_x_870_ = v_tail_877_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_beq___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__1___boxed(lean_object* v_x_880_, lean_object* v_x_881_){
_start:
{
uint8_t v_res_882_; lean_object* v_r_883_; 
v_res_882_ = lp_mathlib_List_beq___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__1(v_x_880_, v_x_881_);
lean_dec(v_x_881_);
lean_dec(v_x_880_);
v_r_883_ = lean_box(v_res_882_);
return v_r_883_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__0___redArg(lean_object* v_m_884_, lean_object* v_a_885_){
_start:
{
lean_object* v_buckets_886_; lean_object* v___x_887_; uint64_t v___y_889_; 
v_buckets_886_ = lean_ctor_get(v_m_884_, 1);
v___x_887_ = lean_array_get_size(v_buckets_886_);
if (lean_obj_tag(v_a_885_) == 0)
{
uint64_t v___x_903_; 
v___x_903_ = 1723ULL;
v___y_889_ = v___x_903_;
goto v___jp_888_;
}
else
{
uint64_t v_hash_904_; 
v_hash_904_ = lean_ctor_get_uint64(v_a_885_, sizeof(void*)*2);
v___y_889_ = v_hash_904_;
goto v___jp_888_;
}
v___jp_888_:
{
uint64_t v___x_890_; uint64_t v___x_891_; uint64_t v_fold_892_; uint64_t v___x_893_; uint64_t v___x_894_; uint64_t v___x_895_; size_t v___x_896_; size_t v___x_897_; size_t v___x_898_; size_t v___x_899_; size_t v___x_900_; lean_object* v___x_901_; uint8_t v___x_902_; 
v___x_890_ = 32ULL;
v___x_891_ = lean_uint64_shift_right(v___y_889_, v___x_890_);
v_fold_892_ = lean_uint64_xor(v___y_889_, v___x_891_);
v___x_893_ = 16ULL;
v___x_894_ = lean_uint64_shift_right(v_fold_892_, v___x_893_);
v___x_895_ = lean_uint64_xor(v_fold_892_, v___x_894_);
v___x_896_ = lean_uint64_to_usize(v___x_895_);
v___x_897_ = lean_usize_of_nat(v___x_887_);
v___x_898_ = ((size_t)1ULL);
v___x_899_ = lean_usize_sub(v___x_897_, v___x_898_);
v___x_900_ = lean_usize_land(v___x_896_, v___x_899_);
v___x_901_ = lean_array_uget_borrowed(v_buckets_886_, v___x_900_);
v___x_902_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__spec__0_spec__0_spec__1___redArg(v_a_885_, v___x_901_);
return v___x_902_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__0___redArg___boxed(lean_object* v_m_905_, lean_object* v_a_906_){
_start:
{
uint8_t v_res_907_; lean_object* v_r_908_; 
v_res_907_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__0___redArg(v_m_905_, v_a_906_);
lean_dec(v_a_906_);
lean_dec_ref(v_m_905_);
v_r_908_ = lean_box(v_res_907_);
return v_r_908_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__4___lam__0(lean_object* v_exceptions_914_, lean_object* v_x_915_, lean_object* v_i_916_, lean_object* v_ranges_917_){
_start:
{
if (lean_obj_tag(v_i_916_) == 0)
{
lean_object* v_i_918_; lean_object* v_toElabInfo_919_; lean_object* v_mctxBefore_920_; lean_object* v_goalsBefore_921_; lean_object* v_mctxAfter_922_; lean_object* v_goalsAfter_923_; lean_object* v_stx_924_; uint8_t v___x_925_; lean_object* v___x_926_; 
v_i_918_ = lean_ctor_get(v_i_916_, 0);
lean_inc_ref(v_i_918_);
lean_dec_ref_known(v_i_916_, 1);
v_toElabInfo_919_ = lean_ctor_get(v_i_918_, 0);
lean_inc_ref(v_toElabInfo_919_);
v_mctxBefore_920_ = lean_ctor_get(v_i_918_, 1);
lean_inc_ref(v_mctxBefore_920_);
v_goalsBefore_921_ = lean_ctor_get(v_i_918_, 2);
lean_inc(v_goalsBefore_921_);
v_mctxAfter_922_ = lean_ctor_get(v_i_918_, 3);
lean_inc_ref(v_mctxAfter_922_);
v_goalsAfter_923_ = lean_ctor_get(v_i_918_, 4);
lean_inc(v_goalsAfter_923_);
lean_dec_ref(v_i_918_);
v_stx_924_ = lean_ctor_get(v_toElabInfo_919_, 1);
lean_inc(v_stx_924_);
lean_dec_ref(v_toElabInfo_919_);
v___x_925_ = 1;
v___x_926_ = l_Lean_Syntax_getRange_x3f(v_stx_924_, v___x_925_);
if (lean_obj_tag(v___x_926_) == 1)
{
lean_object* v_val_927_; uint8_t v___y_929_; lean_object* v_kind_931_; uint8_t v___x_932_; 
v_val_927_ = lean_ctor_get(v___x_926_, 0);
lean_inc(v_val_927_);
lean_dec_ref_known(v___x_926_, 1);
v_kind_931_ = l_Lean_Syntax_getKind(v_stx_924_);
v___x_932_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__0___redArg(v_exceptions_914_, v_kind_931_);
if (v___x_932_ == 0)
{
uint8_t v___x_933_; 
v___x_933_ = lp_mathlib_List_beq___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__1(v_goalsAfter_923_, v_goalsBefore_921_);
lean_dec(v_goalsBefore_921_);
lean_dec(v_goalsAfter_923_);
if (v___x_933_ == 0)
{
lean_object* v___x_934_; 
lean_dec(v_kind_931_);
lean_dec_ref(v_mctxAfter_922_);
lean_dec_ref(v_mctxBefore_920_);
v___x_934_ = lean_array_push(v_ranges_917_, v_val_927_);
return v___x_934_;
}
else
{
if (v___x_932_ == 0)
{
lean_object* v___x_935_; uint8_t v___x_936_; 
v___x_935_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__4___lam__0___closed__1));
v___x_936_ = lean_name_eq(v_kind_931_, v___x_935_);
lean_dec(v_kind_931_);
if (v___x_936_ == 0)
{
lean_dec_ref(v_mctxAfter_922_);
lean_dec_ref(v_mctxBefore_920_);
v___y_929_ = v___x_936_;
goto v___jp_928_;
}
else
{
lean_object* v___x_937_; lean_object* v___x_938_; uint8_t v___x_939_; 
v___x_937_ = lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames(v_mctxBefore_920_);
lean_dec_ref(v_mctxBefore_920_);
v___x_938_ = lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getNames(v_mctxAfter_922_);
lean_dec_ref(v_mctxAfter_922_);
v___x_939_ = lp_mathlib_List_beq___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__2(v___x_937_, v___x_938_);
lean_dec(v___x_938_);
lean_dec(v___x_937_);
if (v___x_939_ == 0)
{
v___y_929_ = v___x_936_;
goto v___jp_928_;
}
else
{
lean_dec(v_val_927_);
return v_ranges_917_;
}
}
}
else
{
lean_object* v___x_940_; 
lean_dec(v_kind_931_);
lean_dec_ref(v_mctxAfter_922_);
lean_dec_ref(v_mctxBefore_920_);
v___x_940_ = lean_array_push(v_ranges_917_, v_val_927_);
return v___x_940_;
}
}
}
else
{
lean_object* v___x_941_; 
lean_dec(v_kind_931_);
lean_dec(v_goalsAfter_923_);
lean_dec_ref(v_mctxAfter_922_);
lean_dec(v_goalsBefore_921_);
lean_dec_ref(v_mctxBefore_920_);
v___x_941_ = lean_array_push(v_ranges_917_, v_val_927_);
return v___x_941_;
}
v___jp_928_:
{
if (v___y_929_ == 0)
{
lean_dec(v_val_927_);
return v_ranges_917_;
}
else
{
lean_object* v___x_930_; 
v___x_930_ = lean_array_push(v_ranges_917_, v_val_927_);
return v___x_930_;
}
}
}
else
{
lean_dec(v___x_926_);
lean_dec(v_stx_924_);
lean_dec(v_goalsAfter_923_);
lean_dec_ref(v_mctxAfter_922_);
lean_dec(v_goalsBefore_921_);
lean_dec_ref(v_mctxBefore_920_);
return v_ranges_917_;
}
}
else
{
lean_dec_ref(v_i_916_);
return v_ranges_917_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__4___lam__0___boxed(lean_object* v_exceptions_942_, lean_object* v_x_943_, lean_object* v_i_944_, lean_object* v_ranges_945_){
_start:
{
lean_object* v_res_946_; 
v_res_946_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__4___lam__0(v_exceptions_942_, v_x_943_, v_i_944_, v_ranges_945_);
lean_dec_ref(v_x_943_);
lean_dec_ref(v_exceptions_942_);
return v_res_946_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__4(lean_object* v_exceptions_947_, lean_object* v_as_948_, size_t v_i_949_, size_t v_stop_950_, lean_object* v_b_951_){
_start:
{
uint8_t v___x_952_; 
v___x_952_ = lean_usize_dec_eq(v_i_949_, v_stop_950_);
if (v___x_952_ == 0)
{
lean_object* v___f_953_; lean_object* v___x_954_; lean_object* v___x_955_; size_t v___x_956_; size_t v___x_957_; 
lean_inc_ref(v_exceptions_947_);
v___f_953_ = lean_alloc_closure((void*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__4___lam__0___boxed), 4, 1);
lean_closure_set(v___f_953_, 0, v_exceptions_947_);
v___x_954_ = lean_array_uget_borrowed(v_as_948_, v_i_949_);
lean_inc(v___x_954_);
v___x_955_ = l_Lean_Elab_InfoTree_foldInfo___redArg(v___f_953_, v_b_951_, v___x_954_);
v___x_956_ = ((size_t)1ULL);
v___x_957_ = lean_usize_add(v_i_949_, v___x_956_);
v_i_949_ = v___x_957_;
v_b_951_ = v___x_955_;
goto _start;
}
else
{
lean_dec_ref(v_exceptions_947_);
return v_b_951_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__4___boxed(lean_object* v_exceptions_959_, lean_object* v_as_960_, lean_object* v_i_961_, lean_object* v_stop_962_, lean_object* v_b_963_){
_start:
{
size_t v_i_boxed_964_; size_t v_stop_boxed_965_; lean_object* v_res_966_; 
v_i_boxed_964_ = lean_unbox_usize(v_i_961_);
lean_dec(v_i_961_);
v_stop_boxed_965_ = lean_unbox_usize(v_stop_962_);
lean_dec(v_stop_962_);
v_res_966_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__4(v_exceptions_959_, v_as_960_, v_i_boxed_964_, v_stop_boxed_965_, v_b_963_);
lean_dec_ref(v_as_960_);
return v_res_966_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__5(lean_object* v_exceptions_967_, lean_object* v_x_968_, lean_object* v_x_969_){
_start:
{
if (lean_obj_tag(v_x_968_) == 0)
{
lean_object* v_cs_970_; lean_object* v___x_971_; lean_object* v___x_972_; uint8_t v___x_973_; 
v_cs_970_ = lean_ctor_get(v_x_968_, 0);
v___x_971_ = lean_unsigned_to_nat(0u);
v___x_972_ = lean_array_get_size(v_cs_970_);
v___x_973_ = lean_nat_dec_lt(v___x_971_, v___x_972_);
if (v___x_973_ == 0)
{
lean_dec_ref(v_exceptions_967_);
return v_x_969_;
}
else
{
uint8_t v___x_974_; 
v___x_974_ = lean_nat_dec_le(v___x_972_, v___x_972_);
if (v___x_974_ == 0)
{
if (v___x_973_ == 0)
{
lean_dec_ref(v_exceptions_967_);
return v_x_969_;
}
else
{
size_t v___x_975_; size_t v___x_976_; lean_object* v___x_977_; 
v___x_975_ = ((size_t)0ULL);
v___x_976_ = lean_usize_of_nat(v___x_972_);
v___x_977_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__3_spec__4(v_exceptions_967_, v_cs_970_, v___x_975_, v___x_976_, v_x_969_);
return v___x_977_;
}
}
else
{
size_t v___x_978_; size_t v___x_979_; lean_object* v___x_980_; 
v___x_978_ = ((size_t)0ULL);
v___x_979_ = lean_usize_of_nat(v___x_972_);
v___x_980_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__3_spec__4(v_exceptions_967_, v_cs_970_, v___x_978_, v___x_979_, v_x_969_);
return v___x_980_;
}
}
}
else
{
lean_object* v_vs_981_; lean_object* v___x_982_; lean_object* v___x_983_; uint8_t v___x_984_; 
v_vs_981_ = lean_ctor_get(v_x_968_, 0);
v___x_982_ = lean_unsigned_to_nat(0u);
v___x_983_ = lean_array_get_size(v_vs_981_);
v___x_984_ = lean_nat_dec_lt(v___x_982_, v___x_983_);
if (v___x_984_ == 0)
{
lean_dec_ref(v_exceptions_967_);
return v_x_969_;
}
else
{
uint8_t v___x_985_; 
v___x_985_ = lean_nat_dec_le(v___x_983_, v___x_983_);
if (v___x_985_ == 0)
{
if (v___x_984_ == 0)
{
lean_dec_ref(v_exceptions_967_);
return v_x_969_;
}
else
{
size_t v___x_986_; size_t v___x_987_; lean_object* v___x_988_; 
v___x_986_ = ((size_t)0ULL);
v___x_987_ = lean_usize_of_nat(v___x_983_);
v___x_988_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__4(v_exceptions_967_, v_vs_981_, v___x_986_, v___x_987_, v_x_969_);
return v___x_988_;
}
}
else
{
size_t v___x_989_; size_t v___x_990_; lean_object* v___x_991_; 
v___x_989_ = ((size_t)0ULL);
v___x_990_ = lean_usize_of_nat(v___x_983_);
v___x_991_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__4(v_exceptions_967_, v_vs_981_, v___x_989_, v___x_990_, v_x_969_);
return v___x_991_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__3_spec__4(lean_object* v_exceptions_992_, lean_object* v_as_993_, size_t v_i_994_, size_t v_stop_995_, lean_object* v_b_996_){
_start:
{
uint8_t v___x_997_; 
v___x_997_ = lean_usize_dec_eq(v_i_994_, v_stop_995_);
if (v___x_997_ == 0)
{
lean_object* v___x_998_; lean_object* v___x_999_; size_t v___x_1000_; size_t v___x_1001_; 
v___x_998_ = lean_array_uget_borrowed(v_as_993_, v_i_994_);
lean_inc_ref(v_exceptions_992_);
v___x_999_ = lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__5(v_exceptions_992_, v___x_998_, v_b_996_);
v___x_1000_ = ((size_t)1ULL);
v___x_1001_ = lean_usize_add(v_i_994_, v___x_1000_);
v_i_994_ = v___x_1001_;
v_b_996_ = v___x_999_;
goto _start;
}
else
{
lean_dec_ref(v_exceptions_992_);
return v_b_996_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__3_spec__4___boxed(lean_object* v_exceptions_1003_, lean_object* v_as_1004_, lean_object* v_i_1005_, lean_object* v_stop_1006_, lean_object* v_b_1007_){
_start:
{
size_t v_i_boxed_1008_; size_t v_stop_boxed_1009_; lean_object* v_res_1010_; 
v_i_boxed_1008_ = lean_unbox_usize(v_i_1005_);
lean_dec(v_i_1005_);
v_stop_boxed_1009_ = lean_unbox_usize(v_stop_1006_);
lean_dec(v_stop_1006_);
v_res_1010_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__3_spec__4(v_exceptions_1003_, v_as_1004_, v_i_boxed_1008_, v_stop_boxed_1009_, v_b_1007_);
lean_dec_ref(v_as_1004_);
return v_res_1010_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__5___boxed(lean_object* v_exceptions_1011_, lean_object* v_x_1012_, lean_object* v_x_1013_){
_start:
{
lean_object* v_res_1014_; 
v_res_1014_ = lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__5(v_exceptions_1011_, v_x_1012_, v_x_1013_);
lean_dec_ref(v_x_1012_);
return v_res_1014_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__3___closed__0(void){
_start:
{
lean_object* v___x_1015_; 
v___x_1015_ = l_Lean_instInhabitedPersistentArrayNode_default(lean_box(0));
return v___x_1015_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__3(lean_object* v_exceptions_1016_, lean_object* v_x_1017_, size_t v_x_1018_, size_t v_x_1019_, lean_object* v_x_1020_){
_start:
{
if (lean_obj_tag(v_x_1017_) == 0)
{
lean_object* v_cs_1021_; lean_object* v___x_1022_; size_t v___x_1023_; lean_object* v_j_1024_; lean_object* v___x_1025_; size_t v___x_1026_; size_t v___x_1027_; size_t v___x_1028_; size_t v___x_1029_; size_t v___x_1030_; size_t v___x_1031_; lean_object* v___x_1032_; lean_object* v___x_1033_; lean_object* v___x_1034_; lean_object* v___x_1035_; uint8_t v___x_1036_; 
v_cs_1021_ = lean_ctor_get(v_x_1017_, 0);
v___x_1022_ = lean_obj_once(&lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__3___closed__0, &lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__3___closed__0_once, _init_lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__3___closed__0);
v___x_1023_ = lean_usize_shift_right(v_x_1018_, v_x_1019_);
v_j_1024_ = lean_usize_to_nat(v___x_1023_);
v___x_1025_ = lean_array_get_borrowed(v___x_1022_, v_cs_1021_, v_j_1024_);
v___x_1026_ = ((size_t)1ULL);
v___x_1027_ = lean_usize_shift_left(v___x_1026_, v_x_1019_);
v___x_1028_ = lean_usize_sub(v___x_1027_, v___x_1026_);
v___x_1029_ = lean_usize_land(v_x_1018_, v___x_1028_);
v___x_1030_ = ((size_t)5ULL);
v___x_1031_ = lean_usize_sub(v_x_1019_, v___x_1030_);
lean_inc_ref(v_exceptions_1016_);
v___x_1032_ = lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__3(v_exceptions_1016_, v___x_1025_, v___x_1029_, v___x_1031_, v_x_1020_);
v___x_1033_ = lean_unsigned_to_nat(1u);
v___x_1034_ = lean_nat_add(v_j_1024_, v___x_1033_);
lean_dec(v_j_1024_);
v___x_1035_ = lean_array_get_size(v_cs_1021_);
v___x_1036_ = lean_nat_dec_lt(v___x_1034_, v___x_1035_);
if (v___x_1036_ == 0)
{
lean_dec(v___x_1034_);
lean_dec_ref(v_exceptions_1016_);
return v___x_1032_;
}
else
{
uint8_t v___x_1037_; 
v___x_1037_ = lean_nat_dec_le(v___x_1035_, v___x_1035_);
if (v___x_1037_ == 0)
{
if (v___x_1036_ == 0)
{
lean_dec(v___x_1034_);
lean_dec_ref(v_exceptions_1016_);
return v___x_1032_;
}
else
{
size_t v___x_1038_; size_t v___x_1039_; lean_object* v___x_1040_; 
v___x_1038_ = lean_usize_of_nat(v___x_1034_);
lean_dec(v___x_1034_);
v___x_1039_ = lean_usize_of_nat(v___x_1035_);
v___x_1040_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__3_spec__4(v_exceptions_1016_, v_cs_1021_, v___x_1038_, v___x_1039_, v___x_1032_);
return v___x_1040_;
}
}
else
{
size_t v___x_1041_; size_t v___x_1042_; lean_object* v___x_1043_; 
v___x_1041_ = lean_usize_of_nat(v___x_1034_);
lean_dec(v___x_1034_);
v___x_1042_ = lean_usize_of_nat(v___x_1035_);
v___x_1043_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__3_spec__4(v_exceptions_1016_, v_cs_1021_, v___x_1041_, v___x_1042_, v___x_1032_);
return v___x_1043_;
}
}
}
else
{
lean_object* v_vs_1044_; lean_object* v___x_1045_; lean_object* v___x_1046_; uint8_t v___x_1047_; 
v_vs_1044_ = lean_ctor_get(v_x_1017_, 0);
v___x_1045_ = lean_usize_to_nat(v_x_1018_);
v___x_1046_ = lean_array_get_size(v_vs_1044_);
v___x_1047_ = lean_nat_dec_lt(v___x_1045_, v___x_1046_);
if (v___x_1047_ == 0)
{
lean_dec(v___x_1045_);
lean_dec_ref(v_exceptions_1016_);
return v_x_1020_;
}
else
{
uint8_t v___x_1048_; 
v___x_1048_ = lean_nat_dec_le(v___x_1046_, v___x_1046_);
if (v___x_1048_ == 0)
{
if (v___x_1047_ == 0)
{
lean_dec(v___x_1045_);
lean_dec_ref(v_exceptions_1016_);
return v_x_1020_;
}
else
{
size_t v___x_1049_; size_t v___x_1050_; lean_object* v___x_1051_; 
v___x_1049_ = lean_usize_of_nat(v___x_1045_);
lean_dec(v___x_1045_);
v___x_1050_ = lean_usize_of_nat(v___x_1046_);
v___x_1051_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__4(v_exceptions_1016_, v_vs_1044_, v___x_1049_, v___x_1050_, v_x_1020_);
return v___x_1051_;
}
}
else
{
size_t v___x_1052_; size_t v___x_1053_; lean_object* v___x_1054_; 
v___x_1052_ = lean_usize_of_nat(v___x_1045_);
lean_dec(v___x_1045_);
v___x_1053_ = lean_usize_of_nat(v___x_1046_);
v___x_1054_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__4(v_exceptions_1016_, v_vs_1044_, v___x_1052_, v___x_1053_, v_x_1020_);
return v___x_1054_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__3___boxed(lean_object* v_exceptions_1055_, lean_object* v_x_1056_, lean_object* v_x_1057_, lean_object* v_x_1058_, lean_object* v_x_1059_){
_start:
{
size_t v_x_3566__boxed_1060_; size_t v_x_3567__boxed_1061_; lean_object* v_res_1062_; 
v_x_3566__boxed_1060_ = lean_unbox_usize(v_x_1057_);
lean_dec(v_x_1057_);
v_x_3567__boxed_1061_ = lean_unbox_usize(v_x_1058_);
lean_dec(v_x_1058_);
v_res_1062_ = lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__3(v_exceptions_1055_, v_x_1056_, v_x_3566__boxed_1060_, v_x_3567__boxed_1061_, v_x_1059_);
lean_dec_ref(v_x_1056_);
return v_res_1062_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3(lean_object* v_exceptions_1063_, lean_object* v_t_1064_, lean_object* v_init_1065_, lean_object* v_start_1066_){
_start:
{
lean_object* v___x_1067_; uint8_t v___x_1068_; 
v___x_1067_ = lean_unsigned_to_nat(0u);
v___x_1068_ = lean_nat_dec_eq(v_start_1066_, v___x_1067_);
if (v___x_1068_ == 0)
{
lean_object* v_root_1069_; lean_object* v_tail_1070_; size_t v_shift_1071_; lean_object* v_tailOff_1072_; uint8_t v___x_1073_; 
v_root_1069_ = lean_ctor_get(v_t_1064_, 0);
v_tail_1070_ = lean_ctor_get(v_t_1064_, 1);
v_shift_1071_ = lean_ctor_get_usize(v_t_1064_, 4);
v_tailOff_1072_ = lean_ctor_get(v_t_1064_, 3);
v___x_1073_ = lean_nat_dec_le(v_tailOff_1072_, v_start_1066_);
if (v___x_1073_ == 0)
{
size_t v___x_1074_; lean_object* v___x_1075_; lean_object* v___x_1076_; uint8_t v___x_1077_; 
v___x_1074_ = lean_usize_of_nat(v_start_1066_);
lean_inc_ref(v_exceptions_1063_);
v___x_1075_ = lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__3(v_exceptions_1063_, v_root_1069_, v___x_1074_, v_shift_1071_, v_init_1065_);
v___x_1076_ = lean_array_get_size(v_tail_1070_);
v___x_1077_ = lean_nat_dec_lt(v___x_1067_, v___x_1076_);
if (v___x_1077_ == 0)
{
lean_dec_ref(v_exceptions_1063_);
return v___x_1075_;
}
else
{
uint8_t v___x_1078_; 
v___x_1078_ = lean_nat_dec_le(v___x_1076_, v___x_1076_);
if (v___x_1078_ == 0)
{
if (v___x_1077_ == 0)
{
lean_dec_ref(v_exceptions_1063_);
return v___x_1075_;
}
else
{
size_t v___x_1079_; size_t v___x_1080_; lean_object* v___x_1081_; 
v___x_1079_ = ((size_t)0ULL);
v___x_1080_ = lean_usize_of_nat(v___x_1076_);
v___x_1081_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__4(v_exceptions_1063_, v_tail_1070_, v___x_1079_, v___x_1080_, v___x_1075_);
return v___x_1081_;
}
}
else
{
size_t v___x_1082_; size_t v___x_1083_; lean_object* v___x_1084_; 
v___x_1082_ = ((size_t)0ULL);
v___x_1083_ = lean_usize_of_nat(v___x_1076_);
v___x_1084_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__4(v_exceptions_1063_, v_tail_1070_, v___x_1082_, v___x_1083_, v___x_1075_);
return v___x_1084_;
}
}
}
else
{
lean_object* v___x_1085_; lean_object* v___x_1086_; uint8_t v___x_1087_; 
v___x_1085_ = lean_nat_sub(v_start_1066_, v_tailOff_1072_);
v___x_1086_ = lean_array_get_size(v_tail_1070_);
v___x_1087_ = lean_nat_dec_lt(v___x_1085_, v___x_1086_);
if (v___x_1087_ == 0)
{
lean_dec(v___x_1085_);
lean_dec_ref(v_exceptions_1063_);
return v_init_1065_;
}
else
{
uint8_t v___x_1088_; 
v___x_1088_ = lean_nat_dec_le(v___x_1086_, v___x_1086_);
if (v___x_1088_ == 0)
{
if (v___x_1087_ == 0)
{
lean_dec(v___x_1085_);
lean_dec_ref(v_exceptions_1063_);
return v_init_1065_;
}
else
{
size_t v___x_1089_; size_t v___x_1090_; lean_object* v___x_1091_; 
v___x_1089_ = lean_usize_of_nat(v___x_1085_);
lean_dec(v___x_1085_);
v___x_1090_ = lean_usize_of_nat(v___x_1086_);
v___x_1091_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__4(v_exceptions_1063_, v_tail_1070_, v___x_1089_, v___x_1090_, v_init_1065_);
return v___x_1091_;
}
}
else
{
size_t v___x_1092_; size_t v___x_1093_; lean_object* v___x_1094_; 
v___x_1092_ = lean_usize_of_nat(v___x_1085_);
lean_dec(v___x_1085_);
v___x_1093_ = lean_usize_of_nat(v___x_1086_);
v___x_1094_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__4(v_exceptions_1063_, v_tail_1070_, v___x_1092_, v___x_1093_, v_init_1065_);
return v___x_1094_;
}
}
}
}
else
{
lean_object* v_root_1095_; lean_object* v_tail_1096_; lean_object* v___x_1097_; lean_object* v___x_1098_; uint8_t v___x_1099_; 
v_root_1095_ = lean_ctor_get(v_t_1064_, 0);
v_tail_1096_ = lean_ctor_get(v_t_1064_, 1);
lean_inc_ref(v_exceptions_1063_);
v___x_1097_ = lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__5(v_exceptions_1063_, v_root_1095_, v_init_1065_);
v___x_1098_ = lean_array_get_size(v_tail_1096_);
v___x_1099_ = lean_nat_dec_lt(v___x_1067_, v___x_1098_);
if (v___x_1099_ == 0)
{
lean_dec_ref(v_exceptions_1063_);
return v___x_1097_;
}
else
{
uint8_t v___x_1100_; 
v___x_1100_ = lean_nat_dec_le(v___x_1098_, v___x_1098_);
if (v___x_1100_ == 0)
{
if (v___x_1099_ == 0)
{
lean_dec_ref(v_exceptions_1063_);
return v___x_1097_;
}
else
{
size_t v___x_1101_; size_t v___x_1102_; lean_object* v___x_1103_; 
v___x_1101_ = ((size_t)0ULL);
v___x_1102_ = lean_usize_of_nat(v___x_1098_);
v___x_1103_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__4(v_exceptions_1063_, v_tail_1096_, v___x_1101_, v___x_1102_, v___x_1097_);
return v___x_1103_;
}
}
else
{
size_t v___x_1104_; size_t v___x_1105_; lean_object* v___x_1106_; 
v___x_1104_ = ((size_t)0ULL);
v___x_1105_ = lean_usize_of_nat(v___x_1098_);
v___x_1106_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3_spec__4(v_exceptions_1063_, v_tail_1096_, v___x_1104_, v___x_1105_, v___x_1097_);
return v___x_1106_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3___boxed(lean_object* v_exceptions_1107_, lean_object* v_t_1108_, lean_object* v_init_1109_, lean_object* v_start_1110_){
_start:
{
lean_object* v_res_1111_; 
v_res_1111_ = lp_mathlib_Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3(v_exceptions_1107_, v_t_1108_, v_init_1109_, v_start_1110_);
lean_dec(v_start_1110_);
lean_dec_ref(v_t_1108_);
return v_res_1111_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics(lean_object* v_exceptions_1114_, lean_object* v_trees_1115_, lean_object* v_a_1116_){
_start:
{
lean_object* v___x_1118_; lean_object* v___x_1119_; lean_object* v_ranges_1120_; lean_object* v___x_1121_; size_t v_sz_1122_; size_t v___x_1123_; lean_object* v___x_1124_; 
v___x_1118_ = lean_unsigned_to_nat(0u);
v___x_1119_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics___closed__0));
v_ranges_1120_ = lp_mathlib_Lean_PersistentArray_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__3(v_exceptions_1114_, v_trees_1115_, v___x_1119_, v___x_1118_);
v___x_1121_ = lean_box(0);
v_sz_1122_ = lean_array_size(v_ranges_1120_);
v___x_1123_ = ((size_t)0ULL);
v___x_1124_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__5(v_ranges_1120_, v_sz_1122_, v___x_1123_, v___x_1121_, v_a_1116_);
lean_dec_ref(v_ranges_1120_);
if (lean_obj_tag(v___x_1124_) == 0)
{
lean_object* v___x_1126_; uint8_t v_isShared_1127_; uint8_t v_isSharedCheck_1131_; 
v_isSharedCheck_1131_ = !lean_is_exclusive(v___x_1124_);
if (v_isSharedCheck_1131_ == 0)
{
lean_object* v_unused_1132_; 
v_unused_1132_ = lean_ctor_get(v___x_1124_, 0);
lean_dec(v_unused_1132_);
v___x_1126_ = v___x_1124_;
v_isShared_1127_ = v_isSharedCheck_1131_;
goto v_resetjp_1125_;
}
else
{
lean_dec(v___x_1124_);
v___x_1126_ = lean_box(0);
v_isShared_1127_ = v_isSharedCheck_1131_;
goto v_resetjp_1125_;
}
v_resetjp_1125_:
{
lean_object* v___x_1129_; 
if (v_isShared_1127_ == 0)
{
lean_ctor_set(v___x_1126_, 0, v___x_1121_);
v___x_1129_ = v___x_1126_;
goto v_reusejp_1128_;
}
else
{
lean_object* v_reuseFailAlloc_1130_; 
v_reuseFailAlloc_1130_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1130_, 0, v___x_1121_);
v___x_1129_ = v_reuseFailAlloc_1130_;
goto v_reusejp_1128_;
}
v_reusejp_1128_:
{
return v___x_1129_;
}
}
}
else
{
return v___x_1124_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics___boxed(lean_object* v_exceptions_1133_, lean_object* v_trees_1134_, lean_object* v_a_1135_, lean_object* v_a_1136_){
_start:
{
lean_object* v_res_1137_; 
v_res_1137_ = lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics(v_exceptions_1133_, v_trees_1134_, v_a_1135_);
lean_dec(v_a_1135_);
lean_dec_ref(v_trees_1134_);
return v_res_1137_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__0(lean_object* v_00_u03b2_1138_, lean_object* v_m_1139_, lean_object* v_a_1140_){
_start:
{
uint8_t v___x_1141_; 
v___x_1141_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__0___redArg(v_m_1139_, v_a_1140_);
return v___x_1141_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__0___boxed(lean_object* v_00_u03b2_1142_, lean_object* v_m_1143_, lean_object* v_a_1144_){
_start:
{
uint8_t v_res_1145_; lean_object* v_r_1146_; 
v_res_1145_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__0(v_00_u03b2_1142_, v_m_1143_, v_a_1144_);
lean_dec(v_a_1144_);
lean_dec_ref(v_m_1143_);
v_r_1146_ = lean_box(v_res_1145_);
return v_r_1146_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__4(lean_object* v_00_u03b2_1147_, lean_object* v_m_1148_, lean_object* v_a_1149_){
_start:
{
lean_object* v___x_1150_; 
v___x_1150_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__4___redArg(v_m_1148_, v_a_1149_);
return v___x_1150_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__4___boxed(lean_object* v_00_u03b2_1151_, lean_object* v_m_1152_, lean_object* v_a_1153_){
_start:
{
lean_object* v_res_1154_; 
v_res_1154_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__4(v_00_u03b2_1151_, v_m_1152_, v_a_1153_);
lean_dec_ref(v_a_1153_);
return v_res_1154_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__4_spec__7(lean_object* v_00_u03b2_1155_, lean_object* v_a_1156_, lean_object* v_x_1157_){
_start:
{
uint8_t v___x_1158_; 
v___x_1158_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__4_spec__7___redArg(v_a_1156_, v_x_1157_);
return v___x_1158_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__4_spec__7___boxed(lean_object* v_00_u03b2_1159_, lean_object* v_a_1160_, lean_object* v_x_1161_){
_start:
{
uint8_t v_res_1162_; lean_object* v_r_1163_; 
v_res_1162_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__4_spec__7(v_00_u03b2_1159_, v_a_1160_, v_x_1161_);
lean_dec(v_x_1161_);
lean_dec_ref(v_a_1160_);
v_r_1163_ = lean_box(v_res_1162_);
return v_r_1163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__4_spec__8(lean_object* v_00_u03b2_1164_, lean_object* v_a_1165_, lean_object* v_x_1166_){
_start:
{
lean_object* v___x_1167_; 
v___x_1167_ = lp_mathlib_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__4_spec__8___redArg(v_a_1165_, v_x_1166_);
return v___x_1167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__4_spec__8___boxed(lean_object* v_00_u03b2_1168_, lean_object* v_a_1169_, lean_object* v_x_1170_){
_start:
{
lean_object* v_res_1171_; 
v_res_1171_ = lp_mathlib_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__4_spec__8(v_00_u03b2_1168_, v_a_1169_, v_x_1170_);
lean_dec_ref(v_a_1169_);
return v_res_1171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__0(lean_object* v_a_1172_){
_start:
{
lean_object* v___x_1173_; 
v___x_1173_ = lean_nat_to_int(v_a_1172_);
return v___x_1173_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getInfoTrees___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__3___redArg(lean_object* v___y_1174_){
_start:
{
lean_object* v___x_1176_; lean_object* v_infoState_1177_; lean_object* v_trees_1178_; lean_object* v___x_1179_; 
v___x_1176_ = lean_st_ref_get(v___y_1174_);
v_infoState_1177_ = lean_ctor_get(v___x_1176_, 8);
lean_inc_ref(v_infoState_1177_);
lean_dec(v___x_1176_);
v_trees_1178_ = lean_ctor_get(v_infoState_1177_, 2);
lean_inc_ref(v_trees_1178_);
lean_dec_ref(v_infoState_1177_);
v___x_1179_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1179_, 0, v_trees_1178_);
return v___x_1179_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getInfoTrees___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__3___redArg___boxed(lean_object* v___y_1180_, lean_object* v___y_1181_){
_start:
{
lean_object* v_res_1182_; 
v_res_1182_ = lp_mathlib_Lean_Elab_getInfoTrees___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__3___redArg(v___y_1180_);
lean_dec(v___y_1180_);
return v_res_1182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getInfoTrees___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__3(lean_object* v___y_1183_, lean_object* v___y_1184_){
_start:
{
lean_object* v___x_1186_; 
v___x_1186_ = lp_mathlib_Lean_Elab_getInfoTrees___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__3___redArg(v___y_1184_);
return v___x_1186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getInfoTrees___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__3___boxed(lean_object* v___y_1187_, lean_object* v___y_1188_, lean_object* v___y_1189_){
_start:
{
lean_object* v_res_1190_; 
v_res_1190_ = lp_mathlib_Lean_Elab_getInfoTrees___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__3(v___y_1187_, v___y_1188_);
lean_dec(v___y_1188_);
lean_dec_ref(v___y_1187_);
return v_res_1190_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__7___redArg___lam__0(lean_object* v_r_1191_){
_start:
{
lean_object* v_start_1192_; lean_object* v_stop_1193_; lean_object* v___x_1195_; uint8_t v_isShared_1196_; uint8_t v_isSharedCheck_1202_; 
v_start_1192_ = lean_ctor_get(v_r_1191_, 0);
v_stop_1193_ = lean_ctor_get(v_r_1191_, 1);
v_isSharedCheck_1202_ = !lean_is_exclusive(v_r_1191_);
if (v_isSharedCheck_1202_ == 0)
{
v___x_1195_ = v_r_1191_;
v_isShared_1196_ = v_isSharedCheck_1202_;
goto v_resetjp_1194_;
}
else
{
lean_inc(v_stop_1193_);
lean_inc(v_start_1192_);
lean_dec(v_r_1191_);
v___x_1195_ = lean_box(0);
v_isShared_1196_ = v_isSharedCheck_1202_;
goto v_resetjp_1194_;
}
v_resetjp_1194_:
{
lean_object* v___x_1197_; lean_object* v___x_1198_; lean_object* v___x_1200_; 
v___x_1197_ = lean_nat_to_int(v_stop_1193_);
v___x_1198_ = lean_int_neg(v___x_1197_);
lean_dec(v___x_1197_);
if (v_isShared_1196_ == 0)
{
lean_ctor_set(v___x_1195_, 1, v___x_1198_);
v___x_1200_ = v___x_1195_;
goto v_reusejp_1199_;
}
else
{
lean_object* v_reuseFailAlloc_1201_; 
v_reuseFailAlloc_1201_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1201_, 0, v_start_1192_);
lean_ctor_set(v_reuseFailAlloc_1201_, 1, v___x_1198_);
v___x_1200_ = v_reuseFailAlloc_1201_;
goto v_reusejp_1199_;
}
v_reusejp_1199_:
{
return v___x_1200_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__7_spec__11___redArg(lean_object* v_hi_1205_, lean_object* v_pivot_1206_, lean_object* v_as_1207_, lean_object* v_i_1208_, lean_object* v_k_1209_){
_start:
{
uint8_t v___x_1214_; 
v___x_1214_ = lean_nat_dec_lt(v_k_1209_, v_hi_1205_);
if (v___x_1214_ == 0)
{
lean_object* v___x_1215_; lean_object* v___x_1216_; 
lean_dec(v_k_1209_);
lean_dec_ref(v_pivot_1206_);
v___x_1215_ = lean_array_fswap(v_as_1207_, v_i_1208_, v_hi_1205_);
v___x_1216_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1216_, 0, v_i_1208_);
lean_ctor_set(v___x_1216_, 1, v___x_1215_);
return v___x_1216_;
}
else
{
lean_object* v___x_1217_; lean_object* v_fst_1218_; lean_object* v_fst_1219_; lean_object* v___f_1220_; lean_object* v___f_1221_; lean_object* v___x_1222_; lean_object* v___x_1223_; lean_object* v___x_15690__overap_1224_; lean_object* v___x_1225_; uint8_t v___x_1226_; 
v___x_1217_ = lean_array_fget_borrowed(v_as_1207_, v_k_1209_);
v_fst_1218_ = lean_ctor_get(v___x_1217_, 0);
v_fst_1219_ = lean_ctor_get(v_pivot_1206_, 0);
v___f_1220_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__7_spec__11___redArg___closed__0));
v___f_1221_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__7_spec__11___redArg___closed__1));
lean_inc(v_fst_1218_);
v___x_1222_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__7___redArg___lam__0(v_fst_1218_);
lean_inc(v_fst_1219_);
v___x_1223_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__7___redArg___lam__0(v_fst_1219_);
v___x_15690__overap_1224_ = l_lexOrd___redArg(v___f_1220_, v___f_1221_);
v___x_1225_ = lean_apply_2(v___x_15690__overap_1224_, v___x_1222_, v___x_1223_);
v___x_1226_ = lean_unbox(v___x_1225_);
if (v___x_1226_ == 0)
{
if (v___x_1214_ == 0)
{
goto v___jp_1210_;
}
else
{
lean_object* v___x_1227_; lean_object* v___x_1228_; lean_object* v___x_1229_; lean_object* v___x_1230_; 
v___x_1227_ = lean_array_fswap(v_as_1207_, v_i_1208_, v_k_1209_);
v___x_1228_ = lean_unsigned_to_nat(1u);
v___x_1229_ = lean_nat_add(v_i_1208_, v___x_1228_);
lean_dec(v_i_1208_);
v___x_1230_ = lean_nat_add(v_k_1209_, v___x_1228_);
lean_dec(v_k_1209_);
v_as_1207_ = v___x_1227_;
v_i_1208_ = v___x_1229_;
v_k_1209_ = v___x_1230_;
goto _start;
}
}
else
{
goto v___jp_1210_;
}
}
v___jp_1210_:
{
lean_object* v___x_1211_; lean_object* v___x_1212_; 
v___x_1211_ = lean_unsigned_to_nat(1u);
v___x_1212_ = lean_nat_add(v_k_1209_, v___x_1211_);
lean_dec(v_k_1209_);
v_k_1209_ = v___x_1212_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__7_spec__11___redArg___boxed(lean_object* v_hi_1232_, lean_object* v_pivot_1233_, lean_object* v_as_1234_, lean_object* v_i_1235_, lean_object* v_k_1236_){
_start:
{
lean_object* v_res_1237_; 
v_res_1237_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__7_spec__11___redArg(v_hi_1232_, v_pivot_1233_, v_as_1234_, v_i_1235_, v_k_1236_);
lean_dec(v_hi_1232_);
return v_res_1237_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__7___redArg___lam__1(lean_object* v___f_1238_, uint8_t v___x_1239_, lean_object* v_x1_1240_, lean_object* v_x2_1241_){
_start:
{
lean_object* v_fst_1242_; lean_object* v_fst_1243_; lean_object* v___f_1244_; lean_object* v___f_1245_; lean_object* v___x_1246_; lean_object* v___x_1247_; lean_object* v___x_15985__overap_1248_; lean_object* v___x_1249_; uint8_t v___x_1250_; 
v_fst_1242_ = lean_ctor_get(v_x1_1240_, 0);
lean_inc(v_fst_1242_);
lean_dec_ref(v_x1_1240_);
v_fst_1243_ = lean_ctor_get(v_x2_1241_, 0);
lean_inc(v_fst_1243_);
lean_dec_ref(v_x2_1241_);
v___f_1244_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__7_spec__11___redArg___closed__0));
v___f_1245_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__7_spec__11___redArg___closed__1));
lean_inc_ref(v___f_1238_);
v___x_1246_ = lean_apply_1(v___f_1238_, v_fst_1242_);
v___x_1247_ = lean_apply_1(v___f_1238_, v_fst_1243_);
v___x_15985__overap_1248_ = l_lexOrd___redArg(v___f_1244_, v___f_1245_);
v___x_1249_ = lean_apply_2(v___x_15985__overap_1248_, v___x_1246_, v___x_1247_);
v___x_1250_ = lean_unbox(v___x_1249_);
if (v___x_1250_ == 0)
{
return v___x_1239_;
}
else
{
uint8_t v___x_1251_; 
v___x_1251_ = 0;
return v___x_1251_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__7___redArg___lam__1___boxed(lean_object* v___f_1252_, lean_object* v___x_1253_, lean_object* v_x1_1254_, lean_object* v_x2_1255_){
_start:
{
uint8_t v___x_16086__boxed_1256_; uint8_t v_res_1257_; lean_object* v_r_1258_; 
v___x_16086__boxed_1256_ = lean_unbox(v___x_1253_);
v_res_1257_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__7___redArg___lam__1(v___f_1252_, v___x_16086__boxed_1256_, v_x1_1254_, v_x2_1255_);
v_r_1258_ = lean_box(v_res_1257_);
return v_r_1258_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__7___redArg(lean_object* v_n_1260_, lean_object* v_as_1261_, lean_object* v_lo_1262_, lean_object* v_hi_1263_){
_start:
{
lean_object* v___y_1265_; uint8_t v___x_1275_; 
v___x_1275_ = lean_nat_dec_lt(v_lo_1262_, v_hi_1263_);
if (v___x_1275_ == 0)
{
lean_dec(v_lo_1262_);
return v_as_1261_;
}
else
{
lean_object* v___f_1276_; lean_object* v___x_1277_; lean_object* v___x_1278_; lean_object* v_mid_1279_; lean_object* v___y_1281_; lean_object* v___y_1287_; lean_object* v___x_1292_; lean_object* v___x_1293_; uint8_t v___x_1294_; 
v___f_1276_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__7___redArg___closed__0));
v___x_1277_ = lean_nat_add(v_lo_1262_, v_hi_1263_);
v___x_1278_ = lean_unsigned_to_nat(1u);
v_mid_1279_ = lean_nat_shiftr(v___x_1277_, v___x_1278_);
lean_dec(v___x_1277_);
v___x_1292_ = lean_array_fget_borrowed(v_as_1261_, v_mid_1279_);
v___x_1293_ = lean_array_fget_borrowed(v_as_1261_, v_lo_1262_);
lean_inc(v___x_1293_);
lean_inc(v___x_1292_);
v___x_1294_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__7___redArg___lam__1(v___f_1276_, v___x_1275_, v___x_1292_, v___x_1293_);
if (v___x_1294_ == 0)
{
v___y_1287_ = v_as_1261_;
goto v___jp_1286_;
}
else
{
lean_object* v___x_1295_; 
v___x_1295_ = lean_array_fswap(v_as_1261_, v_lo_1262_, v_mid_1279_);
v___y_1287_ = v___x_1295_;
goto v___jp_1286_;
}
v___jp_1280_:
{
lean_object* v___x_1282_; lean_object* v___x_1283_; uint8_t v___x_1284_; 
v___x_1282_ = lean_array_fget_borrowed(v___y_1281_, v_mid_1279_);
v___x_1283_ = lean_array_fget_borrowed(v___y_1281_, v_hi_1263_);
lean_inc(v___x_1283_);
lean_inc(v___x_1282_);
v___x_1284_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__7___redArg___lam__1(v___f_1276_, v___x_1275_, v___x_1282_, v___x_1283_);
if (v___x_1284_ == 0)
{
lean_dec(v_mid_1279_);
v___y_1265_ = v___y_1281_;
goto v___jp_1264_;
}
else
{
lean_object* v___x_1285_; 
v___x_1285_ = lean_array_fswap(v___y_1281_, v_mid_1279_, v_hi_1263_);
lean_dec(v_mid_1279_);
v___y_1265_ = v___x_1285_;
goto v___jp_1264_;
}
}
v___jp_1286_:
{
lean_object* v___x_1288_; lean_object* v___x_1289_; uint8_t v___x_1290_; 
v___x_1288_ = lean_array_fget_borrowed(v___y_1287_, v_hi_1263_);
v___x_1289_ = lean_array_fget_borrowed(v___y_1287_, v_lo_1262_);
lean_inc(v___x_1289_);
lean_inc(v___x_1288_);
v___x_1290_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__7___redArg___lam__1(v___f_1276_, v___x_1275_, v___x_1288_, v___x_1289_);
if (v___x_1290_ == 0)
{
v___y_1281_ = v___y_1287_;
goto v___jp_1280_;
}
else
{
lean_object* v___x_1291_; 
v___x_1291_ = lean_array_fswap(v___y_1287_, v_lo_1262_, v_hi_1263_);
v___y_1281_ = v___x_1291_;
goto v___jp_1280_;
}
}
}
v___jp_1264_:
{
lean_object* v_pivot_1266_; lean_object* v___x_1267_; lean_object* v_fst_1268_; lean_object* v_snd_1269_; uint8_t v___x_1270_; 
v_pivot_1266_ = lean_array_fget(v___y_1265_, v_hi_1263_);
lean_inc_n(v_lo_1262_, 2);
v___x_1267_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__7_spec__11___redArg(v_hi_1263_, v_pivot_1266_, v___y_1265_, v_lo_1262_, v_lo_1262_);
v_fst_1268_ = lean_ctor_get(v___x_1267_, 0);
lean_inc(v_fst_1268_);
v_snd_1269_ = lean_ctor_get(v___x_1267_, 1);
lean_inc(v_snd_1269_);
lean_dec_ref(v___x_1267_);
v___x_1270_ = lean_nat_dec_le(v_hi_1263_, v_fst_1268_);
if (v___x_1270_ == 0)
{
lean_object* v___x_1271_; lean_object* v___x_1272_; lean_object* v___x_1273_; 
v___x_1271_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__7___redArg(v_n_1260_, v_snd_1269_, v_lo_1262_, v_fst_1268_);
v___x_1272_ = lean_unsigned_to_nat(1u);
v___x_1273_ = lean_nat_add(v_fst_1268_, v___x_1272_);
lean_dec(v_fst_1268_);
v_as_1261_ = v___x_1271_;
v_lo_1262_ = v___x_1273_;
goto _start;
}
else
{
lean_dec(v_fst_1268_);
lean_dec(v_lo_1262_);
return v_snd_1269_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__7___redArg___boxed(lean_object* v_n_1296_, lean_object* v_as_1297_, lean_object* v_lo_1298_, lean_object* v_hi_1299_){
_start:
{
lean_object* v_res_1300_; 
v_res_1300_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__7___redArg(v_n_1296_, v_as_1297_, v_lo_1298_, v_hi_1299_);
lean_dec(v_hi_1299_);
lean_dec(v_n_1296_);
return v_res_1300_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__8(lean_object* v_x_1301_, lean_object* v_x_1302_){
_start:
{
if (lean_obj_tag(v_x_1302_) == 0)
{
return v_x_1301_;
}
else
{
lean_object* v_key_1303_; lean_object* v_value_1304_; lean_object* v_tail_1305_; lean_object* v___x_1306_; lean_object* v___x_1307_; 
v_key_1303_ = lean_ctor_get(v_x_1302_, 0);
v_value_1304_ = lean_ctor_get(v_x_1302_, 1);
v_tail_1305_ = lean_ctor_get(v_x_1302_, 2);
lean_inc(v_value_1304_);
lean_inc(v_key_1303_);
v___x_1306_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1306_, 0, v_key_1303_);
lean_ctor_set(v___x_1306_, 1, v_value_1304_);
v___x_1307_ = lean_array_push(v_x_1301_, v___x_1306_);
v_x_1301_ = v___x_1307_;
v_x_1302_ = v_tail_1305_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__8___boxed(lean_object* v_x_1309_, lean_object* v_x_1310_){
_start:
{
lean_object* v_res_1311_; 
v_res_1311_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__8(v_x_1309_, v_x_1310_);
lean_dec(v_x_1310_);
return v_res_1311_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__9(lean_object* v_as_1312_, size_t v_i_1313_, size_t v_stop_1314_, lean_object* v_b_1315_){
_start:
{
uint8_t v___x_1316_; 
v___x_1316_ = lean_usize_dec_eq(v_i_1313_, v_stop_1314_);
if (v___x_1316_ == 0)
{
lean_object* v___x_1317_; lean_object* v___x_1318_; size_t v___x_1319_; size_t v___x_1320_; 
v___x_1317_ = lean_array_uget_borrowed(v_as_1312_, v_i_1313_);
v___x_1318_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__8(v_b_1315_, v___x_1317_);
v___x_1319_ = ((size_t)1ULL);
v___x_1320_ = lean_usize_add(v_i_1313_, v___x_1319_);
v_i_1313_ = v___x_1320_;
v_b_1315_ = v___x_1318_;
goto _start;
}
else
{
return v_b_1315_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__9___boxed(lean_object* v_as_1322_, lean_object* v_i_1323_, lean_object* v_stop_1324_, lean_object* v_b_1325_){
_start:
{
size_t v_i_boxed_1326_; size_t v_stop_boxed_1327_; lean_object* v_res_1328_; 
v_i_boxed_1326_ = lean_unbox_usize(v_i_1323_);
lean_dec(v_i_1323_);
v_stop_boxed_1327_ = lean_unbox_usize(v_stop_1324_);
lean_dec(v_stop_1324_);
v_res_1328_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__9(v_as_1322_, v_i_boxed_1326_, v_stop_boxed_1327_, v_b_1325_);
lean_dec_ref(v_as_1322_);
return v_res_1328_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__10_spec__15_spec__19_spec__25_spec__30___redArg(lean_object* v_x_1329_, lean_object* v_x_1330_){
_start:
{
if (lean_obj_tag(v_x_1330_) == 0)
{
return v_x_1329_;
}
else
{
lean_object* v_key_1331_; lean_object* v_value_1332_; lean_object* v_tail_1333_; lean_object* v___x_1335_; uint8_t v_isShared_1336_; uint8_t v_isSharedCheck_1356_; 
v_key_1331_ = lean_ctor_get(v_x_1330_, 0);
v_value_1332_ = lean_ctor_get(v_x_1330_, 1);
v_tail_1333_ = lean_ctor_get(v_x_1330_, 2);
v_isSharedCheck_1356_ = !lean_is_exclusive(v_x_1330_);
if (v_isSharedCheck_1356_ == 0)
{
v___x_1335_ = v_x_1330_;
v_isShared_1336_ = v_isSharedCheck_1356_;
goto v_resetjp_1334_;
}
else
{
lean_inc(v_tail_1333_);
lean_inc(v_value_1332_);
lean_inc(v_key_1331_);
lean_dec(v_x_1330_);
v___x_1335_ = lean_box(0);
v_isShared_1336_ = v_isSharedCheck_1356_;
goto v_resetjp_1334_;
}
v_resetjp_1334_:
{
lean_object* v___x_1337_; uint64_t v___x_1338_; uint64_t v___x_1339_; uint64_t v___x_1340_; uint64_t v_fold_1341_; uint64_t v___x_1342_; uint64_t v___x_1343_; uint64_t v___x_1344_; size_t v___x_1345_; size_t v___x_1346_; size_t v___x_1347_; size_t v___x_1348_; size_t v___x_1349_; lean_object* v___x_1350_; lean_object* v___x_1352_; 
v___x_1337_ = lean_array_get_size(v_x_1329_);
v___x_1338_ = l_Lean_Syntax_instHashableRange_hash(v_key_1331_);
v___x_1339_ = 32ULL;
v___x_1340_ = lean_uint64_shift_right(v___x_1338_, v___x_1339_);
v_fold_1341_ = lean_uint64_xor(v___x_1338_, v___x_1340_);
v___x_1342_ = 16ULL;
v___x_1343_ = lean_uint64_shift_right(v_fold_1341_, v___x_1342_);
v___x_1344_ = lean_uint64_xor(v_fold_1341_, v___x_1343_);
v___x_1345_ = lean_uint64_to_usize(v___x_1344_);
v___x_1346_ = lean_usize_of_nat(v___x_1337_);
v___x_1347_ = ((size_t)1ULL);
v___x_1348_ = lean_usize_sub(v___x_1346_, v___x_1347_);
v___x_1349_ = lean_usize_land(v___x_1345_, v___x_1348_);
v___x_1350_ = lean_array_uget_borrowed(v_x_1329_, v___x_1349_);
lean_inc(v___x_1350_);
if (v_isShared_1336_ == 0)
{
lean_ctor_set(v___x_1335_, 2, v___x_1350_);
v___x_1352_ = v___x_1335_;
goto v_reusejp_1351_;
}
else
{
lean_object* v_reuseFailAlloc_1355_; 
v_reuseFailAlloc_1355_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1355_, 0, v_key_1331_);
lean_ctor_set(v_reuseFailAlloc_1355_, 1, v_value_1332_);
lean_ctor_set(v_reuseFailAlloc_1355_, 2, v___x_1350_);
v___x_1352_ = v_reuseFailAlloc_1355_;
goto v_reusejp_1351_;
}
v_reusejp_1351_:
{
lean_object* v___x_1353_; 
v___x_1353_ = lean_array_uset(v_x_1329_, v___x_1349_, v___x_1352_);
v_x_1329_ = v___x_1353_;
v_x_1330_ = v_tail_1333_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__10_spec__15_spec__19_spec__25___redArg(lean_object* v_i_1357_, lean_object* v_source_1358_, lean_object* v_target_1359_){
_start:
{
lean_object* v___x_1360_; uint8_t v___x_1361_; 
v___x_1360_ = lean_array_get_size(v_source_1358_);
v___x_1361_ = lean_nat_dec_lt(v_i_1357_, v___x_1360_);
if (v___x_1361_ == 0)
{
lean_dec_ref(v_source_1358_);
lean_dec(v_i_1357_);
return v_target_1359_;
}
else
{
lean_object* v_es_1362_; lean_object* v___x_1363_; lean_object* v_source_1364_; lean_object* v_target_1365_; lean_object* v___x_1366_; lean_object* v___x_1367_; 
v_es_1362_ = lean_array_fget(v_source_1358_, v_i_1357_);
v___x_1363_ = lean_box(0);
v_source_1364_ = lean_array_fset(v_source_1358_, v_i_1357_, v___x_1363_);
v_target_1365_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__10_spec__15_spec__19_spec__25_spec__30___redArg(v_target_1359_, v_es_1362_);
v___x_1366_ = lean_unsigned_to_nat(1u);
v___x_1367_ = lean_nat_add(v_i_1357_, v___x_1366_);
lean_dec(v_i_1357_);
v_i_1357_ = v___x_1367_;
v_source_1358_ = v_source_1364_;
v_target_1359_ = v_target_1365_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__10_spec__15_spec__19___redArg(lean_object* v_data_1369_){
_start:
{
lean_object* v___x_1370_; lean_object* v___x_1371_; lean_object* v_nbuckets_1372_; lean_object* v___x_1373_; lean_object* v___x_1374_; lean_object* v___x_1375_; lean_object* v___x_1376_; 
v___x_1370_ = lean_array_get_size(v_data_1369_);
v___x_1371_ = lean_unsigned_to_nat(2u);
v_nbuckets_1372_ = lean_nat_mul(v___x_1370_, v___x_1371_);
v___x_1373_ = lean_unsigned_to_nat(0u);
v___x_1374_ = lean_box(0);
v___x_1375_ = lean_mk_array(v_nbuckets_1372_, v___x_1374_);
v___x_1376_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__10_spec__15_spec__19_spec__25___redArg(v___x_1373_, v_data_1369_, v___x_1375_);
return v___x_1376_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__10_spec__15_spec__20___redArg(lean_object* v_a_1377_, lean_object* v_b_1378_, lean_object* v_x_1379_){
_start:
{
if (lean_obj_tag(v_x_1379_) == 0)
{
lean_dec(v_b_1378_);
lean_dec_ref(v_a_1377_);
return v_x_1379_;
}
else
{
lean_object* v_key_1380_; lean_object* v_value_1381_; lean_object* v_tail_1382_; lean_object* v___x_1384_; uint8_t v_isShared_1385_; uint8_t v_isSharedCheck_1394_; 
v_key_1380_ = lean_ctor_get(v_x_1379_, 0);
v_value_1381_ = lean_ctor_get(v_x_1379_, 1);
v_tail_1382_ = lean_ctor_get(v_x_1379_, 2);
v_isSharedCheck_1394_ = !lean_is_exclusive(v_x_1379_);
if (v_isSharedCheck_1394_ == 0)
{
v___x_1384_ = v_x_1379_;
v_isShared_1385_ = v_isSharedCheck_1394_;
goto v_resetjp_1383_;
}
else
{
lean_inc(v_tail_1382_);
lean_inc(v_value_1381_);
lean_inc(v_key_1380_);
lean_dec(v_x_1379_);
v___x_1384_ = lean_box(0);
v_isShared_1385_ = v_isSharedCheck_1394_;
goto v_resetjp_1383_;
}
v_resetjp_1383_:
{
uint8_t v___x_1386_; 
v___x_1386_ = l_Lean_Syntax_instBEqRange_beq(v_key_1380_, v_a_1377_);
if (v___x_1386_ == 0)
{
lean_object* v___x_1387_; lean_object* v___x_1389_; 
v___x_1387_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__10_spec__15_spec__20___redArg(v_a_1377_, v_b_1378_, v_tail_1382_);
if (v_isShared_1385_ == 0)
{
lean_ctor_set(v___x_1384_, 2, v___x_1387_);
v___x_1389_ = v___x_1384_;
goto v_reusejp_1388_;
}
else
{
lean_object* v_reuseFailAlloc_1390_; 
v_reuseFailAlloc_1390_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1390_, 0, v_key_1380_);
lean_ctor_set(v_reuseFailAlloc_1390_, 1, v_value_1381_);
lean_ctor_set(v_reuseFailAlloc_1390_, 2, v___x_1387_);
v___x_1389_ = v_reuseFailAlloc_1390_;
goto v_reusejp_1388_;
}
v_reusejp_1388_:
{
return v___x_1389_;
}
}
else
{
lean_object* v___x_1392_; 
lean_dec(v_value_1381_);
lean_dec(v_key_1380_);
if (v_isShared_1385_ == 0)
{
lean_ctor_set(v___x_1384_, 1, v_b_1378_);
lean_ctor_set(v___x_1384_, 0, v_a_1377_);
v___x_1392_ = v___x_1384_;
goto v_reusejp_1391_;
}
else
{
lean_object* v_reuseFailAlloc_1393_; 
v_reuseFailAlloc_1393_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1393_, 0, v_a_1377_);
lean_ctor_set(v_reuseFailAlloc_1393_, 1, v_b_1378_);
lean_ctor_set(v_reuseFailAlloc_1393_, 2, v_tail_1382_);
v___x_1392_ = v_reuseFailAlloc_1393_;
goto v_reusejp_1391_;
}
v_reusejp_1391_:
{
return v___x_1392_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__10_spec__15___redArg(lean_object* v_m_1395_, lean_object* v_a_1396_, lean_object* v_b_1397_){
_start:
{
lean_object* v_size_1398_; lean_object* v_buckets_1399_; lean_object* v___x_1401_; uint8_t v_isShared_1402_; uint8_t v_isSharedCheck_1442_; 
v_size_1398_ = lean_ctor_get(v_m_1395_, 0);
v_buckets_1399_ = lean_ctor_get(v_m_1395_, 1);
v_isSharedCheck_1442_ = !lean_is_exclusive(v_m_1395_);
if (v_isSharedCheck_1442_ == 0)
{
v___x_1401_ = v_m_1395_;
v_isShared_1402_ = v_isSharedCheck_1442_;
goto v_resetjp_1400_;
}
else
{
lean_inc(v_buckets_1399_);
lean_inc(v_size_1398_);
lean_dec(v_m_1395_);
v___x_1401_ = lean_box(0);
v_isShared_1402_ = v_isSharedCheck_1442_;
goto v_resetjp_1400_;
}
v_resetjp_1400_:
{
lean_object* v___x_1403_; uint64_t v___x_1404_; uint64_t v___x_1405_; uint64_t v___x_1406_; uint64_t v_fold_1407_; uint64_t v___x_1408_; uint64_t v___x_1409_; uint64_t v___x_1410_; size_t v___x_1411_; size_t v___x_1412_; size_t v___x_1413_; size_t v___x_1414_; size_t v___x_1415_; lean_object* v_bkt_1416_; uint8_t v___x_1417_; 
v___x_1403_ = lean_array_get_size(v_buckets_1399_);
v___x_1404_ = l_Lean_Syntax_instHashableRange_hash(v_a_1396_);
v___x_1405_ = 32ULL;
v___x_1406_ = lean_uint64_shift_right(v___x_1404_, v___x_1405_);
v_fold_1407_ = lean_uint64_xor(v___x_1404_, v___x_1406_);
v___x_1408_ = 16ULL;
v___x_1409_ = lean_uint64_shift_right(v_fold_1407_, v___x_1408_);
v___x_1410_ = lean_uint64_xor(v_fold_1407_, v___x_1409_);
v___x_1411_ = lean_uint64_to_usize(v___x_1410_);
v___x_1412_ = lean_usize_of_nat(v___x_1403_);
v___x_1413_ = ((size_t)1ULL);
v___x_1414_ = lean_usize_sub(v___x_1412_, v___x_1413_);
v___x_1415_ = lean_usize_land(v___x_1411_, v___x_1414_);
v_bkt_1416_ = lean_array_uget_borrowed(v_buckets_1399_, v___x_1415_);
v___x_1417_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics_spec__4_spec__7___redArg(v_a_1396_, v_bkt_1416_);
if (v___x_1417_ == 0)
{
lean_object* v___x_1418_; lean_object* v_size_x27_1419_; lean_object* v___x_1420_; lean_object* v_buckets_x27_1421_; lean_object* v___x_1422_; lean_object* v___x_1423_; lean_object* v___x_1424_; lean_object* v___x_1425_; lean_object* v___x_1426_; uint8_t v___x_1427_; 
v___x_1418_ = lean_unsigned_to_nat(1u);
v_size_x27_1419_ = lean_nat_add(v_size_1398_, v___x_1418_);
lean_dec(v_size_1398_);
lean_inc(v_bkt_1416_);
v___x_1420_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1420_, 0, v_a_1396_);
lean_ctor_set(v___x_1420_, 1, v_b_1397_);
lean_ctor_set(v___x_1420_, 2, v_bkt_1416_);
v_buckets_x27_1421_ = lean_array_uset(v_buckets_1399_, v___x_1415_, v___x_1420_);
v___x_1422_ = lean_unsigned_to_nat(4u);
v___x_1423_ = lean_nat_mul(v_size_x27_1419_, v___x_1422_);
v___x_1424_ = lean_unsigned_to_nat(3u);
v___x_1425_ = lean_nat_div(v___x_1423_, v___x_1424_);
lean_dec(v___x_1423_);
v___x_1426_ = lean_array_get_size(v_buckets_x27_1421_);
v___x_1427_ = lean_nat_dec_le(v___x_1425_, v___x_1426_);
lean_dec(v___x_1425_);
if (v___x_1427_ == 0)
{
lean_object* v_val_1428_; lean_object* v___x_1430_; 
v_val_1428_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__10_spec__15_spec__19___redArg(v_buckets_x27_1421_);
if (v_isShared_1402_ == 0)
{
lean_ctor_set(v___x_1401_, 1, v_val_1428_);
lean_ctor_set(v___x_1401_, 0, v_size_x27_1419_);
v___x_1430_ = v___x_1401_;
goto v_reusejp_1429_;
}
else
{
lean_object* v_reuseFailAlloc_1431_; 
v_reuseFailAlloc_1431_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1431_, 0, v_size_x27_1419_);
lean_ctor_set(v_reuseFailAlloc_1431_, 1, v_val_1428_);
v___x_1430_ = v_reuseFailAlloc_1431_;
goto v_reusejp_1429_;
}
v_reusejp_1429_:
{
return v___x_1430_;
}
}
else
{
lean_object* v___x_1433_; 
if (v_isShared_1402_ == 0)
{
lean_ctor_set(v___x_1401_, 1, v_buckets_x27_1421_);
lean_ctor_set(v___x_1401_, 0, v_size_x27_1419_);
v___x_1433_ = v___x_1401_;
goto v_reusejp_1432_;
}
else
{
lean_object* v_reuseFailAlloc_1434_; 
v_reuseFailAlloc_1434_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1434_, 0, v_size_x27_1419_);
lean_ctor_set(v_reuseFailAlloc_1434_, 1, v_buckets_x27_1421_);
v___x_1433_ = v_reuseFailAlloc_1434_;
goto v_reusejp_1432_;
}
v_reusejp_1432_:
{
return v___x_1433_;
}
}
}
else
{
lean_object* v___x_1435_; lean_object* v_buckets_x27_1436_; lean_object* v___x_1437_; lean_object* v___x_1438_; lean_object* v___x_1440_; 
lean_inc(v_bkt_1416_);
v___x_1435_ = lean_box(0);
v_buckets_x27_1436_ = lean_array_uset(v_buckets_1399_, v___x_1415_, v___x_1435_);
v___x_1437_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__10_spec__15_spec__20___redArg(v_a_1396_, v_b_1397_, v_bkt_1416_);
v___x_1438_ = lean_array_uset(v_buckets_x27_1436_, v___x_1415_, v___x_1437_);
if (v_isShared_1402_ == 0)
{
lean_ctor_set(v___x_1401_, 1, v___x_1438_);
v___x_1440_ = v___x_1401_;
goto v_reusejp_1439_;
}
else
{
lean_object* v_reuseFailAlloc_1441_; 
v_reuseFailAlloc_1441_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1441_, 0, v_size_1398_);
lean_ctor_set(v_reuseFailAlloc_1441_, 1, v___x_1438_);
v___x_1440_ = v_reuseFailAlloc_1441_;
goto v_reusejp_1439_;
}
v_reusejp_1439_:
{
return v___x_1440_;
}
}
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__5_spec__8_spec__11___redArg(lean_object* v_keys_1443_, lean_object* v_i_1444_, lean_object* v_k_1445_){
_start:
{
lean_object* v___x_1446_; uint8_t v___x_1447_; 
v___x_1446_ = lean_array_get_size(v_keys_1443_);
v___x_1447_ = lean_nat_dec_lt(v_i_1444_, v___x_1446_);
if (v___x_1447_ == 0)
{
lean_dec(v_i_1444_);
return v___x_1447_;
}
else
{
lean_object* v_k_x27_1448_; uint8_t v___x_1449_; 
v_k_x27_1448_ = lean_array_fget_borrowed(v_keys_1443_, v_i_1444_);
v___x_1449_ = lean_name_eq(v_k_1445_, v_k_x27_1448_);
if (v___x_1449_ == 0)
{
lean_object* v___x_1450_; lean_object* v___x_1451_; 
v___x_1450_ = lean_unsigned_to_nat(1u);
v___x_1451_ = lean_nat_add(v_i_1444_, v___x_1450_);
lean_dec(v_i_1444_);
v_i_1444_ = v___x_1451_;
goto _start;
}
else
{
lean_dec(v_i_1444_);
return v___x_1449_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__5_spec__8_spec__11___redArg___boxed(lean_object* v_keys_1453_, lean_object* v_i_1454_, lean_object* v_k_1455_){
_start:
{
uint8_t v_res_1456_; lean_object* v_r_1457_; 
v_res_1456_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__5_spec__8_spec__11___redArg(v_keys_1453_, v_i_1454_, v_k_1455_);
lean_dec(v_k_1455_);
lean_dec_ref(v_keys_1453_);
v_r_1457_ = lean_box(v_res_1456_);
return v_r_1457_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__5_spec__8___redArg(lean_object* v_x_1458_, size_t v_x_1459_, lean_object* v_x_1460_){
_start:
{
if (lean_obj_tag(v_x_1458_) == 0)
{
lean_object* v_es_1461_; lean_object* v___x_1462_; size_t v___x_1463_; size_t v___x_1464_; lean_object* v_j_1465_; lean_object* v___x_1466_; 
v_es_1461_ = lean_ctor_get(v_x_1458_, 0);
v___x_1462_ = lean_box(2);
v___x_1463_ = ((size_t)31ULL);
v___x_1464_ = lean_usize_land(v_x_1459_, v___x_1463_);
v_j_1465_ = lean_usize_to_nat(v___x_1464_);
v___x_1466_ = lean_array_get_borrowed(v___x_1462_, v_es_1461_, v_j_1465_);
lean_dec(v_j_1465_);
switch(lean_obj_tag(v___x_1466_))
{
case 0:
{
lean_object* v_key_1467_; uint8_t v___x_1468_; 
v_key_1467_ = lean_ctor_get(v___x_1466_, 0);
v___x_1468_ = lean_name_eq(v_x_1460_, v_key_1467_);
return v___x_1468_;
}
case 1:
{
lean_object* v_node_1469_; size_t v___x_1470_; size_t v___x_1471_; 
v_node_1469_ = lean_ctor_get(v___x_1466_, 0);
v___x_1470_ = ((size_t)5ULL);
v___x_1471_ = lean_usize_shift_right(v_x_1459_, v___x_1470_);
v_x_1458_ = v_node_1469_;
v_x_1459_ = v___x_1471_;
goto _start;
}
default: 
{
uint8_t v___x_1473_; 
v___x_1473_ = 0;
return v___x_1473_;
}
}
}
else
{
lean_object* v_ks_1474_; lean_object* v___x_1475_; uint8_t v___x_1476_; 
v_ks_1474_ = lean_ctor_get(v_x_1458_, 0);
v___x_1475_ = lean_unsigned_to_nat(0u);
v___x_1476_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__5_spec__8_spec__11___redArg(v_ks_1474_, v___x_1475_, v_x_1460_);
return v___x_1476_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__5_spec__8___redArg___boxed(lean_object* v_x_1477_, lean_object* v_x_1478_, lean_object* v_x_1479_){
_start:
{
size_t v_x_16407__boxed_1480_; uint8_t v_res_1481_; lean_object* v_r_1482_; 
v_x_16407__boxed_1480_ = lean_unbox_usize(v_x_1478_);
lean_dec(v_x_1478_);
v_res_1481_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__5_spec__8___redArg(v_x_1477_, v_x_16407__boxed_1480_, v_x_1479_);
lean_dec(v_x_1479_);
lean_dec_ref(v_x_1477_);
v_r_1482_ = lean_box(v_res_1481_);
return v_r_1482_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__5___redArg(lean_object* v_x_1483_, lean_object* v_x_1484_){
_start:
{
uint64_t v___y_1486_; 
if (lean_obj_tag(v_x_1484_) == 0)
{
uint64_t v___x_1489_; 
v___x_1489_ = 1723ULL;
v___y_1486_ = v___x_1489_;
goto v___jp_1485_;
}
else
{
uint64_t v_hash_1490_; 
v_hash_1490_ = lean_ctor_get_uint64(v_x_1484_, sizeof(void*)*2);
v___y_1486_ = v_hash_1490_;
goto v___jp_1485_;
}
v___jp_1485_:
{
size_t v___x_1487_; uint8_t v___x_1488_; 
v___x_1487_ = lean_uint64_to_usize(v___y_1486_);
v___x_1488_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__5_spec__8___redArg(v_x_1483_, v___x_1487_, v_x_1484_);
return v___x_1488_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__5___redArg___boxed(lean_object* v_x_1491_, lean_object* v_x_1492_){
_start:
{
uint8_t v_res_1493_; lean_object* v_r_1494_; 
v_res_1493_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__5___redArg(v_x_1491_, v_x_1492_);
lean_dec(v_x_1492_);
lean_dec_ref(v_x_1491_);
v_r_1494_ = lean_box(v_res_1493_);
return v_r_1494_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__10(lean_object* v___x_1495_, lean_object* v___x_1496_, uint8_t v___y_1497_, lean_object* v_ignoreTacticKinds_1498_, lean_object* v_stx_1499_, lean_object* v_a_1500_){
_start:
{
lean_object* v___y_1503_; uint8_t v___y_1504_; 
if (lean_obj_tag(v_stx_1499_) == 1)
{
lean_object* v_kind_1522_; lean_object* v_args_1523_; lean_object* v___y_1525_; lean_object* v___y_1529_; uint8_t v___x_1530_; 
v_kind_1522_ = lean_ctor_get(v_stx_1499_, 1);
v_args_1523_ = lean_ctor_get(v_stx_1499_, 2);
v___x_1530_ = lp_mathlib_Mathlib_Linter_UnusedTactic_isIgnoreTacticKind(v_ignoreTacticKinds_1498_, v_kind_1522_);
if (v___x_1530_ == 0)
{
lean_object* v___x_1531_; lean_object* v___x_1532_; uint8_t v___x_1533_; 
v___x_1531_ = lean_unsigned_to_nat(0u);
v___x_1532_ = lean_array_get_size(v_args_1523_);
v___x_1533_ = lean_nat_dec_lt(v___x_1531_, v___x_1532_);
if (v___x_1533_ == 0)
{
v___y_1525_ = v_a_1500_;
goto v___jp_1524_;
}
else
{
lean_object* v___x_1534_; uint8_t v___x_1535_; 
v___x_1534_ = lean_box(0);
v___x_1535_ = lean_nat_dec_le(v___x_1532_, v___x_1532_);
if (v___x_1535_ == 0)
{
if (v___x_1533_ == 0)
{
v___y_1525_ = v_a_1500_;
goto v___jp_1524_;
}
else
{
size_t v___x_1536_; size_t v___x_1537_; lean_object* v___x_1538_; 
v___x_1536_ = ((size_t)0ULL);
v___x_1537_ = lean_usize_of_nat(v___x_1532_);
v___x_1538_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__10_spec__16(v___x_1495_, v___x_1496_, v___y_1497_, v_ignoreTacticKinds_1498_, v_args_1523_, v___x_1536_, v___x_1537_, v___x_1534_, v_a_1500_);
v___y_1529_ = v___x_1538_;
goto v___jp_1528_;
}
}
else
{
size_t v___x_1539_; size_t v___x_1540_; lean_object* v___x_1541_; 
v___x_1539_ = ((size_t)0ULL);
v___x_1540_ = lean_usize_of_nat(v___x_1532_);
v___x_1541_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__10_spec__16(v___x_1495_, v___x_1496_, v___y_1497_, v_ignoreTacticKinds_1498_, v_args_1523_, v___x_1539_, v___x_1540_, v___x_1534_, v_a_1500_);
v___y_1529_ = v___x_1541_;
goto v___jp_1528_;
}
}
}
else
{
v___y_1525_ = v_a_1500_;
goto v___jp_1524_;
}
v___jp_1524_:
{
uint8_t v___x_1526_; 
v___x_1526_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__5___redArg(v___x_1495_, v_kind_1522_);
if (v___x_1526_ == 0)
{
uint8_t v___x_1527_; 
v___x_1527_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__5___redArg(v___x_1496_, v_kind_1522_);
v___y_1503_ = v___y_1525_;
v___y_1504_ = v___x_1527_;
goto v___jp_1502_;
}
else
{
v___y_1503_ = v___y_1525_;
v___y_1504_ = v___y_1497_;
goto v___jp_1502_;
}
}
v___jp_1528_:
{
if (lean_obj_tag(v___y_1529_) == 0)
{
lean_dec_ref_known(v___y_1529_, 1);
v___y_1525_ = v_a_1500_;
goto v___jp_1524_;
}
else
{
lean_dec_ref_known(v_stx_1499_, 3);
return v___y_1529_;
}
}
}
else
{
lean_object* v___x_1542_; lean_object* v___x_1543_; 
lean_dec(v_stx_1499_);
v___x_1542_ = lean_box(0);
v___x_1543_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1543_, 0, v___x_1542_);
return v___x_1543_;
}
v___jp_1502_:
{
if (v___y_1504_ == 0)
{
lean_object* v___x_1505_; lean_object* v___x_1506_; 
lean_dec(v_stx_1499_);
v___x_1505_ = lean_box(0);
v___x_1506_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1506_, 0, v___x_1505_);
return v___x_1506_;
}
else
{
lean_object* v___x_1507_; 
v___x_1507_ = l_Lean_Syntax_getRange_x3f(v_stx_1499_, v___y_1504_);
if (lean_obj_tag(v___x_1507_) == 1)
{
lean_object* v_val_1508_; lean_object* v___x_1510_; uint8_t v_isShared_1511_; uint8_t v_isSharedCheck_1519_; 
v_val_1508_ = lean_ctor_get(v___x_1507_, 0);
v_isSharedCheck_1519_ = !lean_is_exclusive(v___x_1507_);
if (v_isSharedCheck_1519_ == 0)
{
v___x_1510_ = v___x_1507_;
v_isShared_1511_ = v_isSharedCheck_1519_;
goto v_resetjp_1509_;
}
else
{
lean_inc(v_val_1508_);
lean_dec(v___x_1507_);
v___x_1510_ = lean_box(0);
v_isShared_1511_ = v_isSharedCheck_1519_;
goto v_resetjp_1509_;
}
v_resetjp_1509_:
{
lean_object* v___x_1512_; lean_object* v___x_1513_; lean_object* v___x_1514_; lean_object* v___x_1515_; lean_object* v___x_1517_; 
v___x_1512_ = lean_st_ref_take(v___y_1503_);
v___x_1513_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__10_spec__15___redArg(v___x_1512_, v_val_1508_, v_stx_1499_);
v___x_1514_ = lean_st_ref_set(v___y_1503_, v___x_1513_);
v___x_1515_ = lean_box(0);
if (v_isShared_1511_ == 0)
{
lean_ctor_set_tag(v___x_1510_, 0);
lean_ctor_set(v___x_1510_, 0, v___x_1515_);
v___x_1517_ = v___x_1510_;
goto v_reusejp_1516_;
}
else
{
lean_object* v_reuseFailAlloc_1518_; 
v_reuseFailAlloc_1518_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1518_, 0, v___x_1515_);
v___x_1517_ = v_reuseFailAlloc_1518_;
goto v_reusejp_1516_;
}
v_reusejp_1516_:
{
return v___x_1517_;
}
}
}
else
{
lean_object* v___x_1520_; lean_object* v___x_1521_; 
lean_dec(v___x_1507_);
lean_dec(v_stx_1499_);
v___x_1520_ = lean_box(0);
v___x_1521_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1521_, 0, v___x_1520_);
return v___x_1521_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__10_spec__16(lean_object* v___x_1544_, lean_object* v___x_1545_, uint8_t v___y_1546_, lean_object* v_ignoreTacticKinds_1547_, lean_object* v_as_1548_, size_t v_i_1549_, size_t v_stop_1550_, lean_object* v_b_1551_, lean_object* v___y_1552_){
_start:
{
uint8_t v___x_1554_; 
v___x_1554_ = lean_usize_dec_eq(v_i_1549_, v_stop_1550_);
if (v___x_1554_ == 0)
{
lean_object* v___x_1555_; lean_object* v___x_1556_; 
v___x_1555_ = lean_array_uget_borrowed(v_as_1548_, v_i_1549_);
lean_inc(v___x_1555_);
v___x_1556_ = lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__10(v___x_1544_, v___x_1545_, v___y_1546_, v_ignoreTacticKinds_1547_, v___x_1555_, v___y_1552_);
if (lean_obj_tag(v___x_1556_) == 0)
{
lean_object* v_a_1557_; size_t v___x_1558_; size_t v___x_1559_; 
v_a_1557_ = lean_ctor_get(v___x_1556_, 0);
lean_inc(v_a_1557_);
lean_dec_ref_known(v___x_1556_, 1);
v___x_1558_ = ((size_t)1ULL);
v___x_1559_ = lean_usize_add(v_i_1549_, v___x_1558_);
v_i_1549_ = v___x_1559_;
v_b_1551_ = v_a_1557_;
goto _start;
}
else
{
return v___x_1556_;
}
}
else
{
lean_object* v___x_1561_; 
v___x_1561_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1561_, 0, v_b_1551_);
return v___x_1561_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__10_spec__16___boxed(lean_object* v___x_1562_, lean_object* v___x_1563_, lean_object* v___y_1564_, lean_object* v_ignoreTacticKinds_1565_, lean_object* v_as_1566_, lean_object* v_i_1567_, lean_object* v_stop_1568_, lean_object* v_b_1569_, lean_object* v___y_1570_, lean_object* v___y_1571_){
_start:
{
uint8_t v___y_16465__boxed_1572_; size_t v_i_boxed_1573_; size_t v_stop_boxed_1574_; lean_object* v_res_1575_; 
v___y_16465__boxed_1572_ = lean_unbox(v___y_1564_);
v_i_boxed_1573_ = lean_unbox_usize(v_i_1567_);
lean_dec(v_i_1567_);
v_stop_boxed_1574_ = lean_unbox_usize(v_stop_1568_);
lean_dec(v_stop_1568_);
v_res_1575_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__10_spec__16(v___x_1562_, v___x_1563_, v___y_16465__boxed_1572_, v_ignoreTacticKinds_1565_, v_as_1566_, v_i_boxed_1573_, v_stop_boxed_1574_, v_b_1569_, v___y_1570_);
lean_dec(v___y_1570_);
lean_dec_ref(v_as_1566_);
lean_dec_ref(v_ignoreTacticKinds_1565_);
lean_dec_ref(v___x_1563_);
lean_dec_ref(v___x_1562_);
return v_res_1575_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__10___boxed(lean_object* v___x_1576_, lean_object* v___x_1577_, lean_object* v___y_1578_, lean_object* v_ignoreTacticKinds_1579_, lean_object* v_stx_1580_, lean_object* v_a_1581_, lean_object* v_a_1582_){
_start:
{
uint8_t v___y_16479__boxed_1583_; lean_object* v_res_1584_; 
v___y_16479__boxed_1583_ = lean_unbox(v___y_1578_);
v_res_1584_ = lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__10(v___x_1576_, v___x_1577_, v___y_16479__boxed_1583_, v_ignoreTacticKinds_1579_, v_stx_1580_, v_a_1581_);
lean_dec(v_a_1581_);
lean_dec_ref(v_ignoreTacticKinds_1579_);
lean_dec_ref(v___x_1577_);
lean_dec_ref(v___x_1576_);
return v_res_1584_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__11_spec__18_spec__24___redArg(lean_object* v_a_1585_, lean_object* v_b_1586_, lean_object* v_x_1587_){
_start:
{
if (lean_obj_tag(v_x_1587_) == 0)
{
lean_dec(v_b_1586_);
lean_dec(v_a_1585_);
return v_x_1587_;
}
else
{
lean_object* v_key_1588_; lean_object* v_value_1589_; lean_object* v_tail_1590_; lean_object* v___x_1592_; uint8_t v_isShared_1593_; uint8_t v_isSharedCheck_1602_; 
v_key_1588_ = lean_ctor_get(v_x_1587_, 0);
v_value_1589_ = lean_ctor_get(v_x_1587_, 1);
v_tail_1590_ = lean_ctor_get(v_x_1587_, 2);
v_isSharedCheck_1602_ = !lean_is_exclusive(v_x_1587_);
if (v_isSharedCheck_1602_ == 0)
{
v___x_1592_ = v_x_1587_;
v_isShared_1593_ = v_isSharedCheck_1602_;
goto v_resetjp_1591_;
}
else
{
lean_inc(v_tail_1590_);
lean_inc(v_value_1589_);
lean_inc(v_key_1588_);
lean_dec(v_x_1587_);
v___x_1592_ = lean_box(0);
v_isShared_1593_ = v_isSharedCheck_1602_;
goto v_resetjp_1591_;
}
v_resetjp_1591_:
{
uint8_t v___x_1594_; 
v___x_1594_ = lean_name_eq(v_key_1588_, v_a_1585_);
if (v___x_1594_ == 0)
{
lean_object* v___x_1595_; lean_object* v___x_1597_; 
v___x_1595_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__11_spec__18_spec__24___redArg(v_a_1585_, v_b_1586_, v_tail_1590_);
if (v_isShared_1593_ == 0)
{
lean_ctor_set(v___x_1592_, 2, v___x_1595_);
v___x_1597_ = v___x_1592_;
goto v_reusejp_1596_;
}
else
{
lean_object* v_reuseFailAlloc_1598_; 
v_reuseFailAlloc_1598_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1598_, 0, v_key_1588_);
lean_ctor_set(v_reuseFailAlloc_1598_, 1, v_value_1589_);
lean_ctor_set(v_reuseFailAlloc_1598_, 2, v___x_1595_);
v___x_1597_ = v_reuseFailAlloc_1598_;
goto v_reusejp_1596_;
}
v_reusejp_1596_:
{
return v___x_1597_;
}
}
else
{
lean_object* v___x_1600_; 
lean_dec(v_value_1589_);
lean_dec(v_key_1588_);
if (v_isShared_1593_ == 0)
{
lean_ctor_set(v___x_1592_, 1, v_b_1586_);
lean_ctor_set(v___x_1592_, 0, v_a_1585_);
v___x_1600_ = v___x_1592_;
goto v_reusejp_1599_;
}
else
{
lean_object* v_reuseFailAlloc_1601_; 
v_reuseFailAlloc_1601_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1601_, 0, v_a_1585_);
lean_ctor_set(v_reuseFailAlloc_1601_, 1, v_b_1586_);
lean_ctor_set(v_reuseFailAlloc_1601_, 2, v_tail_1590_);
v___x_1600_ = v_reuseFailAlloc_1601_;
goto v_reusejp_1599_;
}
v_reusejp_1599_:
{
return v___x_1600_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__11_spec__18___redArg(lean_object* v_m_1603_, lean_object* v_a_1604_, lean_object* v_b_1605_){
_start:
{
lean_object* v_size_1606_; lean_object* v_buckets_1607_; lean_object* v___x_1609_; uint8_t v_isShared_1610_; uint8_t v_isSharedCheck_1653_; 
v_size_1606_ = lean_ctor_get(v_m_1603_, 0);
v_buckets_1607_ = lean_ctor_get(v_m_1603_, 1);
v_isSharedCheck_1653_ = !lean_is_exclusive(v_m_1603_);
if (v_isSharedCheck_1653_ == 0)
{
v___x_1609_ = v_m_1603_;
v_isShared_1610_ = v_isSharedCheck_1653_;
goto v_resetjp_1608_;
}
else
{
lean_inc(v_buckets_1607_);
lean_inc(v_size_1606_);
lean_dec(v_m_1603_);
v___x_1609_ = lean_box(0);
v_isShared_1610_ = v_isSharedCheck_1653_;
goto v_resetjp_1608_;
}
v_resetjp_1608_:
{
lean_object* v___x_1611_; uint64_t v___y_1613_; 
v___x_1611_ = lean_array_get_size(v_buckets_1607_);
if (lean_obj_tag(v_a_1604_) == 0)
{
uint64_t v___x_1651_; 
v___x_1651_ = 1723ULL;
v___y_1613_ = v___x_1651_;
goto v___jp_1612_;
}
else
{
uint64_t v_hash_1652_; 
v_hash_1652_ = lean_ctor_get_uint64(v_a_1604_, sizeof(void*)*2);
v___y_1613_ = v_hash_1652_;
goto v___jp_1612_;
}
v___jp_1612_:
{
uint64_t v___x_1614_; uint64_t v___x_1615_; uint64_t v_fold_1616_; uint64_t v___x_1617_; uint64_t v___x_1618_; uint64_t v___x_1619_; size_t v___x_1620_; size_t v___x_1621_; size_t v___x_1622_; size_t v___x_1623_; size_t v___x_1624_; lean_object* v_bkt_1625_; uint8_t v___x_1626_; 
v___x_1614_ = 32ULL;
v___x_1615_ = lean_uint64_shift_right(v___y_1613_, v___x_1614_);
v_fold_1616_ = lean_uint64_xor(v___y_1613_, v___x_1615_);
v___x_1617_ = 16ULL;
v___x_1618_ = lean_uint64_shift_right(v_fold_1616_, v___x_1617_);
v___x_1619_ = lean_uint64_xor(v_fold_1616_, v___x_1618_);
v___x_1620_ = lean_uint64_to_usize(v___x_1619_);
v___x_1621_ = lean_usize_of_nat(v___x_1611_);
v___x_1622_ = ((size_t)1ULL);
v___x_1623_ = lean_usize_sub(v___x_1621_, v___x_1622_);
v___x_1624_ = lean_usize_land(v___x_1620_, v___x_1623_);
v_bkt_1625_ = lean_array_uget_borrowed(v_buckets_1607_, v___x_1624_);
v___x_1626_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__spec__0_spec__0_spec__1___redArg(v_a_1604_, v_bkt_1625_);
if (v___x_1626_ == 0)
{
lean_object* v___x_1627_; lean_object* v_size_x27_1628_; lean_object* v___x_1629_; lean_object* v_buckets_x27_1630_; lean_object* v___x_1631_; lean_object* v___x_1632_; lean_object* v___x_1633_; lean_object* v___x_1634_; lean_object* v___x_1635_; uint8_t v___x_1636_; 
v___x_1627_ = lean_unsigned_to_nat(1u);
v_size_x27_1628_ = lean_nat_add(v_size_1606_, v___x_1627_);
lean_dec(v_size_1606_);
lean_inc(v_bkt_1625_);
v___x_1629_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1629_, 0, v_a_1604_);
lean_ctor_set(v___x_1629_, 1, v_b_1605_);
lean_ctor_set(v___x_1629_, 2, v_bkt_1625_);
v_buckets_x27_1630_ = lean_array_uset(v_buckets_1607_, v___x_1624_, v___x_1629_);
v___x_1631_ = lean_unsigned_to_nat(4u);
v___x_1632_ = lean_nat_mul(v_size_x27_1628_, v___x_1631_);
v___x_1633_ = lean_unsigned_to_nat(3u);
v___x_1634_ = lean_nat_div(v___x_1632_, v___x_1633_);
lean_dec(v___x_1632_);
v___x_1635_ = lean_array_get_size(v_buckets_x27_1630_);
v___x_1636_ = lean_nat_dec_le(v___x_1634_, v___x_1635_);
lean_dec(v___x_1634_);
if (v___x_1636_ == 0)
{
lean_object* v_val_1637_; lean_object* v___x_1639_; 
v_val_1637_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__spec__0_spec__0_spec__2___redArg(v_buckets_x27_1630_);
if (v_isShared_1610_ == 0)
{
lean_ctor_set(v___x_1609_, 1, v_val_1637_);
lean_ctor_set(v___x_1609_, 0, v_size_x27_1628_);
v___x_1639_ = v___x_1609_;
goto v_reusejp_1638_;
}
else
{
lean_object* v_reuseFailAlloc_1640_; 
v_reuseFailAlloc_1640_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1640_, 0, v_size_x27_1628_);
lean_ctor_set(v_reuseFailAlloc_1640_, 1, v_val_1637_);
v___x_1639_ = v_reuseFailAlloc_1640_;
goto v_reusejp_1638_;
}
v_reusejp_1638_:
{
return v___x_1639_;
}
}
else
{
lean_object* v___x_1642_; 
if (v_isShared_1610_ == 0)
{
lean_ctor_set(v___x_1609_, 1, v_buckets_x27_1630_);
lean_ctor_set(v___x_1609_, 0, v_size_x27_1628_);
v___x_1642_ = v___x_1609_;
goto v_reusejp_1641_;
}
else
{
lean_object* v_reuseFailAlloc_1643_; 
v_reuseFailAlloc_1643_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1643_, 0, v_size_x27_1628_);
lean_ctor_set(v_reuseFailAlloc_1643_, 1, v_buckets_x27_1630_);
v___x_1642_ = v_reuseFailAlloc_1643_;
goto v_reusejp_1641_;
}
v_reusejp_1641_:
{
return v___x_1642_;
}
}
}
else
{
lean_object* v___x_1644_; lean_object* v_buckets_x27_1645_; lean_object* v___x_1646_; lean_object* v___x_1647_; lean_object* v___x_1649_; 
lean_inc(v_bkt_1625_);
v___x_1644_ = lean_box(0);
v_buckets_x27_1645_ = lean_array_uset(v_buckets_1607_, v___x_1624_, v___x_1644_);
v___x_1646_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__11_spec__18_spec__24___redArg(v_a_1604_, v_b_1605_, v_bkt_1625_);
v___x_1647_ = lean_array_uset(v_buckets_x27_1645_, v___x_1624_, v___x_1646_);
if (v_isShared_1610_ == 0)
{
lean_ctor_set(v___x_1609_, 1, v___x_1647_);
v___x_1649_ = v___x_1609_;
goto v_reusejp_1648_;
}
else
{
lean_object* v_reuseFailAlloc_1650_; 
v_reuseFailAlloc_1650_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1650_, 0, v_size_1606_);
lean_ctor_set(v_reuseFailAlloc_1650_, 1, v___x_1647_);
v___x_1649_ = v_reuseFailAlloc_1650_;
goto v_reusejp_1648_;
}
v_reusejp_1648_:
{
return v___x_1649_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__11_spec__19(lean_object* v_a_1654_, lean_object* v_a_1655_){
_start:
{
if (lean_obj_tag(v_a_1654_) == 0)
{
lean_object* v___x_1656_; 
v___x_1656_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1656_, 0, v_a_1655_);
return v___x_1656_;
}
else
{
lean_object* v_key_1657_; lean_object* v_value_1658_; lean_object* v_tail_1659_; lean_object* v_r_1660_; 
v_key_1657_ = lean_ctor_get(v_a_1654_, 0);
lean_inc(v_key_1657_);
v_value_1658_ = lean_ctor_get(v_a_1654_, 1);
lean_inc(v_value_1658_);
v_tail_1659_ = lean_ctor_get(v_a_1654_, 2);
lean_inc(v_tail_1659_);
lean_dec_ref_known(v_a_1654_, 3);
v_r_1660_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__11_spec__18___redArg(v_a_1655_, v_key_1657_, v_value_1658_);
v_a_1654_ = v_tail_1659_;
v_a_1655_ = v_r_1660_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__11_spec__20(lean_object* v_as_1662_, size_t v_sz_1663_, size_t v_i_1664_, lean_object* v_b_1665_){
_start:
{
uint8_t v___x_1666_; 
v___x_1666_ = lean_usize_dec_lt(v_i_1664_, v_sz_1663_);
if (v___x_1666_ == 0)
{
return v_b_1665_;
}
else
{
lean_object* v_a_1667_; lean_object* v___x_1668_; 
v_a_1667_ = lean_array_uget_borrowed(v_as_1662_, v_i_1664_);
lean_inc(v_a_1667_);
v___x_1668_ = lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__11_spec__19(v_a_1667_, v_b_1665_);
if (lean_obj_tag(v___x_1668_) == 0)
{
lean_object* v_a_1669_; 
v_a_1669_ = lean_ctor_get(v___x_1668_, 0);
lean_inc(v_a_1669_);
lean_dec_ref_known(v___x_1668_, 1);
return v_a_1669_;
}
else
{
lean_object* v_a_1670_; size_t v___x_1671_; size_t v___x_1672_; 
v_a_1670_ = lean_ctor_get(v___x_1668_, 0);
lean_inc(v_a_1670_);
lean_dec_ref_known(v___x_1668_, 1);
v___x_1671_ = ((size_t)1ULL);
v___x_1672_ = lean_usize_add(v_i_1664_, v___x_1671_);
v_i_1664_ = v___x_1672_;
v_b_1665_ = v_a_1670_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__11_spec__20___boxed(lean_object* v_as_1674_, lean_object* v_sz_1675_, lean_object* v_i_1676_, lean_object* v_b_1677_){
_start:
{
size_t v_sz_boxed_1678_; size_t v_i_boxed_1679_; lean_object* v_res_1680_; 
v_sz_boxed_1678_ = lean_unbox_usize(v_sz_1675_);
lean_dec(v_sz_1675_);
v_i_boxed_1679_ = lean_unbox_usize(v_i_1676_);
lean_dec(v_i_1676_);
v_res_1680_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__11_spec__20(v_as_1674_, v_sz_boxed_1678_, v_i_boxed_1679_, v_b_1677_);
lean_dec_ref(v_as_1674_);
return v_res_1680_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__11(lean_object* v_m_1681_, lean_object* v_l_1682_){
_start:
{
lean_object* v_buckets_1683_; size_t v_sz_1684_; size_t v___x_1685_; lean_object* v___x_1686_; 
v_buckets_1683_ = lean_ctor_get(v_l_1682_, 1);
v_sz_1684_ = lean_array_size(v_buckets_1683_);
v___x_1685_ = ((size_t)0ULL);
v___x_1686_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__11_spec__20(v_buckets_1683_, v_sz_1684_, v___x_1685_, v_m_1681_);
return v___x_1686_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__11___boxed(lean_object* v_m_1687_, lean_object* v_l_1688_){
_start:
{
lean_object* v_res_1689_; 
v_res_1689_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__11(v_m_1687_, v_l_1688_);
lean_dec_ref(v_l_1688_);
return v_res_1689_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__2_spec__3_spec__5___redArg(lean_object* v_keys_1690_, lean_object* v_vals_1691_, lean_object* v_i_1692_, lean_object* v_k_1693_){
_start:
{
lean_object* v___x_1694_; uint8_t v___x_1695_; 
v___x_1694_ = lean_array_get_size(v_keys_1690_);
v___x_1695_ = lean_nat_dec_lt(v_i_1692_, v___x_1694_);
if (v___x_1695_ == 0)
{
lean_object* v___x_1696_; 
lean_dec(v_i_1692_);
v___x_1696_ = lean_box(0);
return v___x_1696_;
}
else
{
lean_object* v_k_x27_1697_; uint8_t v___x_1698_; 
v_k_x27_1697_ = lean_array_fget_borrowed(v_keys_1690_, v_i_1692_);
v___x_1698_ = lean_name_eq(v_k_1693_, v_k_x27_1697_);
if (v___x_1698_ == 0)
{
lean_object* v___x_1699_; lean_object* v___x_1700_; 
v___x_1699_ = lean_unsigned_to_nat(1u);
v___x_1700_ = lean_nat_add(v_i_1692_, v___x_1699_);
lean_dec(v_i_1692_);
v_i_1692_ = v___x_1700_;
goto _start;
}
else
{
lean_object* v___x_1702_; lean_object* v___x_1703_; 
v___x_1702_ = lean_array_fget_borrowed(v_vals_1691_, v_i_1692_);
lean_dec(v_i_1692_);
lean_inc(v___x_1702_);
v___x_1703_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1703_, 0, v___x_1702_);
return v___x_1703_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__2_spec__3_spec__5___redArg___boxed(lean_object* v_keys_1704_, lean_object* v_vals_1705_, lean_object* v_i_1706_, lean_object* v_k_1707_){
_start:
{
lean_object* v_res_1708_; 
v_res_1708_ = lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__2_spec__3_spec__5___redArg(v_keys_1704_, v_vals_1705_, v_i_1706_, v_k_1707_);
lean_dec(v_k_1707_);
lean_dec_ref(v_vals_1705_);
lean_dec_ref(v_keys_1704_);
return v_res_1708_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__2_spec__3___redArg(lean_object* v_x_1709_, size_t v_x_1710_, lean_object* v_x_1711_){
_start:
{
if (lean_obj_tag(v_x_1709_) == 0)
{
lean_object* v_es_1712_; lean_object* v___x_1713_; size_t v___x_1714_; size_t v___x_1715_; lean_object* v_j_1716_; lean_object* v___x_1717_; 
v_es_1712_ = lean_ctor_get(v_x_1709_, 0);
v___x_1713_ = lean_box(2);
v___x_1714_ = ((size_t)31ULL);
v___x_1715_ = lean_usize_land(v_x_1710_, v___x_1714_);
v_j_1716_ = lean_usize_to_nat(v___x_1715_);
v___x_1717_ = lean_array_get_borrowed(v___x_1713_, v_es_1712_, v_j_1716_);
lean_dec(v_j_1716_);
switch(lean_obj_tag(v___x_1717_))
{
case 0:
{
lean_object* v_key_1718_; lean_object* v_val_1719_; uint8_t v___x_1720_; 
v_key_1718_ = lean_ctor_get(v___x_1717_, 0);
v_val_1719_ = lean_ctor_get(v___x_1717_, 1);
v___x_1720_ = lean_name_eq(v_x_1711_, v_key_1718_);
if (v___x_1720_ == 0)
{
lean_object* v___x_1721_; 
v___x_1721_ = lean_box(0);
return v___x_1721_;
}
else
{
lean_object* v___x_1722_; 
lean_inc(v_val_1719_);
v___x_1722_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1722_, 0, v_val_1719_);
return v___x_1722_;
}
}
case 1:
{
lean_object* v_node_1723_; size_t v___x_1724_; size_t v___x_1725_; 
v_node_1723_ = lean_ctor_get(v___x_1717_, 0);
v___x_1724_ = ((size_t)5ULL);
v___x_1725_ = lean_usize_shift_right(v_x_1710_, v___x_1724_);
v_x_1709_ = v_node_1723_;
v_x_1710_ = v___x_1725_;
goto _start;
}
default: 
{
lean_object* v___x_1727_; 
v___x_1727_ = lean_box(0);
return v___x_1727_;
}
}
}
else
{
lean_object* v_ks_1728_; lean_object* v_vs_1729_; lean_object* v___x_1730_; lean_object* v___x_1731_; 
v_ks_1728_ = lean_ctor_get(v_x_1709_, 0);
v_vs_1729_ = lean_ctor_get(v_x_1709_, 1);
v___x_1730_ = lean_unsigned_to_nat(0u);
v___x_1731_ = lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__2_spec__3_spec__5___redArg(v_ks_1728_, v_vs_1729_, v___x_1730_, v_x_1711_);
return v___x_1731_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__2_spec__3___redArg___boxed(lean_object* v_x_1732_, lean_object* v_x_1733_, lean_object* v_x_1734_){
_start:
{
size_t v_x_16761__boxed_1735_; lean_object* v_res_1736_; 
v_x_16761__boxed_1735_ = lean_unbox_usize(v_x_1733_);
lean_dec(v_x_1733_);
v_res_1736_ = lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__2_spec__3___redArg(v_x_1732_, v_x_16761__boxed_1735_, v_x_1734_);
lean_dec(v_x_1734_);
lean_dec_ref(v_x_1732_);
return v_res_1736_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__2___redArg(lean_object* v_x_1737_, lean_object* v_x_1738_){
_start:
{
uint64_t v___y_1740_; 
if (lean_obj_tag(v_x_1738_) == 0)
{
uint64_t v___x_1743_; 
v___x_1743_ = 1723ULL;
v___y_1740_ = v___x_1743_;
goto v___jp_1739_;
}
else
{
uint64_t v_hash_1744_; 
v_hash_1744_ = lean_ctor_get_uint64(v_x_1738_, sizeof(void*)*2);
v___y_1740_ = v_hash_1744_;
goto v___jp_1739_;
}
v___jp_1739_:
{
size_t v___x_1741_; lean_object* v___x_1742_; 
v___x_1741_ = lean_uint64_to_usize(v___y_1740_);
v___x_1742_ = lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__2_spec__3___redArg(v_x_1737_, v___x_1741_, v_x_1738_);
return v___x_1742_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__2___redArg___boxed(lean_object* v_x_1745_, lean_object* v_x_1746_){
_start:
{
lean_object* v_res_1747_; 
v_res_1747_ = lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__2___redArg(v_x_1745_, v_x_1746_);
lean_dec(v_x_1746_);
lean_dec_ref(v_x_1745_);
return v_res_1747_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__12(lean_object* v_a_1748_, lean_object* v_a_1749_){
_start:
{
if (lean_obj_tag(v_a_1748_) == 0)
{
lean_object* v___x_1750_; 
v___x_1750_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1750_, 0, v_a_1749_);
return v___x_1750_;
}
else
{
lean_object* v_key_1751_; lean_object* v_value_1752_; lean_object* v_tail_1753_; lean_object* v_r_1754_; 
v_key_1751_ = lean_ctor_get(v_a_1748_, 0);
lean_inc(v_key_1751_);
v_value_1752_ = lean_ctor_get(v_a_1748_, 1);
lean_inc(v_value_1752_);
v_tail_1753_ = lean_ctor_get(v_a_1748_, 2);
lean_inc(v_tail_1753_);
lean_dec_ref_known(v_a_1748_, 3);
v_r_1754_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2__spec__0_spec__0___redArg(v_a_1749_, v_key_1751_, v_value_1752_);
v_a_1748_ = v_tail_1753_;
v_a_1749_ = v_r_1754_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__13(lean_object* v_as_1756_, size_t v_sz_1757_, size_t v_i_1758_, lean_object* v_b_1759_){
_start:
{
uint8_t v___x_1760_; 
v___x_1760_ = lean_usize_dec_lt(v_i_1758_, v_sz_1757_);
if (v___x_1760_ == 0)
{
return v_b_1759_;
}
else
{
lean_object* v_a_1761_; lean_object* v___x_1762_; 
v_a_1761_ = lean_array_uget_borrowed(v_as_1756_, v_i_1758_);
lean_inc(v_a_1761_);
v___x_1762_ = lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__12(v_a_1761_, v_b_1759_);
if (lean_obj_tag(v___x_1762_) == 0)
{
lean_object* v_a_1763_; 
v_a_1763_ = lean_ctor_get(v___x_1762_, 0);
lean_inc(v_a_1763_);
lean_dec_ref_known(v___x_1762_, 1);
return v_a_1763_;
}
else
{
lean_object* v_a_1764_; size_t v___x_1765_; size_t v___x_1766_; 
v_a_1764_ = lean_ctor_get(v___x_1762_, 0);
lean_inc(v_a_1764_);
lean_dec_ref_known(v___x_1762_, 1);
v___x_1765_ = ((size_t)1ULL);
v___x_1766_ = lean_usize_add(v_i_1758_, v___x_1765_);
v_i_1758_ = v___x_1766_;
v_b_1759_ = v_a_1764_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__13___boxed(lean_object* v_as_1768_, lean_object* v_sz_1769_, lean_object* v_i_1770_, lean_object* v_b_1771_){
_start:
{
size_t v_sz_boxed_1772_; size_t v_i_boxed_1773_; lean_object* v_res_1774_; 
v_sz_boxed_1772_ = lean_unbox_usize(v_sz_1769_);
lean_dec(v_sz_1769_);
v_i_boxed_1773_ = lean_unbox_usize(v_i_1770_);
lean_dec(v_i_1770_);
v_res_1774_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__13(v_as_1768_, v_sz_boxed_1772_, v_i_boxed_1773_, v_b_1771_);
lean_dec_ref(v_as_1768_);
return v_res_1774_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__1_spec__1___redArg(lean_object* v_o_1775_, lean_object* v___y_1776_){
_start:
{
lean_object* v___x_1778_; lean_object* v_env_1779_; lean_object* v___x_1780_; lean_object* v_toEnvExtension_1781_; lean_object* v_asyncMode_1782_; lean_object* v___x_1783_; lean_object* v___x_1784_; lean_object* v___x_1785_; lean_object* v_merged_1786_; lean_object* v___x_1788_; uint8_t v_isShared_1789_; uint8_t v_isSharedCheck_1794_; 
v___x_1778_ = lean_st_ref_get(v___y_1776_);
v_env_1779_ = lean_ctor_get(v___x_1778_, 0);
lean_inc_ref(v_env_1779_);
lean_dec(v___x_1778_);
v___x_1780_ = l_Lean_Linter_linterSetsExt;
v_toEnvExtension_1781_ = lean_ctor_get(v___x_1780_, 0);
v_asyncMode_1782_ = lean_ctor_get(v_toEnvExtension_1781_, 2);
v___x_1783_ = l_Lean_Linter_instInhabitedLinterSetsState_default;
v___x_1784_ = lean_box(0);
v___x_1785_ = l_Lean_PersistentEnvExtension_getState___redArg(v___x_1783_, v___x_1780_, v_env_1779_, v_asyncMode_1782_, v___x_1784_);
v_merged_1786_ = lean_ctor_get(v___x_1785_, 0);
v_isSharedCheck_1794_ = !lean_is_exclusive(v___x_1785_);
if (v_isSharedCheck_1794_ == 0)
{
lean_object* v_unused_1795_; 
v_unused_1795_ = lean_ctor_get(v___x_1785_, 1);
lean_dec(v_unused_1795_);
v___x_1788_ = v___x_1785_;
v_isShared_1789_ = v_isSharedCheck_1794_;
goto v_resetjp_1787_;
}
else
{
lean_inc(v_merged_1786_);
lean_dec(v___x_1785_);
v___x_1788_ = lean_box(0);
v_isShared_1789_ = v_isSharedCheck_1794_;
goto v_resetjp_1787_;
}
v_resetjp_1787_:
{
lean_object* v___x_1791_; 
if (v_isShared_1789_ == 0)
{
lean_ctor_set(v___x_1788_, 1, v_merged_1786_);
lean_ctor_set(v___x_1788_, 0, v_o_1775_);
v___x_1791_ = v___x_1788_;
goto v_reusejp_1790_;
}
else
{
lean_object* v_reuseFailAlloc_1793_; 
v_reuseFailAlloc_1793_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1793_, 0, v_o_1775_);
lean_ctor_set(v_reuseFailAlloc_1793_, 1, v_merged_1786_);
v___x_1791_ = v_reuseFailAlloc_1793_;
goto v_reusejp_1790_;
}
v_reusejp_1790_:
{
lean_object* v___x_1792_; 
v___x_1792_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1792_, 0, v___x_1791_);
return v___x_1792_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__1_spec__1___redArg___boxed(lean_object* v_o_1796_, lean_object* v___y_1797_, lean_object* v___y_1798_){
_start:
{
lean_object* v_res_1799_; 
v_res_1799_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__1_spec__1___redArg(v_o_1796_, v___y_1797_);
lean_dec(v___y_1797_);
return v_res_1799_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__1(lean_object* v___y_1800_, lean_object* v___y_1801_){
_start:
{
lean_object* v___x_1803_; lean_object* v_scopes_1804_; lean_object* v___x_1805_; lean_object* v___x_1806_; lean_object* v_opts_1807_; lean_object* v___x_1808_; 
v___x_1803_ = lean_st_ref_get(v___y_1801_);
v_scopes_1804_ = lean_ctor_get(v___x_1803_, 2);
lean_inc(v_scopes_1804_);
lean_dec(v___x_1803_);
v___x_1805_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_1806_ = l_List_head_x21___redArg(v___x_1805_, v_scopes_1804_);
lean_dec(v_scopes_1804_);
v_opts_1807_ = lean_ctor_get(v___x_1806_, 1);
lean_inc_ref(v_opts_1807_);
lean_dec(v___x_1806_);
v___x_1808_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__1_spec__1___redArg(v_opts_1807_, v___y_1801_);
return v___x_1808_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__1___boxed(lean_object* v___y_1809_, lean_object* v___y_1810_, lean_object* v___y_1811_){
_start:
{
lean_object* v_res_1812_; 
v_res_1812_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__1(v___y_1809_, v___y_1810_);
lean_dec(v___y_1810_);
lean_dec_ref(v___y_1809_);
return v_res_1812_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8___lam__0(uint8_t v___y_1814_, uint8_t v_suppressElabErrors_1815_, lean_object* v_x_1816_){
_start:
{
if (lean_obj_tag(v_x_1816_) == 1)
{
lean_object* v_pre_1817_; 
v_pre_1817_ = lean_ctor_get(v_x_1816_, 0);
if (lean_obj_tag(v_pre_1817_) == 0)
{
lean_object* v_str_1818_; lean_object* v___x_1819_; uint8_t v___x_1820_; 
v_str_1818_ = lean_ctor_get(v_x_1816_, 1);
v___x_1819_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8___lam__0___closed__0));
v___x_1820_ = lean_string_dec_eq(v_str_1818_, v___x_1819_);
if (v___x_1820_ == 0)
{
return v___y_1814_;
}
else
{
return v_suppressElabErrors_1815_;
}
}
else
{
return v___y_1814_;
}
}
else
{
return v___y_1814_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8___lam__0___boxed(lean_object* v___y_1821_, lean_object* v_suppressElabErrors_1822_, lean_object* v_x_1823_){
_start:
{
uint8_t v___y_16904__boxed_1824_; uint8_t v_suppressElabErrors_boxed_1825_; uint8_t v_res_1826_; lean_object* v_r_1827_; 
v___y_16904__boxed_1824_ = lean_unbox(v___y_1821_);
v_suppressElabErrors_boxed_1825_ = lean_unbox(v_suppressElabErrors_1822_);
v_res_1826_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8___lam__0(v___y_16904__boxed_1824_, v_suppressElabErrors_boxed_1825_, v_x_1823_);
lean_dec(v_x_1823_);
v_r_1827_ = lean_box(v_res_1826_);
return v_r_1827_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8_spec__18___redArg___closed__0(void){
_start:
{
lean_object* v___x_1828_; 
v___x_1828_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_1828_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8_spec__18___redArg___closed__1(void){
_start:
{
lean_object* v___x_1829_; lean_object* v___x_1830_; 
v___x_1829_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8_spec__18___redArg___closed__0, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8_spec__18___redArg___closed__0_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8_spec__18___redArg___closed__0);
v___x_1830_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1830_, 0, v___x_1829_);
return v___x_1830_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8_spec__18___redArg___closed__2(void){
_start:
{
lean_object* v___x_1831_; lean_object* v___x_1832_; lean_object* v___x_1833_; 
v___x_1831_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8_spec__18___redArg___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8_spec__18___redArg___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8_spec__18___redArg___closed__1);
v___x_1832_ = lean_unsigned_to_nat(0u);
v___x_1833_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_1833_, 0, v___x_1832_);
lean_ctor_set(v___x_1833_, 1, v___x_1832_);
lean_ctor_set(v___x_1833_, 2, v___x_1832_);
lean_ctor_set(v___x_1833_, 3, v___x_1832_);
lean_ctor_set(v___x_1833_, 4, v___x_1831_);
lean_ctor_set(v___x_1833_, 5, v___x_1831_);
lean_ctor_set(v___x_1833_, 6, v___x_1831_);
lean_ctor_set(v___x_1833_, 7, v___x_1831_);
lean_ctor_set(v___x_1833_, 8, v___x_1831_);
lean_ctor_set(v___x_1833_, 9, v___x_1831_);
return v___x_1833_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8_spec__18___redArg___closed__3(void){
_start:
{
lean_object* v___x_1834_; lean_object* v___x_1835_; lean_object* v___x_1836_; 
v___x_1834_ = lean_unsigned_to_nat(32u);
v___x_1835_ = lean_mk_empty_array_with_capacity(v___x_1834_);
v___x_1836_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1836_, 0, v___x_1835_);
return v___x_1836_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8_spec__18___redArg___closed__4(void){
_start:
{
size_t v___x_1837_; lean_object* v___x_1838_; lean_object* v___x_1839_; lean_object* v___x_1840_; lean_object* v___x_1841_; lean_object* v___x_1842_; 
v___x_1837_ = ((size_t)5ULL);
v___x_1838_ = lean_unsigned_to_nat(0u);
v___x_1839_ = lean_unsigned_to_nat(32u);
v___x_1840_ = lean_mk_empty_array_with_capacity(v___x_1839_);
v___x_1841_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8_spec__18___redArg___closed__3, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8_spec__18___redArg___closed__3_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8_spec__18___redArg___closed__3);
v___x_1842_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_1842_, 0, v___x_1841_);
lean_ctor_set(v___x_1842_, 1, v___x_1840_);
lean_ctor_set(v___x_1842_, 2, v___x_1838_);
lean_ctor_set(v___x_1842_, 3, v___x_1838_);
lean_ctor_set_usize(v___x_1842_, 4, v___x_1837_);
return v___x_1842_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8_spec__18___redArg___closed__5(void){
_start:
{
lean_object* v___x_1843_; lean_object* v___x_1844_; lean_object* v___x_1845_; lean_object* v___x_1846_; 
v___x_1843_ = lean_box(1);
v___x_1844_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8_spec__18___redArg___closed__4, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8_spec__18___redArg___closed__4_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8_spec__18___redArg___closed__4);
v___x_1845_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8_spec__18___redArg___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8_spec__18___redArg___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8_spec__18___redArg___closed__1);
v___x_1846_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1846_, 0, v___x_1845_);
lean_ctor_set(v___x_1846_, 1, v___x_1844_);
lean_ctor_set(v___x_1846_, 2, v___x_1843_);
return v___x_1846_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8_spec__18___redArg(lean_object* v_msgData_1847_, lean_object* v___y_1848_){
_start:
{
lean_object* v___x_1850_; lean_object* v_env_1851_; lean_object* v___x_1852_; lean_object* v_scopes_1853_; lean_object* v___x_1854_; lean_object* v___x_1855_; lean_object* v_opts_1856_; lean_object* v___x_1857_; lean_object* v___x_1858_; lean_object* v___x_1859_; lean_object* v___x_1860_; lean_object* v___x_1861_; 
v___x_1850_ = lean_st_ref_get(v___y_1848_);
v_env_1851_ = lean_ctor_get(v___x_1850_, 0);
lean_inc_ref(v_env_1851_);
lean_dec(v___x_1850_);
v___x_1852_ = lean_st_ref_get(v___y_1848_);
v_scopes_1853_ = lean_ctor_get(v___x_1852_, 2);
lean_inc(v_scopes_1853_);
lean_dec(v___x_1852_);
v___x_1854_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_1855_ = l_List_head_x21___redArg(v___x_1854_, v_scopes_1853_);
lean_dec(v_scopes_1853_);
v_opts_1856_ = lean_ctor_get(v___x_1855_, 1);
lean_inc_ref(v_opts_1856_);
lean_dec(v___x_1855_);
v___x_1857_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8_spec__18___redArg___closed__2, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8_spec__18___redArg___closed__2_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8_spec__18___redArg___closed__2);
v___x_1858_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8_spec__18___redArg___closed__5, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8_spec__18___redArg___closed__5_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8_spec__18___redArg___closed__5);
v___x_1859_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1859_, 0, v_env_1851_);
lean_ctor_set(v___x_1859_, 1, v___x_1857_);
lean_ctor_set(v___x_1859_, 2, v___x_1858_);
lean_ctor_set(v___x_1859_, 3, v_opts_1856_);
v___x_1860_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_1860_, 0, v___x_1859_);
lean_ctor_set(v___x_1860_, 1, v_msgData_1847_);
v___x_1861_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1861_, 0, v___x_1860_);
return v___x_1861_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8_spec__18___redArg___boxed(lean_object* v_msgData_1862_, lean_object* v___y_1863_, lean_object* v___y_1864_){
_start:
{
lean_object* v_res_1865_; 
v_res_1865_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8_spec__18___redArg(v_msgData_1862_, v___y_1863_);
lean_dec(v___y_1863_);
return v_res_1865_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8_spec__19(lean_object* v_opts_1866_, lean_object* v_opt_1867_){
_start:
{
lean_object* v_name_1868_; lean_object* v_defValue_1869_; lean_object* v_map_1870_; lean_object* v___x_1871_; 
v_name_1868_ = lean_ctor_get(v_opt_1867_, 0);
v_defValue_1869_ = lean_ctor_get(v_opt_1867_, 1);
v_map_1870_ = lean_ctor_get(v_opts_1866_, 0);
v___x_1871_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_1870_, v_name_1868_);
if (lean_obj_tag(v___x_1871_) == 0)
{
uint8_t v___x_1872_; 
v___x_1872_ = lean_unbox(v_defValue_1869_);
return v___x_1872_;
}
else
{
lean_object* v_val_1873_; 
v_val_1873_ = lean_ctor_get(v___x_1871_, 0);
lean_inc(v_val_1873_);
lean_dec_ref_known(v___x_1871_, 1);
if (lean_obj_tag(v_val_1873_) == 1)
{
uint8_t v_v_1874_; 
v_v_1874_ = lean_ctor_get_uint8(v_val_1873_, 0);
lean_dec_ref_known(v_val_1873_, 0);
return v_v_1874_;
}
else
{
uint8_t v___x_1875_; 
lean_dec(v_val_1873_);
v___x_1875_ = lean_unbox(v_defValue_1869_);
return v___x_1875_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8_spec__19___boxed(lean_object* v_opts_1876_, lean_object* v_opt_1877_){
_start:
{
uint8_t v_res_1878_; lean_object* v_r_1879_; 
v_res_1878_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8_spec__19(v_opts_1876_, v_opt_1877_);
lean_dec_ref(v_opt_1877_);
lean_dec_ref(v_opts_1876_);
v_r_1879_ = lean_box(v_res_1878_);
return v_r_1879_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8(lean_object* v_ref_1881_, lean_object* v_msgData_1882_, uint8_t v_severity_1883_, uint8_t v_isSilent_1884_, lean_object* v___y_1885_, lean_object* v___y_1886_){
_start:
{
lean_object* v___y_1889_; lean_object* v___y_1890_; lean_object* v___y_1891_; lean_object* v___y_1892_; lean_object* v___y_1893_; uint8_t v___y_1894_; uint8_t v___y_1895_; lean_object* v___y_1896_; uint8_t v___y_1953_; lean_object* v___y_1954_; uint8_t v___y_1955_; uint8_t v___y_1956_; lean_object* v___y_1957_; uint8_t v___y_1981_; lean_object* v___y_1982_; uint8_t v___y_1983_; uint8_t v___y_1984_; lean_object* v___y_1985_; uint8_t v___y_1989_; uint8_t v___y_1990_; uint8_t v___y_1991_; uint8_t v___x_2006_; uint8_t v___y_2008_; uint8_t v___y_2009_; uint8_t v___y_2010_; uint8_t v___y_2012_; uint8_t v___x_2024_; 
v___x_2006_ = 2;
v___x_2024_ = l_Lean_instBEqMessageSeverity_beq(v_severity_1883_, v___x_2006_);
if (v___x_2024_ == 0)
{
v___y_2012_ = v___x_2024_;
goto v___jp_2011_;
}
else
{
uint8_t v___x_2025_; 
lean_inc_ref(v_msgData_1882_);
v___x_2025_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_1882_);
v___y_2012_ = v___x_2025_;
goto v___jp_2011_;
}
v___jp_1888_:
{
lean_object* v___x_1897_; 
v___x_1897_ = l_Lean_Elab_Command_getScope___redArg(v___y_1896_);
if (lean_obj_tag(v___x_1897_) == 0)
{
lean_object* v_a_1898_; lean_object* v___x_1899_; 
v_a_1898_ = lean_ctor_get(v___x_1897_, 0);
lean_inc(v_a_1898_);
lean_dec_ref_known(v___x_1897_, 1);
v___x_1899_ = l_Lean_Elab_Command_getScope___redArg(v___y_1896_);
if (lean_obj_tag(v___x_1899_) == 0)
{
lean_object* v_a_1900_; lean_object* v___x_1902_; uint8_t v_isShared_1903_; uint8_t v_isSharedCheck_1935_; 
v_a_1900_ = lean_ctor_get(v___x_1899_, 0);
v_isSharedCheck_1935_ = !lean_is_exclusive(v___x_1899_);
if (v_isSharedCheck_1935_ == 0)
{
v___x_1902_ = v___x_1899_;
v_isShared_1903_ = v_isSharedCheck_1935_;
goto v_resetjp_1901_;
}
else
{
lean_inc(v_a_1900_);
lean_dec(v___x_1899_);
v___x_1902_ = lean_box(0);
v_isShared_1903_ = v_isSharedCheck_1935_;
goto v_resetjp_1901_;
}
v_resetjp_1901_:
{
lean_object* v___x_1904_; lean_object* v_currNamespace_1905_; lean_object* v_openDecls_1906_; lean_object* v_env_1907_; lean_object* v_messages_1908_; lean_object* v_scopes_1909_; lean_object* v_usedQuotCtxts_1910_; lean_object* v_nextMacroScope_1911_; lean_object* v_maxRecDepth_1912_; lean_object* v_ngen_1913_; lean_object* v_auxDeclNGen_1914_; lean_object* v_infoState_1915_; lean_object* v_traceState_1916_; lean_object* v_snapshotTasks_1917_; lean_object* v_prevLinterStates_1918_; lean_object* v___x_1920_; uint8_t v_isShared_1921_; uint8_t v_isSharedCheck_1934_; 
v___x_1904_ = lean_st_ref_take(v___y_1896_);
v_currNamespace_1905_ = lean_ctor_get(v_a_1898_, 2);
lean_inc(v_currNamespace_1905_);
lean_dec(v_a_1898_);
v_openDecls_1906_ = lean_ctor_get(v_a_1900_, 3);
lean_inc(v_openDecls_1906_);
lean_dec(v_a_1900_);
v_env_1907_ = lean_ctor_get(v___x_1904_, 0);
v_messages_1908_ = lean_ctor_get(v___x_1904_, 1);
v_scopes_1909_ = lean_ctor_get(v___x_1904_, 2);
v_usedQuotCtxts_1910_ = lean_ctor_get(v___x_1904_, 3);
v_nextMacroScope_1911_ = lean_ctor_get(v___x_1904_, 4);
v_maxRecDepth_1912_ = lean_ctor_get(v___x_1904_, 5);
v_ngen_1913_ = lean_ctor_get(v___x_1904_, 6);
v_auxDeclNGen_1914_ = lean_ctor_get(v___x_1904_, 7);
v_infoState_1915_ = lean_ctor_get(v___x_1904_, 8);
v_traceState_1916_ = lean_ctor_get(v___x_1904_, 9);
v_snapshotTasks_1917_ = lean_ctor_get(v___x_1904_, 10);
v_prevLinterStates_1918_ = lean_ctor_get(v___x_1904_, 11);
v_isSharedCheck_1934_ = !lean_is_exclusive(v___x_1904_);
if (v_isSharedCheck_1934_ == 0)
{
v___x_1920_ = v___x_1904_;
v_isShared_1921_ = v_isSharedCheck_1934_;
goto v_resetjp_1919_;
}
else
{
lean_inc(v_prevLinterStates_1918_);
lean_inc(v_snapshotTasks_1917_);
lean_inc(v_traceState_1916_);
lean_inc(v_infoState_1915_);
lean_inc(v_auxDeclNGen_1914_);
lean_inc(v_ngen_1913_);
lean_inc(v_maxRecDepth_1912_);
lean_inc(v_nextMacroScope_1911_);
lean_inc(v_usedQuotCtxts_1910_);
lean_inc(v_scopes_1909_);
lean_inc(v_messages_1908_);
lean_inc(v_env_1907_);
lean_dec(v___x_1904_);
v___x_1920_ = lean_box(0);
v_isShared_1921_ = v_isSharedCheck_1934_;
goto v_resetjp_1919_;
}
v_resetjp_1919_:
{
lean_object* v___x_1922_; lean_object* v___x_1923_; lean_object* v___x_1924_; lean_object* v___x_1925_; lean_object* v___x_1927_; 
v___x_1922_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1922_, 0, v_currNamespace_1905_);
lean_ctor_set(v___x_1922_, 1, v_openDecls_1906_);
v___x_1923_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1923_, 0, v___x_1922_);
lean_ctor_set(v___x_1923_, 1, v___y_1890_);
lean_inc_ref(v___y_1893_);
lean_inc_ref(v___y_1892_);
v___x_1924_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_1924_, 0, v___y_1892_);
lean_ctor_set(v___x_1924_, 1, v___y_1889_);
lean_ctor_set(v___x_1924_, 2, v___y_1891_);
lean_ctor_set(v___x_1924_, 3, v___y_1893_);
lean_ctor_set(v___x_1924_, 4, v___x_1923_);
lean_ctor_set_uint8(v___x_1924_, sizeof(void*)*5, v___y_1895_);
lean_ctor_set_uint8(v___x_1924_, sizeof(void*)*5 + 1, v___y_1894_);
lean_ctor_set_uint8(v___x_1924_, sizeof(void*)*5 + 2, v_isSilent_1884_);
v___x_1925_ = l_Lean_MessageLog_add(v___x_1924_, v_messages_1908_);
if (v_isShared_1921_ == 0)
{
lean_ctor_set(v___x_1920_, 1, v___x_1925_);
v___x_1927_ = v___x_1920_;
goto v_reusejp_1926_;
}
else
{
lean_object* v_reuseFailAlloc_1933_; 
v_reuseFailAlloc_1933_ = lean_alloc_ctor(0, 12, 0);
lean_ctor_set(v_reuseFailAlloc_1933_, 0, v_env_1907_);
lean_ctor_set(v_reuseFailAlloc_1933_, 1, v___x_1925_);
lean_ctor_set(v_reuseFailAlloc_1933_, 2, v_scopes_1909_);
lean_ctor_set(v_reuseFailAlloc_1933_, 3, v_usedQuotCtxts_1910_);
lean_ctor_set(v_reuseFailAlloc_1933_, 4, v_nextMacroScope_1911_);
lean_ctor_set(v_reuseFailAlloc_1933_, 5, v_maxRecDepth_1912_);
lean_ctor_set(v_reuseFailAlloc_1933_, 6, v_ngen_1913_);
lean_ctor_set(v_reuseFailAlloc_1933_, 7, v_auxDeclNGen_1914_);
lean_ctor_set(v_reuseFailAlloc_1933_, 8, v_infoState_1915_);
lean_ctor_set(v_reuseFailAlloc_1933_, 9, v_traceState_1916_);
lean_ctor_set(v_reuseFailAlloc_1933_, 10, v_snapshotTasks_1917_);
lean_ctor_set(v_reuseFailAlloc_1933_, 11, v_prevLinterStates_1918_);
v___x_1927_ = v_reuseFailAlloc_1933_;
goto v_reusejp_1926_;
}
v_reusejp_1926_:
{
lean_object* v___x_1928_; lean_object* v___x_1929_; lean_object* v___x_1931_; 
v___x_1928_ = lean_st_ref_set(v___y_1896_, v___x_1927_);
v___x_1929_ = lean_box(0);
if (v_isShared_1903_ == 0)
{
lean_ctor_set(v___x_1902_, 0, v___x_1929_);
v___x_1931_ = v___x_1902_;
goto v_reusejp_1930_;
}
else
{
lean_object* v_reuseFailAlloc_1932_; 
v_reuseFailAlloc_1932_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1932_, 0, v___x_1929_);
v___x_1931_ = v_reuseFailAlloc_1932_;
goto v_reusejp_1930_;
}
v_reusejp_1930_:
{
return v___x_1931_;
}
}
}
}
}
else
{
lean_object* v_a_1936_; lean_object* v___x_1938_; uint8_t v_isShared_1939_; uint8_t v_isSharedCheck_1943_; 
lean_dec(v_a_1898_);
lean_dec(v___y_1891_);
lean_dec_ref(v___y_1890_);
lean_dec_ref(v___y_1889_);
v_a_1936_ = lean_ctor_get(v___x_1899_, 0);
v_isSharedCheck_1943_ = !lean_is_exclusive(v___x_1899_);
if (v_isSharedCheck_1943_ == 0)
{
v___x_1938_ = v___x_1899_;
v_isShared_1939_ = v_isSharedCheck_1943_;
goto v_resetjp_1937_;
}
else
{
lean_inc(v_a_1936_);
lean_dec(v___x_1899_);
v___x_1938_ = lean_box(0);
v_isShared_1939_ = v_isSharedCheck_1943_;
goto v_resetjp_1937_;
}
v_resetjp_1937_:
{
lean_object* v___x_1941_; 
if (v_isShared_1939_ == 0)
{
v___x_1941_ = v___x_1938_;
goto v_reusejp_1940_;
}
else
{
lean_object* v_reuseFailAlloc_1942_; 
v_reuseFailAlloc_1942_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1942_, 0, v_a_1936_);
v___x_1941_ = v_reuseFailAlloc_1942_;
goto v_reusejp_1940_;
}
v_reusejp_1940_:
{
return v___x_1941_;
}
}
}
}
else
{
lean_object* v_a_1944_; lean_object* v___x_1946_; uint8_t v_isShared_1947_; uint8_t v_isSharedCheck_1951_; 
lean_dec(v___y_1891_);
lean_dec_ref(v___y_1890_);
lean_dec_ref(v___y_1889_);
v_a_1944_ = lean_ctor_get(v___x_1897_, 0);
v_isSharedCheck_1951_ = !lean_is_exclusive(v___x_1897_);
if (v_isSharedCheck_1951_ == 0)
{
v___x_1946_ = v___x_1897_;
v_isShared_1947_ = v_isSharedCheck_1951_;
goto v_resetjp_1945_;
}
else
{
lean_inc(v_a_1944_);
lean_dec(v___x_1897_);
v___x_1946_ = lean_box(0);
v_isShared_1947_ = v_isSharedCheck_1951_;
goto v_resetjp_1945_;
}
v_resetjp_1945_:
{
lean_object* v___x_1949_; 
if (v_isShared_1947_ == 0)
{
v___x_1949_ = v___x_1946_;
goto v_reusejp_1948_;
}
else
{
lean_object* v_reuseFailAlloc_1950_; 
v_reuseFailAlloc_1950_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1950_, 0, v_a_1944_);
v___x_1949_ = v_reuseFailAlloc_1950_;
goto v_reusejp_1948_;
}
v_reusejp_1948_:
{
return v___x_1949_;
}
}
}
}
v___jp_1952_:
{
lean_object* v_fileName_1958_; lean_object* v_fileMap_1959_; uint8_t v_suppressElabErrors_1960_; lean_object* v___x_1961_; lean_object* v___x_1962_; lean_object* v_a_1963_; lean_object* v___x_1965_; uint8_t v_isShared_1966_; uint8_t v_isSharedCheck_1979_; 
v_fileName_1958_ = lean_ctor_get(v___y_1885_, 0);
v_fileMap_1959_ = lean_ctor_get(v___y_1885_, 1);
v_suppressElabErrors_1960_ = lean_ctor_get_uint8(v___y_1885_, sizeof(void*)*10);
v___x_1961_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_1882_);
v___x_1962_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8_spec__18___redArg(v___x_1961_, v___y_1886_);
v_a_1963_ = lean_ctor_get(v___x_1962_, 0);
v_isSharedCheck_1979_ = !lean_is_exclusive(v___x_1962_);
if (v_isSharedCheck_1979_ == 0)
{
v___x_1965_ = v___x_1962_;
v_isShared_1966_ = v_isSharedCheck_1979_;
goto v_resetjp_1964_;
}
else
{
lean_inc(v_a_1963_);
lean_dec(v___x_1962_);
v___x_1965_ = lean_box(0);
v_isShared_1966_ = v_isSharedCheck_1979_;
goto v_resetjp_1964_;
}
v_resetjp_1964_:
{
lean_object* v___x_1967_; lean_object* v___x_1968_; lean_object* v___x_1969_; lean_object* v___x_1970_; 
lean_inc_ref_n(v_fileMap_1959_, 2);
v___x_1967_ = l_Lean_FileMap_toPosition(v_fileMap_1959_, v___y_1954_);
lean_dec(v___y_1954_);
v___x_1968_ = l_Lean_FileMap_toPosition(v_fileMap_1959_, v___y_1957_);
lean_dec(v___y_1957_);
v___x_1969_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1969_, 0, v___x_1968_);
v___x_1970_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8___closed__0));
if (v_suppressElabErrors_1960_ == 0)
{
lean_del_object(v___x_1965_);
v___y_1889_ = v___x_1967_;
v___y_1890_ = v_a_1963_;
v___y_1891_ = v___x_1969_;
v___y_1892_ = v_fileName_1958_;
v___y_1893_ = v___x_1970_;
v___y_1894_ = v___y_1955_;
v___y_1895_ = v___y_1956_;
v___y_1896_ = v___y_1886_;
goto v___jp_1888_;
}
else
{
lean_object* v___x_1971_; lean_object* v___x_1972_; lean_object* v___f_1973_; uint8_t v___x_1974_; 
v___x_1971_ = lean_box(v___y_1953_);
v___x_1972_ = lean_box(v_suppressElabErrors_1960_);
v___f_1973_ = lean_alloc_closure((void*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8___lam__0___boxed), 3, 2);
lean_closure_set(v___f_1973_, 0, v___x_1971_);
lean_closure_set(v___f_1973_, 1, v___x_1972_);
lean_inc(v_a_1963_);
v___x_1974_ = l_Lean_MessageData_hasTag(v___f_1973_, v_a_1963_);
if (v___x_1974_ == 0)
{
lean_object* v___x_1975_; lean_object* v___x_1977_; 
lean_dec_ref_known(v___x_1969_, 1);
lean_dec_ref(v___x_1967_);
lean_dec(v_a_1963_);
v___x_1975_ = lean_box(0);
if (v_isShared_1966_ == 0)
{
lean_ctor_set(v___x_1965_, 0, v___x_1975_);
v___x_1977_ = v___x_1965_;
goto v_reusejp_1976_;
}
else
{
lean_object* v_reuseFailAlloc_1978_; 
v_reuseFailAlloc_1978_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1978_, 0, v___x_1975_);
v___x_1977_ = v_reuseFailAlloc_1978_;
goto v_reusejp_1976_;
}
v_reusejp_1976_:
{
return v___x_1977_;
}
}
else
{
lean_del_object(v___x_1965_);
v___y_1889_ = v___x_1967_;
v___y_1890_ = v_a_1963_;
v___y_1891_ = v___x_1969_;
v___y_1892_ = v_fileName_1958_;
v___y_1893_ = v___x_1970_;
v___y_1894_ = v___y_1955_;
v___y_1895_ = v___y_1956_;
v___y_1896_ = v___y_1886_;
goto v___jp_1888_;
}
}
}
}
v___jp_1980_:
{
lean_object* v___x_1986_; 
v___x_1986_ = l_Lean_Syntax_getTailPos_x3f(v___y_1982_, v___y_1984_);
lean_dec(v___y_1982_);
if (lean_obj_tag(v___x_1986_) == 0)
{
lean_inc(v___y_1985_);
v___y_1953_ = v___y_1981_;
v___y_1954_ = v___y_1985_;
v___y_1955_ = v___y_1983_;
v___y_1956_ = v___y_1984_;
v___y_1957_ = v___y_1985_;
goto v___jp_1952_;
}
else
{
lean_object* v_val_1987_; 
v_val_1987_ = lean_ctor_get(v___x_1986_, 0);
lean_inc(v_val_1987_);
lean_dec_ref_known(v___x_1986_, 1);
v___y_1953_ = v___y_1981_;
v___y_1954_ = v___y_1985_;
v___y_1955_ = v___y_1983_;
v___y_1956_ = v___y_1984_;
v___y_1957_ = v_val_1987_;
goto v___jp_1952_;
}
}
v___jp_1988_:
{
lean_object* v___x_1992_; 
v___x_1992_ = l_Lean_Elab_Command_getRef___redArg(v___y_1885_);
if (lean_obj_tag(v___x_1992_) == 0)
{
lean_object* v_a_1993_; lean_object* v_ref_1994_; lean_object* v___x_1995_; 
v_a_1993_ = lean_ctor_get(v___x_1992_, 0);
lean_inc(v_a_1993_);
lean_dec_ref_known(v___x_1992_, 1);
v_ref_1994_ = l_Lean_replaceRef(v_ref_1881_, v_a_1993_);
lean_dec(v_a_1993_);
v___x_1995_ = l_Lean_Syntax_getPos_x3f(v_ref_1994_, v___y_1990_);
if (lean_obj_tag(v___x_1995_) == 0)
{
lean_object* v___x_1996_; 
v___x_1996_ = lean_unsigned_to_nat(0u);
v___y_1981_ = v___y_1989_;
v___y_1982_ = v_ref_1994_;
v___y_1983_ = v___y_1991_;
v___y_1984_ = v___y_1990_;
v___y_1985_ = v___x_1996_;
goto v___jp_1980_;
}
else
{
lean_object* v_val_1997_; 
v_val_1997_ = lean_ctor_get(v___x_1995_, 0);
lean_inc(v_val_1997_);
lean_dec_ref_known(v___x_1995_, 1);
v___y_1981_ = v___y_1989_;
v___y_1982_ = v_ref_1994_;
v___y_1983_ = v___y_1991_;
v___y_1984_ = v___y_1990_;
v___y_1985_ = v_val_1997_;
goto v___jp_1980_;
}
}
else
{
lean_object* v_a_1998_; lean_object* v___x_2000_; uint8_t v_isShared_2001_; uint8_t v_isSharedCheck_2005_; 
lean_dec_ref(v_msgData_1882_);
v_a_1998_ = lean_ctor_get(v___x_1992_, 0);
v_isSharedCheck_2005_ = !lean_is_exclusive(v___x_1992_);
if (v_isSharedCheck_2005_ == 0)
{
v___x_2000_ = v___x_1992_;
v_isShared_2001_ = v_isSharedCheck_2005_;
goto v_resetjp_1999_;
}
else
{
lean_inc(v_a_1998_);
lean_dec(v___x_1992_);
v___x_2000_ = lean_box(0);
v_isShared_2001_ = v_isSharedCheck_2005_;
goto v_resetjp_1999_;
}
v_resetjp_1999_:
{
lean_object* v___x_2003_; 
if (v_isShared_2001_ == 0)
{
v___x_2003_ = v___x_2000_;
goto v_reusejp_2002_;
}
else
{
lean_object* v_reuseFailAlloc_2004_; 
v_reuseFailAlloc_2004_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2004_, 0, v_a_1998_);
v___x_2003_ = v_reuseFailAlloc_2004_;
goto v_reusejp_2002_;
}
v_reusejp_2002_:
{
return v___x_2003_;
}
}
}
}
v___jp_2007_:
{
if (v___y_2010_ == 0)
{
v___y_1989_ = v___y_2008_;
v___y_1990_ = v___y_2009_;
v___y_1991_ = v_severity_1883_;
goto v___jp_1988_;
}
else
{
v___y_1989_ = v___y_2008_;
v___y_1990_ = v___y_2009_;
v___y_1991_ = v___x_2006_;
goto v___jp_1988_;
}
}
v___jp_2011_:
{
if (v___y_2012_ == 0)
{
lean_object* v___x_2013_; lean_object* v_scopes_2014_; lean_object* v___x_2015_; lean_object* v___x_2016_; lean_object* v_opts_2017_; uint8_t v___x_2018_; uint8_t v___x_2019_; 
v___x_2013_ = lean_st_ref_get(v___y_1886_);
v_scopes_2014_ = lean_ctor_get(v___x_2013_, 2);
lean_inc(v_scopes_2014_);
lean_dec(v___x_2013_);
v___x_2015_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_2016_ = l_List_head_x21___redArg(v___x_2015_, v_scopes_2014_);
lean_dec(v_scopes_2014_);
v_opts_2017_ = lean_ctor_get(v___x_2016_, 1);
lean_inc_ref(v_opts_2017_);
lean_dec(v___x_2016_);
v___x_2018_ = 1;
v___x_2019_ = l_Lean_instBEqMessageSeverity_beq(v_severity_1883_, v___x_2018_);
if (v___x_2019_ == 0)
{
lean_dec_ref(v_opts_2017_);
v___y_2008_ = v___y_2012_;
v___y_2009_ = v___y_2012_;
v___y_2010_ = v___x_2019_;
goto v___jp_2007_;
}
else
{
lean_object* v___x_2020_; uint8_t v___x_2021_; 
v___x_2020_ = l_Lean_warningAsError;
v___x_2021_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8_spec__19(v_opts_2017_, v___x_2020_);
lean_dec_ref(v_opts_2017_);
v___y_2008_ = v___y_2012_;
v___y_2009_ = v___y_2012_;
v___y_2010_ = v___x_2021_;
goto v___jp_2007_;
}
}
else
{
lean_object* v___x_2022_; lean_object* v___x_2023_; 
lean_dec_ref(v_msgData_1882_);
v___x_2022_ = lean_box(0);
v___x_2023_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2023_, 0, v___x_2022_);
return v___x_2023_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8___boxed(lean_object* v_ref_2026_, lean_object* v_msgData_2027_, lean_object* v_severity_2028_, lean_object* v_isSilent_2029_, lean_object* v___y_2030_, lean_object* v___y_2031_, lean_object* v___y_2032_){
_start:
{
uint8_t v_severity_boxed_2033_; uint8_t v_isSilent_boxed_2034_; lean_object* v_res_2035_; 
v_severity_boxed_2033_ = lean_unbox(v_severity_2028_);
v_isSilent_boxed_2034_ = lean_unbox(v_isSilent_2029_);
v_res_2035_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8(v_ref_2026_, v_msgData_2027_, v_severity_boxed_2033_, v_isSilent_boxed_2034_, v___y_2030_, v___y_2031_);
lean_dec(v___y_2031_);
lean_dec_ref(v___y_2030_);
lean_dec(v_ref_2026_);
return v_res_2035_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6(lean_object* v_ref_2036_, lean_object* v_msgData_2037_, lean_object* v___y_2038_, lean_object* v___y_2039_){
_start:
{
uint8_t v___x_2041_; uint8_t v___x_2042_; lean_object* v___x_2043_; 
v___x_2041_ = 1;
v___x_2042_ = 0;
v___x_2043_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8(v_ref_2036_, v_msgData_2037_, v___x_2041_, v___x_2042_, v___y_2038_, v___y_2039_);
return v___x_2043_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6___boxed(lean_object* v_ref_2044_, lean_object* v_msgData_2045_, lean_object* v___y_2046_, lean_object* v___y_2047_, lean_object* v___y_2048_){
_start:
{
lean_object* v_res_2049_; 
v_res_2049_ = lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6(v_ref_2044_, v_msgData_2045_, v___y_2046_, v___y_2047_);
lean_dec(v___y_2047_);
lean_dec_ref(v___y_2046_);
lean_dec(v_ref_2044_);
return v_res_2049_;
}
}
static lean_object* _init_lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4___closed__1(void){
_start:
{
lean_object* v___x_2051_; lean_object* v___x_2052_; 
v___x_2051_ = ((lean_object*)(lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4___closed__0));
v___x_2052_ = l_Lean_stringToMessageData(v___x_2051_);
return v___x_2052_;
}
}
static lean_object* _init_lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4___closed__3(void){
_start:
{
lean_object* v___x_2054_; lean_object* v___x_2055_; 
v___x_2054_ = ((lean_object*)(lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4___closed__2));
v___x_2055_ = l_Lean_stringToMessageData(v___x_2054_);
return v___x_2055_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4(lean_object* v_linterOption_2056_, lean_object* v_stx_2057_, lean_object* v_msg_2058_, lean_object* v___y_2059_, lean_object* v___y_2060_){
_start:
{
lean_object* v_name_2062_; lean_object* v___x_2064_; uint8_t v_isShared_2065_; uint8_t v_isSharedCheck_2080_; 
v_name_2062_ = lean_ctor_get(v_linterOption_2056_, 0);
v_isSharedCheck_2080_ = !lean_is_exclusive(v_linterOption_2056_);
if (v_isSharedCheck_2080_ == 0)
{
lean_object* v_unused_2081_; 
v_unused_2081_ = lean_ctor_get(v_linterOption_2056_, 1);
lean_dec(v_unused_2081_);
v___x_2064_ = v_linterOption_2056_;
v_isShared_2065_ = v_isSharedCheck_2080_;
goto v_resetjp_2063_;
}
else
{
lean_inc(v_name_2062_);
lean_dec(v_linterOption_2056_);
v___x_2064_ = lean_box(0);
v_isShared_2065_ = v_isSharedCheck_2080_;
goto v_resetjp_2063_;
}
v_resetjp_2063_:
{
lean_object* v___x_2066_; lean_object* v___x_2067_; lean_object* v___x_2069_; 
v___x_2066_ = lean_obj_once(&lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4___closed__1, &lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4___closed__1_once, _init_lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4___closed__1);
lean_inc(v_name_2062_);
v___x_2067_ = l_Lean_MessageData_ofName(v_name_2062_);
if (v_isShared_2065_ == 0)
{
lean_ctor_set_tag(v___x_2064_, 7);
lean_ctor_set(v___x_2064_, 1, v___x_2067_);
lean_ctor_set(v___x_2064_, 0, v___x_2066_);
v___x_2069_ = v___x_2064_;
goto v_reusejp_2068_;
}
else
{
lean_object* v_reuseFailAlloc_2079_; 
v_reuseFailAlloc_2079_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2079_, 0, v___x_2066_);
lean_ctor_set(v_reuseFailAlloc_2079_, 1, v___x_2067_);
v___x_2069_ = v_reuseFailAlloc_2079_;
goto v_reusejp_2068_;
}
v_reusejp_2068_:
{
lean_object* v___x_2070_; lean_object* v___x_2071_; lean_object* v_disable_2072_; lean_object* v___x_2073_; lean_object* v___x_2074_; lean_object* v___x_2075_; lean_object* v___x_2076_; lean_object* v___x_2077_; lean_object* v___x_2078_; 
v___x_2070_ = lean_obj_once(&lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4___closed__3, &lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4___closed__3_once, _init_lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4___closed__3);
v___x_2071_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2071_, 0, v___x_2069_);
lean_ctor_set(v___x_2071_, 1, v___x_2070_);
v_disable_2072_ = l_Lean_MessageData_note(v___x_2071_);
v___x_2073_ = l_Lean_Linter_linterMessageTag;
v___x_2074_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2074_, 0, v_msg_2058_);
lean_ctor_set(v___x_2074_, 1, v_disable_2072_);
v___x_2075_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_2075_, 0, v___x_2073_);
lean_ctor_set(v___x_2075_, 1, v___x_2074_);
v___x_2076_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_2076_, 0, v_name_2062_);
lean_ctor_set(v___x_2076_, 1, v___x_2075_);
lean_inc(v_stx_2057_);
v___x_2077_ = lean_alloc_ctor(11, 2, 0);
lean_ctor_set(v___x_2077_, 0, v_stx_2057_);
lean_ctor_set(v___x_2077_, 1, v___x_2076_);
v___x_2078_ = lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6(v_stx_2057_, v___x_2077_, v___y_2059_, v___y_2060_);
lean_dec(v_stx_2057_);
return v___x_2078_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4___boxed(lean_object* v_linterOption_2082_, lean_object* v_stx_2083_, lean_object* v_msg_2084_, lean_object* v___y_2085_, lean_object* v___y_2086_, lean_object* v___y_2087_){
_start:
{
lean_object* v_res_2088_; 
v_res_2088_ = lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4(v_linterOption_2082_, v_stx_2083_, v_msg_2084_, v___y_2085_, v___y_2086_);
lean_dec(v___y_2086_);
lean_dec_ref(v___y_2085_);
return v_res_2088_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___closed__8(void){
_start:
{
lean_object* v___x_2107_; lean_object* v___x_2108_; 
v___x_2107_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___closed__7));
v___x_2108_ = l_Lean_stringToMessageData(v___x_2107_);
return v___x_2108_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___closed__10(void){
_start:
{
lean_object* v___x_2110_; lean_object* v___x_2111_; 
v___x_2110_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___closed__9));
v___x_2111_ = l_Lean_stringToMessageData(v___x_2110_);
return v___x_2111_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6(lean_object* v_as_2112_, size_t v_sz_2113_, size_t v_i_2114_, lean_object* v_b_2115_, lean_object* v___y_2116_, lean_object* v___y_2117_){
_start:
{
lean_object* v_a_2120_; uint8_t v___x_2124_; 
v___x_2124_ = lean_usize_dec_lt(v_i_2114_, v_sz_2113_);
if (v___x_2124_ == 0)
{
lean_object* v___x_2125_; 
v___x_2125_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2125_, 0, v_b_2115_);
return v___x_2125_;
}
else
{
lean_object* v_a_2126_; lean_object* v_fst_2127_; lean_object* v_snd_2128_; lean_object* v___x_2130_; uint8_t v_isShared_2131_; uint8_t v_isSharedCheck_2169_; 
v_a_2126_ = lean_array_uget(v_as_2112_, v_i_2114_);
v_fst_2127_ = lean_ctor_get(v_a_2126_, 0);
v_snd_2128_ = lean_ctor_get(v_a_2126_, 1);
v_isSharedCheck_2169_ = !lean_is_exclusive(v_a_2126_);
if (v_isSharedCheck_2169_ == 0)
{
v___x_2130_ = v_a_2126_;
v_isShared_2131_ = v_isSharedCheck_2169_;
goto v_resetjp_2129_;
}
else
{
lean_inc(v_snd_2128_);
lean_inc(v_fst_2127_);
lean_dec(v_a_2126_);
v___x_2130_ = lean_box(0);
v_isShared_2131_ = v_isSharedCheck_2169_;
goto v_resetjp_2129_;
}
v_resetjp_2129_:
{
lean_object* v___x_2132_; lean_object* v___x_2133_; lean_object* v___x_2134_; uint8_t v___x_2135_; 
v___x_2132_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___closed__0));
lean_inc(v_snd_2128_);
v___x_2133_ = l_Lean_Syntax_getKind(v_snd_2128_);
v___x_2134_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___closed__6));
v___x_2135_ = l_List_elem___redArg(v___x_2132_, v___x_2133_, v___x_2134_);
if (v___x_2135_ == 0)
{
lean_object* v_start_2136_; lean_object* v_stop_2137_; lean_object* v_start_2138_; lean_object* v_stop_2139_; lean_object* v___x_2140_; uint8_t v___y_2142_; uint8_t v___x_2167_; 
v_start_2136_ = lean_ctor_get(v_b_2115_, 0);
v_stop_2137_ = lean_ctor_get(v_b_2115_, 1);
v_start_2138_ = lean_ctor_get(v_fst_2127_, 0);
v_stop_2139_ = lean_ctor_get(v_fst_2127_, 1);
v___x_2140_ = lp_mathlib_Mathlib_Linter_linter_unusedTactic;
v___x_2167_ = lean_nat_dec_le(v_start_2136_, v_start_2138_);
if (v___x_2167_ == 0)
{
v___y_2142_ = v___x_2167_;
goto v___jp_2141_;
}
else
{
uint8_t v___x_2168_; 
v___x_2168_ = lean_nat_dec_le(v_stop_2139_, v_stop_2137_);
v___y_2142_ = v___x_2168_;
goto v___jp_2141_;
}
v___jp_2141_:
{
if (v___y_2142_ == 0)
{
lean_object* v___x_2144_; uint8_t v_isShared_2145_; uint8_t v_isSharedCheck_2164_; 
v_isSharedCheck_2164_ = !lean_is_exclusive(v_b_2115_);
if (v_isSharedCheck_2164_ == 0)
{
lean_object* v_unused_2165_; lean_object* v_unused_2166_; 
v_unused_2165_ = lean_ctor_get(v_b_2115_, 1);
lean_dec(v_unused_2165_);
v_unused_2166_ = lean_ctor_get(v_b_2115_, 0);
lean_dec(v_unused_2166_);
v___x_2144_ = v_b_2115_;
v_isShared_2145_ = v_isSharedCheck_2164_;
goto v_resetjp_2143_;
}
else
{
lean_dec(v_b_2115_);
v___x_2144_ = lean_box(0);
v_isShared_2145_ = v_isSharedCheck_2164_;
goto v_resetjp_2143_;
}
v_resetjp_2143_:
{
lean_object* v___x_2146_; lean_object* v___x_2147_; lean_object* v___x_2149_; 
v___x_2146_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___closed__8, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___closed__8_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___closed__8);
lean_inc(v_snd_2128_);
v___x_2147_ = l_Lean_MessageData_ofSyntax(v_snd_2128_);
if (v_isShared_2145_ == 0)
{
lean_ctor_set_tag(v___x_2144_, 7);
lean_ctor_set(v___x_2144_, 1, v___x_2147_);
lean_ctor_set(v___x_2144_, 0, v___x_2146_);
v___x_2149_ = v___x_2144_;
goto v_reusejp_2148_;
}
else
{
lean_object* v_reuseFailAlloc_2163_; 
v_reuseFailAlloc_2163_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2163_, 0, v___x_2146_);
lean_ctor_set(v_reuseFailAlloc_2163_, 1, v___x_2147_);
v___x_2149_ = v_reuseFailAlloc_2163_;
goto v_reusejp_2148_;
}
v_reusejp_2148_:
{
lean_object* v___x_2150_; lean_object* v___x_2152_; 
v___x_2150_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___closed__10, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___closed__10_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___closed__10);
if (v_isShared_2131_ == 0)
{
lean_ctor_set_tag(v___x_2130_, 7);
lean_ctor_set(v___x_2130_, 1, v___x_2150_);
lean_ctor_set(v___x_2130_, 0, v___x_2149_);
v___x_2152_ = v___x_2130_;
goto v_reusejp_2151_;
}
else
{
lean_object* v_reuseFailAlloc_2162_; 
v_reuseFailAlloc_2162_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2162_, 0, v___x_2149_);
lean_ctor_set(v_reuseFailAlloc_2162_, 1, v___x_2150_);
v___x_2152_ = v_reuseFailAlloc_2162_;
goto v_reusejp_2151_;
}
v_reusejp_2151_:
{
lean_object* v___x_2153_; 
v___x_2153_ = lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4(v___x_2140_, v_snd_2128_, v___x_2152_, v___y_2116_, v___y_2117_);
if (lean_obj_tag(v___x_2153_) == 0)
{
lean_dec_ref_known(v___x_2153_, 1);
v_a_2120_ = v_fst_2127_;
goto v___jp_2119_;
}
else
{
lean_object* v_a_2154_; lean_object* v___x_2156_; uint8_t v_isShared_2157_; uint8_t v_isSharedCheck_2161_; 
lean_dec(v_fst_2127_);
v_a_2154_ = lean_ctor_get(v___x_2153_, 0);
v_isSharedCheck_2161_ = !lean_is_exclusive(v___x_2153_);
if (v_isSharedCheck_2161_ == 0)
{
v___x_2156_ = v___x_2153_;
v_isShared_2157_ = v_isSharedCheck_2161_;
goto v_resetjp_2155_;
}
else
{
lean_inc(v_a_2154_);
lean_dec(v___x_2153_);
v___x_2156_ = lean_box(0);
v_isShared_2157_ = v_isSharedCheck_2161_;
goto v_resetjp_2155_;
}
v_resetjp_2155_:
{
lean_object* v___x_2159_; 
if (v_isShared_2157_ == 0)
{
v___x_2159_ = v___x_2156_;
goto v_reusejp_2158_;
}
else
{
lean_object* v_reuseFailAlloc_2160_; 
v_reuseFailAlloc_2160_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2160_, 0, v_a_2154_);
v___x_2159_ = v_reuseFailAlloc_2160_;
goto v_reusejp_2158_;
}
v_reusejp_2158_:
{
return v___x_2159_;
}
}
}
}
}
}
}
else
{
lean_del_object(v___x_2130_);
lean_dec(v_snd_2128_);
lean_dec(v_fst_2127_);
v_a_2120_ = v_b_2115_;
goto v___jp_2119_;
}
}
}
else
{
lean_del_object(v___x_2130_);
lean_dec(v_snd_2128_);
lean_dec(v_fst_2127_);
v_a_2120_ = v_b_2115_;
goto v___jp_2119_;
}
}
}
v___jp_2119_:
{
size_t v___x_2121_; size_t v___x_2122_; 
v___x_2121_ = ((size_t)1ULL);
v___x_2122_ = lean_usize_add(v_i_2114_, v___x_2121_);
v_i_2114_ = v___x_2122_;
v_b_2115_ = v_a_2120_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___boxed(lean_object* v_as_2170_, lean_object* v_sz_2171_, lean_object* v_i_2172_, lean_object* v_b_2173_, lean_object* v___y_2174_, lean_object* v___y_2175_, lean_object* v___y_2176_){
_start:
{
size_t v_sz_boxed_2177_; size_t v_i_boxed_2178_; lean_object* v_res_2179_; 
v_sz_boxed_2177_ = lean_unbox_usize(v_sz_2171_);
lean_dec(v_sz_2171_);
v_i_boxed_2178_ = lean_unbox_usize(v_i_2172_);
lean_dec(v_i_2172_);
v_res_2179_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6(v_as_2170_, v_sz_boxed_2177_, v_i_boxed_2178_, v_b_2173_, v___y_2174_, v___y_2175_);
lean_dec(v___y_2175_);
lean_dec_ref(v___y_2174_);
lean_dec_ref(v_as_2170_);
return v_res_2179_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___lam__0___closed__0(void){
_start:
{
lean_object* v___x_2180_; lean_object* v___x_2181_; lean_object* v___x_2182_; 
v___x_2180_ = lean_box(0);
v___x_2181_ = lean_unsigned_to_nat(16u);
v___x_2182_ = lean_mk_array(v___x_2181_, v___x_2180_);
return v___x_2182_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___lam__0___closed__1(void){
_start:
{
lean_object* v___x_2183_; lean_object* v___x_2184_; lean_object* v___x_2185_; 
v___x_2183_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___lam__0___closed__0, &lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___lam__0___closed__0_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___lam__0___closed__0);
v___x_2184_ = lean_unsigned_to_nat(0u);
v___x_2185_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2185_, 0, v___x_2184_);
lean_ctor_set(v___x_2185_, 1, v___x_2183_);
return v___x_2185_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___lam__0___closed__7(void){
_start:
{
lean_object* v___x_2193_; lean_object* v___x_2194_; lean_object* v___x_2195_; 
v___x_2193_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___lam__0___closed__6));
v___x_2194_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6___closed__0));
v___x_2195_ = l_Std_HashSet_instInhabited(lean_box(0), v___x_2194_, v___x_2193_);
return v___x_2195_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___lam__0(lean_object* v_stx_2196_, lean_object* v___y_2197_, lean_object* v___y_2198_){
_start:
{
lean_object* v___y_2201_; lean_object* v___y_2202_; lean_object* v___y_2224_; lean_object* v___y_2225_; lean_object* v___y_2226_; lean_object* v___y_2227_; lean_object* v___y_2228_; lean_object* v___y_2231_; lean_object* v___y_2232_; lean_object* v___y_2233_; lean_object* v___y_2234_; lean_object* v___y_2235_; lean_object* v___y_2238_; lean_object* v___y_2239_; lean_object* v___y_2247_; lean_object* v___y_2248_; lean_object* v___y_2249_; lean_object* v___y_2277_; uint8_t v___y_2278_; lean_object* v___y_2279_; lean_object* v___y_2280_; lean_object* v___y_2281_; lean_object* v___x_2289_; lean_object* v_a_2290_; lean_object* v___x_2292_; uint8_t v_isShared_2293_; uint8_t v_isSharedCheck_2355_; 
v___x_2289_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__1(v___y_2197_, v___y_2198_);
v_a_2290_ = lean_ctor_get(v___x_2289_, 0);
v_isSharedCheck_2355_ = !lean_is_exclusive(v___x_2289_);
if (v_isSharedCheck_2355_ == 0)
{
v___x_2292_ = v___x_2289_;
v_isShared_2293_ = v_isSharedCheck_2355_;
goto v_resetjp_2291_;
}
else
{
lean_inc(v_a_2290_);
lean_dec(v___x_2289_);
v___x_2292_ = lean_box(0);
v_isShared_2293_ = v_isSharedCheck_2355_;
goto v_resetjp_2291_;
}
v___jp_2200_:
{
size_t v_sz_2203_; size_t v___x_2204_; lean_object* v___x_2205_; 
v_sz_2203_ = lean_array_size(v___y_2202_);
v___x_2204_ = ((size_t)0ULL);
v___x_2205_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__6(v___y_2202_, v_sz_2203_, v___x_2204_, v___y_2201_, v___y_2197_, v___y_2198_);
lean_dec_ref(v___y_2202_);
if (lean_obj_tag(v___x_2205_) == 0)
{
lean_object* v___x_2207_; uint8_t v_isShared_2208_; uint8_t v_isSharedCheck_2213_; 
v_isSharedCheck_2213_ = !lean_is_exclusive(v___x_2205_);
if (v_isSharedCheck_2213_ == 0)
{
lean_object* v_unused_2214_; 
v_unused_2214_ = lean_ctor_get(v___x_2205_, 0);
lean_dec(v_unused_2214_);
v___x_2207_ = v___x_2205_;
v_isShared_2208_ = v_isSharedCheck_2213_;
goto v_resetjp_2206_;
}
else
{
lean_dec(v___x_2205_);
v___x_2207_ = lean_box(0);
v_isShared_2208_ = v_isSharedCheck_2213_;
goto v_resetjp_2206_;
}
v_resetjp_2206_:
{
lean_object* v___x_2209_; lean_object* v___x_2211_; 
v___x_2209_ = lean_box(0);
if (v_isShared_2208_ == 0)
{
lean_ctor_set(v___x_2207_, 0, v___x_2209_);
v___x_2211_ = v___x_2207_;
goto v_reusejp_2210_;
}
else
{
lean_object* v_reuseFailAlloc_2212_; 
v_reuseFailAlloc_2212_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2212_, 0, v___x_2209_);
v___x_2211_ = v_reuseFailAlloc_2212_;
goto v_reusejp_2210_;
}
v_reusejp_2210_:
{
return v___x_2211_;
}
}
}
else
{
lean_object* v_a_2215_; lean_object* v___x_2217_; uint8_t v_isShared_2218_; uint8_t v_isSharedCheck_2222_; 
v_a_2215_ = lean_ctor_get(v___x_2205_, 0);
v_isSharedCheck_2222_ = !lean_is_exclusive(v___x_2205_);
if (v_isSharedCheck_2222_ == 0)
{
v___x_2217_ = v___x_2205_;
v_isShared_2218_ = v_isSharedCheck_2222_;
goto v_resetjp_2216_;
}
else
{
lean_inc(v_a_2215_);
lean_dec(v___x_2205_);
v___x_2217_ = lean_box(0);
v_isShared_2218_ = v_isSharedCheck_2222_;
goto v_resetjp_2216_;
}
v_resetjp_2216_:
{
lean_object* v___x_2220_; 
if (v_isShared_2218_ == 0)
{
v___x_2220_ = v___x_2217_;
goto v_reusejp_2219_;
}
else
{
lean_object* v_reuseFailAlloc_2221_; 
v_reuseFailAlloc_2221_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2221_, 0, v_a_2215_);
v___x_2220_ = v_reuseFailAlloc_2221_;
goto v_reusejp_2219_;
}
v_reusejp_2219_:
{
return v___x_2220_;
}
}
}
}
v___jp_2223_:
{
lean_object* v___x_2229_; 
v___x_2229_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__7___redArg(v___y_2224_, v___y_2227_, v___y_2226_, v___y_2228_);
lean_dec(v___y_2228_);
lean_dec(v___y_2224_);
v___y_2201_ = v___y_2225_;
v___y_2202_ = v___x_2229_;
goto v___jp_2200_;
}
v___jp_2230_:
{
uint8_t v___x_2236_; 
v___x_2236_ = lean_nat_dec_le(v___y_2235_, v___y_2233_);
if (v___x_2236_ == 0)
{
lean_dec(v___y_2233_);
lean_inc(v___y_2235_);
v___y_2224_ = v___y_2231_;
v___y_2225_ = v___y_2232_;
v___y_2226_ = v___y_2235_;
v___y_2227_ = v___y_2234_;
v___y_2228_ = v___y_2235_;
goto v___jp_2223_;
}
else
{
v___y_2224_ = v___y_2231_;
v___y_2225_ = v___y_2232_;
v___y_2226_ = v___y_2235_;
v___y_2227_ = v___y_2234_;
v___y_2228_ = v___y_2233_;
goto v___jp_2223_;
}
}
v___jp_2237_:
{
lean_object* v___x_2240_; lean_object* v___x_2241_; uint8_t v___x_2242_; 
lean_inc_n(v___y_2238_, 2);
v___x_2240_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2240_, 0, v___y_2238_);
lean_ctor_set(v___x_2240_, 1, v___y_2238_);
v___x_2241_ = lean_array_get_size(v___y_2239_);
v___x_2242_ = lean_nat_dec_eq(v___x_2241_, v___y_2238_);
if (v___x_2242_ == 0)
{
lean_object* v___x_2243_; lean_object* v___x_2244_; uint8_t v___x_2245_; 
v___x_2243_ = lean_unsigned_to_nat(1u);
v___x_2244_ = lean_nat_sub(v___x_2241_, v___x_2243_);
v___x_2245_ = lean_nat_dec_le(v___y_2238_, v___x_2244_);
if (v___x_2245_ == 0)
{
lean_dec(v___y_2238_);
lean_inc(v___x_2244_);
v___y_2231_ = v___x_2241_;
v___y_2232_ = v___x_2240_;
v___y_2233_ = v___x_2244_;
v___y_2234_ = v___y_2239_;
v___y_2235_ = v___x_2244_;
goto v___jp_2230_;
}
else
{
v___y_2231_ = v___x_2241_;
v___y_2232_ = v___x_2240_;
v___y_2233_ = v___x_2244_;
v___y_2234_ = v___y_2239_;
v___y_2235_ = v___y_2238_;
goto v___jp_2230_;
}
}
else
{
lean_dec(v___y_2238_);
v___y_2201_ = v___x_2240_;
v___y_2202_ = v___y_2239_;
goto v___jp_2200_;
}
}
v___jp_2246_:
{
if (lean_obj_tag(v___y_2249_) == 0)
{
lean_object* v___x_2250_; lean_object* v_size_2251_; lean_object* v_buckets_2252_; lean_object* v___x_2253_; lean_object* v___x_2254_; uint8_t v___x_2255_; 
lean_dec_ref_known(v___y_2249_, 1);
v___x_2250_ = lean_st_ref_get(v___y_2247_);
lean_dec(v___y_2247_);
v_size_2251_ = lean_ctor_get(v___x_2250_, 0);
lean_inc(v_size_2251_);
v_buckets_2252_ = lean_ctor_get(v___x_2250_, 1);
lean_inc_ref(v_buckets_2252_);
lean_dec(v___x_2250_);
v___x_2253_ = lean_mk_empty_array_with_capacity(v_size_2251_);
lean_dec(v_size_2251_);
v___x_2254_ = lean_array_get_size(v_buckets_2252_);
v___x_2255_ = lean_nat_dec_lt(v___y_2248_, v___x_2254_);
if (v___x_2255_ == 0)
{
lean_dec_ref(v_buckets_2252_);
v___y_2238_ = v___y_2248_;
v___y_2239_ = v___x_2253_;
goto v___jp_2237_;
}
else
{
uint8_t v___x_2256_; 
v___x_2256_ = lean_nat_dec_le(v___x_2254_, v___x_2254_);
if (v___x_2256_ == 0)
{
if (v___x_2255_ == 0)
{
lean_dec_ref(v_buckets_2252_);
v___y_2238_ = v___y_2248_;
v___y_2239_ = v___x_2253_;
goto v___jp_2237_;
}
else
{
size_t v___x_2257_; size_t v___x_2258_; lean_object* v___x_2259_; 
v___x_2257_ = ((size_t)0ULL);
v___x_2258_ = lean_usize_of_nat(v___x_2254_);
v___x_2259_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__9(v_buckets_2252_, v___x_2257_, v___x_2258_, v___x_2253_);
lean_dec_ref(v_buckets_2252_);
v___y_2238_ = v___y_2248_;
v___y_2239_ = v___x_2259_;
goto v___jp_2237_;
}
}
else
{
size_t v___x_2260_; size_t v___x_2261_; lean_object* v___x_2262_; 
v___x_2260_ = ((size_t)0ULL);
v___x_2261_ = lean_usize_of_nat(v___x_2254_);
v___x_2262_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__9(v_buckets_2252_, v___x_2260_, v___x_2261_, v___x_2253_);
lean_dec_ref(v_buckets_2252_);
v___y_2238_ = v___y_2248_;
v___y_2239_ = v___x_2262_;
goto v___jp_2237_;
}
}
}
else
{
lean_object* v_a_2263_; lean_object* v___x_2265_; uint8_t v_isShared_2266_; uint8_t v_isSharedCheck_2275_; 
lean_dec(v___y_2248_);
lean_dec(v___y_2247_);
v_a_2263_ = lean_ctor_get(v___y_2249_, 0);
v_isSharedCheck_2275_ = !lean_is_exclusive(v___y_2249_);
if (v_isSharedCheck_2275_ == 0)
{
v___x_2265_ = v___y_2249_;
v_isShared_2266_ = v_isSharedCheck_2275_;
goto v_resetjp_2264_;
}
else
{
lean_inc(v_a_2263_);
lean_dec(v___y_2249_);
v___x_2265_ = lean_box(0);
v_isShared_2266_ = v_isSharedCheck_2275_;
goto v_resetjp_2264_;
}
v_resetjp_2264_:
{
lean_object* v_ref_2267_; lean_object* v___x_2268_; lean_object* v___x_2269_; lean_object* v___x_2270_; lean_object* v___x_2271_; lean_object* v___x_2273_; 
v_ref_2267_ = lean_ctor_get(v___y_2197_, 7);
v___x_2268_ = lean_io_error_to_string(v_a_2263_);
v___x_2269_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2269_, 0, v___x_2268_);
v___x_2270_ = l_Lean_MessageData_ofFormat(v___x_2269_);
lean_inc(v_ref_2267_);
v___x_2271_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2271_, 0, v_ref_2267_);
lean_ctor_set(v___x_2271_, 1, v___x_2270_);
if (v_isShared_2266_ == 0)
{
lean_ctor_set(v___x_2265_, 0, v___x_2271_);
v___x_2273_ = v___x_2265_;
goto v_reusejp_2272_;
}
else
{
lean_object* v_reuseFailAlloc_2274_; 
v_reuseFailAlloc_2274_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2274_, 0, v___x_2271_);
v___x_2273_ = v_reuseFailAlloc_2274_;
goto v_reusejp_2272_;
}
v_reusejp_2272_:
{
return v___x_2273_;
}
}
}
}
v___jp_2276_:
{
lean_object* v___x_2282_; lean_object* v___x_2283_; lean_object* v___x_2284_; lean_object* v___x_2285_; lean_object* v___x_2286_; lean_object* v___x_2287_; 
v___x_2282_ = lean_unsigned_to_nat(0u);
v___x_2283_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___lam__0___closed__1, &lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___lam__0___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___lam__0___closed__1);
v___x_2284_ = lean_st_mk_ref(v___x_2283_);
v___x_2285_ = lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_ignoreTacticKindsRef;
v___x_2286_ = lean_st_ref_get(v___x_2285_);
v___x_2287_ = lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__10(v___y_2279_, v___y_2280_, v___y_2278_, v___x_2286_, v_stx_2196_, v___x_2284_);
lean_dec(v___x_2286_);
lean_dec_ref(v___y_2280_);
lean_dec_ref(v___y_2279_);
if (lean_obj_tag(v___x_2287_) == 0)
{
lean_object* v___x_2288_; 
lean_dec_ref_known(v___x_2287_, 1);
v___x_2288_ = lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_eraseUsedTactics(v___y_2281_, v___y_2277_, v___x_2284_);
lean_dec_ref(v___y_2277_);
v___y_2247_ = v___x_2284_;
v___y_2248_ = v___x_2282_;
v___y_2249_ = v___x_2288_;
goto v___jp_2246_;
}
else
{
lean_dec_ref(v___y_2281_);
lean_dec_ref(v___y_2277_);
v___y_2247_ = v___x_2284_;
v___y_2248_ = v___x_2282_;
v___y_2249_ = v___x_2287_;
goto v___jp_2246_;
}
}
v_resetjp_2291_:
{
lean_object* v___x_2294_; uint8_t v___y_2296_; lean_object* v___x_2351_; uint8_t v___x_2352_; 
v___x_2294_ = lean_st_ref_get(v___y_2198_);
v___x_2351_ = lp_mathlib_Mathlib_Linter_linter_unusedTactic;
v___x_2352_ = l_Lean_Linter_getLinterValue(v___x_2351_, v_a_2290_);
lean_dec(v_a_2290_);
if (v___x_2352_ == 0)
{
lean_dec(v___x_2294_);
v___y_2296_ = v___x_2352_;
goto v___jp_2295_;
}
else
{
lean_object* v_infoState_2353_; uint8_t v_enabled_2354_; 
v_infoState_2353_ = lean_ctor_get(v___x_2294_, 8);
lean_inc_ref(v_infoState_2353_);
lean_dec(v___x_2294_);
v_enabled_2354_ = lean_ctor_get_uint8(v_infoState_2353_, sizeof(void*)*3);
lean_dec_ref(v_infoState_2353_);
v___y_2296_ = v_enabled_2354_;
goto v___jp_2295_;
}
v___jp_2295_:
{
if (v___y_2296_ == 0)
{
lean_object* v___x_2297_; lean_object* v___x_2299_; 
lean_dec(v_stx_2196_);
v___x_2297_ = lean_box(0);
if (v_isShared_2293_ == 0)
{
lean_ctor_set(v___x_2292_, 0, v___x_2297_);
v___x_2299_ = v___x_2292_;
goto v_reusejp_2298_;
}
else
{
lean_object* v_reuseFailAlloc_2300_; 
v_reuseFailAlloc_2300_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2300_, 0, v___x_2297_);
v___x_2299_ = v_reuseFailAlloc_2300_;
goto v_reusejp_2298_;
}
v_reusejp_2298_:
{
return v___x_2299_;
}
}
else
{
lean_object* v___x_2301_; lean_object* v_messages_2302_; uint8_t v___x_2303_; 
v___x_2301_ = lean_st_ref_get(v___y_2198_);
v_messages_2302_ = lean_ctor_get(v___x_2301_, 1);
lean_inc_ref(v_messages_2302_);
lean_dec(v___x_2301_);
v___x_2303_ = l_Lean_MessageLog_hasErrors(v_messages_2302_);
lean_dec_ref(v_messages_2302_);
if (v___x_2303_ == 0)
{
lean_object* v___x_2304_; lean_object* v_env_2305_; lean_object* v___x_2306_; lean_object* v_ext_2307_; lean_object* v_toEnvExtension_2308_; lean_object* v_asyncMode_2309_; lean_object* v___x_2310_; lean_object* v___x_2311_; lean_object* v_categories_2312_; lean_object* v___x_2313_; lean_object* v___x_2314_; 
v___x_2304_ = lean_st_ref_get(v___y_2198_);
v_env_2305_ = lean_ctor_get(v___x_2304_, 0);
lean_inc_ref_n(v_env_2305_, 2);
lean_dec(v___x_2304_);
v___x_2306_ = l_Lean_Parser_parserExtension;
v_ext_2307_ = lean_ctor_get(v___x_2306_, 1);
v_toEnvExtension_2308_ = lean_ctor_get(v_ext_2307_, 0);
v_asyncMode_2309_ = lean_ctor_get(v_toEnvExtension_2308_, 2);
v___x_2310_ = l_Lean_Parser_ParserExtension_instInhabitedState_default;
v___x_2311_ = l_Lean_ScopedEnvExtension_getState___redArg(v___x_2310_, v___x_2306_, v_env_2305_, v_asyncMode_2309_);
v_categories_2312_ = lean_ctor_get(v___x_2311_, 2);
lean_inc_ref(v_categories_2312_);
lean_dec(v___x_2311_);
v___x_2313_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___lam__0___closed__3));
v___x_2314_ = lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__2___redArg(v_categories_2312_, v___x_2313_);
if (lean_obj_tag(v___x_2314_) == 0)
{
lean_object* v___x_2315_; lean_object* v___x_2317_; 
lean_dec_ref(v_categories_2312_);
lean_dec_ref(v_env_2305_);
lean_dec(v_stx_2196_);
v___x_2315_ = lean_box(0);
if (v_isShared_2293_ == 0)
{
lean_ctor_set(v___x_2292_, 0, v___x_2315_);
v___x_2317_ = v___x_2292_;
goto v_reusejp_2316_;
}
else
{
lean_object* v_reuseFailAlloc_2318_; 
v_reuseFailAlloc_2318_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2318_, 0, v___x_2315_);
v___x_2317_ = v_reuseFailAlloc_2318_;
goto v_reusejp_2316_;
}
v_reusejp_2316_:
{
return v___x_2317_;
}
}
else
{
lean_object* v_val_2319_; lean_object* v___x_2320_; lean_object* v___x_2321_; 
v_val_2319_ = lean_ctor_get(v___x_2314_, 0);
lean_inc(v_val_2319_);
lean_dec_ref_known(v___x_2314_, 1);
v___x_2320_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___lam__0___closed__5));
v___x_2321_ = lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__2___redArg(v_categories_2312_, v___x_2320_);
lean_dec_ref(v_categories_2312_);
if (lean_obj_tag(v___x_2321_) == 0)
{
lean_object* v___x_2322_; lean_object* v___x_2324_; 
lean_dec(v_val_2319_);
lean_dec_ref(v_env_2305_);
lean_dec(v_stx_2196_);
v___x_2322_ = lean_box(0);
if (v_isShared_2293_ == 0)
{
lean_ctor_set(v___x_2292_, 0, v___x_2322_);
v___x_2324_ = v___x_2292_;
goto v_reusejp_2323_;
}
else
{
lean_object* v_reuseFailAlloc_2325_; 
v_reuseFailAlloc_2325_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2325_, 0, v___x_2322_);
v___x_2324_ = v_reuseFailAlloc_2325_;
goto v_reusejp_2323_;
}
v_reusejp_2323_:
{
return v___x_2324_;
}
}
else
{
lean_object* v_val_2326_; lean_object* v___x_2327_; lean_object* v_a_2328_; lean_object* v___x_2329_; lean_object* v___x_2330_; lean_object* v_kinds_2331_; lean_object* v_kinds_2332_; lean_object* v___x_2333_; lean_object* v_toEnvExtension_2334_; lean_object* v_asyncMode_2335_; lean_object* v_size_2336_; lean_object* v_buckets_2337_; lean_object* v___x_2338_; lean_object* v___x_2339_; lean_object* v_r_2340_; lean_object* v_size_2341_; uint8_t v___x_2342_; 
lean_del_object(v___x_2292_);
v_val_2326_ = lean_ctor_get(v___x_2321_, 0);
lean_inc(v_val_2326_);
lean_dec_ref_known(v___x_2321_, 1);
v___x_2327_ = lp_mathlib_Lean_Elab_getInfoTrees___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__3___redArg(v___y_2198_);
v_a_2328_ = lean_ctor_get(v___x_2327_, 0);
lean_inc(v_a_2328_);
lean_dec_ref(v___x_2327_);
v___x_2329_ = lp_mathlib_Mathlib_Linter_UnusedTactic_allowedRef;
v___x_2330_ = lean_st_ref_get(v___x_2329_);
v_kinds_2331_ = lean_ctor_get(v_val_2319_, 1);
lean_inc_ref(v_kinds_2331_);
lean_dec(v_val_2319_);
v_kinds_2332_ = lean_ctor_get(v_val_2326_, 1);
lean_inc_ref(v_kinds_2332_);
lean_dec(v_val_2326_);
v___x_2333_ = lp_mathlib_Mathlib_Linter_UnusedTactic_allowedUnusedTacticExt;
v_toEnvExtension_2334_ = lean_ctor_get(v___x_2333_, 0);
v_asyncMode_2335_ = lean_ctor_get(v_toEnvExtension_2334_, 2);
v_size_2336_ = lean_ctor_get(v___x_2330_, 0);
lean_inc(v_size_2336_);
v_buckets_2337_ = lean_ctor_get(v___x_2330_, 1);
lean_inc_ref(v_buckets_2337_);
v___x_2338_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___lam__0___closed__7, &lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___lam__0___closed__7_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___lam__0___closed__7);
v___x_2339_ = lean_box(0);
v_r_2340_ = l_Lean_SimplePersistentEnvExtension_getState___redArg(v___x_2338_, v___x_2333_, v_env_2305_, v_asyncMode_2335_, v___x_2339_);
v_size_2341_ = lean_ctor_get(v_r_2340_, 0);
lean_inc(v_size_2341_);
v___x_2342_ = lean_nat_dec_le(v_size_2336_, v_size_2341_);
lean_dec(v_size_2341_);
lean_dec(v_size_2336_);
if (v___x_2342_ == 0)
{
lean_object* v___x_2343_; 
lean_dec_ref(v_buckets_2337_);
v___x_2343_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__11(v___x_2330_, v_r_2340_);
lean_dec(v_r_2340_);
v___y_2277_ = v_a_2328_;
v___y_2278_ = v___y_2296_;
v___y_2279_ = v_kinds_2331_;
v___y_2280_ = v_kinds_2332_;
v___y_2281_ = v___x_2343_;
goto v___jp_2276_;
}
else
{
size_t v_sz_2344_; size_t v___x_2345_; lean_object* v___x_2346_; 
lean_dec(v___x_2330_);
v_sz_2344_ = lean_array_size(v_buckets_2337_);
v___x_2345_ = ((size_t)0ULL);
v___x_2346_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__13(v_buckets_2337_, v_sz_2344_, v___x_2345_, v_r_2340_);
lean_dec_ref(v_buckets_2337_);
v___y_2277_ = v_a_2328_;
v___y_2278_ = v___y_2296_;
v___y_2279_ = v_kinds_2331_;
v___y_2280_ = v_kinds_2332_;
v___y_2281_ = v___x_2346_;
goto v___jp_2276_;
}
}
}
}
else
{
lean_object* v___x_2347_; lean_object* v___x_2349_; 
lean_dec(v_stx_2196_);
v___x_2347_ = lean_box(0);
if (v_isShared_2293_ == 0)
{
lean_ctor_set(v___x_2292_, 0, v___x_2347_);
v___x_2349_ = v___x_2292_;
goto v_reusejp_2348_;
}
else
{
lean_object* v_reuseFailAlloc_2350_; 
v_reuseFailAlloc_2350_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2350_, 0, v___x_2347_);
v___x_2349_ = v_reuseFailAlloc_2350_;
goto v_reusejp_2348_;
}
v_reusejp_2348_:
{
return v___x_2349_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___lam__0___boxed(lean_object* v_stx_2356_, lean_object* v___y_2357_, lean_object* v___y_2358_, lean_object* v___y_2359_){
_start:
{
lean_object* v_res_2360_; 
v_res_2360_ = lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter___lam__0(v_stx_2356_, v___y_2357_, v___y_2358_);
lean_dec(v___y_2358_);
lean_dec_ref(v___y_2357_);
return v_res_2360_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__1_spec__1(lean_object* v_o_2400_, lean_object* v___y_2401_, lean_object* v___y_2402_){
_start:
{
lean_object* v___x_2404_; 
v___x_2404_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__1_spec__1___redArg(v_o_2400_, v___y_2402_);
return v___x_2404_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__1_spec__1___boxed(lean_object* v_o_2405_, lean_object* v___y_2406_, lean_object* v___y_2407_, lean_object* v___y_2408_){
_start:
{
lean_object* v_res_2409_; 
v_res_2409_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__1_spec__1(v_o_2405_, v___y_2406_, v___y_2407_);
lean_dec(v___y_2407_);
lean_dec_ref(v___y_2406_);
return v_res_2409_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__2(lean_object* v_00_u03b2_2410_, lean_object* v_x_2411_, lean_object* v_x_2412_){
_start:
{
lean_object* v___x_2413_; 
v___x_2413_ = lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__2___redArg(v_x_2411_, v_x_2412_);
return v___x_2413_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__2___boxed(lean_object* v_00_u03b2_2414_, lean_object* v_x_2415_, lean_object* v_x_2416_){
_start:
{
lean_object* v_res_2417_; 
v_res_2417_ = lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__2(v_00_u03b2_2414_, v_x_2415_, v_x_2416_);
lean_dec(v_x_2416_);
lean_dec_ref(v_x_2415_);
return v_res_2417_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__5(lean_object* v_00_u03b2_2418_, lean_object* v_x_2419_, lean_object* v_x_2420_){
_start:
{
uint8_t v___x_2421_; 
v___x_2421_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__5___redArg(v_x_2419_, v_x_2420_);
return v___x_2421_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__5___boxed(lean_object* v_00_u03b2_2422_, lean_object* v_x_2423_, lean_object* v_x_2424_){
_start:
{
uint8_t v_res_2425_; lean_object* v_r_2426_; 
v_res_2425_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__5(v_00_u03b2_2422_, v_x_2423_, v_x_2424_);
lean_dec(v_x_2424_);
lean_dec_ref(v_x_2423_);
v_r_2426_ = lean_box(v_res_2425_);
return v_r_2426_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__7(lean_object* v_n_2427_, lean_object* v_as_2428_, lean_object* v_lo_2429_, lean_object* v_hi_2430_, lean_object* v_w_2431_, lean_object* v_hlo_2432_, lean_object* v_hhi_2433_){
_start:
{
lean_object* v___x_2434_; 
v___x_2434_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__7___redArg(v_n_2427_, v_as_2428_, v_lo_2429_, v_hi_2430_);
return v___x_2434_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__7___boxed(lean_object* v_n_2435_, lean_object* v_as_2436_, lean_object* v_lo_2437_, lean_object* v_hi_2438_, lean_object* v_w_2439_, lean_object* v_hlo_2440_, lean_object* v_hhi_2441_){
_start:
{
lean_object* v_res_2442_; 
v_res_2442_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__7(v_n_2435_, v_as_2436_, v_lo_2437_, v_hi_2438_, v_w_2439_, v_hlo_2440_, v_hhi_2441_);
lean_dec(v_hi_2438_);
lean_dec(v_n_2435_);
return v_res_2442_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__2_spec__3(lean_object* v_00_u03b2_2443_, lean_object* v_x_2444_, size_t v_x_2445_, lean_object* v_x_2446_){
_start:
{
lean_object* v___x_2447_; 
v___x_2447_ = lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__2_spec__3___redArg(v_x_2444_, v_x_2445_, v_x_2446_);
return v___x_2447_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__2_spec__3___boxed(lean_object* v_00_u03b2_2448_, lean_object* v_x_2449_, lean_object* v_x_2450_, lean_object* v_x_2451_){
_start:
{
size_t v_x_18016__boxed_2452_; lean_object* v_res_2453_; 
v_x_18016__boxed_2452_ = lean_unbox_usize(v_x_2450_);
lean_dec(v_x_2450_);
v_res_2453_ = lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__2_spec__3(v_00_u03b2_2448_, v_x_2449_, v_x_18016__boxed_2452_, v_x_2451_);
lean_dec(v_x_2451_);
lean_dec_ref(v_x_2449_);
return v_res_2453_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__5_spec__8(lean_object* v_00_u03b2_2454_, lean_object* v_x_2455_, size_t v_x_2456_, lean_object* v_x_2457_){
_start:
{
uint8_t v___x_2458_; 
v___x_2458_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__5_spec__8___redArg(v_x_2455_, v_x_2456_, v_x_2457_);
return v___x_2458_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__5_spec__8___boxed(lean_object* v_00_u03b2_2459_, lean_object* v_x_2460_, lean_object* v_x_2461_, lean_object* v_x_2462_){
_start:
{
size_t v_x_18027__boxed_2463_; uint8_t v_res_2464_; lean_object* v_r_2465_; 
v_x_18027__boxed_2463_ = lean_unbox_usize(v_x_2461_);
lean_dec(v_x_2461_);
v_res_2464_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__5_spec__8(v_00_u03b2_2459_, v_x_2460_, v_x_18027__boxed_2463_, v_x_2462_);
lean_dec(v_x_2462_);
lean_dec_ref(v_x_2460_);
v_r_2465_ = lean_box(v_res_2464_);
return v_r_2465_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__7_spec__11(lean_object* v_n_2466_, lean_object* v_lo_2467_, lean_object* v_hi_2468_, lean_object* v_hhi_2469_, lean_object* v_pivot_2470_, lean_object* v_as_2471_, lean_object* v_i_2472_, lean_object* v_k_2473_, lean_object* v_ilo_2474_, lean_object* v_ik_2475_, lean_object* v_w_2476_){
_start:
{
lean_object* v___x_2477_; 
v___x_2477_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__7_spec__11___redArg(v_hi_2468_, v_pivot_2470_, v_as_2471_, v_i_2472_, v_k_2473_);
return v___x_2477_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__7_spec__11___boxed(lean_object* v_n_2478_, lean_object* v_lo_2479_, lean_object* v_hi_2480_, lean_object* v_hhi_2481_, lean_object* v_pivot_2482_, lean_object* v_as_2483_, lean_object* v_i_2484_, lean_object* v_k_2485_, lean_object* v_ilo_2486_, lean_object* v_ik_2487_, lean_object* v_w_2488_){
_start:
{
lean_object* v_res_2489_; 
v_res_2489_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__7_spec__11(v_n_2478_, v_lo_2479_, v_hi_2480_, v_hhi_2481_, v_pivot_2482_, v_as_2483_, v_i_2484_, v_k_2485_, v_ilo_2486_, v_ik_2487_, v_w_2488_);
lean_dec(v_hi_2480_);
lean_dec(v_lo_2479_);
lean_dec(v_n_2478_);
return v_res_2489_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__10_spec__15(lean_object* v_00_u03b2_2490_, lean_object* v_m_2491_, lean_object* v_a_2492_, lean_object* v_b_2493_){
_start:
{
lean_object* v___x_2494_; 
v___x_2494_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__10_spec__15___redArg(v_m_2491_, v_a_2492_, v_b_2493_);
return v___x_2494_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__11_spec__18(lean_object* v_00_u03b2_2495_, lean_object* v_m_2496_, lean_object* v_a_2497_, lean_object* v_b_2498_){
_start:
{
lean_object* v___x_2499_; 
v___x_2499_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__11_spec__18___redArg(v_m_2496_, v_a_2497_, v_b_2498_);
return v___x_2499_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__2_spec__3_spec__5(lean_object* v_00_u03b2_2500_, lean_object* v_keys_2501_, lean_object* v_vals_2502_, lean_object* v_heq_2503_, lean_object* v_i_2504_, lean_object* v_k_2505_){
_start:
{
lean_object* v___x_2506_; 
v___x_2506_ = lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__2_spec__3_spec__5___redArg(v_keys_2501_, v_vals_2502_, v_i_2504_, v_k_2505_);
return v___x_2506_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__2_spec__3_spec__5___boxed(lean_object* v_00_u03b2_2507_, lean_object* v_keys_2508_, lean_object* v_vals_2509_, lean_object* v_heq_2510_, lean_object* v_i_2511_, lean_object* v_k_2512_){
_start:
{
lean_object* v_res_2513_; 
v_res_2513_ = lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__2_spec__3_spec__5(v_00_u03b2_2507_, v_keys_2508_, v_vals_2509_, v_heq_2510_, v_i_2511_, v_k_2512_);
lean_dec(v_k_2512_);
lean_dec_ref(v_vals_2509_);
lean_dec_ref(v_keys_2508_);
return v_res_2513_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8_spec__18(lean_object* v_msgData_2514_, lean_object* v___y_2515_, lean_object* v___y_2516_){
_start:
{
lean_object* v___x_2518_; 
v___x_2518_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8_spec__18___redArg(v_msgData_2514_, v___y_2516_);
return v___x_2518_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8_spec__18___boxed(lean_object* v_msgData_2519_, lean_object* v___y_2520_, lean_object* v___y_2521_, lean_object* v___y_2522_){
_start:
{
lean_object* v_res_2523_; 
v_res_2523_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__4_spec__6_spec__8_spec__18(v_msgData_2519_, v___y_2520_, v___y_2521_);
lean_dec(v___y_2521_);
lean_dec_ref(v___y_2520_);
return v_res_2523_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__5_spec__8_spec__11(lean_object* v_00_u03b2_2524_, lean_object* v_keys_2525_, lean_object* v_vals_2526_, lean_object* v_heq_2527_, lean_object* v_i_2528_, lean_object* v_k_2529_){
_start:
{
uint8_t v___x_2530_; 
v___x_2530_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__5_spec__8_spec__11___redArg(v_keys_2525_, v_i_2528_, v_k_2529_);
return v___x_2530_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__5_spec__8_spec__11___boxed(lean_object* v_00_u03b2_2531_, lean_object* v_keys_2532_, lean_object* v_vals_2533_, lean_object* v_heq_2534_, lean_object* v_i_2535_, lean_object* v_k_2536_){
_start:
{
uint8_t v_res_2537_; lean_object* v_r_2538_; 
v_res_2537_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__5_spec__8_spec__11(v_00_u03b2_2531_, v_keys_2532_, v_vals_2533_, v_heq_2534_, v_i_2535_, v_k_2536_);
lean_dec(v_k_2536_);
lean_dec_ref(v_vals_2533_);
lean_dec_ref(v_keys_2532_);
v_r_2538_ = lean_box(v_res_2537_);
return v_r_2538_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__10_spec__15_spec__19(lean_object* v_00_u03b2_2539_, lean_object* v_data_2540_){
_start:
{
lean_object* v___x_2541_; 
v___x_2541_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__10_spec__15_spec__19___redArg(v_data_2540_);
return v___x_2541_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__10_spec__15_spec__20(lean_object* v_00_u03b2_2542_, lean_object* v_a_2543_, lean_object* v_b_2544_, lean_object* v_x_2545_){
_start:
{
lean_object* v___x_2546_; 
v___x_2546_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__10_spec__15_spec__20___redArg(v_a_2543_, v_b_2544_, v_x_2545_);
return v___x_2546_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__11_spec__18_spec__24(lean_object* v_00_u03b2_2547_, lean_object* v_a_2548_, lean_object* v_b_2549_, lean_object* v_x_2550_){
_start:
{
lean_object* v___x_2551_; 
v___x_2551_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__11_spec__18_spec__24___redArg(v_a_2548_, v_b_2549_, v_x_2550_);
return v___x_2551_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__10_spec__15_spec__19_spec__25(lean_object* v_00_u03b2_2552_, lean_object* v_i_2553_, lean_object* v_source_2554_, lean_object* v_target_2555_){
_start:
{
lean_object* v___x_2556_; 
v___x_2556_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__10_spec__15_spec__19_spec__25___redArg(v_i_2553_, v_source_2554_, v_target_2555_);
return v___x_2556_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__10_spec__15_spec__19_spec__25_spec__30(lean_object* v_00_u03b2_2557_, lean_object* v_x_2558_, lean_object* v_x_2559_){
_start:
{
lean_object* v___x_2560_; 
v___x_2560_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_getTactics___at___00__private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter_spec__10_spec__15_spec__19_spec__25_spec__30___redArg(v_x_2558_, v_x_2559_);
return v___x_2560_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_300650473____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_2562_; lean_object* v___x_2563_; 
v___x_2562_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_unusedTacticLinter));
v___x_2563_ = l_Lean_Elab_Command_addLinter(v___x_2562_);
return v___x_2563_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_300650473____hygCtx___hyg_2____boxed(lean_object* v_a_2564_){
_start:
{
lean_object* v_res_2565_; 
v_res_2565_ = lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_300650473____hygCtx___hyg_2_();
return v_res_2565_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Tactic_Unreachable(uint8_t builtin);
lean_object* runtime_initialize_Lean_Parser_Syntax(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_UnusedTacticExtension(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_UnusedTactic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Tactic_Unreachable(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Parser_Syntax(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_UnusedTacticExtension(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Server_InfoUtils(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_Header(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Linter_UnusedTactic(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Server_InfoUtils(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_Header(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_4034122115____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Linter_linter_unusedTactic = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Linter_linter_unusedTactic);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_1538057914____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_ignoreTacticKindsRef = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_ignoreTacticKindsRef);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linter_UnusedTactic_0__Mathlib_Linter_UnusedTactic_initFn_00___x40_Mathlib_Tactic_Linter_UnusedTactic_300650473____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Server_InfoUtils(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_Header(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Tactic_Unreachable(uint8_t builtin);
lean_object* initialize_Lean_Parser_Syntax(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_UnusedTacticExtension(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Linter_UnusedTactic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Server_InfoUtils(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linter_Header(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Tactic_Unreachable(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Parser_Syntax(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linter_UnusedTacticExtension(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_UnusedTactic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Linter_UnusedTactic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Linter_UnusedTactic(builtin);
}
#ifdef __cplusplus
}
#endif
