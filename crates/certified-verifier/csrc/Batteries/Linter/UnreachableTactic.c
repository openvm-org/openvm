// Lean compiler output
// Module: Batteries.Linter.UnreachableTactic
// Imports: public import Init public meta import Init public meta import Lean.Elab.Command public meta import Lean.Parser.Syntax public meta import Init.Try public meta import Batteries.Tactic.Unreachable public meta import Lean.Linter.Basic
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
lean_object* lean_array_get_size(lean_object*);
uint64_t l_Lean_Syntax_instHashableRange_hash(lean_object*);
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fswap(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* l_instOrdNat___lam__0___boxed(lean_object*, lean_object*);
lean_object* l_instOrdInt___lam__0___boxed(lean_object*, lean_object*);
lean_object* lean_nat_to_int(lean_object*);
lean_object* lean_int_neg(lean_object*);
lean_object* l_lexOrd___redArg(lean_object*, lean_object*);
lean_object* l_Lean_instInhabitedPersistentArrayNode_default(lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
lean_object* lean_usize_to_nat(size_t);
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
size_t lean_usize_shift_left(size_t, size_t);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* l_Lean_Syntax_getRange_x3f(lean_object*, uint8_t);
lean_object* lean_st_ref_take(lean_object*);
uint8_t l_Lean_Syntax_instBEqRange_beq(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_register_option(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* l_Lean_MessageData_note(lean_object*);
extern lean_object* l_Lean_Linter_linterMessageTag;
lean_object* l_Lean_Elab_Command_getScope___redArg(lean_object*);
lean_object* l_Lean_MessageLog_add(lean_object*, lean_object*);
lean_object* l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
extern lean_object* l_Lean_Elab_Command_instInhabitedScope_default;
lean_object* l_List_head_x21___redArg(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
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
lean_object* l_Lean_Name_beq___boxed(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getKind(lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
uint8_t l_List_elem___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_nat_shiftr(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_io_error_to_string(lean_object*);
extern lean_object* l_Lean_Linter_linterSetsExt;
extern lean_object* l_Lean_Linter_instInhabitedLinterSetsState_default;
lean_object* l_Lean_PersistentEnvExtension_getState___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_MessageLog_hasErrors(lean_object*);
extern lean_object* l_Lean_Parser_parserExtension;
extern lean_object* l_Lean_Parser_ParserExtension_instInhabitedState_default;
lean_object* l_Lean_ScopedEnvExtension_getState___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
lean_object* l_Lean_NameHashSet_insert(lean_object*, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_NameHashSet_contains(lean_object*, lean_object*);
uint8_t l_Lean_Linter_getLinterValue(lean_object*, lean_object*);
lean_object* l_Lean_withSetOptionIn___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadEIO(lean_object*);
lean_object* l_StateRefT_x27_instMonad___redArg(lean_object*);
lean_object* l_Lean_Syntax_instBEqRange_beq___boxed(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_instHashableRange_hash___boxed(lean_object*);
lean_object* l_Std_DHashMap_Internal_Raw_u2080_insert___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
lean_object* l_Lean_Elab_Command_addLinter(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Option_register___at___00__private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4__spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Option_register___at___00__private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn___closed__0_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "linter"};
static const lean_object* lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn___closed__0_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn___closed__0_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4__value;
static const lean_string_object lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn___closed__1_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "unreachableTactic"};
static const lean_object* lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn___closed__1_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn___closed__1_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4__value;
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn___closed__2_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn___closed__0_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(186, 218, 113, 226, 101, 176, 32, 79)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn___closed__2_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn___closed__2_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn___closed__1_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(111, 213, 187, 125, 185, 252, 173, 63)}};
static const lean_object* lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn___closed__2_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn___closed__2_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4__value;
static const lean_string_object lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn___closed__3_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 39, .m_capacity = 39, .m_length = 38, .m_data = "enable the 'unreachable tactic' linter"};
static const lean_object* lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn___closed__3_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn___closed__3_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4__value;
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn___closed__4_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn___closed__3_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn___closed__4_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn___closed__4_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4__value;
static const lean_string_object lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn___closed__5_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Batteries"};
static const lean_object* lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn___closed__5_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn___closed__5_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4__value;
static const lean_string_object lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn___closed__6_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Linter"};
static const lean_object* lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn___closed__6_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn___closed__6_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4__value;
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn___closed__7_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn___closed__5_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn___closed__7_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn___closed__7_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn___closed__6_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(200, 5, 203, 188, 211, 209, 34, 248)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn___closed__7_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn___closed__7_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn___closed__0_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(53, 186, 4, 149, 36, 28, 231, 98)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn___closed__7_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn___closed__7_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn___closed__1_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(44, 97, 159, 208, 176, 57, 61, 205)}};
static const lean_object* lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn___closed__7_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn___closed__7_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_linter_unreachableTactic;
LEAN_EXPORT uint8_t lp_batteries_Batteries_Linter_UnreachableTactic_getLinterUnreachableTactic(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnreachableTactic_getLinterUnreachableTactic___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__spec__0_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__spec__0_spec__1_spec__2_spec__3___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__spec__0_spec__1_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__spec__0_spec__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__spec__0___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__0_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__0_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_;
static lean_once_cell_t lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__1_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__1_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_;
static const lean_string_object lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__2_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__2_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__2_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value;
static const lean_string_object lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__3_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__3_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__3_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value;
static const lean_string_object lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__4_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__4_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__4_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value;
static const lean_string_object lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__5_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "binderTactic"};
static const lean_object* lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__5_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__5_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__6_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__2_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__6_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__6_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__3_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__6_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__6_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__4_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__6_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__6_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__5_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(17, 181, 78, 34, 190, 12, 180, 92)}};
static const lean_object* lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__6_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__6_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value;
static lean_once_cell_t lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__7_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__7_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_;
static const lean_string_object lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__8_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "dynamicQuot"};
static const lean_object* lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__8_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__8_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__9_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__2_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__9_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__9_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__3_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__9_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__9_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__4_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__9_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__9_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__8_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(116, 123, 139, 164, 173, 191, 116, 242)}};
static const lean_object* lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__9_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__9_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value;
static lean_once_cell_t lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__10_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__10_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_;
static const lean_string_object lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__11_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__11_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__11_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value;
static const lean_string_object lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__12_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "quotSeq"};
static const lean_object* lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__12_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__12_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__13_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__2_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__13_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__13_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__3_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__13_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__13_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__11_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__13_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__13_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__12_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(171, 67, 133, 150, 48, 85, 223, 184)}};
static const lean_object* lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__13_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__13_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value;
static lean_once_cell_t lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__14_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__14_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_;
static const lean_string_object lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__15_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "tacticStop_"};
static const lean_object* lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__15_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__15_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__16_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__2_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__16_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__16_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__3_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__16_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__16_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__11_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__16_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__16_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__15_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(186, 187, 217, 116, 133, 153, 2, 108)}};
static const lean_object* lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__16_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__16_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value;
static lean_once_cell_t lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__17_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__17_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_;
static const lean_string_object lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__18_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Command"};
static const lean_object* lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__18_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__18_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value;
static const lean_string_object lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__19_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "notation"};
static const lean_object* lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__19_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__19_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__20_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__2_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__20_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__20_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__3_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__20_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__20_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__18_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__20_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__20_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__19_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(13, 34, 53, 7, 182, 20, 8, 182)}};
static const lean_object* lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__20_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__20_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value;
static lean_once_cell_t lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__21_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__21_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_;
static const lean_string_object lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__22_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "mixfix"};
static const lean_object* lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__22_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__22_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__23_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__2_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__23_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__23_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__3_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__23_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__23_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__18_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__23_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__23_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__22_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(1, 31, 80, 86, 44, 46, 155, 0)}};
static const lean_object* lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__23_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__23_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value;
static lean_once_cell_t lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__24_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__24_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_;
static const lean_string_object lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__25_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "registerTryTactic"};
static const lean_object* lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__25_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__25_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__26_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__2_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__26_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__26_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__3_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__26_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__26_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__18_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__26_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__26_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__25_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(64, 133, 180, 171, 152, 84, 222, 30)}};
static const lean_object* lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__26_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__26_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value;
static lean_once_cell_t lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__27_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__27_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_;
static const lean_string_object lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__28_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "discharger"};
static const lean_object* lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__28_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__28_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__29_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__2_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__29_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__29_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__3_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__29_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__29_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__11_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__29_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__29_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__28_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(233, 186, 255, 143, 150, 72, 152, 71)}};
static const lean_object* lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__29_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__29_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value;
static lean_once_cell_t lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__30_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__30_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__spec__0_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__spec__0_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__spec__0_spec__1_spec__2_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnreachableTactic_ignoreTacticKindsRef;
static const lean_string_object lp_batteries_Batteries_Linter_UnreachableTactic_isIgnoreTacticKind___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "quot"};
static const lean_object* lp_batteries_Batteries_Linter_UnreachableTactic_isIgnoreTacticKind___closed__0 = (const lean_object*)&lp_batteries_Batteries_Linter_UnreachableTactic_isIgnoreTacticKind___closed__0_value;
LEAN_EXPORT uint8_t lp_batteries_Batteries_Linter_UnreachableTactic_isIgnoreTacticKind(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnreachableTactic_isIgnoreTacticKind___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnreachableTactic_addIgnoreTacticKind(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnreachableTactic_addIgnoreTacticKind___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_batteries_Batteries_Linter_UnreachableTactic_getTactics___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Linter_UnreachableTactic_getTactics___closed__0;
static lean_once_cell_t lp_batteries_Batteries_Linter_UnreachableTactic_getTactics___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Linter_UnreachableTactic_getTactics___closed__1;
static const lean_closure_object lp_batteries_Batteries_Linter_UnreachableTactic_getTactics___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Syntax_instBEqRange_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Batteries_Linter_UnreachableTactic_getTactics___closed__2 = (const lean_object*)&lp_batteries_Batteries_Linter_UnreachableTactic_getTactics___closed__2_value;
static const lean_closure_object lp_batteries_Batteries_Linter_UnreachableTactic_getTactics___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Syntax_instHashableRange_hash___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Batteries_Linter_UnreachableTactic_getTactics___closed__3 = (const lean_object*)&lp_batteries_Batteries_Linter_UnreachableTactic_getTactics___closed__3_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnreachableTactic_getTactics___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnreachableTactic_getTactics(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnreachableTactic_getTactics___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnreachableTactic_getTactics___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnreachableTactic_eraseUsedTactics_spec__2_spec__5___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnreachableTactic_eraseUsedTactics_spec__2_spec__5___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnreachableTactic_eraseUsedTactics_spec__2_spec__6___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnreachableTactic_eraseUsedTactics_spec__2_spec__6___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnreachableTactic_eraseUsedTactics_spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnreachableTactic_eraseUsedTactics_spec__2___redArg___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnreachableTactic_eraseUsedTacticsList_spec__0_spec__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnreachableTactic_eraseUsedTacticsList_spec__0_spec__0___closed__0;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnreachableTactic_eraseUsedTactics(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnreachableTactic_eraseUsedTacticsList_spec__0_spec__1(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_forMAux___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnreachableTactic_eraseUsedTacticsList_spec__0_spec__0_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnreachableTactic_eraseUsedTacticsList_spec__0_spec__0_spec__3(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnreachableTactic_eraseUsedTacticsList_spec__0_spec__0(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_forMFrom0___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnreachableTactic_eraseUsedTacticsList_spec__0_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_forM___at___00Batteries_Linter_UnreachableTactic_eraseUsedTacticsList_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnreachableTactic_eraseUsedTacticsList(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnreachableTactic_eraseUsedTacticsList___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnreachableTactic_eraseUsedTacticsList_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnreachableTactic_eraseUsedTacticsList_spec__0_spec__0_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_forMFrom0___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnreachableTactic_eraseUsedTacticsList_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnreachableTactic_eraseUsedTactics___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_forMAux___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnreachableTactic_eraseUsedTacticsList_spec__0_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_forM___at___00Batteries_Linter_UnreachableTactic_eraseUsedTacticsList_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnreachableTactic_eraseUsedTacticsList_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnreachableTactic_eraseUsedTactics_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnreachableTactic_eraseUsedTactics_spec__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnreachableTactic_eraseUsedTactics_spec__2_spec__5(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnreachableTactic_eraseUsedTactics_spec__2_spec__5___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnreachableTactic_eraseUsedTactics_spec__2_spec__6(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnreachableTactic_eraseUsedTactics_spec__2_spec__6___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nat_cast___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_getInfoTrees___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__4___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_getInfoTrees___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__4___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_getInfoTrees___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_getInfoTrees___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__4___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnreachableTactic_getTactics___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__10_spec__15_spec__20___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnreachableTactic_getTactics___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__10_spec__15_spec__19_spec__22_spec__24___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnreachableTactic_getTactics___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__10_spec__15_spec__19_spec__22___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnreachableTactic_getTactics___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__10_spec__15_spec__19___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnreachableTactic_getTactics___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__10_spec__15___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__9_spec__13_spec__16___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__9_spec__13_spec__16___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__9_spec__13___redArg(lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__9_spec__13___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_contains___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__9___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_contains___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__9___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnreachableTactic_getTactics___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__10(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Linter_UnreachableTactic_getTactics___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__10_spec__16(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Linter_UnreachableTactic_getTactics___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__10_spec__16___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnreachableTactic_getTactics___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3_spec__14(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3_spec__14___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3_spec__13___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3_spec__13___redArg___closed__0;
static lean_once_cell_t lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3_spec__13___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3_spec__13___redArg___closed__1;
static lean_once_cell_t lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3_spec__13___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3_spec__13___redArg___closed__2;
static lean_once_cell_t lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3_spec__13___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3_spec__13___redArg___closed__3;
static lean_once_cell_t lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3_spec__13___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3_spec__13___redArg___closed__4;
static lean_once_cell_t lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3_spec__13___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3_spec__13___redArg___closed__5;
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3_spec__13___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3_spec__13___redArg___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_batteries_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3___lam__0___closed__0 = (const lean_object*)&lp_batteries_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3___lam__0___closed__0_value;
LEAN_EXPORT uint8_t lp_batteries_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3___lam__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_batteries_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3___closed__0 = (const lean_object*)&lp_batteries_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 46, .m_capacity = 46, .m_length = 45, .m_data = "This linter can be disabled with `set_option "};
static const lean_object* lp_batteries_Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1___closed__0 = (const lean_object*)&lp_batteries_Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1___closed__0_value;
static lean_once_cell_t lp_batteries_Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1___closed__1;
static const lean_string_object lp_batteries_Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = " false`"};
static const lean_object* lp_batteries_Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1___closed__2 = (const lean_object*)&lp_batteries_Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1___closed__2_value;
static lean_once_cell_t lp_batteries_Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1___closed__3;
LEAN_EXPORT lean_object* lp_batteries_Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__5___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 30, .m_capacity = 30, .m_length = 29, .m_data = "this tactic is never executed"};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__5___closed__0 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__5___closed__0_value;
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__5___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__5___closed__0_value)}};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__5___closed__1 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__5___closed__1_value;
static lean_once_cell_t lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__5___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__5___closed__2;
static const lean_closure_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__5___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Name_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__5___closed__3 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__5___closed__3_value;
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__5___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "unreachable"};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__5___closed__4 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__5___closed__4_value;
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__5___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn___closed__5_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__5___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__5___closed__5_value_aux_0),((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__11_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__5___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__5___closed__5_value_aux_1),((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__5___closed__4_value),LEAN_SCALAR_PTR_LITERAL(138, 69, 16, 178, 93, 143, 143, 50)}};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__5___closed__5 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__5___closed__5_value;
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__5___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "unreachableConv"};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__5___closed__6 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__5___closed__6_value;
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__5___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn___closed__5_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__5___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__5___closed__7_value_aux_0),((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__11_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__5___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__5___closed__7_value_aux_1),((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__5___closed__6_value),LEAN_SCALAR_PTR_LITERAL(180, 51, 125, 100, 108, 230, 32, 33)}};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__5___closed__7 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__5___closed__7_value;
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__5___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__5___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__5___closed__8 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__5___closed__8_value;
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__5___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__5___closed__5_value),((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__5___closed__8_value)}};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__5___closed__9 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__5___closed__9_value;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__5(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__6___redArg___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instOrdNat___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__6___redArg___lam__1___closed__0 = (const lean_object*)&lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__6___redArg___lam__1___closed__0_value;
static const lean_closure_object lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__6___redArg___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instOrdInt___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__6___redArg___lam__1___closed__1 = (const lean_object*)&lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__6___redArg___lam__1___closed__1_value;
LEAN_EXPORT uint8_t lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__6___redArg___lam__1(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__6___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__6___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__6_spec__9___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__6_spec__9___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__6___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__6___redArg___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__6___redArg___closed__0 = (const lean_object*)&lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__6___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__6___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__3_spec__5_spec__8___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__3_spec__5_spec__8___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__3_spec__5___redArg(lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__3_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_find_x3f___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__3___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_find_x3f___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__3___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__7(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__7___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__8(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__2_spec__3___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__2_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Linter_getLinterOptions___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Linter_getLinterOptions___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__2___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "tactic"};
static const lean_object* lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___lam__0___closed__0 = (const lean_object*)&lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___lam__0___closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(99, 76, 33, 121, 85, 143, 17, 224)}};
static const lean_object* lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___lam__0___closed__1 = (const lean_object*)&lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___lam__0___closed__1_value;
static const lean_string_object lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "conv"};
static const lean_object* lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___lam__0___closed__2 = (const lean_object*)&lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___lam__0___closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(232, 67, 39, 189, 45, 247, 54, 81)}};
static const lean_object* lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___lam__0___closed__3 = (const lean_object*)&lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___lam__0___closed__3_value;
static lean_once_cell_t lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___lam__0___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___lam__0___closed__4;
static lean_once_cell_t lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___lam__0___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___lam__0___closed__5;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___lam__0___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___closed__0 = (const lean_object*)&lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___closed__0_value;
static const lean_closure_object lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_withSetOptionIn___boxed, .m_arity = 6, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___closed__0_value)} };
static const lean_object* lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___closed__1 = (const lean_object*)&lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___closed__1_value;
static const lean_string_object lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "UnreachableTactic"};
static const lean_object* lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___closed__2 = (const lean_object*)&lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___closed__2_value;
static const lean_string_object lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "unreachableTacticLinter"};
static const lean_object* lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___closed__3 = (const lean_object*)&lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___closed__3_value;
static const lean_ctor_object lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn___closed__5_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___closed__4_value_aux_0),((lean_object*)&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn___closed__6_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(200, 5, 203, 188, 211, 209, 34, 248)}};
static const lean_ctor_object lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___closed__4_value_aux_1),((lean_object*)&lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___closed__2_value),LEAN_SCALAR_PTR_LITERAL(179, 42, 10, 234, 98, 130, 52, 131)}};
static const lean_ctor_object lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___closed__4_value_aux_2),((lean_object*)&lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___closed__3_value),LEAN_SCALAR_PTR_LITERAL(185, 1, 169, 232, 96, 7, 66, 186)}};
static const lean_object* lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___closed__4 = (const lean_object*)&lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___closed__4_value;
static const lean_ctor_object lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___closed__1_value),((lean_object*)&lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___closed__4_value)}};
static const lean_object* lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___closed__5 = (const lean_object*)&lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___closed__5_value;
LEAN_EXPORT const lean_object* lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter = (const lean_object*)&lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___closed__5_value;
LEAN_EXPORT lean_object* lp_batteries_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__2_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_find_x3f___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_find_x3f___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_contains___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__9(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_contains___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__9___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__3_spec__5(lean_object*, lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__3_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__6_spec__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__6_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__9_spec__13(lean_object*, lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__9_spec__13___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnreachableTactic_getTactics___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__10_spec__15(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3_spec__13(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3_spec__13___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__3_spec__5_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__3_spec__5_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__9_spec__13_spec__16(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__9_spec__13_spec__16___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnreachableTactic_getTactics___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__10_spec__15_spec__19(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnreachableTactic_getTactics___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__10_spec__15_spec__20(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnreachableTactic_getTactics___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__10_spec__15_spec__19_spec__22(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnreachableTactic_getTactics___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__10_spec__15_spec__19_spec__22_spec__24(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn_00___x40_Batteries_Linter_UnreachableTactic_179727238____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn_00___x40_Batteries_Linter_UnreachableTactic_179727238____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Option_register___at___00__private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4__spec__0(lean_object* v_name_1_, lean_object* v_decl_2_, lean_object* v_ref_3_){
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
LEAN_EXPORT lean_object* lp_batteries_Lean_Option_register___at___00__private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4__spec__0___boxed(lean_object* v_name_29_, lean_object* v_decl_30_, lean_object* v_ref_31_, lean_object* v_a_32_){
_start:
{
lean_object* v_res_33_; 
v_res_33_ = lp_batteries_Lean_Option_register___at___00__private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4__spec__0(v_name_29_, v_decl_30_, v_ref_31_);
lean_dec_ref(v_decl_30_);
return v_res_33_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_53_; lean_object* v___x_54_; lean_object* v___x_55_; lean_object* v___x_56_; 
v___x_53_ = ((lean_object*)(lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn___closed__2_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4_));
v___x_54_ = ((lean_object*)(lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn___closed__4_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4_));
v___x_55_ = ((lean_object*)(lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn___closed__7_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4_));
v___x_56_ = lp_batteries_Lean_Option_register___at___00__private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4__spec__0(v___x_53_, v___x_54_, v___x_55_);
return v___x_56_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4____boxed(lean_object* v_a_57_){
_start:
{
lean_object* v_res_58_; 
v_res_58_ = lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4_();
return v_res_58_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Batteries_Linter_UnreachableTactic_getLinterUnreachableTactic(lean_object* v_o_59_){
_start:
{
lean_object* v___x_60_; uint8_t v___x_61_; 
v___x_60_ = lp_batteries_Batteries_Linter_linter_unreachableTactic;
v___x_61_ = l_Lean_Linter_getLinterValue(v___x_60_, v_o_59_);
return v___x_61_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnreachableTactic_getLinterUnreachableTactic___boxed(lean_object* v_o_62_){
_start:
{
uint8_t v_res_63_; lean_object* v_r_64_; 
v_res_63_ = lp_batteries_Batteries_Linter_UnreachableTactic_getLinterUnreachableTactic(v_o_62_);
lean_dec_ref(v_o_62_);
v_r_64_ = lean_box(v_res_63_);
return v_r_64_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__spec__0_spec__0___redArg(lean_object* v_a_65_, lean_object* v_x_66_){
_start:
{
if (lean_obj_tag(v_x_66_) == 0)
{
uint8_t v___x_67_; 
v___x_67_ = 0;
return v___x_67_;
}
else
{
lean_object* v_key_68_; lean_object* v_tail_69_; uint8_t v___x_70_; 
v_key_68_ = lean_ctor_get(v_x_66_, 0);
v_tail_69_ = lean_ctor_get(v_x_66_, 2);
v___x_70_ = lean_name_eq(v_key_68_, v_a_65_);
if (v___x_70_ == 0)
{
v_x_66_ = v_tail_69_;
goto _start;
}
else
{
return v___x_70_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__spec__0_spec__0___redArg___boxed(lean_object* v_a_72_, lean_object* v_x_73_){
_start:
{
uint8_t v_res_74_; lean_object* v_r_75_; 
v_res_74_ = lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__spec__0_spec__0___redArg(v_a_72_, v_x_73_);
lean_dec(v_x_73_);
lean_dec(v_a_72_);
v_r_75_ = lean_box(v_res_74_);
return v_r_75_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__spec__0_spec__1_spec__2_spec__3___redArg(lean_object* v_x_76_, lean_object* v_x_77_){
_start:
{
if (lean_obj_tag(v_x_77_) == 0)
{
return v_x_76_;
}
else
{
lean_object* v_key_78_; lean_object* v_value_79_; lean_object* v_tail_80_; lean_object* v___x_82_; uint8_t v_isShared_83_; uint8_t v_isSharedCheck_106_; 
v_key_78_ = lean_ctor_get(v_x_77_, 0);
v_value_79_ = lean_ctor_get(v_x_77_, 1);
v_tail_80_ = lean_ctor_get(v_x_77_, 2);
v_isSharedCheck_106_ = !lean_is_exclusive(v_x_77_);
if (v_isSharedCheck_106_ == 0)
{
v___x_82_ = v_x_77_;
v_isShared_83_ = v_isSharedCheck_106_;
goto v_resetjp_81_;
}
else
{
lean_inc(v_tail_80_);
lean_inc(v_value_79_);
lean_inc(v_key_78_);
lean_dec(v_x_77_);
v___x_82_ = lean_box(0);
v_isShared_83_ = v_isSharedCheck_106_;
goto v_resetjp_81_;
}
v_resetjp_81_:
{
lean_object* v___x_84_; uint64_t v___y_86_; 
v___x_84_ = lean_array_get_size(v_x_76_);
if (lean_obj_tag(v_key_78_) == 0)
{
uint64_t v___x_104_; 
v___x_104_ = 1723ULL;
v___y_86_ = v___x_104_;
goto v___jp_85_;
}
else
{
uint64_t v_hash_105_; 
v_hash_105_ = lean_ctor_get_uint64(v_key_78_, sizeof(void*)*2);
v___y_86_ = v_hash_105_;
goto v___jp_85_;
}
v___jp_85_:
{
uint64_t v___x_87_; uint64_t v___x_88_; uint64_t v_fold_89_; uint64_t v___x_90_; uint64_t v___x_91_; uint64_t v___x_92_; size_t v___x_93_; size_t v___x_94_; size_t v___x_95_; size_t v___x_96_; size_t v___x_97_; lean_object* v___x_98_; lean_object* v___x_100_; 
v___x_87_ = 32ULL;
v___x_88_ = lean_uint64_shift_right(v___y_86_, v___x_87_);
v_fold_89_ = lean_uint64_xor(v___y_86_, v___x_88_);
v___x_90_ = 16ULL;
v___x_91_ = lean_uint64_shift_right(v_fold_89_, v___x_90_);
v___x_92_ = lean_uint64_xor(v_fold_89_, v___x_91_);
v___x_93_ = lean_uint64_to_usize(v___x_92_);
v___x_94_ = lean_usize_of_nat(v___x_84_);
v___x_95_ = ((size_t)1ULL);
v___x_96_ = lean_usize_sub(v___x_94_, v___x_95_);
v___x_97_ = lean_usize_land(v___x_93_, v___x_96_);
v___x_98_ = lean_array_uget_borrowed(v_x_76_, v___x_97_);
lean_inc(v___x_98_);
if (v_isShared_83_ == 0)
{
lean_ctor_set(v___x_82_, 2, v___x_98_);
v___x_100_ = v___x_82_;
goto v_reusejp_99_;
}
else
{
lean_object* v_reuseFailAlloc_103_; 
v_reuseFailAlloc_103_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_103_, 0, v_key_78_);
lean_ctor_set(v_reuseFailAlloc_103_, 1, v_value_79_);
lean_ctor_set(v_reuseFailAlloc_103_, 2, v___x_98_);
v___x_100_ = v_reuseFailAlloc_103_;
goto v_reusejp_99_;
}
v_reusejp_99_:
{
lean_object* v___x_101_; 
v___x_101_ = lean_array_uset(v_x_76_, v___x_97_, v___x_100_);
v_x_76_ = v___x_101_;
v_x_77_ = v_tail_80_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__spec__0_spec__1_spec__2___redArg(lean_object* v_i_107_, lean_object* v_source_108_, lean_object* v_target_109_){
_start:
{
lean_object* v___x_110_; uint8_t v___x_111_; 
v___x_110_ = lean_array_get_size(v_source_108_);
v___x_111_ = lean_nat_dec_lt(v_i_107_, v___x_110_);
if (v___x_111_ == 0)
{
lean_dec_ref(v_source_108_);
lean_dec(v_i_107_);
return v_target_109_;
}
else
{
lean_object* v_es_112_; lean_object* v___x_113_; lean_object* v_source_114_; lean_object* v_target_115_; lean_object* v___x_116_; lean_object* v___x_117_; 
v_es_112_ = lean_array_fget(v_source_108_, v_i_107_);
v___x_113_ = lean_box(0);
v_source_114_ = lean_array_fset(v_source_108_, v_i_107_, v___x_113_);
v_target_115_ = lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__spec__0_spec__1_spec__2_spec__3___redArg(v_target_109_, v_es_112_);
v___x_116_ = lean_unsigned_to_nat(1u);
v___x_117_ = lean_nat_add(v_i_107_, v___x_116_);
lean_dec(v_i_107_);
v_i_107_ = v___x_117_;
v_source_108_ = v_source_114_;
v_target_109_ = v_target_115_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__spec__0_spec__1___redArg(lean_object* v_data_119_){
_start:
{
lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v_nbuckets_122_; lean_object* v___x_123_; lean_object* v___x_124_; lean_object* v___x_125_; lean_object* v___x_126_; 
v___x_120_ = lean_array_get_size(v_data_119_);
v___x_121_ = lean_unsigned_to_nat(2u);
v_nbuckets_122_ = lean_nat_mul(v___x_120_, v___x_121_);
v___x_123_ = lean_unsigned_to_nat(0u);
v___x_124_ = lean_box(0);
v___x_125_ = lean_mk_array(v_nbuckets_122_, v___x_124_);
v___x_126_ = lp_batteries___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__spec__0_spec__1_spec__2___redArg(v___x_123_, v_data_119_, v___x_125_);
return v___x_126_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__spec__0___redArg(lean_object* v_m_127_, lean_object* v_a_128_, lean_object* v_b_129_){
_start:
{
lean_object* v_size_130_; lean_object* v_buckets_131_; lean_object* v___x_132_; uint64_t v___y_134_; 
v_size_130_ = lean_ctor_get(v_m_127_, 0);
v_buckets_131_ = lean_ctor_get(v_m_127_, 1);
v___x_132_ = lean_array_get_size(v_buckets_131_);
if (lean_obj_tag(v_a_128_) == 0)
{
uint64_t v___x_171_; 
v___x_171_ = 1723ULL;
v___y_134_ = v___x_171_;
goto v___jp_133_;
}
else
{
uint64_t v_hash_172_; 
v_hash_172_ = lean_ctor_get_uint64(v_a_128_, sizeof(void*)*2);
v___y_134_ = v_hash_172_;
goto v___jp_133_;
}
v___jp_133_:
{
uint64_t v___x_135_; uint64_t v___x_136_; uint64_t v_fold_137_; uint64_t v___x_138_; uint64_t v___x_139_; uint64_t v___x_140_; size_t v___x_141_; size_t v___x_142_; size_t v___x_143_; size_t v___x_144_; size_t v___x_145_; lean_object* v_bkt_146_; uint8_t v___x_147_; 
v___x_135_ = 32ULL;
v___x_136_ = lean_uint64_shift_right(v___y_134_, v___x_135_);
v_fold_137_ = lean_uint64_xor(v___y_134_, v___x_136_);
v___x_138_ = 16ULL;
v___x_139_ = lean_uint64_shift_right(v_fold_137_, v___x_138_);
v___x_140_ = lean_uint64_xor(v_fold_137_, v___x_139_);
v___x_141_ = lean_uint64_to_usize(v___x_140_);
v___x_142_ = lean_usize_of_nat(v___x_132_);
v___x_143_ = ((size_t)1ULL);
v___x_144_ = lean_usize_sub(v___x_142_, v___x_143_);
v___x_145_ = lean_usize_land(v___x_141_, v___x_144_);
v_bkt_146_ = lean_array_uget_borrowed(v_buckets_131_, v___x_145_);
v___x_147_ = lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__spec__0_spec__0___redArg(v_a_128_, v_bkt_146_);
if (v___x_147_ == 0)
{
lean_object* v___x_149_; uint8_t v_isShared_150_; uint8_t v_isSharedCheck_168_; 
lean_inc_ref(v_buckets_131_);
lean_inc(v_size_130_);
v_isSharedCheck_168_ = !lean_is_exclusive(v_m_127_);
if (v_isSharedCheck_168_ == 0)
{
lean_object* v_unused_169_; lean_object* v_unused_170_; 
v_unused_169_ = lean_ctor_get(v_m_127_, 1);
lean_dec(v_unused_169_);
v_unused_170_ = lean_ctor_get(v_m_127_, 0);
lean_dec(v_unused_170_);
v___x_149_ = v_m_127_;
v_isShared_150_ = v_isSharedCheck_168_;
goto v_resetjp_148_;
}
else
{
lean_dec(v_m_127_);
v___x_149_ = lean_box(0);
v_isShared_150_ = v_isSharedCheck_168_;
goto v_resetjp_148_;
}
v_resetjp_148_:
{
lean_object* v___x_151_; lean_object* v_size_x27_152_; lean_object* v___x_153_; lean_object* v_buckets_x27_154_; lean_object* v___x_155_; lean_object* v___x_156_; lean_object* v___x_157_; lean_object* v___x_158_; lean_object* v___x_159_; uint8_t v___x_160_; 
v___x_151_ = lean_unsigned_to_nat(1u);
v_size_x27_152_ = lean_nat_add(v_size_130_, v___x_151_);
lean_dec(v_size_130_);
lean_inc(v_bkt_146_);
v___x_153_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_153_, 0, v_a_128_);
lean_ctor_set(v___x_153_, 1, v_b_129_);
lean_ctor_set(v___x_153_, 2, v_bkt_146_);
v_buckets_x27_154_ = lean_array_uset(v_buckets_131_, v___x_145_, v___x_153_);
v___x_155_ = lean_unsigned_to_nat(4u);
v___x_156_ = lean_nat_mul(v_size_x27_152_, v___x_155_);
v___x_157_ = lean_unsigned_to_nat(3u);
v___x_158_ = lean_nat_div(v___x_156_, v___x_157_);
lean_dec(v___x_156_);
v___x_159_ = lean_array_get_size(v_buckets_x27_154_);
v___x_160_ = lean_nat_dec_le(v___x_158_, v___x_159_);
lean_dec(v___x_158_);
if (v___x_160_ == 0)
{
lean_object* v_val_161_; lean_object* v___x_163_; 
v_val_161_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__spec__0_spec__1___redArg(v_buckets_x27_154_);
if (v_isShared_150_ == 0)
{
lean_ctor_set(v___x_149_, 1, v_val_161_);
lean_ctor_set(v___x_149_, 0, v_size_x27_152_);
v___x_163_ = v___x_149_;
goto v_reusejp_162_;
}
else
{
lean_object* v_reuseFailAlloc_164_; 
v_reuseFailAlloc_164_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_164_, 0, v_size_x27_152_);
lean_ctor_set(v_reuseFailAlloc_164_, 1, v_val_161_);
v___x_163_ = v_reuseFailAlloc_164_;
goto v_reusejp_162_;
}
v_reusejp_162_:
{
return v___x_163_;
}
}
else
{
lean_object* v___x_166_; 
if (v_isShared_150_ == 0)
{
lean_ctor_set(v___x_149_, 1, v_buckets_x27_154_);
lean_ctor_set(v___x_149_, 0, v_size_x27_152_);
v___x_166_ = v___x_149_;
goto v_reusejp_165_;
}
else
{
lean_object* v_reuseFailAlloc_167_; 
v_reuseFailAlloc_167_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_167_, 0, v_size_x27_152_);
lean_ctor_set(v_reuseFailAlloc_167_, 1, v_buckets_x27_154_);
v___x_166_ = v_reuseFailAlloc_167_;
goto v_reusejp_165_;
}
v_reusejp_165_:
{
return v___x_166_;
}
}
}
}
else
{
lean_dec(v_b_129_);
lean_dec(v_a_128_);
return v_m_127_;
}
}
}
}
static lean_object* _init_lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__0_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_173_; lean_object* v___x_174_; lean_object* v___x_175_; 
v___x_173_ = lean_box(0);
v___x_174_ = lean_unsigned_to_nat(16u);
v___x_175_ = lean_mk_array(v___x_174_, v___x_173_);
return v___x_175_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__1_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_176_; lean_object* v___x_177_; lean_object* v___x_178_; 
v___x_176_ = lean_obj_once(&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__0_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__0_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__0_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_);
v___x_177_ = lean_unsigned_to_nat(0u);
v___x_178_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_178_, 0, v___x_177_);
lean_ctor_set(v___x_178_, 1, v___x_176_);
return v___x_178_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__7_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_188_; lean_object* v___x_189_; lean_object* v___x_190_; 
v___x_188_ = ((lean_object*)(lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__6_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_));
v___x_189_ = lean_obj_once(&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__1_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__1_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__1_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_);
v___x_190_ = l_Lean_NameHashSet_insert(v___x_189_, v___x_188_);
return v___x_190_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__10_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_197_; lean_object* v___x_198_; lean_object* v___x_199_; lean_object* v___x_200_; 
v___x_197_ = lean_box(0);
v___x_198_ = ((lean_object*)(lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__9_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_));
v___x_199_ = lean_obj_once(&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__7_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__7_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__7_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_);
v___x_200_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__spec__0___redArg(v___x_199_, v___x_198_, v___x_197_);
return v___x_200_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__14_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_208_; lean_object* v___x_209_; lean_object* v___x_210_; lean_object* v___x_211_; 
v___x_208_ = lean_box(0);
v___x_209_ = ((lean_object*)(lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__13_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_));
v___x_210_ = lean_obj_once(&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__10_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__10_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__10_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_);
v___x_211_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__spec__0___redArg(v___x_210_, v___x_209_, v___x_208_);
return v___x_211_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__17_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_218_; lean_object* v___x_219_; lean_object* v___x_220_; lean_object* v___x_221_; 
v___x_218_ = lean_box(0);
v___x_219_ = ((lean_object*)(lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__16_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_));
v___x_220_ = lean_obj_once(&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__14_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__14_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__14_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_);
v___x_221_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__spec__0___redArg(v___x_220_, v___x_219_, v___x_218_);
return v___x_221_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__21_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v___x_231_; lean_object* v___x_232_; 
v___x_229_ = lean_box(0);
v___x_230_ = ((lean_object*)(lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__20_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_));
v___x_231_ = lean_obj_once(&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__17_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__17_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__17_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_);
v___x_232_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__spec__0___redArg(v___x_231_, v___x_230_, v___x_229_);
return v___x_232_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__24_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_239_; lean_object* v___x_240_; lean_object* v___x_241_; lean_object* v___x_242_; 
v___x_239_ = lean_box(0);
v___x_240_ = ((lean_object*)(lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__23_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_));
v___x_241_ = lean_obj_once(&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__21_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__21_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__21_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_);
v___x_242_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__spec__0___redArg(v___x_241_, v___x_240_, v___x_239_);
return v___x_242_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__27_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_249_; lean_object* v___x_250_; lean_object* v___x_251_; lean_object* v___x_252_; 
v___x_249_ = lean_box(0);
v___x_250_ = ((lean_object*)(lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__26_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_));
v___x_251_ = lean_obj_once(&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__24_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__24_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__24_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_);
v___x_252_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__spec__0___redArg(v___x_251_, v___x_250_, v___x_249_);
return v___x_252_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__30_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_259_; lean_object* v___x_260_; lean_object* v___x_261_; lean_object* v___x_262_; 
v___x_259_ = lean_box(0);
v___x_260_ = ((lean_object*)(lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__29_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_));
v___x_261_ = lean_obj_once(&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__27_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__27_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__27_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_);
v___x_262_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__spec__0___redArg(v___x_261_, v___x_260_, v___x_259_);
return v___x_262_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_264_; lean_object* v___x_265_; lean_object* v___x_266_; 
v___x_264_ = lean_obj_once(&lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__30_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__30_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn___closed__30_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_);
v___x_265_ = lean_st_mk_ref(v___x_264_);
v___x_266_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_266_, 0, v___x_265_);
return v___x_266_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2____boxed(lean_object* v_a_267_){
_start:
{
lean_object* v_res_268_; 
v_res_268_ = lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_();
return v_res_268_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__spec__0(lean_object* v_00_u03b2_269_, lean_object* v_m_270_, lean_object* v_a_271_, lean_object* v_b_272_){
_start:
{
lean_object* v___x_273_; 
v___x_273_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__spec__0___redArg(v_m_270_, v_a_271_, v_b_272_);
return v___x_273_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__spec__0_spec__0(lean_object* v_00_u03b2_274_, lean_object* v_a_275_, lean_object* v_x_276_){
_start:
{
uint8_t v___x_277_; 
v___x_277_ = lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__spec__0_spec__0___redArg(v_a_275_, v_x_276_);
return v___x_277_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__spec__0_spec__0___boxed(lean_object* v_00_u03b2_278_, lean_object* v_a_279_, lean_object* v_x_280_){
_start:
{
uint8_t v_res_281_; lean_object* v_r_282_; 
v_res_281_ = lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__spec__0_spec__0(v_00_u03b2_278_, v_a_279_, v_x_280_);
lean_dec(v_x_280_);
lean_dec(v_a_279_);
v_r_282_ = lean_box(v_res_281_);
return v_r_282_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__spec__0_spec__1(lean_object* v_00_u03b2_283_, lean_object* v_data_284_){
_start:
{
lean_object* v___x_285_; 
v___x_285_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__spec__0_spec__1___redArg(v_data_284_);
return v___x_285_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__spec__0_spec__1_spec__2(lean_object* v_00_u03b2_286_, lean_object* v_i_287_, lean_object* v_source_288_, lean_object* v_target_289_){
_start:
{
lean_object* v___x_290_; 
v___x_290_ = lp_batteries___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__spec__0_spec__1_spec__2___redArg(v_i_287_, v_source_288_, v_target_289_);
return v___x_290_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__spec__0_spec__1_spec__2_spec__3(lean_object* v_00_u03b2_291_, lean_object* v_x_292_, lean_object* v_x_293_){
_start:
{
lean_object* v___x_294_; 
v___x_294_ = lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2__spec__0_spec__1_spec__2_spec__3___redArg(v_x_292_, v_x_293_);
return v___x_294_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Batteries_Linter_UnreachableTactic_isIgnoreTacticKind(lean_object* v_ignoreTacticKinds_296_, lean_object* v_k_297_){
_start:
{
if (lean_obj_tag(v_k_297_) == 1)
{
lean_object* v_str_298_; lean_object* v___x_299_; uint8_t v___x_300_; 
v_str_298_ = lean_ctor_get(v_k_297_, 1);
v___x_299_ = ((lean_object*)(lp_batteries_Batteries_Linter_UnreachableTactic_isIgnoreTacticKind___closed__0));
v___x_300_ = lean_string_dec_eq(v_str_298_, v___x_299_);
if (v___x_300_ == 0)
{
uint8_t v___x_301_; 
v___x_301_ = l_Lean_NameHashSet_contains(v_ignoreTacticKinds_296_, v_k_297_);
return v___x_301_;
}
else
{
return v___x_300_;
}
}
else
{
uint8_t v___x_302_; 
v___x_302_ = l_Lean_NameHashSet_contains(v_ignoreTacticKinds_296_, v_k_297_);
return v___x_302_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnreachableTactic_isIgnoreTacticKind___boxed(lean_object* v_ignoreTacticKinds_303_, lean_object* v_k_304_){
_start:
{
uint8_t v_res_305_; lean_object* v_r_306_; 
v_res_305_ = lp_batteries_Batteries_Linter_UnreachableTactic_isIgnoreTacticKind(v_ignoreTacticKinds_303_, v_k_304_);
lean_dec(v_k_304_);
lean_dec_ref(v_ignoreTacticKinds_303_);
v_r_306_ = lean_box(v_res_305_);
return v_r_306_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnreachableTactic_addIgnoreTacticKind(lean_object* v_kind_307_){
_start:
{
lean_object* v___x_309_; lean_object* v___x_310_; lean_object* v___x_311_; lean_object* v___x_312_; lean_object* v___x_313_; 
v___x_309_ = lp_batteries_Batteries_Linter_UnreachableTactic_ignoreTacticKindsRef;
v___x_310_ = lean_st_ref_take(v___x_309_);
v___x_311_ = l_Lean_NameHashSet_insert(v___x_310_, v_kind_307_);
v___x_312_ = lean_st_ref_set(v___x_309_, v___x_311_);
v___x_313_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_313_, 0, v___x_312_);
return v___x_313_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnreachableTactic_addIgnoreTacticKind___boxed(lean_object* v_kind_314_, lean_object* v_a_315_){
_start:
{
lean_object* v_res_316_; 
v_res_316_ = lp_batteries_Batteries_Linter_UnreachableTactic_addIgnoreTacticKind(v_kind_314_);
return v_res_316_;
}
}
static lean_object* _init_lp_batteries_Batteries_Linter_UnreachableTactic_getTactics___closed__0(void){
_start:
{
lean_object* v___x_317_; 
v___x_317_ = l_instMonadEIO(lean_box(0));
return v___x_317_;
}
}
static lean_object* _init_lp_batteries_Batteries_Linter_UnreachableTactic_getTactics___closed__1(void){
_start:
{
lean_object* v___x_318_; lean_object* v___x_319_; 
v___x_318_ = lean_obj_once(&lp_batteries_Batteries_Linter_UnreachableTactic_getTactics___closed__0, &lp_batteries_Batteries_Linter_UnreachableTactic_getTactics___closed__0_once, _init_lp_batteries_Batteries_Linter_UnreachableTactic_getTactics___closed__0);
v___x_319_ = l_StateRefT_x27_instMonad___redArg(v___x_318_);
return v___x_319_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnreachableTactic_getTactics___lam__0___boxed(lean_object* v_ignoreTacticKinds_322_, lean_object* v_isTacKind_323_, lean_object* v_x_324_, lean_object* v___y_325_, lean_object* v___y_326_, lean_object* v___y_327_){
_start:
{
lean_object* v_res_328_; 
v_res_328_ = lp_batteries_Batteries_Linter_UnreachableTactic_getTactics___lam__0(v_ignoreTacticKinds_322_, v_isTacKind_323_, v_x_324_, v___y_325_, v___y_326_);
lean_dec(v___y_326_);
return v_res_328_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnreachableTactic_getTactics(lean_object* v_ignoreTacticKinds_329_, lean_object* v_isTacKind_330_, lean_object* v_stx_331_, lean_object* v_a_332_){
_start:
{
lean_object* v___x_334_; 
v___x_334_ = lean_obj_once(&lp_batteries_Batteries_Linter_UnreachableTactic_getTactics___closed__1, &lp_batteries_Batteries_Linter_UnreachableTactic_getTactics___closed__1_once, _init_lp_batteries_Batteries_Linter_UnreachableTactic_getTactics___closed__1);
if (lean_obj_tag(v_stx_331_) == 1)
{
lean_object* v_kind_335_; lean_object* v_args_336_; lean_object* v___y_338_; lean_object* v___y_362_; uint8_t v___x_363_; 
v_kind_335_ = lean_ctor_get(v_stx_331_, 1);
v_args_336_ = lean_ctor_get(v_stx_331_, 2);
v___x_363_ = lp_batteries_Batteries_Linter_UnreachableTactic_isIgnoreTacticKind(v_ignoreTacticKinds_329_, v_kind_335_);
if (v___x_363_ == 0)
{
lean_object* v___x_364_; lean_object* v___x_365_; uint8_t v___x_366_; 
v___x_364_ = lean_unsigned_to_nat(0u);
v___x_365_ = lean_array_get_size(v_args_336_);
v___x_366_ = lean_nat_dec_lt(v___x_364_, v___x_365_);
if (v___x_366_ == 0)
{
lean_dec_ref(v_ignoreTacticKinds_329_);
v___y_338_ = v_a_332_;
goto v___jp_337_;
}
else
{
lean_object* v___f_367_; lean_object* v___x_368_; uint8_t v___x_369_; 
lean_inc_ref(v_isTacKind_330_);
v___f_367_ = lean_alloc_closure((void*)(lp_batteries_Batteries_Linter_UnreachableTactic_getTactics___lam__0___boxed), 6, 2);
lean_closure_set(v___f_367_, 0, v_ignoreTacticKinds_329_);
lean_closure_set(v___f_367_, 1, v_isTacKind_330_);
v___x_368_ = lean_box(0);
v___x_369_ = lean_nat_dec_le(v___x_365_, v___x_365_);
if (v___x_369_ == 0)
{
if (v___x_366_ == 0)
{
lean_dec_ref(v___f_367_);
v___y_338_ = v_a_332_;
goto v___jp_337_;
}
else
{
size_t v___x_370_; size_t v___x_371_; lean_object* v___x_1198__overap_372_; lean_object* v___x_373_; 
v___x_370_ = ((size_t)0ULL);
v___x_371_ = lean_usize_of_nat(v___x_365_);
lean_inc_ref(v_args_336_);
v___x_1198__overap_372_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_334_, v___f_367_, v_args_336_, v___x_370_, v___x_371_, v___x_368_);
lean_inc(v_a_332_);
v___x_373_ = lean_apply_2(v___x_1198__overap_372_, v_a_332_, lean_box(0));
v___y_362_ = v___x_373_;
goto v___jp_361_;
}
}
else
{
size_t v___x_374_; size_t v___x_375_; lean_object* v___x_1202__overap_376_; lean_object* v___x_377_; 
v___x_374_ = ((size_t)0ULL);
v___x_375_ = lean_usize_of_nat(v___x_365_);
lean_inc_ref(v_args_336_);
v___x_1202__overap_376_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_334_, v___f_367_, v_args_336_, v___x_374_, v___x_375_, v___x_368_);
lean_inc(v_a_332_);
v___x_377_ = lean_apply_2(v___x_1202__overap_376_, v_a_332_, lean_box(0));
v___y_362_ = v___x_377_;
goto v___jp_361_;
}
}
}
else
{
lean_dec_ref(v_ignoreTacticKinds_329_);
v___y_338_ = v_a_332_;
goto v___jp_337_;
}
v___jp_337_:
{
lean_object* v___x_339_; uint8_t v___x_340_; 
lean_inc(v_kind_335_);
v___x_339_ = lean_apply_1(v_isTacKind_330_, v_kind_335_);
v___x_340_ = lean_unbox(v___x_339_);
if (v___x_340_ == 0)
{
lean_object* v___x_341_; lean_object* v___x_342_; 
lean_dec_ref_known(v_stx_331_, 3);
v___x_341_ = lean_box(0);
v___x_342_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_342_, 0, v___x_341_);
return v___x_342_;
}
else
{
uint8_t v___x_343_; lean_object* v___x_344_; 
v___x_343_ = lean_unbox(v___x_339_);
v___x_344_ = l_Lean_Syntax_getRange_x3f(v_stx_331_, v___x_343_);
if (lean_obj_tag(v___x_344_) == 1)
{
lean_object* v_val_345_; lean_object* v___x_347_; uint8_t v_isShared_348_; uint8_t v_isSharedCheck_358_; 
v_val_345_ = lean_ctor_get(v___x_344_, 0);
v_isSharedCheck_358_ = !lean_is_exclusive(v___x_344_);
if (v_isSharedCheck_358_ == 0)
{
v___x_347_ = v___x_344_;
v_isShared_348_ = v_isSharedCheck_358_;
goto v_resetjp_346_;
}
else
{
lean_inc(v_val_345_);
lean_dec(v___x_344_);
v___x_347_ = lean_box(0);
v_isShared_348_ = v_isSharedCheck_358_;
goto v_resetjp_346_;
}
v_resetjp_346_:
{
lean_object* v___x_349_; lean_object* v___x_350_; lean_object* v___x_351_; lean_object* v___x_352_; lean_object* v___x_353_; lean_object* v___x_354_; lean_object* v___x_356_; 
v___x_349_ = lean_st_ref_take(v___y_338_);
v___x_350_ = ((lean_object*)(lp_batteries_Batteries_Linter_UnreachableTactic_getTactics___closed__2));
v___x_351_ = ((lean_object*)(lp_batteries_Batteries_Linter_UnreachableTactic_getTactics___closed__3));
v___x_352_ = l_Std_DHashMap_Internal_Raw_u2080_insert___redArg(v___x_350_, v___x_351_, v___x_349_, v_val_345_, v_stx_331_);
v___x_353_ = lean_st_ref_set(v___y_338_, v___x_352_);
v___x_354_ = lean_box(0);
if (v_isShared_348_ == 0)
{
lean_ctor_set_tag(v___x_347_, 0);
lean_ctor_set(v___x_347_, 0, v___x_354_);
v___x_356_ = v___x_347_;
goto v_reusejp_355_;
}
else
{
lean_object* v_reuseFailAlloc_357_; 
v_reuseFailAlloc_357_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_357_, 0, v___x_354_);
v___x_356_ = v_reuseFailAlloc_357_;
goto v_reusejp_355_;
}
v_reusejp_355_:
{
return v___x_356_;
}
}
}
else
{
lean_object* v___x_359_; lean_object* v___x_360_; 
lean_dec(v___x_344_);
lean_dec_ref_known(v_stx_331_, 3);
v___x_359_ = lean_box(0);
v___x_360_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_360_, 0, v___x_359_);
return v___x_360_;
}
}
}
v___jp_361_:
{
if (lean_obj_tag(v___y_362_) == 0)
{
lean_dec_ref_known(v___y_362_, 1);
v___y_338_ = v_a_332_;
goto v___jp_337_;
}
else
{
lean_dec_ref_known(v_stx_331_, 3);
lean_dec_ref(v_isTacKind_330_);
return v___y_362_;
}
}
}
else
{
lean_object* v___x_378_; lean_object* v___x_379_; 
lean_dec(v_stx_331_);
lean_dec_ref(v_isTacKind_330_);
lean_dec_ref(v_ignoreTacticKinds_329_);
v___x_378_ = lean_box(0);
v___x_379_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_379_, 0, v___x_378_);
return v___x_379_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnreachableTactic_getTactics___lam__0(lean_object* v_ignoreTacticKinds_380_, lean_object* v_isTacKind_381_, lean_object* v_x_382_, lean_object* v___y_383_, lean_object* v___y_384_){
_start:
{
lean_object* v___x_386_; 
v___x_386_ = lp_batteries_Batteries_Linter_UnreachableTactic_getTactics(v_ignoreTacticKinds_380_, v_isTacKind_381_, v___y_383_, v___y_384_);
return v___x_386_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnreachableTactic_getTactics___boxed(lean_object* v_ignoreTacticKinds_387_, lean_object* v_isTacKind_388_, lean_object* v_stx_389_, lean_object* v_a_390_, lean_object* v_a_391_){
_start:
{
lean_object* v_res_392_; 
v_res_392_ = lp_batteries_Batteries_Linter_UnreachableTactic_getTactics(v_ignoreTacticKinds_387_, v_isTacKind_388_, v_stx_389_, v_a_390_);
lean_dec(v_a_390_);
return v_res_392_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnreachableTactic_eraseUsedTactics_spec__2_spec__5___redArg(lean_object* v_a_393_, lean_object* v_x_394_){
_start:
{
if (lean_obj_tag(v_x_394_) == 0)
{
uint8_t v___x_395_; 
v___x_395_ = 0;
return v___x_395_;
}
else
{
lean_object* v_key_396_; lean_object* v_tail_397_; uint8_t v___x_398_; 
v_key_396_ = lean_ctor_get(v_x_394_, 0);
v_tail_397_ = lean_ctor_get(v_x_394_, 2);
v___x_398_ = l_Lean_Syntax_instBEqRange_beq(v_key_396_, v_a_393_);
if (v___x_398_ == 0)
{
v_x_394_ = v_tail_397_;
goto _start;
}
else
{
return v___x_398_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnreachableTactic_eraseUsedTactics_spec__2_spec__5___redArg___boxed(lean_object* v_a_400_, lean_object* v_x_401_){
_start:
{
uint8_t v_res_402_; lean_object* v_r_403_; 
v_res_402_ = lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnreachableTactic_eraseUsedTactics_spec__2_spec__5___redArg(v_a_400_, v_x_401_);
lean_dec(v_x_401_);
lean_dec_ref(v_a_400_);
v_r_403_ = lean_box(v_res_402_);
return v_r_403_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnreachableTactic_eraseUsedTactics_spec__2_spec__6___redArg(lean_object* v_a_404_, lean_object* v_x_405_){
_start:
{
if (lean_obj_tag(v_x_405_) == 0)
{
return v_x_405_;
}
else
{
lean_object* v_key_406_; lean_object* v_value_407_; lean_object* v_tail_408_; lean_object* v___x_410_; uint8_t v_isShared_411_; uint8_t v_isSharedCheck_417_; 
v_key_406_ = lean_ctor_get(v_x_405_, 0);
v_value_407_ = lean_ctor_get(v_x_405_, 1);
v_tail_408_ = lean_ctor_get(v_x_405_, 2);
v_isSharedCheck_417_ = !lean_is_exclusive(v_x_405_);
if (v_isSharedCheck_417_ == 0)
{
v___x_410_ = v_x_405_;
v_isShared_411_ = v_isSharedCheck_417_;
goto v_resetjp_409_;
}
else
{
lean_inc(v_tail_408_);
lean_inc(v_value_407_);
lean_inc(v_key_406_);
lean_dec(v_x_405_);
v___x_410_ = lean_box(0);
v_isShared_411_ = v_isSharedCheck_417_;
goto v_resetjp_409_;
}
v_resetjp_409_:
{
uint8_t v___x_412_; 
v___x_412_ = l_Lean_Syntax_instBEqRange_beq(v_key_406_, v_a_404_);
if (v___x_412_ == 0)
{
lean_object* v___x_413_; lean_object* v___x_415_; 
v___x_413_ = lp_batteries_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnreachableTactic_eraseUsedTactics_spec__2_spec__6___redArg(v_a_404_, v_tail_408_);
if (v_isShared_411_ == 0)
{
lean_ctor_set(v___x_410_, 2, v___x_413_);
v___x_415_ = v___x_410_;
goto v_reusejp_414_;
}
else
{
lean_object* v_reuseFailAlloc_416_; 
v_reuseFailAlloc_416_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_416_, 0, v_key_406_);
lean_ctor_set(v_reuseFailAlloc_416_, 1, v_value_407_);
lean_ctor_set(v_reuseFailAlloc_416_, 2, v___x_413_);
v___x_415_ = v_reuseFailAlloc_416_;
goto v_reusejp_414_;
}
v_reusejp_414_:
{
return v___x_415_;
}
}
else
{
lean_del_object(v___x_410_);
lean_dec(v_value_407_);
lean_dec(v_key_406_);
return v_tail_408_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnreachableTactic_eraseUsedTactics_spec__2_spec__6___redArg___boxed(lean_object* v_a_418_, lean_object* v_x_419_){
_start:
{
lean_object* v_res_420_; 
v_res_420_ = lp_batteries_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnreachableTactic_eraseUsedTactics_spec__2_spec__6___redArg(v_a_418_, v_x_419_);
lean_dec_ref(v_a_418_);
return v_res_420_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnreachableTactic_eraseUsedTactics_spec__2___redArg(lean_object* v_m_421_, lean_object* v_a_422_){
_start:
{
lean_object* v_size_423_; lean_object* v_buckets_424_; lean_object* v___x_425_; uint64_t v___x_426_; uint64_t v___x_427_; uint64_t v___x_428_; uint64_t v_fold_429_; uint64_t v___x_430_; uint64_t v___x_431_; uint64_t v___x_432_; size_t v___x_433_; size_t v___x_434_; size_t v___x_435_; size_t v___x_436_; size_t v___x_437_; lean_object* v_bkt_438_; uint8_t v___x_439_; 
v_size_423_ = lean_ctor_get(v_m_421_, 0);
v_buckets_424_ = lean_ctor_get(v_m_421_, 1);
v___x_425_ = lean_array_get_size(v_buckets_424_);
v___x_426_ = l_Lean_Syntax_instHashableRange_hash(v_a_422_);
v___x_427_ = 32ULL;
v___x_428_ = lean_uint64_shift_right(v___x_426_, v___x_427_);
v_fold_429_ = lean_uint64_xor(v___x_426_, v___x_428_);
v___x_430_ = 16ULL;
v___x_431_ = lean_uint64_shift_right(v_fold_429_, v___x_430_);
v___x_432_ = lean_uint64_xor(v_fold_429_, v___x_431_);
v___x_433_ = lean_uint64_to_usize(v___x_432_);
v___x_434_ = lean_usize_of_nat(v___x_425_);
v___x_435_ = ((size_t)1ULL);
v___x_436_ = lean_usize_sub(v___x_434_, v___x_435_);
v___x_437_ = lean_usize_land(v___x_433_, v___x_436_);
v_bkt_438_ = lean_array_uget_borrowed(v_buckets_424_, v___x_437_);
v___x_439_ = lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnreachableTactic_eraseUsedTactics_spec__2_spec__5___redArg(v_a_422_, v_bkt_438_);
if (v___x_439_ == 0)
{
return v_m_421_;
}
else
{
lean_object* v___x_441_; uint8_t v_isShared_442_; uint8_t v_isSharedCheck_452_; 
lean_inc(v_bkt_438_);
lean_inc_ref(v_buckets_424_);
lean_inc(v_size_423_);
v_isSharedCheck_452_ = !lean_is_exclusive(v_m_421_);
if (v_isSharedCheck_452_ == 0)
{
lean_object* v_unused_453_; lean_object* v_unused_454_; 
v_unused_453_ = lean_ctor_get(v_m_421_, 1);
lean_dec(v_unused_453_);
v_unused_454_ = lean_ctor_get(v_m_421_, 0);
lean_dec(v_unused_454_);
v___x_441_ = v_m_421_;
v_isShared_442_ = v_isSharedCheck_452_;
goto v_resetjp_440_;
}
else
{
lean_dec(v_m_421_);
v___x_441_ = lean_box(0);
v_isShared_442_ = v_isSharedCheck_452_;
goto v_resetjp_440_;
}
v_resetjp_440_:
{
lean_object* v___x_443_; lean_object* v_buckets_x27_444_; lean_object* v___x_445_; lean_object* v___x_446_; lean_object* v___x_447_; lean_object* v___x_448_; lean_object* v___x_450_; 
v___x_443_ = lean_box(0);
v_buckets_x27_444_ = lean_array_uset(v_buckets_424_, v___x_437_, v___x_443_);
v___x_445_ = lean_unsigned_to_nat(1u);
v___x_446_ = lean_nat_sub(v_size_423_, v___x_445_);
lean_dec(v_size_423_);
v___x_447_ = lp_batteries_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnreachableTactic_eraseUsedTactics_spec__2_spec__6___redArg(v_a_422_, v_bkt_438_);
v___x_448_ = lean_array_uset(v_buckets_x27_444_, v___x_437_, v___x_447_);
if (v_isShared_442_ == 0)
{
lean_ctor_set(v___x_441_, 1, v___x_448_);
lean_ctor_set(v___x_441_, 0, v___x_446_);
v___x_450_ = v___x_441_;
goto v_reusejp_449_;
}
else
{
lean_object* v_reuseFailAlloc_451_; 
v_reuseFailAlloc_451_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_451_, 0, v___x_446_);
lean_ctor_set(v_reuseFailAlloc_451_, 1, v___x_448_);
v___x_450_ = v_reuseFailAlloc_451_;
goto v_reusejp_449_;
}
v_reusejp_449_:
{
return v___x_450_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnreachableTactic_eraseUsedTactics_spec__2___redArg___boxed(lean_object* v_m_455_, lean_object* v_a_456_){
_start:
{
lean_object* v_res_457_; 
v_res_457_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnreachableTactic_eraseUsedTactics_spec__2___redArg(v_m_455_, v_a_456_);
lean_dec_ref(v_a_456_);
return v_res_457_;
}
}
static lean_object* _init_lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnreachableTactic_eraseUsedTacticsList_spec__0_spec__0___closed__0(void){
_start:
{
lean_object* v___x_458_; 
v___x_458_ = l_Lean_instInhabitedPersistentArrayNode_default(lean_box(0));
return v___x_458_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnreachableTactic_eraseUsedTactics(lean_object* v_x_459_, lean_object* v_a_460_){
_start:
{
switch(lean_obj_tag(v_x_459_))
{
case 0:
{
lean_object* v_t_462_; 
v_t_462_ = lean_ctor_get(v_x_459_, 1);
lean_inc_ref(v_t_462_);
lean_dec_ref_known(v_x_459_, 2);
v_x_459_ = v_t_462_;
goto _start;
}
case 1:
{
lean_object* v_i_464_; 
v_i_464_ = lean_ctor_get(v_x_459_, 0);
if (lean_obj_tag(v_i_464_) == 0)
{
lean_object* v_i_465_; lean_object* v_toElabInfo_466_; lean_object* v_children_467_; lean_object* v_stx_468_; uint8_t v___x_469_; lean_object* v___x_470_; 
v_i_465_ = lean_ctor_get(v_i_464_, 0);
v_toElabInfo_466_ = lean_ctor_get(v_i_465_, 0);
lean_inc_ref(v_toElabInfo_466_);
v_children_467_ = lean_ctor_get(v_x_459_, 1);
lean_inc_ref(v_children_467_);
lean_dec_ref_known(v_x_459_, 2);
v_stx_468_ = lean_ctor_get(v_toElabInfo_466_, 1);
lean_inc(v_stx_468_);
lean_dec_ref(v_toElabInfo_466_);
v___x_469_ = 1;
v___x_470_ = l_Lean_Syntax_getRange_x3f(v_stx_468_, v___x_469_);
lean_dec(v_stx_468_);
if (lean_obj_tag(v___x_470_) == 1)
{
lean_object* v_val_471_; lean_object* v___x_472_; lean_object* v___x_473_; lean_object* v___x_474_; lean_object* v___x_475_; 
v_val_471_ = lean_ctor_get(v___x_470_, 0);
lean_inc(v_val_471_);
lean_dec_ref_known(v___x_470_, 1);
v___x_472_ = lean_st_ref_take(v_a_460_);
v___x_473_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnreachableTactic_eraseUsedTactics_spec__2___redArg(v___x_472_, v_val_471_);
lean_dec(v_val_471_);
v___x_474_ = lean_st_ref_set(v_a_460_, v___x_473_);
v___x_475_ = lp_batteries_Batteries_Linter_UnreachableTactic_eraseUsedTacticsList(v_children_467_, v_a_460_);
return v___x_475_;
}
else
{
lean_object* v___x_476_; 
lean_dec(v___x_470_);
v___x_476_ = lp_batteries_Batteries_Linter_UnreachableTactic_eraseUsedTacticsList(v_children_467_, v_a_460_);
return v___x_476_;
}
}
else
{
lean_object* v_children_477_; lean_object* v___x_478_; 
v_children_477_ = lean_ctor_get(v_x_459_, 1);
lean_inc_ref(v_children_477_);
lean_dec_ref_known(v_x_459_, 2);
v___x_478_ = lp_batteries_Batteries_Linter_UnreachableTactic_eraseUsedTacticsList(v_children_477_, v_a_460_);
return v___x_478_;
}
}
default: 
{
lean_object* v___x_480_; uint8_t v_isShared_481_; uint8_t v_isSharedCheck_486_; 
v_isSharedCheck_486_ = !lean_is_exclusive(v_x_459_);
if (v_isSharedCheck_486_ == 0)
{
lean_object* v_unused_487_; 
v_unused_487_ = lean_ctor_get(v_x_459_, 0);
lean_dec(v_unused_487_);
v___x_480_ = v_x_459_;
v_isShared_481_ = v_isSharedCheck_486_;
goto v_resetjp_479_;
}
else
{
lean_dec(v_x_459_);
v___x_480_ = lean_box(0);
v_isShared_481_ = v_isSharedCheck_486_;
goto v_resetjp_479_;
}
v_resetjp_479_:
{
lean_object* v___x_482_; lean_object* v___x_484_; 
v___x_482_ = lean_box(0);
if (v_isShared_481_ == 0)
{
lean_ctor_set_tag(v___x_480_, 0);
lean_ctor_set(v___x_480_, 0, v___x_482_);
v___x_484_ = v___x_480_;
goto v_reusejp_483_;
}
else
{
lean_object* v_reuseFailAlloc_485_; 
v_reuseFailAlloc_485_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_485_, 0, v___x_482_);
v___x_484_ = v_reuseFailAlloc_485_;
goto v_reusejp_483_;
}
v_reusejp_483_:
{
return v___x_484_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnreachableTactic_eraseUsedTacticsList_spec__0_spec__1(lean_object* v_as_488_, size_t v_i_489_, size_t v_stop_490_, lean_object* v_b_491_, lean_object* v___y_492_){
_start:
{
uint8_t v___x_494_; 
v___x_494_ = lean_usize_dec_eq(v_i_489_, v_stop_490_);
if (v___x_494_ == 0)
{
lean_object* v___x_495_; lean_object* v___x_496_; 
v___x_495_ = lean_array_uget_borrowed(v_as_488_, v_i_489_);
lean_inc(v___x_495_);
v___x_496_ = lp_batteries_Batteries_Linter_UnreachableTactic_eraseUsedTactics(v___x_495_, v___y_492_);
if (lean_obj_tag(v___x_496_) == 0)
{
lean_object* v_a_497_; size_t v___x_498_; size_t v___x_499_; 
v_a_497_ = lean_ctor_get(v___x_496_, 0);
lean_inc(v_a_497_);
lean_dec_ref_known(v___x_496_, 1);
v___x_498_ = ((size_t)1ULL);
v___x_499_ = lean_usize_add(v_i_489_, v___x_498_);
v_i_489_ = v___x_499_;
v_b_491_ = v_a_497_;
goto _start;
}
else
{
return v___x_496_;
}
}
else
{
lean_object* v___x_501_; 
v___x_501_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_501_, 0, v_b_491_);
return v___x_501_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_forMAux___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnreachableTactic_eraseUsedTacticsList_spec__0_spec__0_spec__2(lean_object* v_x_502_, lean_object* v___y_503_){
_start:
{
if (lean_obj_tag(v_x_502_) == 0)
{
lean_object* v_cs_505_; lean_object* v___x_507_; uint8_t v_isShared_508_; uint8_t v_isSharedCheck_526_; 
v_cs_505_ = lean_ctor_get(v_x_502_, 0);
v_isSharedCheck_526_ = !lean_is_exclusive(v_x_502_);
if (v_isSharedCheck_526_ == 0)
{
v___x_507_ = v_x_502_;
v_isShared_508_ = v_isSharedCheck_526_;
goto v_resetjp_506_;
}
else
{
lean_inc(v_cs_505_);
lean_dec(v_x_502_);
v___x_507_ = lean_box(0);
v_isShared_508_ = v_isSharedCheck_526_;
goto v_resetjp_506_;
}
v_resetjp_506_:
{
lean_object* v___x_509_; lean_object* v___x_510_; lean_object* v___x_511_; uint8_t v___x_512_; 
v___x_509_ = lean_unsigned_to_nat(0u);
v___x_510_ = lean_array_get_size(v_cs_505_);
v___x_511_ = lean_box(0);
v___x_512_ = lean_nat_dec_lt(v___x_509_, v___x_510_);
if (v___x_512_ == 0)
{
lean_object* v___x_514_; 
lean_dec_ref(v_cs_505_);
if (v_isShared_508_ == 0)
{
lean_ctor_set(v___x_507_, 0, v___x_511_);
v___x_514_ = v___x_507_;
goto v_reusejp_513_;
}
else
{
lean_object* v_reuseFailAlloc_515_; 
v_reuseFailAlloc_515_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_515_, 0, v___x_511_);
v___x_514_ = v_reuseFailAlloc_515_;
goto v_reusejp_513_;
}
v_reusejp_513_:
{
return v___x_514_;
}
}
else
{
uint8_t v___x_516_; 
v___x_516_ = lean_nat_dec_le(v___x_510_, v___x_510_);
if (v___x_516_ == 0)
{
if (v___x_512_ == 0)
{
lean_object* v___x_518_; 
lean_dec_ref(v_cs_505_);
if (v_isShared_508_ == 0)
{
lean_ctor_set(v___x_507_, 0, v___x_511_);
v___x_518_ = v___x_507_;
goto v_reusejp_517_;
}
else
{
lean_object* v_reuseFailAlloc_519_; 
v_reuseFailAlloc_519_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_519_, 0, v___x_511_);
v___x_518_ = v_reuseFailAlloc_519_;
goto v_reusejp_517_;
}
v_reusejp_517_:
{
return v___x_518_;
}
}
else
{
size_t v___x_520_; size_t v___x_521_; lean_object* v___x_522_; 
lean_del_object(v___x_507_);
v___x_520_ = ((size_t)0ULL);
v___x_521_ = lean_usize_of_nat(v___x_510_);
v___x_522_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnreachableTactic_eraseUsedTacticsList_spec__0_spec__0_spec__3(v_cs_505_, v___x_520_, v___x_521_, v___x_511_, v___y_503_);
lean_dec_ref(v_cs_505_);
return v___x_522_;
}
}
else
{
size_t v___x_523_; size_t v___x_524_; lean_object* v___x_525_; 
lean_del_object(v___x_507_);
v___x_523_ = ((size_t)0ULL);
v___x_524_ = lean_usize_of_nat(v___x_510_);
v___x_525_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnreachableTactic_eraseUsedTacticsList_spec__0_spec__0_spec__3(v_cs_505_, v___x_523_, v___x_524_, v___x_511_, v___y_503_);
lean_dec_ref(v_cs_505_);
return v___x_525_;
}
}
}
}
else
{
lean_object* v_vs_527_; lean_object* v___x_529_; uint8_t v_isShared_530_; uint8_t v_isSharedCheck_548_; 
v_vs_527_ = lean_ctor_get(v_x_502_, 0);
v_isSharedCheck_548_ = !lean_is_exclusive(v_x_502_);
if (v_isSharedCheck_548_ == 0)
{
v___x_529_ = v_x_502_;
v_isShared_530_ = v_isSharedCheck_548_;
goto v_resetjp_528_;
}
else
{
lean_inc(v_vs_527_);
lean_dec(v_x_502_);
v___x_529_ = lean_box(0);
v_isShared_530_ = v_isSharedCheck_548_;
goto v_resetjp_528_;
}
v_resetjp_528_:
{
lean_object* v___x_531_; lean_object* v___x_532_; lean_object* v___x_533_; uint8_t v___x_534_; 
v___x_531_ = lean_unsigned_to_nat(0u);
v___x_532_ = lean_array_get_size(v_vs_527_);
v___x_533_ = lean_box(0);
v___x_534_ = lean_nat_dec_lt(v___x_531_, v___x_532_);
if (v___x_534_ == 0)
{
lean_object* v___x_536_; 
lean_dec_ref(v_vs_527_);
if (v_isShared_530_ == 0)
{
lean_ctor_set_tag(v___x_529_, 0);
lean_ctor_set(v___x_529_, 0, v___x_533_);
v___x_536_ = v___x_529_;
goto v_reusejp_535_;
}
else
{
lean_object* v_reuseFailAlloc_537_; 
v_reuseFailAlloc_537_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_537_, 0, v___x_533_);
v___x_536_ = v_reuseFailAlloc_537_;
goto v_reusejp_535_;
}
v_reusejp_535_:
{
return v___x_536_;
}
}
else
{
uint8_t v___x_538_; 
v___x_538_ = lean_nat_dec_le(v___x_532_, v___x_532_);
if (v___x_538_ == 0)
{
if (v___x_534_ == 0)
{
lean_object* v___x_540_; 
lean_dec_ref(v_vs_527_);
if (v_isShared_530_ == 0)
{
lean_ctor_set_tag(v___x_529_, 0);
lean_ctor_set(v___x_529_, 0, v___x_533_);
v___x_540_ = v___x_529_;
goto v_reusejp_539_;
}
else
{
lean_object* v_reuseFailAlloc_541_; 
v_reuseFailAlloc_541_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_541_, 0, v___x_533_);
v___x_540_ = v_reuseFailAlloc_541_;
goto v_reusejp_539_;
}
v_reusejp_539_:
{
return v___x_540_;
}
}
else
{
size_t v___x_542_; size_t v___x_543_; lean_object* v___x_544_; 
lean_del_object(v___x_529_);
v___x_542_ = ((size_t)0ULL);
v___x_543_ = lean_usize_of_nat(v___x_532_);
v___x_544_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnreachableTactic_eraseUsedTacticsList_spec__0_spec__1(v_vs_527_, v___x_542_, v___x_543_, v___x_533_, v___y_503_);
lean_dec_ref(v_vs_527_);
return v___x_544_;
}
}
else
{
size_t v___x_545_; size_t v___x_546_; lean_object* v___x_547_; 
lean_del_object(v___x_529_);
v___x_545_ = ((size_t)0ULL);
v___x_546_ = lean_usize_of_nat(v___x_532_);
v___x_547_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnreachableTactic_eraseUsedTacticsList_spec__0_spec__1(v_vs_527_, v___x_545_, v___x_546_, v___x_533_, v___y_503_);
lean_dec_ref(v_vs_527_);
return v___x_547_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnreachableTactic_eraseUsedTacticsList_spec__0_spec__0_spec__3(lean_object* v_as_549_, size_t v_i_550_, size_t v_stop_551_, lean_object* v_b_552_, lean_object* v___y_553_){
_start:
{
uint8_t v___x_555_; 
v___x_555_ = lean_usize_dec_eq(v_i_550_, v_stop_551_);
if (v___x_555_ == 0)
{
lean_object* v___x_556_; lean_object* v___x_557_; 
v___x_556_ = lean_array_uget_borrowed(v_as_549_, v_i_550_);
lean_inc(v___x_556_);
v___x_557_ = lp_batteries_Lean_PersistentArray_forMAux___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnreachableTactic_eraseUsedTacticsList_spec__0_spec__0_spec__2(v___x_556_, v___y_553_);
if (lean_obj_tag(v___x_557_) == 0)
{
lean_object* v_a_558_; size_t v___x_559_; size_t v___x_560_; 
v_a_558_ = lean_ctor_get(v___x_557_, 0);
lean_inc(v_a_558_);
lean_dec_ref_known(v___x_557_, 1);
v___x_559_ = ((size_t)1ULL);
v___x_560_ = lean_usize_add(v_i_550_, v___x_559_);
v_i_550_ = v___x_560_;
v_b_552_ = v_a_558_;
goto _start;
}
else
{
return v___x_557_;
}
}
else
{
lean_object* v___x_562_; 
v___x_562_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_562_, 0, v_b_552_);
return v___x_562_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnreachableTactic_eraseUsedTacticsList_spec__0_spec__0(lean_object* v_x_563_, size_t v_x_564_, size_t v_x_565_, lean_object* v___y_566_){
_start:
{
if (lean_obj_tag(v_x_563_) == 0)
{
lean_object* v_cs_568_; lean_object* v___x_569_; size_t v___x_570_; lean_object* v_j_571_; lean_object* v___x_572_; size_t v___x_573_; size_t v___x_574_; size_t v___x_575_; size_t v___x_576_; size_t v___x_577_; size_t v___x_578_; lean_object* v___x_579_; 
v_cs_568_ = lean_ctor_get(v_x_563_, 0);
lean_inc_ref(v_cs_568_);
lean_dec_ref_known(v_x_563_, 1);
v___x_569_ = lean_obj_once(&lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnreachableTactic_eraseUsedTacticsList_spec__0_spec__0___closed__0, &lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnreachableTactic_eraseUsedTacticsList_spec__0_spec__0___closed__0_once, _init_lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnreachableTactic_eraseUsedTacticsList_spec__0_spec__0___closed__0);
v___x_570_ = lean_usize_shift_right(v_x_564_, v_x_565_);
v_j_571_ = lean_usize_to_nat(v___x_570_);
v___x_572_ = lean_array_get_borrowed(v___x_569_, v_cs_568_, v_j_571_);
v___x_573_ = ((size_t)1ULL);
v___x_574_ = lean_usize_shift_left(v___x_573_, v_x_565_);
v___x_575_ = lean_usize_sub(v___x_574_, v___x_573_);
v___x_576_ = lean_usize_land(v_x_564_, v___x_575_);
v___x_577_ = ((size_t)5ULL);
v___x_578_ = lean_usize_sub(v_x_565_, v___x_577_);
lean_inc(v___x_572_);
v___x_579_ = lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnreachableTactic_eraseUsedTacticsList_spec__0_spec__0(v___x_572_, v___x_576_, v___x_578_, v___y_566_);
if (lean_obj_tag(v___x_579_) == 0)
{
lean_object* v___x_581_; uint8_t v_isShared_582_; uint8_t v_isSharedCheck_601_; 
v_isSharedCheck_601_ = !lean_is_exclusive(v___x_579_);
if (v_isSharedCheck_601_ == 0)
{
lean_object* v_unused_602_; 
v_unused_602_ = lean_ctor_get(v___x_579_, 0);
lean_dec(v_unused_602_);
v___x_581_ = v___x_579_;
v_isShared_582_ = v_isSharedCheck_601_;
goto v_resetjp_580_;
}
else
{
lean_dec(v___x_579_);
v___x_581_ = lean_box(0);
v_isShared_582_ = v_isSharedCheck_601_;
goto v_resetjp_580_;
}
v_resetjp_580_:
{
lean_object* v___x_583_; lean_object* v___x_584_; lean_object* v___x_585_; lean_object* v___x_586_; uint8_t v___x_587_; 
v___x_583_ = lean_unsigned_to_nat(1u);
v___x_584_ = lean_nat_add(v_j_571_, v___x_583_);
lean_dec(v_j_571_);
v___x_585_ = lean_array_get_size(v_cs_568_);
v___x_586_ = lean_box(0);
v___x_587_ = lean_nat_dec_lt(v___x_584_, v___x_585_);
if (v___x_587_ == 0)
{
lean_object* v___x_589_; 
lean_dec(v___x_584_);
lean_dec_ref(v_cs_568_);
if (v_isShared_582_ == 0)
{
lean_ctor_set(v___x_581_, 0, v___x_586_);
v___x_589_ = v___x_581_;
goto v_reusejp_588_;
}
else
{
lean_object* v_reuseFailAlloc_590_; 
v_reuseFailAlloc_590_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_590_, 0, v___x_586_);
v___x_589_ = v_reuseFailAlloc_590_;
goto v_reusejp_588_;
}
v_reusejp_588_:
{
return v___x_589_;
}
}
else
{
uint8_t v___x_591_; 
v___x_591_ = lean_nat_dec_le(v___x_585_, v___x_585_);
if (v___x_591_ == 0)
{
if (v___x_587_ == 0)
{
lean_object* v___x_593_; 
lean_dec(v___x_584_);
lean_dec_ref(v_cs_568_);
if (v_isShared_582_ == 0)
{
lean_ctor_set(v___x_581_, 0, v___x_586_);
v___x_593_ = v___x_581_;
goto v_reusejp_592_;
}
else
{
lean_object* v_reuseFailAlloc_594_; 
v_reuseFailAlloc_594_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_594_, 0, v___x_586_);
v___x_593_ = v_reuseFailAlloc_594_;
goto v_reusejp_592_;
}
v_reusejp_592_:
{
return v___x_593_;
}
}
else
{
size_t v___x_595_; size_t v___x_596_; lean_object* v___x_597_; 
lean_del_object(v___x_581_);
v___x_595_ = lean_usize_of_nat(v___x_584_);
lean_dec(v___x_584_);
v___x_596_ = lean_usize_of_nat(v___x_585_);
v___x_597_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnreachableTactic_eraseUsedTacticsList_spec__0_spec__0_spec__3(v_cs_568_, v___x_595_, v___x_596_, v___x_586_, v___y_566_);
lean_dec_ref(v_cs_568_);
return v___x_597_;
}
}
else
{
size_t v___x_598_; size_t v___x_599_; lean_object* v___x_600_; 
lean_del_object(v___x_581_);
v___x_598_ = lean_usize_of_nat(v___x_584_);
lean_dec(v___x_584_);
v___x_599_ = lean_usize_of_nat(v___x_585_);
v___x_600_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnreachableTactic_eraseUsedTacticsList_spec__0_spec__0_spec__3(v_cs_568_, v___x_598_, v___x_599_, v___x_586_, v___y_566_);
lean_dec_ref(v_cs_568_);
return v___x_600_;
}
}
}
}
else
{
lean_dec(v_j_571_);
lean_dec_ref(v_cs_568_);
return v___x_579_;
}
}
else
{
lean_object* v_vs_603_; lean_object* v___x_605_; uint8_t v_isShared_606_; uint8_t v_isSharedCheck_624_; 
v_vs_603_ = lean_ctor_get(v_x_563_, 0);
v_isSharedCheck_624_ = !lean_is_exclusive(v_x_563_);
if (v_isSharedCheck_624_ == 0)
{
v___x_605_ = v_x_563_;
v_isShared_606_ = v_isSharedCheck_624_;
goto v_resetjp_604_;
}
else
{
lean_inc(v_vs_603_);
lean_dec(v_x_563_);
v___x_605_ = lean_box(0);
v_isShared_606_ = v_isSharedCheck_624_;
goto v_resetjp_604_;
}
v_resetjp_604_:
{
lean_object* v___x_607_; lean_object* v___x_608_; lean_object* v___x_609_; uint8_t v___x_610_; 
v___x_607_ = lean_usize_to_nat(v_x_564_);
v___x_608_ = lean_array_get_size(v_vs_603_);
v___x_609_ = lean_box(0);
v___x_610_ = lean_nat_dec_lt(v___x_607_, v___x_608_);
if (v___x_610_ == 0)
{
lean_object* v___x_612_; 
lean_dec(v___x_607_);
lean_dec_ref(v_vs_603_);
if (v_isShared_606_ == 0)
{
lean_ctor_set_tag(v___x_605_, 0);
lean_ctor_set(v___x_605_, 0, v___x_609_);
v___x_612_ = v___x_605_;
goto v_reusejp_611_;
}
else
{
lean_object* v_reuseFailAlloc_613_; 
v_reuseFailAlloc_613_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_613_, 0, v___x_609_);
v___x_612_ = v_reuseFailAlloc_613_;
goto v_reusejp_611_;
}
v_reusejp_611_:
{
return v___x_612_;
}
}
else
{
uint8_t v___x_614_; 
v___x_614_ = lean_nat_dec_le(v___x_608_, v___x_608_);
if (v___x_614_ == 0)
{
if (v___x_610_ == 0)
{
lean_object* v___x_616_; 
lean_dec(v___x_607_);
lean_dec_ref(v_vs_603_);
if (v_isShared_606_ == 0)
{
lean_ctor_set_tag(v___x_605_, 0);
lean_ctor_set(v___x_605_, 0, v___x_609_);
v___x_616_ = v___x_605_;
goto v_reusejp_615_;
}
else
{
lean_object* v_reuseFailAlloc_617_; 
v_reuseFailAlloc_617_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_617_, 0, v___x_609_);
v___x_616_ = v_reuseFailAlloc_617_;
goto v_reusejp_615_;
}
v_reusejp_615_:
{
return v___x_616_;
}
}
else
{
size_t v___x_618_; size_t v___x_619_; lean_object* v___x_620_; 
lean_del_object(v___x_605_);
v___x_618_ = lean_usize_of_nat(v___x_607_);
lean_dec(v___x_607_);
v___x_619_ = lean_usize_of_nat(v___x_608_);
v___x_620_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnreachableTactic_eraseUsedTacticsList_spec__0_spec__1(v_vs_603_, v___x_618_, v___x_619_, v___x_609_, v___y_566_);
lean_dec_ref(v_vs_603_);
return v___x_620_;
}
}
else
{
size_t v___x_621_; size_t v___x_622_; lean_object* v___x_623_; 
lean_del_object(v___x_605_);
v___x_621_ = lean_usize_of_nat(v___x_607_);
lean_dec(v___x_607_);
v___x_622_ = lean_usize_of_nat(v___x_608_);
v___x_623_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnreachableTactic_eraseUsedTacticsList_spec__0_spec__1(v_vs_603_, v___x_621_, v___x_622_, v___x_609_, v___y_566_);
lean_dec_ref(v_vs_603_);
return v___x_623_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_forMFrom0___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnreachableTactic_eraseUsedTacticsList_spec__0_spec__2(lean_object* v_t_625_, lean_object* v___y_626_){
_start:
{
lean_object* v_root_628_; lean_object* v_tail_629_; lean_object* v___x_630_; 
v_root_628_ = lean_ctor_get(v_t_625_, 0);
lean_inc_ref(v_root_628_);
v_tail_629_ = lean_ctor_get(v_t_625_, 1);
lean_inc_ref(v_tail_629_);
lean_dec_ref(v_t_625_);
v___x_630_ = lp_batteries_Lean_PersistentArray_forMAux___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnreachableTactic_eraseUsedTacticsList_spec__0_spec__0_spec__2(v_root_628_, v___y_626_);
if (lean_obj_tag(v___x_630_) == 0)
{
lean_object* v___x_632_; uint8_t v_isShared_633_; uint8_t v_isSharedCheck_651_; 
v_isSharedCheck_651_ = !lean_is_exclusive(v___x_630_);
if (v_isSharedCheck_651_ == 0)
{
lean_object* v_unused_652_; 
v_unused_652_ = lean_ctor_get(v___x_630_, 0);
lean_dec(v_unused_652_);
v___x_632_ = v___x_630_;
v_isShared_633_ = v_isSharedCheck_651_;
goto v_resetjp_631_;
}
else
{
lean_dec(v___x_630_);
v___x_632_ = lean_box(0);
v_isShared_633_ = v_isSharedCheck_651_;
goto v_resetjp_631_;
}
v_resetjp_631_:
{
lean_object* v___x_634_; lean_object* v___x_635_; lean_object* v___x_636_; uint8_t v___x_637_; 
v___x_634_ = lean_unsigned_to_nat(0u);
v___x_635_ = lean_array_get_size(v_tail_629_);
v___x_636_ = lean_box(0);
v___x_637_ = lean_nat_dec_lt(v___x_634_, v___x_635_);
if (v___x_637_ == 0)
{
lean_object* v___x_639_; 
lean_dec_ref(v_tail_629_);
if (v_isShared_633_ == 0)
{
lean_ctor_set(v___x_632_, 0, v___x_636_);
v___x_639_ = v___x_632_;
goto v_reusejp_638_;
}
else
{
lean_object* v_reuseFailAlloc_640_; 
v_reuseFailAlloc_640_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_640_, 0, v___x_636_);
v___x_639_ = v_reuseFailAlloc_640_;
goto v_reusejp_638_;
}
v_reusejp_638_:
{
return v___x_639_;
}
}
else
{
uint8_t v___x_641_; 
v___x_641_ = lean_nat_dec_le(v___x_635_, v___x_635_);
if (v___x_641_ == 0)
{
if (v___x_637_ == 0)
{
lean_object* v___x_643_; 
lean_dec_ref(v_tail_629_);
if (v_isShared_633_ == 0)
{
lean_ctor_set(v___x_632_, 0, v___x_636_);
v___x_643_ = v___x_632_;
goto v_reusejp_642_;
}
else
{
lean_object* v_reuseFailAlloc_644_; 
v_reuseFailAlloc_644_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_644_, 0, v___x_636_);
v___x_643_ = v_reuseFailAlloc_644_;
goto v_reusejp_642_;
}
v_reusejp_642_:
{
return v___x_643_;
}
}
else
{
size_t v___x_645_; size_t v___x_646_; lean_object* v___x_647_; 
lean_del_object(v___x_632_);
v___x_645_ = ((size_t)0ULL);
v___x_646_ = lean_usize_of_nat(v___x_635_);
v___x_647_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnreachableTactic_eraseUsedTacticsList_spec__0_spec__1(v_tail_629_, v___x_645_, v___x_646_, v___x_636_, v___y_626_);
lean_dec_ref(v_tail_629_);
return v___x_647_;
}
}
else
{
size_t v___x_648_; size_t v___x_649_; lean_object* v___x_650_; 
lean_del_object(v___x_632_);
v___x_648_ = ((size_t)0ULL);
v___x_649_ = lean_usize_of_nat(v___x_635_);
v___x_650_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnreachableTactic_eraseUsedTacticsList_spec__0_spec__1(v_tail_629_, v___x_648_, v___x_649_, v___x_636_, v___y_626_);
lean_dec_ref(v_tail_629_);
return v___x_650_;
}
}
}
}
else
{
lean_dec_ref(v_tail_629_);
return v___x_630_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_forM___at___00Batteries_Linter_UnreachableTactic_eraseUsedTacticsList_spec__0(lean_object* v_t_653_, lean_object* v_start_654_, lean_object* v___y_655_){
_start:
{
lean_object* v___x_657_; uint8_t v___x_658_; 
v___x_657_ = lean_unsigned_to_nat(0u);
v___x_658_ = lean_nat_dec_eq(v_start_654_, v___x_657_);
if (v___x_658_ == 0)
{
lean_object* v_root_659_; lean_object* v_tail_660_; size_t v_shift_661_; lean_object* v_tailOff_662_; uint8_t v___x_663_; 
v_root_659_ = lean_ctor_get(v_t_653_, 0);
lean_inc_ref(v_root_659_);
v_tail_660_ = lean_ctor_get(v_t_653_, 1);
lean_inc_ref(v_tail_660_);
v_shift_661_ = lean_ctor_get_usize(v_t_653_, 4);
v_tailOff_662_ = lean_ctor_get(v_t_653_, 3);
lean_inc(v_tailOff_662_);
lean_dec_ref(v_t_653_);
v___x_663_ = lean_nat_dec_le(v_tailOff_662_, v_start_654_);
if (v___x_663_ == 0)
{
size_t v___x_664_; lean_object* v___x_665_; 
lean_dec(v_tailOff_662_);
v___x_664_ = lean_usize_of_nat(v_start_654_);
v___x_665_ = lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnreachableTactic_eraseUsedTacticsList_spec__0_spec__0(v_root_659_, v___x_664_, v_shift_661_, v___y_655_);
if (lean_obj_tag(v___x_665_) == 0)
{
lean_object* v___x_667_; uint8_t v_isShared_668_; uint8_t v_isSharedCheck_685_; 
v_isSharedCheck_685_ = !lean_is_exclusive(v___x_665_);
if (v_isSharedCheck_685_ == 0)
{
lean_object* v_unused_686_; 
v_unused_686_ = lean_ctor_get(v___x_665_, 0);
lean_dec(v_unused_686_);
v___x_667_ = v___x_665_;
v_isShared_668_ = v_isSharedCheck_685_;
goto v_resetjp_666_;
}
else
{
lean_dec(v___x_665_);
v___x_667_ = lean_box(0);
v_isShared_668_ = v_isSharedCheck_685_;
goto v_resetjp_666_;
}
v_resetjp_666_:
{
lean_object* v___x_669_; lean_object* v___x_670_; uint8_t v___x_671_; 
v___x_669_ = lean_array_get_size(v_tail_660_);
v___x_670_ = lean_box(0);
v___x_671_ = lean_nat_dec_lt(v___x_657_, v___x_669_);
if (v___x_671_ == 0)
{
lean_object* v___x_673_; 
lean_dec_ref(v_tail_660_);
if (v_isShared_668_ == 0)
{
lean_ctor_set(v___x_667_, 0, v___x_670_);
v___x_673_ = v___x_667_;
goto v_reusejp_672_;
}
else
{
lean_object* v_reuseFailAlloc_674_; 
v_reuseFailAlloc_674_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_674_, 0, v___x_670_);
v___x_673_ = v_reuseFailAlloc_674_;
goto v_reusejp_672_;
}
v_reusejp_672_:
{
return v___x_673_;
}
}
else
{
uint8_t v___x_675_; 
v___x_675_ = lean_nat_dec_le(v___x_669_, v___x_669_);
if (v___x_675_ == 0)
{
if (v___x_671_ == 0)
{
lean_object* v___x_677_; 
lean_dec_ref(v_tail_660_);
if (v_isShared_668_ == 0)
{
lean_ctor_set(v___x_667_, 0, v___x_670_);
v___x_677_ = v___x_667_;
goto v_reusejp_676_;
}
else
{
lean_object* v_reuseFailAlloc_678_; 
v_reuseFailAlloc_678_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_678_, 0, v___x_670_);
v___x_677_ = v_reuseFailAlloc_678_;
goto v_reusejp_676_;
}
v_reusejp_676_:
{
return v___x_677_;
}
}
else
{
size_t v___x_679_; size_t v___x_680_; lean_object* v___x_681_; 
lean_del_object(v___x_667_);
v___x_679_ = ((size_t)0ULL);
v___x_680_ = lean_usize_of_nat(v___x_669_);
v___x_681_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnreachableTactic_eraseUsedTacticsList_spec__0_spec__1(v_tail_660_, v___x_679_, v___x_680_, v___x_670_, v___y_655_);
lean_dec_ref(v_tail_660_);
return v___x_681_;
}
}
else
{
size_t v___x_682_; size_t v___x_683_; lean_object* v___x_684_; 
lean_del_object(v___x_667_);
v___x_682_ = ((size_t)0ULL);
v___x_683_ = lean_usize_of_nat(v___x_669_);
v___x_684_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnreachableTactic_eraseUsedTacticsList_spec__0_spec__1(v_tail_660_, v___x_682_, v___x_683_, v___x_670_, v___y_655_);
lean_dec_ref(v_tail_660_);
return v___x_684_;
}
}
}
}
else
{
lean_dec_ref(v_tail_660_);
return v___x_665_;
}
}
else
{
lean_object* v___x_687_; lean_object* v___x_688_; lean_object* v___x_689_; uint8_t v___x_690_; 
lean_dec_ref(v_root_659_);
v___x_687_ = lean_nat_sub(v_start_654_, v_tailOff_662_);
lean_dec(v_tailOff_662_);
v___x_688_ = lean_array_get_size(v_tail_660_);
v___x_689_ = lean_box(0);
v___x_690_ = lean_nat_dec_lt(v___x_687_, v___x_688_);
if (v___x_690_ == 0)
{
lean_object* v___x_691_; 
lean_dec(v___x_687_);
lean_dec_ref(v_tail_660_);
v___x_691_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_691_, 0, v___x_689_);
return v___x_691_;
}
else
{
uint8_t v___x_692_; 
v___x_692_ = lean_nat_dec_le(v___x_688_, v___x_688_);
if (v___x_692_ == 0)
{
if (v___x_690_ == 0)
{
lean_object* v___x_693_; 
lean_dec(v___x_687_);
lean_dec_ref(v_tail_660_);
v___x_693_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_693_, 0, v___x_689_);
return v___x_693_;
}
else
{
size_t v___x_694_; size_t v___x_695_; lean_object* v___x_696_; 
v___x_694_ = lean_usize_of_nat(v___x_687_);
lean_dec(v___x_687_);
v___x_695_ = lean_usize_of_nat(v___x_688_);
v___x_696_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnreachableTactic_eraseUsedTacticsList_spec__0_spec__1(v_tail_660_, v___x_694_, v___x_695_, v___x_689_, v___y_655_);
lean_dec_ref(v_tail_660_);
return v___x_696_;
}
}
else
{
size_t v___x_697_; size_t v___x_698_; lean_object* v___x_699_; 
v___x_697_ = lean_usize_of_nat(v___x_687_);
lean_dec(v___x_687_);
v___x_698_ = lean_usize_of_nat(v___x_688_);
v___x_699_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnreachableTactic_eraseUsedTacticsList_spec__0_spec__1(v_tail_660_, v___x_697_, v___x_698_, v___x_689_, v___y_655_);
lean_dec_ref(v_tail_660_);
return v___x_699_;
}
}
}
}
else
{
lean_object* v___x_700_; 
v___x_700_ = lp_batteries_Lean_PersistentArray_forMFrom0___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnreachableTactic_eraseUsedTacticsList_spec__0_spec__2(v_t_653_, v___y_655_);
return v___x_700_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnreachableTactic_eraseUsedTacticsList(lean_object* v_trees_701_, lean_object* v_a_702_){
_start:
{
lean_object* v___x_704_; lean_object* v___x_705_; 
v___x_704_ = lean_unsigned_to_nat(0u);
v___x_705_ = lp_batteries_Lean_PersistentArray_forM___at___00Batteries_Linter_UnreachableTactic_eraseUsedTacticsList_spec__0(v_trees_701_, v___x_704_, v_a_702_);
return v___x_705_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnreachableTactic_eraseUsedTacticsList___boxed(lean_object* v_trees_706_, lean_object* v_a_707_, lean_object* v_a_708_){
_start:
{
lean_object* v_res_709_; 
v_res_709_ = lp_batteries_Batteries_Linter_UnreachableTactic_eraseUsedTacticsList(v_trees_706_, v_a_707_);
lean_dec(v_a_707_);
return v_res_709_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnreachableTactic_eraseUsedTacticsList_spec__0_spec__1___boxed(lean_object* v_as_710_, lean_object* v_i_711_, lean_object* v_stop_712_, lean_object* v_b_713_, lean_object* v___y_714_, lean_object* v___y_715_){
_start:
{
size_t v_i_boxed_716_; size_t v_stop_boxed_717_; lean_object* v_res_718_; 
v_i_boxed_716_ = lean_unbox_usize(v_i_711_);
lean_dec(v_i_711_);
v_stop_boxed_717_ = lean_unbox_usize(v_stop_712_);
lean_dec(v_stop_712_);
v_res_718_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnreachableTactic_eraseUsedTacticsList_spec__0_spec__1(v_as_710_, v_i_boxed_716_, v_stop_boxed_717_, v_b_713_, v___y_714_);
lean_dec(v___y_714_);
lean_dec_ref(v_as_710_);
return v_res_718_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnreachableTactic_eraseUsedTacticsList_spec__0_spec__0_spec__3___boxed(lean_object* v_as_719_, lean_object* v_i_720_, lean_object* v_stop_721_, lean_object* v_b_722_, lean_object* v___y_723_, lean_object* v___y_724_){
_start:
{
size_t v_i_boxed_725_; size_t v_stop_boxed_726_; lean_object* v_res_727_; 
v_i_boxed_725_ = lean_unbox_usize(v_i_720_);
lean_dec(v_i_720_);
v_stop_boxed_726_ = lean_unbox_usize(v_stop_721_);
lean_dec(v_stop_721_);
v_res_727_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnreachableTactic_eraseUsedTacticsList_spec__0_spec__0_spec__3(v_as_719_, v_i_boxed_725_, v_stop_boxed_726_, v_b_722_, v___y_723_);
lean_dec(v___y_723_);
lean_dec_ref(v_as_719_);
return v_res_727_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_forMFrom0___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnreachableTactic_eraseUsedTacticsList_spec__0_spec__2___boxed(lean_object* v_t_728_, lean_object* v___y_729_, lean_object* v___y_730_){
_start:
{
lean_object* v_res_731_; 
v_res_731_ = lp_batteries_Lean_PersistentArray_forMFrom0___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnreachableTactic_eraseUsedTacticsList_spec__0_spec__2(v_t_728_, v___y_729_);
lean_dec(v___y_729_);
return v_res_731_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnreachableTactic_eraseUsedTactics___boxed(lean_object* v_x_732_, lean_object* v_a_733_, lean_object* v_a_734_){
_start:
{
lean_object* v_res_735_; 
v_res_735_ = lp_batteries_Batteries_Linter_UnreachableTactic_eraseUsedTactics(v_x_732_, v_a_733_);
lean_dec(v_a_733_);
return v_res_735_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_forMAux___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnreachableTactic_eraseUsedTacticsList_spec__0_spec__0_spec__2___boxed(lean_object* v_x_736_, lean_object* v___y_737_, lean_object* v___y_738_){
_start:
{
lean_object* v_res_739_; 
v_res_739_ = lp_batteries_Lean_PersistentArray_forMAux___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnreachableTactic_eraseUsedTacticsList_spec__0_spec__0_spec__2(v_x_736_, v___y_737_);
lean_dec(v___y_737_);
return v_res_739_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_forM___at___00Batteries_Linter_UnreachableTactic_eraseUsedTacticsList_spec__0___boxed(lean_object* v_t_740_, lean_object* v_start_741_, lean_object* v___y_742_, lean_object* v___y_743_){
_start:
{
lean_object* v_res_744_; 
v_res_744_ = lp_batteries_Lean_PersistentArray_forM___at___00Batteries_Linter_UnreachableTactic_eraseUsedTacticsList_spec__0(v_t_740_, v_start_741_, v___y_742_);
lean_dec(v___y_742_);
lean_dec(v_start_741_);
return v_res_744_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnreachableTactic_eraseUsedTacticsList_spec__0_spec__0___boxed(lean_object* v_x_745_, lean_object* v_x_746_, lean_object* v_x_747_, lean_object* v___y_748_, lean_object* v___y_749_){
_start:
{
size_t v_x_2929__boxed_750_; size_t v_x_2930__boxed_751_; lean_object* v_res_752_; 
v_x_2929__boxed_750_ = lean_unbox_usize(v_x_746_);
lean_dec(v_x_746_);
v_x_2930__boxed_751_ = lean_unbox_usize(v_x_747_);
lean_dec(v_x_747_);
v_res_752_ = lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnreachableTactic_eraseUsedTacticsList_spec__0_spec__0(v_x_745_, v_x_2929__boxed_750_, v_x_2930__boxed_751_, v___y_748_);
lean_dec(v___y_748_);
return v_res_752_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnreachableTactic_eraseUsedTactics_spec__2(lean_object* v_00_u03b2_753_, lean_object* v_m_754_, lean_object* v_a_755_){
_start:
{
lean_object* v___x_756_; 
v___x_756_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnreachableTactic_eraseUsedTactics_spec__2___redArg(v_m_754_, v_a_755_);
return v___x_756_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnreachableTactic_eraseUsedTactics_spec__2___boxed(lean_object* v_00_u03b2_757_, lean_object* v_m_758_, lean_object* v_a_759_){
_start:
{
lean_object* v_res_760_; 
v_res_760_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnreachableTactic_eraseUsedTactics_spec__2(v_00_u03b2_757_, v_m_758_, v_a_759_);
lean_dec_ref(v_a_759_);
return v_res_760_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnreachableTactic_eraseUsedTactics_spec__2_spec__5(lean_object* v_00_u03b2_761_, lean_object* v_a_762_, lean_object* v_x_763_){
_start:
{
uint8_t v___x_764_; 
v___x_764_ = lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnreachableTactic_eraseUsedTactics_spec__2_spec__5___redArg(v_a_762_, v_x_763_);
return v___x_764_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnreachableTactic_eraseUsedTactics_spec__2_spec__5___boxed(lean_object* v_00_u03b2_765_, lean_object* v_a_766_, lean_object* v_x_767_){
_start:
{
uint8_t v_res_768_; lean_object* v_r_769_; 
v_res_768_ = lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnreachableTactic_eraseUsedTactics_spec__2_spec__5(v_00_u03b2_765_, v_a_766_, v_x_767_);
lean_dec(v_x_767_);
lean_dec_ref(v_a_766_);
v_r_769_ = lean_box(v_res_768_);
return v_r_769_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnreachableTactic_eraseUsedTactics_spec__2_spec__6(lean_object* v_00_u03b2_770_, lean_object* v_a_771_, lean_object* v_x_772_){
_start:
{
lean_object* v___x_773_; 
v___x_773_ = lp_batteries_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnreachableTactic_eraseUsedTactics_spec__2_spec__6___redArg(v_a_771_, v_x_772_);
return v___x_773_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnreachableTactic_eraseUsedTactics_spec__2_spec__6___boxed(lean_object* v_00_u03b2_774_, lean_object* v_a_775_, lean_object* v_x_776_){
_start:
{
lean_object* v_res_777_; 
v_res_777_ = lp_batteries_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnreachableTactic_eraseUsedTactics_spec__2_spec__6(v_00_u03b2_774_, v_a_775_, v_x_776_);
lean_dec_ref(v_a_775_);
return v_res_777_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nat_cast___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__0(lean_object* v_a_778_){
_start:
{
lean_object* v___x_779_; 
v___x_779_ = lean_nat_to_int(v_a_778_);
return v___x_779_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_getInfoTrees___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__4___redArg(lean_object* v___y_780_){
_start:
{
lean_object* v___x_782_; lean_object* v_infoState_783_; lean_object* v_trees_784_; lean_object* v___x_785_; 
v___x_782_ = lean_st_ref_get(v___y_780_);
v_infoState_783_ = lean_ctor_get(v___x_782_, 8);
lean_inc_ref(v_infoState_783_);
lean_dec(v___x_782_);
v_trees_784_ = lean_ctor_get(v_infoState_783_, 2);
lean_inc_ref(v_trees_784_);
lean_dec_ref(v_infoState_783_);
v___x_785_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_785_, 0, v_trees_784_);
return v___x_785_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_getInfoTrees___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__4___redArg___boxed(lean_object* v___y_786_, lean_object* v___y_787_){
_start:
{
lean_object* v_res_788_; 
v_res_788_ = lp_batteries_Lean_Elab_getInfoTrees___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__4___redArg(v___y_786_);
lean_dec(v___y_786_);
return v_res_788_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_getInfoTrees___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__4(lean_object* v___y_789_, lean_object* v___y_790_){
_start:
{
lean_object* v___x_792_; 
v___x_792_ = lp_batteries_Lean_Elab_getInfoTrees___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__4___redArg(v___y_790_);
return v___x_792_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_getInfoTrees___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__4___boxed(lean_object* v___y_793_, lean_object* v___y_794_, lean_object* v___y_795_){
_start:
{
lean_object* v_res_796_; 
v_res_796_ = lp_batteries_Lean_Elab_getInfoTrees___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__4(v___y_793_, v___y_794_);
lean_dec(v___y_794_);
lean_dec_ref(v___y_793_);
return v_res_796_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnreachableTactic_getTactics___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__10_spec__15_spec__20___redArg(lean_object* v_a_797_, lean_object* v_b_798_, lean_object* v_x_799_){
_start:
{
if (lean_obj_tag(v_x_799_) == 0)
{
lean_dec(v_b_798_);
lean_dec_ref(v_a_797_);
return v_x_799_;
}
else
{
lean_object* v_key_800_; lean_object* v_value_801_; lean_object* v_tail_802_; lean_object* v___x_804_; uint8_t v_isShared_805_; uint8_t v_isSharedCheck_814_; 
v_key_800_ = lean_ctor_get(v_x_799_, 0);
v_value_801_ = lean_ctor_get(v_x_799_, 1);
v_tail_802_ = lean_ctor_get(v_x_799_, 2);
v_isSharedCheck_814_ = !lean_is_exclusive(v_x_799_);
if (v_isSharedCheck_814_ == 0)
{
v___x_804_ = v_x_799_;
v_isShared_805_ = v_isSharedCheck_814_;
goto v_resetjp_803_;
}
else
{
lean_inc(v_tail_802_);
lean_inc(v_value_801_);
lean_inc(v_key_800_);
lean_dec(v_x_799_);
v___x_804_ = lean_box(0);
v_isShared_805_ = v_isSharedCheck_814_;
goto v_resetjp_803_;
}
v_resetjp_803_:
{
uint8_t v___x_806_; 
v___x_806_ = l_Lean_Syntax_instBEqRange_beq(v_key_800_, v_a_797_);
if (v___x_806_ == 0)
{
lean_object* v___x_807_; lean_object* v___x_809_; 
v___x_807_ = lp_batteries_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnreachableTactic_getTactics___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__10_spec__15_spec__20___redArg(v_a_797_, v_b_798_, v_tail_802_);
if (v_isShared_805_ == 0)
{
lean_ctor_set(v___x_804_, 2, v___x_807_);
v___x_809_ = v___x_804_;
goto v_reusejp_808_;
}
else
{
lean_object* v_reuseFailAlloc_810_; 
v_reuseFailAlloc_810_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_810_, 0, v_key_800_);
lean_ctor_set(v_reuseFailAlloc_810_, 1, v_value_801_);
lean_ctor_set(v_reuseFailAlloc_810_, 2, v___x_807_);
v___x_809_ = v_reuseFailAlloc_810_;
goto v_reusejp_808_;
}
v_reusejp_808_:
{
return v___x_809_;
}
}
else
{
lean_object* v___x_812_; 
lean_dec(v_value_801_);
lean_dec(v_key_800_);
if (v_isShared_805_ == 0)
{
lean_ctor_set(v___x_804_, 1, v_b_798_);
lean_ctor_set(v___x_804_, 0, v_a_797_);
v___x_812_ = v___x_804_;
goto v_reusejp_811_;
}
else
{
lean_object* v_reuseFailAlloc_813_; 
v_reuseFailAlloc_813_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_813_, 0, v_a_797_);
lean_ctor_set(v_reuseFailAlloc_813_, 1, v_b_798_);
lean_ctor_set(v_reuseFailAlloc_813_, 2, v_tail_802_);
v___x_812_ = v_reuseFailAlloc_813_;
goto v_reusejp_811_;
}
v_reusejp_811_:
{
return v___x_812_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnreachableTactic_getTactics___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__10_spec__15_spec__19_spec__22_spec__24___redArg(lean_object* v_x_815_, lean_object* v_x_816_){
_start:
{
if (lean_obj_tag(v_x_816_) == 0)
{
return v_x_815_;
}
else
{
lean_object* v_key_817_; lean_object* v_value_818_; lean_object* v_tail_819_; lean_object* v___x_821_; uint8_t v_isShared_822_; uint8_t v_isSharedCheck_842_; 
v_key_817_ = lean_ctor_get(v_x_816_, 0);
v_value_818_ = lean_ctor_get(v_x_816_, 1);
v_tail_819_ = lean_ctor_get(v_x_816_, 2);
v_isSharedCheck_842_ = !lean_is_exclusive(v_x_816_);
if (v_isSharedCheck_842_ == 0)
{
v___x_821_ = v_x_816_;
v_isShared_822_ = v_isSharedCheck_842_;
goto v_resetjp_820_;
}
else
{
lean_inc(v_tail_819_);
lean_inc(v_value_818_);
lean_inc(v_key_817_);
lean_dec(v_x_816_);
v___x_821_ = lean_box(0);
v_isShared_822_ = v_isSharedCheck_842_;
goto v_resetjp_820_;
}
v_resetjp_820_:
{
lean_object* v___x_823_; uint64_t v___x_824_; uint64_t v___x_825_; uint64_t v___x_826_; uint64_t v_fold_827_; uint64_t v___x_828_; uint64_t v___x_829_; uint64_t v___x_830_; size_t v___x_831_; size_t v___x_832_; size_t v___x_833_; size_t v___x_834_; size_t v___x_835_; lean_object* v___x_836_; lean_object* v___x_838_; 
v___x_823_ = lean_array_get_size(v_x_815_);
v___x_824_ = l_Lean_Syntax_instHashableRange_hash(v_key_817_);
v___x_825_ = 32ULL;
v___x_826_ = lean_uint64_shift_right(v___x_824_, v___x_825_);
v_fold_827_ = lean_uint64_xor(v___x_824_, v___x_826_);
v___x_828_ = 16ULL;
v___x_829_ = lean_uint64_shift_right(v_fold_827_, v___x_828_);
v___x_830_ = lean_uint64_xor(v_fold_827_, v___x_829_);
v___x_831_ = lean_uint64_to_usize(v___x_830_);
v___x_832_ = lean_usize_of_nat(v___x_823_);
v___x_833_ = ((size_t)1ULL);
v___x_834_ = lean_usize_sub(v___x_832_, v___x_833_);
v___x_835_ = lean_usize_land(v___x_831_, v___x_834_);
v___x_836_ = lean_array_uget_borrowed(v_x_815_, v___x_835_);
lean_inc(v___x_836_);
if (v_isShared_822_ == 0)
{
lean_ctor_set(v___x_821_, 2, v___x_836_);
v___x_838_ = v___x_821_;
goto v_reusejp_837_;
}
else
{
lean_object* v_reuseFailAlloc_841_; 
v_reuseFailAlloc_841_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_841_, 0, v_key_817_);
lean_ctor_set(v_reuseFailAlloc_841_, 1, v_value_818_);
lean_ctor_set(v_reuseFailAlloc_841_, 2, v___x_836_);
v___x_838_ = v_reuseFailAlloc_841_;
goto v_reusejp_837_;
}
v_reusejp_837_:
{
lean_object* v___x_839_; 
v___x_839_ = lean_array_uset(v_x_815_, v___x_835_, v___x_838_);
v_x_815_ = v___x_839_;
v_x_816_ = v_tail_819_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnreachableTactic_getTactics___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__10_spec__15_spec__19_spec__22___redArg(lean_object* v_i_843_, lean_object* v_source_844_, lean_object* v_target_845_){
_start:
{
lean_object* v___x_846_; uint8_t v___x_847_; 
v___x_846_ = lean_array_get_size(v_source_844_);
v___x_847_ = lean_nat_dec_lt(v_i_843_, v___x_846_);
if (v___x_847_ == 0)
{
lean_dec_ref(v_source_844_);
lean_dec(v_i_843_);
return v_target_845_;
}
else
{
lean_object* v_es_848_; lean_object* v___x_849_; lean_object* v_source_850_; lean_object* v_target_851_; lean_object* v___x_852_; lean_object* v___x_853_; 
v_es_848_ = lean_array_fget(v_source_844_, v_i_843_);
v___x_849_ = lean_box(0);
v_source_850_ = lean_array_fset(v_source_844_, v_i_843_, v___x_849_);
v_target_851_ = lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnreachableTactic_getTactics___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__10_spec__15_spec__19_spec__22_spec__24___redArg(v_target_845_, v_es_848_);
v___x_852_ = lean_unsigned_to_nat(1u);
v___x_853_ = lean_nat_add(v_i_843_, v___x_852_);
lean_dec(v_i_843_);
v_i_843_ = v___x_853_;
v_source_844_ = v_source_850_;
v_target_845_ = v_target_851_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnreachableTactic_getTactics___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__10_spec__15_spec__19___redArg(lean_object* v_data_855_){
_start:
{
lean_object* v___x_856_; lean_object* v___x_857_; lean_object* v_nbuckets_858_; lean_object* v___x_859_; lean_object* v___x_860_; lean_object* v___x_861_; lean_object* v___x_862_; 
v___x_856_ = lean_array_get_size(v_data_855_);
v___x_857_ = lean_unsigned_to_nat(2u);
v_nbuckets_858_ = lean_nat_mul(v___x_856_, v___x_857_);
v___x_859_ = lean_unsigned_to_nat(0u);
v___x_860_ = lean_box(0);
v___x_861_ = lean_mk_array(v_nbuckets_858_, v___x_860_);
v___x_862_ = lp_batteries___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnreachableTactic_getTactics___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__10_spec__15_spec__19_spec__22___redArg(v___x_859_, v_data_855_, v___x_861_);
return v___x_862_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnreachableTactic_getTactics___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__10_spec__15___redArg(lean_object* v_m_863_, lean_object* v_a_864_, lean_object* v_b_865_){
_start:
{
lean_object* v_size_866_; lean_object* v_buckets_867_; lean_object* v___x_869_; uint8_t v_isShared_870_; uint8_t v_isSharedCheck_910_; 
v_size_866_ = lean_ctor_get(v_m_863_, 0);
v_buckets_867_ = lean_ctor_get(v_m_863_, 1);
v_isSharedCheck_910_ = !lean_is_exclusive(v_m_863_);
if (v_isSharedCheck_910_ == 0)
{
v___x_869_ = v_m_863_;
v_isShared_870_ = v_isSharedCheck_910_;
goto v_resetjp_868_;
}
else
{
lean_inc(v_buckets_867_);
lean_inc(v_size_866_);
lean_dec(v_m_863_);
v___x_869_ = lean_box(0);
v_isShared_870_ = v_isSharedCheck_910_;
goto v_resetjp_868_;
}
v_resetjp_868_:
{
lean_object* v___x_871_; uint64_t v___x_872_; uint64_t v___x_873_; uint64_t v___x_874_; uint64_t v_fold_875_; uint64_t v___x_876_; uint64_t v___x_877_; uint64_t v___x_878_; size_t v___x_879_; size_t v___x_880_; size_t v___x_881_; size_t v___x_882_; size_t v___x_883_; lean_object* v_bkt_884_; uint8_t v___x_885_; 
v___x_871_ = lean_array_get_size(v_buckets_867_);
v___x_872_ = l_Lean_Syntax_instHashableRange_hash(v_a_864_);
v___x_873_ = 32ULL;
v___x_874_ = lean_uint64_shift_right(v___x_872_, v___x_873_);
v_fold_875_ = lean_uint64_xor(v___x_872_, v___x_874_);
v___x_876_ = 16ULL;
v___x_877_ = lean_uint64_shift_right(v_fold_875_, v___x_876_);
v___x_878_ = lean_uint64_xor(v_fold_875_, v___x_877_);
v___x_879_ = lean_uint64_to_usize(v___x_878_);
v___x_880_ = lean_usize_of_nat(v___x_871_);
v___x_881_ = ((size_t)1ULL);
v___x_882_ = lean_usize_sub(v___x_880_, v___x_881_);
v___x_883_ = lean_usize_land(v___x_879_, v___x_882_);
v_bkt_884_ = lean_array_uget_borrowed(v_buckets_867_, v___x_883_);
v___x_885_ = lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnreachableTactic_eraseUsedTactics_spec__2_spec__5___redArg(v_a_864_, v_bkt_884_);
if (v___x_885_ == 0)
{
lean_object* v___x_886_; lean_object* v_size_x27_887_; lean_object* v___x_888_; lean_object* v_buckets_x27_889_; lean_object* v___x_890_; lean_object* v___x_891_; lean_object* v___x_892_; lean_object* v___x_893_; lean_object* v___x_894_; uint8_t v___x_895_; 
v___x_886_ = lean_unsigned_to_nat(1u);
v_size_x27_887_ = lean_nat_add(v_size_866_, v___x_886_);
lean_dec(v_size_866_);
lean_inc(v_bkt_884_);
v___x_888_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_888_, 0, v_a_864_);
lean_ctor_set(v___x_888_, 1, v_b_865_);
lean_ctor_set(v___x_888_, 2, v_bkt_884_);
v_buckets_x27_889_ = lean_array_uset(v_buckets_867_, v___x_883_, v___x_888_);
v___x_890_ = lean_unsigned_to_nat(4u);
v___x_891_ = lean_nat_mul(v_size_x27_887_, v___x_890_);
v___x_892_ = lean_unsigned_to_nat(3u);
v___x_893_ = lean_nat_div(v___x_891_, v___x_892_);
lean_dec(v___x_891_);
v___x_894_ = lean_array_get_size(v_buckets_x27_889_);
v___x_895_ = lean_nat_dec_le(v___x_893_, v___x_894_);
lean_dec(v___x_893_);
if (v___x_895_ == 0)
{
lean_object* v_val_896_; lean_object* v___x_898_; 
v_val_896_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnreachableTactic_getTactics___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__10_spec__15_spec__19___redArg(v_buckets_x27_889_);
if (v_isShared_870_ == 0)
{
lean_ctor_set(v___x_869_, 1, v_val_896_);
lean_ctor_set(v___x_869_, 0, v_size_x27_887_);
v___x_898_ = v___x_869_;
goto v_reusejp_897_;
}
else
{
lean_object* v_reuseFailAlloc_899_; 
v_reuseFailAlloc_899_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_899_, 0, v_size_x27_887_);
lean_ctor_set(v_reuseFailAlloc_899_, 1, v_val_896_);
v___x_898_ = v_reuseFailAlloc_899_;
goto v_reusejp_897_;
}
v_reusejp_897_:
{
return v___x_898_;
}
}
else
{
lean_object* v___x_901_; 
if (v_isShared_870_ == 0)
{
lean_ctor_set(v___x_869_, 1, v_buckets_x27_889_);
lean_ctor_set(v___x_869_, 0, v_size_x27_887_);
v___x_901_ = v___x_869_;
goto v_reusejp_900_;
}
else
{
lean_object* v_reuseFailAlloc_902_; 
v_reuseFailAlloc_902_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_902_, 0, v_size_x27_887_);
lean_ctor_set(v_reuseFailAlloc_902_, 1, v_buckets_x27_889_);
v___x_901_ = v_reuseFailAlloc_902_;
goto v_reusejp_900_;
}
v_reusejp_900_:
{
return v___x_901_;
}
}
}
else
{
lean_object* v___x_903_; lean_object* v_buckets_x27_904_; lean_object* v___x_905_; lean_object* v___x_906_; lean_object* v___x_908_; 
lean_inc(v_bkt_884_);
v___x_903_ = lean_box(0);
v_buckets_x27_904_ = lean_array_uset(v_buckets_867_, v___x_883_, v___x_903_);
v___x_905_ = lp_batteries_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnreachableTactic_getTactics___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__10_spec__15_spec__20___redArg(v_a_864_, v_b_865_, v_bkt_884_);
v___x_906_ = lean_array_uset(v_buckets_x27_904_, v___x_883_, v___x_905_);
if (v_isShared_870_ == 0)
{
lean_ctor_set(v___x_869_, 1, v___x_906_);
v___x_908_ = v___x_869_;
goto v_reusejp_907_;
}
else
{
lean_object* v_reuseFailAlloc_909_; 
v_reuseFailAlloc_909_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_909_, 0, v_size_866_);
lean_ctor_set(v_reuseFailAlloc_909_, 1, v___x_906_);
v___x_908_ = v_reuseFailAlloc_909_;
goto v_reusejp_907_;
}
v_reusejp_907_:
{
return v___x_908_;
}
}
}
}
}
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__9_spec__13_spec__16___redArg(lean_object* v_keys_911_, lean_object* v_i_912_, lean_object* v_k_913_){
_start:
{
lean_object* v___x_914_; uint8_t v___x_915_; 
v___x_914_ = lean_array_get_size(v_keys_911_);
v___x_915_ = lean_nat_dec_lt(v_i_912_, v___x_914_);
if (v___x_915_ == 0)
{
lean_dec(v_i_912_);
return v___x_915_;
}
else
{
lean_object* v_k_x27_916_; uint8_t v___x_917_; 
v_k_x27_916_ = lean_array_fget_borrowed(v_keys_911_, v_i_912_);
v___x_917_ = lean_name_eq(v_k_913_, v_k_x27_916_);
if (v___x_917_ == 0)
{
lean_object* v___x_918_; lean_object* v___x_919_; 
v___x_918_ = lean_unsigned_to_nat(1u);
v___x_919_ = lean_nat_add(v_i_912_, v___x_918_);
lean_dec(v_i_912_);
v_i_912_ = v___x_919_;
goto _start;
}
else
{
lean_dec(v_i_912_);
return v___x_917_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__9_spec__13_spec__16___redArg___boxed(lean_object* v_keys_921_, lean_object* v_i_922_, lean_object* v_k_923_){
_start:
{
uint8_t v_res_924_; lean_object* v_r_925_; 
v_res_924_ = lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__9_spec__13_spec__16___redArg(v_keys_921_, v_i_922_, v_k_923_);
lean_dec(v_k_923_);
lean_dec_ref(v_keys_921_);
v_r_925_ = lean_box(v_res_924_);
return v_r_925_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__9_spec__13___redArg(lean_object* v_x_926_, size_t v_x_927_, lean_object* v_x_928_){
_start:
{
if (lean_obj_tag(v_x_926_) == 0)
{
lean_object* v_es_929_; lean_object* v___x_930_; size_t v___x_931_; size_t v___x_932_; lean_object* v_j_933_; lean_object* v___x_934_; 
v_es_929_ = lean_ctor_get(v_x_926_, 0);
v___x_930_ = lean_box(2);
v___x_931_ = ((size_t)31ULL);
v___x_932_ = lean_usize_land(v_x_927_, v___x_931_);
v_j_933_ = lean_usize_to_nat(v___x_932_);
v___x_934_ = lean_array_get_borrowed(v___x_930_, v_es_929_, v_j_933_);
lean_dec(v_j_933_);
switch(lean_obj_tag(v___x_934_))
{
case 0:
{
lean_object* v_key_935_; uint8_t v___x_936_; 
v_key_935_ = lean_ctor_get(v___x_934_, 0);
v___x_936_ = lean_name_eq(v_x_928_, v_key_935_);
return v___x_936_;
}
case 1:
{
lean_object* v_node_937_; size_t v___x_938_; size_t v___x_939_; 
v_node_937_ = lean_ctor_get(v___x_934_, 0);
v___x_938_ = ((size_t)5ULL);
v___x_939_ = lean_usize_shift_right(v_x_927_, v___x_938_);
v_x_926_ = v_node_937_;
v_x_927_ = v___x_939_;
goto _start;
}
default: 
{
uint8_t v___x_941_; 
v___x_941_ = 0;
return v___x_941_;
}
}
}
else
{
lean_object* v_ks_942_; lean_object* v___x_943_; uint8_t v___x_944_; 
v_ks_942_ = lean_ctor_get(v_x_926_, 0);
v___x_943_ = lean_unsigned_to_nat(0u);
v___x_944_ = lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__9_spec__13_spec__16___redArg(v_ks_942_, v___x_943_, v_x_928_);
return v___x_944_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__9_spec__13___redArg___boxed(lean_object* v_x_945_, lean_object* v_x_946_, lean_object* v_x_947_){
_start:
{
size_t v_x_13396__boxed_948_; uint8_t v_res_949_; lean_object* v_r_950_; 
v_x_13396__boxed_948_ = lean_unbox_usize(v_x_946_);
lean_dec(v_x_946_);
v_res_949_ = lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__9_spec__13___redArg(v_x_945_, v_x_13396__boxed_948_, v_x_947_);
lean_dec(v_x_947_);
lean_dec_ref(v_x_945_);
v_r_950_ = lean_box(v_res_949_);
return v_r_950_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_contains___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__9___redArg(lean_object* v_x_951_, lean_object* v_x_952_){
_start:
{
uint64_t v___y_954_; 
if (lean_obj_tag(v_x_952_) == 0)
{
uint64_t v___x_957_; 
v___x_957_ = 1723ULL;
v___y_954_ = v___x_957_;
goto v___jp_953_;
}
else
{
uint64_t v_hash_958_; 
v_hash_958_ = lean_ctor_get_uint64(v_x_952_, sizeof(void*)*2);
v___y_954_ = v_hash_958_;
goto v___jp_953_;
}
v___jp_953_:
{
size_t v___x_955_; uint8_t v___x_956_; 
v___x_955_ = lean_uint64_to_usize(v___y_954_);
v___x_956_ = lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__9_spec__13___redArg(v_x_951_, v___x_955_, v_x_952_);
return v___x_956_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_contains___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__9___redArg___boxed(lean_object* v_x_959_, lean_object* v_x_960_){
_start:
{
uint8_t v_res_961_; lean_object* v_r_962_; 
v_res_961_ = lp_batteries_Lean_PersistentHashMap_contains___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__9___redArg(v_x_959_, v_x_960_);
lean_dec(v_x_960_);
lean_dec_ref(v_x_959_);
v_r_962_ = lean_box(v_res_961_);
return v_r_962_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnreachableTactic_getTactics___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__10(lean_object* v___x_963_, lean_object* v___x_964_, uint8_t v___y_965_, lean_object* v_ignoreTacticKinds_966_, lean_object* v_stx_967_, lean_object* v_a_968_){
_start:
{
lean_object* v___y_971_; uint8_t v___y_972_; 
if (lean_obj_tag(v_stx_967_) == 1)
{
lean_object* v_kind_990_; lean_object* v_args_991_; lean_object* v___y_993_; lean_object* v___y_997_; uint8_t v___x_998_; 
v_kind_990_ = lean_ctor_get(v_stx_967_, 1);
v_args_991_ = lean_ctor_get(v_stx_967_, 2);
v___x_998_ = lp_batteries_Batteries_Linter_UnreachableTactic_isIgnoreTacticKind(v_ignoreTacticKinds_966_, v_kind_990_);
if (v___x_998_ == 0)
{
lean_object* v___x_999_; lean_object* v___x_1000_; uint8_t v___x_1001_; 
v___x_999_ = lean_unsigned_to_nat(0u);
v___x_1000_ = lean_array_get_size(v_args_991_);
v___x_1001_ = lean_nat_dec_lt(v___x_999_, v___x_1000_);
if (v___x_1001_ == 0)
{
v___y_993_ = v_a_968_;
goto v___jp_992_;
}
else
{
lean_object* v___x_1002_; uint8_t v___x_1003_; 
v___x_1002_ = lean_box(0);
v___x_1003_ = lean_nat_dec_le(v___x_1000_, v___x_1000_);
if (v___x_1003_ == 0)
{
if (v___x_1001_ == 0)
{
v___y_993_ = v_a_968_;
goto v___jp_992_;
}
else
{
size_t v___x_1004_; size_t v___x_1005_; lean_object* v___x_1006_; 
v___x_1004_ = ((size_t)0ULL);
v___x_1005_ = lean_usize_of_nat(v___x_1000_);
v___x_1006_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Linter_UnreachableTactic_getTactics___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__10_spec__16(v___x_963_, v___x_964_, v___y_965_, v_ignoreTacticKinds_966_, v_args_991_, v___x_1004_, v___x_1005_, v___x_1002_, v_a_968_);
v___y_997_ = v___x_1006_;
goto v___jp_996_;
}
}
else
{
size_t v___x_1007_; size_t v___x_1008_; lean_object* v___x_1009_; 
v___x_1007_ = ((size_t)0ULL);
v___x_1008_ = lean_usize_of_nat(v___x_1000_);
v___x_1009_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Linter_UnreachableTactic_getTactics___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__10_spec__16(v___x_963_, v___x_964_, v___y_965_, v_ignoreTacticKinds_966_, v_args_991_, v___x_1007_, v___x_1008_, v___x_1002_, v_a_968_);
v___y_997_ = v___x_1009_;
goto v___jp_996_;
}
}
}
else
{
v___y_993_ = v_a_968_;
goto v___jp_992_;
}
v___jp_992_:
{
uint8_t v___x_994_; 
v___x_994_ = lp_batteries_Lean_PersistentHashMap_contains___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__9___redArg(v___x_963_, v_kind_990_);
if (v___x_994_ == 0)
{
uint8_t v___x_995_; 
v___x_995_ = lp_batteries_Lean_PersistentHashMap_contains___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__9___redArg(v___x_964_, v_kind_990_);
v___y_971_ = v___y_993_;
v___y_972_ = v___x_995_;
goto v___jp_970_;
}
else
{
v___y_971_ = v___y_993_;
v___y_972_ = v___y_965_;
goto v___jp_970_;
}
}
v___jp_996_:
{
if (lean_obj_tag(v___y_997_) == 0)
{
lean_dec_ref_known(v___y_997_, 1);
v___y_993_ = v_a_968_;
goto v___jp_992_;
}
else
{
lean_dec_ref_known(v_stx_967_, 3);
return v___y_997_;
}
}
}
else
{
lean_object* v___x_1010_; lean_object* v___x_1011_; 
lean_dec(v_stx_967_);
v___x_1010_ = lean_box(0);
v___x_1011_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1011_, 0, v___x_1010_);
return v___x_1011_;
}
v___jp_970_:
{
if (v___y_972_ == 0)
{
lean_object* v___x_973_; lean_object* v___x_974_; 
lean_dec(v_stx_967_);
v___x_973_ = lean_box(0);
v___x_974_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_974_, 0, v___x_973_);
return v___x_974_;
}
else
{
lean_object* v___x_975_; 
v___x_975_ = l_Lean_Syntax_getRange_x3f(v_stx_967_, v___y_972_);
if (lean_obj_tag(v___x_975_) == 1)
{
lean_object* v_val_976_; lean_object* v___x_978_; uint8_t v_isShared_979_; uint8_t v_isSharedCheck_987_; 
v_val_976_ = lean_ctor_get(v___x_975_, 0);
v_isSharedCheck_987_ = !lean_is_exclusive(v___x_975_);
if (v_isSharedCheck_987_ == 0)
{
v___x_978_ = v___x_975_;
v_isShared_979_ = v_isSharedCheck_987_;
goto v_resetjp_977_;
}
else
{
lean_inc(v_val_976_);
lean_dec(v___x_975_);
v___x_978_ = lean_box(0);
v_isShared_979_ = v_isSharedCheck_987_;
goto v_resetjp_977_;
}
v_resetjp_977_:
{
lean_object* v___x_980_; lean_object* v___x_981_; lean_object* v___x_982_; lean_object* v___x_983_; lean_object* v___x_985_; 
v___x_980_ = lean_st_ref_take(v___y_971_);
v___x_981_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnreachableTactic_getTactics___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__10_spec__15___redArg(v___x_980_, v_val_976_, v_stx_967_);
v___x_982_ = lean_st_ref_set(v___y_971_, v___x_981_);
v___x_983_ = lean_box(0);
if (v_isShared_979_ == 0)
{
lean_ctor_set_tag(v___x_978_, 0);
lean_ctor_set(v___x_978_, 0, v___x_983_);
v___x_985_ = v___x_978_;
goto v_reusejp_984_;
}
else
{
lean_object* v_reuseFailAlloc_986_; 
v_reuseFailAlloc_986_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_986_, 0, v___x_983_);
v___x_985_ = v_reuseFailAlloc_986_;
goto v_reusejp_984_;
}
v_reusejp_984_:
{
return v___x_985_;
}
}
}
else
{
lean_object* v___x_988_; lean_object* v___x_989_; 
lean_dec(v___x_975_);
lean_dec(v_stx_967_);
v___x_988_ = lean_box(0);
v___x_989_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_989_, 0, v___x_988_);
return v___x_989_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Linter_UnreachableTactic_getTactics___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__10_spec__16(lean_object* v___x_1012_, lean_object* v___x_1013_, uint8_t v___y_1014_, lean_object* v_ignoreTacticKinds_1015_, lean_object* v_as_1016_, size_t v_i_1017_, size_t v_stop_1018_, lean_object* v_b_1019_, lean_object* v___y_1020_){
_start:
{
uint8_t v___x_1022_; 
v___x_1022_ = lean_usize_dec_eq(v_i_1017_, v_stop_1018_);
if (v___x_1022_ == 0)
{
lean_object* v___x_1023_; lean_object* v___x_1024_; 
v___x_1023_ = lean_array_uget_borrowed(v_as_1016_, v_i_1017_);
lean_inc(v___x_1023_);
v___x_1024_ = lp_batteries_Batteries_Linter_UnreachableTactic_getTactics___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__10(v___x_1012_, v___x_1013_, v___y_1014_, v_ignoreTacticKinds_1015_, v___x_1023_, v___y_1020_);
if (lean_obj_tag(v___x_1024_) == 0)
{
lean_object* v_a_1025_; size_t v___x_1026_; size_t v___x_1027_; 
v_a_1025_ = lean_ctor_get(v___x_1024_, 0);
lean_inc(v_a_1025_);
lean_dec_ref_known(v___x_1024_, 1);
v___x_1026_ = ((size_t)1ULL);
v___x_1027_ = lean_usize_add(v_i_1017_, v___x_1026_);
v_i_1017_ = v___x_1027_;
v_b_1019_ = v_a_1025_;
goto _start;
}
else
{
return v___x_1024_;
}
}
else
{
lean_object* v___x_1029_; 
v___x_1029_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1029_, 0, v_b_1019_);
return v___x_1029_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Linter_UnreachableTactic_getTactics___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__10_spec__16___boxed(lean_object* v___x_1030_, lean_object* v___x_1031_, lean_object* v___y_1032_, lean_object* v_ignoreTacticKinds_1033_, lean_object* v_as_1034_, lean_object* v_i_1035_, lean_object* v_stop_1036_, lean_object* v_b_1037_, lean_object* v___y_1038_, lean_object* v___y_1039_){
_start:
{
uint8_t v___y_13454__boxed_1040_; size_t v_i_boxed_1041_; size_t v_stop_boxed_1042_; lean_object* v_res_1043_; 
v___y_13454__boxed_1040_ = lean_unbox(v___y_1032_);
v_i_boxed_1041_ = lean_unbox_usize(v_i_1035_);
lean_dec(v_i_1035_);
v_stop_boxed_1042_ = lean_unbox_usize(v_stop_1036_);
lean_dec(v_stop_1036_);
v_res_1043_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Linter_UnreachableTactic_getTactics___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__10_spec__16(v___x_1030_, v___x_1031_, v___y_13454__boxed_1040_, v_ignoreTacticKinds_1033_, v_as_1034_, v_i_boxed_1041_, v_stop_boxed_1042_, v_b_1037_, v___y_1038_);
lean_dec(v___y_1038_);
lean_dec_ref(v_as_1034_);
lean_dec_ref(v_ignoreTacticKinds_1033_);
lean_dec_ref(v___x_1031_);
lean_dec_ref(v___x_1030_);
return v_res_1043_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnreachableTactic_getTactics___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__10___boxed(lean_object* v___x_1044_, lean_object* v___x_1045_, lean_object* v___y_1046_, lean_object* v_ignoreTacticKinds_1047_, lean_object* v_stx_1048_, lean_object* v_a_1049_, lean_object* v_a_1050_){
_start:
{
uint8_t v___y_13468__boxed_1051_; lean_object* v_res_1052_; 
v___y_13468__boxed_1051_ = lean_unbox(v___y_1046_);
v_res_1052_ = lp_batteries_Batteries_Linter_UnreachableTactic_getTactics___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__10(v___x_1044_, v___x_1045_, v___y_13468__boxed_1051_, v_ignoreTacticKinds_1047_, v_stx_1048_, v_a_1049_);
lean_dec(v_a_1049_);
lean_dec_ref(v_ignoreTacticKinds_1047_);
lean_dec_ref(v___x_1045_);
lean_dec_ref(v___x_1044_);
return v_res_1052_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3_spec__14(lean_object* v_opts_1053_, lean_object* v_opt_1054_){
_start:
{
lean_object* v_name_1055_; lean_object* v_defValue_1056_; lean_object* v_map_1057_; lean_object* v___x_1058_; 
v_name_1055_ = lean_ctor_get(v_opt_1054_, 0);
v_defValue_1056_ = lean_ctor_get(v_opt_1054_, 1);
v_map_1057_ = lean_ctor_get(v_opts_1053_, 0);
v___x_1058_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_1057_, v_name_1055_);
if (lean_obj_tag(v___x_1058_) == 0)
{
uint8_t v___x_1059_; 
v___x_1059_ = lean_unbox(v_defValue_1056_);
return v___x_1059_;
}
else
{
lean_object* v_val_1060_; 
v_val_1060_ = lean_ctor_get(v___x_1058_, 0);
lean_inc(v_val_1060_);
lean_dec_ref_known(v___x_1058_, 1);
if (lean_obj_tag(v_val_1060_) == 1)
{
uint8_t v_v_1061_; 
v_v_1061_ = lean_ctor_get_uint8(v_val_1060_, 0);
lean_dec_ref_known(v_val_1060_, 0);
return v_v_1061_;
}
else
{
uint8_t v___x_1062_; 
lean_dec(v_val_1060_);
v___x_1062_ = lean_unbox(v_defValue_1056_);
return v___x_1062_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3_spec__14___boxed(lean_object* v_opts_1063_, lean_object* v_opt_1064_){
_start:
{
uint8_t v_res_1065_; lean_object* v_r_1066_; 
v_res_1065_ = lp_batteries_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3_spec__14(v_opts_1063_, v_opt_1064_);
lean_dec_ref(v_opt_1064_);
lean_dec_ref(v_opts_1063_);
v_r_1066_ = lean_box(v_res_1065_);
return v_r_1066_;
}
}
static lean_object* _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3_spec__13___redArg___closed__0(void){
_start:
{
lean_object* v___x_1067_; 
v___x_1067_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_1067_;
}
}
static lean_object* _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3_spec__13___redArg___closed__1(void){
_start:
{
lean_object* v___x_1068_; lean_object* v___x_1069_; 
v___x_1068_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3_spec__13___redArg___closed__0, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3_spec__13___redArg___closed__0_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3_spec__13___redArg___closed__0);
v___x_1069_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1069_, 0, v___x_1068_);
return v___x_1069_;
}
}
static lean_object* _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3_spec__13___redArg___closed__2(void){
_start:
{
lean_object* v___x_1070_; lean_object* v___x_1071_; lean_object* v___x_1072_; 
v___x_1070_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3_spec__13___redArg___closed__1, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3_spec__13___redArg___closed__1_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3_spec__13___redArg___closed__1);
v___x_1071_ = lean_unsigned_to_nat(0u);
v___x_1072_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_1072_, 0, v___x_1071_);
lean_ctor_set(v___x_1072_, 1, v___x_1071_);
lean_ctor_set(v___x_1072_, 2, v___x_1071_);
lean_ctor_set(v___x_1072_, 3, v___x_1071_);
lean_ctor_set(v___x_1072_, 4, v___x_1070_);
lean_ctor_set(v___x_1072_, 5, v___x_1070_);
lean_ctor_set(v___x_1072_, 6, v___x_1070_);
lean_ctor_set(v___x_1072_, 7, v___x_1070_);
lean_ctor_set(v___x_1072_, 8, v___x_1070_);
lean_ctor_set(v___x_1072_, 9, v___x_1070_);
return v___x_1072_;
}
}
static lean_object* _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3_spec__13___redArg___closed__3(void){
_start:
{
lean_object* v___x_1073_; lean_object* v___x_1074_; lean_object* v___x_1075_; 
v___x_1073_ = lean_unsigned_to_nat(32u);
v___x_1074_ = lean_mk_empty_array_with_capacity(v___x_1073_);
v___x_1075_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1075_, 0, v___x_1074_);
return v___x_1075_;
}
}
static lean_object* _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3_spec__13___redArg___closed__4(void){
_start:
{
size_t v___x_1076_; lean_object* v___x_1077_; lean_object* v___x_1078_; lean_object* v___x_1079_; lean_object* v___x_1080_; lean_object* v___x_1081_; 
v___x_1076_ = ((size_t)5ULL);
v___x_1077_ = lean_unsigned_to_nat(0u);
v___x_1078_ = lean_unsigned_to_nat(32u);
v___x_1079_ = lean_mk_empty_array_with_capacity(v___x_1078_);
v___x_1080_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3_spec__13___redArg___closed__3, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3_spec__13___redArg___closed__3_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3_spec__13___redArg___closed__3);
v___x_1081_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_1081_, 0, v___x_1080_);
lean_ctor_set(v___x_1081_, 1, v___x_1079_);
lean_ctor_set(v___x_1081_, 2, v___x_1077_);
lean_ctor_set(v___x_1081_, 3, v___x_1077_);
lean_ctor_set_usize(v___x_1081_, 4, v___x_1076_);
return v___x_1081_;
}
}
static lean_object* _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3_spec__13___redArg___closed__5(void){
_start:
{
lean_object* v___x_1082_; lean_object* v___x_1083_; lean_object* v___x_1084_; lean_object* v___x_1085_; 
v___x_1082_ = lean_box(1);
v___x_1083_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3_spec__13___redArg___closed__4, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3_spec__13___redArg___closed__4_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3_spec__13___redArg___closed__4);
v___x_1084_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3_spec__13___redArg___closed__1, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3_spec__13___redArg___closed__1_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3_spec__13___redArg___closed__1);
v___x_1085_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1085_, 0, v___x_1084_);
lean_ctor_set(v___x_1085_, 1, v___x_1083_);
lean_ctor_set(v___x_1085_, 2, v___x_1082_);
return v___x_1085_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3_spec__13___redArg(lean_object* v_msgData_1086_, lean_object* v___y_1087_){
_start:
{
lean_object* v___x_1089_; lean_object* v_env_1090_; lean_object* v___x_1091_; lean_object* v_scopes_1092_; lean_object* v___x_1093_; lean_object* v___x_1094_; lean_object* v_opts_1095_; lean_object* v___x_1096_; lean_object* v___x_1097_; lean_object* v___x_1098_; lean_object* v___x_1099_; lean_object* v___x_1100_; 
v___x_1089_ = lean_st_ref_get(v___y_1087_);
v_env_1090_ = lean_ctor_get(v___x_1089_, 0);
lean_inc_ref(v_env_1090_);
lean_dec(v___x_1089_);
v___x_1091_ = lean_st_ref_get(v___y_1087_);
v_scopes_1092_ = lean_ctor_get(v___x_1091_, 2);
lean_inc(v_scopes_1092_);
lean_dec(v___x_1091_);
v___x_1093_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_1094_ = l_List_head_x21___redArg(v___x_1093_, v_scopes_1092_);
lean_dec(v_scopes_1092_);
v_opts_1095_ = lean_ctor_get(v___x_1094_, 1);
lean_inc_ref(v_opts_1095_);
lean_dec(v___x_1094_);
v___x_1096_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3_spec__13___redArg___closed__2, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3_spec__13___redArg___closed__2_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3_spec__13___redArg___closed__2);
v___x_1097_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3_spec__13___redArg___closed__5, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3_spec__13___redArg___closed__5_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3_spec__13___redArg___closed__5);
v___x_1098_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1098_, 0, v_env_1090_);
lean_ctor_set(v___x_1098_, 1, v___x_1096_);
lean_ctor_set(v___x_1098_, 2, v___x_1097_);
lean_ctor_set(v___x_1098_, 3, v_opts_1095_);
v___x_1099_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_1099_, 0, v___x_1098_);
lean_ctor_set(v___x_1099_, 1, v_msgData_1086_);
v___x_1100_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1100_, 0, v___x_1099_);
return v___x_1100_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3_spec__13___redArg___boxed(lean_object* v_msgData_1101_, lean_object* v___y_1102_, lean_object* v___y_1103_){
_start:
{
lean_object* v_res_1104_; 
v_res_1104_ = lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3_spec__13___redArg(v_msgData_1101_, v___y_1102_);
lean_dec(v___y_1102_);
return v_res_1104_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3___lam__0(uint8_t v___y_1106_, uint8_t v_suppressElabErrors_1107_, lean_object* v_x_1108_){
_start:
{
if (lean_obj_tag(v_x_1108_) == 1)
{
lean_object* v_pre_1109_; 
v_pre_1109_ = lean_ctor_get(v_x_1108_, 0);
if (lean_obj_tag(v_pre_1109_) == 0)
{
lean_object* v_str_1110_; lean_object* v___x_1111_; uint8_t v___x_1112_; 
v_str_1110_ = lean_ctor_get(v_x_1108_, 1);
v___x_1111_ = ((lean_object*)(lp_batteries_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3___lam__0___closed__0));
v___x_1112_ = lean_string_dec_eq(v_str_1110_, v___x_1111_);
if (v___x_1112_ == 0)
{
return v___y_1106_;
}
else
{
return v_suppressElabErrors_1107_;
}
}
else
{
return v___y_1106_;
}
}
else
{
return v___y_1106_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3___lam__0___boxed(lean_object* v___y_1113_, lean_object* v_suppressElabErrors_1114_, lean_object* v_x_1115_){
_start:
{
uint8_t v___y_13694__boxed_1116_; uint8_t v_suppressElabErrors_boxed_1117_; uint8_t v_res_1118_; lean_object* v_r_1119_; 
v___y_13694__boxed_1116_ = lean_unbox(v___y_1113_);
v_suppressElabErrors_boxed_1117_ = lean_unbox(v_suppressElabErrors_1114_);
v_res_1118_ = lp_batteries_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3___lam__0(v___y_13694__boxed_1116_, v_suppressElabErrors_boxed_1117_, v_x_1115_);
lean_dec(v_x_1115_);
v_r_1119_ = lean_box(v_res_1118_);
return v_r_1119_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3(lean_object* v_ref_1121_, lean_object* v_msgData_1122_, uint8_t v_severity_1123_, uint8_t v_isSilent_1124_, lean_object* v___y_1125_, lean_object* v___y_1126_){
_start:
{
lean_object* v___y_1129_; lean_object* v___y_1130_; lean_object* v___y_1131_; lean_object* v___y_1132_; uint8_t v___y_1133_; lean_object* v___y_1134_; uint8_t v___y_1135_; lean_object* v___y_1136_; uint8_t v___y_1193_; lean_object* v___y_1194_; uint8_t v___y_1195_; uint8_t v___y_1196_; lean_object* v___y_1197_; uint8_t v___y_1221_; lean_object* v___y_1222_; uint8_t v___y_1223_; uint8_t v___y_1224_; lean_object* v___y_1225_; uint8_t v___y_1229_; uint8_t v___y_1230_; uint8_t v___y_1231_; uint8_t v___x_1246_; uint8_t v___y_1248_; uint8_t v___y_1249_; uint8_t v___y_1250_; uint8_t v___y_1252_; uint8_t v___x_1264_; 
v___x_1246_ = 2;
v___x_1264_ = l_Lean_instBEqMessageSeverity_beq(v_severity_1123_, v___x_1246_);
if (v___x_1264_ == 0)
{
v___y_1252_ = v___x_1264_;
goto v___jp_1251_;
}
else
{
uint8_t v___x_1265_; 
lean_inc_ref(v_msgData_1122_);
v___x_1265_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_1122_);
v___y_1252_ = v___x_1265_;
goto v___jp_1251_;
}
v___jp_1128_:
{
lean_object* v___x_1137_; 
v___x_1137_ = l_Lean_Elab_Command_getScope___redArg(v___y_1136_);
if (lean_obj_tag(v___x_1137_) == 0)
{
lean_object* v_a_1138_; lean_object* v___x_1139_; 
v_a_1138_ = lean_ctor_get(v___x_1137_, 0);
lean_inc(v_a_1138_);
lean_dec_ref_known(v___x_1137_, 1);
v___x_1139_ = l_Lean_Elab_Command_getScope___redArg(v___y_1136_);
if (lean_obj_tag(v___x_1139_) == 0)
{
lean_object* v_a_1140_; lean_object* v___x_1142_; uint8_t v_isShared_1143_; uint8_t v_isSharedCheck_1175_; 
v_a_1140_ = lean_ctor_get(v___x_1139_, 0);
v_isSharedCheck_1175_ = !lean_is_exclusive(v___x_1139_);
if (v_isSharedCheck_1175_ == 0)
{
v___x_1142_ = v___x_1139_;
v_isShared_1143_ = v_isSharedCheck_1175_;
goto v_resetjp_1141_;
}
else
{
lean_inc(v_a_1140_);
lean_dec(v___x_1139_);
v___x_1142_ = lean_box(0);
v_isShared_1143_ = v_isSharedCheck_1175_;
goto v_resetjp_1141_;
}
v_resetjp_1141_:
{
lean_object* v___x_1144_; lean_object* v_currNamespace_1145_; lean_object* v_openDecls_1146_; lean_object* v_env_1147_; lean_object* v_messages_1148_; lean_object* v_scopes_1149_; lean_object* v_usedQuotCtxts_1150_; lean_object* v_nextMacroScope_1151_; lean_object* v_maxRecDepth_1152_; lean_object* v_ngen_1153_; lean_object* v_auxDeclNGen_1154_; lean_object* v_infoState_1155_; lean_object* v_traceState_1156_; lean_object* v_snapshotTasks_1157_; lean_object* v_prevLinterStates_1158_; lean_object* v___x_1160_; uint8_t v_isShared_1161_; uint8_t v_isSharedCheck_1174_; 
v___x_1144_ = lean_st_ref_take(v___y_1136_);
v_currNamespace_1145_ = lean_ctor_get(v_a_1138_, 2);
lean_inc(v_currNamespace_1145_);
lean_dec(v_a_1138_);
v_openDecls_1146_ = lean_ctor_get(v_a_1140_, 3);
lean_inc(v_openDecls_1146_);
lean_dec(v_a_1140_);
v_env_1147_ = lean_ctor_get(v___x_1144_, 0);
v_messages_1148_ = lean_ctor_get(v___x_1144_, 1);
v_scopes_1149_ = lean_ctor_get(v___x_1144_, 2);
v_usedQuotCtxts_1150_ = lean_ctor_get(v___x_1144_, 3);
v_nextMacroScope_1151_ = lean_ctor_get(v___x_1144_, 4);
v_maxRecDepth_1152_ = lean_ctor_get(v___x_1144_, 5);
v_ngen_1153_ = lean_ctor_get(v___x_1144_, 6);
v_auxDeclNGen_1154_ = lean_ctor_get(v___x_1144_, 7);
v_infoState_1155_ = lean_ctor_get(v___x_1144_, 8);
v_traceState_1156_ = lean_ctor_get(v___x_1144_, 9);
v_snapshotTasks_1157_ = lean_ctor_get(v___x_1144_, 10);
v_prevLinterStates_1158_ = lean_ctor_get(v___x_1144_, 11);
v_isSharedCheck_1174_ = !lean_is_exclusive(v___x_1144_);
if (v_isSharedCheck_1174_ == 0)
{
v___x_1160_ = v___x_1144_;
v_isShared_1161_ = v_isSharedCheck_1174_;
goto v_resetjp_1159_;
}
else
{
lean_inc(v_prevLinterStates_1158_);
lean_inc(v_snapshotTasks_1157_);
lean_inc(v_traceState_1156_);
lean_inc(v_infoState_1155_);
lean_inc(v_auxDeclNGen_1154_);
lean_inc(v_ngen_1153_);
lean_inc(v_maxRecDepth_1152_);
lean_inc(v_nextMacroScope_1151_);
lean_inc(v_usedQuotCtxts_1150_);
lean_inc(v_scopes_1149_);
lean_inc(v_messages_1148_);
lean_inc(v_env_1147_);
lean_dec(v___x_1144_);
v___x_1160_ = lean_box(0);
v_isShared_1161_ = v_isSharedCheck_1174_;
goto v_resetjp_1159_;
}
v_resetjp_1159_:
{
lean_object* v___x_1162_; lean_object* v___x_1163_; lean_object* v___x_1164_; lean_object* v___x_1165_; lean_object* v___x_1167_; 
v___x_1162_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1162_, 0, v_currNamespace_1145_);
lean_ctor_set(v___x_1162_, 1, v_openDecls_1146_);
v___x_1163_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1163_, 0, v___x_1162_);
lean_ctor_set(v___x_1163_, 1, v___y_1129_);
lean_inc_ref(v___y_1130_);
lean_inc_ref(v___y_1132_);
v___x_1164_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_1164_, 0, v___y_1132_);
lean_ctor_set(v___x_1164_, 1, v___y_1131_);
lean_ctor_set(v___x_1164_, 2, v___y_1134_);
lean_ctor_set(v___x_1164_, 3, v___y_1130_);
lean_ctor_set(v___x_1164_, 4, v___x_1163_);
lean_ctor_set_uint8(v___x_1164_, sizeof(void*)*5, v___y_1135_);
lean_ctor_set_uint8(v___x_1164_, sizeof(void*)*5 + 1, v___y_1133_);
lean_ctor_set_uint8(v___x_1164_, sizeof(void*)*5 + 2, v_isSilent_1124_);
v___x_1165_ = l_Lean_MessageLog_add(v___x_1164_, v_messages_1148_);
if (v_isShared_1161_ == 0)
{
lean_ctor_set(v___x_1160_, 1, v___x_1165_);
v___x_1167_ = v___x_1160_;
goto v_reusejp_1166_;
}
else
{
lean_object* v_reuseFailAlloc_1173_; 
v_reuseFailAlloc_1173_ = lean_alloc_ctor(0, 12, 0);
lean_ctor_set(v_reuseFailAlloc_1173_, 0, v_env_1147_);
lean_ctor_set(v_reuseFailAlloc_1173_, 1, v___x_1165_);
lean_ctor_set(v_reuseFailAlloc_1173_, 2, v_scopes_1149_);
lean_ctor_set(v_reuseFailAlloc_1173_, 3, v_usedQuotCtxts_1150_);
lean_ctor_set(v_reuseFailAlloc_1173_, 4, v_nextMacroScope_1151_);
lean_ctor_set(v_reuseFailAlloc_1173_, 5, v_maxRecDepth_1152_);
lean_ctor_set(v_reuseFailAlloc_1173_, 6, v_ngen_1153_);
lean_ctor_set(v_reuseFailAlloc_1173_, 7, v_auxDeclNGen_1154_);
lean_ctor_set(v_reuseFailAlloc_1173_, 8, v_infoState_1155_);
lean_ctor_set(v_reuseFailAlloc_1173_, 9, v_traceState_1156_);
lean_ctor_set(v_reuseFailAlloc_1173_, 10, v_snapshotTasks_1157_);
lean_ctor_set(v_reuseFailAlloc_1173_, 11, v_prevLinterStates_1158_);
v___x_1167_ = v_reuseFailAlloc_1173_;
goto v_reusejp_1166_;
}
v_reusejp_1166_:
{
lean_object* v___x_1168_; lean_object* v___x_1169_; lean_object* v___x_1171_; 
v___x_1168_ = lean_st_ref_set(v___y_1136_, v___x_1167_);
v___x_1169_ = lean_box(0);
if (v_isShared_1143_ == 0)
{
lean_ctor_set(v___x_1142_, 0, v___x_1169_);
v___x_1171_ = v___x_1142_;
goto v_reusejp_1170_;
}
else
{
lean_object* v_reuseFailAlloc_1172_; 
v_reuseFailAlloc_1172_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1172_, 0, v___x_1169_);
v___x_1171_ = v_reuseFailAlloc_1172_;
goto v_reusejp_1170_;
}
v_reusejp_1170_:
{
return v___x_1171_;
}
}
}
}
}
else
{
lean_object* v_a_1176_; lean_object* v___x_1178_; uint8_t v_isShared_1179_; uint8_t v_isSharedCheck_1183_; 
lean_dec(v_a_1138_);
lean_dec(v___y_1134_);
lean_dec_ref(v___y_1131_);
lean_dec_ref(v___y_1129_);
v_a_1176_ = lean_ctor_get(v___x_1139_, 0);
v_isSharedCheck_1183_ = !lean_is_exclusive(v___x_1139_);
if (v_isSharedCheck_1183_ == 0)
{
v___x_1178_ = v___x_1139_;
v_isShared_1179_ = v_isSharedCheck_1183_;
goto v_resetjp_1177_;
}
else
{
lean_inc(v_a_1176_);
lean_dec(v___x_1139_);
v___x_1178_ = lean_box(0);
v_isShared_1179_ = v_isSharedCheck_1183_;
goto v_resetjp_1177_;
}
v_resetjp_1177_:
{
lean_object* v___x_1181_; 
if (v_isShared_1179_ == 0)
{
v___x_1181_ = v___x_1178_;
goto v_reusejp_1180_;
}
else
{
lean_object* v_reuseFailAlloc_1182_; 
v_reuseFailAlloc_1182_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1182_, 0, v_a_1176_);
v___x_1181_ = v_reuseFailAlloc_1182_;
goto v_reusejp_1180_;
}
v_reusejp_1180_:
{
return v___x_1181_;
}
}
}
}
else
{
lean_object* v_a_1184_; lean_object* v___x_1186_; uint8_t v_isShared_1187_; uint8_t v_isSharedCheck_1191_; 
lean_dec(v___y_1134_);
lean_dec_ref(v___y_1131_);
lean_dec_ref(v___y_1129_);
v_a_1184_ = lean_ctor_get(v___x_1137_, 0);
v_isSharedCheck_1191_ = !lean_is_exclusive(v___x_1137_);
if (v_isSharedCheck_1191_ == 0)
{
v___x_1186_ = v___x_1137_;
v_isShared_1187_ = v_isSharedCheck_1191_;
goto v_resetjp_1185_;
}
else
{
lean_inc(v_a_1184_);
lean_dec(v___x_1137_);
v___x_1186_ = lean_box(0);
v_isShared_1187_ = v_isSharedCheck_1191_;
goto v_resetjp_1185_;
}
v_resetjp_1185_:
{
lean_object* v___x_1189_; 
if (v_isShared_1187_ == 0)
{
v___x_1189_ = v___x_1186_;
goto v_reusejp_1188_;
}
else
{
lean_object* v_reuseFailAlloc_1190_; 
v_reuseFailAlloc_1190_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1190_, 0, v_a_1184_);
v___x_1189_ = v_reuseFailAlloc_1190_;
goto v_reusejp_1188_;
}
v_reusejp_1188_:
{
return v___x_1189_;
}
}
}
}
v___jp_1192_:
{
lean_object* v_fileName_1198_; lean_object* v_fileMap_1199_; uint8_t v_suppressElabErrors_1200_; lean_object* v___x_1201_; lean_object* v___x_1202_; lean_object* v_a_1203_; lean_object* v___x_1205_; uint8_t v_isShared_1206_; uint8_t v_isSharedCheck_1219_; 
v_fileName_1198_ = lean_ctor_get(v___y_1125_, 0);
v_fileMap_1199_ = lean_ctor_get(v___y_1125_, 1);
v_suppressElabErrors_1200_ = lean_ctor_get_uint8(v___y_1125_, sizeof(void*)*10);
v___x_1201_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_1122_);
v___x_1202_ = lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3_spec__13___redArg(v___x_1201_, v___y_1126_);
v_a_1203_ = lean_ctor_get(v___x_1202_, 0);
v_isSharedCheck_1219_ = !lean_is_exclusive(v___x_1202_);
if (v_isSharedCheck_1219_ == 0)
{
v___x_1205_ = v___x_1202_;
v_isShared_1206_ = v_isSharedCheck_1219_;
goto v_resetjp_1204_;
}
else
{
lean_inc(v_a_1203_);
lean_dec(v___x_1202_);
v___x_1205_ = lean_box(0);
v_isShared_1206_ = v_isSharedCheck_1219_;
goto v_resetjp_1204_;
}
v_resetjp_1204_:
{
lean_object* v___x_1207_; lean_object* v___x_1208_; lean_object* v___x_1209_; lean_object* v___x_1210_; 
lean_inc_ref_n(v_fileMap_1199_, 2);
v___x_1207_ = l_Lean_FileMap_toPosition(v_fileMap_1199_, v___y_1194_);
lean_dec(v___y_1194_);
v___x_1208_ = l_Lean_FileMap_toPosition(v_fileMap_1199_, v___y_1197_);
lean_dec(v___y_1197_);
v___x_1209_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1209_, 0, v___x_1208_);
v___x_1210_ = ((lean_object*)(lp_batteries_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3___closed__0));
if (v_suppressElabErrors_1200_ == 0)
{
lean_del_object(v___x_1205_);
v___y_1129_ = v_a_1203_;
v___y_1130_ = v___x_1210_;
v___y_1131_ = v___x_1207_;
v___y_1132_ = v_fileName_1198_;
v___y_1133_ = v___y_1195_;
v___y_1134_ = v___x_1209_;
v___y_1135_ = v___y_1196_;
v___y_1136_ = v___y_1126_;
goto v___jp_1128_;
}
else
{
lean_object* v___x_1211_; lean_object* v___x_1212_; lean_object* v___f_1213_; uint8_t v___x_1214_; 
v___x_1211_ = lean_box(v___y_1193_);
v___x_1212_ = lean_box(v_suppressElabErrors_1200_);
v___f_1213_ = lean_alloc_closure((void*)(lp_batteries_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3___lam__0___boxed), 3, 2);
lean_closure_set(v___f_1213_, 0, v___x_1211_);
lean_closure_set(v___f_1213_, 1, v___x_1212_);
lean_inc(v_a_1203_);
v___x_1214_ = l_Lean_MessageData_hasTag(v___f_1213_, v_a_1203_);
if (v___x_1214_ == 0)
{
lean_object* v___x_1215_; lean_object* v___x_1217_; 
lean_dec_ref_known(v___x_1209_, 1);
lean_dec_ref(v___x_1207_);
lean_dec(v_a_1203_);
v___x_1215_ = lean_box(0);
if (v_isShared_1206_ == 0)
{
lean_ctor_set(v___x_1205_, 0, v___x_1215_);
v___x_1217_ = v___x_1205_;
goto v_reusejp_1216_;
}
else
{
lean_object* v_reuseFailAlloc_1218_; 
v_reuseFailAlloc_1218_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1218_, 0, v___x_1215_);
v___x_1217_ = v_reuseFailAlloc_1218_;
goto v_reusejp_1216_;
}
v_reusejp_1216_:
{
return v___x_1217_;
}
}
else
{
lean_del_object(v___x_1205_);
v___y_1129_ = v_a_1203_;
v___y_1130_ = v___x_1210_;
v___y_1131_ = v___x_1207_;
v___y_1132_ = v_fileName_1198_;
v___y_1133_ = v___y_1195_;
v___y_1134_ = v___x_1209_;
v___y_1135_ = v___y_1196_;
v___y_1136_ = v___y_1126_;
goto v___jp_1128_;
}
}
}
}
v___jp_1220_:
{
lean_object* v___x_1226_; 
v___x_1226_ = l_Lean_Syntax_getTailPos_x3f(v___y_1222_, v___y_1224_);
lean_dec(v___y_1222_);
if (lean_obj_tag(v___x_1226_) == 0)
{
lean_inc(v___y_1225_);
v___y_1193_ = v___y_1221_;
v___y_1194_ = v___y_1225_;
v___y_1195_ = v___y_1223_;
v___y_1196_ = v___y_1224_;
v___y_1197_ = v___y_1225_;
goto v___jp_1192_;
}
else
{
lean_object* v_val_1227_; 
v_val_1227_ = lean_ctor_get(v___x_1226_, 0);
lean_inc(v_val_1227_);
lean_dec_ref_known(v___x_1226_, 1);
v___y_1193_ = v___y_1221_;
v___y_1194_ = v___y_1225_;
v___y_1195_ = v___y_1223_;
v___y_1196_ = v___y_1224_;
v___y_1197_ = v_val_1227_;
goto v___jp_1192_;
}
}
v___jp_1228_:
{
lean_object* v___x_1232_; 
v___x_1232_ = l_Lean_Elab_Command_getRef___redArg(v___y_1125_);
if (lean_obj_tag(v___x_1232_) == 0)
{
lean_object* v_a_1233_; lean_object* v_ref_1234_; lean_object* v___x_1235_; 
v_a_1233_ = lean_ctor_get(v___x_1232_, 0);
lean_inc(v_a_1233_);
lean_dec_ref_known(v___x_1232_, 1);
v_ref_1234_ = l_Lean_replaceRef(v_ref_1121_, v_a_1233_);
lean_dec(v_a_1233_);
v___x_1235_ = l_Lean_Syntax_getPos_x3f(v_ref_1234_, v___y_1230_);
if (lean_obj_tag(v___x_1235_) == 0)
{
lean_object* v___x_1236_; 
v___x_1236_ = lean_unsigned_to_nat(0u);
v___y_1221_ = v___y_1229_;
v___y_1222_ = v_ref_1234_;
v___y_1223_ = v___y_1231_;
v___y_1224_ = v___y_1230_;
v___y_1225_ = v___x_1236_;
goto v___jp_1220_;
}
else
{
lean_object* v_val_1237_; 
v_val_1237_ = lean_ctor_get(v___x_1235_, 0);
lean_inc(v_val_1237_);
lean_dec_ref_known(v___x_1235_, 1);
v___y_1221_ = v___y_1229_;
v___y_1222_ = v_ref_1234_;
v___y_1223_ = v___y_1231_;
v___y_1224_ = v___y_1230_;
v___y_1225_ = v_val_1237_;
goto v___jp_1220_;
}
}
else
{
lean_object* v_a_1238_; lean_object* v___x_1240_; uint8_t v_isShared_1241_; uint8_t v_isSharedCheck_1245_; 
lean_dec_ref(v_msgData_1122_);
v_a_1238_ = lean_ctor_get(v___x_1232_, 0);
v_isSharedCheck_1245_ = !lean_is_exclusive(v___x_1232_);
if (v_isSharedCheck_1245_ == 0)
{
v___x_1240_ = v___x_1232_;
v_isShared_1241_ = v_isSharedCheck_1245_;
goto v_resetjp_1239_;
}
else
{
lean_inc(v_a_1238_);
lean_dec(v___x_1232_);
v___x_1240_ = lean_box(0);
v_isShared_1241_ = v_isSharedCheck_1245_;
goto v_resetjp_1239_;
}
v_resetjp_1239_:
{
lean_object* v___x_1243_; 
if (v_isShared_1241_ == 0)
{
v___x_1243_ = v___x_1240_;
goto v_reusejp_1242_;
}
else
{
lean_object* v_reuseFailAlloc_1244_; 
v_reuseFailAlloc_1244_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1244_, 0, v_a_1238_);
v___x_1243_ = v_reuseFailAlloc_1244_;
goto v_reusejp_1242_;
}
v_reusejp_1242_:
{
return v___x_1243_;
}
}
}
}
v___jp_1247_:
{
if (v___y_1250_ == 0)
{
v___y_1229_ = v___y_1248_;
v___y_1230_ = v___y_1249_;
v___y_1231_ = v_severity_1123_;
goto v___jp_1228_;
}
else
{
v___y_1229_ = v___y_1248_;
v___y_1230_ = v___y_1249_;
v___y_1231_ = v___x_1246_;
goto v___jp_1228_;
}
}
v___jp_1251_:
{
if (v___y_1252_ == 0)
{
lean_object* v___x_1253_; lean_object* v_scopes_1254_; lean_object* v___x_1255_; lean_object* v___x_1256_; lean_object* v_opts_1257_; uint8_t v___x_1258_; uint8_t v___x_1259_; 
v___x_1253_ = lean_st_ref_get(v___y_1126_);
v_scopes_1254_ = lean_ctor_get(v___x_1253_, 2);
lean_inc(v_scopes_1254_);
lean_dec(v___x_1253_);
v___x_1255_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_1256_ = l_List_head_x21___redArg(v___x_1255_, v_scopes_1254_);
lean_dec(v_scopes_1254_);
v_opts_1257_ = lean_ctor_get(v___x_1256_, 1);
lean_inc_ref(v_opts_1257_);
lean_dec(v___x_1256_);
v___x_1258_ = 1;
v___x_1259_ = l_Lean_instBEqMessageSeverity_beq(v_severity_1123_, v___x_1258_);
if (v___x_1259_ == 0)
{
lean_dec_ref(v_opts_1257_);
v___y_1248_ = v___y_1252_;
v___y_1249_ = v___y_1252_;
v___y_1250_ = v___x_1259_;
goto v___jp_1247_;
}
else
{
lean_object* v___x_1260_; uint8_t v___x_1261_; 
v___x_1260_ = l_Lean_warningAsError;
v___x_1261_ = lp_batteries_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3_spec__14(v_opts_1257_, v___x_1260_);
lean_dec_ref(v_opts_1257_);
v___y_1248_ = v___y_1252_;
v___y_1249_ = v___y_1252_;
v___y_1250_ = v___x_1261_;
goto v___jp_1247_;
}
}
else
{
lean_object* v___x_1262_; lean_object* v___x_1263_; 
lean_dec_ref(v_msgData_1122_);
v___x_1262_ = lean_box(0);
v___x_1263_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1263_, 0, v___x_1262_);
return v___x_1263_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3___boxed(lean_object* v_ref_1266_, lean_object* v_msgData_1267_, lean_object* v_severity_1268_, lean_object* v_isSilent_1269_, lean_object* v___y_1270_, lean_object* v___y_1271_, lean_object* v___y_1272_){
_start:
{
uint8_t v_severity_boxed_1273_; uint8_t v_isSilent_boxed_1274_; lean_object* v_res_1275_; 
v_severity_boxed_1273_ = lean_unbox(v_severity_1268_);
v_isSilent_boxed_1274_ = lean_unbox(v_isSilent_1269_);
v_res_1275_ = lp_batteries_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3(v_ref_1266_, v_msgData_1267_, v_severity_boxed_1273_, v_isSilent_boxed_1274_, v___y_1270_, v___y_1271_);
lean_dec(v___y_1271_);
lean_dec_ref(v___y_1270_);
lean_dec(v_ref_1266_);
return v_res_1275_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1(lean_object* v_ref_1276_, lean_object* v_msgData_1277_, lean_object* v___y_1278_, lean_object* v___y_1279_){
_start:
{
uint8_t v___x_1281_; uint8_t v___x_1282_; lean_object* v___x_1283_; 
v___x_1281_ = 1;
v___x_1282_ = 0;
v___x_1283_ = lp_batteries_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3(v_ref_1276_, v_msgData_1277_, v___x_1281_, v___x_1282_, v___y_1278_, v___y_1279_);
return v___x_1283_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1___boxed(lean_object* v_ref_1284_, lean_object* v_msgData_1285_, lean_object* v___y_1286_, lean_object* v___y_1287_, lean_object* v___y_1288_){
_start:
{
lean_object* v_res_1289_; 
v_res_1289_ = lp_batteries_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1(v_ref_1284_, v_msgData_1285_, v___y_1286_, v___y_1287_);
lean_dec(v___y_1287_);
lean_dec_ref(v___y_1286_);
lean_dec(v_ref_1284_);
return v_res_1289_;
}
}
static lean_object* _init_lp_batteries_Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1___closed__1(void){
_start:
{
lean_object* v___x_1291_; lean_object* v___x_1292_; 
v___x_1291_ = ((lean_object*)(lp_batteries_Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1___closed__0));
v___x_1292_ = l_Lean_stringToMessageData(v___x_1291_);
return v___x_1292_;
}
}
static lean_object* _init_lp_batteries_Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1___closed__3(void){
_start:
{
lean_object* v___x_1294_; lean_object* v___x_1295_; 
v___x_1294_ = ((lean_object*)(lp_batteries_Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1___closed__2));
v___x_1295_ = l_Lean_stringToMessageData(v___x_1294_);
return v___x_1295_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1(lean_object* v_linterOption_1296_, lean_object* v_stx_1297_, lean_object* v_msg_1298_, lean_object* v___y_1299_, lean_object* v___y_1300_){
_start:
{
lean_object* v_name_1302_; lean_object* v___x_1304_; uint8_t v_isShared_1305_; uint8_t v_isSharedCheck_1320_; 
v_name_1302_ = lean_ctor_get(v_linterOption_1296_, 0);
v_isSharedCheck_1320_ = !lean_is_exclusive(v_linterOption_1296_);
if (v_isSharedCheck_1320_ == 0)
{
lean_object* v_unused_1321_; 
v_unused_1321_ = lean_ctor_get(v_linterOption_1296_, 1);
lean_dec(v_unused_1321_);
v___x_1304_ = v_linterOption_1296_;
v_isShared_1305_ = v_isSharedCheck_1320_;
goto v_resetjp_1303_;
}
else
{
lean_inc(v_name_1302_);
lean_dec(v_linterOption_1296_);
v___x_1304_ = lean_box(0);
v_isShared_1305_ = v_isSharedCheck_1320_;
goto v_resetjp_1303_;
}
v_resetjp_1303_:
{
lean_object* v___x_1306_; lean_object* v___x_1307_; lean_object* v___x_1309_; 
v___x_1306_ = lean_obj_once(&lp_batteries_Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1___closed__1, &lp_batteries_Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1___closed__1_once, _init_lp_batteries_Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1___closed__1);
lean_inc(v_name_1302_);
v___x_1307_ = l_Lean_MessageData_ofName(v_name_1302_);
if (v_isShared_1305_ == 0)
{
lean_ctor_set_tag(v___x_1304_, 7);
lean_ctor_set(v___x_1304_, 1, v___x_1307_);
lean_ctor_set(v___x_1304_, 0, v___x_1306_);
v___x_1309_ = v___x_1304_;
goto v_reusejp_1308_;
}
else
{
lean_object* v_reuseFailAlloc_1319_; 
v_reuseFailAlloc_1319_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1319_, 0, v___x_1306_);
lean_ctor_set(v_reuseFailAlloc_1319_, 1, v___x_1307_);
v___x_1309_ = v_reuseFailAlloc_1319_;
goto v_reusejp_1308_;
}
v_reusejp_1308_:
{
lean_object* v___x_1310_; lean_object* v___x_1311_; lean_object* v_disable_1312_; lean_object* v___x_1313_; lean_object* v___x_1314_; lean_object* v___x_1315_; lean_object* v___x_1316_; lean_object* v___x_1317_; lean_object* v___x_1318_; 
v___x_1310_ = lean_obj_once(&lp_batteries_Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1___closed__3, &lp_batteries_Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1___closed__3_once, _init_lp_batteries_Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1___closed__3);
v___x_1311_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1311_, 0, v___x_1309_);
lean_ctor_set(v___x_1311_, 1, v___x_1310_);
v_disable_1312_ = l_Lean_MessageData_note(v___x_1311_);
v___x_1313_ = l_Lean_Linter_linterMessageTag;
v___x_1314_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1314_, 0, v_msg_1298_);
lean_ctor_set(v___x_1314_, 1, v_disable_1312_);
v___x_1315_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_1315_, 0, v___x_1313_);
lean_ctor_set(v___x_1315_, 1, v___x_1314_);
v___x_1316_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_1316_, 0, v_name_1302_);
lean_ctor_set(v___x_1316_, 1, v___x_1315_);
lean_inc(v_stx_1297_);
v___x_1317_ = lean_alloc_ctor(11, 2, 0);
lean_ctor_set(v___x_1317_, 0, v_stx_1297_);
lean_ctor_set(v___x_1317_, 1, v___x_1316_);
v___x_1318_ = lp_batteries_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1(v_stx_1297_, v___x_1317_, v___y_1299_, v___y_1300_);
lean_dec(v_stx_1297_);
return v___x_1318_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1___boxed(lean_object* v_linterOption_1322_, lean_object* v_stx_1323_, lean_object* v_msg_1324_, lean_object* v___y_1325_, lean_object* v___y_1326_, lean_object* v___y_1327_){
_start:
{
lean_object* v_res_1328_; 
v_res_1328_ = lp_batteries_Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1(v_linterOption_1322_, v_stx_1323_, v_msg_1324_, v___y_1325_, v___y_1326_);
lean_dec(v___y_1326_);
lean_dec_ref(v___y_1325_);
return v_res_1328_;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__5___closed__2(void){
_start:
{
lean_object* v___x_1332_; lean_object* v___x_1333_; 
v___x_1332_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__5___closed__1));
v___x_1333_ = l_Lean_MessageData_ofFormat(v___x_1332_);
return v___x_1333_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__5(lean_object* v_as_1351_, size_t v_sz_1352_, size_t v_i_1353_, lean_object* v_b_1354_, lean_object* v___y_1355_, lean_object* v___y_1356_){
_start:
{
lean_object* v_a_1359_; uint8_t v___x_1363_; 
v___x_1363_ = lean_usize_dec_lt(v_i_1353_, v_sz_1352_);
if (v___x_1363_ == 0)
{
lean_object* v___x_1364_; 
v___x_1364_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1364_, 0, v_b_1354_);
return v___x_1364_;
}
else
{
lean_object* v_a_1365_; lean_object* v_fst_1366_; lean_object* v_snd_1367_; uint8_t v___y_1369_; lean_object* v___x_1381_; lean_object* v___x_1382_; lean_object* v___x_1383_; uint8_t v___x_1384_; 
v_a_1365_ = lean_array_uget_borrowed(v_as_1351_, v_i_1353_);
v_fst_1366_ = lean_ctor_get(v_a_1365_, 0);
v_snd_1367_ = lean_ctor_get(v_a_1365_, 1);
v___x_1381_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__5___closed__3));
lean_inc(v_snd_1367_);
v___x_1382_ = l_Lean_Syntax_getKind(v_snd_1367_);
v___x_1383_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__5___closed__9));
v___x_1384_ = l_List_elem___redArg(v___x_1381_, v___x_1382_, v___x_1383_);
if (v___x_1384_ == 0)
{
lean_object* v_start_1385_; lean_object* v_stop_1386_; lean_object* v_start_1387_; lean_object* v_stop_1388_; uint8_t v___x_1389_; 
v_start_1385_ = lean_ctor_get(v_b_1354_, 0);
v_stop_1386_ = lean_ctor_get(v_b_1354_, 1);
v_start_1387_ = lean_ctor_get(v_fst_1366_, 0);
v_stop_1388_ = lean_ctor_get(v_fst_1366_, 1);
v___x_1389_ = lean_nat_dec_le(v_start_1385_, v_start_1387_);
if (v___x_1389_ == 0)
{
v___y_1369_ = v___x_1389_;
goto v___jp_1368_;
}
else
{
uint8_t v___x_1390_; 
v___x_1390_ = lean_nat_dec_le(v_stop_1388_, v_stop_1386_);
v___y_1369_ = v___x_1390_;
goto v___jp_1368_;
}
}
else
{
v_a_1359_ = v_b_1354_;
goto v___jp_1358_;
}
v___jp_1368_:
{
if (v___y_1369_ == 0)
{
lean_object* v___x_1370_; lean_object* v___x_1371_; lean_object* v___x_1372_; 
lean_dec_ref(v_b_1354_);
v___x_1370_ = lp_batteries_Batteries_Linter_linter_unreachableTactic;
v___x_1371_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__5___closed__2, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__5___closed__2_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__5___closed__2);
lean_inc(v_snd_1367_);
v___x_1372_ = lp_batteries_Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1(v___x_1370_, v_snd_1367_, v___x_1371_, v___y_1355_, v___y_1356_);
if (lean_obj_tag(v___x_1372_) == 0)
{
lean_dec_ref_known(v___x_1372_, 1);
lean_inc(v_fst_1366_);
v_a_1359_ = v_fst_1366_;
goto v___jp_1358_;
}
else
{
lean_object* v_a_1373_; lean_object* v___x_1375_; uint8_t v_isShared_1376_; uint8_t v_isSharedCheck_1380_; 
v_a_1373_ = lean_ctor_get(v___x_1372_, 0);
v_isSharedCheck_1380_ = !lean_is_exclusive(v___x_1372_);
if (v_isSharedCheck_1380_ == 0)
{
v___x_1375_ = v___x_1372_;
v_isShared_1376_ = v_isSharedCheck_1380_;
goto v_resetjp_1374_;
}
else
{
lean_inc(v_a_1373_);
lean_dec(v___x_1372_);
v___x_1375_ = lean_box(0);
v_isShared_1376_ = v_isSharedCheck_1380_;
goto v_resetjp_1374_;
}
v_resetjp_1374_:
{
lean_object* v___x_1378_; 
if (v_isShared_1376_ == 0)
{
v___x_1378_ = v___x_1375_;
goto v_reusejp_1377_;
}
else
{
lean_object* v_reuseFailAlloc_1379_; 
v_reuseFailAlloc_1379_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1379_, 0, v_a_1373_);
v___x_1378_ = v_reuseFailAlloc_1379_;
goto v_reusejp_1377_;
}
v_reusejp_1377_:
{
return v___x_1378_;
}
}
}
}
else
{
v_a_1359_ = v_b_1354_;
goto v___jp_1358_;
}
}
}
v___jp_1358_:
{
size_t v___x_1360_; size_t v___x_1361_; 
v___x_1360_ = ((size_t)1ULL);
v___x_1361_ = lean_usize_add(v_i_1353_, v___x_1360_);
v_i_1353_ = v___x_1361_;
v_b_1354_ = v_a_1359_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__5___boxed(lean_object* v_as_1391_, lean_object* v_sz_1392_, lean_object* v_i_1393_, lean_object* v_b_1394_, lean_object* v___y_1395_, lean_object* v___y_1396_, lean_object* v___y_1397_){
_start:
{
size_t v_sz_boxed_1398_; size_t v_i_boxed_1399_; lean_object* v_res_1400_; 
v_sz_boxed_1398_ = lean_unbox_usize(v_sz_1392_);
lean_dec(v_sz_1392_);
v_i_boxed_1399_ = lean_unbox_usize(v_i_1393_);
lean_dec(v_i_1393_);
v_res_1400_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__5(v_as_1391_, v_sz_boxed_1398_, v_i_boxed_1399_, v_b_1394_, v___y_1395_, v___y_1396_);
lean_dec(v___y_1396_);
lean_dec_ref(v___y_1395_);
lean_dec_ref(v_as_1391_);
return v_res_1400_;
}
}
LEAN_EXPORT uint8_t lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__6___redArg___lam__1(lean_object* v___f_1403_, uint8_t v___x_1404_, lean_object* v_x1_1405_, lean_object* v_x2_1406_){
_start:
{
lean_object* v_fst_1407_; lean_object* v_fst_1408_; lean_object* v___f_1409_; lean_object* v___f_1410_; lean_object* v___x_1411_; lean_object* v___x_1412_; lean_object* v___x_13157__overap_1413_; lean_object* v___x_1414_; uint8_t v___x_1415_; 
v_fst_1407_ = lean_ctor_get(v_x1_1405_, 0);
lean_inc(v_fst_1407_);
lean_dec_ref(v_x1_1405_);
v_fst_1408_ = lean_ctor_get(v_x2_1406_, 0);
lean_inc(v_fst_1408_);
lean_dec_ref(v_x2_1406_);
v___f_1409_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__6___redArg___lam__1___closed__0));
v___f_1410_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__6___redArg___lam__1___closed__1));
lean_inc_ref(v___f_1403_);
v___x_1411_ = lean_apply_1(v___f_1403_, v_fst_1407_);
v___x_1412_ = lean_apply_1(v___f_1403_, v_fst_1408_);
v___x_13157__overap_1413_ = l_lexOrd___redArg(v___f_1409_, v___f_1410_);
v___x_1414_ = lean_apply_2(v___x_13157__overap_1413_, v___x_1411_, v___x_1412_);
v___x_1415_ = lean_unbox(v___x_1414_);
if (v___x_1415_ == 0)
{
return v___x_1404_;
}
else
{
uint8_t v___x_1416_; 
v___x_1416_ = 0;
return v___x_1416_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__6___redArg___lam__1___boxed(lean_object* v___f_1417_, lean_object* v___x_1418_, lean_object* v_x1_1419_, lean_object* v_x2_1420_){
_start:
{
uint8_t v___x_14179__boxed_1421_; uint8_t v_res_1422_; lean_object* v_r_1423_; 
v___x_14179__boxed_1421_ = lean_unbox(v___x_1418_);
v_res_1422_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__6___redArg___lam__1(v___f_1417_, v___x_14179__boxed_1421_, v_x1_1419_, v_x2_1420_);
v_r_1423_ = lean_box(v_res_1422_);
return v_r_1423_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__6___redArg___lam__0(lean_object* v_r_1424_){
_start:
{
lean_object* v_start_1425_; lean_object* v_stop_1426_; lean_object* v___x_1428_; uint8_t v_isShared_1429_; uint8_t v_isSharedCheck_1435_; 
v_start_1425_ = lean_ctor_get(v_r_1424_, 0);
v_stop_1426_ = lean_ctor_get(v_r_1424_, 1);
v_isSharedCheck_1435_ = !lean_is_exclusive(v_r_1424_);
if (v_isSharedCheck_1435_ == 0)
{
v___x_1428_ = v_r_1424_;
v_isShared_1429_ = v_isSharedCheck_1435_;
goto v_resetjp_1427_;
}
else
{
lean_inc(v_stop_1426_);
lean_inc(v_start_1425_);
lean_dec(v_r_1424_);
v___x_1428_ = lean_box(0);
v_isShared_1429_ = v_isSharedCheck_1435_;
goto v_resetjp_1427_;
}
v_resetjp_1427_:
{
lean_object* v___x_1430_; lean_object* v___x_1431_; lean_object* v___x_1433_; 
v___x_1430_ = lean_nat_to_int(v_stop_1426_);
v___x_1431_ = lean_int_neg(v___x_1430_);
lean_dec(v___x_1430_);
if (v_isShared_1429_ == 0)
{
lean_ctor_set(v___x_1428_, 1, v___x_1431_);
v___x_1433_ = v___x_1428_;
goto v_reusejp_1432_;
}
else
{
lean_object* v_reuseFailAlloc_1434_; 
v_reuseFailAlloc_1434_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1434_, 0, v_start_1425_);
lean_ctor_set(v_reuseFailAlloc_1434_, 1, v___x_1431_);
v___x_1433_ = v_reuseFailAlloc_1434_;
goto v_reusejp_1432_;
}
v_reusejp_1432_:
{
return v___x_1433_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__6_spec__9___redArg(lean_object* v_hi_1436_, lean_object* v_pivot_1437_, lean_object* v_as_1438_, lean_object* v_i_1439_, lean_object* v_k_1440_){
_start:
{
uint8_t v___x_1445_; 
v___x_1445_ = lean_nat_dec_lt(v_k_1440_, v_hi_1436_);
if (v___x_1445_ == 0)
{
lean_object* v___x_1446_; lean_object* v___x_1447_; 
lean_dec(v_k_1440_);
lean_dec_ref(v_pivot_1437_);
v___x_1446_ = lean_array_fswap(v_as_1438_, v_i_1439_, v_hi_1436_);
v___x_1447_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1447_, 0, v_i_1439_);
lean_ctor_set(v___x_1447_, 1, v___x_1446_);
return v___x_1447_;
}
else
{
lean_object* v___x_1448_; lean_object* v_fst_1449_; lean_object* v_fst_1450_; lean_object* v___f_1451_; lean_object* v___f_1452_; lean_object* v___x_1453_; lean_object* v___x_1454_; lean_object* v___x_12895__overap_1455_; lean_object* v___x_1456_; uint8_t v___x_1457_; 
v___x_1448_ = lean_array_fget_borrowed(v_as_1438_, v_k_1440_);
v_fst_1449_ = lean_ctor_get(v___x_1448_, 0);
v_fst_1450_ = lean_ctor_get(v_pivot_1437_, 0);
v___f_1451_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__6___redArg___lam__1___closed__0));
v___f_1452_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__6___redArg___lam__1___closed__1));
lean_inc(v_fst_1449_);
v___x_1453_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__6___redArg___lam__0(v_fst_1449_);
lean_inc(v_fst_1450_);
v___x_1454_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__6___redArg___lam__0(v_fst_1450_);
v___x_12895__overap_1455_ = l_lexOrd___redArg(v___f_1451_, v___f_1452_);
v___x_1456_ = lean_apply_2(v___x_12895__overap_1455_, v___x_1453_, v___x_1454_);
v___x_1457_ = lean_unbox(v___x_1456_);
if (v___x_1457_ == 0)
{
if (v___x_1445_ == 0)
{
goto v___jp_1441_;
}
else
{
lean_object* v___x_1458_; lean_object* v___x_1459_; lean_object* v___x_1460_; lean_object* v___x_1461_; 
v___x_1458_ = lean_array_fswap(v_as_1438_, v_i_1439_, v_k_1440_);
v___x_1459_ = lean_unsigned_to_nat(1u);
v___x_1460_ = lean_nat_add(v_i_1439_, v___x_1459_);
lean_dec(v_i_1439_);
v___x_1461_ = lean_nat_add(v_k_1440_, v___x_1459_);
lean_dec(v_k_1440_);
v_as_1438_ = v___x_1458_;
v_i_1439_ = v___x_1460_;
v_k_1440_ = v___x_1461_;
goto _start;
}
}
else
{
goto v___jp_1441_;
}
}
v___jp_1441_:
{
lean_object* v___x_1442_; lean_object* v___x_1443_; 
v___x_1442_ = lean_unsigned_to_nat(1u);
v___x_1443_ = lean_nat_add(v_k_1440_, v___x_1442_);
lean_dec(v_k_1440_);
v_k_1440_ = v___x_1443_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__6_spec__9___redArg___boxed(lean_object* v_hi_1463_, lean_object* v_pivot_1464_, lean_object* v_as_1465_, lean_object* v_i_1466_, lean_object* v_k_1467_){
_start:
{
lean_object* v_res_1468_; 
v_res_1468_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__6_spec__9___redArg(v_hi_1463_, v_pivot_1464_, v_as_1465_, v_i_1466_, v_k_1467_);
lean_dec(v_hi_1463_);
return v_res_1468_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__6___redArg(lean_object* v_n_1470_, lean_object* v_as_1471_, lean_object* v_lo_1472_, lean_object* v_hi_1473_){
_start:
{
lean_object* v___y_1475_; uint8_t v___x_1485_; 
v___x_1485_ = lean_nat_dec_lt(v_lo_1472_, v_hi_1473_);
if (v___x_1485_ == 0)
{
lean_dec(v_lo_1472_);
return v_as_1471_;
}
else
{
lean_object* v___f_1486_; lean_object* v___x_1487_; lean_object* v___x_1488_; lean_object* v_mid_1489_; lean_object* v___y_1491_; lean_object* v___y_1497_; lean_object* v___x_1502_; lean_object* v___x_1503_; uint8_t v___x_1504_; 
v___f_1486_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__6___redArg___closed__0));
v___x_1487_ = lean_nat_add(v_lo_1472_, v_hi_1473_);
v___x_1488_ = lean_unsigned_to_nat(1u);
v_mid_1489_ = lean_nat_shiftr(v___x_1487_, v___x_1488_);
lean_dec(v___x_1487_);
v___x_1502_ = lean_array_fget_borrowed(v_as_1471_, v_mid_1489_);
v___x_1503_ = lean_array_fget_borrowed(v_as_1471_, v_lo_1472_);
lean_inc(v___x_1503_);
lean_inc(v___x_1502_);
v___x_1504_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__6___redArg___lam__1(v___f_1486_, v___x_1485_, v___x_1502_, v___x_1503_);
if (v___x_1504_ == 0)
{
v___y_1497_ = v_as_1471_;
goto v___jp_1496_;
}
else
{
lean_object* v___x_1505_; 
v___x_1505_ = lean_array_fswap(v_as_1471_, v_lo_1472_, v_mid_1489_);
v___y_1497_ = v___x_1505_;
goto v___jp_1496_;
}
v___jp_1490_:
{
lean_object* v___x_1492_; lean_object* v___x_1493_; uint8_t v___x_1494_; 
v___x_1492_ = lean_array_fget_borrowed(v___y_1491_, v_mid_1489_);
v___x_1493_ = lean_array_fget_borrowed(v___y_1491_, v_hi_1473_);
lean_inc(v___x_1493_);
lean_inc(v___x_1492_);
v___x_1494_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__6___redArg___lam__1(v___f_1486_, v___x_1485_, v___x_1492_, v___x_1493_);
if (v___x_1494_ == 0)
{
lean_dec(v_mid_1489_);
v___y_1475_ = v___y_1491_;
goto v___jp_1474_;
}
else
{
lean_object* v___x_1495_; 
v___x_1495_ = lean_array_fswap(v___y_1491_, v_mid_1489_, v_hi_1473_);
lean_dec(v_mid_1489_);
v___y_1475_ = v___x_1495_;
goto v___jp_1474_;
}
}
v___jp_1496_:
{
lean_object* v___x_1498_; lean_object* v___x_1499_; uint8_t v___x_1500_; 
v___x_1498_ = lean_array_fget_borrowed(v___y_1497_, v_hi_1473_);
v___x_1499_ = lean_array_fget_borrowed(v___y_1497_, v_lo_1472_);
lean_inc(v___x_1499_);
lean_inc(v___x_1498_);
v___x_1500_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__6___redArg___lam__1(v___f_1486_, v___x_1485_, v___x_1498_, v___x_1499_);
if (v___x_1500_ == 0)
{
v___y_1491_ = v___y_1497_;
goto v___jp_1490_;
}
else
{
lean_object* v___x_1501_; 
v___x_1501_ = lean_array_fswap(v___y_1497_, v_lo_1472_, v_hi_1473_);
v___y_1491_ = v___x_1501_;
goto v___jp_1490_;
}
}
}
v___jp_1474_:
{
lean_object* v_pivot_1476_; lean_object* v___x_1477_; lean_object* v_fst_1478_; lean_object* v_snd_1479_; uint8_t v___x_1480_; 
v_pivot_1476_ = lean_array_fget(v___y_1475_, v_hi_1473_);
lean_inc_n(v_lo_1472_, 2);
v___x_1477_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__6_spec__9___redArg(v_hi_1473_, v_pivot_1476_, v___y_1475_, v_lo_1472_, v_lo_1472_);
v_fst_1478_ = lean_ctor_get(v___x_1477_, 0);
lean_inc(v_fst_1478_);
v_snd_1479_ = lean_ctor_get(v___x_1477_, 1);
lean_inc(v_snd_1479_);
lean_dec_ref(v___x_1477_);
v___x_1480_ = lean_nat_dec_le(v_hi_1473_, v_fst_1478_);
if (v___x_1480_ == 0)
{
lean_object* v___x_1481_; lean_object* v___x_1482_; lean_object* v___x_1483_; 
v___x_1481_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__6___redArg(v_n_1470_, v_snd_1479_, v_lo_1472_, v_fst_1478_);
v___x_1482_ = lean_unsigned_to_nat(1u);
v___x_1483_ = lean_nat_add(v_fst_1478_, v___x_1482_);
lean_dec(v_fst_1478_);
v_as_1471_ = v___x_1481_;
v_lo_1472_ = v___x_1483_;
goto _start;
}
else
{
lean_dec(v_fst_1478_);
lean_dec(v_lo_1472_);
return v_snd_1479_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__6___redArg___boxed(lean_object* v_n_1506_, lean_object* v_as_1507_, lean_object* v_lo_1508_, lean_object* v_hi_1509_){
_start:
{
lean_object* v_res_1510_; 
v_res_1510_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__6___redArg(v_n_1506_, v_as_1507_, v_lo_1508_, v_hi_1509_);
lean_dec(v_hi_1509_);
lean_dec(v_n_1506_);
return v_res_1510_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__3_spec__5_spec__8___redArg(lean_object* v_keys_1511_, lean_object* v_vals_1512_, lean_object* v_i_1513_, lean_object* v_k_1514_){
_start:
{
lean_object* v___x_1515_; uint8_t v___x_1516_; 
v___x_1515_ = lean_array_get_size(v_keys_1511_);
v___x_1516_ = lean_nat_dec_lt(v_i_1513_, v___x_1515_);
if (v___x_1516_ == 0)
{
lean_object* v___x_1517_; 
lean_dec(v_i_1513_);
v___x_1517_ = lean_box(0);
return v___x_1517_;
}
else
{
lean_object* v_k_x27_1518_; uint8_t v___x_1519_; 
v_k_x27_1518_ = lean_array_fget_borrowed(v_keys_1511_, v_i_1513_);
v___x_1519_ = lean_name_eq(v_k_1514_, v_k_x27_1518_);
if (v___x_1519_ == 0)
{
lean_object* v___x_1520_; lean_object* v___x_1521_; 
v___x_1520_ = lean_unsigned_to_nat(1u);
v___x_1521_ = lean_nat_add(v_i_1513_, v___x_1520_);
lean_dec(v_i_1513_);
v_i_1513_ = v___x_1521_;
goto _start;
}
else
{
lean_object* v___x_1523_; lean_object* v___x_1524_; 
v___x_1523_ = lean_array_fget_borrowed(v_vals_1512_, v_i_1513_);
lean_dec(v_i_1513_);
lean_inc(v___x_1523_);
v___x_1524_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1524_, 0, v___x_1523_);
return v___x_1524_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__3_spec__5_spec__8___redArg___boxed(lean_object* v_keys_1525_, lean_object* v_vals_1526_, lean_object* v_i_1527_, lean_object* v_k_1528_){
_start:
{
lean_object* v_res_1529_; 
v_res_1529_ = lp_batteries_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__3_spec__5_spec__8___redArg(v_keys_1525_, v_vals_1526_, v_i_1527_, v_k_1528_);
lean_dec(v_k_1528_);
lean_dec_ref(v_vals_1526_);
lean_dec_ref(v_keys_1525_);
return v_res_1529_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__3_spec__5___redArg(lean_object* v_x_1530_, size_t v_x_1531_, lean_object* v_x_1532_){
_start:
{
if (lean_obj_tag(v_x_1530_) == 0)
{
lean_object* v_es_1533_; lean_object* v___x_1534_; size_t v___x_1535_; size_t v___x_1536_; lean_object* v_j_1537_; lean_object* v___x_1538_; 
v_es_1533_ = lean_ctor_get(v_x_1530_, 0);
v___x_1534_ = lean_box(2);
v___x_1535_ = ((size_t)31ULL);
v___x_1536_ = lean_usize_land(v_x_1531_, v___x_1535_);
v_j_1537_ = lean_usize_to_nat(v___x_1536_);
v___x_1538_ = lean_array_get_borrowed(v___x_1534_, v_es_1533_, v_j_1537_);
lean_dec(v_j_1537_);
switch(lean_obj_tag(v___x_1538_))
{
case 0:
{
lean_object* v_key_1539_; lean_object* v_val_1540_; uint8_t v___x_1541_; 
v_key_1539_ = lean_ctor_get(v___x_1538_, 0);
v_val_1540_ = lean_ctor_get(v___x_1538_, 1);
v___x_1541_ = lean_name_eq(v_x_1532_, v_key_1539_);
if (v___x_1541_ == 0)
{
lean_object* v___x_1542_; 
v___x_1542_ = lean_box(0);
return v___x_1542_;
}
else
{
lean_object* v___x_1543_; 
lean_inc(v_val_1540_);
v___x_1543_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1543_, 0, v_val_1540_);
return v___x_1543_;
}
}
case 1:
{
lean_object* v_node_1544_; size_t v___x_1545_; size_t v___x_1546_; 
v_node_1544_ = lean_ctor_get(v___x_1538_, 0);
v___x_1545_ = ((size_t)5ULL);
v___x_1546_ = lean_usize_shift_right(v_x_1531_, v___x_1545_);
v_x_1530_ = v_node_1544_;
v_x_1531_ = v___x_1546_;
goto _start;
}
default: 
{
lean_object* v___x_1548_; 
v___x_1548_ = lean_box(0);
return v___x_1548_;
}
}
}
else
{
lean_object* v_ks_1549_; lean_object* v_vs_1550_; lean_object* v___x_1551_; lean_object* v___x_1552_; 
v_ks_1549_ = lean_ctor_get(v_x_1530_, 0);
v_vs_1550_ = lean_ctor_get(v_x_1530_, 1);
v___x_1551_ = lean_unsigned_to_nat(0u);
v___x_1552_ = lp_batteries_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__3_spec__5_spec__8___redArg(v_ks_1549_, v_vs_1550_, v___x_1551_, v_x_1532_);
return v___x_1552_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__3_spec__5___redArg___boxed(lean_object* v_x_1553_, lean_object* v_x_1554_, lean_object* v_x_1555_){
_start:
{
size_t v_x_14346__boxed_1556_; lean_object* v_res_1557_; 
v_x_14346__boxed_1556_ = lean_unbox_usize(v_x_1554_);
lean_dec(v_x_1554_);
v_res_1557_ = lp_batteries_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__3_spec__5___redArg(v_x_1553_, v_x_14346__boxed_1556_, v_x_1555_);
lean_dec(v_x_1555_);
lean_dec_ref(v_x_1553_);
return v_res_1557_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_find_x3f___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__3___redArg(lean_object* v_x_1558_, lean_object* v_x_1559_){
_start:
{
uint64_t v___y_1561_; 
if (lean_obj_tag(v_x_1559_) == 0)
{
uint64_t v___x_1564_; 
v___x_1564_ = 1723ULL;
v___y_1561_ = v___x_1564_;
goto v___jp_1560_;
}
else
{
uint64_t v_hash_1565_; 
v_hash_1565_ = lean_ctor_get_uint64(v_x_1559_, sizeof(void*)*2);
v___y_1561_ = v_hash_1565_;
goto v___jp_1560_;
}
v___jp_1560_:
{
size_t v___x_1562_; lean_object* v___x_1563_; 
v___x_1562_ = lean_uint64_to_usize(v___y_1561_);
v___x_1563_ = lp_batteries_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__3_spec__5___redArg(v_x_1558_, v___x_1562_, v_x_1559_);
return v___x_1563_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_find_x3f___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__3___redArg___boxed(lean_object* v_x_1566_, lean_object* v_x_1567_){
_start:
{
lean_object* v_res_1568_; 
v_res_1568_ = lp_batteries_Lean_PersistentHashMap_find_x3f___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__3___redArg(v_x_1566_, v_x_1567_);
lean_dec(v_x_1567_);
lean_dec_ref(v_x_1566_);
return v_res_1568_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__7(lean_object* v_x_1569_, lean_object* v_x_1570_){
_start:
{
if (lean_obj_tag(v_x_1570_) == 0)
{
return v_x_1569_;
}
else
{
lean_object* v_key_1571_; lean_object* v_value_1572_; lean_object* v_tail_1573_; lean_object* v___x_1574_; lean_object* v___x_1575_; 
v_key_1571_ = lean_ctor_get(v_x_1570_, 0);
v_value_1572_ = lean_ctor_get(v_x_1570_, 1);
v_tail_1573_ = lean_ctor_get(v_x_1570_, 2);
lean_inc(v_value_1572_);
lean_inc(v_key_1571_);
v___x_1574_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1574_, 0, v_key_1571_);
lean_ctor_set(v___x_1574_, 1, v_value_1572_);
v___x_1575_ = lean_array_push(v_x_1569_, v___x_1574_);
v_x_1569_ = v___x_1575_;
v_x_1570_ = v_tail_1573_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__7___boxed(lean_object* v_x_1577_, lean_object* v_x_1578_){
_start:
{
lean_object* v_res_1579_; 
v_res_1579_ = lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__7(v_x_1577_, v_x_1578_);
lean_dec(v_x_1578_);
return v_res_1579_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__8(lean_object* v_as_1580_, size_t v_i_1581_, size_t v_stop_1582_, lean_object* v_b_1583_){
_start:
{
uint8_t v___x_1584_; 
v___x_1584_ = lean_usize_dec_eq(v_i_1581_, v_stop_1582_);
if (v___x_1584_ == 0)
{
lean_object* v___x_1585_; lean_object* v___x_1586_; size_t v___x_1587_; size_t v___x_1588_; 
v___x_1585_ = lean_array_uget_borrowed(v_as_1580_, v_i_1581_);
v___x_1586_ = lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__7(v_b_1583_, v___x_1585_);
v___x_1587_ = ((size_t)1ULL);
v___x_1588_ = lean_usize_add(v_i_1581_, v___x_1587_);
v_i_1581_ = v___x_1588_;
v_b_1583_ = v___x_1586_;
goto _start;
}
else
{
return v_b_1583_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__8___boxed(lean_object* v_as_1590_, lean_object* v_i_1591_, lean_object* v_stop_1592_, lean_object* v_b_1593_){
_start:
{
size_t v_i_boxed_1594_; size_t v_stop_boxed_1595_; lean_object* v_res_1596_; 
v_i_boxed_1594_ = lean_unbox_usize(v_i_1591_);
lean_dec(v_i_1591_);
v_stop_boxed_1595_ = lean_unbox_usize(v_stop_1592_);
lean_dec(v_stop_1592_);
v_res_1596_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__8(v_as_1590_, v_i_boxed_1594_, v_stop_boxed_1595_, v_b_1593_);
lean_dec_ref(v_as_1590_);
return v_res_1596_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__2_spec__3___redArg(lean_object* v_o_1597_, lean_object* v___y_1598_){
_start:
{
lean_object* v___x_1600_; lean_object* v_env_1601_; lean_object* v___x_1602_; lean_object* v_toEnvExtension_1603_; lean_object* v_asyncMode_1604_; lean_object* v___x_1605_; lean_object* v___x_1606_; lean_object* v___x_1607_; lean_object* v_merged_1608_; lean_object* v___x_1610_; uint8_t v_isShared_1611_; uint8_t v_isSharedCheck_1616_; 
v___x_1600_ = lean_st_ref_get(v___y_1598_);
v_env_1601_ = lean_ctor_get(v___x_1600_, 0);
lean_inc_ref(v_env_1601_);
lean_dec(v___x_1600_);
v___x_1602_ = l_Lean_Linter_linterSetsExt;
v_toEnvExtension_1603_ = lean_ctor_get(v___x_1602_, 0);
v_asyncMode_1604_ = lean_ctor_get(v_toEnvExtension_1603_, 2);
v___x_1605_ = l_Lean_Linter_instInhabitedLinterSetsState_default;
v___x_1606_ = lean_box(0);
v___x_1607_ = l_Lean_PersistentEnvExtension_getState___redArg(v___x_1605_, v___x_1602_, v_env_1601_, v_asyncMode_1604_, v___x_1606_);
v_merged_1608_ = lean_ctor_get(v___x_1607_, 0);
v_isSharedCheck_1616_ = !lean_is_exclusive(v___x_1607_);
if (v_isSharedCheck_1616_ == 0)
{
lean_object* v_unused_1617_; 
v_unused_1617_ = lean_ctor_get(v___x_1607_, 1);
lean_dec(v_unused_1617_);
v___x_1610_ = v___x_1607_;
v_isShared_1611_ = v_isSharedCheck_1616_;
goto v_resetjp_1609_;
}
else
{
lean_inc(v_merged_1608_);
lean_dec(v___x_1607_);
v___x_1610_ = lean_box(0);
v_isShared_1611_ = v_isSharedCheck_1616_;
goto v_resetjp_1609_;
}
v_resetjp_1609_:
{
lean_object* v___x_1613_; 
if (v_isShared_1611_ == 0)
{
lean_ctor_set(v___x_1610_, 1, v_merged_1608_);
lean_ctor_set(v___x_1610_, 0, v_o_1597_);
v___x_1613_ = v___x_1610_;
goto v_reusejp_1612_;
}
else
{
lean_object* v_reuseFailAlloc_1615_; 
v_reuseFailAlloc_1615_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1615_, 0, v_o_1597_);
lean_ctor_set(v_reuseFailAlloc_1615_, 1, v_merged_1608_);
v___x_1613_ = v_reuseFailAlloc_1615_;
goto v_reusejp_1612_;
}
v_reusejp_1612_:
{
lean_object* v___x_1614_; 
v___x_1614_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1614_, 0, v___x_1613_);
return v___x_1614_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__2_spec__3___redArg___boxed(lean_object* v_o_1618_, lean_object* v___y_1619_, lean_object* v___y_1620_){
_start:
{
lean_object* v_res_1621_; 
v_res_1621_ = lp_batteries_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__2_spec__3___redArg(v_o_1618_, v___y_1619_);
lean_dec(v___y_1619_);
return v_res_1621_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Linter_getLinterOptions___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__2(lean_object* v___y_1622_, lean_object* v___y_1623_){
_start:
{
lean_object* v___x_1625_; lean_object* v_scopes_1626_; lean_object* v___x_1627_; lean_object* v___x_1628_; lean_object* v_opts_1629_; lean_object* v___x_1630_; 
v___x_1625_ = lean_st_ref_get(v___y_1623_);
v_scopes_1626_ = lean_ctor_get(v___x_1625_, 2);
lean_inc(v_scopes_1626_);
lean_dec(v___x_1625_);
v___x_1627_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_1628_ = l_List_head_x21___redArg(v___x_1627_, v_scopes_1626_);
lean_dec(v_scopes_1626_);
v_opts_1629_ = lean_ctor_get(v___x_1628_, 1);
lean_inc_ref(v_opts_1629_);
lean_dec(v___x_1628_);
v___x_1630_ = lp_batteries_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__2_spec__3___redArg(v_opts_1629_, v___y_1623_);
return v___x_1630_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Linter_getLinterOptions___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__2___boxed(lean_object* v___y_1631_, lean_object* v___y_1632_, lean_object* v___y_1633_){
_start:
{
lean_object* v_res_1634_; 
v_res_1634_ = lp_batteries_Lean_Linter_getLinterOptions___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__2(v___y_1631_, v___y_1632_);
lean_dec(v___y_1632_);
lean_dec_ref(v___y_1631_);
return v_res_1634_;
}
}
static lean_object* _init_lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___lam__0___closed__4(void){
_start:
{
lean_object* v___x_1641_; lean_object* v___x_1642_; lean_object* v___x_1643_; 
v___x_1641_ = lean_box(0);
v___x_1642_ = lean_unsigned_to_nat(16u);
v___x_1643_ = lean_mk_array(v___x_1642_, v___x_1641_);
return v___x_1643_;
}
}
static lean_object* _init_lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___lam__0___closed__5(void){
_start:
{
lean_object* v___x_1644_; lean_object* v___x_1645_; lean_object* v___x_1646_; 
v___x_1644_ = lean_obj_once(&lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___lam__0___closed__4, &lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___lam__0___closed__4_once, _init_lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___lam__0___closed__4);
v___x_1645_ = lean_unsigned_to_nat(0u);
v___x_1646_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1646_, 0, v___x_1645_);
lean_ctor_set(v___x_1646_, 1, v___x_1644_);
return v___x_1646_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___lam__0(lean_object* v_stx_1647_, lean_object* v___y_1648_, lean_object* v___y_1649_){
_start:
{
lean_object* v___y_1652_; lean_object* v___y_1653_; lean_object* v___y_1675_; lean_object* v___y_1676_; lean_object* v___y_1677_; lean_object* v___y_1678_; lean_object* v___y_1679_; lean_object* v___y_1682_; lean_object* v___y_1683_; lean_object* v___y_1684_; lean_object* v___y_1685_; lean_object* v___y_1686_; lean_object* v___y_1689_; lean_object* v___y_1690_; lean_object* v___y_1698_; lean_object* v___y_1699_; lean_object* v___y_1700_; lean_object* v___x_1727_; lean_object* v_a_1728_; lean_object* v___x_1730_; uint8_t v_isShared_1731_; uint8_t v_isSharedCheck_1783_; 
v___x_1727_ = lp_batteries_Lean_Linter_getLinterOptions___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__2(v___y_1648_, v___y_1649_);
v_a_1728_ = lean_ctor_get(v___x_1727_, 0);
v_isSharedCheck_1783_ = !lean_is_exclusive(v___x_1727_);
if (v_isSharedCheck_1783_ == 0)
{
v___x_1730_ = v___x_1727_;
v_isShared_1731_ = v_isSharedCheck_1783_;
goto v_resetjp_1729_;
}
else
{
lean_inc(v_a_1728_);
lean_dec(v___x_1727_);
v___x_1730_ = lean_box(0);
v_isShared_1731_ = v_isSharedCheck_1783_;
goto v_resetjp_1729_;
}
v___jp_1651_:
{
size_t v_sz_1654_; size_t v___x_1655_; lean_object* v___x_1656_; 
v_sz_1654_ = lean_array_size(v___y_1653_);
v___x_1655_ = ((size_t)0ULL);
v___x_1656_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__5(v___y_1653_, v_sz_1654_, v___x_1655_, v___y_1652_, v___y_1648_, v___y_1649_);
lean_dec_ref(v___y_1653_);
if (lean_obj_tag(v___x_1656_) == 0)
{
lean_object* v___x_1658_; uint8_t v_isShared_1659_; uint8_t v_isSharedCheck_1664_; 
v_isSharedCheck_1664_ = !lean_is_exclusive(v___x_1656_);
if (v_isSharedCheck_1664_ == 0)
{
lean_object* v_unused_1665_; 
v_unused_1665_ = lean_ctor_get(v___x_1656_, 0);
lean_dec(v_unused_1665_);
v___x_1658_ = v___x_1656_;
v_isShared_1659_ = v_isSharedCheck_1664_;
goto v_resetjp_1657_;
}
else
{
lean_dec(v___x_1656_);
v___x_1658_ = lean_box(0);
v_isShared_1659_ = v_isSharedCheck_1664_;
goto v_resetjp_1657_;
}
v_resetjp_1657_:
{
lean_object* v___x_1660_; lean_object* v___x_1662_; 
v___x_1660_ = lean_box(0);
if (v_isShared_1659_ == 0)
{
lean_ctor_set(v___x_1658_, 0, v___x_1660_);
v___x_1662_ = v___x_1658_;
goto v_reusejp_1661_;
}
else
{
lean_object* v_reuseFailAlloc_1663_; 
v_reuseFailAlloc_1663_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1663_, 0, v___x_1660_);
v___x_1662_ = v_reuseFailAlloc_1663_;
goto v_reusejp_1661_;
}
v_reusejp_1661_:
{
return v___x_1662_;
}
}
}
else
{
lean_object* v_a_1666_; lean_object* v___x_1668_; uint8_t v_isShared_1669_; uint8_t v_isSharedCheck_1673_; 
v_a_1666_ = lean_ctor_get(v___x_1656_, 0);
v_isSharedCheck_1673_ = !lean_is_exclusive(v___x_1656_);
if (v_isSharedCheck_1673_ == 0)
{
v___x_1668_ = v___x_1656_;
v_isShared_1669_ = v_isSharedCheck_1673_;
goto v_resetjp_1667_;
}
else
{
lean_inc(v_a_1666_);
lean_dec(v___x_1656_);
v___x_1668_ = lean_box(0);
v_isShared_1669_ = v_isSharedCheck_1673_;
goto v_resetjp_1667_;
}
v_resetjp_1667_:
{
lean_object* v___x_1671_; 
if (v_isShared_1669_ == 0)
{
v___x_1671_ = v___x_1668_;
goto v_reusejp_1670_;
}
else
{
lean_object* v_reuseFailAlloc_1672_; 
v_reuseFailAlloc_1672_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1672_, 0, v_a_1666_);
v___x_1671_ = v_reuseFailAlloc_1672_;
goto v_reusejp_1670_;
}
v_reusejp_1670_:
{
return v___x_1671_;
}
}
}
}
v___jp_1674_:
{
lean_object* v___x_1680_; 
v___x_1680_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__6___redArg(v___y_1675_, v___y_1677_, v___y_1678_, v___y_1679_);
lean_dec(v___y_1679_);
lean_dec(v___y_1675_);
v___y_1652_ = v___y_1676_;
v___y_1653_ = v___x_1680_;
goto v___jp_1651_;
}
v___jp_1681_:
{
uint8_t v___x_1687_; 
v___x_1687_ = lean_nat_dec_le(v___y_1686_, v___y_1683_);
if (v___x_1687_ == 0)
{
lean_dec(v___y_1683_);
lean_inc(v___y_1686_);
v___y_1675_ = v___y_1682_;
v___y_1676_ = v___y_1684_;
v___y_1677_ = v___y_1685_;
v___y_1678_ = v___y_1686_;
v___y_1679_ = v___y_1686_;
goto v___jp_1674_;
}
else
{
v___y_1675_ = v___y_1682_;
v___y_1676_ = v___y_1684_;
v___y_1677_ = v___y_1685_;
v___y_1678_ = v___y_1686_;
v___y_1679_ = v___y_1683_;
goto v___jp_1674_;
}
}
v___jp_1688_:
{
lean_object* v___x_1691_; lean_object* v___x_1692_; uint8_t v___x_1693_; 
lean_inc_n(v___y_1689_, 2);
v___x_1691_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1691_, 0, v___y_1689_);
lean_ctor_set(v___x_1691_, 1, v___y_1689_);
v___x_1692_ = lean_array_get_size(v___y_1690_);
v___x_1693_ = lean_nat_dec_eq(v___x_1692_, v___y_1689_);
if (v___x_1693_ == 0)
{
lean_object* v___x_1694_; lean_object* v___x_1695_; uint8_t v___x_1696_; 
v___x_1694_ = lean_unsigned_to_nat(1u);
v___x_1695_ = lean_nat_sub(v___x_1692_, v___x_1694_);
v___x_1696_ = lean_nat_dec_le(v___y_1689_, v___x_1695_);
if (v___x_1696_ == 0)
{
lean_dec(v___y_1689_);
lean_inc(v___x_1695_);
v___y_1682_ = v___x_1692_;
v___y_1683_ = v___x_1695_;
v___y_1684_ = v___x_1691_;
v___y_1685_ = v___y_1690_;
v___y_1686_ = v___x_1695_;
goto v___jp_1681_;
}
else
{
v___y_1682_ = v___x_1692_;
v___y_1683_ = v___x_1695_;
v___y_1684_ = v___x_1691_;
v___y_1685_ = v___y_1690_;
v___y_1686_ = v___y_1689_;
goto v___jp_1681_;
}
}
else
{
lean_dec(v___y_1689_);
v___y_1652_ = v___x_1691_;
v___y_1653_ = v___y_1690_;
goto v___jp_1651_;
}
}
v___jp_1697_:
{
if (lean_obj_tag(v___y_1700_) == 0)
{
lean_object* v___x_1701_; lean_object* v_size_1702_; lean_object* v_buckets_1703_; lean_object* v___x_1704_; lean_object* v___x_1705_; uint8_t v___x_1706_; 
lean_dec_ref_known(v___y_1700_, 1);
v___x_1701_ = lean_st_ref_get(v___y_1699_);
lean_dec(v___y_1699_);
v_size_1702_ = lean_ctor_get(v___x_1701_, 0);
lean_inc(v_size_1702_);
v_buckets_1703_ = lean_ctor_get(v___x_1701_, 1);
lean_inc_ref(v_buckets_1703_);
lean_dec(v___x_1701_);
v___x_1704_ = lean_mk_empty_array_with_capacity(v_size_1702_);
lean_dec(v_size_1702_);
v___x_1705_ = lean_array_get_size(v_buckets_1703_);
v___x_1706_ = lean_nat_dec_lt(v___y_1698_, v___x_1705_);
if (v___x_1706_ == 0)
{
lean_dec_ref(v_buckets_1703_);
v___y_1689_ = v___y_1698_;
v___y_1690_ = v___x_1704_;
goto v___jp_1688_;
}
else
{
uint8_t v___x_1707_; 
v___x_1707_ = lean_nat_dec_le(v___x_1705_, v___x_1705_);
if (v___x_1707_ == 0)
{
if (v___x_1706_ == 0)
{
lean_dec_ref(v_buckets_1703_);
v___y_1689_ = v___y_1698_;
v___y_1690_ = v___x_1704_;
goto v___jp_1688_;
}
else
{
size_t v___x_1708_; size_t v___x_1709_; lean_object* v___x_1710_; 
v___x_1708_ = ((size_t)0ULL);
v___x_1709_ = lean_usize_of_nat(v___x_1705_);
v___x_1710_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__8(v_buckets_1703_, v___x_1708_, v___x_1709_, v___x_1704_);
lean_dec_ref(v_buckets_1703_);
v___y_1689_ = v___y_1698_;
v___y_1690_ = v___x_1710_;
goto v___jp_1688_;
}
}
else
{
size_t v___x_1711_; size_t v___x_1712_; lean_object* v___x_1713_; 
v___x_1711_ = ((size_t)0ULL);
v___x_1712_ = lean_usize_of_nat(v___x_1705_);
v___x_1713_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__8(v_buckets_1703_, v___x_1711_, v___x_1712_, v___x_1704_);
lean_dec_ref(v_buckets_1703_);
v___y_1689_ = v___y_1698_;
v___y_1690_ = v___x_1713_;
goto v___jp_1688_;
}
}
}
else
{
lean_object* v_a_1714_; lean_object* v___x_1716_; uint8_t v_isShared_1717_; uint8_t v_isSharedCheck_1726_; 
lean_dec(v___y_1699_);
lean_dec(v___y_1698_);
v_a_1714_ = lean_ctor_get(v___y_1700_, 0);
v_isSharedCheck_1726_ = !lean_is_exclusive(v___y_1700_);
if (v_isSharedCheck_1726_ == 0)
{
v___x_1716_ = v___y_1700_;
v_isShared_1717_ = v_isSharedCheck_1726_;
goto v_resetjp_1715_;
}
else
{
lean_inc(v_a_1714_);
lean_dec(v___y_1700_);
v___x_1716_ = lean_box(0);
v_isShared_1717_ = v_isSharedCheck_1726_;
goto v_resetjp_1715_;
}
v_resetjp_1715_:
{
lean_object* v_ref_1718_; lean_object* v___x_1719_; lean_object* v___x_1720_; lean_object* v___x_1721_; lean_object* v___x_1722_; lean_object* v___x_1724_; 
v_ref_1718_ = lean_ctor_get(v___y_1648_, 7);
v___x_1719_ = lean_io_error_to_string(v_a_1714_);
v___x_1720_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1720_, 0, v___x_1719_);
v___x_1721_ = l_Lean_MessageData_ofFormat(v___x_1720_);
lean_inc(v_ref_1718_);
v___x_1722_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1722_, 0, v_ref_1718_);
lean_ctor_set(v___x_1722_, 1, v___x_1721_);
if (v_isShared_1717_ == 0)
{
lean_ctor_set(v___x_1716_, 0, v___x_1722_);
v___x_1724_ = v___x_1716_;
goto v_reusejp_1723_;
}
else
{
lean_object* v_reuseFailAlloc_1725_; 
v_reuseFailAlloc_1725_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1725_, 0, v___x_1722_);
v___x_1724_ = v_reuseFailAlloc_1725_;
goto v_reusejp_1723_;
}
v_reusejp_1723_:
{
return v___x_1724_;
}
}
}
}
v_resetjp_1729_:
{
lean_object* v___x_1732_; uint8_t v___y_1734_; uint8_t v___x_1780_; 
v___x_1732_ = lean_st_ref_get(v___y_1649_);
v___x_1780_ = lp_batteries_Batteries_Linter_UnreachableTactic_getLinterUnreachableTactic(v_a_1728_);
lean_dec(v_a_1728_);
if (v___x_1780_ == 0)
{
lean_dec(v___x_1732_);
v___y_1734_ = v___x_1780_;
goto v___jp_1733_;
}
else
{
lean_object* v_infoState_1781_; uint8_t v_enabled_1782_; 
v_infoState_1781_ = lean_ctor_get(v___x_1732_, 8);
lean_inc_ref(v_infoState_1781_);
lean_dec(v___x_1732_);
v_enabled_1782_ = lean_ctor_get_uint8(v_infoState_1781_, sizeof(void*)*3);
lean_dec_ref(v_infoState_1781_);
v___y_1734_ = v_enabled_1782_;
goto v___jp_1733_;
}
v___jp_1733_:
{
if (v___y_1734_ == 0)
{
lean_object* v___x_1735_; lean_object* v___x_1737_; 
lean_dec(v_stx_1647_);
v___x_1735_ = lean_box(0);
if (v_isShared_1731_ == 0)
{
lean_ctor_set(v___x_1730_, 0, v___x_1735_);
v___x_1737_ = v___x_1730_;
goto v_reusejp_1736_;
}
else
{
lean_object* v_reuseFailAlloc_1738_; 
v_reuseFailAlloc_1738_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1738_, 0, v___x_1735_);
v___x_1737_ = v_reuseFailAlloc_1738_;
goto v_reusejp_1736_;
}
v_reusejp_1736_:
{
return v___x_1737_;
}
}
else
{
lean_object* v___x_1739_; lean_object* v_messages_1740_; uint8_t v___x_1741_; 
v___x_1739_ = lean_st_ref_get(v___y_1649_);
v_messages_1740_ = lean_ctor_get(v___x_1739_, 1);
lean_inc_ref(v_messages_1740_);
lean_dec(v___x_1739_);
v___x_1741_ = l_Lean_MessageLog_hasErrors(v_messages_1740_);
lean_dec_ref(v_messages_1740_);
if (v___x_1741_ == 0)
{
lean_object* v___x_1742_; lean_object* v_env_1743_; lean_object* v___x_1744_; lean_object* v_ext_1745_; lean_object* v_toEnvExtension_1746_; lean_object* v_asyncMode_1747_; lean_object* v___x_1748_; lean_object* v___x_1749_; lean_object* v_categories_1750_; lean_object* v___x_1751_; lean_object* v___x_1752_; 
v___x_1742_ = lean_st_ref_get(v___y_1649_);
v_env_1743_ = lean_ctor_get(v___x_1742_, 0);
lean_inc_ref(v_env_1743_);
lean_dec(v___x_1742_);
v___x_1744_ = l_Lean_Parser_parserExtension;
v_ext_1745_ = lean_ctor_get(v___x_1744_, 1);
v_toEnvExtension_1746_ = lean_ctor_get(v_ext_1745_, 0);
v_asyncMode_1747_ = lean_ctor_get(v_toEnvExtension_1746_, 2);
v___x_1748_ = l_Lean_Parser_ParserExtension_instInhabitedState_default;
v___x_1749_ = l_Lean_ScopedEnvExtension_getState___redArg(v___x_1748_, v___x_1744_, v_env_1743_, v_asyncMode_1747_);
v_categories_1750_ = lean_ctor_get(v___x_1749_, 2);
lean_inc_ref(v_categories_1750_);
lean_dec(v___x_1749_);
v___x_1751_ = ((lean_object*)(lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___lam__0___closed__1));
v___x_1752_ = lp_batteries_Lean_PersistentHashMap_find_x3f___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__3___redArg(v_categories_1750_, v___x_1751_);
if (lean_obj_tag(v___x_1752_) == 0)
{
lean_object* v___x_1753_; lean_object* v___x_1755_; 
lean_dec_ref(v_categories_1750_);
lean_dec(v_stx_1647_);
v___x_1753_ = lean_box(0);
if (v_isShared_1731_ == 0)
{
lean_ctor_set(v___x_1730_, 0, v___x_1753_);
v___x_1755_ = v___x_1730_;
goto v_reusejp_1754_;
}
else
{
lean_object* v_reuseFailAlloc_1756_; 
v_reuseFailAlloc_1756_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1756_, 0, v___x_1753_);
v___x_1755_ = v_reuseFailAlloc_1756_;
goto v_reusejp_1754_;
}
v_reusejp_1754_:
{
return v___x_1755_;
}
}
else
{
lean_object* v_val_1757_; lean_object* v___x_1758_; lean_object* v___x_1759_; 
v_val_1757_ = lean_ctor_get(v___x_1752_, 0);
lean_inc(v_val_1757_);
lean_dec_ref_known(v___x_1752_, 1);
v___x_1758_ = ((lean_object*)(lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___lam__0___closed__3));
v___x_1759_ = lp_batteries_Lean_PersistentHashMap_find_x3f___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__3___redArg(v_categories_1750_, v___x_1758_);
lean_dec_ref(v_categories_1750_);
if (lean_obj_tag(v___x_1759_) == 0)
{
lean_object* v___x_1760_; lean_object* v___x_1762_; 
lean_dec(v_val_1757_);
lean_dec(v_stx_1647_);
v___x_1760_ = lean_box(0);
if (v_isShared_1731_ == 0)
{
lean_ctor_set(v___x_1730_, 0, v___x_1760_);
v___x_1762_ = v___x_1730_;
goto v_reusejp_1761_;
}
else
{
lean_object* v_reuseFailAlloc_1763_; 
v_reuseFailAlloc_1763_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1763_, 0, v___x_1760_);
v___x_1762_ = v_reuseFailAlloc_1763_;
goto v_reusejp_1761_;
}
v_reusejp_1761_:
{
return v___x_1762_;
}
}
else
{
lean_object* v_val_1764_; lean_object* v___x_1765_; lean_object* v_a_1766_; lean_object* v___x_1767_; lean_object* v___x_1768_; lean_object* v___x_1769_; lean_object* v___x_1770_; lean_object* v___x_1771_; lean_object* v_kinds_1772_; lean_object* v_kinds_1773_; lean_object* v___x_1774_; 
lean_del_object(v___x_1730_);
v_val_1764_ = lean_ctor_get(v___x_1759_, 0);
lean_inc(v_val_1764_);
lean_dec_ref_known(v___x_1759_, 1);
v___x_1765_ = lp_batteries_Lean_Elab_getInfoTrees___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__4___redArg(v___y_1649_);
v_a_1766_ = lean_ctor_get(v___x_1765_, 0);
lean_inc(v_a_1766_);
lean_dec_ref(v___x_1765_);
v___x_1767_ = lean_unsigned_to_nat(0u);
v___x_1768_ = lean_obj_once(&lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___lam__0___closed__5, &lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___lam__0___closed__5_once, _init_lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___lam__0___closed__5);
v___x_1769_ = lean_st_mk_ref(v___x_1768_);
v___x_1770_ = lp_batteries_Batteries_Linter_UnreachableTactic_ignoreTacticKindsRef;
v___x_1771_ = lean_st_ref_get(v___x_1770_);
v_kinds_1772_ = lean_ctor_get(v_val_1757_, 1);
lean_inc_ref(v_kinds_1772_);
lean_dec(v_val_1757_);
v_kinds_1773_ = lean_ctor_get(v_val_1764_, 1);
lean_inc_ref(v_kinds_1773_);
lean_dec(v_val_1764_);
v___x_1774_ = lp_batteries_Batteries_Linter_UnreachableTactic_getTactics___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__10(v_kinds_1772_, v_kinds_1773_, v___y_1734_, v___x_1771_, v_stx_1647_, v___x_1769_);
lean_dec(v___x_1771_);
lean_dec_ref(v_kinds_1773_);
lean_dec_ref(v_kinds_1772_);
if (lean_obj_tag(v___x_1774_) == 0)
{
lean_object* v___x_1775_; 
lean_dec_ref_known(v___x_1774_, 1);
v___x_1775_ = lp_batteries_Batteries_Linter_UnreachableTactic_eraseUsedTacticsList(v_a_1766_, v___x_1769_);
v___y_1698_ = v___x_1767_;
v___y_1699_ = v___x_1769_;
v___y_1700_ = v___x_1775_;
goto v___jp_1697_;
}
else
{
lean_dec(v_a_1766_);
v___y_1698_ = v___x_1767_;
v___y_1699_ = v___x_1769_;
v___y_1700_ = v___x_1774_;
goto v___jp_1697_;
}
}
}
}
else
{
lean_object* v___x_1776_; lean_object* v___x_1778_; 
lean_dec(v_stx_1647_);
v___x_1776_ = lean_box(0);
if (v_isShared_1731_ == 0)
{
lean_ctor_set(v___x_1730_, 0, v___x_1776_);
v___x_1778_ = v___x_1730_;
goto v_reusejp_1777_;
}
else
{
lean_object* v_reuseFailAlloc_1779_; 
v_reuseFailAlloc_1779_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1779_, 0, v___x_1776_);
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
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___lam__0___boxed(lean_object* v_stx_1784_, lean_object* v___y_1785_, lean_object* v___y_1786_, lean_object* v___y_1787_){
_start:
{
lean_object* v_res_1788_; 
v_res_1788_ = lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter___lam__0(v_stx_1784_, v___y_1785_, v___y_1786_);
lean_dec(v___y_1786_);
lean_dec_ref(v___y_1785_);
return v_res_1788_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__2_spec__3(lean_object* v_o_1803_, lean_object* v___y_1804_, lean_object* v___y_1805_){
_start:
{
lean_object* v___x_1807_; 
v___x_1807_ = lp_batteries_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__2_spec__3___redArg(v_o_1803_, v___y_1805_);
return v___x_1807_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__2_spec__3___boxed(lean_object* v_o_1808_, lean_object* v___y_1809_, lean_object* v___y_1810_, lean_object* v___y_1811_){
_start:
{
lean_object* v_res_1812_; 
v_res_1812_ = lp_batteries_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__2_spec__3(v_o_1808_, v___y_1809_, v___y_1810_);
lean_dec(v___y_1810_);
lean_dec_ref(v___y_1809_);
return v_res_1812_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_find_x3f___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__3(lean_object* v_00_u03b2_1813_, lean_object* v_x_1814_, lean_object* v_x_1815_){
_start:
{
lean_object* v___x_1816_; 
v___x_1816_ = lp_batteries_Lean_PersistentHashMap_find_x3f___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__3___redArg(v_x_1814_, v_x_1815_);
return v___x_1816_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_find_x3f___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__3___boxed(lean_object* v_00_u03b2_1817_, lean_object* v_x_1818_, lean_object* v_x_1819_){
_start:
{
lean_object* v_res_1820_; 
v_res_1820_ = lp_batteries_Lean_PersistentHashMap_find_x3f___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__3(v_00_u03b2_1817_, v_x_1818_, v_x_1819_);
lean_dec(v_x_1819_);
lean_dec_ref(v_x_1818_);
return v_res_1820_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__6(lean_object* v_n_1821_, lean_object* v_as_1822_, lean_object* v_lo_1823_, lean_object* v_hi_1824_, lean_object* v_w_1825_, lean_object* v_hlo_1826_, lean_object* v_hhi_1827_){
_start:
{
lean_object* v___x_1828_; 
v___x_1828_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__6___redArg(v_n_1821_, v_as_1822_, v_lo_1823_, v_hi_1824_);
return v___x_1828_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__6___boxed(lean_object* v_n_1829_, lean_object* v_as_1830_, lean_object* v_lo_1831_, lean_object* v_hi_1832_, lean_object* v_w_1833_, lean_object* v_hlo_1834_, lean_object* v_hhi_1835_){
_start:
{
lean_object* v_res_1836_; 
v_res_1836_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__6(v_n_1829_, v_as_1830_, v_lo_1831_, v_hi_1832_, v_w_1833_, v_hlo_1834_, v_hhi_1835_);
lean_dec(v_hi_1832_);
lean_dec(v_n_1829_);
return v_res_1836_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_contains___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__9(lean_object* v_00_u03b2_1837_, lean_object* v_x_1838_, lean_object* v_x_1839_){
_start:
{
uint8_t v___x_1840_; 
v___x_1840_ = lp_batteries_Lean_PersistentHashMap_contains___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__9___redArg(v_x_1838_, v_x_1839_);
return v___x_1840_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_contains___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__9___boxed(lean_object* v_00_u03b2_1841_, lean_object* v_x_1842_, lean_object* v_x_1843_){
_start:
{
uint8_t v_res_1844_; lean_object* v_r_1845_; 
v_res_1844_ = lp_batteries_Lean_PersistentHashMap_contains___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__9(v_00_u03b2_1841_, v_x_1842_, v_x_1843_);
lean_dec(v_x_1843_);
lean_dec_ref(v_x_1842_);
v_r_1845_ = lean_box(v_res_1844_);
return v_r_1845_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__3_spec__5(lean_object* v_00_u03b2_1846_, lean_object* v_x_1847_, size_t v_x_1848_, lean_object* v_x_1849_){
_start:
{
lean_object* v___x_1850_; 
v___x_1850_ = lp_batteries_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__3_spec__5___redArg(v_x_1847_, v_x_1848_, v_x_1849_);
return v___x_1850_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__3_spec__5___boxed(lean_object* v_00_u03b2_1851_, lean_object* v_x_1852_, lean_object* v_x_1853_, lean_object* v_x_1854_){
_start:
{
size_t v_x_14851__boxed_1855_; lean_object* v_res_1856_; 
v_x_14851__boxed_1855_ = lean_unbox_usize(v_x_1853_);
lean_dec(v_x_1853_);
v_res_1856_ = lp_batteries_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__3_spec__5(v_00_u03b2_1851_, v_x_1852_, v_x_14851__boxed_1855_, v_x_1854_);
lean_dec(v_x_1854_);
lean_dec_ref(v_x_1852_);
return v_res_1856_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__6_spec__9(lean_object* v_n_1857_, lean_object* v_lo_1858_, lean_object* v_hi_1859_, lean_object* v_hhi_1860_, lean_object* v_pivot_1861_, lean_object* v_as_1862_, lean_object* v_i_1863_, lean_object* v_k_1864_, lean_object* v_ilo_1865_, lean_object* v_ik_1866_, lean_object* v_w_1867_){
_start:
{
lean_object* v___x_1868_; 
v___x_1868_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__6_spec__9___redArg(v_hi_1859_, v_pivot_1861_, v_as_1862_, v_i_1863_, v_k_1864_);
return v___x_1868_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__6_spec__9___boxed(lean_object* v_n_1869_, lean_object* v_lo_1870_, lean_object* v_hi_1871_, lean_object* v_hhi_1872_, lean_object* v_pivot_1873_, lean_object* v_as_1874_, lean_object* v_i_1875_, lean_object* v_k_1876_, lean_object* v_ilo_1877_, lean_object* v_ik_1878_, lean_object* v_w_1879_){
_start:
{
lean_object* v_res_1880_; 
v_res_1880_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__6_spec__9(v_n_1869_, v_lo_1870_, v_hi_1871_, v_hhi_1872_, v_pivot_1873_, v_as_1874_, v_i_1875_, v_k_1876_, v_ilo_1877_, v_ik_1878_, v_w_1879_);
lean_dec(v_hi_1871_);
lean_dec(v_lo_1870_);
lean_dec(v_n_1869_);
return v_res_1880_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__9_spec__13(lean_object* v_00_u03b2_1881_, lean_object* v_x_1882_, size_t v_x_1883_, lean_object* v_x_1884_){
_start:
{
uint8_t v___x_1885_; 
v___x_1885_ = lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__9_spec__13___redArg(v_x_1882_, v_x_1883_, v_x_1884_);
return v___x_1885_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__9_spec__13___boxed(lean_object* v_00_u03b2_1886_, lean_object* v_x_1887_, lean_object* v_x_1888_, lean_object* v_x_1889_){
_start:
{
size_t v_x_14864__boxed_1890_; uint8_t v_res_1891_; lean_object* v_r_1892_; 
v_x_14864__boxed_1890_ = lean_unbox_usize(v_x_1888_);
lean_dec(v_x_1888_);
v_res_1891_ = lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__9_spec__13(v_00_u03b2_1886_, v_x_1887_, v_x_14864__boxed_1890_, v_x_1889_);
lean_dec(v_x_1889_);
lean_dec_ref(v_x_1887_);
v_r_1892_ = lean_box(v_res_1891_);
return v_r_1892_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnreachableTactic_getTactics___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__10_spec__15(lean_object* v_00_u03b2_1893_, lean_object* v_m_1894_, lean_object* v_a_1895_, lean_object* v_b_1896_){
_start:
{
lean_object* v___x_1897_; 
v___x_1897_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnreachableTactic_getTactics___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__10_spec__15___redArg(v_m_1894_, v_a_1895_, v_b_1896_);
return v___x_1897_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3_spec__13(lean_object* v_msgData_1898_, lean_object* v___y_1899_, lean_object* v___y_1900_){
_start:
{
lean_object* v___x_1902_; 
v___x_1902_ = lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3_spec__13___redArg(v_msgData_1898_, v___y_1900_);
return v___x_1902_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3_spec__13___boxed(lean_object* v_msgData_1903_, lean_object* v___y_1904_, lean_object* v___y_1905_, lean_object* v___y_1906_){
_start:
{
lean_object* v_res_1907_; 
v_res_1907_ = lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__1_spec__1_spec__3_spec__13(v_msgData_1903_, v___y_1904_, v___y_1905_);
lean_dec(v___y_1905_);
lean_dec_ref(v___y_1904_);
return v_res_1907_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__3_spec__5_spec__8(lean_object* v_00_u03b2_1908_, lean_object* v_keys_1909_, lean_object* v_vals_1910_, lean_object* v_heq_1911_, lean_object* v_i_1912_, lean_object* v_k_1913_){
_start:
{
lean_object* v___x_1914_; 
v___x_1914_ = lp_batteries_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__3_spec__5_spec__8___redArg(v_keys_1909_, v_vals_1910_, v_i_1912_, v_k_1913_);
return v___x_1914_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__3_spec__5_spec__8___boxed(lean_object* v_00_u03b2_1915_, lean_object* v_keys_1916_, lean_object* v_vals_1917_, lean_object* v_heq_1918_, lean_object* v_i_1919_, lean_object* v_k_1920_){
_start:
{
lean_object* v_res_1921_; 
v_res_1921_ = lp_batteries_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__3_spec__5_spec__8(v_00_u03b2_1915_, v_keys_1916_, v_vals_1917_, v_heq_1918_, v_i_1919_, v_k_1920_);
lean_dec(v_k_1920_);
lean_dec_ref(v_vals_1917_);
lean_dec_ref(v_keys_1916_);
return v_res_1921_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__9_spec__13_spec__16(lean_object* v_00_u03b2_1922_, lean_object* v_keys_1923_, lean_object* v_vals_1924_, lean_object* v_heq_1925_, lean_object* v_i_1926_, lean_object* v_k_1927_){
_start:
{
uint8_t v___x_1928_; 
v___x_1928_ = lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__9_spec__13_spec__16___redArg(v_keys_1923_, v_i_1926_, v_k_1927_);
return v___x_1928_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__9_spec__13_spec__16___boxed(lean_object* v_00_u03b2_1929_, lean_object* v_keys_1930_, lean_object* v_vals_1931_, lean_object* v_heq_1932_, lean_object* v_i_1933_, lean_object* v_k_1934_){
_start:
{
uint8_t v_res_1935_; lean_object* v_r_1936_; 
v_res_1935_ = lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__9_spec__13_spec__16(v_00_u03b2_1929_, v_keys_1930_, v_vals_1931_, v_heq_1932_, v_i_1933_, v_k_1934_);
lean_dec(v_k_1934_);
lean_dec_ref(v_vals_1931_);
lean_dec_ref(v_keys_1930_);
v_r_1936_ = lean_box(v_res_1935_);
return v_r_1936_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnreachableTactic_getTactics___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__10_spec__15_spec__19(lean_object* v_00_u03b2_1937_, lean_object* v_data_1938_){
_start:
{
lean_object* v___x_1939_; 
v___x_1939_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnreachableTactic_getTactics___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__10_spec__15_spec__19___redArg(v_data_1938_);
return v___x_1939_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnreachableTactic_getTactics___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__10_spec__15_spec__20(lean_object* v_00_u03b2_1940_, lean_object* v_a_1941_, lean_object* v_b_1942_, lean_object* v_x_1943_){
_start:
{
lean_object* v___x_1944_; 
v___x_1944_ = lp_batteries_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnreachableTactic_getTactics___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__10_spec__15_spec__20___redArg(v_a_1941_, v_b_1942_, v_x_1943_);
return v___x_1944_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnreachableTactic_getTactics___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__10_spec__15_spec__19_spec__22(lean_object* v_00_u03b2_1945_, lean_object* v_i_1946_, lean_object* v_source_1947_, lean_object* v_target_1948_){
_start:
{
lean_object* v___x_1949_; 
v___x_1949_ = lp_batteries___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnreachableTactic_getTactics___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__10_spec__15_spec__19_spec__22___redArg(v_i_1946_, v_source_1947_, v_target_1948_);
return v___x_1949_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnreachableTactic_getTactics___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__10_spec__15_spec__19_spec__22_spec__24(lean_object* v_00_u03b2_1950_, lean_object* v_x_1951_, lean_object* v_x_1952_){
_start:
{
lean_object* v___x_1953_; 
v___x_1953_ = lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnreachableTactic_getTactics___at___00Batteries_Linter_UnreachableTactic_unreachableTacticLinter_spec__10_spec__15_spec__19_spec__22_spec__24___redArg(v_x_1951_, v_x_1952_);
return v___x_1953_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn_00___x40_Batteries_Linter_UnreachableTactic_179727238____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_1955_; lean_object* v___x_1956_; 
v___x_1955_ = ((lean_object*)(lp_batteries_Batteries_Linter_UnreachableTactic_unreachableTacticLinter));
v___x_1956_ = l_Lean_Elab_Command_addLinter(v___x_1955_);
return v___x_1956_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn_00___x40_Batteries_Linter_UnreachableTactic_179727238____hygCtx___hyg_2____boxed(lean_object* v_a_1957_){
_start:
{
lean_object* v_res_1958_; 
v_res_1958_ = lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn_00___x40_Batteries_Linter_UnreachableTactic_179727238____hygCtx___hyg_2_();
return v_res_1958_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_batteries_Batteries_Linter_UnreachableTactic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize_runtime_module();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Command(uint8_t builtin);
lean_object* runtime_initialize_Lean_Parser_Syntax(uint8_t builtin);
lean_object* runtime_initialize_Init_Try(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Tactic_Unreachable(uint8_t builtin);
lean_object* runtime_initialize_Lean_Linter_Basic(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_batteries_Batteries_Linter_UnreachableTactic(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Command(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Parser_Syntax(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Init_Try(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Tactic_Unreachable(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Linter_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_initFn_00___x40_Batteries_Linter_UnreachableTactic_2857764029____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_batteries_Batteries_Linter_linter_unreachableTactic = lean_io_result_get_value(res);
lean_mark_persistent(lp_batteries_Batteries_Linter_linter_unreachableTactic);
lean_dec_ref(res);
res = lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn_00___x40_Batteries_Linter_UnreachableTactic_949854657____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_batteries_Batteries_Linter_UnreachableTactic_ignoreTacticKindsRef = lean_io_result_get_value(res);
lean_mark_persistent(lp_batteries_Batteries_Linter_UnreachableTactic_ignoreTacticKindsRef);
lean_dec_ref(res);
res = lp_batteries___private_Batteries_Linter_UnreachableTactic_0__Batteries_Linter_UnreachableTactic_initFn_00___x40_Batteries_Linter_UnreachableTactic_179727238____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Command(uint8_t builtin);
lean_object* initialize_Lean_Parser_Syntax(uint8_t builtin);
lean_object* initialize_Init_Try(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Tactic_Unreachable(uint8_t builtin);
lean_object* initialize_Lean_Linter_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Linter_UnreachableTactic(uint8_t builtin) {
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
res = initialize_Lean_Parser_Syntax(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init_Try(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Tactic_Unreachable(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Linter_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Linter_UnreachableTactic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_batteries_Batteries_Linter_UnreachableTactic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_batteries_Batteries_Linter_UnreachableTactic(builtin);
}
#ifdef __cplusplus
}
#endif
