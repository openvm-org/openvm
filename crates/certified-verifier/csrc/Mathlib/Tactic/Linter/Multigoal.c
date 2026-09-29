// Lean compiler output
// Module: Mathlib.Tactic.Linter.Multigoal
// Imports: public import Init public meta import Init public meta import Lean.Elab.Command public meta import Mathlib.Tactic.Linter.Header public import Lean.Parser.Term
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
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__6(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__2___boxed(lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_instForIn_x27InferInstanceMembershipOfMonad___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instForInOfForIn_x27___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_register_option(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_PrettyPrinter_ppTactic(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* l_Lean_Syntax_getHeadInfo(lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
lean_object* l_List_lengthTR___redArg(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getKind(lean_object*);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* l_Lean_PersistentArray_toArray___redArg(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_liftCoreM___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* l_Lean_MessageData_note(lean_object*);
extern lean_object* l_Lean_Linter_linterMessageTag;
lean_object* l_Lean_Elab_Command_getScope___redArg(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Lean_MessageLog_add(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(lean_object*);
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
lean_object* l_Lean_Name_beq___boxed(lean_object*, lean_object*);
extern lean_object* l_Lean_Linter_linterSetsExt;
extern lean_object* l_Lean_Linter_instInhabitedLinterSetsState_default;
lean_object* l_Lean_PersistentEnvExtension_getState___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Linter_getLinterValue(lean_object*, lean_object*);
uint8_t l_Lean_MessageLog_hasErrors(lean_object*);
lean_object* l_Lean_withSetOptionIn___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_hash___override___boxed(lean_object*);
lean_object* l_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_addLinter(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4__spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "style"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "multiGoal"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(186, 218, 113, 226, 101, 176, 32, 79)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(105, 62, 218, 153, 100, 142, 29, 251)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(15, 42, 169, 25, 251, 166, 61, 117)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 28, .m_capacity = 28, .m_length = 27, .m_data = "enable the multiGoal linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(120, 131, 127, 204, 79, 169, 80, 92)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(101, 237, 90, 120, 51, 59, 46, 172)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4__value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(98, 189, 128, 85, 154, 50, 252, 160)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4__value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(248, 248, 118, 61, 250, 208, 250, 54)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_linter_style_multiGoal;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Name_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__0_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Name_hash___override___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "injections"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__6_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__6_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__4_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__6_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__5_value),LEAN_SCALAR_PTR_LITERAL(100, 83, 22, 173, 194, 141, 92, 179)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__6_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "substVars"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__8_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__8_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__4_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__8_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__7_value),LEAN_SCALAR_PTR_LITERAL(164, 80, 240, 20, 13, 181, 46, 131)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__8_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Batteries"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__9_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "tacticPick_goal-_"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__9_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__11_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__4_value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__11_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__10_value),LEAN_SCALAR_PTR_LITERAL(121, 117, 22, 172, 18, 18, 128, 239)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__11_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "case'"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__12_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__13_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__13_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__13_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__13_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__13_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__4_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__13_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__12_value),LEAN_SCALAR_PTR_LITERAL(134, 21, 185, 205, 238, 88, 7, 106)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__13_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "tactic#adaptation_note_"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__14_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__14_value),LEAN_SCALAR_PTR_LITERAL(37, 97, 231, 128, 245, 117, 181, 171)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__15_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "tacticSleep_heartbeats_"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__16_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__16_value),LEAN_SCALAR_PTR_LITERAL(88, 4, 66, 45, 18, 210, 176, 193)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__17_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "case"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__18 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__18_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__19_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__19_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__19_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__19_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__19_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__4_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__19_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__18_value),LEAN_SCALAR_PTR_LITERAL(216, 244, 120, 128, 139, 198, 139, 51)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__19 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__19_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "constructor"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__20 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__20_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__21_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__21_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__21_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__21_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__21_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__4_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__21_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__20_value),LEAN_SCALAR_PTR_LITERAL(144, 188, 57, 91, 27, 124, 155, 13)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__21 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__21_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "tacticAssumption'"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__22 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__22_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__23_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__23_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__23_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__4_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__23_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__22_value),LEAN_SCALAR_PTR_LITERAL(107, 131, 17, 59, 169, 254, 12, 140)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__23 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__23_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "induction"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__24 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__24_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__25_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__25_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__25_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__25_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__25_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__4_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__25_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__24_value),LEAN_SCALAR_PTR_LITERAL(231, 196, 247, 144, 178, 6, 178, 16)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__25 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__25_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "cases"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__26 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__26_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__27_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__27_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__27_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__27_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__27_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__4_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__27_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__26_value),LEAN_SCALAR_PTR_LITERAL(197, 49, 98, 208, 150, 151, 163, 74)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__27 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__27_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "intros"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__28 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__28_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__29_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__29_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__29_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__29_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__29_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__4_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__29_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__28_value),LEAN_SCALAR_PTR_LITERAL(26, 175, 18, 116, 252, 50, 128, 45)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__29 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__29_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "skip"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__30 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__30_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__31_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__31_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__31_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__31_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__31_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__4_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__31_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__30_value),LEAN_SCALAR_PTR_LITERAL(244, 42, 145, 170, 145, 147, 228, 105)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__31 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__31_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticOn_goal-_=>_"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__32 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__32_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__33_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__9_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__33_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__33_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__4_value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__33_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__32_value),LEAN_SCALAR_PTR_LITERAL(243, 56, 227, 189, 147, 207, 104, 76)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__33 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__33_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSwap_var__,,"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__34 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__34_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__35_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__35_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__35_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__4_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__35_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__34_value),LEAN_SCALAR_PTR_LITERAL(78, 115, 87, 127, 138, 34, 160, 62)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__35 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__35_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "tacticRepeat_"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__36 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__36_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__37_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__37_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__37_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__37_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__37_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__4_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__37_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__36_value),LEAN_SCALAR_PTR_LITERAL(149, 101, 42, 245, 144, 172, 68, 230)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__37 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__37_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "tacticTry_"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__38 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__38_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__39_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__39_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__39_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__39_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__39_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__4_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__39_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__38_value),LEAN_SCALAR_PTR_LITERAL(34, 109, 187, 155, 23, 130, 33, 152)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__39 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__39_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "paren"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__40 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__40_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__41_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__41_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__41_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__41_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__41_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__4_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__41_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__40_value),LEAN_SCALAR_PTR_LITERAL(117, 253, 122, 28, 77, 248, 149, 120)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__41 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__41_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Grind"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__42 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__42_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "focus"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__43 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__43_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__44_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__44_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__44_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__44_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__44_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__4_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__44_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__44_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__42_value),LEAN_SCALAR_PTR_LITERAL(148, 105, 19, 51, 118, 250, 248, 43)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__44_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__44_value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__43_value),LEAN_SCALAR_PTR_LITERAL(76, 68, 95, 128, 15, 244, 165, 139)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__44 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__44_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__45_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "next"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__45 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__45_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__46_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__46_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__46_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__46_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__46_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__4_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__46_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__46_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__42_value),LEAN_SCALAR_PTR_LITERAL(148, 105, 19, 51, 118, 250, 248, 43)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__46_value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__45_value),LEAN_SCALAR_PTR_LITERAL(122, 67, 127, 148, 132, 17, 131, 108)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__46 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__46_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__47_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__47_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__47_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__47_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__47_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__4_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__47_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__47_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__42_value),LEAN_SCALAR_PTR_LITERAL(148, 105, 19, 51, 118, 250, 248, 43)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__47_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__47_value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__26_value),LEAN_SCALAR_PTR_LITERAL(255, 233, 158, 17, 45, 135, 214, 137)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__47 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__47_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__48_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "tacticSwap"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__48 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__48_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__49_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__9_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__49_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__49_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__4_value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__49_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__49_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__48_value),LEAN_SCALAR_PTR_LITERAL(131, 195, 160, 50, 159, 119, 204, 253)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__49 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__49_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__50_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "rotateLeft"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__50 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__50_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__51_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__51_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__51_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__51_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__51_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__4_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__51_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__51_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__50_value),LEAN_SCALAR_PTR_LITERAL(63, 201, 198, 124, 10, 198, 250, 123)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__51 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__51_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__52_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "rotateRight"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__52 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__52_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__53_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__53_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__53_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__53_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__53_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__4_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__53_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__53_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__52_value),LEAN_SCALAR_PTR_LITERAL(98, 177, 153, 112, 69, 167, 66, 136)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__53 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__53_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__54_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "grindSeq1Indented"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__54 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__54_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__55_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__55_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__55_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__55_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__55_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__4_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__55_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__55_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__42_value),LEAN_SCALAR_PTR_LITERAL(148, 105, 19, 51, 118, 250, 248, 43)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__55_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__55_value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__54_value),LEAN_SCALAR_PTR_LITERAL(35, 114, 22, 139, 17, 175, 241, 184)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__55 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__55_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__56_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "grindSeq"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__56 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__56_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__57_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__57_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__57_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__57_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__57_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__4_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__57_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__57_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__42_value),LEAN_SCALAR_PTR_LITERAL(148, 105, 19, 51, 118, 250, 248, 43)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__57_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__57_value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__56_value),LEAN_SCALAR_PTR_LITERAL(158, 229, 98, 59, 247, 194, 34, 174)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__57 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__57_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__58_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 7, .m_data = "grind·_"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__58 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__58_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__59_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__59_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__59_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__59_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__59_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__4_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__59_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__59_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__42_value),LEAN_SCALAR_PTR_LITERAL(148, 105, 19, 51, 118, 250, 248, 43)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__59_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__59_value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__58_value),LEAN_SCALAR_PTR_LITERAL(27, 208, 22, 131, 194, 122, 241, 171)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__59 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__59_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__60_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "grindSeqBracketed"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__60 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__60_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__61_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__61_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__61_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__61_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__61_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__4_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__61_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__61_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__42_value),LEAN_SCALAR_PTR_LITERAL(148, 105, 19, 51, 118, 250, 248, 43)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__61_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__61_value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__60_value),LEAN_SCALAR_PTR_LITERAL(218, 10, 37, 72, 63, 213, 137, 199)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__61 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__61_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__62_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "grind_<;>_"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__62 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__62_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__63_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__63_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__63_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__63_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__63_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__4_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__63_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__63_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__42_value),LEAN_SCALAR_PTR_LITERAL(148, 105, 19, 51, 118, 250, 248, 43)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__63_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__63_value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__62_value),LEAN_SCALAR_PTR_LITERAL(104, 7, 229, 204, 205, 179, 221, 240)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__63 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__63_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__64_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__64_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__64_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__64_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__64_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__4_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__64_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__64_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__42_value),LEAN_SCALAR_PTR_LITERAL(148, 105, 19, 51, 118, 250, 248, 43)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__64_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__64_value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__30_value),LEAN_SCALAR_PTR_LITERAL(206, 95, 123, 110, 162, 109, 248, 53)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__64 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__64_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__65_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "else"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__65 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__65_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__66_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__65_value),LEAN_SCALAR_PTR_LITERAL(205, 140, 41, 106, 106, 114, 66, 206)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__66 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__66_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__67_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "tacticNext_=>_"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__67 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__67_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__68_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__68_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__68_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__68_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__68_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__4_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__68_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__68_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__67_value),LEAN_SCALAR_PTR_LITERAL(90, 21, 53, 2, 17, 158, 67, 66)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__68 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__68_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__69_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__69 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__69_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__70_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__70_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__70_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__70_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__70_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__4_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__70_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__70_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__69_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__70 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__70_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__71_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__71 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__71_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__72_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__72_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__72_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__72_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__72_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__4_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__72_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__72_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__71_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__72 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__72_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__73_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__43_value),LEAN_SCALAR_PTR_LITERAL(175, 11, 46, 34, 110, 2, 85, 154)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__73 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__73_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__74_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__74_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__74_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__74_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__74_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__4_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__74_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__74_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__43_value),LEAN_SCALAR_PTR_LITERAL(198, 223, 207, 6, 131, 57, 182, 221)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__74 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__74_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__75_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "<;>"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__75 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__75_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__76_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__75_value),LEAN_SCALAR_PTR_LITERAL(142, 129, 147, 41, 246, 11, 224, 230)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__76 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__76_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__77_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "tactic_<;>_"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__77 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__77_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__78_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__78_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__78_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__78_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__78_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__4_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__78_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__78_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__77_value),LEAN_SCALAR_PTR_LITERAL(31, 118, 44, 159, 195, 11, 47, 176)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__78 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__78_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__79_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "{"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__79 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__79_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__80_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__79_value),LEAN_SCALAR_PTR_LITERAL(111, 179, 85, 121, 215, 224, 31, 91)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__80 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__80_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__81_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__81 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__81_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__82_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__81_value),LEAN_SCALAR_PTR_LITERAL(238, 60, 149, 138, 55, 149, 59, 229)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__82 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__82_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__83_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__83 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__83_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__84_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__83_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__84 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__84_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__85_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "then"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__85 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__85_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__86_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__85_value),LEAN_SCALAR_PTR_LITERAL(20, 94, 39, 114, 80, 133, 157, 222)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__86 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__86_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__87_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__87 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__87_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__88_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "cdot"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__88 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__88_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__89_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__89_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__89_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__89_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__89_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__87_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__89_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__89_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__88_value),LEAN_SCALAR_PTR_LITERAL(215, 94, 65, 66, 49, 100, 151, 85)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__89 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__89_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__90_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__90_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__90_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__88_value),LEAN_SCALAR_PTR_LITERAL(238, 151, 138, 49, 249, 18, 254, 242)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__90 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__90_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__91_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "cdotTk"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__91 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__91_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__92_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__92_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__92_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__91_value),LEAN_SCALAR_PTR_LITERAL(117, 126, 44, 217, 38, 3, 69, 145)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__92 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__92_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__93_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeqBracketed"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__93 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__93_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__94_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__94_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__94_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__94_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__94_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__4_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__94_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__94_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__93_value),LEAN_SCALAR_PTR_LITERAL(142, 80, 121, 250, 245, 54, 71, 145)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__94 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__94_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__95_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ";"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__95 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__95_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__96_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__95_value),LEAN_SCALAR_PTR_LITERAL(101, 218, 47, 72, 64, 31, 83, 55)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__96 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__96_value;
static const lean_array_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__97_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*47, .m_other = 0, .m_tag = 246}, .m_size = 47, .m_capacity = 47, .m_data = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__89_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__90_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__92_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__94_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__96_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__76_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__78_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__80_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__82_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__84_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__86_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__66_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__68_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__70_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__72_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__73_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__74_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__55_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__57_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__59_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__61_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__63_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__64_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__44_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__46_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__47_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__49_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__51_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__53_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__31_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__33_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__35_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__37_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__39_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__41_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__19_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__21_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__23_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__25_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__27_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__29_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__6_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__8_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__11_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__13_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__15_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__17_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__97 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__97_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__98_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__98 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__98_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__99_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__1___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__99 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__99_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__100_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__2___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__100 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__100_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__101_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__3, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__101 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__101_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__102_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__4___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__102 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__102_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__103_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__5___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__103 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__103_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__104_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__6, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__104 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__104_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__105_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__98_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__99_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__105 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__105_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__106_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__105_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__100_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__101_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__102_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__103_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__106 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__106_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__107_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__106_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__104_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__107 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__107_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__108_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Array_instForIn_x27InferInstanceMembershipOfMonad___redArg___lam__0, .m_arity = 5, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__107_value)} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__108 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__108_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__109_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instForInOfForIn_x27___redArg___lam__1, .m_arity = 5, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__108_value)} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__109 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__109_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__110_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__110;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__111_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__111;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__112_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__112;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Conv"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "conv"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__2_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__2_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__4_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__2_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__2_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__0_value),LEAN_SCALAR_PTR_LITERAL(51, 212, 92, 235, 115, 8, 100, 36)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__2_value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__1_value),LEAN_SCALAR_PTR_LITERAL(192, 39, 103, 162, 58, 5, 181, 114)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "convLHS"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__4_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__4_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__0_value),LEAN_SCALAR_PTR_LITERAL(82, 42, 44, 190, 203, 225, 51, 226)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__4_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__3_value),LEAN_SCALAR_PTR_LITERAL(9, 199, 252, 210, 27, 127, 215, 31)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "convRHS"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__6_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__4_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__6_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__0_value),LEAN_SCALAR_PTR_LITERAL(82, 42, 44, 190, 203, 225, 51, 226)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__6_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__5_value),LEAN_SCALAR_PTR_LITERAL(69, 141, 229, 94, 0, 2, 204, 200)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__6_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "first"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__8_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__8_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__4_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__8_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__7_value),LEAN_SCALAR_PTR_LITERAL(59, 232, 35, 17, 172, 62, 48, 174)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__8_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "repeat'"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__10_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__10_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__10_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__10_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__4_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__10_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__9_value),LEAN_SCALAR_PTR_LITERAL(199, 67, 182, 138, 186, 187, 207, 59)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__10_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "tacticIterate____"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__12_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__12_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__12_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__12_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__12_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__4_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__12_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__11_value),LEAN_SCALAR_PTR_LITERAL(237, 47, 56, 55, 14, 165, 150, 141)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__12_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "anyGoals"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__13_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__14_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__14_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__14_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__14_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__4_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__14_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__13_value),LEAN_SCALAR_PTR_LITERAL(168, 19, 163, 3, 232, 106, 175, 32)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__14_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "allGoals"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__15_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__16_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__16_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__16_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__16_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__4_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__16_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__15_value),LEAN_SCALAR_PTR_LITERAL(105, 66, 138, 83, 251, 171, 29, 196)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__16_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "failIfSuccess"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__17_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__18_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__18_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__18_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__18_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__18_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__4_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__18_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__17_value),LEAN_SCALAR_PTR_LITERAL(227, 159, 155, 237, 20, 68, 221, 246)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__18 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__18_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__19_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__19_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__19_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__19_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__19_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__4_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__19_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__19_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__42_value),LEAN_SCALAR_PTR_LITERAL(148, 105, 19, 51, 118, 250, 248, 43)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__19_value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__13_value),LEAN_SCALAR_PTR_LITERAL(194, 3, 10, 44, 226, 136, 80, 42)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__19 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__19_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__20_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__20_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__20_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__20_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__20_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__4_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__20_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__20_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__42_value),LEAN_SCALAR_PTR_LITERAL(148, 105, 19, 51, 118, 250, 248, 43)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__20_value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__15_value),LEAN_SCALAR_PTR_LITERAL(131, 176, 42, 44, 172, 202, 38, 34)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__20 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__20_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__21_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__21_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__21_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__21_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__21_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__4_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__21_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__21_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__42_value),LEAN_SCALAR_PTR_LITERAL(148, 105, 19, 51, 118, 250, 248, 43)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__21_value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__7_value),LEAN_SCALAR_PTR_LITERAL(1, 60, 110, 192, 46, 198, 252, 25)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__21 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__21_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__22_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__22_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__22_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__22_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__22_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__4_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__22_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__22_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__42_value),LEAN_SCALAR_PTR_LITERAL(148, 105, 19, 51, 118, 250, 248, 43)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__22_value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__17_value),LEAN_SCALAR_PTR_LITERAL(9, 235, 219, 147, 187, 132, 195, 48)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__22 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__22_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "grindRepeat_"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__23 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__23_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__24_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__24_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__24_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__24_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__24_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__4_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__24_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__24_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__42_value),LEAN_SCALAR_PTR_LITERAL(148, 105, 19, 51, 118, 250, 248, 43)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__24_value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__23_value),LEAN_SCALAR_PTR_LITERAL(163, 93, 145, 161, 123, 119, 39, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__24 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__24_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "successIfFailWithMsg"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__25 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__25_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__26_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__26_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__26_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__4_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__26_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__25_value),LEAN_SCALAR_PTR_LITERAL(244, 58, 29, 249, 84, 193, 89, 5)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__26 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__26_value;
static const lean_array_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*16, .m_other = 0, .m_tag = 246}, .m_size = 16, .m_capacity = 16, .m_data = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__2_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__4_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__6_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__8_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__37_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__10_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__12_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__14_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__16_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__18_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__19_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__20_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__21_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__22_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__24_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__26_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__27 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__27_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__28_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__28;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch;
LEAN_EXPORT uint8_t lp_mathlib_List_elem___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_elem___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__2___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_filterTR_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_filterTR_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__1_spec__3___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__1_spec__3___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__5(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__0_spec__0_spec__1_spec__7_spec__12___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__0_spec__0_spec__1_spec__7___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__0_spec__0_spec__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__0_spec__1(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__4_spec__7_spec__9(size_t, size_t, lean_object*);
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals___closed__0;
static const lean_array_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals___closed__1_value;
static const lean_array_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapM___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__4_spec__8(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__4_spec__7(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_mapM___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__4(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapM___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__4_spec__8___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__4_spec__7_spec__9___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__1_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__0_spec__0_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__0_spec__0_spec__1_spec__7(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__0_spec__0_spec__1_spec__7_spec__12(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getInfoTrees___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getInfoTrees___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getInfoTrees___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getInfoTrees___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "(failed to pretty print)"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___lam__0___closed__0_value)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___lam__0___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___lam__0___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = " goals"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___lam__1___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___lam__1___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___lam__1___closed__0_value)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___lam__1___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___lam__1___closed__1_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "1 goal"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___lam__1___closed__2 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___lam__1___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___lam__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___lam__1___closed__2_value)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___lam__1___closed__3 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___lam__1___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___lam__1(lean_object*);
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__2;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__3;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__4;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4_spec__7___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4_spec__8(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4_spec__8___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4___lam__0___closed__0_value;
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4___lam__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 46, .m_capacity = 46, .m_length = 45, .m_data = "This linter can be disabled with `set_option "};
static const lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2___closed__0 = (const lean_object*)&lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2___closed__1;
static const lean_string_object lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = " false`"};
static const lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2___closed__2 = (const lean_object*)&lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "The following tactic starts with "};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__1;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = " and ends with "};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__2 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__2_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__3;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ", "};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__4 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__4_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__5;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = " of which "};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__6 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__6_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__7;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = " not operated on."};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__8 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__8_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__9;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 76, .m_capacity = 76, .m_length = 74, .m_data = "\nPlease focus on the current goal, for instance using `·` (typed as \"\\.\")."};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__10 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__10_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__11;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "are"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__12 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__12_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "is"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__13 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__13_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__4_spec__7_spec__11___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__4_spec__7_spec__11___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__4_spec__7_spec__11___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__4_spec__7_spec__11(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__4_spec__7_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__4_spec__7(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__4_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__4_spec__6_spec__9_spec__12___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__4_spec__6_spec__9_spec__12___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__4_spec__6_spec__9_spec__12___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__4_spec__6_spec__9_spec__12(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__4_spec__6_spec__9_spec__12___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__4_spec__6_spec__9(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__4_spec__6_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__4_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__4_spec__6_spec__8(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__4_spec__6_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__4_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___lam__0___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__0_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_withSetOptionIn___boxed, .m_arity = 6, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__0_value)} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__2_value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__4_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__4_value),LEAN_SCALAR_PTR_LITERAL(191, 70, 156, 159, 11, 54, 216, 94)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__5_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(37, 204, 154, 235, 250, 222, 148, 114)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__6_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Multigoal"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__6_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__7_value),LEAN_SCALAR_PTR_LITERAL(205, 246, 240, 81, 60, 210, 166, 119)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(56, 86, 166, 233, 207, 65, 227, 32)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__9_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(97, 166, 134, 163, 39, 110, 15, 14)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__10_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(59, 62, 231, 18, 147, 213, 30, 206)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__11_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Style"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__12_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__11_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__12_value),LEAN_SCALAR_PTR_LITERAL(79, 92, 28, 37, 135, 46, 104, 102)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__13_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__13_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(217, 57, 139, 127, 27, 72, 105, 24)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__14_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "multiGoalLinter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__15_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__14_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__15_value),LEAN_SCALAR_PTR_LITERAL(242, 42, 19, 255, 107, 241, 175, 209)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__16_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__16_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__17_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___closed__17_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4_spec__7(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_initFn_00___x40_Mathlib_Tactic_Linter_Multigoal_276707340____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_initFn_00___x40_Mathlib_Tactic_Linter_Multigoal_276707340____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4__spec__0(lean_object* v_name_1_, lean_object* v_decl_2_, lean_object* v_ref_3_){
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4__spec__0___boxed(lean_object* v_name_29_, lean_object* v_decl_30_, lean_object* v_ref_31_, lean_object* v_a_32_){
_start:
{
lean_object* v_res_33_; 
v_res_33_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4__spec__0(v_name_29_, v_decl_30_, v_ref_31_);
lean_dec_ref(v_decl_30_);
return v_res_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_56_; lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v___x_59_; 
v___x_56_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4_));
v___x_57_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4_));
v___x_58_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4_));
v___x_59_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4__spec__0(v___x_56_, v___x_57_, v___x_58_);
return v___x_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4____boxed(lean_object* v_a_60_){
_start:
{
lean_object* v_res_61_; 
v_res_61_ = lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4_();
return v_res_61_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__110(void){
_start:
{
lean_object* v___x_436_; lean_object* v___x_437_; lean_object* v___x_438_; 
v___x_436_ = lean_box(0);
v___x_437_ = lean_unsigned_to_nat(16u);
v___x_438_ = lean_mk_array(v___x_437_, v___x_436_);
return v___x_438_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__111(void){
_start:
{
lean_object* v___x_439_; lean_object* v___x_440_; lean_object* v___x_441_; 
v___x_439_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__110, &lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__110_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__110);
v___x_440_ = lean_unsigned_to_nat(0u);
v___x_441_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_441_, 0, v___x_440_);
lean_ctor_set(v___x_441_, 1, v___x_439_);
return v___x_441_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__112(void){
_start:
{
lean_object* v___x_442_; lean_object* v___x_443_; lean_object* v___x_444_; lean_object* v___x_445_; lean_object* v___f_446_; lean_object* v___x_447_; 
v___x_442_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__97));
v___x_443_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__111, &lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__111_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__111);
v___x_444_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__1));
v___x_445_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__0));
v___f_446_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__109));
v___x_447_ = l_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___redArg(v___f_446_, v___x_445_, v___x_444_, v___x_443_, v___x_442_);
return v___x_447_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions(void){
_start:
{
lean_object* v___x_448_; 
v___x_448_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__112, &lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__112_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__112);
return v___x_448_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__28(void){
_start:
{
lean_object* v___x_575_; lean_object* v___x_576_; lean_object* v___x_577_; lean_object* v___x_578_; lean_object* v___f_579_; lean_object* v___x_580_; 
v___x_575_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__27));
v___x_576_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__111, &lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__111_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__111);
v___x_577_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__1));
v___x_578_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__0));
v___f_579_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__109));
v___x_580_ = l_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___redArg(v___f_579_, v___x_578_, v___x_577_, v___x_576_, v___x_575_);
return v___x_580_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch(void){
_start:
{
lean_object* v___x_581_; 
v___x_581_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__28, &lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__28_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__28);
return v___x_581_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_elem___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__2(lean_object* v_a_582_, lean_object* v_x_583_){
_start:
{
if (lean_obj_tag(v_x_583_) == 0)
{
uint8_t v___x_584_; 
v___x_584_ = 0;
return v___x_584_;
}
else
{
lean_object* v_head_585_; lean_object* v_tail_586_; uint8_t v___x_587_; 
v_head_585_ = lean_ctor_get(v_x_583_, 0);
v_tail_586_ = lean_ctor_get(v_x_583_, 1);
v___x_587_ = l_Lean_instBEqMVarId_beq(v_a_582_, v_head_585_);
if (v___x_587_ == 0)
{
v_x_583_ = v_tail_586_;
goto _start;
}
else
{
return v___x_587_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_elem___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__2___boxed(lean_object* v_a_589_, lean_object* v_x_590_){
_start:
{
uint8_t v_res_591_; lean_object* v_r_592_; 
v_res_591_ = lp_mathlib_List_elem___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__2(v_a_589_, v_x_590_);
lean_dec(v_x_590_);
lean_dec(v_a_589_);
v_r_592_ = lean_box(v_res_591_);
return v_r_592_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_filterTR_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__3(lean_object* v___x_593_, lean_object* v_a_594_, lean_object* v_a_595_){
_start:
{
if (lean_obj_tag(v_a_594_) == 0)
{
lean_object* v___x_596_; 
v___x_596_ = l_List_reverse___redArg(v_a_595_);
return v___x_596_;
}
else
{
lean_object* v_head_597_; lean_object* v_tail_598_; lean_object* v___x_600_; uint8_t v_isShared_601_; uint8_t v_isSharedCheck_608_; 
v_head_597_ = lean_ctor_get(v_a_594_, 0);
v_tail_598_ = lean_ctor_get(v_a_594_, 1);
v_isSharedCheck_608_ = !lean_is_exclusive(v_a_594_);
if (v_isSharedCheck_608_ == 0)
{
v___x_600_ = v_a_594_;
v_isShared_601_ = v_isSharedCheck_608_;
goto v_resetjp_599_;
}
else
{
lean_inc(v_tail_598_);
lean_inc(v_head_597_);
lean_dec(v_a_594_);
v___x_600_ = lean_box(0);
v_isShared_601_ = v_isSharedCheck_608_;
goto v_resetjp_599_;
}
v_resetjp_599_:
{
uint8_t v___x_602_; 
v___x_602_ = lp_mathlib_List_elem___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__2(v_head_597_, v___x_593_);
if (v___x_602_ == 0)
{
lean_del_object(v___x_600_);
lean_dec(v_head_597_);
v_a_594_ = v_tail_598_;
goto _start;
}
else
{
lean_object* v___x_605_; 
if (v_isShared_601_ == 0)
{
lean_ctor_set(v___x_600_, 1, v_a_595_);
v___x_605_ = v___x_600_;
goto v_reusejp_604_;
}
else
{
lean_object* v_reuseFailAlloc_607_; 
v_reuseFailAlloc_607_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_607_, 0, v_head_597_);
lean_ctor_set(v_reuseFailAlloc_607_, 1, v_a_595_);
v___x_605_ = v_reuseFailAlloc_607_;
goto v_reusejp_604_;
}
v_reusejp_604_:
{
v_a_594_ = v_tail_598_;
v_a_595_ = v___x_605_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_filterTR_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__3___boxed(lean_object* v___x_609_, lean_object* v_a_610_, lean_object* v_a_611_){
_start:
{
lean_object* v_res_612_; 
v_res_612_ = lp_mathlib_List_filterTR_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__3(v___x_609_, v_a_610_, v_a_611_);
lean_dec(v___x_609_);
return v_res_612_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__1_spec__3___redArg(lean_object* v_a_613_, lean_object* v_x_614_){
_start:
{
if (lean_obj_tag(v_x_614_) == 0)
{
uint8_t v___x_615_; 
v___x_615_ = 0;
return v___x_615_;
}
else
{
lean_object* v_key_616_; lean_object* v_tail_617_; uint8_t v___x_618_; 
v_key_616_ = lean_ctor_get(v_x_614_, 0);
v_tail_617_ = lean_ctor_get(v_x_614_, 2);
v___x_618_ = lean_name_eq(v_key_616_, v_a_613_);
if (v___x_618_ == 0)
{
v_x_614_ = v_tail_617_;
goto _start;
}
else
{
return v___x_618_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__1_spec__3___redArg___boxed(lean_object* v_a_620_, lean_object* v_x_621_){
_start:
{
uint8_t v_res_622_; lean_object* v_r_623_; 
v_res_622_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__1_spec__3___redArg(v_a_620_, v_x_621_);
lean_dec(v_x_621_);
lean_dec(v_a_620_);
v_r_623_ = lean_box(v_res_622_);
return v_r_623_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__1___redArg(lean_object* v_m_624_, lean_object* v_a_625_){
_start:
{
lean_object* v_buckets_626_; lean_object* v___x_627_; uint64_t v___y_629_; 
v_buckets_626_ = lean_ctor_get(v_m_624_, 1);
v___x_627_ = lean_array_get_size(v_buckets_626_);
if (lean_obj_tag(v_a_625_) == 0)
{
uint64_t v___x_643_; 
v___x_643_ = 1723ULL;
v___y_629_ = v___x_643_;
goto v___jp_628_;
}
else
{
uint64_t v_hash_644_; 
v_hash_644_ = lean_ctor_get_uint64(v_a_625_, sizeof(void*)*2);
v___y_629_ = v_hash_644_;
goto v___jp_628_;
}
v___jp_628_:
{
uint64_t v___x_630_; uint64_t v___x_631_; uint64_t v_fold_632_; uint64_t v___x_633_; uint64_t v___x_634_; uint64_t v___x_635_; size_t v___x_636_; size_t v___x_637_; size_t v___x_638_; size_t v___x_639_; size_t v___x_640_; lean_object* v___x_641_; uint8_t v___x_642_; 
v___x_630_ = 32ULL;
v___x_631_ = lean_uint64_shift_right(v___y_629_, v___x_630_);
v_fold_632_ = lean_uint64_xor(v___y_629_, v___x_631_);
v___x_633_ = 16ULL;
v___x_634_ = lean_uint64_shift_right(v_fold_632_, v___x_633_);
v___x_635_ = lean_uint64_xor(v_fold_632_, v___x_634_);
v___x_636_ = lean_uint64_to_usize(v___x_635_);
v___x_637_ = lean_usize_of_nat(v___x_627_);
v___x_638_ = ((size_t)1ULL);
v___x_639_ = lean_usize_sub(v___x_637_, v___x_638_);
v___x_640_ = lean_usize_land(v___x_636_, v___x_639_);
v___x_641_ = lean_array_uget_borrowed(v_buckets_626_, v___x_640_);
v___x_642_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__1_spec__3___redArg(v_a_625_, v___x_641_);
return v___x_642_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__1___redArg___boxed(lean_object* v_m_645_, lean_object* v_a_646_){
_start:
{
uint8_t v_res_647_; lean_object* v_r_648_; 
v_res_647_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__1___redArg(v_m_645_, v_a_646_);
lean_dec(v_a_646_);
lean_dec_ref(v_m_645_);
v_r_648_ = lean_box(v_res_647_);
return v_r_648_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__5(lean_object* v_as_649_, size_t v_i_650_, size_t v_stop_651_, lean_object* v_b_652_){
_start:
{
uint8_t v___x_653_; 
v___x_653_ = lean_usize_dec_eq(v_i_650_, v_stop_651_);
if (v___x_653_ == 0)
{
lean_object* v___x_654_; lean_object* v___x_655_; size_t v___x_656_; size_t v___x_657_; 
v___x_654_ = lean_array_uget_borrowed(v_as_649_, v_i_650_);
v___x_655_ = l_Array_append___redArg(v_b_652_, v___x_654_);
v___x_656_ = ((size_t)1ULL);
v___x_657_ = lean_usize_add(v_i_650_, v___x_656_);
v_i_650_ = v___x_657_;
v_b_652_ = v___x_655_;
goto _start;
}
else
{
return v_b_652_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__5___boxed(lean_object* v_as_659_, lean_object* v_i_660_, lean_object* v_stop_661_, lean_object* v_b_662_){
_start:
{
size_t v_i_boxed_663_; size_t v_stop_boxed_664_; lean_object* v_res_665_; 
v_i_boxed_663_ = lean_unbox_usize(v_i_660_);
lean_dec(v_i_660_);
v_stop_boxed_664_ = lean_unbox_usize(v_stop_661_);
lean_dec(v_stop_661_);
v_res_665_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__5(v_as_659_, v_i_boxed_663_, v_stop_boxed_664_, v_b_662_);
lean_dec_ref(v_as_659_);
return v_res_665_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__0_spec__0_spec__1_spec__7_spec__12___redArg(lean_object* v_x_666_, lean_object* v_x_667_){
_start:
{
if (lean_obj_tag(v_x_667_) == 0)
{
return v_x_666_;
}
else
{
lean_object* v_key_668_; lean_object* v_value_669_; lean_object* v_tail_670_; lean_object* v___x_672_; uint8_t v_isShared_673_; uint8_t v_isSharedCheck_696_; 
v_key_668_ = lean_ctor_get(v_x_667_, 0);
v_value_669_ = lean_ctor_get(v_x_667_, 1);
v_tail_670_ = lean_ctor_get(v_x_667_, 2);
v_isSharedCheck_696_ = !lean_is_exclusive(v_x_667_);
if (v_isSharedCheck_696_ == 0)
{
v___x_672_ = v_x_667_;
v_isShared_673_ = v_isSharedCheck_696_;
goto v_resetjp_671_;
}
else
{
lean_inc(v_tail_670_);
lean_inc(v_value_669_);
lean_inc(v_key_668_);
lean_dec(v_x_667_);
v___x_672_ = lean_box(0);
v_isShared_673_ = v_isSharedCheck_696_;
goto v_resetjp_671_;
}
v_resetjp_671_:
{
lean_object* v___x_674_; uint64_t v___y_676_; 
v___x_674_ = lean_array_get_size(v_x_666_);
if (lean_obj_tag(v_key_668_) == 0)
{
uint64_t v___x_694_; 
v___x_694_ = 1723ULL;
v___y_676_ = v___x_694_;
goto v___jp_675_;
}
else
{
uint64_t v_hash_695_; 
v_hash_695_ = lean_ctor_get_uint64(v_key_668_, sizeof(void*)*2);
v___y_676_ = v_hash_695_;
goto v___jp_675_;
}
v___jp_675_:
{
uint64_t v___x_677_; uint64_t v___x_678_; uint64_t v_fold_679_; uint64_t v___x_680_; uint64_t v___x_681_; uint64_t v___x_682_; size_t v___x_683_; size_t v___x_684_; size_t v___x_685_; size_t v___x_686_; size_t v___x_687_; lean_object* v___x_688_; lean_object* v___x_690_; 
v___x_677_ = 32ULL;
v___x_678_ = lean_uint64_shift_right(v___y_676_, v___x_677_);
v_fold_679_ = lean_uint64_xor(v___y_676_, v___x_678_);
v___x_680_ = 16ULL;
v___x_681_ = lean_uint64_shift_right(v_fold_679_, v___x_680_);
v___x_682_ = lean_uint64_xor(v_fold_679_, v___x_681_);
v___x_683_ = lean_uint64_to_usize(v___x_682_);
v___x_684_ = lean_usize_of_nat(v___x_674_);
v___x_685_ = ((size_t)1ULL);
v___x_686_ = lean_usize_sub(v___x_684_, v___x_685_);
v___x_687_ = lean_usize_land(v___x_683_, v___x_686_);
v___x_688_ = lean_array_uget_borrowed(v_x_666_, v___x_687_);
lean_inc(v___x_688_);
if (v_isShared_673_ == 0)
{
lean_ctor_set(v___x_672_, 2, v___x_688_);
v___x_690_ = v___x_672_;
goto v_reusejp_689_;
}
else
{
lean_object* v_reuseFailAlloc_693_; 
v_reuseFailAlloc_693_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_693_, 0, v_key_668_);
lean_ctor_set(v_reuseFailAlloc_693_, 1, v_value_669_);
lean_ctor_set(v_reuseFailAlloc_693_, 2, v___x_688_);
v___x_690_ = v_reuseFailAlloc_693_;
goto v_reusejp_689_;
}
v_reusejp_689_:
{
lean_object* v___x_691_; 
v___x_691_ = lean_array_uset(v_x_666_, v___x_687_, v___x_690_);
v_x_666_ = v___x_691_;
v_x_667_ = v_tail_670_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__0_spec__0_spec__1_spec__7___redArg(lean_object* v_i_697_, lean_object* v_source_698_, lean_object* v_target_699_){
_start:
{
lean_object* v___x_700_; uint8_t v___x_701_; 
v___x_700_ = lean_array_get_size(v_source_698_);
v___x_701_ = lean_nat_dec_lt(v_i_697_, v___x_700_);
if (v___x_701_ == 0)
{
lean_dec_ref(v_source_698_);
lean_dec(v_i_697_);
return v_target_699_;
}
else
{
lean_object* v_es_702_; lean_object* v___x_703_; lean_object* v_source_704_; lean_object* v_target_705_; lean_object* v___x_706_; lean_object* v___x_707_; 
v_es_702_ = lean_array_fget(v_source_698_, v_i_697_);
v___x_703_ = lean_box(0);
v_source_704_ = lean_array_fset(v_source_698_, v_i_697_, v___x_703_);
v_target_705_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__0_spec__0_spec__1_spec__7_spec__12___redArg(v_target_699_, v_es_702_);
v___x_706_ = lean_unsigned_to_nat(1u);
v___x_707_ = lean_nat_add(v_i_697_, v___x_706_);
lean_dec(v_i_697_);
v_i_697_ = v___x_707_;
v_source_698_ = v_source_704_;
v_target_699_ = v_target_705_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__0_spec__0_spec__1___redArg(lean_object* v_data_709_){
_start:
{
lean_object* v___x_710_; lean_object* v___x_711_; lean_object* v_nbuckets_712_; lean_object* v___x_713_; lean_object* v___x_714_; lean_object* v___x_715_; lean_object* v___x_716_; 
v___x_710_ = lean_array_get_size(v_data_709_);
v___x_711_ = lean_unsigned_to_nat(2u);
v_nbuckets_712_ = lean_nat_mul(v___x_710_, v___x_711_);
v___x_713_ = lean_unsigned_to_nat(0u);
v___x_714_ = lean_box(0);
v___x_715_ = lean_mk_array(v_nbuckets_712_, v___x_714_);
v___x_716_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__0_spec__0_spec__1_spec__7___redArg(v___x_713_, v_data_709_, v___x_715_);
return v___x_716_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__0_spec__0___redArg(lean_object* v_m_717_, lean_object* v_a_718_, lean_object* v_b_719_){
_start:
{
lean_object* v_size_720_; lean_object* v_buckets_721_; lean_object* v___x_722_; uint64_t v___y_724_; 
v_size_720_ = lean_ctor_get(v_m_717_, 0);
v_buckets_721_ = lean_ctor_get(v_m_717_, 1);
v___x_722_ = lean_array_get_size(v_buckets_721_);
if (lean_obj_tag(v_a_718_) == 0)
{
uint64_t v___x_761_; 
v___x_761_ = 1723ULL;
v___y_724_ = v___x_761_;
goto v___jp_723_;
}
else
{
uint64_t v_hash_762_; 
v_hash_762_ = lean_ctor_get_uint64(v_a_718_, sizeof(void*)*2);
v___y_724_ = v_hash_762_;
goto v___jp_723_;
}
v___jp_723_:
{
uint64_t v___x_725_; uint64_t v___x_726_; uint64_t v_fold_727_; uint64_t v___x_728_; uint64_t v___x_729_; uint64_t v___x_730_; size_t v___x_731_; size_t v___x_732_; size_t v___x_733_; size_t v___x_734_; size_t v___x_735_; lean_object* v_bkt_736_; uint8_t v___x_737_; 
v___x_725_ = 32ULL;
v___x_726_ = lean_uint64_shift_right(v___y_724_, v___x_725_);
v_fold_727_ = lean_uint64_xor(v___y_724_, v___x_726_);
v___x_728_ = 16ULL;
v___x_729_ = lean_uint64_shift_right(v_fold_727_, v___x_728_);
v___x_730_ = lean_uint64_xor(v_fold_727_, v___x_729_);
v___x_731_ = lean_uint64_to_usize(v___x_730_);
v___x_732_ = lean_usize_of_nat(v___x_722_);
v___x_733_ = ((size_t)1ULL);
v___x_734_ = lean_usize_sub(v___x_732_, v___x_733_);
v___x_735_ = lean_usize_land(v___x_731_, v___x_734_);
v_bkt_736_ = lean_array_uget_borrowed(v_buckets_721_, v___x_735_);
v___x_737_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__1_spec__3___redArg(v_a_718_, v_bkt_736_);
if (v___x_737_ == 0)
{
lean_object* v___x_739_; uint8_t v_isShared_740_; uint8_t v_isSharedCheck_758_; 
lean_inc_ref(v_buckets_721_);
lean_inc(v_size_720_);
v_isSharedCheck_758_ = !lean_is_exclusive(v_m_717_);
if (v_isSharedCheck_758_ == 0)
{
lean_object* v_unused_759_; lean_object* v_unused_760_; 
v_unused_759_ = lean_ctor_get(v_m_717_, 1);
lean_dec(v_unused_759_);
v_unused_760_ = lean_ctor_get(v_m_717_, 0);
lean_dec(v_unused_760_);
v___x_739_ = v_m_717_;
v_isShared_740_ = v_isSharedCheck_758_;
goto v_resetjp_738_;
}
else
{
lean_dec(v_m_717_);
v___x_739_ = lean_box(0);
v_isShared_740_ = v_isSharedCheck_758_;
goto v_resetjp_738_;
}
v_resetjp_738_:
{
lean_object* v___x_741_; lean_object* v_size_x27_742_; lean_object* v___x_743_; lean_object* v_buckets_x27_744_; lean_object* v___x_745_; lean_object* v___x_746_; lean_object* v___x_747_; lean_object* v___x_748_; lean_object* v___x_749_; uint8_t v___x_750_; 
v___x_741_ = lean_unsigned_to_nat(1u);
v_size_x27_742_ = lean_nat_add(v_size_720_, v___x_741_);
lean_dec(v_size_720_);
lean_inc(v_bkt_736_);
v___x_743_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_743_, 0, v_a_718_);
lean_ctor_set(v___x_743_, 1, v_b_719_);
lean_ctor_set(v___x_743_, 2, v_bkt_736_);
v_buckets_x27_744_ = lean_array_uset(v_buckets_721_, v___x_735_, v___x_743_);
v___x_745_ = lean_unsigned_to_nat(4u);
v___x_746_ = lean_nat_mul(v_size_x27_742_, v___x_745_);
v___x_747_ = lean_unsigned_to_nat(3u);
v___x_748_ = lean_nat_div(v___x_746_, v___x_747_);
lean_dec(v___x_746_);
v___x_749_ = lean_array_get_size(v_buckets_x27_744_);
v___x_750_ = lean_nat_dec_le(v___x_748_, v___x_749_);
lean_dec(v___x_748_);
if (v___x_750_ == 0)
{
lean_object* v_val_751_; lean_object* v___x_753_; 
v_val_751_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__0_spec__0_spec__1___redArg(v_buckets_x27_744_);
if (v_isShared_740_ == 0)
{
lean_ctor_set(v___x_739_, 1, v_val_751_);
lean_ctor_set(v___x_739_, 0, v_size_x27_742_);
v___x_753_ = v___x_739_;
goto v_reusejp_752_;
}
else
{
lean_object* v_reuseFailAlloc_754_; 
v_reuseFailAlloc_754_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_754_, 0, v_size_x27_742_);
lean_ctor_set(v_reuseFailAlloc_754_, 1, v_val_751_);
v___x_753_ = v_reuseFailAlloc_754_;
goto v_reusejp_752_;
}
v_reusejp_752_:
{
return v___x_753_;
}
}
else
{
lean_object* v___x_756_; 
if (v_isShared_740_ == 0)
{
lean_ctor_set(v___x_739_, 1, v_buckets_x27_744_);
lean_ctor_set(v___x_739_, 0, v_size_x27_742_);
v___x_756_ = v___x_739_;
goto v_reusejp_755_;
}
else
{
lean_object* v_reuseFailAlloc_757_; 
v_reuseFailAlloc_757_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_757_, 0, v_size_x27_742_);
lean_ctor_set(v_reuseFailAlloc_757_, 1, v_buckets_x27_744_);
v___x_756_ = v_reuseFailAlloc_757_;
goto v_reusejp_755_;
}
v_reusejp_755_:
{
return v___x_756_;
}
}
}
}
else
{
lean_dec(v_b_719_);
lean_dec(v_a_718_);
return v_m_717_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__0_spec__1(lean_object* v_as_763_, size_t v_sz_764_, size_t v_i_765_, lean_object* v_b_766_){
_start:
{
uint8_t v___x_767_; 
v___x_767_ = lean_usize_dec_lt(v_i_765_, v_sz_764_);
if (v___x_767_ == 0)
{
return v_b_766_;
}
else
{
lean_object* v_a_768_; lean_object* v___x_769_; lean_object* v_r_770_; size_t v___x_771_; size_t v___x_772_; 
v_a_768_ = lean_array_uget_borrowed(v_as_763_, v_i_765_);
v___x_769_ = lean_box(0);
lean_inc(v_a_768_);
v_r_770_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__0_spec__0___redArg(v_b_766_, v_a_768_, v___x_769_);
v___x_771_ = ((size_t)1ULL);
v___x_772_ = lean_usize_add(v_i_765_, v___x_771_);
v_i_765_ = v___x_772_;
v_b_766_ = v_r_770_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__0_spec__1___boxed(lean_object* v_as_774_, lean_object* v_sz_775_, lean_object* v_i_776_, lean_object* v_b_777_){
_start:
{
size_t v_sz_boxed_778_; size_t v_i_boxed_779_; lean_object* v_res_780_; 
v_sz_boxed_778_ = lean_unbox_usize(v_sz_775_);
lean_dec(v_sz_775_);
v_i_boxed_779_ = lean_unbox_usize(v_i_776_);
lean_dec(v_i_776_);
v_res_780_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__0_spec__1(v_as_774_, v_sz_boxed_778_, v_i_boxed_779_, v_b_777_);
lean_dec_ref(v_as_774_);
return v_res_780_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__0(lean_object* v_m_781_, lean_object* v_l_782_){
_start:
{
size_t v_sz_783_; size_t v___x_784_; lean_object* v___x_785_; 
v_sz_783_ = lean_array_size(v_l_782_);
v___x_784_ = ((size_t)0ULL);
v___x_785_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__0_spec__1(v_l_782_, v_sz_783_, v___x_784_, v_m_781_);
return v___x_785_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__0___boxed(lean_object* v_m_786_, lean_object* v_l_787_){
_start:
{
lean_object* v_res_788_; 
v_res_788_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__0(v_m_786_, v_l_787_);
lean_dec_ref(v_l_787_);
return v_res_788_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__4_spec__7_spec__9(size_t v_sz_789_, size_t v_i_790_, lean_object* v_bs_791_){
_start:
{
uint8_t v___x_792_; 
v___x_792_ = lean_usize_dec_lt(v_i_790_, v_sz_789_);
if (v___x_792_ == 0)
{
return v_bs_791_;
}
else
{
lean_object* v_v_793_; lean_object* v___x_794_; lean_object* v_bs_x27_795_; lean_object* v___x_796_; size_t v___x_797_; size_t v___x_798_; lean_object* v___x_799_; 
v_v_793_ = lean_array_uget(v_bs_791_, v_i_790_);
v___x_794_ = lean_unsigned_to_nat(0u);
v_bs_x27_795_ = lean_array_uset(v_bs_791_, v_i_790_, v___x_794_);
v___x_796_ = lp_mathlib_Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__4_spec__7(v_v_793_);
v___x_797_ = ((size_t)1ULL);
v___x_798_ = lean_usize_add(v_i_790_, v___x_797_);
v___x_799_ = lean_array_uset(v_bs_x27_795_, v_i_790_, v___x_796_);
v_i_790_ = v___x_798_;
v_bs_791_ = v___x_799_;
goto _start;
}
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals___closed__0(void){
_start:
{
lean_object* v___x_801_; lean_object* v___x_802_; lean_object* v___x_803_; 
v___x_801_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch___closed__27));
v___x_802_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__111, &lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__111_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__111);
v___x_803_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__0(v___x_802_, v___x_801_);
return v___x_803_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals(lean_object* v_x_808_){
_start:
{
lean_object* v___y_810_; lean_object* v___y_811_; lean_object* v___y_812_; lean_object* v___y_813_; lean_object* v___y_814_; lean_object* v___y_815_; lean_object* v___y_816_; lean_object* v___y_817_; lean_object* v___y_818_; lean_object* v___y_819_; lean_object* v___y_820_; lean_object* v___y_821_; lean_object* v___y_822_; lean_object* v___y_823_; uint8_t v___y_824_; 
switch(lean_obj_tag(v_x_808_))
{
case 1:
{
lean_object* v_i_966_; lean_object* v_children_967_; lean_object* v___y_969_; lean_object* v___x_992_; lean_object* v___x_993_; lean_object* v___x_994_; lean_object* v___x_995_; lean_object* v___x_996_; uint8_t v___x_997_; 
v_i_966_ = lean_ctor_get(v_x_808_, 0);
lean_inc_ref(v_i_966_);
v_children_967_ = lean_ctor_get(v_x_808_, 1);
lean_inc_ref(v_children_967_);
lean_dec_ref_known(v_x_808_, 2);
v___x_992_ = lp_mathlib_Lean_PersistentArray_mapM___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__4(v_children_967_);
v___x_993_ = l_Lean_PersistentArray_toArray___redArg(v___x_992_);
lean_dec_ref(v___x_992_);
v___x_994_ = lean_unsigned_to_nat(0u);
v___x_995_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals___closed__2));
v___x_996_ = lean_array_get_size(v___x_993_);
v___x_997_ = lean_nat_dec_lt(v___x_994_, v___x_996_);
if (v___x_997_ == 0)
{
lean_dec_ref(v___x_993_);
v___y_969_ = v___x_995_;
goto v___jp_968_;
}
else
{
uint8_t v___x_998_; 
v___x_998_ = lean_nat_dec_le(v___x_996_, v___x_996_);
if (v___x_998_ == 0)
{
if (v___x_997_ == 0)
{
lean_dec_ref(v___x_993_);
v___y_969_ = v___x_995_;
goto v___jp_968_;
}
else
{
size_t v___x_999_; size_t v___x_1000_; lean_object* v___x_1001_; 
v___x_999_ = ((size_t)0ULL);
v___x_1000_ = lean_usize_of_nat(v___x_996_);
v___x_1001_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__5(v___x_993_, v___x_999_, v___x_1000_, v___x_995_);
lean_dec_ref(v___x_993_);
v___y_969_ = v___x_1001_;
goto v___jp_968_;
}
}
else
{
size_t v___x_1002_; size_t v___x_1003_; lean_object* v___x_1004_; 
v___x_1002_ = ((size_t)0ULL);
v___x_1003_ = lean_usize_of_nat(v___x_996_);
v___x_1004_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__5(v___x_993_, v___x_1002_, v___x_1003_, v___x_995_);
lean_dec_ref(v___x_993_);
v___y_969_ = v___x_1004_;
goto v___jp_968_;
}
}
v___jp_968_:
{
if (lean_obj_tag(v_i_966_) == 0)
{
lean_object* v_i_970_; lean_object* v___x_971_; lean_object* v___x_972_; lean_object* v___x_973_; lean_object* v___x_974_; lean_object* v___x_975_; lean_object* v___x_976_; lean_object* v_toElabInfo_977_; lean_object* v_goalsBefore_978_; lean_object* v_goalsAfter_979_; lean_object* v_stx_980_; lean_object* v___x_981_; lean_object* v___x_982_; lean_object* v___x_983_; lean_object* v___x_984_; uint8_t v___x_985_; 
v_i_970_ = lean_ctor_get(v_i_966_, 0);
lean_inc_ref(v_i_970_);
lean_dec_ref_known(v_i_966_, 1);
v___x_971_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__2));
v___x_972_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__3));
v___x_973_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__4));
v___x_974_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4_));
v___x_975_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__37));
v___x_976_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__42));
v_toElabInfo_977_ = lean_ctor_get(v_i_970_, 0);
lean_inc_ref(v_toElabInfo_977_);
v_goalsBefore_978_ = lean_ctor_get(v_i_970_, 2);
lean_inc(v_goalsBefore_978_);
v_goalsAfter_979_ = lean_ctor_get(v_i_970_, 4);
lean_inc(v_goalsAfter_979_);
lean_dec_ref(v_i_970_);
v_stx_980_ = lean_ctor_get(v_toElabInfo_977_, 1);
lean_inc_n(v_stx_980_, 2);
lean_dec_ref(v_toElabInfo_977_);
v___x_981_ = lean_unsigned_to_nat(0u);
v___x_982_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__111, &lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__111_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__111);
v___x_983_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals___closed__0, &lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals___closed__0_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals___closed__0);
v___x_984_ = l_Lean_Syntax_getKind(v_stx_980_);
v___x_985_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__1___redArg(v___x_983_, v___x_984_);
if (v___x_985_ == 0)
{
lean_object* v___x_986_; lean_object* v___x_987_; uint8_t v___x_988_; 
v___x_986_ = l_List_lengthTR___redArg(v_goalsBefore_978_);
v___x_987_ = lean_unsigned_to_nat(1u);
v___x_988_ = lean_nat_dec_eq(v___x_986_, v___x_987_);
if (v___x_988_ == 0)
{
v___y_810_ = v___x_984_;
v___y_811_ = v_goalsAfter_979_;
v___y_812_ = v_goalsBefore_978_;
v___y_813_ = v___y_969_;
v___y_814_ = v___x_981_;
v___y_815_ = v___x_971_;
v___y_816_ = v_stx_980_;
v___y_817_ = v___x_976_;
v___y_818_ = v___x_982_;
v___y_819_ = v___x_974_;
v___y_820_ = v___x_972_;
v___y_821_ = v___x_975_;
v___y_822_ = v___x_986_;
v___y_823_ = v___x_973_;
v___y_824_ = v___x_988_;
goto v___jp_809_;
}
else
{
lean_object* v___x_989_; uint8_t v___x_990_; 
v___x_989_ = l_List_lengthTR___redArg(v_goalsAfter_979_);
v___x_990_ = lean_nat_dec_le(v___x_989_, v___x_987_);
lean_dec(v___x_989_);
v___y_810_ = v___x_984_;
v___y_811_ = v_goalsAfter_979_;
v___y_812_ = v_goalsBefore_978_;
v___y_813_ = v___y_969_;
v___y_814_ = v___x_981_;
v___y_815_ = v___x_971_;
v___y_816_ = v_stx_980_;
v___y_817_ = v___x_976_;
v___y_818_ = v___x_982_;
v___y_819_ = v___x_974_;
v___y_820_ = v___x_972_;
v___y_821_ = v___x_975_;
v___y_822_ = v___x_986_;
v___y_823_ = v___x_973_;
v___y_824_ = v___x_990_;
goto v___jp_809_;
}
}
else
{
lean_object* v___x_991_; 
lean_dec(v___x_984_);
lean_dec(v_stx_980_);
lean_dec(v_goalsAfter_979_);
lean_dec(v_goalsBefore_978_);
lean_dec_ref(v___y_969_);
v___x_991_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals___closed__1));
return v___x_991_;
}
}
else
{
lean_dec_ref(v_i_966_);
return v___y_969_;
}
}
}
case 0:
{
lean_object* v_t_1005_; 
v_t_1005_ = lean_ctor_get(v_x_808_, 1);
lean_inc_ref(v_t_1005_);
lean_dec_ref_known(v_x_808_, 2);
v_x_808_ = v_t_1005_;
goto _start;
}
default: 
{
lean_object* v___x_1007_; 
lean_dec_ref(v_x_808_);
v___x_1007_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals___closed__2));
return v___x_1007_;
}
}
v___jp_809_:
{
if (v___y_824_ == 0)
{
lean_object* v___x_825_; 
v___x_825_ = l_Lean_Syntax_getHeadInfo(v___y_816_);
if (lean_obj_tag(v___x_825_) == 0)
{
lean_object* v___x_826_; lean_object* v_backgroundGoals_827_; lean_object* v___x_828_; uint8_t v___x_829_; 
lean_dec_ref_known(v___x_825_, 4);
v___x_826_ = lean_box(0);
lean_inc(v___y_811_);
v_backgroundGoals_827_ = lp_mathlib_List_filterTR_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__3(v___y_812_, v___y_811_, v___x_826_);
lean_dec(v___y_812_);
v___x_828_ = l_List_lengthTR___redArg(v_backgroundGoals_827_);
lean_dec(v_backgroundGoals_827_);
v___x_829_ = lean_nat_dec_eq(v___x_828_, v___y_814_);
if (v___x_829_ == 0)
{
lean_object* v___x_830_; lean_object* v___x_831_; lean_object* v___x_832_; lean_object* v___x_833_; lean_object* v___x_834_; lean_object* v___x_835_; lean_object* v___x_836_; lean_object* v___x_837_; lean_object* v___x_838_; lean_object* v___x_839_; lean_object* v___x_840_; lean_object* v___x_841_; lean_object* v___x_842_; lean_object* v___x_843_; lean_object* v___x_844_; lean_object* v___x_845_; lean_object* v___x_846_; lean_object* v___x_847_; lean_object* v___x_848_; lean_object* v___x_849_; lean_object* v___x_850_; lean_object* v___x_851_; lean_object* v___x_852_; lean_object* v___x_853_; lean_object* v___x_854_; lean_object* v___x_855_; lean_object* v___x_856_; lean_object* v___x_857_; lean_object* v___x_858_; lean_object* v___x_859_; lean_object* v___x_860_; lean_object* v___x_861_; lean_object* v___x_862_; lean_object* v___x_863_; lean_object* v___x_864_; lean_object* v___x_865_; lean_object* v___x_866_; lean_object* v___x_867_; lean_object* v___x_868_; lean_object* v___x_869_; lean_object* v___x_870_; lean_object* v___x_871_; lean_object* v___x_872_; lean_object* v___x_873_; lean_object* v___x_874_; lean_object* v___x_875_; lean_object* v___x_876_; lean_object* v___x_877_; lean_object* v___x_878_; lean_object* v___x_879_; lean_object* v___x_880_; lean_object* v___x_881_; lean_object* v___x_882_; lean_object* v___x_883_; lean_object* v___x_884_; lean_object* v___x_885_; lean_object* v___x_886_; lean_object* v___x_887_; lean_object* v___x_888_; lean_object* v___x_889_; lean_object* v___x_890_; lean_object* v___x_891_; lean_object* v___x_892_; lean_object* v___x_893_; lean_object* v___x_894_; lean_object* v___x_895_; lean_object* v___x_896_; lean_object* v___x_897_; lean_object* v___x_898_; lean_object* v___x_899_; lean_object* v___x_900_; lean_object* v___x_901_; lean_object* v___x_902_; lean_object* v___x_903_; lean_object* v___x_904_; lean_object* v___x_905_; lean_object* v___x_906_; lean_object* v___x_907_; lean_object* v___x_908_; lean_object* v___x_909_; lean_object* v___x_910_; lean_object* v___x_911_; lean_object* v___x_912_; lean_object* v___x_913_; lean_object* v___x_914_; lean_object* v___x_915_; lean_object* v___x_916_; lean_object* v___x_917_; lean_object* v___x_918_; lean_object* v___x_919_; lean_object* v___x_920_; lean_object* v___x_921_; lean_object* v___x_922_; lean_object* v___x_923_; lean_object* v___x_924_; lean_object* v___x_925_; lean_object* v___x_926_; lean_object* v___x_927_; lean_object* v___x_928_; lean_object* v___x_929_; lean_object* v___x_930_; lean_object* v___x_931_; lean_object* v___x_932_; lean_object* v___x_933_; lean_object* v___x_934_; lean_object* v___x_935_; lean_object* v___x_936_; lean_object* v___x_937_; lean_object* v___x_938_; lean_object* v___x_939_; lean_object* v___x_940_; lean_object* v___x_941_; lean_object* v___x_942_; lean_object* v___x_943_; lean_object* v___x_944_; lean_object* v___x_945_; lean_object* v___x_946_; lean_object* v___x_947_; lean_object* v___x_948_; lean_object* v___x_949_; lean_object* v___x_950_; lean_object* v___x_951_; lean_object* v___x_952_; lean_object* v___x_953_; lean_object* v___x_954_; lean_object* v___x_955_; lean_object* v___x_956_; lean_object* v___x_957_; lean_object* v___x_958_; lean_object* v___x_959_; uint8_t v___x_960_; 
v___x_830_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__5));
lean_inc_ref_n(v___y_823_, 33);
lean_inc_ref_n(v___y_820_, 29);
lean_inc_ref_n(v___y_815_, 31);
v___x_831_ = l_Lean_Name_mkStr4(v___y_815_, v___y_820_, v___y_823_, v___x_830_);
v___x_832_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__7));
v___x_833_ = l_Lean_Name_mkStr4(v___y_815_, v___y_820_, v___y_823_, v___x_832_);
v___x_834_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__9));
v___x_835_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__10));
v___x_836_ = l_Lean_Name_mkStr3(v___x_834_, v___y_823_, v___x_835_);
v___x_837_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__12));
v___x_838_ = l_Lean_Name_mkStr4(v___y_815_, v___y_820_, v___y_823_, v___x_837_);
v___x_839_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__15));
v___x_840_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__17));
v___x_841_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__18));
v___x_842_ = l_Lean_Name_mkStr4(v___y_815_, v___y_820_, v___y_823_, v___x_841_);
v___x_843_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__20));
v___x_844_ = l_Lean_Name_mkStr4(v___y_815_, v___y_820_, v___y_823_, v___x_843_);
v___x_845_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__22));
lean_inc_ref_n(v___y_819_, 2);
v___x_846_ = l_Lean_Name_mkStr3(v___y_819_, v___y_823_, v___x_845_);
v___x_847_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__24));
v___x_848_ = l_Lean_Name_mkStr4(v___y_815_, v___y_820_, v___y_823_, v___x_847_);
v___x_849_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__26));
v___x_850_ = l_Lean_Name_mkStr4(v___y_815_, v___y_820_, v___y_823_, v___x_849_);
v___x_851_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__28));
v___x_852_ = l_Lean_Name_mkStr4(v___y_815_, v___y_820_, v___y_823_, v___x_851_);
v___x_853_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__30));
v___x_854_ = l_Lean_Name_mkStr4(v___y_815_, v___y_820_, v___y_823_, v___x_853_);
v___x_855_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__32));
v___x_856_ = l_Lean_Name_mkStr3(v___x_834_, v___y_823_, v___x_855_);
v___x_857_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__34));
v___x_858_ = l_Lean_Name_mkStr3(v___y_819_, v___y_823_, v___x_857_);
v___x_859_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__38));
v___x_860_ = l_Lean_Name_mkStr4(v___y_815_, v___y_820_, v___y_823_, v___x_859_);
v___x_861_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__40));
v___x_862_ = l_Lean_Name_mkStr4(v___y_815_, v___y_820_, v___y_823_, v___x_861_);
v___x_863_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__43));
lean_inc_ref_n(v___y_817_, 9);
v___x_864_ = l_Lean_Name_mkStr5(v___y_815_, v___y_820_, v___y_823_, v___y_817_, v___x_863_);
v___x_865_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__45));
v___x_866_ = l_Lean_Name_mkStr5(v___y_815_, v___y_820_, v___y_823_, v___y_817_, v___x_865_);
v___x_867_ = l_Lean_Name_mkStr5(v___y_815_, v___y_820_, v___y_823_, v___y_817_, v___x_849_);
v___x_868_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__48));
v___x_869_ = l_Lean_Name_mkStr3(v___x_834_, v___y_823_, v___x_868_);
v___x_870_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__50));
v___x_871_ = l_Lean_Name_mkStr4(v___y_815_, v___y_820_, v___y_823_, v___x_870_);
v___x_872_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__52));
v___x_873_ = l_Lean_Name_mkStr4(v___y_815_, v___y_820_, v___y_823_, v___x_872_);
v___x_874_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__54));
v___x_875_ = l_Lean_Name_mkStr5(v___y_815_, v___y_820_, v___y_823_, v___y_817_, v___x_874_);
v___x_876_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__56));
v___x_877_ = l_Lean_Name_mkStr5(v___y_815_, v___y_820_, v___y_823_, v___y_817_, v___x_876_);
v___x_878_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__58));
v___x_879_ = l_Lean_Name_mkStr5(v___y_815_, v___y_820_, v___y_823_, v___y_817_, v___x_878_);
v___x_880_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__60));
v___x_881_ = l_Lean_Name_mkStr5(v___y_815_, v___y_820_, v___y_823_, v___y_817_, v___x_880_);
v___x_882_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__62));
v___x_883_ = l_Lean_Name_mkStr5(v___y_815_, v___y_820_, v___y_823_, v___y_817_, v___x_882_);
v___x_884_ = l_Lean_Name_mkStr5(v___y_815_, v___y_820_, v___y_823_, v___y_817_, v___x_853_);
v___x_885_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__66));
v___x_886_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__67));
v___x_887_ = l_Lean_Name_mkStr4(v___y_815_, v___y_820_, v___y_823_, v___x_886_);
v___x_888_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__69));
v___x_889_ = l_Lean_Name_mkStr4(v___y_815_, v___y_820_, v___y_823_, v___x_888_);
v___x_890_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__71));
v___x_891_ = l_Lean_Name_mkStr4(v___y_815_, v___y_820_, v___y_823_, v___x_890_);
v___x_892_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__73));
v___x_893_ = l_Lean_Name_mkStr4(v___y_815_, v___y_820_, v___y_823_, v___x_863_);
v___x_894_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__76));
v___x_895_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__77));
v___x_896_ = l_Lean_Name_mkStr4(v___y_815_, v___y_820_, v___y_823_, v___x_895_);
v___x_897_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__80));
v___x_898_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__82));
v___x_899_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__84));
v___x_900_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__86));
v___x_901_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__87));
v___x_902_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__88));
v___x_903_ = l_Lean_Name_mkStr4(v___y_815_, v___y_820_, v___x_901_, v___x_902_);
v___x_904_ = l_Lean_Name_mkStr2(v___y_815_, v___x_902_);
v___x_905_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__91));
v___x_906_ = l_Lean_Name_mkStr2(v___y_815_, v___x_905_);
v___x_907_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__93));
v___x_908_ = l_Lean_Name_mkStr4(v___y_815_, v___y_820_, v___y_823_, v___x_907_);
v___x_909_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions___closed__96));
v___x_910_ = lean_unsigned_to_nat(47u);
v___x_911_ = lean_mk_empty_array_with_capacity(v___x_910_);
v___x_912_ = lean_array_push(v___x_911_, v___x_903_);
v___x_913_ = lean_array_push(v___x_912_, v___x_904_);
v___x_914_ = lean_array_push(v___x_913_, v___x_906_);
v___x_915_ = lean_array_push(v___x_914_, v___x_908_);
v___x_916_ = lean_array_push(v___x_915_, v___x_909_);
v___x_917_ = lean_array_push(v___x_916_, v___x_894_);
v___x_918_ = lean_array_push(v___x_917_, v___x_896_);
v___x_919_ = lean_array_push(v___x_918_, v___x_897_);
v___x_920_ = lean_array_push(v___x_919_, v___x_898_);
v___x_921_ = lean_array_push(v___x_920_, v___x_899_);
v___x_922_ = lean_array_push(v___x_921_, v___x_900_);
v___x_923_ = lean_array_push(v___x_922_, v___x_885_);
v___x_924_ = lean_array_push(v___x_923_, v___x_887_);
v___x_925_ = lean_array_push(v___x_924_, v___x_889_);
v___x_926_ = lean_array_push(v___x_925_, v___x_891_);
v___x_927_ = lean_array_push(v___x_926_, v___x_892_);
v___x_928_ = lean_array_push(v___x_927_, v___x_893_);
v___x_929_ = lean_array_push(v___x_928_, v___x_875_);
v___x_930_ = lean_array_push(v___x_929_, v___x_877_);
v___x_931_ = lean_array_push(v___x_930_, v___x_879_);
v___x_932_ = lean_array_push(v___x_931_, v___x_881_);
v___x_933_ = lean_array_push(v___x_932_, v___x_883_);
v___x_934_ = lean_array_push(v___x_933_, v___x_884_);
v___x_935_ = lean_array_push(v___x_934_, v___x_864_);
v___x_936_ = lean_array_push(v___x_935_, v___x_866_);
v___x_937_ = lean_array_push(v___x_936_, v___x_867_);
v___x_938_ = lean_array_push(v___x_937_, v___x_869_);
v___x_939_ = lean_array_push(v___x_938_, v___x_871_);
v___x_940_ = lean_array_push(v___x_939_, v___x_873_);
v___x_941_ = lean_array_push(v___x_940_, v___x_854_);
v___x_942_ = lean_array_push(v___x_941_, v___x_856_);
v___x_943_ = lean_array_push(v___x_942_, v___x_858_);
lean_inc(v___y_821_);
v___x_944_ = lean_array_push(v___x_943_, v___y_821_);
v___x_945_ = lean_array_push(v___x_944_, v___x_860_);
v___x_946_ = lean_array_push(v___x_945_, v___x_862_);
v___x_947_ = lean_array_push(v___x_946_, v___x_842_);
v___x_948_ = lean_array_push(v___x_947_, v___x_844_);
v___x_949_ = lean_array_push(v___x_948_, v___x_846_);
v___x_950_ = lean_array_push(v___x_949_, v___x_848_);
v___x_951_ = lean_array_push(v___x_950_, v___x_850_);
v___x_952_ = lean_array_push(v___x_951_, v___x_852_);
v___x_953_ = lean_array_push(v___x_952_, v___x_831_);
v___x_954_ = lean_array_push(v___x_953_, v___x_833_);
v___x_955_ = lean_array_push(v___x_954_, v___x_836_);
v___x_956_ = lean_array_push(v___x_955_, v___x_838_);
v___x_957_ = lean_array_push(v___x_956_, v___x_839_);
v___x_958_ = lean_array_push(v___x_957_, v___x_840_);
lean_inc_ref(v___y_818_);
v___x_959_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__0(v___y_818_, v___x_958_);
lean_dec_ref(v___x_958_);
v___x_960_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__1___redArg(v___x_959_, v___y_810_);
lean_dec(v___y_810_);
lean_dec_ref(v___x_959_);
if (v___x_960_ == 0)
{
lean_object* v___x_961_; lean_object* v___x_962_; lean_object* v___x_963_; lean_object* v___x_964_; lean_object* v___x_965_; 
v___x_961_ = l_List_lengthTR___redArg(v___y_811_);
lean_dec(v___y_811_);
v___x_962_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_962_, 0, v___x_961_);
lean_ctor_set(v___x_962_, 1, v___x_828_);
v___x_963_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_963_, 0, v___y_822_);
lean_ctor_set(v___x_963_, 1, v___x_962_);
v___x_964_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_964_, 0, v___y_816_);
lean_ctor_set(v___x_964_, 1, v___x_963_);
v___x_965_ = lean_array_push(v___y_813_, v___x_964_);
return v___x_965_;
}
else
{
lean_dec(v___x_828_);
lean_dec(v___y_822_);
lean_dec(v___y_816_);
lean_dec(v___y_811_);
return v___y_813_;
}
}
else
{
lean_dec(v___x_828_);
lean_dec(v___y_822_);
lean_dec(v___y_816_);
lean_dec(v___y_811_);
lean_dec(v___y_810_);
return v___y_813_;
}
}
else
{
lean_dec(v___x_825_);
lean_dec(v___y_822_);
lean_dec(v___y_816_);
lean_dec(v___y_812_);
lean_dec(v___y_811_);
lean_dec(v___y_810_);
return v___y_813_;
}
}
else
{
lean_dec(v___y_822_);
lean_dec(v___y_816_);
lean_dec(v___y_812_);
lean_dec(v___y_811_);
lean_dec(v___y_810_);
return v___y_813_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapM___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__4_spec__8(size_t v_sz_1008_, size_t v_i_1009_, lean_object* v_bs_1010_){
_start:
{
uint8_t v___x_1011_; 
v___x_1011_ = lean_usize_dec_lt(v_i_1009_, v_sz_1008_);
if (v___x_1011_ == 0)
{
return v_bs_1010_;
}
else
{
lean_object* v_v_1012_; lean_object* v___x_1013_; lean_object* v_bs_x27_1014_; lean_object* v___x_1015_; size_t v___x_1016_; size_t v___x_1017_; lean_object* v___x_1018_; 
v_v_1012_ = lean_array_uget(v_bs_1010_, v_i_1009_);
v___x_1013_ = lean_unsigned_to_nat(0u);
v_bs_x27_1014_ = lean_array_uset(v_bs_1010_, v_i_1009_, v___x_1013_);
v___x_1015_ = lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals(v_v_1012_);
v___x_1016_ = ((size_t)1ULL);
v___x_1017_ = lean_usize_add(v_i_1009_, v___x_1016_);
v___x_1018_ = lean_array_uset(v_bs_x27_1014_, v_i_1009_, v___x_1015_);
v_i_1009_ = v___x_1017_;
v_bs_1010_ = v___x_1018_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__4_spec__7(lean_object* v_x_1020_){
_start:
{
if (lean_obj_tag(v_x_1020_) == 0)
{
lean_object* v_cs_1021_; lean_object* v___x_1023_; uint8_t v_isShared_1024_; uint8_t v_isSharedCheck_1031_; 
v_cs_1021_ = lean_ctor_get(v_x_1020_, 0);
v_isSharedCheck_1031_ = !lean_is_exclusive(v_x_1020_);
if (v_isSharedCheck_1031_ == 0)
{
v___x_1023_ = v_x_1020_;
v_isShared_1024_ = v_isSharedCheck_1031_;
goto v_resetjp_1022_;
}
else
{
lean_inc(v_cs_1021_);
lean_dec(v_x_1020_);
v___x_1023_ = lean_box(0);
v_isShared_1024_ = v_isSharedCheck_1031_;
goto v_resetjp_1022_;
}
v_resetjp_1022_:
{
size_t v_sz_1025_; size_t v___x_1026_; lean_object* v___x_1027_; lean_object* v___x_1029_; 
v_sz_1025_ = lean_array_size(v_cs_1021_);
v___x_1026_ = ((size_t)0ULL);
v___x_1027_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__4_spec__7_spec__9(v_sz_1025_, v___x_1026_, v_cs_1021_);
if (v_isShared_1024_ == 0)
{
lean_ctor_set(v___x_1023_, 0, v___x_1027_);
v___x_1029_ = v___x_1023_;
goto v_reusejp_1028_;
}
else
{
lean_object* v_reuseFailAlloc_1030_; 
v_reuseFailAlloc_1030_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1030_, 0, v___x_1027_);
v___x_1029_ = v_reuseFailAlloc_1030_;
goto v_reusejp_1028_;
}
v_reusejp_1028_:
{
return v___x_1029_;
}
}
}
else
{
lean_object* v_vs_1032_; lean_object* v___x_1034_; uint8_t v_isShared_1035_; uint8_t v_isSharedCheck_1042_; 
v_vs_1032_ = lean_ctor_get(v_x_1020_, 0);
v_isSharedCheck_1042_ = !lean_is_exclusive(v_x_1020_);
if (v_isSharedCheck_1042_ == 0)
{
v___x_1034_ = v_x_1020_;
v_isShared_1035_ = v_isSharedCheck_1042_;
goto v_resetjp_1033_;
}
else
{
lean_inc(v_vs_1032_);
lean_dec(v_x_1020_);
v___x_1034_ = lean_box(0);
v_isShared_1035_ = v_isSharedCheck_1042_;
goto v_resetjp_1033_;
}
v_resetjp_1033_:
{
size_t v_sz_1036_; size_t v___x_1037_; lean_object* v___x_1038_; lean_object* v___x_1040_; 
v_sz_1036_ = lean_array_size(v_vs_1032_);
v___x_1037_ = ((size_t)0ULL);
v___x_1038_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapM___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__4_spec__8(v_sz_1036_, v___x_1037_, v_vs_1032_);
if (v_isShared_1035_ == 0)
{
lean_ctor_set(v___x_1034_, 0, v___x_1038_);
v___x_1040_ = v___x_1034_;
goto v_reusejp_1039_;
}
else
{
lean_object* v_reuseFailAlloc_1041_; 
v_reuseFailAlloc_1041_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1041_, 0, v___x_1038_);
v___x_1040_ = v_reuseFailAlloc_1041_;
goto v_reusejp_1039_;
}
v_reusejp_1039_:
{
return v___x_1040_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_mapM___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__4(lean_object* v_t_1043_){
_start:
{
lean_object* v_root_1044_; lean_object* v_tail_1045_; lean_object* v_size_1046_; size_t v_shift_1047_; lean_object* v_tailOff_1048_; lean_object* v___x_1050_; uint8_t v_isShared_1051_; uint8_t v_isSharedCheck_1059_; 
v_root_1044_ = lean_ctor_get(v_t_1043_, 0);
v_tail_1045_ = lean_ctor_get(v_t_1043_, 1);
v_size_1046_ = lean_ctor_get(v_t_1043_, 2);
v_shift_1047_ = lean_ctor_get_usize(v_t_1043_, 4);
v_tailOff_1048_ = lean_ctor_get(v_t_1043_, 3);
v_isSharedCheck_1059_ = !lean_is_exclusive(v_t_1043_);
if (v_isSharedCheck_1059_ == 0)
{
v___x_1050_ = v_t_1043_;
v_isShared_1051_ = v_isSharedCheck_1059_;
goto v_resetjp_1049_;
}
else
{
lean_inc(v_tailOff_1048_);
lean_inc(v_size_1046_);
lean_inc(v_tail_1045_);
lean_inc(v_root_1044_);
lean_dec(v_t_1043_);
v___x_1050_ = lean_box(0);
v_isShared_1051_ = v_isSharedCheck_1059_;
goto v_resetjp_1049_;
}
v_resetjp_1049_:
{
lean_object* v___x_1052_; size_t v_sz_1053_; size_t v___x_1054_; lean_object* v___x_1055_; lean_object* v___x_1057_; 
v___x_1052_ = lp_mathlib_Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__4_spec__7(v_root_1044_);
v_sz_1053_ = lean_array_size(v_tail_1045_);
v___x_1054_ = ((size_t)0ULL);
v___x_1055_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapM___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__4_spec__8(v_sz_1053_, v___x_1054_, v_tail_1045_);
if (v_isShared_1051_ == 0)
{
lean_ctor_set(v___x_1050_, 1, v___x_1055_);
lean_ctor_set(v___x_1050_, 0, v___x_1052_);
v___x_1057_ = v___x_1050_;
goto v_reusejp_1056_;
}
else
{
lean_object* v_reuseFailAlloc_1058_; 
v_reuseFailAlloc_1058_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v_reuseFailAlloc_1058_, 0, v___x_1052_);
lean_ctor_set(v_reuseFailAlloc_1058_, 1, v___x_1055_);
lean_ctor_set(v_reuseFailAlloc_1058_, 2, v_size_1046_);
lean_ctor_set(v_reuseFailAlloc_1058_, 3, v_tailOff_1048_);
lean_ctor_set_usize(v_reuseFailAlloc_1058_, 4, v_shift_1047_);
v___x_1057_ = v_reuseFailAlloc_1058_;
goto v_reusejp_1056_;
}
v_reusejp_1056_:
{
return v___x_1057_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapM___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__4_spec__8___boxed(lean_object* v_sz_1060_, lean_object* v_i_1061_, lean_object* v_bs_1062_){
_start:
{
size_t v_sz_boxed_1063_; size_t v_i_boxed_1064_; lean_object* v_res_1065_; 
v_sz_boxed_1063_ = lean_unbox_usize(v_sz_1060_);
lean_dec(v_sz_1060_);
v_i_boxed_1064_ = lean_unbox_usize(v_i_1061_);
lean_dec(v_i_1061_);
v_res_1065_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapM___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__4_spec__8(v_sz_boxed_1063_, v_i_boxed_1064_, v_bs_1062_);
return v_res_1065_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__4_spec__7_spec__9___boxed(lean_object* v_sz_1066_, lean_object* v_i_1067_, lean_object* v_bs_1068_){
_start:
{
size_t v_sz_boxed_1069_; size_t v_i_boxed_1070_; lean_object* v_res_1071_; 
v_sz_boxed_1069_ = lean_unbox_usize(v_sz_1066_);
lean_dec(v_sz_1066_);
v_i_boxed_1070_ = lean_unbox_usize(v_i_1067_);
lean_dec(v_i_1067_);
v_res_1071_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__4_spec__7_spec__9(v_sz_boxed_1069_, v_i_boxed_1070_, v_bs_1068_);
return v_res_1071_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__1(lean_object* v_00_u03b2_1072_, lean_object* v_m_1073_, lean_object* v_a_1074_){
_start:
{
uint8_t v___x_1075_; 
v___x_1075_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__1___redArg(v_m_1073_, v_a_1074_);
return v___x_1075_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__1___boxed(lean_object* v_00_u03b2_1076_, lean_object* v_m_1077_, lean_object* v_a_1078_){
_start:
{
uint8_t v_res_1079_; lean_object* v_r_1080_; 
v_res_1079_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__1(v_00_u03b2_1076_, v_m_1077_, v_a_1078_);
lean_dec(v_a_1078_);
lean_dec_ref(v_m_1077_);
v_r_1080_ = lean_box(v_res_1079_);
return v_r_1080_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__0_spec__0(lean_object* v_00_u03b2_1081_, lean_object* v_m_1082_, lean_object* v_a_1083_, lean_object* v_b_1084_){
_start:
{
lean_object* v___x_1085_; 
v___x_1085_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__0_spec__0___redArg(v_m_1082_, v_a_1083_, v_b_1084_);
return v___x_1085_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__1_spec__3(lean_object* v_00_u03b2_1086_, lean_object* v_a_1087_, lean_object* v_x_1088_){
_start:
{
uint8_t v___x_1089_; 
v___x_1089_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__1_spec__3___redArg(v_a_1087_, v_x_1088_);
return v___x_1089_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__1_spec__3___boxed(lean_object* v_00_u03b2_1090_, lean_object* v_a_1091_, lean_object* v_x_1092_){
_start:
{
uint8_t v_res_1093_; lean_object* v_r_1094_; 
v_res_1093_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__1_spec__3(v_00_u03b2_1090_, v_a_1091_, v_x_1092_);
lean_dec(v_x_1092_);
lean_dec(v_a_1091_);
v_r_1094_ = lean_box(v_res_1093_);
return v_r_1094_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__0_spec__0_spec__1(lean_object* v_00_u03b2_1095_, lean_object* v_data_1096_){
_start:
{
lean_object* v___x_1097_; 
v___x_1097_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__0_spec__0_spec__1___redArg(v_data_1096_);
return v___x_1097_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__0_spec__0_spec__1_spec__7(lean_object* v_00_u03b2_1098_, lean_object* v_i_1099_, lean_object* v_source_1100_, lean_object* v_target_1101_){
_start:
{
lean_object* v___x_1102_; 
v___x_1102_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__0_spec__0_spec__1_spec__7___redArg(v_i_1099_, v_source_1100_, v_target_1101_);
return v___x_1102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__0_spec__0_spec__1_spec__7_spec__12(lean_object* v_00_u03b2_1103_, lean_object* v_x_1104_, lean_object* v_x_1105_){
_start:
{
lean_object* v___x_1106_; 
v___x_1106_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals_spec__0_spec__0_spec__1_spec__7_spec__12___redArg(v_x_1104_, v_x_1105_);
return v___x_1106_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getInfoTrees___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__1___redArg(lean_object* v___y_1107_){
_start:
{
lean_object* v___x_1109_; lean_object* v_infoState_1110_; lean_object* v_trees_1111_; lean_object* v___x_1112_; 
v___x_1109_ = lean_st_ref_get(v___y_1107_);
v_infoState_1110_ = lean_ctor_get(v___x_1109_, 8);
lean_inc_ref(v_infoState_1110_);
lean_dec(v___x_1109_);
v_trees_1111_ = lean_ctor_get(v_infoState_1110_, 2);
lean_inc_ref(v_trees_1111_);
lean_dec_ref(v_infoState_1110_);
v___x_1112_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1112_, 0, v_trees_1111_);
return v___x_1112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getInfoTrees___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__1___redArg___boxed(lean_object* v___y_1113_, lean_object* v___y_1114_){
_start:
{
lean_object* v_res_1115_; 
v_res_1115_ = lp_mathlib_Lean_Elab_getInfoTrees___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__1___redArg(v___y_1113_);
lean_dec(v___y_1113_);
return v_res_1115_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getInfoTrees___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__1(lean_object* v___y_1116_, lean_object* v___y_1117_){
_start:
{
lean_object* v___x_1119_; 
v___x_1119_ = lp_mathlib_Lean_Elab_getInfoTrees___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__1___redArg(v___y_1117_);
return v___x_1119_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getInfoTrees___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__1___boxed(lean_object* v___y_1120_, lean_object* v___y_1121_, lean_object* v___y_1122_){
_start:
{
lean_object* v_res_1123_; 
v_res_1123_ = lp_mathlib_Lean_Elab_getInfoTrees___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__1(v___y_1120_, v___y_1121_);
lean_dec(v___y_1121_);
lean_dec_ref(v___y_1120_);
return v_res_1123_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___lam__0(lean_object* v_fst_1127_, lean_object* v___y_1128_, lean_object* v___y_1129_){
_start:
{
lean_object* v___x_1131_; 
v___x_1131_ = l_Lean_PrettyPrinter_ppTactic(v_fst_1127_, v___y_1128_, v___y_1129_);
if (lean_obj_tag(v___x_1131_) == 0)
{
return v___x_1131_;
}
else
{
lean_object* v_a_1132_; uint8_t v___y_1134_; uint8_t v___x_1144_; 
v_a_1132_ = lean_ctor_get(v___x_1131_, 0);
lean_inc(v_a_1132_);
v___x_1144_ = l_Lean_Exception_isInterrupt(v_a_1132_);
if (v___x_1144_ == 0)
{
uint8_t v___x_1145_; 
v___x_1145_ = l_Lean_Exception_isRuntime(v_a_1132_);
v___y_1134_ = v___x_1145_;
goto v___jp_1133_;
}
else
{
lean_dec(v_a_1132_);
v___y_1134_ = v___x_1144_;
goto v___jp_1133_;
}
v___jp_1133_:
{
if (v___y_1134_ == 0)
{
lean_object* v___x_1136_; uint8_t v_isShared_1137_; uint8_t v_isSharedCheck_1142_; 
v_isSharedCheck_1142_ = !lean_is_exclusive(v___x_1131_);
if (v_isSharedCheck_1142_ == 0)
{
lean_object* v_unused_1143_; 
v_unused_1143_ = lean_ctor_get(v___x_1131_, 0);
lean_dec(v_unused_1143_);
v___x_1136_ = v___x_1131_;
v_isShared_1137_ = v_isSharedCheck_1142_;
goto v_resetjp_1135_;
}
else
{
lean_dec(v___x_1131_);
v___x_1136_ = lean_box(0);
v_isShared_1137_ = v_isSharedCheck_1142_;
goto v_resetjp_1135_;
}
v_resetjp_1135_:
{
lean_object* v___x_1138_; lean_object* v___x_1140_; 
v___x_1138_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___lam__0___closed__1));
if (v_isShared_1137_ == 0)
{
lean_ctor_set_tag(v___x_1136_, 0);
lean_ctor_set(v___x_1136_, 0, v___x_1138_);
v___x_1140_ = v___x_1136_;
goto v_reusejp_1139_;
}
else
{
lean_object* v_reuseFailAlloc_1141_; 
v_reuseFailAlloc_1141_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1141_, 0, v___x_1138_);
v___x_1140_ = v_reuseFailAlloc_1141_;
goto v_reusejp_1139_;
}
v_reusejp_1139_:
{
return v___x_1140_;
}
}
}
else
{
return v___x_1131_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___lam__0___boxed(lean_object* v_fst_1146_, lean_object* v___y_1147_, lean_object* v___y_1148_, lean_object* v___y_1149_){
_start:
{
lean_object* v_res_1150_; 
v_res_1150_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___lam__0(v_fst_1146_, v___y_1147_, v___y_1148_);
lean_dec(v___y_1148_);
lean_dec_ref(v___y_1147_);
return v_res_1150_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___lam__1(lean_object* v_k_1157_){
_start:
{
lean_object* v___x_1158_; uint8_t v___x_1159_; 
v___x_1158_ = lean_unsigned_to_nat(1u);
v___x_1159_ = lean_nat_dec_eq(v_k_1157_, v___x_1158_);
if (v___x_1159_ == 0)
{
lean_object* v___x_1160_; lean_object* v___x_1161_; lean_object* v___x_1162_; lean_object* v___x_1163_; 
v___x_1160_ = l_Nat_reprFast(v_k_1157_);
v___x_1161_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1161_, 0, v___x_1160_);
v___x_1162_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___lam__1___closed__1));
v___x_1163_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1163_, 0, v___x_1161_);
lean_ctor_set(v___x_1163_, 1, v___x_1162_);
return v___x_1163_;
}
else
{
lean_object* v___x_1164_; 
lean_dec(v_k_1157_);
v___x_1164_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___lam__1___closed__3));
return v___x_1164_;
}
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__0(void){
_start:
{
lean_object* v___x_1165_; 
v___x_1165_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_1165_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__1(void){
_start:
{
lean_object* v___x_1166_; lean_object* v___x_1167_; 
v___x_1166_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__0, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__0_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__0);
v___x_1167_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1167_, 0, v___x_1166_);
return v___x_1167_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__2(void){
_start:
{
lean_object* v___x_1168_; lean_object* v___x_1169_; lean_object* v___x_1170_; 
v___x_1168_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__1);
v___x_1169_ = lean_unsigned_to_nat(0u);
v___x_1170_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_1170_, 0, v___x_1169_);
lean_ctor_set(v___x_1170_, 1, v___x_1169_);
lean_ctor_set(v___x_1170_, 2, v___x_1169_);
lean_ctor_set(v___x_1170_, 3, v___x_1169_);
lean_ctor_set(v___x_1170_, 4, v___x_1168_);
lean_ctor_set(v___x_1170_, 5, v___x_1168_);
lean_ctor_set(v___x_1170_, 6, v___x_1168_);
lean_ctor_set(v___x_1170_, 7, v___x_1168_);
lean_ctor_set(v___x_1170_, 8, v___x_1168_);
lean_ctor_set(v___x_1170_, 9, v___x_1168_);
return v___x_1170_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__3(void){
_start:
{
lean_object* v___x_1171_; lean_object* v___x_1172_; lean_object* v___x_1173_; 
v___x_1171_ = lean_unsigned_to_nat(32u);
v___x_1172_ = lean_mk_empty_array_with_capacity(v___x_1171_);
v___x_1173_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1173_, 0, v___x_1172_);
return v___x_1173_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__4(void){
_start:
{
size_t v___x_1174_; lean_object* v___x_1175_; lean_object* v___x_1176_; lean_object* v___x_1177_; lean_object* v___x_1178_; lean_object* v___x_1179_; 
v___x_1174_ = ((size_t)5ULL);
v___x_1175_ = lean_unsigned_to_nat(0u);
v___x_1176_ = lean_unsigned_to_nat(32u);
v___x_1177_ = lean_mk_empty_array_with_capacity(v___x_1176_);
v___x_1178_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__3, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__3_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__3);
v___x_1179_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_1179_, 0, v___x_1178_);
lean_ctor_set(v___x_1179_, 1, v___x_1177_);
lean_ctor_set(v___x_1179_, 2, v___x_1175_);
lean_ctor_set(v___x_1179_, 3, v___x_1175_);
lean_ctor_set_usize(v___x_1179_, 4, v___x_1174_);
return v___x_1179_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__5(void){
_start:
{
lean_object* v___x_1180_; lean_object* v___x_1181_; lean_object* v___x_1182_; lean_object* v___x_1183_; 
v___x_1180_ = lean_box(1);
v___x_1181_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__4, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__4_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__4);
v___x_1182_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__1);
v___x_1183_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1183_, 0, v___x_1182_);
lean_ctor_set(v___x_1183_, 1, v___x_1181_);
lean_ctor_set(v___x_1183_, 2, v___x_1180_);
return v___x_1183_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4_spec__7___redArg(lean_object* v_msgData_1184_, lean_object* v___y_1185_){
_start:
{
lean_object* v___x_1187_; lean_object* v_env_1188_; lean_object* v___x_1189_; lean_object* v_scopes_1190_; lean_object* v___x_1191_; lean_object* v___x_1192_; lean_object* v_opts_1193_; lean_object* v___x_1194_; lean_object* v___x_1195_; lean_object* v___x_1196_; lean_object* v___x_1197_; lean_object* v___x_1198_; 
v___x_1187_ = lean_st_ref_get(v___y_1185_);
v_env_1188_ = lean_ctor_get(v___x_1187_, 0);
lean_inc_ref(v_env_1188_);
lean_dec(v___x_1187_);
v___x_1189_ = lean_st_ref_get(v___y_1185_);
v_scopes_1190_ = lean_ctor_get(v___x_1189_, 2);
lean_inc(v_scopes_1190_);
lean_dec(v___x_1189_);
v___x_1191_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_1192_ = l_List_head_x21___redArg(v___x_1191_, v_scopes_1190_);
lean_dec(v_scopes_1190_);
v_opts_1193_ = lean_ctor_get(v___x_1192_, 1);
lean_inc_ref(v_opts_1193_);
lean_dec(v___x_1192_);
v___x_1194_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__2, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__2_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__2);
v___x_1195_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__5, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__5_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__5);
v___x_1196_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1196_, 0, v_env_1188_);
lean_ctor_set(v___x_1196_, 1, v___x_1194_);
lean_ctor_set(v___x_1196_, 2, v___x_1195_);
lean_ctor_set(v___x_1196_, 3, v_opts_1193_);
v___x_1197_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_1197_, 0, v___x_1196_);
lean_ctor_set(v___x_1197_, 1, v_msgData_1184_);
v___x_1198_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1198_, 0, v___x_1197_);
return v___x_1198_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4_spec__7___redArg___boxed(lean_object* v_msgData_1199_, lean_object* v___y_1200_, lean_object* v___y_1201_){
_start:
{
lean_object* v_res_1202_; 
v_res_1202_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4_spec__7___redArg(v_msgData_1199_, v___y_1200_);
lean_dec(v___y_1200_);
return v_res_1202_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4_spec__8(lean_object* v_opts_1203_, lean_object* v_opt_1204_){
_start:
{
lean_object* v_name_1205_; lean_object* v_defValue_1206_; lean_object* v_map_1207_; lean_object* v___x_1208_; 
v_name_1205_ = lean_ctor_get(v_opt_1204_, 0);
v_defValue_1206_ = lean_ctor_get(v_opt_1204_, 1);
v_map_1207_ = lean_ctor_get(v_opts_1203_, 0);
v___x_1208_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_1207_, v_name_1205_);
if (lean_obj_tag(v___x_1208_) == 0)
{
uint8_t v___x_1209_; 
v___x_1209_ = lean_unbox(v_defValue_1206_);
return v___x_1209_;
}
else
{
lean_object* v_val_1210_; 
v_val_1210_ = lean_ctor_get(v___x_1208_, 0);
lean_inc(v_val_1210_);
lean_dec_ref_known(v___x_1208_, 1);
if (lean_obj_tag(v_val_1210_) == 1)
{
uint8_t v_v_1211_; 
v_v_1211_ = lean_ctor_get_uint8(v_val_1210_, 0);
lean_dec_ref_known(v_val_1210_, 0);
return v_v_1211_;
}
else
{
uint8_t v___x_1212_; 
lean_dec(v_val_1210_);
v___x_1212_ = lean_unbox(v_defValue_1206_);
return v___x_1212_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4_spec__8___boxed(lean_object* v_opts_1213_, lean_object* v_opt_1214_){
_start:
{
uint8_t v_res_1215_; lean_object* v_r_1216_; 
v_res_1215_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4_spec__8(v_opts_1213_, v_opt_1214_);
lean_dec_ref(v_opt_1214_);
lean_dec_ref(v_opts_1213_);
v_r_1216_ = lean_box(v_res_1215_);
return v_r_1216_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4___lam__0(uint8_t v___y_1218_, uint8_t v_suppressElabErrors_1219_, lean_object* v_x_1220_){
_start:
{
if (lean_obj_tag(v_x_1220_) == 1)
{
lean_object* v_pre_1221_; 
v_pre_1221_ = lean_ctor_get(v_x_1220_, 0);
if (lean_obj_tag(v_pre_1221_) == 0)
{
lean_object* v_str_1222_; lean_object* v___x_1223_; uint8_t v___x_1224_; 
v_str_1222_ = lean_ctor_get(v_x_1220_, 1);
v___x_1223_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4___lam__0___closed__0));
v___x_1224_ = lean_string_dec_eq(v_str_1222_, v___x_1223_);
if (v___x_1224_ == 0)
{
return v___y_1218_;
}
else
{
return v_suppressElabErrors_1219_;
}
}
else
{
return v___y_1218_;
}
}
else
{
return v___y_1218_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4___lam__0___boxed(lean_object* v___y_1225_, lean_object* v_suppressElabErrors_1226_, lean_object* v_x_1227_){
_start:
{
uint8_t v___y_9274__boxed_1228_; uint8_t v_suppressElabErrors_boxed_1229_; uint8_t v_res_1230_; lean_object* v_r_1231_; 
v___y_9274__boxed_1228_ = lean_unbox(v___y_1225_);
v_suppressElabErrors_boxed_1229_ = lean_unbox(v_suppressElabErrors_1226_);
v_res_1230_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4___lam__0(v___y_9274__boxed_1228_, v_suppressElabErrors_boxed_1229_, v_x_1227_);
lean_dec(v_x_1227_);
v_r_1231_ = lean_box(v_res_1230_);
return v_r_1231_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4(lean_object* v_ref_1233_, lean_object* v_msgData_1234_, uint8_t v_severity_1235_, uint8_t v_isSilent_1236_, lean_object* v___y_1237_, lean_object* v___y_1238_){
_start:
{
lean_object* v___y_1241_; uint8_t v___y_1242_; lean_object* v___y_1243_; lean_object* v___y_1244_; lean_object* v___y_1245_; lean_object* v___y_1246_; uint8_t v___y_1247_; lean_object* v___y_1248_; uint8_t v___y_1305_; lean_object* v___y_1306_; uint8_t v___y_1307_; uint8_t v___y_1308_; lean_object* v___y_1309_; uint8_t v___y_1333_; uint8_t v___y_1334_; lean_object* v___y_1335_; uint8_t v___y_1336_; lean_object* v___y_1337_; uint8_t v___y_1341_; uint8_t v___y_1342_; uint8_t v___y_1343_; uint8_t v___x_1358_; uint8_t v___y_1360_; uint8_t v___y_1361_; uint8_t v___y_1362_; uint8_t v___y_1364_; uint8_t v___x_1376_; 
v___x_1358_ = 2;
v___x_1376_ = l_Lean_instBEqMessageSeverity_beq(v_severity_1235_, v___x_1358_);
if (v___x_1376_ == 0)
{
v___y_1364_ = v___x_1376_;
goto v___jp_1363_;
}
else
{
uint8_t v___x_1377_; 
lean_inc_ref(v_msgData_1234_);
v___x_1377_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_1234_);
v___y_1364_ = v___x_1377_;
goto v___jp_1363_;
}
v___jp_1240_:
{
lean_object* v___x_1249_; 
v___x_1249_ = l_Lean_Elab_Command_getScope___redArg(v___y_1248_);
if (lean_obj_tag(v___x_1249_) == 0)
{
lean_object* v_a_1250_; lean_object* v___x_1251_; 
v_a_1250_ = lean_ctor_get(v___x_1249_, 0);
lean_inc(v_a_1250_);
lean_dec_ref_known(v___x_1249_, 1);
v___x_1251_ = l_Lean_Elab_Command_getScope___redArg(v___y_1248_);
if (lean_obj_tag(v___x_1251_) == 0)
{
lean_object* v_a_1252_; lean_object* v___x_1254_; uint8_t v_isShared_1255_; uint8_t v_isSharedCheck_1287_; 
v_a_1252_ = lean_ctor_get(v___x_1251_, 0);
v_isSharedCheck_1287_ = !lean_is_exclusive(v___x_1251_);
if (v_isSharedCheck_1287_ == 0)
{
v___x_1254_ = v___x_1251_;
v_isShared_1255_ = v_isSharedCheck_1287_;
goto v_resetjp_1253_;
}
else
{
lean_inc(v_a_1252_);
lean_dec(v___x_1251_);
v___x_1254_ = lean_box(0);
v_isShared_1255_ = v_isSharedCheck_1287_;
goto v_resetjp_1253_;
}
v_resetjp_1253_:
{
lean_object* v___x_1256_; lean_object* v_currNamespace_1257_; lean_object* v_openDecls_1258_; lean_object* v_env_1259_; lean_object* v_messages_1260_; lean_object* v_scopes_1261_; lean_object* v_usedQuotCtxts_1262_; lean_object* v_nextMacroScope_1263_; lean_object* v_maxRecDepth_1264_; lean_object* v_ngen_1265_; lean_object* v_auxDeclNGen_1266_; lean_object* v_infoState_1267_; lean_object* v_traceState_1268_; lean_object* v_snapshotTasks_1269_; lean_object* v_prevLinterStates_1270_; lean_object* v___x_1272_; uint8_t v_isShared_1273_; uint8_t v_isSharedCheck_1286_; 
v___x_1256_ = lean_st_ref_take(v___y_1248_);
v_currNamespace_1257_ = lean_ctor_get(v_a_1250_, 2);
lean_inc(v_currNamespace_1257_);
lean_dec(v_a_1250_);
v_openDecls_1258_ = lean_ctor_get(v_a_1252_, 3);
lean_inc(v_openDecls_1258_);
lean_dec(v_a_1252_);
v_env_1259_ = lean_ctor_get(v___x_1256_, 0);
v_messages_1260_ = lean_ctor_get(v___x_1256_, 1);
v_scopes_1261_ = lean_ctor_get(v___x_1256_, 2);
v_usedQuotCtxts_1262_ = lean_ctor_get(v___x_1256_, 3);
v_nextMacroScope_1263_ = lean_ctor_get(v___x_1256_, 4);
v_maxRecDepth_1264_ = lean_ctor_get(v___x_1256_, 5);
v_ngen_1265_ = lean_ctor_get(v___x_1256_, 6);
v_auxDeclNGen_1266_ = lean_ctor_get(v___x_1256_, 7);
v_infoState_1267_ = lean_ctor_get(v___x_1256_, 8);
v_traceState_1268_ = lean_ctor_get(v___x_1256_, 9);
v_snapshotTasks_1269_ = lean_ctor_get(v___x_1256_, 10);
v_prevLinterStates_1270_ = lean_ctor_get(v___x_1256_, 11);
v_isSharedCheck_1286_ = !lean_is_exclusive(v___x_1256_);
if (v_isSharedCheck_1286_ == 0)
{
v___x_1272_ = v___x_1256_;
v_isShared_1273_ = v_isSharedCheck_1286_;
goto v_resetjp_1271_;
}
else
{
lean_inc(v_prevLinterStates_1270_);
lean_inc(v_snapshotTasks_1269_);
lean_inc(v_traceState_1268_);
lean_inc(v_infoState_1267_);
lean_inc(v_auxDeclNGen_1266_);
lean_inc(v_ngen_1265_);
lean_inc(v_maxRecDepth_1264_);
lean_inc(v_nextMacroScope_1263_);
lean_inc(v_usedQuotCtxts_1262_);
lean_inc(v_scopes_1261_);
lean_inc(v_messages_1260_);
lean_inc(v_env_1259_);
lean_dec(v___x_1256_);
v___x_1272_ = lean_box(0);
v_isShared_1273_ = v_isSharedCheck_1286_;
goto v_resetjp_1271_;
}
v_resetjp_1271_:
{
lean_object* v___x_1274_; lean_object* v___x_1275_; lean_object* v___x_1276_; lean_object* v___x_1277_; lean_object* v___x_1279_; 
v___x_1274_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1274_, 0, v_currNamespace_1257_);
lean_ctor_set(v___x_1274_, 1, v_openDecls_1258_);
v___x_1275_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1275_, 0, v___x_1274_);
lean_ctor_set(v___x_1275_, 1, v___y_1244_);
lean_inc_ref(v___y_1245_);
lean_inc_ref(v___y_1243_);
v___x_1276_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_1276_, 0, v___y_1243_);
lean_ctor_set(v___x_1276_, 1, v___y_1246_);
lean_ctor_set(v___x_1276_, 2, v___y_1241_);
lean_ctor_set(v___x_1276_, 3, v___y_1245_);
lean_ctor_set(v___x_1276_, 4, v___x_1275_);
lean_ctor_set_uint8(v___x_1276_, sizeof(void*)*5, v___y_1247_);
lean_ctor_set_uint8(v___x_1276_, sizeof(void*)*5 + 1, v___y_1242_);
lean_ctor_set_uint8(v___x_1276_, sizeof(void*)*5 + 2, v_isSilent_1236_);
v___x_1277_ = l_Lean_MessageLog_add(v___x_1276_, v_messages_1260_);
if (v_isShared_1273_ == 0)
{
lean_ctor_set(v___x_1272_, 1, v___x_1277_);
v___x_1279_ = v___x_1272_;
goto v_reusejp_1278_;
}
else
{
lean_object* v_reuseFailAlloc_1285_; 
v_reuseFailAlloc_1285_ = lean_alloc_ctor(0, 12, 0);
lean_ctor_set(v_reuseFailAlloc_1285_, 0, v_env_1259_);
lean_ctor_set(v_reuseFailAlloc_1285_, 1, v___x_1277_);
lean_ctor_set(v_reuseFailAlloc_1285_, 2, v_scopes_1261_);
lean_ctor_set(v_reuseFailAlloc_1285_, 3, v_usedQuotCtxts_1262_);
lean_ctor_set(v_reuseFailAlloc_1285_, 4, v_nextMacroScope_1263_);
lean_ctor_set(v_reuseFailAlloc_1285_, 5, v_maxRecDepth_1264_);
lean_ctor_set(v_reuseFailAlloc_1285_, 6, v_ngen_1265_);
lean_ctor_set(v_reuseFailAlloc_1285_, 7, v_auxDeclNGen_1266_);
lean_ctor_set(v_reuseFailAlloc_1285_, 8, v_infoState_1267_);
lean_ctor_set(v_reuseFailAlloc_1285_, 9, v_traceState_1268_);
lean_ctor_set(v_reuseFailAlloc_1285_, 10, v_snapshotTasks_1269_);
lean_ctor_set(v_reuseFailAlloc_1285_, 11, v_prevLinterStates_1270_);
v___x_1279_ = v_reuseFailAlloc_1285_;
goto v_reusejp_1278_;
}
v_reusejp_1278_:
{
lean_object* v___x_1280_; lean_object* v___x_1281_; lean_object* v___x_1283_; 
v___x_1280_ = lean_st_ref_set(v___y_1248_, v___x_1279_);
v___x_1281_ = lean_box(0);
if (v_isShared_1255_ == 0)
{
lean_ctor_set(v___x_1254_, 0, v___x_1281_);
v___x_1283_ = v___x_1254_;
goto v_reusejp_1282_;
}
else
{
lean_object* v_reuseFailAlloc_1284_; 
v_reuseFailAlloc_1284_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1284_, 0, v___x_1281_);
v___x_1283_ = v_reuseFailAlloc_1284_;
goto v_reusejp_1282_;
}
v_reusejp_1282_:
{
return v___x_1283_;
}
}
}
}
}
else
{
lean_object* v_a_1288_; lean_object* v___x_1290_; uint8_t v_isShared_1291_; uint8_t v_isSharedCheck_1295_; 
lean_dec(v_a_1250_);
lean_dec_ref(v___y_1246_);
lean_dec_ref(v___y_1244_);
lean_dec(v___y_1241_);
v_a_1288_ = lean_ctor_get(v___x_1251_, 0);
v_isSharedCheck_1295_ = !lean_is_exclusive(v___x_1251_);
if (v_isSharedCheck_1295_ == 0)
{
v___x_1290_ = v___x_1251_;
v_isShared_1291_ = v_isSharedCheck_1295_;
goto v_resetjp_1289_;
}
else
{
lean_inc(v_a_1288_);
lean_dec(v___x_1251_);
v___x_1290_ = lean_box(0);
v_isShared_1291_ = v_isSharedCheck_1295_;
goto v_resetjp_1289_;
}
v_resetjp_1289_:
{
lean_object* v___x_1293_; 
if (v_isShared_1291_ == 0)
{
v___x_1293_ = v___x_1290_;
goto v_reusejp_1292_;
}
else
{
lean_object* v_reuseFailAlloc_1294_; 
v_reuseFailAlloc_1294_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1294_, 0, v_a_1288_);
v___x_1293_ = v_reuseFailAlloc_1294_;
goto v_reusejp_1292_;
}
v_reusejp_1292_:
{
return v___x_1293_;
}
}
}
}
else
{
lean_object* v_a_1296_; lean_object* v___x_1298_; uint8_t v_isShared_1299_; uint8_t v_isSharedCheck_1303_; 
lean_dec_ref(v___y_1246_);
lean_dec_ref(v___y_1244_);
lean_dec(v___y_1241_);
v_a_1296_ = lean_ctor_get(v___x_1249_, 0);
v_isSharedCheck_1303_ = !lean_is_exclusive(v___x_1249_);
if (v_isSharedCheck_1303_ == 0)
{
v___x_1298_ = v___x_1249_;
v_isShared_1299_ = v_isSharedCheck_1303_;
goto v_resetjp_1297_;
}
else
{
lean_inc(v_a_1296_);
lean_dec(v___x_1249_);
v___x_1298_ = lean_box(0);
v_isShared_1299_ = v_isSharedCheck_1303_;
goto v_resetjp_1297_;
}
v_resetjp_1297_:
{
lean_object* v___x_1301_; 
if (v_isShared_1299_ == 0)
{
v___x_1301_ = v___x_1298_;
goto v_reusejp_1300_;
}
else
{
lean_object* v_reuseFailAlloc_1302_; 
v_reuseFailAlloc_1302_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1302_, 0, v_a_1296_);
v___x_1301_ = v_reuseFailAlloc_1302_;
goto v_reusejp_1300_;
}
v_reusejp_1300_:
{
return v___x_1301_;
}
}
}
}
v___jp_1304_:
{
lean_object* v_fileName_1310_; lean_object* v_fileMap_1311_; uint8_t v_suppressElabErrors_1312_; lean_object* v___x_1313_; lean_object* v___x_1314_; lean_object* v_a_1315_; lean_object* v___x_1317_; uint8_t v_isShared_1318_; uint8_t v_isSharedCheck_1331_; 
v_fileName_1310_ = lean_ctor_get(v___y_1237_, 0);
v_fileMap_1311_ = lean_ctor_get(v___y_1237_, 1);
v_suppressElabErrors_1312_ = lean_ctor_get_uint8(v___y_1237_, sizeof(void*)*10);
v___x_1313_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_1234_);
v___x_1314_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4_spec__7___redArg(v___x_1313_, v___y_1238_);
v_a_1315_ = lean_ctor_get(v___x_1314_, 0);
v_isSharedCheck_1331_ = !lean_is_exclusive(v___x_1314_);
if (v_isSharedCheck_1331_ == 0)
{
v___x_1317_ = v___x_1314_;
v_isShared_1318_ = v_isSharedCheck_1331_;
goto v_resetjp_1316_;
}
else
{
lean_inc(v_a_1315_);
lean_dec(v___x_1314_);
v___x_1317_ = lean_box(0);
v_isShared_1318_ = v_isSharedCheck_1331_;
goto v_resetjp_1316_;
}
v_resetjp_1316_:
{
lean_object* v___x_1319_; lean_object* v___x_1320_; lean_object* v___x_1321_; lean_object* v___x_1322_; 
lean_inc_ref_n(v_fileMap_1311_, 2);
v___x_1319_ = l_Lean_FileMap_toPosition(v_fileMap_1311_, v___y_1306_);
lean_dec(v___y_1306_);
v___x_1320_ = l_Lean_FileMap_toPosition(v_fileMap_1311_, v___y_1309_);
lean_dec(v___y_1309_);
v___x_1321_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1321_, 0, v___x_1320_);
v___x_1322_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4___closed__0));
if (v_suppressElabErrors_1312_ == 0)
{
lean_del_object(v___x_1317_);
v___y_1241_ = v___x_1321_;
v___y_1242_ = v___y_1307_;
v___y_1243_ = v_fileName_1310_;
v___y_1244_ = v_a_1315_;
v___y_1245_ = v___x_1322_;
v___y_1246_ = v___x_1319_;
v___y_1247_ = v___y_1308_;
v___y_1248_ = v___y_1238_;
goto v___jp_1240_;
}
else
{
lean_object* v___x_1323_; lean_object* v___x_1324_; lean_object* v___f_1325_; uint8_t v___x_1326_; 
v___x_1323_ = lean_box(v___y_1305_);
v___x_1324_ = lean_box(v_suppressElabErrors_1312_);
v___f_1325_ = lean_alloc_closure((void*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4___lam__0___boxed), 3, 2);
lean_closure_set(v___f_1325_, 0, v___x_1323_);
lean_closure_set(v___f_1325_, 1, v___x_1324_);
lean_inc(v_a_1315_);
v___x_1326_ = l_Lean_MessageData_hasTag(v___f_1325_, v_a_1315_);
if (v___x_1326_ == 0)
{
lean_object* v___x_1327_; lean_object* v___x_1329_; 
lean_dec_ref_known(v___x_1321_, 1);
lean_dec_ref(v___x_1319_);
lean_dec(v_a_1315_);
v___x_1327_ = lean_box(0);
if (v_isShared_1318_ == 0)
{
lean_ctor_set(v___x_1317_, 0, v___x_1327_);
v___x_1329_ = v___x_1317_;
goto v_reusejp_1328_;
}
else
{
lean_object* v_reuseFailAlloc_1330_; 
v_reuseFailAlloc_1330_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1330_, 0, v___x_1327_);
v___x_1329_ = v_reuseFailAlloc_1330_;
goto v_reusejp_1328_;
}
v_reusejp_1328_:
{
return v___x_1329_;
}
}
else
{
lean_del_object(v___x_1317_);
v___y_1241_ = v___x_1321_;
v___y_1242_ = v___y_1307_;
v___y_1243_ = v_fileName_1310_;
v___y_1244_ = v_a_1315_;
v___y_1245_ = v___x_1322_;
v___y_1246_ = v___x_1319_;
v___y_1247_ = v___y_1308_;
v___y_1248_ = v___y_1238_;
goto v___jp_1240_;
}
}
}
}
v___jp_1332_:
{
lean_object* v___x_1338_; 
v___x_1338_ = l_Lean_Syntax_getTailPos_x3f(v___y_1335_, v___y_1336_);
lean_dec(v___y_1335_);
if (lean_obj_tag(v___x_1338_) == 0)
{
lean_inc(v___y_1337_);
v___y_1305_ = v___y_1333_;
v___y_1306_ = v___y_1337_;
v___y_1307_ = v___y_1334_;
v___y_1308_ = v___y_1336_;
v___y_1309_ = v___y_1337_;
goto v___jp_1304_;
}
else
{
lean_object* v_val_1339_; 
v_val_1339_ = lean_ctor_get(v___x_1338_, 0);
lean_inc(v_val_1339_);
lean_dec_ref_known(v___x_1338_, 1);
v___y_1305_ = v___y_1333_;
v___y_1306_ = v___y_1337_;
v___y_1307_ = v___y_1334_;
v___y_1308_ = v___y_1336_;
v___y_1309_ = v_val_1339_;
goto v___jp_1304_;
}
}
v___jp_1340_:
{
lean_object* v___x_1344_; 
v___x_1344_ = l_Lean_Elab_Command_getRef___redArg(v___y_1237_);
if (lean_obj_tag(v___x_1344_) == 0)
{
lean_object* v_a_1345_; lean_object* v_ref_1346_; lean_object* v___x_1347_; 
v_a_1345_ = lean_ctor_get(v___x_1344_, 0);
lean_inc(v_a_1345_);
lean_dec_ref_known(v___x_1344_, 1);
v_ref_1346_ = l_Lean_replaceRef(v_ref_1233_, v_a_1345_);
lean_dec(v_a_1345_);
v___x_1347_ = l_Lean_Syntax_getPos_x3f(v_ref_1346_, v___y_1342_);
if (lean_obj_tag(v___x_1347_) == 0)
{
lean_object* v___x_1348_; 
v___x_1348_ = lean_unsigned_to_nat(0u);
v___y_1333_ = v___y_1341_;
v___y_1334_ = v___y_1343_;
v___y_1335_ = v_ref_1346_;
v___y_1336_ = v___y_1342_;
v___y_1337_ = v___x_1348_;
goto v___jp_1332_;
}
else
{
lean_object* v_val_1349_; 
v_val_1349_ = lean_ctor_get(v___x_1347_, 0);
lean_inc(v_val_1349_);
lean_dec_ref_known(v___x_1347_, 1);
v___y_1333_ = v___y_1341_;
v___y_1334_ = v___y_1343_;
v___y_1335_ = v_ref_1346_;
v___y_1336_ = v___y_1342_;
v___y_1337_ = v_val_1349_;
goto v___jp_1332_;
}
}
else
{
lean_object* v_a_1350_; lean_object* v___x_1352_; uint8_t v_isShared_1353_; uint8_t v_isSharedCheck_1357_; 
lean_dec_ref(v_msgData_1234_);
v_a_1350_ = lean_ctor_get(v___x_1344_, 0);
v_isSharedCheck_1357_ = !lean_is_exclusive(v___x_1344_);
if (v_isSharedCheck_1357_ == 0)
{
v___x_1352_ = v___x_1344_;
v_isShared_1353_ = v_isSharedCheck_1357_;
goto v_resetjp_1351_;
}
else
{
lean_inc(v_a_1350_);
lean_dec(v___x_1344_);
v___x_1352_ = lean_box(0);
v_isShared_1353_ = v_isSharedCheck_1357_;
goto v_resetjp_1351_;
}
v_resetjp_1351_:
{
lean_object* v___x_1355_; 
if (v_isShared_1353_ == 0)
{
v___x_1355_ = v___x_1352_;
goto v_reusejp_1354_;
}
else
{
lean_object* v_reuseFailAlloc_1356_; 
v_reuseFailAlloc_1356_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1356_, 0, v_a_1350_);
v___x_1355_ = v_reuseFailAlloc_1356_;
goto v_reusejp_1354_;
}
v_reusejp_1354_:
{
return v___x_1355_;
}
}
}
}
v___jp_1359_:
{
if (v___y_1362_ == 0)
{
v___y_1341_ = v___y_1360_;
v___y_1342_ = v___y_1361_;
v___y_1343_ = v_severity_1235_;
goto v___jp_1340_;
}
else
{
v___y_1341_ = v___y_1360_;
v___y_1342_ = v___y_1361_;
v___y_1343_ = v___x_1358_;
goto v___jp_1340_;
}
}
v___jp_1363_:
{
if (v___y_1364_ == 0)
{
lean_object* v___x_1365_; lean_object* v_scopes_1366_; lean_object* v___x_1367_; lean_object* v___x_1368_; lean_object* v_opts_1369_; uint8_t v___x_1370_; uint8_t v___x_1371_; 
v___x_1365_ = lean_st_ref_get(v___y_1238_);
v_scopes_1366_ = lean_ctor_get(v___x_1365_, 2);
lean_inc(v_scopes_1366_);
lean_dec(v___x_1365_);
v___x_1367_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_1368_ = l_List_head_x21___redArg(v___x_1367_, v_scopes_1366_);
lean_dec(v_scopes_1366_);
v_opts_1369_ = lean_ctor_get(v___x_1368_, 1);
lean_inc_ref(v_opts_1369_);
lean_dec(v___x_1368_);
v___x_1370_ = 1;
v___x_1371_ = l_Lean_instBEqMessageSeverity_beq(v_severity_1235_, v___x_1370_);
if (v___x_1371_ == 0)
{
lean_dec_ref(v_opts_1369_);
v___y_1360_ = v___y_1364_;
v___y_1361_ = v___y_1364_;
v___y_1362_ = v___x_1371_;
goto v___jp_1359_;
}
else
{
lean_object* v___x_1372_; uint8_t v___x_1373_; 
v___x_1372_ = l_Lean_warningAsError;
v___x_1373_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4_spec__8(v_opts_1369_, v___x_1372_);
lean_dec_ref(v_opts_1369_);
v___y_1360_ = v___y_1364_;
v___y_1361_ = v___y_1364_;
v___y_1362_ = v___x_1373_;
goto v___jp_1359_;
}
}
else
{
lean_object* v___x_1374_; lean_object* v___x_1375_; 
lean_dec_ref(v_msgData_1234_);
v___x_1374_ = lean_box(0);
v___x_1375_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1375_, 0, v___x_1374_);
return v___x_1375_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4___boxed(lean_object* v_ref_1378_, lean_object* v_msgData_1379_, lean_object* v_severity_1380_, lean_object* v_isSilent_1381_, lean_object* v___y_1382_, lean_object* v___y_1383_, lean_object* v___y_1384_){
_start:
{
uint8_t v_severity_boxed_1385_; uint8_t v_isSilent_boxed_1386_; lean_object* v_res_1387_; 
v_severity_boxed_1385_ = lean_unbox(v_severity_1380_);
v_isSilent_boxed_1386_ = lean_unbox(v_isSilent_1381_);
v_res_1387_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4(v_ref_1378_, v_msgData_1379_, v_severity_boxed_1385_, v_isSilent_boxed_1386_, v___y_1382_, v___y_1383_);
lean_dec(v___y_1383_);
lean_dec_ref(v___y_1382_);
lean_dec(v_ref_1378_);
return v_res_1387_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3(lean_object* v_ref_1388_, lean_object* v_msgData_1389_, lean_object* v___y_1390_, lean_object* v___y_1391_){
_start:
{
uint8_t v___x_1393_; uint8_t v___x_1394_; lean_object* v___x_1395_; 
v___x_1393_ = 1;
v___x_1394_ = 0;
v___x_1395_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4(v_ref_1388_, v_msgData_1389_, v___x_1393_, v___x_1394_, v___y_1390_, v___y_1391_);
return v___x_1395_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3___boxed(lean_object* v_ref_1396_, lean_object* v_msgData_1397_, lean_object* v___y_1398_, lean_object* v___y_1399_, lean_object* v___y_1400_){
_start:
{
lean_object* v_res_1401_; 
v_res_1401_ = lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3(v_ref_1396_, v_msgData_1397_, v___y_1398_, v___y_1399_);
lean_dec(v___y_1399_);
lean_dec_ref(v___y_1398_);
lean_dec(v_ref_1396_);
return v_res_1401_;
}
}
static lean_object* _init_lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2___closed__1(void){
_start:
{
lean_object* v___x_1403_; lean_object* v___x_1404_; 
v___x_1403_ = ((lean_object*)(lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2___closed__0));
v___x_1404_ = l_Lean_stringToMessageData(v___x_1403_);
return v___x_1404_;
}
}
static lean_object* _init_lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2___closed__3(void){
_start:
{
lean_object* v___x_1406_; lean_object* v___x_1407_; 
v___x_1406_ = ((lean_object*)(lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2___closed__2));
v___x_1407_ = l_Lean_stringToMessageData(v___x_1406_);
return v___x_1407_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2(lean_object* v_linterOption_1408_, lean_object* v_stx_1409_, lean_object* v_msg_1410_, lean_object* v___y_1411_, lean_object* v___y_1412_){
_start:
{
lean_object* v_name_1414_; lean_object* v___x_1416_; uint8_t v_isShared_1417_; uint8_t v_isSharedCheck_1432_; 
v_name_1414_ = lean_ctor_get(v_linterOption_1408_, 0);
v_isSharedCheck_1432_ = !lean_is_exclusive(v_linterOption_1408_);
if (v_isSharedCheck_1432_ == 0)
{
lean_object* v_unused_1433_; 
v_unused_1433_ = lean_ctor_get(v_linterOption_1408_, 1);
lean_dec(v_unused_1433_);
v___x_1416_ = v_linterOption_1408_;
v_isShared_1417_ = v_isSharedCheck_1432_;
goto v_resetjp_1415_;
}
else
{
lean_inc(v_name_1414_);
lean_dec(v_linterOption_1408_);
v___x_1416_ = lean_box(0);
v_isShared_1417_ = v_isSharedCheck_1432_;
goto v_resetjp_1415_;
}
v_resetjp_1415_:
{
lean_object* v___x_1418_; lean_object* v___x_1419_; lean_object* v___x_1421_; 
v___x_1418_ = lean_obj_once(&lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2___closed__1, &lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2___closed__1_once, _init_lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2___closed__1);
lean_inc(v_name_1414_);
v___x_1419_ = l_Lean_MessageData_ofName(v_name_1414_);
if (v_isShared_1417_ == 0)
{
lean_ctor_set_tag(v___x_1416_, 7);
lean_ctor_set(v___x_1416_, 1, v___x_1419_);
lean_ctor_set(v___x_1416_, 0, v___x_1418_);
v___x_1421_ = v___x_1416_;
goto v_reusejp_1420_;
}
else
{
lean_object* v_reuseFailAlloc_1431_; 
v_reuseFailAlloc_1431_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1431_, 0, v___x_1418_);
lean_ctor_set(v_reuseFailAlloc_1431_, 1, v___x_1419_);
v___x_1421_ = v_reuseFailAlloc_1431_;
goto v_reusejp_1420_;
}
v_reusejp_1420_:
{
lean_object* v___x_1422_; lean_object* v___x_1423_; lean_object* v_disable_1424_; lean_object* v___x_1425_; lean_object* v___x_1426_; lean_object* v___x_1427_; lean_object* v___x_1428_; lean_object* v___x_1429_; lean_object* v___x_1430_; 
v___x_1422_ = lean_obj_once(&lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2___closed__3, &lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2___closed__3_once, _init_lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2___closed__3);
v___x_1423_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1423_, 0, v___x_1421_);
lean_ctor_set(v___x_1423_, 1, v___x_1422_);
v_disable_1424_ = l_Lean_MessageData_note(v___x_1423_);
v___x_1425_ = l_Lean_Linter_linterMessageTag;
v___x_1426_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1426_, 0, v_msg_1410_);
lean_ctor_set(v___x_1426_, 1, v_disable_1424_);
v___x_1427_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_1427_, 0, v___x_1425_);
lean_ctor_set(v___x_1427_, 1, v___x_1426_);
v___x_1428_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_1428_, 0, v_name_1414_);
lean_ctor_set(v___x_1428_, 1, v___x_1427_);
lean_inc(v_stx_1409_);
v___x_1429_ = lean_alloc_ctor(11, 2, 0);
lean_ctor_set(v___x_1429_, 0, v_stx_1409_);
lean_ctor_set(v___x_1429_, 1, v___x_1428_);
v___x_1430_ = lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3(v_stx_1409_, v___x_1429_, v___y_1411_, v___y_1412_);
lean_dec(v_stx_1409_);
return v___x_1430_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2___boxed(lean_object* v_linterOption_1434_, lean_object* v_stx_1435_, lean_object* v_msg_1436_, lean_object* v___y_1437_, lean_object* v___y_1438_, lean_object* v___y_1439_){
_start:
{
lean_object* v_res_1440_; 
v_res_1440_ = lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2(v_linterOption_1434_, v_stx_1435_, v_msg_1436_, v___y_1437_, v___y_1438_);
lean_dec(v___y_1438_);
lean_dec_ref(v___y_1437_);
return v_res_1440_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__1(void){
_start:
{
lean_object* v___x_1442_; lean_object* v___x_1443_; 
v___x_1442_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__0));
v___x_1443_ = l_Lean_stringToMessageData(v___x_1442_);
return v___x_1443_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__3(void){
_start:
{
lean_object* v___x_1445_; lean_object* v___x_1446_; 
v___x_1445_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__2));
v___x_1446_ = l_Lean_stringToMessageData(v___x_1445_);
return v___x_1446_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__5(void){
_start:
{
lean_object* v___x_1448_; lean_object* v___x_1449_; 
v___x_1448_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__4));
v___x_1449_ = l_Lean_stringToMessageData(v___x_1448_);
return v___x_1449_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__7(void){
_start:
{
lean_object* v___x_1451_; lean_object* v___x_1452_; 
v___x_1451_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__6));
v___x_1452_ = l_Lean_stringToMessageData(v___x_1451_);
return v___x_1452_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__9(void){
_start:
{
lean_object* v___x_1454_; lean_object* v___x_1455_; 
v___x_1454_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__8));
v___x_1455_ = l_Lean_stringToMessageData(v___x_1454_);
return v___x_1455_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__11(void){
_start:
{
lean_object* v___x_1457_; lean_object* v___x_1458_; 
v___x_1457_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__10));
v___x_1458_ = l_Lean_stringToMessageData(v___x_1457_);
return v___x_1458_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3(lean_object* v_as_1461_, size_t v_sz_1462_, size_t v_i_1463_, lean_object* v_b_1464_, lean_object* v___y_1465_, lean_object* v___y_1466_){
_start:
{
uint8_t v___x_1468_; 
v___x_1468_ = lean_usize_dec_lt(v_i_1463_, v_sz_1462_);
if (v___x_1468_ == 0)
{
lean_object* v___x_1469_; 
v___x_1469_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1469_, 0, v_b_1464_);
return v___x_1469_;
}
else
{
lean_object* v_a_1470_; lean_object* v_snd_1471_; lean_object* v_snd_1472_; lean_object* v_fst_1473_; lean_object* v___x_1475_; uint8_t v_isShared_1476_; uint8_t v_isSharedCheck_1544_; 
v_a_1470_ = lean_array_uget(v_as_1461_, v_i_1463_);
v_snd_1471_ = lean_ctor_get(v_a_1470_, 1);
lean_inc(v_snd_1471_);
v_snd_1472_ = lean_ctor_get(v_snd_1471_, 1);
lean_inc(v_snd_1472_);
v_fst_1473_ = lean_ctor_get(v_a_1470_, 0);
v_isSharedCheck_1544_ = !lean_is_exclusive(v_a_1470_);
if (v_isSharedCheck_1544_ == 0)
{
lean_object* v_unused_1545_; 
v_unused_1545_ = lean_ctor_get(v_a_1470_, 1);
lean_dec(v_unused_1545_);
v___x_1475_ = v_a_1470_;
v_isShared_1476_ = v_isSharedCheck_1544_;
goto v_resetjp_1474_;
}
else
{
lean_inc(v_fst_1473_);
lean_dec(v_a_1470_);
v___x_1475_ = lean_box(0);
v_isShared_1476_ = v_isSharedCheck_1544_;
goto v_resetjp_1474_;
}
v_resetjp_1474_:
{
lean_object* v_fst_1477_; lean_object* v___x_1479_; uint8_t v_isShared_1480_; uint8_t v_isSharedCheck_1542_; 
v_fst_1477_ = lean_ctor_get(v_snd_1471_, 0);
v_isSharedCheck_1542_ = !lean_is_exclusive(v_snd_1471_);
if (v_isSharedCheck_1542_ == 0)
{
lean_object* v_unused_1543_; 
v_unused_1543_ = lean_ctor_get(v_snd_1471_, 1);
lean_dec(v_unused_1543_);
v___x_1479_ = v_snd_1471_;
v_isShared_1480_ = v_isSharedCheck_1542_;
goto v_resetjp_1478_;
}
else
{
lean_inc(v_fst_1477_);
lean_dec(v_snd_1471_);
v___x_1479_ = lean_box(0);
v_isShared_1480_ = v_isSharedCheck_1542_;
goto v_resetjp_1478_;
}
v_resetjp_1478_:
{
lean_object* v_fst_1481_; lean_object* v_snd_1482_; lean_object* v___x_1484_; uint8_t v_isShared_1485_; uint8_t v_isSharedCheck_1541_; 
v_fst_1481_ = lean_ctor_get(v_snd_1472_, 0);
v_snd_1482_ = lean_ctor_get(v_snd_1472_, 1);
v_isSharedCheck_1541_ = !lean_is_exclusive(v_snd_1472_);
if (v_isSharedCheck_1541_ == 0)
{
v___x_1484_ = v_snd_1472_;
v_isShared_1485_ = v_isSharedCheck_1541_;
goto v_resetjp_1483_;
}
else
{
lean_inc(v_snd_1482_);
lean_inc(v_fst_1481_);
lean_dec(v_snd_1472_);
v___x_1484_ = lean_box(0);
v_isShared_1485_ = v_isSharedCheck_1541_;
goto v_resetjp_1483_;
}
v_resetjp_1483_:
{
lean_object* v___f_1486_; lean_object* v___x_1487_; 
lean_inc(v_fst_1473_);
v___f_1486_ = lean_alloc_closure((void*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___lam__0___boxed), 4, 1);
lean_closure_set(v___f_1486_, 0, v_fst_1473_);
v___x_1487_ = l_Lean_Elab_Command_liftCoreM___redArg(v___f_1486_, v___y_1465_, v___y_1466_);
if (lean_obj_tag(v___x_1487_) == 0)
{
lean_object* v_a_1488_; lean_object* v___x_1489_; lean_object* v___x_1490_; lean_object* v___x_1491_; lean_object* v___x_1492_; lean_object* v___x_1493_; lean_object* v___x_1495_; 
v_a_1488_ = lean_ctor_get(v___x_1487_, 0);
lean_inc(v_a_1488_);
lean_dec_ref_known(v___x_1487_, 1);
v___x_1489_ = lp_mathlib_Mathlib_Linter_linter_style_multiGoal;
v___x_1490_ = lean_box(0);
v___x_1491_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__1, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__1_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__1);
v___x_1492_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___lam__1(v_fst_1477_);
v___x_1493_ = l_Lean_MessageData_ofFormat(v___x_1492_);
if (v_isShared_1485_ == 0)
{
lean_ctor_set_tag(v___x_1484_, 7);
lean_ctor_set(v___x_1484_, 1, v___x_1493_);
lean_ctor_set(v___x_1484_, 0, v___x_1491_);
v___x_1495_ = v___x_1484_;
goto v_reusejp_1494_;
}
else
{
lean_object* v_reuseFailAlloc_1532_; 
v_reuseFailAlloc_1532_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1532_, 0, v___x_1491_);
lean_ctor_set(v_reuseFailAlloc_1532_, 1, v___x_1493_);
v___x_1495_ = v_reuseFailAlloc_1532_;
goto v_reusejp_1494_;
}
v_reusejp_1494_:
{
lean_object* v___x_1496_; lean_object* v___x_1498_; 
v___x_1496_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__3, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__3_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__3);
if (v_isShared_1480_ == 0)
{
lean_ctor_set_tag(v___x_1479_, 7);
lean_ctor_set(v___x_1479_, 1, v___x_1496_);
lean_ctor_set(v___x_1479_, 0, v___x_1495_);
v___x_1498_ = v___x_1479_;
goto v_reusejp_1497_;
}
else
{
lean_object* v_reuseFailAlloc_1531_; 
v_reuseFailAlloc_1531_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1531_, 0, v___x_1495_);
lean_ctor_set(v_reuseFailAlloc_1531_, 1, v___x_1496_);
v___x_1498_ = v_reuseFailAlloc_1531_;
goto v_reusejp_1497_;
}
v_reusejp_1497_:
{
lean_object* v___x_1499_; lean_object* v___x_1500_; lean_object* v___x_1502_; 
v___x_1499_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___lam__1(v_fst_1481_);
v___x_1500_ = l_Lean_MessageData_ofFormat(v___x_1499_);
if (v_isShared_1476_ == 0)
{
lean_ctor_set_tag(v___x_1475_, 7);
lean_ctor_set(v___x_1475_, 1, v___x_1500_);
lean_ctor_set(v___x_1475_, 0, v___x_1498_);
v___x_1502_ = v___x_1475_;
goto v_reusejp_1501_;
}
else
{
lean_object* v_reuseFailAlloc_1530_; 
v_reuseFailAlloc_1530_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1530_, 0, v___x_1498_);
lean_ctor_set(v_reuseFailAlloc_1530_, 1, v___x_1500_);
v___x_1502_ = v_reuseFailAlloc_1530_;
goto v_reusejp_1501_;
}
v_reusejp_1501_:
{
lean_object* v___x_1503_; lean_object* v___x_1504_; lean_object* v___x_1505_; lean_object* v___x_1506_; lean_object* v___x_1507_; lean_object* v___x_1508_; lean_object* v___x_1509_; lean_object* v___x_1510_; lean_object* v___y_1512_; lean_object* v___x_1526_; uint8_t v___x_1527_; 
v___x_1503_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__5, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__5_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__5);
v___x_1504_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1504_, 0, v___x_1502_);
lean_ctor_set(v___x_1504_, 1, v___x_1503_);
lean_inc(v_snd_1482_);
v___x_1505_ = l_Nat_reprFast(v_snd_1482_);
v___x_1506_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1506_, 0, v___x_1505_);
v___x_1507_ = l_Lean_MessageData_ofFormat(v___x_1506_);
v___x_1508_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1508_, 0, v___x_1504_);
lean_ctor_set(v___x_1508_, 1, v___x_1507_);
v___x_1509_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__7, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__7_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__7);
v___x_1510_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1510_, 0, v___x_1508_);
lean_ctor_set(v___x_1510_, 1, v___x_1509_);
v___x_1526_ = lean_unsigned_to_nat(1u);
v___x_1527_ = lean_nat_dec_eq(v_snd_1482_, v___x_1526_);
lean_dec(v_snd_1482_);
if (v___x_1527_ == 0)
{
lean_object* v___x_1528_; 
v___x_1528_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__12));
v___y_1512_ = v___x_1528_;
goto v___jp_1511_;
}
else
{
lean_object* v___x_1529_; 
v___x_1529_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__13));
v___y_1512_ = v___x_1529_;
goto v___jp_1511_;
}
v___jp_1511_:
{
lean_object* v___x_1513_; lean_object* v___x_1514_; lean_object* v___x_1515_; lean_object* v___x_1516_; lean_object* v___x_1517_; lean_object* v___x_1518_; lean_object* v___x_1519_; lean_object* v___x_1520_; lean_object* v___x_1521_; lean_object* v___x_1522_; 
lean_inc_ref(v___y_1512_);
v___x_1513_ = l_Lean_stringToMessageData(v___y_1512_);
v___x_1514_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1514_, 0, v___x_1510_);
lean_ctor_set(v___x_1514_, 1, v___x_1513_);
v___x_1515_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__9, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__9_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__9);
v___x_1516_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1516_, 0, v___x_1514_);
lean_ctor_set(v___x_1516_, 1, v___x_1515_);
v___x_1517_ = l_Lean_MessageData_ofFormat(v_a_1488_);
v___x_1518_ = l_Lean_indentD(v___x_1517_);
v___x_1519_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1519_, 0, v___x_1516_);
lean_ctor_set(v___x_1519_, 1, v___x_1518_);
v___x_1520_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__11, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__11_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___closed__11);
v___x_1521_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1521_, 0, v___x_1519_);
lean_ctor_set(v___x_1521_, 1, v___x_1520_);
v___x_1522_ = lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2(v___x_1489_, v_fst_1473_, v___x_1521_, v___y_1465_, v___y_1466_);
if (lean_obj_tag(v___x_1522_) == 0)
{
size_t v___x_1523_; size_t v___x_1524_; 
lean_dec_ref_known(v___x_1522_, 1);
v___x_1523_ = ((size_t)1ULL);
v___x_1524_ = lean_usize_add(v_i_1463_, v___x_1523_);
v_i_1463_ = v___x_1524_;
v_b_1464_ = v___x_1490_;
goto _start;
}
else
{
return v___x_1522_;
}
}
}
}
}
}
else
{
lean_object* v_a_1533_; lean_object* v___x_1535_; uint8_t v_isShared_1536_; uint8_t v_isSharedCheck_1540_; 
lean_del_object(v___x_1484_);
lean_dec(v_snd_1482_);
lean_dec(v_fst_1481_);
lean_del_object(v___x_1479_);
lean_dec(v_fst_1477_);
lean_del_object(v___x_1475_);
lean_dec(v_fst_1473_);
v_a_1533_ = lean_ctor_get(v___x_1487_, 0);
v_isSharedCheck_1540_ = !lean_is_exclusive(v___x_1487_);
if (v_isSharedCheck_1540_ == 0)
{
v___x_1535_ = v___x_1487_;
v_isShared_1536_ = v_isSharedCheck_1540_;
goto v_resetjp_1534_;
}
else
{
lean_inc(v_a_1533_);
lean_dec(v___x_1487_);
v___x_1535_ = lean_box(0);
v_isShared_1536_ = v_isSharedCheck_1540_;
goto v_resetjp_1534_;
}
v_resetjp_1534_:
{
lean_object* v___x_1538_; 
if (v_isShared_1536_ == 0)
{
v___x_1538_ = v___x_1535_;
goto v_reusejp_1537_;
}
else
{
lean_object* v_reuseFailAlloc_1539_; 
v_reuseFailAlloc_1539_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1539_, 0, v_a_1533_);
v___x_1538_ = v_reuseFailAlloc_1539_;
goto v_reusejp_1537_;
}
v_reusejp_1537_:
{
return v___x_1538_;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3___boxed(lean_object* v_as_1546_, lean_object* v_sz_1547_, lean_object* v_i_1548_, lean_object* v_b_1549_, lean_object* v___y_1550_, lean_object* v___y_1551_, lean_object* v___y_1552_){
_start:
{
size_t v_sz_boxed_1553_; size_t v_i_boxed_1554_; lean_object* v_res_1555_; 
v_sz_boxed_1553_ = lean_unbox_usize(v_sz_1547_);
lean_dec(v_sz_1547_);
v_i_boxed_1554_ = lean_unbox_usize(v_i_1548_);
lean_dec(v_i_1548_);
v_res_1555_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3(v_as_1546_, v_sz_boxed_1553_, v_i_boxed_1554_, v_b_1549_, v___y_1550_, v___y_1551_);
lean_dec(v___y_1551_);
lean_dec_ref(v___y_1550_);
lean_dec_ref(v_as_1546_);
return v_res_1555_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__4_spec__7_spec__11(lean_object* v_as_1559_, size_t v_sz_1560_, size_t v_i_1561_, lean_object* v_b_1562_, lean_object* v___y_1563_, lean_object* v___y_1564_){
_start:
{
uint8_t v___x_1566_; 
v___x_1566_ = lean_usize_dec_lt(v_i_1561_, v_sz_1560_);
if (v___x_1566_ == 0)
{
lean_object* v___x_1567_; 
v___x_1567_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1567_, 0, v_b_1562_);
return v___x_1567_;
}
else
{
lean_object* v___x_1568_; lean_object* v_a_1569_; lean_object* v___x_1570_; size_t v_sz_1571_; size_t v___x_1572_; lean_object* v___x_1573_; 
lean_dec_ref(v_b_1562_);
v___x_1568_ = lean_box(0);
v_a_1569_ = lean_array_uget_borrowed(v_as_1559_, v_i_1561_);
lean_inc(v_a_1569_);
v___x_1570_ = lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals(v_a_1569_);
v_sz_1571_ = lean_array_size(v___x_1570_);
v___x_1572_ = ((size_t)0ULL);
v___x_1573_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3(v___x_1570_, v_sz_1571_, v___x_1572_, v___x_1568_, v___y_1563_, v___y_1564_);
lean_dec_ref(v___x_1570_);
if (lean_obj_tag(v___x_1573_) == 0)
{
lean_object* v___x_1574_; size_t v___x_1575_; size_t v___x_1576_; 
lean_dec_ref_known(v___x_1573_, 1);
v___x_1574_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__4_spec__7_spec__11___closed__0));
v___x_1575_ = ((size_t)1ULL);
v___x_1576_ = lean_usize_add(v_i_1561_, v___x_1575_);
v_i_1561_ = v___x_1576_;
v_b_1562_ = v___x_1574_;
goto _start;
}
else
{
lean_object* v_a_1578_; lean_object* v___x_1580_; uint8_t v_isShared_1581_; uint8_t v_isSharedCheck_1585_; 
v_a_1578_ = lean_ctor_get(v___x_1573_, 0);
v_isSharedCheck_1585_ = !lean_is_exclusive(v___x_1573_);
if (v_isSharedCheck_1585_ == 0)
{
v___x_1580_ = v___x_1573_;
v_isShared_1581_ = v_isSharedCheck_1585_;
goto v_resetjp_1579_;
}
else
{
lean_inc(v_a_1578_);
lean_dec(v___x_1573_);
v___x_1580_ = lean_box(0);
v_isShared_1581_ = v_isSharedCheck_1585_;
goto v_resetjp_1579_;
}
v_resetjp_1579_:
{
lean_object* v___x_1583_; 
if (v_isShared_1581_ == 0)
{
v___x_1583_ = v___x_1580_;
goto v_reusejp_1582_;
}
else
{
lean_object* v_reuseFailAlloc_1584_; 
v_reuseFailAlloc_1584_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1584_, 0, v_a_1578_);
v___x_1583_ = v_reuseFailAlloc_1584_;
goto v_reusejp_1582_;
}
v_reusejp_1582_:
{
return v___x_1583_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__4_spec__7_spec__11___boxed(lean_object* v_as_1586_, lean_object* v_sz_1587_, lean_object* v_i_1588_, lean_object* v_b_1589_, lean_object* v___y_1590_, lean_object* v___y_1591_, lean_object* v___y_1592_){
_start:
{
size_t v_sz_boxed_1593_; size_t v_i_boxed_1594_; lean_object* v_res_1595_; 
v_sz_boxed_1593_ = lean_unbox_usize(v_sz_1587_);
lean_dec(v_sz_1587_);
v_i_boxed_1594_ = lean_unbox_usize(v_i_1588_);
lean_dec(v_i_1588_);
v_res_1595_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__4_spec__7_spec__11(v_as_1586_, v_sz_boxed_1593_, v_i_boxed_1594_, v_b_1589_, v___y_1590_, v___y_1591_);
lean_dec(v___y_1591_);
lean_dec_ref(v___y_1590_);
lean_dec_ref(v_as_1586_);
return v_res_1595_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__4_spec__7(lean_object* v_as_1596_, size_t v_sz_1597_, size_t v_i_1598_, lean_object* v_b_1599_, lean_object* v___y_1600_, lean_object* v___y_1601_){
_start:
{
uint8_t v___x_1603_; 
v___x_1603_ = lean_usize_dec_lt(v_i_1598_, v_sz_1597_);
if (v___x_1603_ == 0)
{
lean_object* v___x_1604_; 
v___x_1604_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1604_, 0, v_b_1599_);
return v___x_1604_;
}
else
{
lean_object* v___x_1605_; lean_object* v_a_1606_; lean_object* v___x_1607_; size_t v_sz_1608_; size_t v___x_1609_; lean_object* v___x_1610_; 
lean_dec_ref(v_b_1599_);
v___x_1605_ = lean_box(0);
v_a_1606_ = lean_array_uget_borrowed(v_as_1596_, v_i_1598_);
lean_inc(v_a_1606_);
v___x_1607_ = lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals(v_a_1606_);
v_sz_1608_ = lean_array_size(v___x_1607_);
v___x_1609_ = ((size_t)0ULL);
v___x_1610_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3(v___x_1607_, v_sz_1608_, v___x_1609_, v___x_1605_, v___y_1600_, v___y_1601_);
lean_dec_ref(v___x_1607_);
if (lean_obj_tag(v___x_1610_) == 0)
{
lean_object* v___x_1611_; size_t v___x_1612_; size_t v___x_1613_; lean_object* v___x_1614_; 
lean_dec_ref_known(v___x_1610_, 1);
v___x_1611_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__4_spec__7_spec__11___closed__0));
v___x_1612_ = ((size_t)1ULL);
v___x_1613_ = lean_usize_add(v_i_1598_, v___x_1612_);
v___x_1614_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__4_spec__7_spec__11(v_as_1596_, v_sz_1597_, v___x_1613_, v___x_1611_, v___y_1600_, v___y_1601_);
return v___x_1614_;
}
else
{
lean_object* v_a_1615_; lean_object* v___x_1617_; uint8_t v_isShared_1618_; uint8_t v_isSharedCheck_1622_; 
v_a_1615_ = lean_ctor_get(v___x_1610_, 0);
v_isSharedCheck_1622_ = !lean_is_exclusive(v___x_1610_);
if (v_isSharedCheck_1622_ == 0)
{
v___x_1617_ = v___x_1610_;
v_isShared_1618_ = v_isSharedCheck_1622_;
goto v_resetjp_1616_;
}
else
{
lean_inc(v_a_1615_);
lean_dec(v___x_1610_);
v___x_1617_ = lean_box(0);
v_isShared_1618_ = v_isSharedCheck_1622_;
goto v_resetjp_1616_;
}
v_resetjp_1616_:
{
lean_object* v___x_1620_; 
if (v_isShared_1618_ == 0)
{
v___x_1620_ = v___x_1617_;
goto v_reusejp_1619_;
}
else
{
lean_object* v_reuseFailAlloc_1621_; 
v_reuseFailAlloc_1621_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1621_, 0, v_a_1615_);
v___x_1620_ = v_reuseFailAlloc_1621_;
goto v_reusejp_1619_;
}
v_reusejp_1619_:
{
return v___x_1620_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__4_spec__7___boxed(lean_object* v_as_1623_, lean_object* v_sz_1624_, lean_object* v_i_1625_, lean_object* v_b_1626_, lean_object* v___y_1627_, lean_object* v___y_1628_, lean_object* v___y_1629_){
_start:
{
size_t v_sz_boxed_1630_; size_t v_i_boxed_1631_; lean_object* v_res_1632_; 
v_sz_boxed_1630_ = lean_unbox_usize(v_sz_1624_);
lean_dec(v_sz_1624_);
v_i_boxed_1631_ = lean_unbox_usize(v_i_1625_);
lean_dec(v_i_1625_);
v_res_1632_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__4_spec__7(v_as_1623_, v_sz_boxed_1630_, v_i_boxed_1631_, v_b_1626_, v___y_1627_, v___y_1628_);
lean_dec(v___y_1628_);
lean_dec_ref(v___y_1627_);
lean_dec_ref(v_as_1623_);
return v_res_1632_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__4_spec__6_spec__9_spec__12(lean_object* v_as_1636_, size_t v_sz_1637_, size_t v_i_1638_, lean_object* v_b_1639_, lean_object* v___y_1640_, lean_object* v___y_1641_){
_start:
{
uint8_t v___x_1643_; 
v___x_1643_ = lean_usize_dec_lt(v_i_1638_, v_sz_1637_);
if (v___x_1643_ == 0)
{
lean_object* v___x_1644_; 
v___x_1644_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1644_, 0, v_b_1639_);
return v___x_1644_;
}
else
{
lean_object* v___x_1645_; lean_object* v_a_1646_; lean_object* v___x_1647_; size_t v_sz_1648_; size_t v___x_1649_; lean_object* v___x_1650_; 
lean_dec_ref(v_b_1639_);
v___x_1645_ = lean_box(0);
v_a_1646_ = lean_array_uget_borrowed(v_as_1636_, v_i_1638_);
lean_inc(v_a_1646_);
v___x_1647_ = lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals(v_a_1646_);
v_sz_1648_ = lean_array_size(v___x_1647_);
v___x_1649_ = ((size_t)0ULL);
v___x_1650_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3(v___x_1647_, v_sz_1648_, v___x_1649_, v___x_1645_, v___y_1640_, v___y_1641_);
lean_dec_ref(v___x_1647_);
if (lean_obj_tag(v___x_1650_) == 0)
{
lean_object* v___x_1651_; size_t v___x_1652_; size_t v___x_1653_; 
lean_dec_ref_known(v___x_1650_, 1);
v___x_1651_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__4_spec__6_spec__9_spec__12___closed__0));
v___x_1652_ = ((size_t)1ULL);
v___x_1653_ = lean_usize_add(v_i_1638_, v___x_1652_);
v_i_1638_ = v___x_1653_;
v_b_1639_ = v___x_1651_;
goto _start;
}
else
{
lean_object* v_a_1655_; lean_object* v___x_1657_; uint8_t v_isShared_1658_; uint8_t v_isSharedCheck_1662_; 
v_a_1655_ = lean_ctor_get(v___x_1650_, 0);
v_isSharedCheck_1662_ = !lean_is_exclusive(v___x_1650_);
if (v_isSharedCheck_1662_ == 0)
{
v___x_1657_ = v___x_1650_;
v_isShared_1658_ = v_isSharedCheck_1662_;
goto v_resetjp_1656_;
}
else
{
lean_inc(v_a_1655_);
lean_dec(v___x_1650_);
v___x_1657_ = lean_box(0);
v_isShared_1658_ = v_isSharedCheck_1662_;
goto v_resetjp_1656_;
}
v_resetjp_1656_:
{
lean_object* v___x_1660_; 
if (v_isShared_1658_ == 0)
{
v___x_1660_ = v___x_1657_;
goto v_reusejp_1659_;
}
else
{
lean_object* v_reuseFailAlloc_1661_; 
v_reuseFailAlloc_1661_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1661_, 0, v_a_1655_);
v___x_1660_ = v_reuseFailAlloc_1661_;
goto v_reusejp_1659_;
}
v_reusejp_1659_:
{
return v___x_1660_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__4_spec__6_spec__9_spec__12___boxed(lean_object* v_as_1663_, lean_object* v_sz_1664_, lean_object* v_i_1665_, lean_object* v_b_1666_, lean_object* v___y_1667_, lean_object* v___y_1668_, lean_object* v___y_1669_){
_start:
{
size_t v_sz_boxed_1670_; size_t v_i_boxed_1671_; lean_object* v_res_1672_; 
v_sz_boxed_1670_ = lean_unbox_usize(v_sz_1664_);
lean_dec(v_sz_1664_);
v_i_boxed_1671_ = lean_unbox_usize(v_i_1665_);
lean_dec(v_i_1665_);
v_res_1672_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__4_spec__6_spec__9_spec__12(v_as_1663_, v_sz_boxed_1670_, v_i_boxed_1671_, v_b_1666_, v___y_1667_, v___y_1668_);
lean_dec(v___y_1668_);
lean_dec_ref(v___y_1667_);
lean_dec_ref(v_as_1663_);
return v_res_1672_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__4_spec__6_spec__9(lean_object* v_as_1673_, size_t v_sz_1674_, size_t v_i_1675_, lean_object* v_b_1676_, lean_object* v___y_1677_, lean_object* v___y_1678_){
_start:
{
uint8_t v___x_1680_; 
v___x_1680_ = lean_usize_dec_lt(v_i_1675_, v_sz_1674_);
if (v___x_1680_ == 0)
{
lean_object* v___x_1681_; 
v___x_1681_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1681_, 0, v_b_1676_);
return v___x_1681_;
}
else
{
lean_object* v___x_1682_; lean_object* v_a_1683_; lean_object* v___x_1684_; size_t v_sz_1685_; size_t v___x_1686_; lean_object* v___x_1687_; 
lean_dec_ref(v_b_1676_);
v___x_1682_ = lean_box(0);
v_a_1683_ = lean_array_uget_borrowed(v_as_1673_, v_i_1675_);
lean_inc(v_a_1683_);
v___x_1684_ = lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_getManyGoals(v_a_1683_);
v_sz_1685_ = lean_array_size(v___x_1684_);
v___x_1686_ = ((size_t)0ULL);
v___x_1687_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__3(v___x_1684_, v_sz_1685_, v___x_1686_, v___x_1682_, v___y_1677_, v___y_1678_);
lean_dec_ref(v___x_1684_);
if (lean_obj_tag(v___x_1687_) == 0)
{
lean_object* v___x_1688_; size_t v___x_1689_; size_t v___x_1690_; lean_object* v___x_1691_; 
lean_dec_ref_known(v___x_1687_, 1);
v___x_1688_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__4_spec__6_spec__9_spec__12___closed__0));
v___x_1689_ = ((size_t)1ULL);
v___x_1690_ = lean_usize_add(v_i_1675_, v___x_1689_);
v___x_1691_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__4_spec__6_spec__9_spec__12(v_as_1673_, v_sz_1674_, v___x_1690_, v___x_1688_, v___y_1677_, v___y_1678_);
return v___x_1691_;
}
else
{
lean_object* v_a_1692_; lean_object* v___x_1694_; uint8_t v_isShared_1695_; uint8_t v_isSharedCheck_1699_; 
v_a_1692_ = lean_ctor_get(v___x_1687_, 0);
v_isSharedCheck_1699_ = !lean_is_exclusive(v___x_1687_);
if (v_isSharedCheck_1699_ == 0)
{
v___x_1694_ = v___x_1687_;
v_isShared_1695_ = v_isSharedCheck_1699_;
goto v_resetjp_1693_;
}
else
{
lean_inc(v_a_1692_);
lean_dec(v___x_1687_);
v___x_1694_ = lean_box(0);
v_isShared_1695_ = v_isSharedCheck_1699_;
goto v_resetjp_1693_;
}
v_resetjp_1693_:
{
lean_object* v___x_1697_; 
if (v_isShared_1695_ == 0)
{
v___x_1697_ = v___x_1694_;
goto v_reusejp_1696_;
}
else
{
lean_object* v_reuseFailAlloc_1698_; 
v_reuseFailAlloc_1698_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1698_, 0, v_a_1692_);
v___x_1697_ = v_reuseFailAlloc_1698_;
goto v_reusejp_1696_;
}
v_reusejp_1696_:
{
return v___x_1697_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__4_spec__6_spec__9___boxed(lean_object* v_as_1700_, lean_object* v_sz_1701_, lean_object* v_i_1702_, lean_object* v_b_1703_, lean_object* v___y_1704_, lean_object* v___y_1705_, lean_object* v___y_1706_){
_start:
{
size_t v_sz_boxed_1707_; size_t v_i_boxed_1708_; lean_object* v_res_1709_; 
v_sz_boxed_1707_ = lean_unbox_usize(v_sz_1701_);
lean_dec(v_sz_1701_);
v_i_boxed_1708_ = lean_unbox_usize(v_i_1702_);
lean_dec(v_i_1702_);
v_res_1709_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__4_spec__6_spec__9(v_as_1700_, v_sz_boxed_1707_, v_i_boxed_1708_, v_b_1703_, v___y_1704_, v___y_1705_);
lean_dec(v___y_1705_);
lean_dec_ref(v___y_1704_);
lean_dec_ref(v_as_1700_);
return v_res_1709_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__4_spec__6(lean_object* v_init_1710_, lean_object* v_n_1711_, lean_object* v_b_1712_, lean_object* v___y_1713_, lean_object* v___y_1714_){
_start:
{
if (lean_obj_tag(v_n_1711_) == 0)
{
lean_object* v_cs_1716_; lean_object* v___x_1717_; lean_object* v___x_1718_; size_t v_sz_1719_; size_t v___x_1720_; lean_object* v___x_1721_; 
v_cs_1716_ = lean_ctor_get(v_n_1711_, 0);
v___x_1717_ = lean_box(0);
v___x_1718_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1718_, 0, v___x_1717_);
lean_ctor_set(v___x_1718_, 1, v_b_1712_);
v_sz_1719_ = lean_array_size(v_cs_1716_);
v___x_1720_ = ((size_t)0ULL);
v___x_1721_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__4_spec__6_spec__8(v_init_1710_, v_cs_1716_, v_sz_1719_, v___x_1720_, v___x_1718_, v___y_1713_, v___y_1714_);
if (lean_obj_tag(v___x_1721_) == 0)
{
lean_object* v_a_1722_; lean_object* v___x_1724_; uint8_t v_isShared_1725_; uint8_t v_isSharedCheck_1736_; 
v_a_1722_ = lean_ctor_get(v___x_1721_, 0);
v_isSharedCheck_1736_ = !lean_is_exclusive(v___x_1721_);
if (v_isSharedCheck_1736_ == 0)
{
v___x_1724_ = v___x_1721_;
v_isShared_1725_ = v_isSharedCheck_1736_;
goto v_resetjp_1723_;
}
else
{
lean_inc(v_a_1722_);
lean_dec(v___x_1721_);
v___x_1724_ = lean_box(0);
v_isShared_1725_ = v_isSharedCheck_1736_;
goto v_resetjp_1723_;
}
v_resetjp_1723_:
{
lean_object* v_fst_1726_; 
v_fst_1726_ = lean_ctor_get(v_a_1722_, 0);
if (lean_obj_tag(v_fst_1726_) == 0)
{
lean_object* v_snd_1727_; lean_object* v___x_1728_; lean_object* v___x_1730_; 
v_snd_1727_ = lean_ctor_get(v_a_1722_, 1);
lean_inc(v_snd_1727_);
lean_dec(v_a_1722_);
v___x_1728_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1728_, 0, v_snd_1727_);
if (v_isShared_1725_ == 0)
{
lean_ctor_set(v___x_1724_, 0, v___x_1728_);
v___x_1730_ = v___x_1724_;
goto v_reusejp_1729_;
}
else
{
lean_object* v_reuseFailAlloc_1731_; 
v_reuseFailAlloc_1731_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1731_, 0, v___x_1728_);
v___x_1730_ = v_reuseFailAlloc_1731_;
goto v_reusejp_1729_;
}
v_reusejp_1729_:
{
return v___x_1730_;
}
}
else
{
lean_object* v_val_1732_; lean_object* v___x_1734_; 
lean_inc_ref(v_fst_1726_);
lean_dec(v_a_1722_);
v_val_1732_ = lean_ctor_get(v_fst_1726_, 0);
lean_inc(v_val_1732_);
lean_dec_ref_known(v_fst_1726_, 1);
if (v_isShared_1725_ == 0)
{
lean_ctor_set(v___x_1724_, 0, v_val_1732_);
v___x_1734_ = v___x_1724_;
goto v_reusejp_1733_;
}
else
{
lean_object* v_reuseFailAlloc_1735_; 
v_reuseFailAlloc_1735_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1735_, 0, v_val_1732_);
v___x_1734_ = v_reuseFailAlloc_1735_;
goto v_reusejp_1733_;
}
v_reusejp_1733_:
{
return v___x_1734_;
}
}
}
}
else
{
lean_object* v_a_1737_; lean_object* v___x_1739_; uint8_t v_isShared_1740_; uint8_t v_isSharedCheck_1744_; 
v_a_1737_ = lean_ctor_get(v___x_1721_, 0);
v_isSharedCheck_1744_ = !lean_is_exclusive(v___x_1721_);
if (v_isSharedCheck_1744_ == 0)
{
v___x_1739_ = v___x_1721_;
v_isShared_1740_ = v_isSharedCheck_1744_;
goto v_resetjp_1738_;
}
else
{
lean_inc(v_a_1737_);
lean_dec(v___x_1721_);
v___x_1739_ = lean_box(0);
v_isShared_1740_ = v_isSharedCheck_1744_;
goto v_resetjp_1738_;
}
v_resetjp_1738_:
{
lean_object* v___x_1742_; 
if (v_isShared_1740_ == 0)
{
v___x_1742_ = v___x_1739_;
goto v_reusejp_1741_;
}
else
{
lean_object* v_reuseFailAlloc_1743_; 
v_reuseFailAlloc_1743_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1743_, 0, v_a_1737_);
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
else
{
lean_object* v_vs_1745_; lean_object* v___x_1746_; lean_object* v___x_1747_; size_t v_sz_1748_; size_t v___x_1749_; lean_object* v___x_1750_; 
v_vs_1745_ = lean_ctor_get(v_n_1711_, 0);
v___x_1746_ = lean_box(0);
v___x_1747_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1747_, 0, v___x_1746_);
lean_ctor_set(v___x_1747_, 1, v_b_1712_);
v_sz_1748_ = lean_array_size(v_vs_1745_);
v___x_1749_ = ((size_t)0ULL);
v___x_1750_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__4_spec__6_spec__9(v_vs_1745_, v_sz_1748_, v___x_1749_, v___x_1747_, v___y_1713_, v___y_1714_);
if (lean_obj_tag(v___x_1750_) == 0)
{
lean_object* v_a_1751_; lean_object* v___x_1753_; uint8_t v_isShared_1754_; uint8_t v_isSharedCheck_1765_; 
v_a_1751_ = lean_ctor_get(v___x_1750_, 0);
v_isSharedCheck_1765_ = !lean_is_exclusive(v___x_1750_);
if (v_isSharedCheck_1765_ == 0)
{
v___x_1753_ = v___x_1750_;
v_isShared_1754_ = v_isSharedCheck_1765_;
goto v_resetjp_1752_;
}
else
{
lean_inc(v_a_1751_);
lean_dec(v___x_1750_);
v___x_1753_ = lean_box(0);
v_isShared_1754_ = v_isSharedCheck_1765_;
goto v_resetjp_1752_;
}
v_resetjp_1752_:
{
lean_object* v_fst_1755_; 
v_fst_1755_ = lean_ctor_get(v_a_1751_, 0);
if (lean_obj_tag(v_fst_1755_) == 0)
{
lean_object* v_snd_1756_; lean_object* v___x_1757_; lean_object* v___x_1759_; 
v_snd_1756_ = lean_ctor_get(v_a_1751_, 1);
lean_inc(v_snd_1756_);
lean_dec(v_a_1751_);
v___x_1757_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1757_, 0, v_snd_1756_);
if (v_isShared_1754_ == 0)
{
lean_ctor_set(v___x_1753_, 0, v___x_1757_);
v___x_1759_ = v___x_1753_;
goto v_reusejp_1758_;
}
else
{
lean_object* v_reuseFailAlloc_1760_; 
v_reuseFailAlloc_1760_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1760_, 0, v___x_1757_);
v___x_1759_ = v_reuseFailAlloc_1760_;
goto v_reusejp_1758_;
}
v_reusejp_1758_:
{
return v___x_1759_;
}
}
else
{
lean_object* v_val_1761_; lean_object* v___x_1763_; 
lean_inc_ref(v_fst_1755_);
lean_dec(v_a_1751_);
v_val_1761_ = lean_ctor_get(v_fst_1755_, 0);
lean_inc(v_val_1761_);
lean_dec_ref_known(v_fst_1755_, 1);
if (v_isShared_1754_ == 0)
{
lean_ctor_set(v___x_1753_, 0, v_val_1761_);
v___x_1763_ = v___x_1753_;
goto v_reusejp_1762_;
}
else
{
lean_object* v_reuseFailAlloc_1764_; 
v_reuseFailAlloc_1764_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1764_, 0, v_val_1761_);
v___x_1763_ = v_reuseFailAlloc_1764_;
goto v_reusejp_1762_;
}
v_reusejp_1762_:
{
return v___x_1763_;
}
}
}
}
else
{
lean_object* v_a_1766_; lean_object* v___x_1768_; uint8_t v_isShared_1769_; uint8_t v_isSharedCheck_1773_; 
v_a_1766_ = lean_ctor_get(v___x_1750_, 0);
v_isSharedCheck_1773_ = !lean_is_exclusive(v___x_1750_);
if (v_isSharedCheck_1773_ == 0)
{
v___x_1768_ = v___x_1750_;
v_isShared_1769_ = v_isSharedCheck_1773_;
goto v_resetjp_1767_;
}
else
{
lean_inc(v_a_1766_);
lean_dec(v___x_1750_);
v___x_1768_ = lean_box(0);
v_isShared_1769_ = v_isSharedCheck_1773_;
goto v_resetjp_1767_;
}
v_resetjp_1767_:
{
lean_object* v___x_1771_; 
if (v_isShared_1769_ == 0)
{
v___x_1771_ = v___x_1768_;
goto v_reusejp_1770_;
}
else
{
lean_object* v_reuseFailAlloc_1772_; 
v_reuseFailAlloc_1772_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1772_, 0, v_a_1766_);
v___x_1771_ = v_reuseFailAlloc_1772_;
goto v_reusejp_1770_;
}
v_reusejp_1770_:
{
return v___x_1771_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__4_spec__6_spec__8(lean_object* v_init_1774_, lean_object* v_as_1775_, size_t v_sz_1776_, size_t v_i_1777_, lean_object* v_b_1778_, lean_object* v___y_1779_, lean_object* v___y_1780_){
_start:
{
uint8_t v___x_1782_; 
v___x_1782_ = lean_usize_dec_lt(v_i_1777_, v_sz_1776_);
if (v___x_1782_ == 0)
{
lean_object* v___x_1783_; 
v___x_1783_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1783_, 0, v_b_1778_);
return v___x_1783_;
}
else
{
lean_object* v_snd_1784_; lean_object* v___x_1786_; uint8_t v_isShared_1787_; uint8_t v_isSharedCheck_1818_; 
v_snd_1784_ = lean_ctor_get(v_b_1778_, 1);
v_isSharedCheck_1818_ = !lean_is_exclusive(v_b_1778_);
if (v_isSharedCheck_1818_ == 0)
{
lean_object* v_unused_1819_; 
v_unused_1819_ = lean_ctor_get(v_b_1778_, 0);
lean_dec(v_unused_1819_);
v___x_1786_ = v_b_1778_;
v_isShared_1787_ = v_isSharedCheck_1818_;
goto v_resetjp_1785_;
}
else
{
lean_inc(v_snd_1784_);
lean_dec(v_b_1778_);
v___x_1786_ = lean_box(0);
v_isShared_1787_ = v_isSharedCheck_1818_;
goto v_resetjp_1785_;
}
v_resetjp_1785_:
{
lean_object* v_a_1788_; lean_object* v___x_1789_; 
v_a_1788_ = lean_array_uget_borrowed(v_as_1775_, v_i_1777_);
lean_inc(v_snd_1784_);
v___x_1789_ = lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__4_spec__6(v_init_1774_, v_a_1788_, v_snd_1784_, v___y_1779_, v___y_1780_);
if (lean_obj_tag(v___x_1789_) == 0)
{
lean_object* v_a_1790_; lean_object* v___x_1792_; uint8_t v_isShared_1793_; uint8_t v_isSharedCheck_1809_; 
v_a_1790_ = lean_ctor_get(v___x_1789_, 0);
v_isSharedCheck_1809_ = !lean_is_exclusive(v___x_1789_);
if (v_isSharedCheck_1809_ == 0)
{
v___x_1792_ = v___x_1789_;
v_isShared_1793_ = v_isSharedCheck_1809_;
goto v_resetjp_1791_;
}
else
{
lean_inc(v_a_1790_);
lean_dec(v___x_1789_);
v___x_1792_ = lean_box(0);
v_isShared_1793_ = v_isSharedCheck_1809_;
goto v_resetjp_1791_;
}
v_resetjp_1791_:
{
if (lean_obj_tag(v_a_1790_) == 0)
{
lean_object* v___x_1794_; lean_object* v___x_1796_; 
v___x_1794_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1794_, 0, v_a_1790_);
if (v_isShared_1787_ == 0)
{
lean_ctor_set(v___x_1786_, 0, v___x_1794_);
v___x_1796_ = v___x_1786_;
goto v_reusejp_1795_;
}
else
{
lean_object* v_reuseFailAlloc_1800_; 
v_reuseFailAlloc_1800_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1800_, 0, v___x_1794_);
lean_ctor_set(v_reuseFailAlloc_1800_, 1, v_snd_1784_);
v___x_1796_ = v_reuseFailAlloc_1800_;
goto v_reusejp_1795_;
}
v_reusejp_1795_:
{
lean_object* v___x_1798_; 
if (v_isShared_1793_ == 0)
{
lean_ctor_set(v___x_1792_, 0, v___x_1796_);
v___x_1798_ = v___x_1792_;
goto v_reusejp_1797_;
}
else
{
lean_object* v_reuseFailAlloc_1799_; 
v_reuseFailAlloc_1799_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1799_, 0, v___x_1796_);
v___x_1798_ = v_reuseFailAlloc_1799_;
goto v_reusejp_1797_;
}
v_reusejp_1797_:
{
return v___x_1798_;
}
}
}
else
{
lean_object* v_a_1801_; lean_object* v___x_1802_; lean_object* v___x_1804_; 
lean_del_object(v___x_1792_);
lean_dec(v_snd_1784_);
v_a_1801_ = lean_ctor_get(v_a_1790_, 0);
lean_inc(v_a_1801_);
lean_dec_ref_known(v_a_1790_, 1);
v___x_1802_ = lean_box(0);
if (v_isShared_1787_ == 0)
{
lean_ctor_set(v___x_1786_, 1, v_a_1801_);
lean_ctor_set(v___x_1786_, 0, v___x_1802_);
v___x_1804_ = v___x_1786_;
goto v_reusejp_1803_;
}
else
{
lean_object* v_reuseFailAlloc_1808_; 
v_reuseFailAlloc_1808_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1808_, 0, v___x_1802_);
lean_ctor_set(v_reuseFailAlloc_1808_, 1, v_a_1801_);
v___x_1804_ = v_reuseFailAlloc_1808_;
goto v_reusejp_1803_;
}
v_reusejp_1803_:
{
size_t v___x_1805_; size_t v___x_1806_; 
v___x_1805_ = ((size_t)1ULL);
v___x_1806_ = lean_usize_add(v_i_1777_, v___x_1805_);
v_i_1777_ = v___x_1806_;
v_b_1778_ = v___x_1804_;
goto _start;
}
}
}
}
else
{
lean_object* v_a_1810_; lean_object* v___x_1812_; uint8_t v_isShared_1813_; uint8_t v_isSharedCheck_1817_; 
lean_del_object(v___x_1786_);
lean_dec(v_snd_1784_);
v_a_1810_ = lean_ctor_get(v___x_1789_, 0);
v_isSharedCheck_1817_ = !lean_is_exclusive(v___x_1789_);
if (v_isSharedCheck_1817_ == 0)
{
v___x_1812_ = v___x_1789_;
v_isShared_1813_ = v_isSharedCheck_1817_;
goto v_resetjp_1811_;
}
else
{
lean_inc(v_a_1810_);
lean_dec(v___x_1789_);
v___x_1812_ = lean_box(0);
v_isShared_1813_ = v_isSharedCheck_1817_;
goto v_resetjp_1811_;
}
v_resetjp_1811_:
{
lean_object* v___x_1815_; 
if (v_isShared_1813_ == 0)
{
v___x_1815_ = v___x_1812_;
goto v_reusejp_1814_;
}
else
{
lean_object* v_reuseFailAlloc_1816_; 
v_reuseFailAlloc_1816_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1816_, 0, v_a_1810_);
v___x_1815_ = v_reuseFailAlloc_1816_;
goto v_reusejp_1814_;
}
v_reusejp_1814_:
{
return v___x_1815_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__4_spec__6_spec__8___boxed(lean_object* v_init_1820_, lean_object* v_as_1821_, lean_object* v_sz_1822_, lean_object* v_i_1823_, lean_object* v_b_1824_, lean_object* v___y_1825_, lean_object* v___y_1826_, lean_object* v___y_1827_){
_start:
{
size_t v_sz_boxed_1828_; size_t v_i_boxed_1829_; lean_object* v_res_1830_; 
v_sz_boxed_1828_ = lean_unbox_usize(v_sz_1822_);
lean_dec(v_sz_1822_);
v_i_boxed_1829_ = lean_unbox_usize(v_i_1823_);
lean_dec(v_i_1823_);
v_res_1830_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__4_spec__6_spec__8(v_init_1820_, v_as_1821_, v_sz_boxed_1828_, v_i_boxed_1829_, v_b_1824_, v___y_1825_, v___y_1826_);
lean_dec(v___y_1826_);
lean_dec_ref(v___y_1825_);
lean_dec_ref(v_as_1821_);
return v_res_1830_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__4_spec__6___boxed(lean_object* v_init_1831_, lean_object* v_n_1832_, lean_object* v_b_1833_, lean_object* v___y_1834_, lean_object* v___y_1835_, lean_object* v___y_1836_){
_start:
{
lean_object* v_res_1837_; 
v_res_1837_ = lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__4_spec__6(v_init_1831_, v_n_1832_, v_b_1833_, v___y_1834_, v___y_1835_);
lean_dec(v___y_1835_);
lean_dec_ref(v___y_1834_);
lean_dec_ref(v_n_1832_);
return v_res_1837_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__4(lean_object* v_t_1838_, lean_object* v_init_1839_, lean_object* v___y_1840_, lean_object* v___y_1841_){
_start:
{
lean_object* v_root_1843_; lean_object* v_tail_1844_; lean_object* v___x_1845_; 
v_root_1843_ = lean_ctor_get(v_t_1838_, 0);
v_tail_1844_ = lean_ctor_get(v_t_1838_, 1);
v___x_1845_ = lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__4_spec__6(v_init_1839_, v_root_1843_, v_init_1839_, v___y_1840_, v___y_1841_);
if (lean_obj_tag(v___x_1845_) == 0)
{
lean_object* v_a_1846_; lean_object* v___x_1848_; uint8_t v_isShared_1849_; uint8_t v_isSharedCheck_1882_; 
v_a_1846_ = lean_ctor_get(v___x_1845_, 0);
v_isSharedCheck_1882_ = !lean_is_exclusive(v___x_1845_);
if (v_isSharedCheck_1882_ == 0)
{
v___x_1848_ = v___x_1845_;
v_isShared_1849_ = v_isSharedCheck_1882_;
goto v_resetjp_1847_;
}
else
{
lean_inc(v_a_1846_);
lean_dec(v___x_1845_);
v___x_1848_ = lean_box(0);
v_isShared_1849_ = v_isSharedCheck_1882_;
goto v_resetjp_1847_;
}
v_resetjp_1847_:
{
if (lean_obj_tag(v_a_1846_) == 0)
{
lean_object* v_a_1850_; lean_object* v___x_1852_; 
v_a_1850_ = lean_ctor_get(v_a_1846_, 0);
lean_inc(v_a_1850_);
lean_dec_ref_known(v_a_1846_, 1);
if (v_isShared_1849_ == 0)
{
lean_ctor_set(v___x_1848_, 0, v_a_1850_);
v___x_1852_ = v___x_1848_;
goto v_reusejp_1851_;
}
else
{
lean_object* v_reuseFailAlloc_1853_; 
v_reuseFailAlloc_1853_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1853_, 0, v_a_1850_);
v___x_1852_ = v_reuseFailAlloc_1853_;
goto v_reusejp_1851_;
}
v_reusejp_1851_:
{
return v___x_1852_;
}
}
else
{
lean_object* v_a_1854_; lean_object* v___x_1855_; lean_object* v___x_1856_; size_t v_sz_1857_; size_t v___x_1858_; lean_object* v___x_1859_; 
lean_del_object(v___x_1848_);
v_a_1854_ = lean_ctor_get(v_a_1846_, 0);
lean_inc(v_a_1854_);
lean_dec_ref_known(v_a_1846_, 1);
v___x_1855_ = lean_box(0);
v___x_1856_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1856_, 0, v___x_1855_);
lean_ctor_set(v___x_1856_, 1, v_a_1854_);
v_sz_1857_ = lean_array_size(v_tail_1844_);
v___x_1858_ = ((size_t)0ULL);
v___x_1859_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__4_spec__7(v_tail_1844_, v_sz_1857_, v___x_1858_, v___x_1856_, v___y_1840_, v___y_1841_);
if (lean_obj_tag(v___x_1859_) == 0)
{
lean_object* v_a_1860_; lean_object* v___x_1862_; uint8_t v_isShared_1863_; uint8_t v_isSharedCheck_1873_; 
v_a_1860_ = lean_ctor_get(v___x_1859_, 0);
v_isSharedCheck_1873_ = !lean_is_exclusive(v___x_1859_);
if (v_isSharedCheck_1873_ == 0)
{
v___x_1862_ = v___x_1859_;
v_isShared_1863_ = v_isSharedCheck_1873_;
goto v_resetjp_1861_;
}
else
{
lean_inc(v_a_1860_);
lean_dec(v___x_1859_);
v___x_1862_ = lean_box(0);
v_isShared_1863_ = v_isSharedCheck_1873_;
goto v_resetjp_1861_;
}
v_resetjp_1861_:
{
lean_object* v_fst_1864_; 
v_fst_1864_ = lean_ctor_get(v_a_1860_, 0);
if (lean_obj_tag(v_fst_1864_) == 0)
{
lean_object* v_snd_1865_; lean_object* v___x_1867_; 
v_snd_1865_ = lean_ctor_get(v_a_1860_, 1);
lean_inc(v_snd_1865_);
lean_dec(v_a_1860_);
if (v_isShared_1863_ == 0)
{
lean_ctor_set(v___x_1862_, 0, v_snd_1865_);
v___x_1867_ = v___x_1862_;
goto v_reusejp_1866_;
}
else
{
lean_object* v_reuseFailAlloc_1868_; 
v_reuseFailAlloc_1868_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1868_, 0, v_snd_1865_);
v___x_1867_ = v_reuseFailAlloc_1868_;
goto v_reusejp_1866_;
}
v_reusejp_1866_:
{
return v___x_1867_;
}
}
else
{
lean_object* v_val_1869_; lean_object* v___x_1871_; 
lean_inc_ref(v_fst_1864_);
lean_dec(v_a_1860_);
v_val_1869_ = lean_ctor_get(v_fst_1864_, 0);
lean_inc(v_val_1869_);
lean_dec_ref_known(v_fst_1864_, 1);
if (v_isShared_1863_ == 0)
{
lean_ctor_set(v___x_1862_, 0, v_val_1869_);
v___x_1871_ = v___x_1862_;
goto v_reusejp_1870_;
}
else
{
lean_object* v_reuseFailAlloc_1872_; 
v_reuseFailAlloc_1872_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1872_, 0, v_val_1869_);
v___x_1871_ = v_reuseFailAlloc_1872_;
goto v_reusejp_1870_;
}
v_reusejp_1870_:
{
return v___x_1871_;
}
}
}
}
else
{
lean_object* v_a_1874_; lean_object* v___x_1876_; uint8_t v_isShared_1877_; uint8_t v_isSharedCheck_1881_; 
v_a_1874_ = lean_ctor_get(v___x_1859_, 0);
v_isSharedCheck_1881_ = !lean_is_exclusive(v___x_1859_);
if (v_isSharedCheck_1881_ == 0)
{
v___x_1876_ = v___x_1859_;
v_isShared_1877_ = v_isSharedCheck_1881_;
goto v_resetjp_1875_;
}
else
{
lean_inc(v_a_1874_);
lean_dec(v___x_1859_);
v___x_1876_ = lean_box(0);
v_isShared_1877_ = v_isSharedCheck_1881_;
goto v_resetjp_1875_;
}
v_resetjp_1875_:
{
lean_object* v___x_1879_; 
if (v_isShared_1877_ == 0)
{
v___x_1879_ = v___x_1876_;
goto v_reusejp_1878_;
}
else
{
lean_object* v_reuseFailAlloc_1880_; 
v_reuseFailAlloc_1880_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1880_, 0, v_a_1874_);
v___x_1879_ = v_reuseFailAlloc_1880_;
goto v_reusejp_1878_;
}
v_reusejp_1878_:
{
return v___x_1879_;
}
}
}
}
}
}
else
{
lean_object* v_a_1883_; lean_object* v___x_1885_; uint8_t v_isShared_1886_; uint8_t v_isSharedCheck_1890_; 
v_a_1883_ = lean_ctor_get(v___x_1845_, 0);
v_isSharedCheck_1890_ = !lean_is_exclusive(v___x_1845_);
if (v_isSharedCheck_1890_ == 0)
{
v___x_1885_ = v___x_1845_;
v_isShared_1886_ = v_isSharedCheck_1890_;
goto v_resetjp_1884_;
}
else
{
lean_inc(v_a_1883_);
lean_dec(v___x_1845_);
v___x_1885_ = lean_box(0);
v_isShared_1886_ = v_isSharedCheck_1890_;
goto v_resetjp_1884_;
}
v_resetjp_1884_:
{
lean_object* v___x_1888_; 
if (v_isShared_1886_ == 0)
{
v___x_1888_ = v___x_1885_;
goto v_reusejp_1887_;
}
else
{
lean_object* v_reuseFailAlloc_1889_; 
v_reuseFailAlloc_1889_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1889_, 0, v_a_1883_);
v___x_1888_ = v_reuseFailAlloc_1889_;
goto v_reusejp_1887_;
}
v_reusejp_1887_:
{
return v___x_1888_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__4___boxed(lean_object* v_t_1891_, lean_object* v_init_1892_, lean_object* v___y_1893_, lean_object* v___y_1894_, lean_object* v___y_1895_){
_start:
{
lean_object* v_res_1896_; 
v_res_1896_ = lp_mathlib_Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__4(v_t_1891_, v_init_1892_, v___y_1893_, v___y_1894_);
lean_dec(v___y_1894_);
lean_dec_ref(v___y_1893_);
lean_dec_ref(v_t_1891_);
return v_res_1896_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__0_spec__0___redArg(lean_object* v_o_1897_, lean_object* v___y_1898_){
_start:
{
lean_object* v___x_1900_; lean_object* v_env_1901_; lean_object* v___x_1902_; lean_object* v_toEnvExtension_1903_; lean_object* v_asyncMode_1904_; lean_object* v___x_1905_; lean_object* v___x_1906_; lean_object* v___x_1907_; lean_object* v_merged_1908_; lean_object* v___x_1910_; uint8_t v_isShared_1911_; uint8_t v_isSharedCheck_1916_; 
v___x_1900_ = lean_st_ref_get(v___y_1898_);
v_env_1901_ = lean_ctor_get(v___x_1900_, 0);
lean_inc_ref(v_env_1901_);
lean_dec(v___x_1900_);
v___x_1902_ = l_Lean_Linter_linterSetsExt;
v_toEnvExtension_1903_ = lean_ctor_get(v___x_1902_, 0);
v_asyncMode_1904_ = lean_ctor_get(v_toEnvExtension_1903_, 2);
v___x_1905_ = l_Lean_Linter_instInhabitedLinterSetsState_default;
v___x_1906_ = lean_box(0);
v___x_1907_ = l_Lean_PersistentEnvExtension_getState___redArg(v___x_1905_, v___x_1902_, v_env_1901_, v_asyncMode_1904_, v___x_1906_);
v_merged_1908_ = lean_ctor_get(v___x_1907_, 0);
v_isSharedCheck_1916_ = !lean_is_exclusive(v___x_1907_);
if (v_isSharedCheck_1916_ == 0)
{
lean_object* v_unused_1917_; 
v_unused_1917_ = lean_ctor_get(v___x_1907_, 1);
lean_dec(v_unused_1917_);
v___x_1910_ = v___x_1907_;
v_isShared_1911_ = v_isSharedCheck_1916_;
goto v_resetjp_1909_;
}
else
{
lean_inc(v_merged_1908_);
lean_dec(v___x_1907_);
v___x_1910_ = lean_box(0);
v_isShared_1911_ = v_isSharedCheck_1916_;
goto v_resetjp_1909_;
}
v_resetjp_1909_:
{
lean_object* v___x_1913_; 
if (v_isShared_1911_ == 0)
{
lean_ctor_set(v___x_1910_, 1, v_merged_1908_);
lean_ctor_set(v___x_1910_, 0, v_o_1897_);
v___x_1913_ = v___x_1910_;
goto v_reusejp_1912_;
}
else
{
lean_object* v_reuseFailAlloc_1915_; 
v_reuseFailAlloc_1915_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1915_, 0, v_o_1897_);
lean_ctor_set(v_reuseFailAlloc_1915_, 1, v_merged_1908_);
v___x_1913_ = v_reuseFailAlloc_1915_;
goto v_reusejp_1912_;
}
v_reusejp_1912_:
{
lean_object* v___x_1914_; 
v___x_1914_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1914_, 0, v___x_1913_);
return v___x_1914_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__0_spec__0___redArg___boxed(lean_object* v_o_1918_, lean_object* v___y_1919_, lean_object* v___y_1920_){
_start:
{
lean_object* v_res_1921_; 
v_res_1921_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__0_spec__0___redArg(v_o_1918_, v___y_1919_);
lean_dec(v___y_1919_);
return v_res_1921_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__0(lean_object* v___y_1922_, lean_object* v___y_1923_){
_start:
{
lean_object* v___x_1925_; lean_object* v_scopes_1926_; lean_object* v___x_1927_; lean_object* v___x_1928_; lean_object* v_opts_1929_; lean_object* v___x_1930_; 
v___x_1925_ = lean_st_ref_get(v___y_1923_);
v_scopes_1926_ = lean_ctor_get(v___x_1925_, 2);
lean_inc(v_scopes_1926_);
lean_dec(v___x_1925_);
v___x_1927_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_1928_ = l_List_head_x21___redArg(v___x_1927_, v_scopes_1926_);
lean_dec(v_scopes_1926_);
v_opts_1929_ = lean_ctor_get(v___x_1928_, 1);
lean_inc_ref(v_opts_1929_);
lean_dec(v___x_1928_);
v___x_1930_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__0_spec__0___redArg(v_opts_1929_, v___y_1923_);
return v___x_1930_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__0___boxed(lean_object* v___y_1931_, lean_object* v___y_1932_, lean_object* v___y_1933_){
_start:
{
lean_object* v_res_1934_; 
v_res_1934_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__0(v___y_1931_, v___y_1932_);
lean_dec(v___y_1932_);
lean_dec_ref(v___y_1931_);
return v_res_1934_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___lam__0(lean_object* v___stx_1935_, lean_object* v___y_1936_, lean_object* v___y_1937_){
_start:
{
lean_object* v___x_1939_; lean_object* v_a_1940_; lean_object* v___x_1942_; uint8_t v_isShared_1943_; uint8_t v_isSharedCheck_1969_; 
v___x_1939_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__0(v___y_1936_, v___y_1937_);
v_a_1940_ = lean_ctor_get(v___x_1939_, 0);
v_isSharedCheck_1969_ = !lean_is_exclusive(v___x_1939_);
if (v_isSharedCheck_1969_ == 0)
{
v___x_1942_ = v___x_1939_;
v_isShared_1943_ = v_isSharedCheck_1969_;
goto v_resetjp_1941_;
}
else
{
lean_inc(v_a_1940_);
lean_dec(v___x_1939_);
v___x_1942_ = lean_box(0);
v_isShared_1943_ = v_isSharedCheck_1969_;
goto v_resetjp_1941_;
}
v_resetjp_1941_:
{
lean_object* v___x_1944_; uint8_t v___x_1945_; 
v___x_1944_ = lp_mathlib_Mathlib_Linter_linter_style_multiGoal;
v___x_1945_ = l_Lean_Linter_getLinterValue(v___x_1944_, v_a_1940_);
lean_dec(v_a_1940_);
if (v___x_1945_ == 0)
{
lean_object* v___x_1946_; lean_object* v___x_1948_; 
v___x_1946_ = lean_box(0);
if (v_isShared_1943_ == 0)
{
lean_ctor_set(v___x_1942_, 0, v___x_1946_);
v___x_1948_ = v___x_1942_;
goto v_reusejp_1947_;
}
else
{
lean_object* v_reuseFailAlloc_1949_; 
v_reuseFailAlloc_1949_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1949_, 0, v___x_1946_);
v___x_1948_ = v_reuseFailAlloc_1949_;
goto v_reusejp_1947_;
}
v_reusejp_1947_:
{
return v___x_1948_;
}
}
else
{
lean_object* v___x_1950_; lean_object* v_messages_1951_; uint8_t v___x_1952_; 
v___x_1950_ = lean_st_ref_get(v___y_1937_);
v_messages_1951_ = lean_ctor_get(v___x_1950_, 1);
lean_inc_ref(v_messages_1951_);
lean_dec(v___x_1950_);
v___x_1952_ = l_Lean_MessageLog_hasErrors(v_messages_1951_);
lean_dec_ref(v_messages_1951_);
if (v___x_1952_ == 0)
{
lean_object* v___x_1953_; lean_object* v_a_1954_; lean_object* v___x_1955_; lean_object* v___x_1956_; 
lean_del_object(v___x_1942_);
v___x_1953_ = lp_mathlib_Lean_Elab_getInfoTrees___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__1___redArg(v___y_1937_);
v_a_1954_ = lean_ctor_get(v___x_1953_, 0);
lean_inc(v_a_1954_);
lean_dec_ref(v___x_1953_);
v___x_1955_ = lean_box(0);
v___x_1956_ = lp_mathlib_Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__4(v_a_1954_, v___x_1955_, v___y_1936_, v___y_1937_);
lean_dec(v_a_1954_);
if (lean_obj_tag(v___x_1956_) == 0)
{
lean_object* v___x_1958_; uint8_t v_isShared_1959_; uint8_t v_isSharedCheck_1963_; 
v_isSharedCheck_1963_ = !lean_is_exclusive(v___x_1956_);
if (v_isSharedCheck_1963_ == 0)
{
lean_object* v_unused_1964_; 
v_unused_1964_ = lean_ctor_get(v___x_1956_, 0);
lean_dec(v_unused_1964_);
v___x_1958_ = v___x_1956_;
v_isShared_1959_ = v_isSharedCheck_1963_;
goto v_resetjp_1957_;
}
else
{
lean_dec(v___x_1956_);
v___x_1958_ = lean_box(0);
v_isShared_1959_ = v_isSharedCheck_1963_;
goto v_resetjp_1957_;
}
v_resetjp_1957_:
{
lean_object* v___x_1961_; 
if (v_isShared_1959_ == 0)
{
lean_ctor_set(v___x_1958_, 0, v___x_1955_);
v___x_1961_ = v___x_1958_;
goto v_reusejp_1960_;
}
else
{
lean_object* v_reuseFailAlloc_1962_; 
v_reuseFailAlloc_1962_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1962_, 0, v___x_1955_);
v___x_1961_ = v_reuseFailAlloc_1962_;
goto v_reusejp_1960_;
}
v_reusejp_1960_:
{
return v___x_1961_;
}
}
}
else
{
return v___x_1956_;
}
}
else
{
lean_object* v___x_1965_; lean_object* v___x_1967_; 
v___x_1965_ = lean_box(0);
if (v_isShared_1943_ == 0)
{
lean_ctor_set(v___x_1942_, 0, v___x_1965_);
v___x_1967_ = v___x_1942_;
goto v_reusejp_1966_;
}
else
{
lean_object* v_reuseFailAlloc_1968_; 
v_reuseFailAlloc_1968_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1968_, 0, v___x_1965_);
v___x_1967_ = v_reuseFailAlloc_1968_;
goto v_reusejp_1966_;
}
v_reusejp_1966_:
{
return v___x_1967_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___lam__0___boxed(lean_object* v___stx_1970_, lean_object* v___y_1971_, lean_object* v___y_1972_, lean_object* v___y_1973_){
_start:
{
lean_object* v_res_1974_; 
v_res_1974_ = lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter___lam__0(v___stx_1970_, v___y_1971_, v___y_1972_);
lean_dec(v___y_1972_);
lean_dec_ref(v___y_1971_);
lean_dec(v___stx_1970_);
return v_res_1974_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__0_spec__0(lean_object* v_o_2019_, lean_object* v___y_2020_, lean_object* v___y_2021_){
_start:
{
lean_object* v___x_2023_; 
v___x_2023_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__0_spec__0___redArg(v_o_2019_, v___y_2021_);
return v___x_2023_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__0_spec__0___boxed(lean_object* v_o_2024_, lean_object* v___y_2025_, lean_object* v___y_2026_, lean_object* v___y_2027_){
_start:
{
lean_object* v_res_2028_; 
v_res_2028_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__0_spec__0(v_o_2024_, v___y_2025_, v___y_2026_);
lean_dec(v___y_2026_);
lean_dec_ref(v___y_2025_);
return v_res_2028_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4_spec__7(lean_object* v_msgData_2029_, lean_object* v___y_2030_, lean_object* v___y_2031_){
_start:
{
lean_object* v___x_2033_; 
v___x_2033_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4_spec__7___redArg(v_msgData_2029_, v___y_2031_);
return v___x_2033_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4_spec__7___boxed(lean_object* v_msgData_2034_, lean_object* v___y_2035_, lean_object* v___y_2036_, lean_object* v___y_2037_){
_start:
{
lean_object* v_res_2038_; 
v_res_2038_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter_spec__2_spec__3_spec__4_spec__7(v_msgData_2034_, v___y_2035_, v___y_2036_);
lean_dec(v___y_2036_);
lean_dec_ref(v___y_2035_);
return v_res_2038_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_initFn_00___x40_Mathlib_Tactic_Linter_Multigoal_276707340____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_2040_; lean_object* v___x_2041_; 
v___x_2040_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_multiGoalLinter));
v___x_2041_ = l_Lean_Elab_Command_addLinter(v___x_2040_);
return v___x_2041_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_initFn_00___x40_Mathlib_Tactic_Linter_Multigoal_276707340____hygCtx___hyg_2____boxed(lean_object* v_a_2042_){
_start:
{
lean_object* v_res_2043_; 
v_res_2043_ = lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_initFn_00___x40_Mathlib_Tactic_Linter_Multigoal_276707340____hygCtx___hyg_2_();
return v_res_2043_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Parser_Term(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_Multigoal(uint8_t builtin) {
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
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_Header(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Linter_Multigoal(uint8_t builtin) {
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
res = lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Multigoal_4046044826____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Linter_linter_style_multiGoal = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Linter_linter_style_multiGoal);
lean_dec_ref(res);
lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions = _init_lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_exclusions);
lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch = _init_lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_ignoreBranch);
res = lp_mathlib___private_Mathlib_Tactic_Linter_Multigoal_0__Mathlib_Linter_Style_multiGoal_initFn_00___x40_Mathlib_Tactic_Linter_Multigoal_276707340____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Command(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_Header(uint8_t builtin);
lean_object* initialize_Lean_Parser_Term(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Linter_Multigoal(uint8_t builtin) {
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
res = initialize_Lean_Parser_Term(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_Multigoal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Linter_Multigoal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Linter_Multigoal(builtin);
}
#ifdef __cplusplus
}
#endif
