// Lean compiler output
// Module: Mathlib.Tactic.Linter.HaveLetLinter
// Imports: public import Init public meta import Init public import Mathlib.Init public meta import Lean.Elab.Command public meta import Lean.Server.InfoUtils
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
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_add(size_t, size_t);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* l_Lean_Name_toString(lean_object*, uint8_t);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_isProp(lean_object*);
lean_object* l_Lean_Meta_ppExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_Lean_LocalDecl_fvarId(lean_object*);
lean_object* l_Lean_Elab_ContextInfo_runMetaM___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_io_error_to_string(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_Elab_Command_getScope___redArg(lean_object*);
lean_object* l_Lean_MessageLog_add(lean_object*, lean_object*);
lean_object* l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(lean_object*);
extern lean_object* l_Lean_Elab_Command_instInhabitedScope_default;
lean_object* l_List_head_x21___redArg(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
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
lean_object* l_Lean_Elab_InfoTree_foldInfo___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_register_option(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getHeadInfo(lean_object*);
lean_object* lean_array_to_list(lean_object*);
uint64_t l_Lean_instHashableMVarId_hash(lean_object*);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_usize_to_nat(size_t);
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Lean_PersistentArray_toList___redArg(lean_object*);
lean_object* l_List_foldl___at___00Array_appendList_spec__0___redArg(lean_object*, lean_object*);
uint8_t l_Lean_instBEqFVarId_beq(lean_object*, lean_object*);
lean_object* l_Lean_LocalDecl_type(lean_object*);
lean_object* l_Lean_LocalDecl_userName(lean_object*);
lean_object* lean_array_mk(lean_object*);
extern lean_object* l_Lean_instInhabitedMetavarDecl_default;
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* l_Lean_MessageData_note(lean_object*);
uint8_t l_Lean_PersistentArray_isEmpty___redArg(lean_object*);
lean_object* l_Lean_withSetOptionIn___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_addLinter(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4__spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "haveLet"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(186, 218, 113, 226, 101, 176, 32, 79)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(138, 221, 202, 13, 117, 178, 153, 230)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 124, .m_capacity = 124, .m_length = 123, .m_data = "enable the `have` vs `let` linter:\n* 0 -- inactive;\n* 1 -- active only on noisy declarations;\n* 2 or more -- always active."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(120, 131, 127, 204, 79, 169, 80, 92)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(101, 237, 90, 120, 51, 59, 46, 172)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(129, 83, 33, 215, 4, 254, 93, 246)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_linter_haveLet;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_isHave_x3f___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_isHave_x3f___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_isHave_x3f___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_isHave_x3f___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_isHave_x3f___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_isHave_x3f___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_isHave_x3f___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_isHave_x3f___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_isHave_x3f___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_isHave_x3f___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "tacticHave__"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_isHave_x3f___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_isHave_x3f___closed__3_value;
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_isHave_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_isHave_x3f___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_InfoTree_foldInfoM___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_InfoTree_foldInfoM___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_InfoTree_foldInfoM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_InfoTree_foldInfoM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_toFormat__propTypes_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_toFormat__propTypes_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_toFormat__propTypes_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_toFormat__propTypes_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_toFormat__propTypes_spec__1_spec__1(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_toFormat__propTypes_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_toFormat__propTypes_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_toFormat__propTypes_spec__1___closed__0 = (const lean_object*)&lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_toFormat__propTypes_spec__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_toFormat__propTypes_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_toFormat__propTypes_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_toFormat__propTypes___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_toFormat__propTypes___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_toFormat__propTypes(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_toFormat__propTypes___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_InfoTree_foldInfoM___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__11___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_InfoTree_foldInfoM___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__11___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_InfoTree_foldInfoM___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__11___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_InfoTree_foldInfoM___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__11___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_InfoTree_foldInfoM___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__11___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_InfoTree_foldInfoM___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__11___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_InfoTree_foldInfoM___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__11(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_InfoTree_foldInfoM___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__8(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_elem___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__6(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_elem___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__6___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_filterTR_loop___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__7(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_filterTR_loop___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_filterMapTR_go___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__0(lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__9___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = " : "};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__9___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__9___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__9___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__9___closed__0_value)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__9___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__9___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__9(uint8_t, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__5(lean_object*, lean_object*);
static const lean_array_object lp_mathlib_List_mapTR_loop___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__3___closed__0 = (const lean_object*)&lp_mathlib_List_mapTR_loop___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__3___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__1_spec__1_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__1_spec__1_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__1_spec__1___redArg(lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_filterMapTR_go___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_filterMapTR_go___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_getMax_x3f___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__10_spec__11(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_getMax_x3f___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__10_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_getMax_x3f___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__10(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_getMax_x3f___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__10___boxed(lean_object*);
static const lean_array_object lp_mathlib_Mathlib_Linter_haveLet_nonPropHaves___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Linter_haveLet_nonPropHaves___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Linter_haveLet_nonPropHaves___lam__0___closed__0_value;
static const lean_array_object lp_mathlib_Mathlib_Linter_haveLet_nonPropHaves___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Linter_haveLet_nonPropHaves___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Linter_haveLet_nonPropHaves___lam__0___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_haveLet_nonPropHaves___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_haveLet_nonPropHaves___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Linter_haveLet_nonPropHaves___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Linter_haveLet_nonPropHaves___lam__0___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Linter_haveLet_nonPropHaves___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Linter_haveLet_nonPropHaves___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_haveLet_nonPropHaves(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_haveLet_nonPropHaves___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__1_spec__1(lean_object*, lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__1_spec__1_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__1_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getInfoTrees___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getInfoTrees___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getInfoTrees___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getInfoTrees___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__1___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3_spec__6___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3_spec__6___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3_spec__6___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3_spec__6___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3_spec__6___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3_spec__6___redArg___closed__2;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3_spec__6___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3_spec__6___redArg___closed__3;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3_spec__6___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3_spec__6___redArg___closed__4;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3_spec__6___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3_spec__6___redArg___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3_spec__6___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3___lam__0___closed__0_value;
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3_spec__7(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3_spec__7___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 46, .m_capacity = 46, .m_length = 45, .m_data = "This linter can be disabled with `set_option "};
static const lean_object* lp_mathlib_Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = " 0`"};
static const lean_object* lp_mathlib_Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "'"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__3___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__3___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__3___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__3___closed__1;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__3___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 68, .m_capacity = 68, .m_length = 67, .m_data = "' is a Type and not a Prop. Consider using 'let' instead of 'have'."};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__3___closed__2 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__3___closed__2_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__3___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__3___closed__3;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__3(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__4_spec__6_spec__10___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__4_spec__6_spec__10___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__4_spec__6_spec__10___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__4_spec__6_spec__10(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__4_spec__6_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__4_spec__6(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__4_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__4_spec__5_spec__8_spec__11___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__4_spec__5_spec__8_spec__11___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__4_spec__5_spec__8_spec__11___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__4_spec__5_spec__8_spec__11(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__4_spec__5_spec__8_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__4_spec__5_spec__8(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__4_spec__5_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__4_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__4_spec__5_spec__7(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__4_spec__5_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__4_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___lam__0___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__0_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_withSetOptionIn___boxed, .m_arity = 6, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__0_value)} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__2_value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__4_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_isHave_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(191, 70, 156, 159, 11, 54, 216, 94)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__5_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(37, 204, 154, 235, 250, 222, 148, 114)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__6_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "HaveLetLinter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__6_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__7_value),LEAN_SCALAR_PTR_LITERAL(85, 90, 217, 157, 124, 151, 161, 87)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(192, 94, 106, 46, 12, 243, 246, 91)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__9_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(201, 229, 138, 85, 43, 170, 70, 85)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__10_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(179, 88, 22, 46, 228, 197, 24, 94)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__11_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(199, 189, 0, 117, 21, 252, 66, 150)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__12_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "haveLetLinter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__13_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__12_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__13_value),LEAN_SCALAR_PTR_LITERAL(100, 240, 12, 197, 79, 133, 127, 166)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__14_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__14_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__15_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___closed__15_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3_spec__6(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_initFn_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_3906617390____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_initFn_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_3906617390____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4__spec__0(lean_object* v_name_1_, lean_object* v_decl_2_, lean_object* v_ref_3_){
_start:
{
lean_object* v_defValue_5_; lean_object* v_descr_6_; lean_object* v_deprecation_x3f_7_; lean_object* v___x_8_; lean_object* v___x_9_; lean_object* v___x_10_; 
v_defValue_5_ = lean_ctor_get(v_decl_2_, 0);
v_descr_6_ = lean_ctor_get(v_decl_2_, 1);
v_deprecation_x3f_7_ = lean_ctor_get(v_decl_2_, 2);
lean_inc(v_defValue_5_);
v___x_8_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_8_, 0, v_defValue_5_);
lean_inc(v_deprecation_x3f_7_);
lean_inc_ref(v_descr_6_);
lean_inc_n(v_name_1_, 2);
v___x_9_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_9_, 0, v_name_1_);
lean_ctor_set(v___x_9_, 1, v_ref_3_);
lean_ctor_set(v___x_9_, 2, v___x_8_);
lean_ctor_set(v___x_9_, 3, v_descr_6_);
lean_ctor_set(v___x_9_, 4, v_deprecation_x3f_7_);
v___x_10_ = lean_register_option(v_name_1_, v___x_9_);
if (lean_obj_tag(v___x_10_) == 0)
{
lean_object* v___x_12_; uint8_t v_isShared_13_; uint8_t v_isSharedCheck_18_; 
v_isSharedCheck_18_ = !lean_is_exclusive(v___x_10_);
if (v_isSharedCheck_18_ == 0)
{
lean_object* v_unused_19_; 
v_unused_19_ = lean_ctor_get(v___x_10_, 0);
lean_dec(v_unused_19_);
v___x_12_ = v___x_10_;
v_isShared_13_ = v_isSharedCheck_18_;
goto v_resetjp_11_;
}
else
{
lean_dec(v___x_10_);
v___x_12_ = lean_box(0);
v_isShared_13_ = v_isSharedCheck_18_;
goto v_resetjp_11_;
}
v_resetjp_11_:
{
lean_object* v___x_14_; lean_object* v___x_16_; 
lean_inc(v_defValue_5_);
v___x_14_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_14_, 0, v_name_1_);
lean_ctor_set(v___x_14_, 1, v_defValue_5_);
if (v_isShared_13_ == 0)
{
lean_ctor_set(v___x_12_, 0, v___x_14_);
v___x_16_ = v___x_12_;
goto v_reusejp_15_;
}
else
{
lean_object* v_reuseFailAlloc_17_; 
v_reuseFailAlloc_17_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_17_, 0, v___x_14_);
v___x_16_ = v_reuseFailAlloc_17_;
goto v_reusejp_15_;
}
v_reusejp_15_:
{
return v___x_16_;
}
}
}
else
{
lean_object* v_a_20_; lean_object* v___x_22_; uint8_t v_isShared_23_; uint8_t v_isSharedCheck_27_; 
lean_dec(v_name_1_);
v_a_20_ = lean_ctor_get(v___x_10_, 0);
v_isSharedCheck_27_ = !lean_is_exclusive(v___x_10_);
if (v_isSharedCheck_27_ == 0)
{
v___x_22_ = v___x_10_;
v_isShared_23_ = v_isSharedCheck_27_;
goto v_resetjp_21_;
}
else
{
lean_inc(v_a_20_);
lean_dec(v___x_10_);
v___x_22_ = lean_box(0);
v_isShared_23_ = v_isSharedCheck_27_;
goto v_resetjp_21_;
}
v_resetjp_21_:
{
lean_object* v___x_25_; 
if (v_isShared_23_ == 0)
{
v___x_25_ = v___x_22_;
goto v_reusejp_24_;
}
else
{
lean_object* v_reuseFailAlloc_26_; 
v_reuseFailAlloc_26_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_26_, 0, v_a_20_);
v___x_25_ = v_reuseFailAlloc_26_;
goto v_reusejp_24_;
}
v_reusejp_24_:
{
return v___x_25_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4__spec__0___boxed(lean_object* v_name_28_, lean_object* v_decl_29_, lean_object* v_ref_30_, lean_object* v_a_31_){
_start:
{
lean_object* v_res_32_; 
v_res_32_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4__spec__0(v_name_28_, v_decl_29_, v_ref_30_);
lean_dec_ref(v_decl_29_);
return v_res_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_51_; lean_object* v___x_52_; lean_object* v___x_53_; lean_object* v___x_54_; 
v___x_51_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4_));
v___x_52_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4_));
v___x_53_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4_));
v___x_54_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4__spec__0(v___x_51_, v___x_52_, v___x_53_);
return v___x_54_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4____boxed(lean_object* v_a_55_){
_start:
{
lean_object* v_res_56_; 
v_res_56_ = lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4_();
return v_res_56_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_isHave_x3f(lean_object* v_x_61_){
_start:
{
if (lean_obj_tag(v_x_61_) == 1)
{
lean_object* v_kind_62_; 
v_kind_62_ = lean_ctor_get(v_x_61_, 1);
if (lean_obj_tag(v_kind_62_) == 1)
{
lean_object* v_pre_63_; 
v_pre_63_ = lean_ctor_get(v_kind_62_, 0);
if (lean_obj_tag(v_pre_63_) == 1)
{
lean_object* v_pre_64_; 
v_pre_64_ = lean_ctor_get(v_pre_63_, 0);
if (lean_obj_tag(v_pre_64_) == 1)
{
lean_object* v_pre_65_; 
v_pre_65_ = lean_ctor_get(v_pre_64_, 0);
if (lean_obj_tag(v_pre_65_) == 1)
{
lean_object* v_pre_66_; 
v_pre_66_ = lean_ctor_get(v_pre_65_, 0);
if (lean_obj_tag(v_pre_66_) == 0)
{
lean_object* v_str_67_; lean_object* v_str_68_; lean_object* v_str_69_; lean_object* v_str_70_; lean_object* v___x_71_; uint8_t v___x_72_; 
v_str_67_ = lean_ctor_get(v_kind_62_, 1);
v_str_68_ = lean_ctor_get(v_pre_63_, 1);
v_str_69_ = lean_ctor_get(v_pre_64_, 1);
v_str_70_ = lean_ctor_get(v_pre_65_, 1);
v___x_71_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_isHave_x3f___closed__0));
v___x_72_ = lean_string_dec_eq(v_str_70_, v___x_71_);
if (v___x_72_ == 0)
{
return v___x_72_;
}
else
{
lean_object* v___x_73_; uint8_t v___x_74_; 
v___x_73_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_isHave_x3f___closed__1));
v___x_74_ = lean_string_dec_eq(v_str_69_, v___x_73_);
if (v___x_74_ == 0)
{
return v___x_74_;
}
else
{
lean_object* v___x_75_; uint8_t v___x_76_; 
v___x_75_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_isHave_x3f___closed__2));
v___x_76_ = lean_string_dec_eq(v_str_68_, v___x_75_);
if (v___x_76_ == 0)
{
return v___x_76_;
}
else
{
lean_object* v___x_77_; uint8_t v___x_78_; 
v___x_77_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_isHave_x3f___closed__3));
v___x_78_ = lean_string_dec_eq(v_str_67_, v___x_77_);
return v___x_78_;
}
}
}
}
else
{
uint8_t v___x_79_; 
v___x_79_ = 0;
return v___x_79_;
}
}
else
{
uint8_t v___x_80_; 
v___x_80_ = 0;
return v___x_80_;
}
}
else
{
uint8_t v___x_81_; 
v___x_81_ = 0;
return v___x_81_;
}
}
else
{
uint8_t v___x_82_; 
v___x_82_ = 0;
return v___x_82_;
}
}
else
{
uint8_t v___x_83_; 
v___x_83_ = 0;
return v___x_83_;
}
}
else
{
uint8_t v___x_84_; 
v___x_84_ = 0;
return v___x_84_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_isHave_x3f___boxed(lean_object* v_x_85_){
_start:
{
uint8_t v_res_86_; lean_object* v_r_87_; 
v_res_86_ = lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_isHave_x3f(v_x_85_);
lean_dec(v_x_85_);
v_r_87_ = lean_box(v_res_86_);
return v_r_87_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_InfoTree_foldInfoM___redArg___lam__0(lean_object* v_f_88_, lean_object* v_ctx_89_, lean_object* v_i_90_, lean_object* v_____do__lift_91_){
_start:
{
lean_object* v___x_92_; 
v___x_92_ = lean_apply_3(v_f_88_, v_ctx_89_, v_i_90_, v_____do__lift_91_);
return v___x_92_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_InfoTree_foldInfoM___redArg___lam__1(lean_object* v_f_93_, lean_object* v_toBind_94_, lean_object* v_ctx_95_, lean_object* v_i_96_, lean_object* v_ma_97_){
_start:
{
lean_object* v___f_98_; lean_object* v___x_99_; 
v___f_98_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_InfoTree_foldInfoM___redArg___lam__0), 4, 3);
lean_closure_set(v___f_98_, 0, v_f_93_);
lean_closure_set(v___f_98_, 1, v_ctx_95_);
lean_closure_set(v___f_98_, 2, v_i_96_);
v___x_99_ = lean_apply_4(v_toBind_94_, lean_box(0), lean_box(0), v_ma_97_, v___f_98_);
return v___x_99_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_InfoTree_foldInfoM___redArg(lean_object* v_inst_100_, lean_object* v_f_101_, lean_object* v_init_102_, lean_object* v_a_103_){
_start:
{
lean_object* v_toApplicative_104_; lean_object* v_toBind_105_; lean_object* v_toPure_106_; lean_object* v___f_107_; lean_object* v___x_108_; lean_object* v___x_109_; 
v_toApplicative_104_ = lean_ctor_get(v_inst_100_, 0);
lean_inc_ref(v_toApplicative_104_);
v_toBind_105_ = lean_ctor_get(v_inst_100_, 1);
lean_inc(v_toBind_105_);
lean_dec_ref(v_inst_100_);
v_toPure_106_ = lean_ctor_get(v_toApplicative_104_, 1);
lean_inc(v_toPure_106_);
lean_dec_ref(v_toApplicative_104_);
v___f_107_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_InfoTree_foldInfoM___redArg___lam__1), 5, 2);
lean_closure_set(v___f_107_, 0, v_f_101_);
lean_closure_set(v___f_107_, 1, v_toBind_105_);
v___x_108_ = lean_apply_2(v_toPure_106_, lean_box(0), v_init_102_);
v___x_109_ = l_Lean_Elab_InfoTree_foldInfo___redArg(v___f_107_, v___x_108_, v_a_103_);
return v___x_109_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_InfoTree_foldInfoM(lean_object* v_00_u03b1_110_, lean_object* v_m_111_, lean_object* v_inst_112_, lean_object* v_f_113_, lean_object* v_init_114_, lean_object* v_a_115_){
_start:
{
lean_object* v___x_116_; 
v___x_116_ = lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_InfoTree_foldInfoM___redArg(v_inst_112_, v_f_113_, v_init_114_, v_a_115_);
return v___x_116_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_toFormat__propTypes_spec__0___redArg(lean_object* v_e_117_, lean_object* v___y_118_){
_start:
{
uint8_t v___x_120_; 
v___x_120_ = l_Lean_Expr_hasMVar(v_e_117_);
if (v___x_120_ == 0)
{
lean_object* v___x_121_; 
v___x_121_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_121_, 0, v_e_117_);
return v___x_121_;
}
else
{
lean_object* v___x_122_; lean_object* v_mctx_123_; lean_object* v___x_124_; lean_object* v_fst_125_; lean_object* v_snd_126_; lean_object* v___x_127_; lean_object* v_cache_128_; lean_object* v_zetaDeltaFVarIds_129_; lean_object* v_postponed_130_; lean_object* v_diag_131_; lean_object* v___x_133_; uint8_t v_isShared_134_; uint8_t v_isSharedCheck_140_; 
v___x_122_ = lean_st_ref_get(v___y_118_);
v_mctx_123_ = lean_ctor_get(v___x_122_, 0);
lean_inc_ref(v_mctx_123_);
lean_dec(v___x_122_);
v___x_124_ = l_Lean_instantiateMVarsCore(v_mctx_123_, v_e_117_);
v_fst_125_ = lean_ctor_get(v___x_124_, 0);
lean_inc(v_fst_125_);
v_snd_126_ = lean_ctor_get(v___x_124_, 1);
lean_inc(v_snd_126_);
lean_dec_ref(v___x_124_);
v___x_127_ = lean_st_ref_take(v___y_118_);
v_cache_128_ = lean_ctor_get(v___x_127_, 1);
v_zetaDeltaFVarIds_129_ = lean_ctor_get(v___x_127_, 2);
v_postponed_130_ = lean_ctor_get(v___x_127_, 3);
v_diag_131_ = lean_ctor_get(v___x_127_, 4);
v_isSharedCheck_140_ = !lean_is_exclusive(v___x_127_);
if (v_isSharedCheck_140_ == 0)
{
lean_object* v_unused_141_; 
v_unused_141_ = lean_ctor_get(v___x_127_, 0);
lean_dec(v_unused_141_);
v___x_133_ = v___x_127_;
v_isShared_134_ = v_isSharedCheck_140_;
goto v_resetjp_132_;
}
else
{
lean_inc(v_diag_131_);
lean_inc(v_postponed_130_);
lean_inc(v_zetaDeltaFVarIds_129_);
lean_inc(v_cache_128_);
lean_dec(v___x_127_);
v___x_133_ = lean_box(0);
v_isShared_134_ = v_isSharedCheck_140_;
goto v_resetjp_132_;
}
v_resetjp_132_:
{
lean_object* v___x_136_; 
if (v_isShared_134_ == 0)
{
lean_ctor_set(v___x_133_, 0, v_snd_126_);
v___x_136_ = v___x_133_;
goto v_reusejp_135_;
}
else
{
lean_object* v_reuseFailAlloc_139_; 
v_reuseFailAlloc_139_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_139_, 0, v_snd_126_);
lean_ctor_set(v_reuseFailAlloc_139_, 1, v_cache_128_);
lean_ctor_set(v_reuseFailAlloc_139_, 2, v_zetaDeltaFVarIds_129_);
lean_ctor_set(v_reuseFailAlloc_139_, 3, v_postponed_130_);
lean_ctor_set(v_reuseFailAlloc_139_, 4, v_diag_131_);
v___x_136_ = v_reuseFailAlloc_139_;
goto v_reusejp_135_;
}
v_reusejp_135_:
{
lean_object* v___x_137_; lean_object* v___x_138_; 
v___x_137_ = lean_st_ref_set(v___y_118_, v___x_136_);
v___x_138_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_138_, 0, v_fst_125_);
return v___x_138_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_toFormat__propTypes_spec__0___redArg___boxed(lean_object* v_e_142_, lean_object* v___y_143_, lean_object* v___y_144_){
_start:
{
lean_object* v_res_145_; 
v_res_145_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_toFormat__propTypes_spec__0___redArg(v_e_142_, v___y_143_);
lean_dec(v___y_143_);
return v_res_145_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_toFormat__propTypes_spec__0(lean_object* v_e_146_, lean_object* v___y_147_, lean_object* v___y_148_, lean_object* v___y_149_, lean_object* v___y_150_){
_start:
{
lean_object* v___x_152_; 
v___x_152_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_toFormat__propTypes_spec__0___redArg(v_e_146_, v___y_148_);
return v___x_152_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_toFormat__propTypes_spec__0___boxed(lean_object* v_e_153_, lean_object* v___y_154_, lean_object* v___y_155_, lean_object* v___y_156_, lean_object* v___y_157_, lean_object* v___y_158_){
_start:
{
lean_object* v_res_159_; 
v_res_159_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_toFormat__propTypes_spec__0(v_e_153_, v___y_154_, v___y_155_, v___y_156_, v___y_157_);
lean_dec(v___y_157_);
lean_dec_ref(v___y_156_);
lean_dec(v___y_155_);
lean_dec_ref(v___y_154_);
return v_res_159_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_toFormat__propTypes_spec__1_spec__1(lean_object* v_as_160_, size_t v_i_161_, size_t v_stop_162_, lean_object* v_b_163_, lean_object* v___y_164_, lean_object* v___y_165_, lean_object* v___y_166_, lean_object* v___y_167_){
_start:
{
lean_object* v_a_170_; uint8_t v___x_174_; 
v___x_174_ = lean_usize_dec_eq(v_i_161_, v_stop_162_);
if (v___x_174_ == 0)
{
lean_object* v___x_175_; lean_object* v_fst_176_; lean_object* v_snd_177_; lean_object* v___x_179_; uint8_t v_isShared_180_; uint8_t v_isSharedCheck_216_; 
v___x_175_ = lean_array_uget(v_as_160_, v_i_161_);
v_fst_176_ = lean_ctor_get(v___x_175_, 0);
v_snd_177_ = lean_ctor_get(v___x_175_, 1);
v_isSharedCheck_216_ = !lean_is_exclusive(v___x_175_);
if (v_isSharedCheck_216_ == 0)
{
v___x_179_ = v___x_175_;
v_isShared_180_ = v_isSharedCheck_216_;
goto v_resetjp_178_;
}
else
{
lean_inc(v_snd_177_);
lean_inc(v_fst_176_);
lean_dec(v___x_175_);
v___x_179_ = lean_box(0);
v_isShared_180_ = v_isSharedCheck_216_;
goto v_resetjp_178_;
}
v_resetjp_178_:
{
lean_object* v___x_181_; 
lean_inc(v_fst_176_);
v___x_181_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_toFormat__propTypes_spec__0___redArg(v_fst_176_, v___y_165_);
if (lean_obj_tag(v___x_181_) == 0)
{
lean_object* v_a_182_; lean_object* v___x_183_; 
v_a_182_ = lean_ctor_get(v___x_181_, 0);
lean_inc(v_a_182_);
lean_dec_ref_known(v___x_181_, 1);
lean_inc(v___y_167_);
lean_inc_ref(v___y_166_);
lean_inc(v___y_165_);
lean_inc_ref(v___y_164_);
v___x_183_ = lean_infer_type(v_a_182_, v___y_164_, v___y_165_, v___y_166_, v___y_167_);
if (lean_obj_tag(v___x_183_) == 0)
{
lean_object* v_a_184_; uint8_t v___x_185_; 
v_a_184_ = lean_ctor_get(v___x_183_, 0);
lean_inc(v_a_184_);
lean_dec_ref_known(v___x_183_, 1);
v___x_185_ = l_Lean_Expr_isProp(v_a_184_);
lean_dec(v_a_184_);
if (v___x_185_ == 0)
{
lean_object* v___x_186_; 
v___x_186_ = l_Lean_Meta_ppExpr(v_fst_176_, v___y_164_, v___y_165_, v___y_166_, v___y_167_);
if (lean_obj_tag(v___x_186_) == 0)
{
lean_object* v_a_187_; lean_object* v___x_189_; 
v_a_187_ = lean_ctor_get(v___x_186_, 0);
lean_inc(v_a_187_);
lean_dec_ref_known(v___x_186_, 1);
if (v_isShared_180_ == 0)
{
lean_ctor_set(v___x_179_, 0, v_a_187_);
v___x_189_ = v___x_179_;
goto v_reusejp_188_;
}
else
{
lean_object* v_reuseFailAlloc_191_; 
v_reuseFailAlloc_191_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_191_, 0, v_a_187_);
lean_ctor_set(v_reuseFailAlloc_191_, 1, v_snd_177_);
v___x_189_ = v_reuseFailAlloc_191_;
goto v_reusejp_188_;
}
v_reusejp_188_:
{
lean_object* v___x_190_; 
v___x_190_ = lean_array_push(v_b_163_, v___x_189_);
v_a_170_ = v___x_190_;
goto v___jp_169_;
}
}
else
{
lean_object* v_a_192_; lean_object* v___x_194_; uint8_t v_isShared_195_; uint8_t v_isSharedCheck_199_; 
lean_del_object(v___x_179_);
lean_dec(v_snd_177_);
lean_dec_ref(v_b_163_);
v_a_192_ = lean_ctor_get(v___x_186_, 0);
v_isSharedCheck_199_ = !lean_is_exclusive(v___x_186_);
if (v_isSharedCheck_199_ == 0)
{
v___x_194_ = v___x_186_;
v_isShared_195_ = v_isSharedCheck_199_;
goto v_resetjp_193_;
}
else
{
lean_inc(v_a_192_);
lean_dec(v___x_186_);
v___x_194_ = lean_box(0);
v_isShared_195_ = v_isSharedCheck_199_;
goto v_resetjp_193_;
}
v_resetjp_193_:
{
lean_object* v___x_197_; 
if (v_isShared_195_ == 0)
{
v___x_197_ = v___x_194_;
goto v_reusejp_196_;
}
else
{
lean_object* v_reuseFailAlloc_198_; 
v_reuseFailAlloc_198_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_198_, 0, v_a_192_);
v___x_197_ = v_reuseFailAlloc_198_;
goto v_reusejp_196_;
}
v_reusejp_196_:
{
return v___x_197_;
}
}
}
}
else
{
lean_del_object(v___x_179_);
lean_dec(v_snd_177_);
lean_dec(v_fst_176_);
v_a_170_ = v_b_163_;
goto v___jp_169_;
}
}
else
{
lean_object* v_a_200_; lean_object* v___x_202_; uint8_t v_isShared_203_; uint8_t v_isSharedCheck_207_; 
lean_del_object(v___x_179_);
lean_dec(v_snd_177_);
lean_dec(v_fst_176_);
lean_dec_ref(v_b_163_);
v_a_200_ = lean_ctor_get(v___x_183_, 0);
v_isSharedCheck_207_ = !lean_is_exclusive(v___x_183_);
if (v_isSharedCheck_207_ == 0)
{
v___x_202_ = v___x_183_;
v_isShared_203_ = v_isSharedCheck_207_;
goto v_resetjp_201_;
}
else
{
lean_inc(v_a_200_);
lean_dec(v___x_183_);
v___x_202_ = lean_box(0);
v_isShared_203_ = v_isSharedCheck_207_;
goto v_resetjp_201_;
}
v_resetjp_201_:
{
lean_object* v___x_205_; 
if (v_isShared_203_ == 0)
{
v___x_205_ = v___x_202_;
goto v_reusejp_204_;
}
else
{
lean_object* v_reuseFailAlloc_206_; 
v_reuseFailAlloc_206_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_206_, 0, v_a_200_);
v___x_205_ = v_reuseFailAlloc_206_;
goto v_reusejp_204_;
}
v_reusejp_204_:
{
return v___x_205_;
}
}
}
}
else
{
lean_object* v_a_208_; lean_object* v___x_210_; uint8_t v_isShared_211_; uint8_t v_isSharedCheck_215_; 
lean_del_object(v___x_179_);
lean_dec(v_snd_177_);
lean_dec(v_fst_176_);
lean_dec_ref(v_b_163_);
v_a_208_ = lean_ctor_get(v___x_181_, 0);
v_isSharedCheck_215_ = !lean_is_exclusive(v___x_181_);
if (v_isSharedCheck_215_ == 0)
{
v___x_210_ = v___x_181_;
v_isShared_211_ = v_isSharedCheck_215_;
goto v_resetjp_209_;
}
else
{
lean_inc(v_a_208_);
lean_dec(v___x_181_);
v___x_210_ = lean_box(0);
v_isShared_211_ = v_isSharedCheck_215_;
goto v_resetjp_209_;
}
v_resetjp_209_:
{
lean_object* v___x_213_; 
if (v_isShared_211_ == 0)
{
v___x_213_ = v___x_210_;
goto v_reusejp_212_;
}
else
{
lean_object* v_reuseFailAlloc_214_; 
v_reuseFailAlloc_214_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_214_, 0, v_a_208_);
v___x_213_ = v_reuseFailAlloc_214_;
goto v_reusejp_212_;
}
v_reusejp_212_:
{
return v___x_213_;
}
}
}
}
}
else
{
lean_object* v___x_217_; 
v___x_217_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_217_, 0, v_b_163_);
return v___x_217_;
}
v___jp_169_:
{
size_t v___x_171_; size_t v___x_172_; 
v___x_171_ = ((size_t)1ULL);
v___x_172_ = lean_usize_add(v_i_161_, v___x_171_);
v_i_161_ = v___x_172_;
v_b_163_ = v_a_170_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_toFormat__propTypes_spec__1_spec__1___boxed(lean_object* v_as_218_, lean_object* v_i_219_, lean_object* v_stop_220_, lean_object* v_b_221_, lean_object* v___y_222_, lean_object* v___y_223_, lean_object* v___y_224_, lean_object* v___y_225_, lean_object* v___y_226_){
_start:
{
size_t v_i_boxed_227_; size_t v_stop_boxed_228_; lean_object* v_res_229_; 
v_i_boxed_227_ = lean_unbox_usize(v_i_219_);
lean_dec(v_i_219_);
v_stop_boxed_228_ = lean_unbox_usize(v_stop_220_);
lean_dec(v_stop_220_);
v_res_229_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_toFormat__propTypes_spec__1_spec__1(v_as_218_, v_i_boxed_227_, v_stop_boxed_228_, v_b_221_, v___y_222_, v___y_223_, v___y_224_, v___y_225_);
lean_dec(v___y_225_);
lean_dec_ref(v___y_224_);
lean_dec(v___y_223_);
lean_dec_ref(v___y_222_);
lean_dec_ref(v_as_218_);
return v_res_229_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_toFormat__propTypes_spec__1(lean_object* v_as_232_, lean_object* v_start_233_, lean_object* v_stop_234_, lean_object* v___y_235_, lean_object* v___y_236_, lean_object* v___y_237_, lean_object* v___y_238_){
_start:
{
lean_object* v___x_240_; uint8_t v___x_241_; 
v___x_240_ = ((lean_object*)(lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_toFormat__propTypes_spec__1___closed__0));
v___x_241_ = lean_nat_dec_lt(v_start_233_, v_stop_234_);
if (v___x_241_ == 0)
{
lean_object* v___x_242_; 
v___x_242_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_242_, 0, v___x_240_);
return v___x_242_;
}
else
{
lean_object* v___x_243_; uint8_t v___x_244_; 
v___x_243_ = lean_array_get_size(v_as_232_);
v___x_244_ = lean_nat_dec_le(v_stop_234_, v___x_243_);
if (v___x_244_ == 0)
{
uint8_t v___x_245_; 
v___x_245_ = lean_nat_dec_lt(v_start_233_, v___x_243_);
if (v___x_245_ == 0)
{
lean_object* v___x_246_; 
v___x_246_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_246_, 0, v___x_240_);
return v___x_246_;
}
else
{
size_t v___x_247_; size_t v___x_248_; lean_object* v___x_249_; 
v___x_247_ = lean_usize_of_nat(v_start_233_);
v___x_248_ = lean_usize_of_nat(v___x_243_);
v___x_249_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_toFormat__propTypes_spec__1_spec__1(v_as_232_, v___x_247_, v___x_248_, v___x_240_, v___y_235_, v___y_236_, v___y_237_, v___y_238_);
return v___x_249_;
}
}
else
{
size_t v___x_250_; size_t v___x_251_; lean_object* v___x_252_; 
v___x_250_ = lean_usize_of_nat(v_start_233_);
v___x_251_ = lean_usize_of_nat(v_stop_234_);
v___x_252_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_toFormat__propTypes_spec__1_spec__1(v_as_232_, v___x_250_, v___x_251_, v___x_240_, v___y_235_, v___y_236_, v___y_237_, v___y_238_);
return v___x_252_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_toFormat__propTypes_spec__1___boxed(lean_object* v_as_253_, lean_object* v_start_254_, lean_object* v_stop_255_, lean_object* v___y_256_, lean_object* v___y_257_, lean_object* v___y_258_, lean_object* v___y_259_, lean_object* v___y_260_){
_start:
{
lean_object* v_res_261_; 
v_res_261_ = lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_toFormat__propTypes_spec__1(v_as_253_, v_start_254_, v_stop_255_, v___y_256_, v___y_257_, v___y_258_, v___y_259_);
lean_dec(v___y_259_);
lean_dec_ref(v___y_258_);
lean_dec(v___y_257_);
lean_dec_ref(v___y_256_);
lean_dec(v_stop_255_);
lean_dec(v_start_254_);
lean_dec_ref(v_as_253_);
return v_res_261_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_toFormat__propTypes___redArg(lean_object* v_ctx_262_, lean_object* v_lc_263_, lean_object* v_es_264_, lean_object* v_a_265_){
_start:
{
lean_object* v___x_267_; lean_object* v___x_268_; lean_object* v___x_269_; lean_object* v___x_270_; 
v___x_267_ = lean_unsigned_to_nat(0u);
v___x_268_ = lean_array_get_size(v_es_264_);
v___x_269_ = lean_alloc_closure((void*)(lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_toFormat__propTypes_spec__1___boxed), 8, 3);
lean_closure_set(v___x_269_, 0, v_es_264_);
lean_closure_set(v___x_269_, 1, v___x_267_);
lean_closure_set(v___x_269_, 2, v___x_268_);
v___x_270_ = l_Lean_Elab_ContextInfo_runMetaM___redArg(v_ctx_262_, v_lc_263_, v___x_269_);
if (lean_obj_tag(v___x_270_) == 0)
{
lean_object* v_a_271_; lean_object* v___x_273_; uint8_t v_isShared_274_; uint8_t v_isSharedCheck_278_; 
v_a_271_ = lean_ctor_get(v___x_270_, 0);
v_isSharedCheck_278_ = !lean_is_exclusive(v___x_270_);
if (v_isSharedCheck_278_ == 0)
{
v___x_273_ = v___x_270_;
v_isShared_274_ = v_isSharedCheck_278_;
goto v_resetjp_272_;
}
else
{
lean_inc(v_a_271_);
lean_dec(v___x_270_);
v___x_273_ = lean_box(0);
v_isShared_274_ = v_isSharedCheck_278_;
goto v_resetjp_272_;
}
v_resetjp_272_:
{
lean_object* v___x_276_; 
if (v_isShared_274_ == 0)
{
v___x_276_ = v___x_273_;
goto v_reusejp_275_;
}
else
{
lean_object* v_reuseFailAlloc_277_; 
v_reuseFailAlloc_277_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_277_, 0, v_a_271_);
v___x_276_ = v_reuseFailAlloc_277_;
goto v_reusejp_275_;
}
v_reusejp_275_:
{
return v___x_276_;
}
}
}
else
{
lean_object* v_a_279_; lean_object* v___x_281_; uint8_t v_isShared_282_; uint8_t v_isSharedCheck_291_; 
v_a_279_ = lean_ctor_get(v___x_270_, 0);
v_isSharedCheck_291_ = !lean_is_exclusive(v___x_270_);
if (v_isSharedCheck_291_ == 0)
{
v___x_281_ = v___x_270_;
v_isShared_282_ = v_isSharedCheck_291_;
goto v_resetjp_280_;
}
else
{
lean_inc(v_a_279_);
lean_dec(v___x_270_);
v___x_281_ = lean_box(0);
v_isShared_282_ = v_isSharedCheck_291_;
goto v_resetjp_280_;
}
v_resetjp_280_:
{
lean_object* v_ref_283_; lean_object* v___x_284_; lean_object* v___x_285_; lean_object* v___x_286_; lean_object* v___x_287_; lean_object* v___x_289_; 
v_ref_283_ = lean_ctor_get(v_a_265_, 7);
v___x_284_ = lean_io_error_to_string(v_a_279_);
v___x_285_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_285_, 0, v___x_284_);
v___x_286_ = l_Lean_MessageData_ofFormat(v___x_285_);
lean_inc(v_ref_283_);
v___x_287_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_287_, 0, v_ref_283_);
lean_ctor_set(v___x_287_, 1, v___x_286_);
if (v_isShared_282_ == 0)
{
lean_ctor_set(v___x_281_, 0, v___x_287_);
v___x_289_ = v___x_281_;
goto v_reusejp_288_;
}
else
{
lean_object* v_reuseFailAlloc_290_; 
v_reuseFailAlloc_290_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_290_, 0, v___x_287_);
v___x_289_ = v_reuseFailAlloc_290_;
goto v_reusejp_288_;
}
v_reusejp_288_:
{
return v___x_289_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_toFormat__propTypes___redArg___boxed(lean_object* v_ctx_292_, lean_object* v_lc_293_, lean_object* v_es_294_, lean_object* v_a_295_, lean_object* v_a_296_){
_start:
{
lean_object* v_res_297_; 
v_res_297_ = lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_toFormat__propTypes___redArg(v_ctx_292_, v_lc_293_, v_es_294_, v_a_295_);
lean_dec_ref(v_a_295_);
return v_res_297_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_toFormat__propTypes(lean_object* v_ctx_298_, lean_object* v_lc_299_, lean_object* v_es_300_, lean_object* v_a_301_, lean_object* v_a_302_){
_start:
{
lean_object* v___x_304_; 
v___x_304_ = lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_toFormat__propTypes___redArg(v_ctx_298_, v_lc_299_, v_es_300_, v_a_301_);
return v___x_304_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_toFormat__propTypes___boxed(lean_object* v_ctx_305_, lean_object* v_lc_306_, lean_object* v_es_307_, lean_object* v_a_308_, lean_object* v_a_309_, lean_object* v_a_310_){
_start:
{
lean_object* v_res_311_; 
v_res_311_ = lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_toFormat__propTypes(v_ctx_305_, v_lc_306_, v_es_307_, v_a_308_, v_a_309_);
lean_dec(v_a_309_);
lean_dec_ref(v_a_308_);
return v_res_311_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_InfoTree_foldInfoM___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__11___redArg___lam__0(lean_object* v_f_312_, lean_object* v_ctx_313_, lean_object* v_i_314_, lean_object* v_ma_315_, lean_object* v___y_316_, lean_object* v___y_317_){
_start:
{
lean_object* v___x_319_; 
lean_inc(v___y_317_);
lean_inc_ref(v___y_316_);
v___x_319_ = lean_apply_3(v_ma_315_, v___y_316_, v___y_317_, lean_box(0));
if (lean_obj_tag(v___x_319_) == 0)
{
lean_object* v_a_320_; lean_object* v___x_321_; 
v_a_320_ = lean_ctor_get(v___x_319_, 0);
lean_inc(v_a_320_);
lean_dec_ref_known(v___x_319_, 1);
lean_inc(v___y_317_);
lean_inc_ref(v___y_316_);
v___x_321_ = lean_apply_6(v_f_312_, v_ctx_313_, v_i_314_, v_a_320_, v___y_316_, v___y_317_, lean_box(0));
return v___x_321_;
}
else
{
lean_dec_ref(v_i_314_);
lean_dec_ref(v_ctx_313_);
lean_dec_ref(v_f_312_);
return v___x_319_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_InfoTree_foldInfoM___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__11___redArg___lam__0___boxed(lean_object* v_f_322_, lean_object* v_ctx_323_, lean_object* v_i_324_, lean_object* v_ma_325_, lean_object* v___y_326_, lean_object* v___y_327_, lean_object* v___y_328_){
_start:
{
lean_object* v_res_329_; 
v_res_329_ = lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_InfoTree_foldInfoM___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__11___redArg___lam__0(v_f_322_, v_ctx_323_, v_i_324_, v_ma_325_, v___y_326_, v___y_327_);
lean_dec(v___y_327_);
lean_dec_ref(v___y_326_);
return v_res_329_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_InfoTree_foldInfoM___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__11___redArg___lam__1(lean_object* v_init_330_, lean_object* v___y_331_, lean_object* v___y_332_){
_start:
{
lean_object* v___x_334_; 
v___x_334_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_334_, 0, v_init_330_);
return v___x_334_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_InfoTree_foldInfoM___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__11___redArg___lam__1___boxed(lean_object* v_init_335_, lean_object* v___y_336_, lean_object* v___y_337_, lean_object* v___y_338_){
_start:
{
lean_object* v_res_339_; 
v_res_339_ = lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_InfoTree_foldInfoM___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__11___redArg___lam__1(v_init_335_, v___y_336_, v___y_337_);
lean_dec(v___y_337_);
lean_dec_ref(v___y_336_);
return v_res_339_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_InfoTree_foldInfoM___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__11___redArg(lean_object* v_f_340_, lean_object* v_init_341_, lean_object* v_a_342_, lean_object* v___y_343_, lean_object* v___y_344_){
_start:
{
lean_object* v___f_346_; lean_object* v___f_347_; lean_object* v___x_1766__overap_348_; lean_object* v___x_349_; 
v___f_346_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_InfoTree_foldInfoM___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__11___redArg___lam__0___boxed), 7, 1);
lean_closure_set(v___f_346_, 0, v_f_340_);
v___f_347_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_InfoTree_foldInfoM___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__11___redArg___lam__1___boxed), 4, 1);
lean_closure_set(v___f_347_, 0, v_init_341_);
v___x_1766__overap_348_ = l_Lean_Elab_InfoTree_foldInfo___redArg(v___f_346_, v___f_347_, v_a_342_);
lean_inc(v___y_344_);
lean_inc_ref(v___y_343_);
v___x_349_ = lean_apply_3(v___x_1766__overap_348_, v___y_343_, v___y_344_, lean_box(0));
return v___x_349_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_InfoTree_foldInfoM___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__11___redArg___boxed(lean_object* v_f_350_, lean_object* v_init_351_, lean_object* v_a_352_, lean_object* v___y_353_, lean_object* v___y_354_, lean_object* v___y_355_){
_start:
{
lean_object* v_res_356_; 
v_res_356_ = lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_InfoTree_foldInfoM___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__11___redArg(v_f_350_, v_init_351_, v_a_352_, v___y_353_, v___y_354_);
lean_dec(v___y_354_);
lean_dec_ref(v___y_353_);
return v_res_356_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_InfoTree_foldInfoM___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__11(lean_object* v_00_u03b1_357_, lean_object* v_f_358_, lean_object* v_init_359_, lean_object* v_a_360_, lean_object* v___y_361_, lean_object* v___y_362_){
_start:
{
lean_object* v___x_364_; 
v___x_364_ = lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_InfoTree_foldInfoM___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__11___redArg(v_f_358_, v_init_359_, v_a_360_, v___y_361_, v___y_362_);
return v___x_364_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_InfoTree_foldInfoM___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__11___boxed(lean_object* v_00_u03b1_365_, lean_object* v_f_366_, lean_object* v_init_367_, lean_object* v_a_368_, lean_object* v___y_369_, lean_object* v___y_370_, lean_object* v___y_371_){
_start:
{
lean_object* v_res_372_; 
v_res_372_ = lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_InfoTree_foldInfoM___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__11(v_00_u03b1_365_, v_f_366_, v_init_367_, v_a_368_, v___y_369_, v___y_370_);
lean_dec(v___y_370_);
lean_dec_ref(v___y_369_);
return v_res_372_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__8(lean_object* v_a_373_, lean_object* v_a_374_){
_start:
{
if (lean_obj_tag(v_a_373_) == 0)
{
lean_object* v___x_375_; 
v___x_375_ = l_List_reverse___redArg(v_a_374_);
return v___x_375_;
}
else
{
lean_object* v_head_376_; lean_object* v_tail_377_; lean_object* v___x_379_; uint8_t v_isShared_380_; uint8_t v_isSharedCheck_388_; 
v_head_376_ = lean_ctor_get(v_a_373_, 0);
v_tail_377_ = lean_ctor_get(v_a_373_, 1);
v_isSharedCheck_388_ = !lean_is_exclusive(v_a_373_);
if (v_isSharedCheck_388_ == 0)
{
v___x_379_ = v_a_373_;
v_isShared_380_ = v_isSharedCheck_388_;
goto v_resetjp_378_;
}
else
{
lean_inc(v_tail_377_);
lean_inc(v_head_376_);
lean_dec(v_a_373_);
v___x_379_ = lean_box(0);
v_isShared_380_ = v_isSharedCheck_388_;
goto v_resetjp_378_;
}
v_resetjp_378_:
{
lean_object* v___x_381_; lean_object* v___x_382_; lean_object* v___x_383_; lean_object* v___x_385_; 
v___x_381_ = l_Lean_LocalDecl_type(v_head_376_);
v___x_382_ = l_Lean_LocalDecl_userName(v_head_376_);
lean_dec(v_head_376_);
v___x_383_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_383_, 0, v___x_381_);
lean_ctor_set(v___x_383_, 1, v___x_382_);
if (v_isShared_380_ == 0)
{
lean_ctor_set(v___x_379_, 1, v_a_374_);
lean_ctor_set(v___x_379_, 0, v___x_383_);
v___x_385_ = v___x_379_;
goto v_reusejp_384_;
}
else
{
lean_object* v_reuseFailAlloc_387_; 
v_reuseFailAlloc_387_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_387_, 0, v___x_383_);
lean_ctor_set(v_reuseFailAlloc_387_, 1, v_a_374_);
v___x_385_ = v_reuseFailAlloc_387_;
goto v_reusejp_384_;
}
v_reusejp_384_:
{
v_a_373_ = v_tail_377_;
v_a_374_ = v___x_385_;
goto _start;
}
}
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_elem___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__6(lean_object* v_a_389_, lean_object* v_x_390_){
_start:
{
if (lean_obj_tag(v_x_390_) == 0)
{
uint8_t v___x_391_; 
v___x_391_ = 0;
return v___x_391_;
}
else
{
lean_object* v_head_392_; lean_object* v_tail_393_; uint8_t v___x_394_; 
v_head_392_ = lean_ctor_get(v_x_390_, 0);
v_tail_393_ = lean_ctor_get(v_x_390_, 1);
v___x_394_ = l_Lean_instBEqFVarId_beq(v_a_389_, v_head_392_);
if (v___x_394_ == 0)
{
v_x_390_ = v_tail_393_;
goto _start;
}
else
{
return v___x_394_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_elem___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__6___boxed(lean_object* v_a_396_, lean_object* v_x_397_){
_start:
{
uint8_t v_res_398_; lean_object* v_r_399_; 
v_res_398_ = lp_mathlib_List_elem___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__6(v_a_396_, v_x_397_);
lean_dec(v_x_397_);
lean_dec(v_a_396_);
v_r_399_ = lean_box(v_res_398_);
return v_r_399_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_filterTR_loop___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__7(lean_object* v_oldFVars_400_, uint8_t v___x_401_, lean_object* v_a_402_, lean_object* v_a_403_){
_start:
{
if (lean_obj_tag(v_a_402_) == 0)
{
lean_object* v___x_404_; 
v___x_404_ = l_List_reverse___redArg(v_a_403_);
return v___x_404_;
}
else
{
lean_object* v_head_405_; lean_object* v_tail_406_; lean_object* v___x_408_; uint8_t v_isShared_409_; uint8_t v_isSharedCheck_418_; 
v_head_405_ = lean_ctor_get(v_a_402_, 0);
v_tail_406_ = lean_ctor_get(v_a_402_, 1);
v_isSharedCheck_418_ = !lean_is_exclusive(v_a_402_);
if (v_isSharedCheck_418_ == 0)
{
v___x_408_ = v_a_402_;
v_isShared_409_ = v_isSharedCheck_418_;
goto v_resetjp_407_;
}
else
{
lean_inc(v_tail_406_);
lean_inc(v_head_405_);
lean_dec(v_a_402_);
v___x_408_ = lean_box(0);
v_isShared_409_ = v_isSharedCheck_418_;
goto v_resetjp_407_;
}
v_resetjp_407_:
{
lean_object* v___x_410_; uint8_t v___x_411_; 
v___x_410_ = l_Lean_LocalDecl_fvarId(v_head_405_);
v___x_411_ = lp_mathlib_List_elem___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__6(v___x_410_, v_oldFVars_400_);
lean_dec(v___x_410_);
if (v___x_411_ == 0)
{
if (v___x_401_ == 0)
{
lean_del_object(v___x_408_);
lean_dec(v_head_405_);
v_a_402_ = v_tail_406_;
goto _start;
}
else
{
lean_object* v___x_414_; 
if (v_isShared_409_ == 0)
{
lean_ctor_set(v___x_408_, 1, v_a_403_);
v___x_414_ = v___x_408_;
goto v_reusejp_413_;
}
else
{
lean_object* v_reuseFailAlloc_416_; 
v_reuseFailAlloc_416_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_416_, 0, v_head_405_);
lean_ctor_set(v_reuseFailAlloc_416_, 1, v_a_403_);
v___x_414_ = v_reuseFailAlloc_416_;
goto v_reusejp_413_;
}
v_reusejp_413_:
{
v_a_402_ = v_tail_406_;
v_a_403_ = v___x_414_;
goto _start;
}
}
}
else
{
lean_del_object(v___x_408_);
lean_dec(v_head_405_);
v_a_402_ = v_tail_406_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_filterTR_loop___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__7___boxed(lean_object* v_oldFVars_419_, lean_object* v___x_420_, lean_object* v_a_421_, lean_object* v_a_422_){
_start:
{
uint8_t v___x_2284__boxed_423_; lean_object* v_res_424_; 
v___x_2284__boxed_423_ = lean_unbox(v___x_420_);
v_res_424_ = lp_mathlib_List_filterTR_loop___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__7(v_oldFVars_419_, v___x_2284__boxed_423_, v_a_421_, v_a_422_);
lean_dec(v_oldFVars_419_);
return v_res_424_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__4(lean_object* v_a_425_, lean_object* v_a_426_){
_start:
{
if (lean_obj_tag(v_a_425_) == 0)
{
lean_object* v___x_427_; 
v___x_427_ = lean_array_to_list(v_a_426_);
return v___x_427_;
}
else
{
lean_object* v_head_428_; lean_object* v_tail_429_; lean_object* v___x_430_; 
v_head_428_ = lean_ctor_get(v_a_425_, 0);
lean_inc(v_head_428_);
v_tail_429_ = lean_ctor_get(v_a_425_, 1);
lean_inc(v_tail_429_);
lean_dec_ref_known(v_a_425_, 2);
v___x_430_ = l_List_foldl___at___00Array_appendList_spec__0___redArg(v_a_426_, v_head_428_);
v_a_425_ = v_tail_429_;
v_a_426_ = v___x_430_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_filterMapTR_go___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__0(lean_object* v_a_432_, lean_object* v_a_433_){
_start:
{
if (lean_obj_tag(v_a_432_) == 0)
{
lean_object* v___x_434_; 
v___x_434_ = lean_array_to_list(v_a_433_);
return v___x_434_;
}
else
{
lean_object* v_head_435_; 
v_head_435_ = lean_ctor_get(v_a_432_, 0);
if (lean_obj_tag(v_head_435_) == 0)
{
lean_object* v_tail_436_; 
v_tail_436_ = lean_ctor_get(v_a_432_, 1);
lean_inc(v_tail_436_);
lean_dec_ref_known(v_a_432_, 2);
v_a_432_ = v_tail_436_;
goto _start;
}
else
{
lean_object* v_tail_438_; lean_object* v_val_439_; lean_object* v___x_440_; 
lean_inc_ref(v_head_435_);
v_tail_438_ = lean_ctor_get(v_a_432_, 1);
lean_inc(v_tail_438_);
lean_dec_ref_known(v_a_432_, 2);
v_val_439_ = lean_ctor_get(v_head_435_, 0);
lean_inc(v_val_439_);
lean_dec_ref_known(v_head_435_, 1);
v___x_440_ = lean_array_push(v_a_433_, v_val_439_);
v_a_432_ = v_tail_438_;
v_a_433_ = v___x_440_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__9(uint8_t v___x_445_, lean_object* v_stx_446_, size_t v_sz_447_, size_t v_i_448_, lean_object* v_bs_449_){
_start:
{
uint8_t v___x_450_; 
v___x_450_ = lean_usize_dec_lt(v_i_448_, v_sz_447_);
if (v___x_450_ == 0)
{
lean_dec(v_stx_446_);
return v_bs_449_;
}
else
{
lean_object* v_v_451_; lean_object* v_fst_452_; lean_object* v_snd_453_; lean_object* v___x_455_; uint8_t v_isShared_456_; uint8_t v_isSharedCheck_471_; 
v_v_451_ = lean_array_uget(v_bs_449_, v_i_448_);
v_fst_452_ = lean_ctor_get(v_v_451_, 0);
v_snd_453_ = lean_ctor_get(v_v_451_, 1);
v_isSharedCheck_471_ = !lean_is_exclusive(v_v_451_);
if (v_isSharedCheck_471_ == 0)
{
v___x_455_ = v_v_451_;
v_isShared_456_ = v_isSharedCheck_471_;
goto v_resetjp_454_;
}
else
{
lean_inc(v_snd_453_);
lean_inc(v_fst_452_);
lean_dec(v_v_451_);
v___x_455_ = lean_box(0);
v_isShared_456_ = v_isSharedCheck_471_;
goto v_resetjp_454_;
}
v_resetjp_454_:
{
lean_object* v___x_457_; lean_object* v_bs_x27_458_; lean_object* v___x_459_; lean_object* v___x_460_; lean_object* v___x_461_; lean_object* v___x_462_; lean_object* v___x_463_; lean_object* v___x_465_; 
v___x_457_ = lean_unsigned_to_nat(0u);
v_bs_x27_458_ = lean_array_uset(v_bs_449_, v_i_448_, v___x_457_);
v___x_459_ = l_Lean_Name_toString(v_snd_453_, v___x_445_);
v___x_460_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_460_, 0, v___x_459_);
v___x_461_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__9___closed__1));
v___x_462_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_462_, 0, v___x_460_);
lean_ctor_set(v___x_462_, 1, v___x_461_);
v___x_463_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_463_, 0, v___x_462_);
lean_ctor_set(v___x_463_, 1, v_fst_452_);
lean_inc(v_stx_446_);
if (v_isShared_456_ == 0)
{
lean_ctor_set(v___x_455_, 1, v___x_463_);
lean_ctor_set(v___x_455_, 0, v_stx_446_);
v___x_465_ = v___x_455_;
goto v_reusejp_464_;
}
else
{
lean_object* v_reuseFailAlloc_470_; 
v_reuseFailAlloc_470_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_470_, 0, v_stx_446_);
lean_ctor_set(v_reuseFailAlloc_470_, 1, v___x_463_);
v___x_465_ = v_reuseFailAlloc_470_;
goto v_reusejp_464_;
}
v_reusejp_464_:
{
size_t v___x_466_; size_t v___x_467_; lean_object* v___x_468_; 
v___x_466_ = ((size_t)1ULL);
v___x_467_ = lean_usize_add(v_i_448_, v___x_466_);
v___x_468_ = lean_array_uset(v_bs_x27_458_, v_i_448_, v___x_465_);
v_i_448_ = v___x_467_;
v_bs_449_ = v___x_468_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__9___boxed(lean_object* v___x_472_, lean_object* v_stx_473_, lean_object* v_sz_474_, lean_object* v_i_475_, lean_object* v_bs_476_){
_start:
{
uint8_t v___x_2363__boxed_477_; size_t v_sz_boxed_478_; size_t v_i_boxed_479_; lean_object* v_res_480_; 
v___x_2363__boxed_477_ = lean_unbox(v___x_472_);
v_sz_boxed_478_ = lean_unbox_usize(v_sz_474_);
lean_dec(v_sz_474_);
v_i_boxed_479_ = lean_unbox_usize(v_i_475_);
lean_dec(v_i_475_);
v_res_480_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__9(v___x_2363__boxed_477_, v_stx_473_, v_sz_boxed_478_, v_i_boxed_479_, v_bs_476_);
return v_res_480_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__5(lean_object* v_a_481_, lean_object* v_a_482_){
_start:
{
if (lean_obj_tag(v_a_481_) == 0)
{
lean_object* v___x_483_; 
v___x_483_ = l_List_reverse___redArg(v_a_482_);
return v___x_483_;
}
else
{
lean_object* v_head_484_; lean_object* v_tail_485_; lean_object* v___x_487_; uint8_t v_isShared_488_; uint8_t v_isSharedCheck_494_; 
v_head_484_ = lean_ctor_get(v_a_481_, 0);
v_tail_485_ = lean_ctor_get(v_a_481_, 1);
v_isSharedCheck_494_ = !lean_is_exclusive(v_a_481_);
if (v_isSharedCheck_494_ == 0)
{
v___x_487_ = v_a_481_;
v_isShared_488_ = v_isSharedCheck_494_;
goto v_resetjp_486_;
}
else
{
lean_inc(v_tail_485_);
lean_inc(v_head_484_);
lean_dec(v_a_481_);
v___x_487_ = lean_box(0);
v_isShared_488_ = v_isSharedCheck_494_;
goto v_resetjp_486_;
}
v_resetjp_486_:
{
lean_object* v___x_489_; lean_object* v___x_491_; 
v___x_489_ = l_Lean_LocalDecl_fvarId(v_head_484_);
lean_dec(v_head_484_);
if (v_isShared_488_ == 0)
{
lean_ctor_set(v___x_487_, 1, v_a_482_);
lean_ctor_set(v___x_487_, 0, v___x_489_);
v___x_491_ = v___x_487_;
goto v_reusejp_490_;
}
else
{
lean_object* v_reuseFailAlloc_493_; 
v_reuseFailAlloc_493_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_493_, 0, v___x_489_);
lean_ctor_set(v_reuseFailAlloc_493_, 1, v_a_482_);
v___x_491_ = v_reuseFailAlloc_493_;
goto v_reusejp_490_;
}
v_reusejp_490_:
{
v_a_481_ = v_tail_485_;
v_a_482_ = v___x_491_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__3(lean_object* v_a_497_, lean_object* v_a_498_){
_start:
{
if (lean_obj_tag(v_a_497_) == 0)
{
lean_object* v___x_499_; 
v___x_499_ = l_List_reverse___redArg(v_a_498_);
return v___x_499_;
}
else
{
lean_object* v_head_500_; lean_object* v_lctx_501_; lean_object* v_tail_502_; lean_object* v___x_504_; uint8_t v_isShared_505_; uint8_t v_isSharedCheck_514_; 
v_head_500_ = lean_ctor_get(v_a_497_, 0);
v_lctx_501_ = lean_ctor_get(v_head_500_, 1);
lean_inc_ref(v_lctx_501_);
v_tail_502_ = lean_ctor_get(v_a_497_, 1);
v_isSharedCheck_514_ = !lean_is_exclusive(v_a_497_);
if (v_isSharedCheck_514_ == 0)
{
lean_object* v_unused_515_; 
v_unused_515_ = lean_ctor_get(v_a_497_, 0);
lean_dec(v_unused_515_);
v___x_504_ = v_a_497_;
v_isShared_505_ = v_isSharedCheck_514_;
goto v_resetjp_503_;
}
else
{
lean_inc(v_tail_502_);
lean_dec(v_a_497_);
v___x_504_ = lean_box(0);
v_isShared_505_ = v_isSharedCheck_514_;
goto v_resetjp_503_;
}
v_resetjp_503_:
{
lean_object* v_decls_506_; lean_object* v___x_507_; lean_object* v___x_508_; lean_object* v___x_509_; lean_object* v___x_511_; 
v_decls_506_ = lean_ctor_get(v_lctx_501_, 1);
lean_inc_ref(v_decls_506_);
lean_dec_ref(v_lctx_501_);
v___x_507_ = l_Lean_PersistentArray_toList___redArg(v_decls_506_);
lean_dec_ref(v_decls_506_);
v___x_508_ = ((lean_object*)(lp_mathlib_List_mapTR_loop___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__3___closed__0));
v___x_509_ = lp_mathlib_List_filterMapTR_go___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__0(v___x_507_, v___x_508_);
if (v_isShared_505_ == 0)
{
lean_ctor_set(v___x_504_, 1, v_a_498_);
lean_ctor_set(v___x_504_, 0, v___x_509_);
v___x_511_ = v___x_504_;
goto v_reusejp_510_;
}
else
{
lean_object* v_reuseFailAlloc_513_; 
v_reuseFailAlloc_513_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_513_, 0, v___x_509_);
lean_ctor_set(v_reuseFailAlloc_513_, 1, v_a_498_);
v___x_511_ = v_reuseFailAlloc_513_;
goto v_reusejp_510_;
}
v_reusejp_510_:
{
v_a_497_ = v_tail_502_;
v_a_498_ = v___x_511_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__1_spec__1_spec__3___redArg(lean_object* v_keys_516_, lean_object* v_vals_517_, lean_object* v_i_518_, lean_object* v_k_519_){
_start:
{
lean_object* v___x_520_; uint8_t v___x_521_; 
v___x_520_ = lean_array_get_size(v_keys_516_);
v___x_521_ = lean_nat_dec_lt(v_i_518_, v___x_520_);
if (v___x_521_ == 0)
{
lean_object* v___x_522_; 
lean_dec(v_i_518_);
v___x_522_ = lean_box(0);
return v___x_522_;
}
else
{
lean_object* v_k_x27_523_; uint8_t v___x_524_; 
v_k_x27_523_ = lean_array_fget_borrowed(v_keys_516_, v_i_518_);
v___x_524_ = l_Lean_instBEqMVarId_beq(v_k_519_, v_k_x27_523_);
if (v___x_524_ == 0)
{
lean_object* v___x_525_; lean_object* v___x_526_; 
v___x_525_ = lean_unsigned_to_nat(1u);
v___x_526_ = lean_nat_add(v_i_518_, v___x_525_);
lean_dec(v_i_518_);
v_i_518_ = v___x_526_;
goto _start;
}
else
{
lean_object* v___x_528_; lean_object* v___x_529_; 
v___x_528_ = lean_array_fget_borrowed(v_vals_517_, v_i_518_);
lean_dec(v_i_518_);
lean_inc(v___x_528_);
v___x_529_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_529_, 0, v___x_528_);
return v___x_529_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__1_spec__1_spec__3___redArg___boxed(lean_object* v_keys_530_, lean_object* v_vals_531_, lean_object* v_i_532_, lean_object* v_k_533_){
_start:
{
lean_object* v_res_534_; 
v_res_534_ = lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__1_spec__1_spec__3___redArg(v_keys_530_, v_vals_531_, v_i_532_, v_k_533_);
lean_dec(v_k_533_);
lean_dec_ref(v_vals_531_);
lean_dec_ref(v_keys_530_);
return v_res_534_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__1_spec__1___redArg(lean_object* v_x_535_, size_t v_x_536_, lean_object* v_x_537_){
_start:
{
if (lean_obj_tag(v_x_535_) == 0)
{
lean_object* v_es_538_; lean_object* v___x_539_; size_t v___x_540_; size_t v___x_541_; lean_object* v_j_542_; lean_object* v___x_543_; 
v_es_538_ = lean_ctor_get(v_x_535_, 0);
v___x_539_ = lean_box(2);
v___x_540_ = ((size_t)31ULL);
v___x_541_ = lean_usize_land(v_x_536_, v___x_540_);
v_j_542_ = lean_usize_to_nat(v___x_541_);
v___x_543_ = lean_array_get_borrowed(v___x_539_, v_es_538_, v_j_542_);
lean_dec(v_j_542_);
switch(lean_obj_tag(v___x_543_))
{
case 0:
{
lean_object* v_key_544_; lean_object* v_val_545_; uint8_t v___x_546_; 
v_key_544_ = lean_ctor_get(v___x_543_, 0);
v_val_545_ = lean_ctor_get(v___x_543_, 1);
v___x_546_ = l_Lean_instBEqMVarId_beq(v_x_537_, v_key_544_);
if (v___x_546_ == 0)
{
lean_object* v___x_547_; 
v___x_547_ = lean_box(0);
return v___x_547_;
}
else
{
lean_object* v___x_548_; 
lean_inc(v_val_545_);
v___x_548_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_548_, 0, v_val_545_);
return v___x_548_;
}
}
case 1:
{
lean_object* v_node_549_; size_t v___x_550_; size_t v___x_551_; 
v_node_549_ = lean_ctor_get(v___x_543_, 0);
v___x_550_ = ((size_t)5ULL);
v___x_551_ = lean_usize_shift_right(v_x_536_, v___x_550_);
v_x_535_ = v_node_549_;
v_x_536_ = v___x_551_;
goto _start;
}
default: 
{
lean_object* v___x_553_; 
v___x_553_ = lean_box(0);
return v___x_553_;
}
}
}
else
{
lean_object* v_ks_554_; lean_object* v_vs_555_; lean_object* v___x_556_; lean_object* v___x_557_; 
v_ks_554_ = lean_ctor_get(v_x_535_, 0);
v_vs_555_ = lean_ctor_get(v_x_535_, 1);
v___x_556_ = lean_unsigned_to_nat(0u);
v___x_557_ = lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__1_spec__1_spec__3___redArg(v_ks_554_, v_vs_555_, v___x_556_, v_x_537_);
return v___x_557_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__1_spec__1___redArg___boxed(lean_object* v_x_558_, lean_object* v_x_559_, lean_object* v_x_560_){
_start:
{
size_t v_x_2496__boxed_561_; lean_object* v_res_562_; 
v_x_2496__boxed_561_ = lean_unbox_usize(v_x_559_);
lean_dec(v_x_559_);
v_res_562_ = lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__1_spec__1___redArg(v_x_558_, v_x_2496__boxed_561_, v_x_560_);
lean_dec(v_x_560_);
lean_dec_ref(v_x_558_);
return v_res_562_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__1___redArg(lean_object* v_x_563_, lean_object* v_x_564_){
_start:
{
uint64_t v___x_565_; size_t v___x_566_; lean_object* v___x_567_; 
v___x_565_ = l_Lean_instHashableMVarId_hash(v_x_564_);
v___x_566_ = lean_uint64_to_usize(v___x_565_);
v___x_567_ = lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__1_spec__1___redArg(v_x_563_, v___x_566_, v_x_564_);
return v___x_567_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__1___redArg___boxed(lean_object* v_x_568_, lean_object* v_x_569_){
_start:
{
lean_object* v_res_570_; 
v_res_570_ = lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__1___redArg(v_x_568_, v_x_569_);
lean_dec(v_x_569_);
lean_dec_ref(v_x_568_);
return v_res_570_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_filterMapTR_go___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__2(lean_object* v_mctx_571_, lean_object* v_a_572_, lean_object* v_a_573_){
_start:
{
if (lean_obj_tag(v_a_572_) == 0)
{
lean_object* v___x_574_; 
v___x_574_ = lean_array_to_list(v_a_573_);
return v___x_574_;
}
else
{
lean_object* v_head_575_; lean_object* v_tail_576_; lean_object* v_decls_577_; lean_object* v___x_578_; 
v_head_575_ = lean_ctor_get(v_a_572_, 0);
v_tail_576_ = lean_ctor_get(v_a_572_, 1);
v_decls_577_ = lean_ctor_get(v_mctx_571_, 5);
v___x_578_ = lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__1___redArg(v_decls_577_, v_head_575_);
if (lean_obj_tag(v___x_578_) == 0)
{
v_a_572_ = v_tail_576_;
goto _start;
}
else
{
lean_object* v_val_580_; lean_object* v___x_581_; 
v_val_580_ = lean_ctor_get(v___x_578_, 0);
lean_inc(v_val_580_);
lean_dec_ref_known(v___x_578_, 1);
v___x_581_ = lean_array_push(v_a_573_, v_val_580_);
v_a_572_ = v_tail_576_;
v_a_573_ = v___x_581_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_filterMapTR_go___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__2___boxed(lean_object* v_mctx_583_, lean_object* v_a_584_, lean_object* v_a_585_){
_start:
{
lean_object* v_res_586_; 
v_res_586_ = lp_mathlib_List_filterMapTR_go___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__2(v_mctx_583_, v_a_584_, v_a_585_);
lean_dec(v_a_584_);
lean_dec_ref(v_mctx_583_);
return v_res_586_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_getMax_x3f___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__10_spec__11(lean_object* v_as_587_, size_t v_i_588_, size_t v_stop_589_, lean_object* v_b_590_){
_start:
{
lean_object* v___y_592_; uint8_t v___x_596_; 
v___x_596_ = lean_usize_dec_eq(v_i_588_, v_stop_589_);
if (v___x_596_ == 0)
{
lean_object* v_index_597_; lean_object* v___x_598_; lean_object* v_index_599_; uint8_t v___x_600_; 
v_index_597_ = lean_ctor_get(v_b_590_, 6);
v___x_598_ = lean_array_uget_borrowed(v_as_587_, v_i_588_);
v_index_599_ = lean_ctor_get(v___x_598_, 6);
v___x_600_ = lean_nat_dec_lt(v_index_597_, v_index_599_);
if (v___x_600_ == 0)
{
v___y_592_ = v_b_590_;
goto v___jp_591_;
}
else
{
v___y_592_ = v___x_598_;
goto v___jp_591_;
}
}
else
{
lean_inc_ref(v_b_590_);
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
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_getMax_x3f___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__10_spec__11___boxed(lean_object* v_as_601_, lean_object* v_i_602_, lean_object* v_stop_603_, lean_object* v_b_604_){
_start:
{
size_t v_i_boxed_605_; size_t v_stop_boxed_606_; lean_object* v_res_607_; 
v_i_boxed_605_ = lean_unbox_usize(v_i_602_);
lean_dec(v_i_602_);
v_stop_boxed_606_ = lean_unbox_usize(v_stop_603_);
lean_dec(v_stop_603_);
v_res_607_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_getMax_x3f___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__10_spec__11(v_as_601_, v_i_boxed_605_, v_stop_boxed_606_, v_b_604_);
lean_dec_ref(v_b_604_);
lean_dec_ref(v_as_601_);
return v_res_607_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_getMax_x3f___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__10(lean_object* v_as_608_){
_start:
{
lean_object* v___x_609_; lean_object* v___x_610_; uint8_t v___x_611_; 
v___x_609_ = lean_unsigned_to_nat(0u);
v___x_610_ = lean_array_get_size(v_as_608_);
v___x_611_ = lean_nat_dec_lt(v___x_609_, v___x_610_);
if (v___x_611_ == 0)
{
lean_object* v___x_612_; 
v___x_612_ = lean_box(0);
return v___x_612_;
}
else
{
lean_object* v_a0_613_; lean_object* v___x_614_; uint8_t v___x_615_; 
v_a0_613_ = lean_array_fget_borrowed(v_as_608_, v___x_609_);
v___x_614_ = lean_unsigned_to_nat(1u);
v___x_615_ = lean_nat_dec_lt(v___x_614_, v___x_610_);
if (v___x_615_ == 0)
{
lean_object* v___x_616_; 
lean_inc(v_a0_613_);
v___x_616_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_616_, 0, v_a0_613_);
return v___x_616_;
}
else
{
uint8_t v___x_617_; 
v___x_617_ = lean_nat_dec_le(v___x_610_, v___x_610_);
if (v___x_617_ == 0)
{
if (v___x_615_ == 0)
{
lean_object* v___x_618_; 
lean_inc(v_a0_613_);
v___x_618_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_618_, 0, v_a0_613_);
return v___x_618_;
}
else
{
size_t v___x_619_; size_t v___x_620_; lean_object* v___x_621_; lean_object* v___x_622_; 
v___x_619_ = ((size_t)1ULL);
v___x_620_ = lean_usize_of_nat(v___x_610_);
v___x_621_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_getMax_x3f___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__10_spec__11(v_as_608_, v___x_619_, v___x_620_, v_a0_613_);
v___x_622_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_622_, 0, v___x_621_);
return v___x_622_;
}
}
else
{
size_t v___x_623_; size_t v___x_624_; lean_object* v___x_625_; lean_object* v___x_626_; 
v___x_623_ = ((size_t)1ULL);
v___x_624_ = lean_usize_of_nat(v___x_610_);
v___x_625_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_getMax_x3f___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__10_spec__11(v_as_608_, v___x_623_, v___x_624_, v_a0_613_);
v___x_626_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_626_, 0, v___x_625_);
return v___x_626_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_getMax_x3f___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__10___boxed(lean_object* v_as_627_){
_start:
{
lean_object* v_res_628_; 
v_res_628_ = lp_mathlib_Array_getMax_x3f___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__10(v_as_627_);
lean_dec_ref(v_as_627_);
return v_res_628_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_haveLet_nonPropHaves___lam__0(lean_object* v_ctx_633_, lean_object* v_info_634_, lean_object* v_args_635_, lean_object* v___y_636_, lean_object* v___y_637_){
_start:
{
lean_object* v_a_640_; 
if (lean_obj_tag(v_info_634_) == 0)
{
lean_object* v_i_643_; lean_object* v_toElabInfo_644_; lean_object* v_goalsBefore_645_; lean_object* v_mctxAfter_646_; lean_object* v_goalsAfter_647_; lean_object* v_stx_648_; lean_object* v___x_649_; 
v_i_643_ = lean_ctor_get(v_info_634_, 0);
lean_inc_ref(v_i_643_);
lean_dec_ref_known(v_info_634_, 1);
v_toElabInfo_644_ = lean_ctor_get(v_i_643_, 0);
lean_inc_ref(v_toElabInfo_644_);
v_goalsBefore_645_ = lean_ctor_get(v_i_643_, 2);
lean_inc(v_goalsBefore_645_);
v_mctxAfter_646_ = lean_ctor_get(v_i_643_, 3);
lean_inc_ref(v_mctxAfter_646_);
v_goalsAfter_647_ = lean_ctor_get(v_i_643_, 4);
lean_inc(v_goalsAfter_647_);
lean_dec_ref(v_i_643_);
v_stx_648_ = lean_ctor_get(v_toElabInfo_644_, 1);
lean_inc(v_stx_648_);
lean_dec_ref(v_toElabInfo_644_);
v___x_649_ = l_Lean_Syntax_getHeadInfo(v_stx_648_);
if (lean_obj_tag(v___x_649_) == 0)
{
uint8_t v___x_650_; lean_object* v___y_652_; 
lean_dec_ref_known(v___x_649_, 4);
v___x_650_ = lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_isHave_x3f(v_stx_648_);
if (v___x_650_ == 0)
{
lean_object* v___x_679_; 
lean_dec(v_stx_648_);
lean_dec(v_goalsAfter_647_);
lean_dec_ref(v_mctxAfter_646_);
lean_dec(v_goalsBefore_645_);
lean_dec_ref(v_ctx_633_);
v___x_679_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_haveLet_nonPropHaves___lam__0___closed__1));
v_a_640_ = v___x_679_;
goto v___jp_639_;
}
else
{
lean_object* v___x_680_; lean_object* v_mvdecls_681_; lean_object* v___x_682_; lean_object* v___x_683_; 
v___x_680_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_haveLet_nonPropHaves___lam__0___closed__0));
v_mvdecls_681_ = lp_mathlib_List_filterMapTR_go___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__2(v_mctxAfter_646_, v_goalsAfter_647_, v___x_680_);
lean_dec(v_goalsAfter_647_);
v___x_682_ = lean_array_mk(v_mvdecls_681_);
v___x_683_ = lp_mathlib_Array_getMax_x3f___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__10(v___x_682_);
lean_dec_ref(v___x_682_);
if (lean_obj_tag(v___x_683_) == 0)
{
lean_object* v___x_684_; 
v___x_684_ = l_Lean_instInhabitedMetavarDecl_default;
v___y_652_ = v___x_684_;
goto v___jp_651_;
}
else
{
lean_object* v_val_685_; 
v_val_685_ = lean_ctor_get(v___x_683_, 0);
lean_inc(v_val_685_);
lean_dec_ref_known(v___x_683_, 1);
v___y_652_ = v_val_685_;
goto v___jp_651_;
}
}
v___jp_651_:
{
lean_object* v_lctx_653_; lean_object* v_decls_654_; lean_object* v___x_655_; lean_object* v_oldMvdecls_656_; lean_object* v___x_657_; lean_object* v___x_658_; lean_object* v___x_659_; lean_object* v_oldFVars_660_; lean_object* v___x_661_; lean_object* v___x_662_; lean_object* v_newDecls_663_; lean_object* v___x_664_; lean_object* v___x_665_; lean_object* v___x_666_; 
v_lctx_653_ = lean_ctor_get(v___y_652_, 1);
lean_inc_ref(v_lctx_653_);
lean_dec_ref(v___y_652_);
v_decls_654_ = lean_ctor_get(v_lctx_653_, 1);
v___x_655_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_haveLet_nonPropHaves___lam__0___closed__0));
v_oldMvdecls_656_ = lp_mathlib_List_filterMapTR_go___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__2(v_mctxAfter_646_, v_goalsBefore_645_, v___x_655_);
lean_dec(v_goalsBefore_645_);
lean_dec_ref(v_mctxAfter_646_);
v___x_657_ = lean_box(0);
v___x_658_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__3(v_oldMvdecls_656_, v___x_657_);
v___x_659_ = lp_mathlib___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__4(v___x_658_, v___x_655_);
v_oldFVars_660_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__5(v___x_659_, v___x_657_);
v___x_661_ = l_Lean_PersistentArray_toList___redArg(v_decls_654_);
v___x_662_ = lp_mathlib_List_filterMapTR_go___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__0(v___x_661_, v___x_655_);
v_newDecls_663_ = lp_mathlib_List_filterTR_loop___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__7(v_oldFVars_660_, v___x_650_, v___x_662_, v___x_657_);
lean_dec(v_oldFVars_660_);
v___x_664_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__8(v_newDecls_663_, v___x_657_);
v___x_665_ = lean_array_mk(v___x_664_);
v___x_666_ = lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_toFormat__propTypes___redArg(v_ctx_633_, v_lctx_653_, v___x_665_, v___y_636_);
if (lean_obj_tag(v___x_666_) == 0)
{
lean_object* v_a_667_; size_t v_sz_668_; size_t v___x_669_; lean_object* v___x_670_; 
v_a_667_ = lean_ctor_get(v___x_666_, 0);
lean_inc(v_a_667_);
lean_dec_ref_known(v___x_666_, 1);
v_sz_668_ = lean_array_size(v_a_667_);
v___x_669_ = ((size_t)0ULL);
v___x_670_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__9(v___x_650_, v_stx_648_, v_sz_668_, v___x_669_, v_a_667_);
v_a_640_ = v___x_670_;
goto v___jp_639_;
}
else
{
lean_object* v_a_671_; lean_object* v___x_673_; uint8_t v_isShared_674_; uint8_t v_isSharedCheck_678_; 
lean_dec(v_stx_648_);
lean_dec_ref(v_args_635_);
v_a_671_ = lean_ctor_get(v___x_666_, 0);
v_isSharedCheck_678_ = !lean_is_exclusive(v___x_666_);
if (v_isSharedCheck_678_ == 0)
{
v___x_673_ = v___x_666_;
v_isShared_674_ = v_isSharedCheck_678_;
goto v_resetjp_672_;
}
else
{
lean_inc(v_a_671_);
lean_dec(v___x_666_);
v___x_673_ = lean_box(0);
v_isShared_674_ = v_isSharedCheck_678_;
goto v_resetjp_672_;
}
v_resetjp_672_:
{
lean_object* v___x_676_; 
if (v_isShared_674_ == 0)
{
v___x_676_ = v___x_673_;
goto v_reusejp_675_;
}
else
{
lean_object* v_reuseFailAlloc_677_; 
v_reuseFailAlloc_677_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_677_, 0, v_a_671_);
v___x_676_ = v_reuseFailAlloc_677_;
goto v_reusejp_675_;
}
v_reusejp_675_:
{
return v___x_676_;
}
}
}
}
}
else
{
lean_object* v___x_686_; 
lean_dec(v___x_649_);
lean_dec(v_stx_648_);
lean_dec(v_goalsAfter_647_);
lean_dec_ref(v_mctxAfter_646_);
lean_dec(v_goalsBefore_645_);
lean_dec_ref(v_ctx_633_);
v___x_686_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_haveLet_nonPropHaves___lam__0___closed__1));
v_a_640_ = v___x_686_;
goto v___jp_639_;
}
}
else
{
lean_object* v___x_687_; 
lean_dec_ref(v_info_634_);
lean_dec_ref(v_ctx_633_);
v___x_687_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_haveLet_nonPropHaves___lam__0___closed__1));
v_a_640_ = v___x_687_;
goto v___jp_639_;
}
v___jp_639_:
{
lean_object* v___x_641_; lean_object* v___x_642_; 
v___x_641_ = l_Array_append___redArg(v_args_635_, v_a_640_);
lean_dec_ref(v_a_640_);
v___x_642_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_642_, 0, v___x_641_);
return v___x_642_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_haveLet_nonPropHaves___lam__0___boxed(lean_object* v_ctx_688_, lean_object* v_info_689_, lean_object* v_args_690_, lean_object* v___y_691_, lean_object* v___y_692_, lean_object* v___y_693_){
_start:
{
lean_object* v_res_694_; 
v_res_694_ = lp_mathlib_Mathlib_Linter_haveLet_nonPropHaves___lam__0(v_ctx_688_, v_info_689_, v_args_690_, v___y_691_, v___y_692_);
lean_dec(v___y_692_);
lean_dec_ref(v___y_691_);
return v_res_694_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_haveLet_nonPropHaves(lean_object* v_a_696_, lean_object* v_a_697_, lean_object* v_a_698_){
_start:
{
lean_object* v___f_700_; lean_object* v___x_701_; lean_object* v___x_702_; 
v___f_700_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_haveLet_nonPropHaves___closed__0));
v___x_701_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_haveLet_nonPropHaves___lam__0___closed__1));
v___x_702_ = lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_InfoTree_foldInfoM___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__11___redArg(v___f_700_, v___x_701_, v_a_696_, v_a_697_, v_a_698_);
return v___x_702_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_haveLet_nonPropHaves___boxed(lean_object* v_a_703_, lean_object* v_a_704_, lean_object* v_a_705_, lean_object* v_a_706_){
_start:
{
lean_object* v_res_707_; 
v_res_707_ = lp_mathlib_Mathlib_Linter_haveLet_nonPropHaves(v_a_703_, v_a_704_, v_a_705_);
lean_dec(v_a_705_);
lean_dec_ref(v_a_704_);
return v_res_707_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__1(lean_object* v_00_u03b2_708_, lean_object* v_x_709_, lean_object* v_x_710_){
_start:
{
lean_object* v___x_711_; 
v___x_711_ = lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__1___redArg(v_x_709_, v_x_710_);
return v___x_711_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__1___boxed(lean_object* v_00_u03b2_712_, lean_object* v_x_713_, lean_object* v_x_714_){
_start:
{
lean_object* v_res_715_; 
v_res_715_ = lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__1(v_00_u03b2_712_, v_x_713_, v_x_714_);
lean_dec(v_x_714_);
lean_dec_ref(v_x_713_);
return v_res_715_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__1_spec__1(lean_object* v_00_u03b2_716_, lean_object* v_x_717_, size_t v_x_718_, lean_object* v_x_719_){
_start:
{
lean_object* v___x_720_; 
v___x_720_ = lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__1_spec__1___redArg(v_x_717_, v_x_718_, v_x_719_);
return v___x_720_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__1_spec__1___boxed(lean_object* v_00_u03b2_721_, lean_object* v_x_722_, lean_object* v_x_723_, lean_object* v_x_724_){
_start:
{
size_t v_x_2757__boxed_725_; lean_object* v_res_726_; 
v_x_2757__boxed_725_ = lean_unbox_usize(v_x_723_);
lean_dec(v_x_723_);
v_res_726_ = lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__1_spec__1(v_00_u03b2_721_, v_x_722_, v_x_2757__boxed_725_, v_x_724_);
lean_dec(v_x_724_);
lean_dec_ref(v_x_722_);
return v_res_726_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__1_spec__1_spec__3(lean_object* v_00_u03b2_727_, lean_object* v_keys_728_, lean_object* v_vals_729_, lean_object* v_heq_730_, lean_object* v_i_731_, lean_object* v_k_732_){
_start:
{
lean_object* v___x_733_; 
v___x_733_ = lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__1_spec__1_spec__3___redArg(v_keys_728_, v_vals_729_, v_i_731_, v_k_732_);
return v___x_733_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__1_spec__1_spec__3___boxed(lean_object* v_00_u03b2_734_, lean_object* v_keys_735_, lean_object* v_vals_736_, lean_object* v_heq_737_, lean_object* v_i_738_, lean_object* v_k_739_){
_start:
{
lean_object* v_res_740_; 
v_res_740_ = lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Mathlib_Linter_haveLet_nonPropHaves_spec__1_spec__1_spec__3(v_00_u03b2_734_, v_keys_735_, v_vals_736_, v_heq_737_, v_i_738_, v_k_739_);
lean_dec(v_k_739_);
lean_dec_ref(v_vals_736_);
lean_dec_ref(v_keys_735_);
return v_res_740_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__0(lean_object* v_opts_741_, lean_object* v_opt_742_){
_start:
{
lean_object* v_name_743_; lean_object* v_defValue_744_; lean_object* v_map_745_; lean_object* v___x_746_; 
v_name_743_ = lean_ctor_get(v_opt_742_, 0);
v_defValue_744_ = lean_ctor_get(v_opt_742_, 1);
v_map_745_ = lean_ctor_get(v_opts_741_, 0);
v___x_746_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_745_, v_name_743_);
if (lean_obj_tag(v___x_746_) == 0)
{
lean_inc(v_defValue_744_);
return v_defValue_744_;
}
else
{
lean_object* v_val_747_; 
v_val_747_ = lean_ctor_get(v___x_746_, 0);
lean_inc(v_val_747_);
lean_dec_ref_known(v___x_746_, 1);
if (lean_obj_tag(v_val_747_) == 3)
{
lean_object* v_v_748_; 
v_v_748_ = lean_ctor_get(v_val_747_, 0);
lean_inc(v_v_748_);
lean_dec_ref_known(v_val_747_, 1);
return v_v_748_;
}
else
{
lean_dec(v_val_747_);
lean_inc(v_defValue_744_);
return v_defValue_744_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__0___boxed(lean_object* v_opts_749_, lean_object* v_opt_750_){
_start:
{
lean_object* v_res_751_; 
v_res_751_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__0(v_opts_749_, v_opt_750_);
lean_dec_ref(v_opt_750_);
lean_dec_ref(v_opts_749_);
return v_res_751_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getInfoTrees___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__1___redArg(lean_object* v___y_752_){
_start:
{
lean_object* v___x_754_; lean_object* v_infoState_755_; lean_object* v_trees_756_; lean_object* v___x_757_; 
v___x_754_ = lean_st_ref_get(v___y_752_);
v_infoState_755_ = lean_ctor_get(v___x_754_, 8);
lean_inc_ref(v_infoState_755_);
lean_dec(v___x_754_);
v_trees_756_ = lean_ctor_get(v_infoState_755_, 2);
lean_inc_ref(v_trees_756_);
lean_dec_ref(v_infoState_755_);
v___x_757_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_757_, 0, v_trees_756_);
return v___x_757_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getInfoTrees___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__1___redArg___boxed(lean_object* v___y_758_, lean_object* v___y_759_){
_start:
{
lean_object* v_res_760_; 
v_res_760_ = lp_mathlib_Lean_Elab_getInfoTrees___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__1___redArg(v___y_758_);
lean_dec(v___y_758_);
return v_res_760_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getInfoTrees___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__1(lean_object* v___y_761_, lean_object* v___y_762_){
_start:
{
lean_object* v___x_764_; 
v___x_764_ = lp_mathlib_Lean_Elab_getInfoTrees___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__1___redArg(v___y_762_);
return v___x_764_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getInfoTrees___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__1___boxed(lean_object* v___y_765_, lean_object* v___y_766_, lean_object* v___y_767_){
_start:
{
lean_object* v_res_768_; 
v_res_768_ = lp_mathlib_Lean_Elab_getInfoTrees___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__1(v___y_765_, v___y_766_);
lean_dec(v___y_766_);
lean_dec_ref(v___y_765_);
return v_res_768_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3_spec__6___redArg___closed__0(void){
_start:
{
lean_object* v___x_769_; 
v___x_769_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_769_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3_spec__6___redArg___closed__1(void){
_start:
{
lean_object* v___x_770_; lean_object* v___x_771_; 
v___x_770_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3_spec__6___redArg___closed__0, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3_spec__6___redArg___closed__0_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3_spec__6___redArg___closed__0);
v___x_771_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_771_, 0, v___x_770_);
return v___x_771_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3_spec__6___redArg___closed__2(void){
_start:
{
lean_object* v___x_772_; lean_object* v___x_773_; lean_object* v___x_774_; 
v___x_772_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3_spec__6___redArg___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3_spec__6___redArg___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3_spec__6___redArg___closed__1);
v___x_773_ = lean_unsigned_to_nat(0u);
v___x_774_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_774_, 0, v___x_773_);
lean_ctor_set(v___x_774_, 1, v___x_773_);
lean_ctor_set(v___x_774_, 2, v___x_773_);
lean_ctor_set(v___x_774_, 3, v___x_773_);
lean_ctor_set(v___x_774_, 4, v___x_772_);
lean_ctor_set(v___x_774_, 5, v___x_772_);
lean_ctor_set(v___x_774_, 6, v___x_772_);
lean_ctor_set(v___x_774_, 7, v___x_772_);
lean_ctor_set(v___x_774_, 8, v___x_772_);
lean_ctor_set(v___x_774_, 9, v___x_772_);
return v___x_774_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3_spec__6___redArg___closed__3(void){
_start:
{
lean_object* v___x_775_; lean_object* v___x_776_; lean_object* v___x_777_; 
v___x_775_ = lean_unsigned_to_nat(32u);
v___x_776_ = lean_mk_empty_array_with_capacity(v___x_775_);
v___x_777_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_777_, 0, v___x_776_);
return v___x_777_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3_spec__6___redArg___closed__4(void){
_start:
{
size_t v___x_778_; lean_object* v___x_779_; lean_object* v___x_780_; lean_object* v___x_781_; lean_object* v___x_782_; lean_object* v___x_783_; 
v___x_778_ = ((size_t)5ULL);
v___x_779_ = lean_unsigned_to_nat(0u);
v___x_780_ = lean_unsigned_to_nat(32u);
v___x_781_ = lean_mk_empty_array_with_capacity(v___x_780_);
v___x_782_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3_spec__6___redArg___closed__3, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3_spec__6___redArg___closed__3_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3_spec__6___redArg___closed__3);
v___x_783_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_783_, 0, v___x_782_);
lean_ctor_set(v___x_783_, 1, v___x_781_);
lean_ctor_set(v___x_783_, 2, v___x_779_);
lean_ctor_set(v___x_783_, 3, v___x_779_);
lean_ctor_set_usize(v___x_783_, 4, v___x_778_);
return v___x_783_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3_spec__6___redArg___closed__5(void){
_start:
{
lean_object* v___x_784_; lean_object* v___x_785_; lean_object* v___x_786_; lean_object* v___x_787_; 
v___x_784_ = lean_box(1);
v___x_785_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3_spec__6___redArg___closed__4, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3_spec__6___redArg___closed__4_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3_spec__6___redArg___closed__4);
v___x_786_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3_spec__6___redArg___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3_spec__6___redArg___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3_spec__6___redArg___closed__1);
v___x_787_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_787_, 0, v___x_786_);
lean_ctor_set(v___x_787_, 1, v___x_785_);
lean_ctor_set(v___x_787_, 2, v___x_784_);
return v___x_787_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3_spec__6___redArg(lean_object* v_msgData_788_, lean_object* v___y_789_){
_start:
{
lean_object* v___x_791_; lean_object* v_env_792_; lean_object* v___x_793_; lean_object* v_scopes_794_; lean_object* v___x_795_; lean_object* v___x_796_; lean_object* v_opts_797_; lean_object* v___x_798_; lean_object* v___x_799_; lean_object* v___x_800_; lean_object* v___x_801_; lean_object* v___x_802_; 
v___x_791_ = lean_st_ref_get(v___y_789_);
v_env_792_ = lean_ctor_get(v___x_791_, 0);
lean_inc_ref(v_env_792_);
lean_dec(v___x_791_);
v___x_793_ = lean_st_ref_get(v___y_789_);
v_scopes_794_ = lean_ctor_get(v___x_793_, 2);
lean_inc(v_scopes_794_);
lean_dec(v___x_793_);
v___x_795_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_796_ = l_List_head_x21___redArg(v___x_795_, v_scopes_794_);
lean_dec(v_scopes_794_);
v_opts_797_ = lean_ctor_get(v___x_796_, 1);
lean_inc_ref(v_opts_797_);
lean_dec(v___x_796_);
v___x_798_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3_spec__6___redArg___closed__2, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3_spec__6___redArg___closed__2_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3_spec__6___redArg___closed__2);
v___x_799_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3_spec__6___redArg___closed__5, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3_spec__6___redArg___closed__5_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3_spec__6___redArg___closed__5);
v___x_800_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_800_, 0, v_env_792_);
lean_ctor_set(v___x_800_, 1, v___x_798_);
lean_ctor_set(v___x_800_, 2, v___x_799_);
lean_ctor_set(v___x_800_, 3, v_opts_797_);
v___x_801_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_801_, 0, v___x_800_);
lean_ctor_set(v___x_801_, 1, v_msgData_788_);
v___x_802_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_802_, 0, v___x_801_);
return v___x_802_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3_spec__6___redArg___boxed(lean_object* v_msgData_803_, lean_object* v___y_804_, lean_object* v___y_805_){
_start:
{
lean_object* v_res_806_; 
v_res_806_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3_spec__6___redArg(v_msgData_803_, v___y_804_);
lean_dec(v___y_804_);
return v_res_806_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3___lam__0(uint8_t v___y_808_, uint8_t v_suppressElabErrors_809_, lean_object* v_x_810_){
_start:
{
if (lean_obj_tag(v_x_810_) == 1)
{
lean_object* v_pre_811_; 
v_pre_811_ = lean_ctor_get(v_x_810_, 0);
if (lean_obj_tag(v_pre_811_) == 0)
{
lean_object* v_str_812_; lean_object* v___x_813_; uint8_t v___x_814_; 
v_str_812_ = lean_ctor_get(v_x_810_, 1);
v___x_813_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3___lam__0___closed__0));
v___x_814_ = lean_string_dec_eq(v_str_812_, v___x_813_);
if (v___x_814_ == 0)
{
return v___y_808_;
}
else
{
return v_suppressElabErrors_809_;
}
}
else
{
return v___y_808_;
}
}
else
{
return v___y_808_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3___lam__0___boxed(lean_object* v___y_815_, lean_object* v_suppressElabErrors_816_, lean_object* v_x_817_){
_start:
{
uint8_t v___y_8066__boxed_818_; uint8_t v_suppressElabErrors_boxed_819_; uint8_t v_res_820_; lean_object* v_r_821_; 
v___y_8066__boxed_818_ = lean_unbox(v___y_815_);
v_suppressElabErrors_boxed_819_ = lean_unbox(v_suppressElabErrors_816_);
v_res_820_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3___lam__0(v___y_8066__boxed_818_, v_suppressElabErrors_boxed_819_, v_x_817_);
lean_dec(v_x_817_);
v_r_821_ = lean_box(v_res_820_);
return v_r_821_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3_spec__7(lean_object* v_opts_822_, lean_object* v_opt_823_){
_start:
{
lean_object* v_name_824_; lean_object* v_defValue_825_; lean_object* v_map_826_; lean_object* v___x_827_; 
v_name_824_ = lean_ctor_get(v_opt_823_, 0);
v_defValue_825_ = lean_ctor_get(v_opt_823_, 1);
v_map_826_ = lean_ctor_get(v_opts_822_, 0);
v___x_827_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_826_, v_name_824_);
if (lean_obj_tag(v___x_827_) == 0)
{
uint8_t v___x_828_; 
v___x_828_ = lean_unbox(v_defValue_825_);
return v___x_828_;
}
else
{
lean_object* v_val_829_; 
v_val_829_ = lean_ctor_get(v___x_827_, 0);
lean_inc(v_val_829_);
lean_dec_ref_known(v___x_827_, 1);
if (lean_obj_tag(v_val_829_) == 1)
{
uint8_t v_v_830_; 
v_v_830_ = lean_ctor_get_uint8(v_val_829_, 0);
lean_dec_ref_known(v_val_829_, 0);
return v_v_830_;
}
else
{
uint8_t v___x_831_; 
lean_dec(v_val_829_);
v___x_831_ = lean_unbox(v_defValue_825_);
return v___x_831_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3_spec__7___boxed(lean_object* v_opts_832_, lean_object* v_opt_833_){
_start:
{
uint8_t v_res_834_; lean_object* v_r_835_; 
v_res_834_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3_spec__7(v_opts_832_, v_opt_833_);
lean_dec_ref(v_opt_833_);
lean_dec_ref(v_opts_832_);
v_r_835_ = lean_box(v_res_834_);
return v_r_835_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3(lean_object* v_ref_837_, lean_object* v_msgData_838_, uint8_t v_severity_839_, uint8_t v_isSilent_840_, lean_object* v___y_841_, lean_object* v___y_842_){
_start:
{
uint8_t v___y_845_; lean_object* v___y_846_; lean_object* v___y_847_; lean_object* v___y_848_; lean_object* v___y_849_; uint8_t v___y_850_; lean_object* v___y_851_; lean_object* v___y_852_; uint8_t v___y_909_; uint8_t v___y_910_; lean_object* v___y_911_; uint8_t v___y_912_; lean_object* v___y_913_; uint8_t v___y_937_; lean_object* v___y_938_; uint8_t v___y_939_; uint8_t v___y_940_; lean_object* v___y_941_; uint8_t v___y_945_; uint8_t v___y_946_; uint8_t v___y_947_; uint8_t v___x_962_; uint8_t v___y_964_; uint8_t v___y_965_; uint8_t v___y_966_; uint8_t v___y_968_; uint8_t v___x_980_; 
v___x_962_ = 2;
v___x_980_ = l_Lean_instBEqMessageSeverity_beq(v_severity_839_, v___x_962_);
if (v___x_980_ == 0)
{
v___y_968_ = v___x_980_;
goto v___jp_967_;
}
else
{
uint8_t v___x_981_; 
lean_inc_ref(v_msgData_838_);
v___x_981_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_838_);
v___y_968_ = v___x_981_;
goto v___jp_967_;
}
v___jp_844_:
{
lean_object* v___x_853_; 
v___x_853_ = l_Lean_Elab_Command_getScope___redArg(v___y_852_);
if (lean_obj_tag(v___x_853_) == 0)
{
lean_object* v_a_854_; lean_object* v___x_855_; 
v_a_854_ = lean_ctor_get(v___x_853_, 0);
lean_inc(v_a_854_);
lean_dec_ref_known(v___x_853_, 1);
v___x_855_ = l_Lean_Elab_Command_getScope___redArg(v___y_852_);
if (lean_obj_tag(v___x_855_) == 0)
{
lean_object* v_a_856_; lean_object* v___x_858_; uint8_t v_isShared_859_; uint8_t v_isSharedCheck_891_; 
v_a_856_ = lean_ctor_get(v___x_855_, 0);
v_isSharedCheck_891_ = !lean_is_exclusive(v___x_855_);
if (v_isSharedCheck_891_ == 0)
{
v___x_858_ = v___x_855_;
v_isShared_859_ = v_isSharedCheck_891_;
goto v_resetjp_857_;
}
else
{
lean_inc(v_a_856_);
lean_dec(v___x_855_);
v___x_858_ = lean_box(0);
v_isShared_859_ = v_isSharedCheck_891_;
goto v_resetjp_857_;
}
v_resetjp_857_:
{
lean_object* v___x_860_; lean_object* v_currNamespace_861_; lean_object* v_openDecls_862_; lean_object* v_env_863_; lean_object* v_messages_864_; lean_object* v_scopes_865_; lean_object* v_usedQuotCtxts_866_; lean_object* v_nextMacroScope_867_; lean_object* v_maxRecDepth_868_; lean_object* v_ngen_869_; lean_object* v_auxDeclNGen_870_; lean_object* v_infoState_871_; lean_object* v_traceState_872_; lean_object* v_snapshotTasks_873_; lean_object* v_prevLinterStates_874_; lean_object* v___x_876_; uint8_t v_isShared_877_; uint8_t v_isSharedCheck_890_; 
v___x_860_ = lean_st_ref_take(v___y_852_);
v_currNamespace_861_ = lean_ctor_get(v_a_854_, 2);
lean_inc(v_currNamespace_861_);
lean_dec(v_a_854_);
v_openDecls_862_ = lean_ctor_get(v_a_856_, 3);
lean_inc(v_openDecls_862_);
lean_dec(v_a_856_);
v_env_863_ = lean_ctor_get(v___x_860_, 0);
v_messages_864_ = lean_ctor_get(v___x_860_, 1);
v_scopes_865_ = lean_ctor_get(v___x_860_, 2);
v_usedQuotCtxts_866_ = lean_ctor_get(v___x_860_, 3);
v_nextMacroScope_867_ = lean_ctor_get(v___x_860_, 4);
v_maxRecDepth_868_ = lean_ctor_get(v___x_860_, 5);
v_ngen_869_ = lean_ctor_get(v___x_860_, 6);
v_auxDeclNGen_870_ = lean_ctor_get(v___x_860_, 7);
v_infoState_871_ = lean_ctor_get(v___x_860_, 8);
v_traceState_872_ = lean_ctor_get(v___x_860_, 9);
v_snapshotTasks_873_ = lean_ctor_get(v___x_860_, 10);
v_prevLinterStates_874_ = lean_ctor_get(v___x_860_, 11);
v_isSharedCheck_890_ = !lean_is_exclusive(v___x_860_);
if (v_isSharedCheck_890_ == 0)
{
v___x_876_ = v___x_860_;
v_isShared_877_ = v_isSharedCheck_890_;
goto v_resetjp_875_;
}
else
{
lean_inc(v_prevLinterStates_874_);
lean_inc(v_snapshotTasks_873_);
lean_inc(v_traceState_872_);
lean_inc(v_infoState_871_);
lean_inc(v_auxDeclNGen_870_);
lean_inc(v_ngen_869_);
lean_inc(v_maxRecDepth_868_);
lean_inc(v_nextMacroScope_867_);
lean_inc(v_usedQuotCtxts_866_);
lean_inc(v_scopes_865_);
lean_inc(v_messages_864_);
lean_inc(v_env_863_);
lean_dec(v___x_860_);
v___x_876_ = lean_box(0);
v_isShared_877_ = v_isSharedCheck_890_;
goto v_resetjp_875_;
}
v_resetjp_875_:
{
lean_object* v___x_878_; lean_object* v___x_879_; lean_object* v___x_880_; lean_object* v___x_881_; lean_object* v___x_883_; 
v___x_878_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_878_, 0, v_currNamespace_861_);
lean_ctor_set(v___x_878_, 1, v_openDecls_862_);
v___x_879_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_879_, 0, v___x_878_);
lean_ctor_set(v___x_879_, 1, v___y_848_);
lean_inc_ref(v___y_847_);
lean_inc_ref(v___y_846_);
v___x_880_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_880_, 0, v___y_846_);
lean_ctor_set(v___x_880_, 1, v___y_849_);
lean_ctor_set(v___x_880_, 2, v___y_851_);
lean_ctor_set(v___x_880_, 3, v___y_847_);
lean_ctor_set(v___x_880_, 4, v___x_879_);
lean_ctor_set_uint8(v___x_880_, sizeof(void*)*5, v___y_850_);
lean_ctor_set_uint8(v___x_880_, sizeof(void*)*5 + 1, v___y_845_);
lean_ctor_set_uint8(v___x_880_, sizeof(void*)*5 + 2, v_isSilent_840_);
v___x_881_ = l_Lean_MessageLog_add(v___x_880_, v_messages_864_);
if (v_isShared_877_ == 0)
{
lean_ctor_set(v___x_876_, 1, v___x_881_);
v___x_883_ = v___x_876_;
goto v_reusejp_882_;
}
else
{
lean_object* v_reuseFailAlloc_889_; 
v_reuseFailAlloc_889_ = lean_alloc_ctor(0, 12, 0);
lean_ctor_set(v_reuseFailAlloc_889_, 0, v_env_863_);
lean_ctor_set(v_reuseFailAlloc_889_, 1, v___x_881_);
lean_ctor_set(v_reuseFailAlloc_889_, 2, v_scopes_865_);
lean_ctor_set(v_reuseFailAlloc_889_, 3, v_usedQuotCtxts_866_);
lean_ctor_set(v_reuseFailAlloc_889_, 4, v_nextMacroScope_867_);
lean_ctor_set(v_reuseFailAlloc_889_, 5, v_maxRecDepth_868_);
lean_ctor_set(v_reuseFailAlloc_889_, 6, v_ngen_869_);
lean_ctor_set(v_reuseFailAlloc_889_, 7, v_auxDeclNGen_870_);
lean_ctor_set(v_reuseFailAlloc_889_, 8, v_infoState_871_);
lean_ctor_set(v_reuseFailAlloc_889_, 9, v_traceState_872_);
lean_ctor_set(v_reuseFailAlloc_889_, 10, v_snapshotTasks_873_);
lean_ctor_set(v_reuseFailAlloc_889_, 11, v_prevLinterStates_874_);
v___x_883_ = v_reuseFailAlloc_889_;
goto v_reusejp_882_;
}
v_reusejp_882_:
{
lean_object* v___x_884_; lean_object* v___x_885_; lean_object* v___x_887_; 
v___x_884_ = lean_st_ref_set(v___y_852_, v___x_883_);
v___x_885_ = lean_box(0);
if (v_isShared_859_ == 0)
{
lean_ctor_set(v___x_858_, 0, v___x_885_);
v___x_887_ = v___x_858_;
goto v_reusejp_886_;
}
else
{
lean_object* v_reuseFailAlloc_888_; 
v_reuseFailAlloc_888_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_888_, 0, v___x_885_);
v___x_887_ = v_reuseFailAlloc_888_;
goto v_reusejp_886_;
}
v_reusejp_886_:
{
return v___x_887_;
}
}
}
}
}
else
{
lean_object* v_a_892_; lean_object* v___x_894_; uint8_t v_isShared_895_; uint8_t v_isSharedCheck_899_; 
lean_dec(v_a_854_);
lean_dec(v___y_851_);
lean_dec_ref(v___y_849_);
lean_dec_ref(v___y_848_);
v_a_892_ = lean_ctor_get(v___x_855_, 0);
v_isSharedCheck_899_ = !lean_is_exclusive(v___x_855_);
if (v_isSharedCheck_899_ == 0)
{
v___x_894_ = v___x_855_;
v_isShared_895_ = v_isSharedCheck_899_;
goto v_resetjp_893_;
}
else
{
lean_inc(v_a_892_);
lean_dec(v___x_855_);
v___x_894_ = lean_box(0);
v_isShared_895_ = v_isSharedCheck_899_;
goto v_resetjp_893_;
}
v_resetjp_893_:
{
lean_object* v___x_897_; 
if (v_isShared_895_ == 0)
{
v___x_897_ = v___x_894_;
goto v_reusejp_896_;
}
else
{
lean_object* v_reuseFailAlloc_898_; 
v_reuseFailAlloc_898_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_898_, 0, v_a_892_);
v___x_897_ = v_reuseFailAlloc_898_;
goto v_reusejp_896_;
}
v_reusejp_896_:
{
return v___x_897_;
}
}
}
}
else
{
lean_object* v_a_900_; lean_object* v___x_902_; uint8_t v_isShared_903_; uint8_t v_isSharedCheck_907_; 
lean_dec(v___y_851_);
lean_dec_ref(v___y_849_);
lean_dec_ref(v___y_848_);
v_a_900_ = lean_ctor_get(v___x_853_, 0);
v_isSharedCheck_907_ = !lean_is_exclusive(v___x_853_);
if (v_isSharedCheck_907_ == 0)
{
v___x_902_ = v___x_853_;
v_isShared_903_ = v_isSharedCheck_907_;
goto v_resetjp_901_;
}
else
{
lean_inc(v_a_900_);
lean_dec(v___x_853_);
v___x_902_ = lean_box(0);
v_isShared_903_ = v_isSharedCheck_907_;
goto v_resetjp_901_;
}
v_resetjp_901_:
{
lean_object* v___x_905_; 
if (v_isShared_903_ == 0)
{
v___x_905_ = v___x_902_;
goto v_reusejp_904_;
}
else
{
lean_object* v_reuseFailAlloc_906_; 
v_reuseFailAlloc_906_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_906_, 0, v_a_900_);
v___x_905_ = v_reuseFailAlloc_906_;
goto v_reusejp_904_;
}
v_reusejp_904_:
{
return v___x_905_;
}
}
}
}
v___jp_908_:
{
lean_object* v_fileName_914_; lean_object* v_fileMap_915_; uint8_t v_suppressElabErrors_916_; lean_object* v___x_917_; lean_object* v___x_918_; lean_object* v_a_919_; lean_object* v___x_921_; uint8_t v_isShared_922_; uint8_t v_isSharedCheck_935_; 
v_fileName_914_ = lean_ctor_get(v___y_841_, 0);
v_fileMap_915_ = lean_ctor_get(v___y_841_, 1);
v_suppressElabErrors_916_ = lean_ctor_get_uint8(v___y_841_, sizeof(void*)*10);
v___x_917_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_838_);
v___x_918_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3_spec__6___redArg(v___x_917_, v___y_842_);
v_a_919_ = lean_ctor_get(v___x_918_, 0);
v_isSharedCheck_935_ = !lean_is_exclusive(v___x_918_);
if (v_isSharedCheck_935_ == 0)
{
v___x_921_ = v___x_918_;
v_isShared_922_ = v_isSharedCheck_935_;
goto v_resetjp_920_;
}
else
{
lean_inc(v_a_919_);
lean_dec(v___x_918_);
v___x_921_ = lean_box(0);
v_isShared_922_ = v_isSharedCheck_935_;
goto v_resetjp_920_;
}
v_resetjp_920_:
{
lean_object* v___x_923_; lean_object* v___x_924_; lean_object* v___x_925_; lean_object* v___x_926_; 
lean_inc_ref_n(v_fileMap_915_, 2);
v___x_923_ = l_Lean_FileMap_toPosition(v_fileMap_915_, v___y_911_);
lean_dec(v___y_911_);
v___x_924_ = l_Lean_FileMap_toPosition(v_fileMap_915_, v___y_913_);
lean_dec(v___y_913_);
v___x_925_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_925_, 0, v___x_924_);
v___x_926_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3___closed__0));
if (v_suppressElabErrors_916_ == 0)
{
lean_del_object(v___x_921_);
v___y_845_ = v___y_910_;
v___y_846_ = v_fileName_914_;
v___y_847_ = v___x_926_;
v___y_848_ = v_a_919_;
v___y_849_ = v___x_923_;
v___y_850_ = v___y_912_;
v___y_851_ = v___x_925_;
v___y_852_ = v___y_842_;
goto v___jp_844_;
}
else
{
lean_object* v___x_927_; lean_object* v___x_928_; lean_object* v___f_929_; uint8_t v___x_930_; 
v___x_927_ = lean_box(v___y_909_);
v___x_928_ = lean_box(v_suppressElabErrors_916_);
v___f_929_ = lean_alloc_closure((void*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3___lam__0___boxed), 3, 2);
lean_closure_set(v___f_929_, 0, v___x_927_);
lean_closure_set(v___f_929_, 1, v___x_928_);
lean_inc(v_a_919_);
v___x_930_ = l_Lean_MessageData_hasTag(v___f_929_, v_a_919_);
if (v___x_930_ == 0)
{
lean_object* v___x_931_; lean_object* v___x_933_; 
lean_dec_ref_known(v___x_925_, 1);
lean_dec_ref(v___x_923_);
lean_dec(v_a_919_);
v___x_931_ = lean_box(0);
if (v_isShared_922_ == 0)
{
lean_ctor_set(v___x_921_, 0, v___x_931_);
v___x_933_ = v___x_921_;
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
else
{
lean_del_object(v___x_921_);
v___y_845_ = v___y_910_;
v___y_846_ = v_fileName_914_;
v___y_847_ = v___x_926_;
v___y_848_ = v_a_919_;
v___y_849_ = v___x_923_;
v___y_850_ = v___y_912_;
v___y_851_ = v___x_925_;
v___y_852_ = v___y_842_;
goto v___jp_844_;
}
}
}
}
v___jp_936_:
{
lean_object* v___x_942_; 
v___x_942_ = l_Lean_Syntax_getTailPos_x3f(v___y_938_, v___y_940_);
lean_dec(v___y_938_);
if (lean_obj_tag(v___x_942_) == 0)
{
lean_inc(v___y_941_);
v___y_909_ = v___y_937_;
v___y_910_ = v___y_939_;
v___y_911_ = v___y_941_;
v___y_912_ = v___y_940_;
v___y_913_ = v___y_941_;
goto v___jp_908_;
}
else
{
lean_object* v_val_943_; 
v_val_943_ = lean_ctor_get(v___x_942_, 0);
lean_inc(v_val_943_);
lean_dec_ref_known(v___x_942_, 1);
v___y_909_ = v___y_937_;
v___y_910_ = v___y_939_;
v___y_911_ = v___y_941_;
v___y_912_ = v___y_940_;
v___y_913_ = v_val_943_;
goto v___jp_908_;
}
}
v___jp_944_:
{
lean_object* v___x_948_; 
v___x_948_ = l_Lean_Elab_Command_getRef___redArg(v___y_841_);
if (lean_obj_tag(v___x_948_) == 0)
{
lean_object* v_a_949_; lean_object* v_ref_950_; lean_object* v___x_951_; 
v_a_949_ = lean_ctor_get(v___x_948_, 0);
lean_inc(v_a_949_);
lean_dec_ref_known(v___x_948_, 1);
v_ref_950_ = l_Lean_replaceRef(v_ref_837_, v_a_949_);
lean_dec(v_a_949_);
v___x_951_ = l_Lean_Syntax_getPos_x3f(v_ref_950_, v___y_946_);
if (lean_obj_tag(v___x_951_) == 0)
{
lean_object* v___x_952_; 
v___x_952_ = lean_unsigned_to_nat(0u);
v___y_937_ = v___y_945_;
v___y_938_ = v_ref_950_;
v___y_939_ = v___y_947_;
v___y_940_ = v___y_946_;
v___y_941_ = v___x_952_;
goto v___jp_936_;
}
else
{
lean_object* v_val_953_; 
v_val_953_ = lean_ctor_get(v___x_951_, 0);
lean_inc(v_val_953_);
lean_dec_ref_known(v___x_951_, 1);
v___y_937_ = v___y_945_;
v___y_938_ = v_ref_950_;
v___y_939_ = v___y_947_;
v___y_940_ = v___y_946_;
v___y_941_ = v_val_953_;
goto v___jp_936_;
}
}
else
{
lean_object* v_a_954_; lean_object* v___x_956_; uint8_t v_isShared_957_; uint8_t v_isSharedCheck_961_; 
lean_dec_ref(v_msgData_838_);
v_a_954_ = lean_ctor_get(v___x_948_, 0);
v_isSharedCheck_961_ = !lean_is_exclusive(v___x_948_);
if (v_isSharedCheck_961_ == 0)
{
v___x_956_ = v___x_948_;
v_isShared_957_ = v_isSharedCheck_961_;
goto v_resetjp_955_;
}
else
{
lean_inc(v_a_954_);
lean_dec(v___x_948_);
v___x_956_ = lean_box(0);
v_isShared_957_ = v_isSharedCheck_961_;
goto v_resetjp_955_;
}
v_resetjp_955_:
{
lean_object* v___x_959_; 
if (v_isShared_957_ == 0)
{
v___x_959_ = v___x_956_;
goto v_reusejp_958_;
}
else
{
lean_object* v_reuseFailAlloc_960_; 
v_reuseFailAlloc_960_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_960_, 0, v_a_954_);
v___x_959_ = v_reuseFailAlloc_960_;
goto v_reusejp_958_;
}
v_reusejp_958_:
{
return v___x_959_;
}
}
}
}
v___jp_963_:
{
if (v___y_966_ == 0)
{
v___y_945_ = v___y_964_;
v___y_946_ = v___y_965_;
v___y_947_ = v_severity_839_;
goto v___jp_944_;
}
else
{
v___y_945_ = v___y_964_;
v___y_946_ = v___y_965_;
v___y_947_ = v___x_962_;
goto v___jp_944_;
}
}
v___jp_967_:
{
if (v___y_968_ == 0)
{
lean_object* v___x_969_; lean_object* v_scopes_970_; lean_object* v___x_971_; lean_object* v___x_972_; lean_object* v_opts_973_; uint8_t v___x_974_; uint8_t v___x_975_; 
v___x_969_ = lean_st_ref_get(v___y_842_);
v_scopes_970_ = lean_ctor_get(v___x_969_, 2);
lean_inc(v_scopes_970_);
lean_dec(v___x_969_);
v___x_971_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_972_ = l_List_head_x21___redArg(v___x_971_, v_scopes_970_);
lean_dec(v_scopes_970_);
v_opts_973_ = lean_ctor_get(v___x_972_, 1);
lean_inc_ref(v_opts_973_);
lean_dec(v___x_972_);
v___x_974_ = 1;
v___x_975_ = l_Lean_instBEqMessageSeverity_beq(v_severity_839_, v___x_974_);
if (v___x_975_ == 0)
{
lean_dec_ref(v_opts_973_);
v___y_964_ = v___y_968_;
v___y_965_ = v___y_968_;
v___y_966_ = v___x_975_;
goto v___jp_963_;
}
else
{
lean_object* v___x_976_; uint8_t v___x_977_; 
v___x_976_ = l_Lean_warningAsError;
v___x_977_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3_spec__7(v_opts_973_, v___x_976_);
lean_dec_ref(v_opts_973_);
v___y_964_ = v___y_968_;
v___y_965_ = v___y_968_;
v___y_966_ = v___x_977_;
goto v___jp_963_;
}
}
else
{
lean_object* v___x_978_; lean_object* v___x_979_; 
lean_dec_ref(v_msgData_838_);
v___x_978_ = lean_box(0);
v___x_979_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_979_, 0, v___x_978_);
return v___x_979_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3___boxed(lean_object* v_ref_982_, lean_object* v_msgData_983_, lean_object* v_severity_984_, lean_object* v_isSilent_985_, lean_object* v___y_986_, lean_object* v___y_987_, lean_object* v___y_988_){
_start:
{
uint8_t v_severity_boxed_989_; uint8_t v_isSilent_boxed_990_; lean_object* v_res_991_; 
v_severity_boxed_989_ = lean_unbox(v_severity_984_);
v_isSilent_boxed_990_ = lean_unbox(v_isSilent_985_);
v_res_991_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3(v_ref_982_, v_msgData_983_, v_severity_boxed_989_, v_isSilent_boxed_990_, v___y_986_, v___y_987_);
lean_dec(v___y_987_);
lean_dec_ref(v___y_986_);
lean_dec(v_ref_982_);
return v_res_991_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2(lean_object* v_ref_992_, lean_object* v_msgData_993_, lean_object* v___y_994_, lean_object* v___y_995_){
_start:
{
uint8_t v___x_997_; uint8_t v___x_998_; lean_object* v___x_999_; 
v___x_997_ = 1;
v___x_998_ = 0;
v___x_999_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3(v_ref_992_, v_msgData_993_, v___x_997_, v___x_998_, v___y_994_, v___y_995_);
return v___x_999_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2___boxed(lean_object* v_ref_1000_, lean_object* v_msgData_1001_, lean_object* v___y_1002_, lean_object* v___y_1003_, lean_object* v___y_1004_){
_start:
{
lean_object* v_res_1005_; 
v_res_1005_ = lp_mathlib_Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2(v_ref_1000_, v_msgData_1001_, v___y_1002_, v___y_1003_);
lean_dec(v___y_1003_);
lean_dec_ref(v___y_1002_);
lean_dec(v_ref_1000_);
return v_res_1005_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2___closed__1(void){
_start:
{
lean_object* v___x_1007_; lean_object* v___x_1008_; 
v___x_1007_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2___closed__0));
v___x_1008_ = l_Lean_stringToMessageData(v___x_1007_);
return v___x_1008_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2___closed__3(void){
_start:
{
lean_object* v___x_1010_; lean_object* v___x_1011_; 
v___x_1010_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2___closed__2));
v___x_1011_ = l_Lean_stringToMessageData(v___x_1010_);
return v___x_1011_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2(lean_object* v_linterOption_1012_, lean_object* v_stx_1013_, lean_object* v_msg_1014_, lean_object* v___y_1015_, lean_object* v___y_1016_){
_start:
{
lean_object* v_name_1018_; lean_object* v___x_1020_; uint8_t v_isShared_1021_; uint8_t v_isSharedCheck_1033_; 
v_name_1018_ = lean_ctor_get(v_linterOption_1012_, 0);
v_isSharedCheck_1033_ = !lean_is_exclusive(v_linterOption_1012_);
if (v_isSharedCheck_1033_ == 0)
{
lean_object* v_unused_1034_; 
v_unused_1034_ = lean_ctor_get(v_linterOption_1012_, 1);
lean_dec(v_unused_1034_);
v___x_1020_ = v_linterOption_1012_;
v_isShared_1021_ = v_isSharedCheck_1033_;
goto v_resetjp_1019_;
}
else
{
lean_inc(v_name_1018_);
lean_dec(v_linterOption_1012_);
v___x_1020_ = lean_box(0);
v_isShared_1021_ = v_isSharedCheck_1033_;
goto v_resetjp_1019_;
}
v_resetjp_1019_:
{
lean_object* v___x_1022_; lean_object* v___x_1023_; lean_object* v___x_1025_; 
v___x_1022_ = lean_obj_once(&lp_mathlib_Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2___closed__1, &lp_mathlib_Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2___closed__1_once, _init_lp_mathlib_Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2___closed__1);
lean_inc(v_name_1018_);
v___x_1023_ = l_Lean_MessageData_ofName(v_name_1018_);
if (v_isShared_1021_ == 0)
{
lean_ctor_set_tag(v___x_1020_, 7);
lean_ctor_set(v___x_1020_, 1, v___x_1023_);
lean_ctor_set(v___x_1020_, 0, v___x_1022_);
v___x_1025_ = v___x_1020_;
goto v_reusejp_1024_;
}
else
{
lean_object* v_reuseFailAlloc_1032_; 
v_reuseFailAlloc_1032_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1032_, 0, v___x_1022_);
lean_ctor_set(v_reuseFailAlloc_1032_, 1, v___x_1023_);
v___x_1025_ = v_reuseFailAlloc_1032_;
goto v_reusejp_1024_;
}
v_reusejp_1024_:
{
lean_object* v___x_1026_; lean_object* v___x_1027_; lean_object* v_disable_1028_; lean_object* v___x_1029_; lean_object* v___x_1030_; lean_object* v___x_1031_; 
v___x_1026_ = lean_obj_once(&lp_mathlib_Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2___closed__3, &lp_mathlib_Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2___closed__3_once, _init_lp_mathlib_Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2___closed__3);
v___x_1027_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1027_, 0, v___x_1025_);
lean_ctor_set(v___x_1027_, 1, v___x_1026_);
v_disable_1028_ = l_Lean_MessageData_note(v___x_1027_);
v___x_1029_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1029_, 0, v_msg_1014_);
lean_ctor_set(v___x_1029_, 1, v_disable_1028_);
v___x_1030_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_1030_, 0, v_name_1018_);
lean_ctor_set(v___x_1030_, 1, v___x_1029_);
v___x_1031_ = lp_mathlib_Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2(v_stx_1013_, v___x_1030_, v___y_1015_, v___y_1016_);
return v___x_1031_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2___boxed(lean_object* v_linterOption_1035_, lean_object* v_stx_1036_, lean_object* v_msg_1037_, lean_object* v___y_1038_, lean_object* v___y_1039_, lean_object* v___y_1040_){
_start:
{
lean_object* v_res_1041_; 
v_res_1041_ = lp_mathlib_Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2(v_linterOption_1035_, v_stx_1036_, v_msg_1037_, v___y_1038_, v___y_1039_);
lean_dec(v___y_1039_);
lean_dec_ref(v___y_1038_);
lean_dec(v_stx_1036_);
return v_res_1041_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__3___closed__1(void){
_start:
{
lean_object* v___x_1043_; lean_object* v___x_1044_; 
v___x_1043_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__3___closed__0));
v___x_1044_ = l_Lean_stringToMessageData(v___x_1043_);
return v___x_1044_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__3___closed__3(void){
_start:
{
lean_object* v___x_1046_; lean_object* v___x_1047_; 
v___x_1046_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__3___closed__2));
v___x_1047_ = l_Lean_stringToMessageData(v___x_1046_);
return v___x_1047_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__3(lean_object* v_as_1048_, size_t v_sz_1049_, size_t v_i_1050_, lean_object* v_b_1051_, lean_object* v___y_1052_, lean_object* v___y_1053_){
_start:
{
uint8_t v___x_1055_; 
v___x_1055_ = lean_usize_dec_lt(v_i_1050_, v_sz_1049_);
if (v___x_1055_ == 0)
{
lean_object* v___x_1056_; 
v___x_1056_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1056_, 0, v_b_1051_);
return v___x_1056_;
}
else
{
lean_object* v_a_1057_; lean_object* v_fst_1058_; lean_object* v_snd_1059_; lean_object* v___x_1061_; uint8_t v_isShared_1062_; uint8_t v_isSharedCheck_1076_; 
v_a_1057_ = lean_array_uget(v_as_1048_, v_i_1050_);
v_fst_1058_ = lean_ctor_get(v_a_1057_, 0);
v_snd_1059_ = lean_ctor_get(v_a_1057_, 1);
v_isSharedCheck_1076_ = !lean_is_exclusive(v_a_1057_);
if (v_isSharedCheck_1076_ == 0)
{
v___x_1061_ = v_a_1057_;
v_isShared_1062_ = v_isSharedCheck_1076_;
goto v_resetjp_1060_;
}
else
{
lean_inc(v_snd_1059_);
lean_inc(v_fst_1058_);
lean_dec(v_a_1057_);
v___x_1061_ = lean_box(0);
v_isShared_1062_ = v_isSharedCheck_1076_;
goto v_resetjp_1060_;
}
v_resetjp_1060_:
{
lean_object* v___x_1063_; lean_object* v___x_1064_; lean_object* v___x_1065_; lean_object* v___x_1067_; 
v___x_1063_ = lp_mathlib_Mathlib_Linter_linter_haveLet;
v___x_1064_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__3___closed__1, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__3___closed__1_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__3___closed__1);
v___x_1065_ = l_Lean_MessageData_ofFormat(v_snd_1059_);
if (v_isShared_1062_ == 0)
{
lean_ctor_set_tag(v___x_1061_, 7);
lean_ctor_set(v___x_1061_, 1, v___x_1065_);
lean_ctor_set(v___x_1061_, 0, v___x_1064_);
v___x_1067_ = v___x_1061_;
goto v_reusejp_1066_;
}
else
{
lean_object* v_reuseFailAlloc_1075_; 
v_reuseFailAlloc_1075_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1075_, 0, v___x_1064_);
lean_ctor_set(v_reuseFailAlloc_1075_, 1, v___x_1065_);
v___x_1067_ = v_reuseFailAlloc_1075_;
goto v_reusejp_1066_;
}
v_reusejp_1066_:
{
lean_object* v___x_1068_; lean_object* v___x_1069_; lean_object* v___x_1070_; 
v___x_1068_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__3___closed__3, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__3___closed__3_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__3___closed__3);
v___x_1069_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1069_, 0, v___x_1067_);
lean_ctor_set(v___x_1069_, 1, v___x_1068_);
v___x_1070_ = lp_mathlib_Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2(v___x_1063_, v_fst_1058_, v___x_1069_, v___y_1052_, v___y_1053_);
lean_dec(v_fst_1058_);
if (lean_obj_tag(v___x_1070_) == 0)
{
lean_object* v___x_1071_; size_t v___x_1072_; size_t v___x_1073_; 
lean_dec_ref_known(v___x_1070_, 1);
v___x_1071_ = lean_box(0);
v___x_1072_ = ((size_t)1ULL);
v___x_1073_ = lean_usize_add(v_i_1050_, v___x_1072_);
v_i_1050_ = v___x_1073_;
v_b_1051_ = v___x_1071_;
goto _start;
}
else
{
return v___x_1070_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__3___boxed(lean_object* v_as_1077_, lean_object* v_sz_1078_, lean_object* v_i_1079_, lean_object* v_b_1080_, lean_object* v___y_1081_, lean_object* v___y_1082_, lean_object* v___y_1083_){
_start:
{
size_t v_sz_boxed_1084_; size_t v_i_boxed_1085_; lean_object* v_res_1086_; 
v_sz_boxed_1084_ = lean_unbox_usize(v_sz_1078_);
lean_dec(v_sz_1078_);
v_i_boxed_1085_ = lean_unbox_usize(v_i_1079_);
lean_dec(v_i_1079_);
v_res_1086_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__3(v_as_1077_, v_sz_boxed_1084_, v_i_boxed_1085_, v_b_1080_, v___y_1081_, v___y_1082_);
lean_dec(v___y_1082_);
lean_dec_ref(v___y_1081_);
lean_dec_ref(v_as_1077_);
return v_res_1086_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__4_spec__6_spec__10(lean_object* v_as_1090_, size_t v_sz_1091_, size_t v_i_1092_, lean_object* v_b_1093_, lean_object* v___y_1094_, lean_object* v___y_1095_){
_start:
{
uint8_t v___x_1097_; 
v___x_1097_ = lean_usize_dec_lt(v_i_1092_, v_sz_1091_);
if (v___x_1097_ == 0)
{
lean_object* v___x_1098_; 
v___x_1098_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1098_, 0, v_b_1093_);
return v___x_1098_;
}
else
{
lean_object* v_a_1099_; lean_object* v___x_1100_; 
lean_dec_ref(v_b_1093_);
v_a_1099_ = lean_array_uget_borrowed(v_as_1090_, v_i_1092_);
lean_inc(v_a_1099_);
v___x_1100_ = lp_mathlib_Mathlib_Linter_haveLet_nonPropHaves(v_a_1099_, v___y_1094_, v___y_1095_);
if (lean_obj_tag(v___x_1100_) == 0)
{
lean_object* v_a_1101_; lean_object* v___x_1102_; size_t v_sz_1103_; size_t v___x_1104_; lean_object* v___x_1105_; 
v_a_1101_ = lean_ctor_get(v___x_1100_, 0);
lean_inc(v_a_1101_);
lean_dec_ref_known(v___x_1100_, 1);
v___x_1102_ = lean_box(0);
v_sz_1103_ = lean_array_size(v_a_1101_);
v___x_1104_ = ((size_t)0ULL);
v___x_1105_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__3(v_a_1101_, v_sz_1103_, v___x_1104_, v___x_1102_, v___y_1094_, v___y_1095_);
lean_dec(v_a_1101_);
if (lean_obj_tag(v___x_1105_) == 0)
{
lean_object* v___x_1106_; size_t v___x_1107_; size_t v___x_1108_; 
lean_dec_ref_known(v___x_1105_, 1);
v___x_1106_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__4_spec__6_spec__10___closed__0));
v___x_1107_ = ((size_t)1ULL);
v___x_1108_ = lean_usize_add(v_i_1092_, v___x_1107_);
v_i_1092_ = v___x_1108_;
v_b_1093_ = v___x_1106_;
goto _start;
}
else
{
lean_object* v_a_1110_; lean_object* v___x_1112_; uint8_t v_isShared_1113_; uint8_t v_isSharedCheck_1117_; 
v_a_1110_ = lean_ctor_get(v___x_1105_, 0);
v_isSharedCheck_1117_ = !lean_is_exclusive(v___x_1105_);
if (v_isSharedCheck_1117_ == 0)
{
v___x_1112_ = v___x_1105_;
v_isShared_1113_ = v_isSharedCheck_1117_;
goto v_resetjp_1111_;
}
else
{
lean_inc(v_a_1110_);
lean_dec(v___x_1105_);
v___x_1112_ = lean_box(0);
v_isShared_1113_ = v_isSharedCheck_1117_;
goto v_resetjp_1111_;
}
v_resetjp_1111_:
{
lean_object* v___x_1115_; 
if (v_isShared_1113_ == 0)
{
v___x_1115_ = v___x_1112_;
goto v_reusejp_1114_;
}
else
{
lean_object* v_reuseFailAlloc_1116_; 
v_reuseFailAlloc_1116_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1116_, 0, v_a_1110_);
v___x_1115_ = v_reuseFailAlloc_1116_;
goto v_reusejp_1114_;
}
v_reusejp_1114_:
{
return v___x_1115_;
}
}
}
}
else
{
lean_object* v_a_1118_; lean_object* v___x_1120_; uint8_t v_isShared_1121_; uint8_t v_isSharedCheck_1125_; 
v_a_1118_ = lean_ctor_get(v___x_1100_, 0);
v_isSharedCheck_1125_ = !lean_is_exclusive(v___x_1100_);
if (v_isSharedCheck_1125_ == 0)
{
v___x_1120_ = v___x_1100_;
v_isShared_1121_ = v_isSharedCheck_1125_;
goto v_resetjp_1119_;
}
else
{
lean_inc(v_a_1118_);
lean_dec(v___x_1100_);
v___x_1120_ = lean_box(0);
v_isShared_1121_ = v_isSharedCheck_1125_;
goto v_resetjp_1119_;
}
v_resetjp_1119_:
{
lean_object* v___x_1123_; 
if (v_isShared_1121_ == 0)
{
v___x_1123_ = v___x_1120_;
goto v_reusejp_1122_;
}
else
{
lean_object* v_reuseFailAlloc_1124_; 
v_reuseFailAlloc_1124_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1124_, 0, v_a_1118_);
v___x_1123_ = v_reuseFailAlloc_1124_;
goto v_reusejp_1122_;
}
v_reusejp_1122_:
{
return v___x_1123_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__4_spec__6_spec__10___boxed(lean_object* v_as_1126_, lean_object* v_sz_1127_, lean_object* v_i_1128_, lean_object* v_b_1129_, lean_object* v___y_1130_, lean_object* v___y_1131_, lean_object* v___y_1132_){
_start:
{
size_t v_sz_boxed_1133_; size_t v_i_boxed_1134_; lean_object* v_res_1135_; 
v_sz_boxed_1133_ = lean_unbox_usize(v_sz_1127_);
lean_dec(v_sz_1127_);
v_i_boxed_1134_ = lean_unbox_usize(v_i_1128_);
lean_dec(v_i_1128_);
v_res_1135_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__4_spec__6_spec__10(v_as_1126_, v_sz_boxed_1133_, v_i_boxed_1134_, v_b_1129_, v___y_1130_, v___y_1131_);
lean_dec(v___y_1131_);
lean_dec_ref(v___y_1130_);
lean_dec_ref(v_as_1126_);
return v_res_1135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__4_spec__6(lean_object* v_as_1136_, size_t v_sz_1137_, size_t v_i_1138_, lean_object* v_b_1139_, lean_object* v___y_1140_, lean_object* v___y_1141_){
_start:
{
uint8_t v___x_1143_; 
v___x_1143_ = lean_usize_dec_lt(v_i_1138_, v_sz_1137_);
if (v___x_1143_ == 0)
{
lean_object* v___x_1144_; 
v___x_1144_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1144_, 0, v_b_1139_);
return v___x_1144_;
}
else
{
lean_object* v_a_1145_; lean_object* v___x_1146_; 
lean_dec_ref(v_b_1139_);
v_a_1145_ = lean_array_uget_borrowed(v_as_1136_, v_i_1138_);
lean_inc(v_a_1145_);
v___x_1146_ = lp_mathlib_Mathlib_Linter_haveLet_nonPropHaves(v_a_1145_, v___y_1140_, v___y_1141_);
if (lean_obj_tag(v___x_1146_) == 0)
{
lean_object* v_a_1147_; lean_object* v___x_1148_; size_t v_sz_1149_; size_t v___x_1150_; lean_object* v___x_1151_; 
v_a_1147_ = lean_ctor_get(v___x_1146_, 0);
lean_inc(v_a_1147_);
lean_dec_ref_known(v___x_1146_, 1);
v___x_1148_ = lean_box(0);
v_sz_1149_ = lean_array_size(v_a_1147_);
v___x_1150_ = ((size_t)0ULL);
v___x_1151_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__3(v_a_1147_, v_sz_1149_, v___x_1150_, v___x_1148_, v___y_1140_, v___y_1141_);
lean_dec(v_a_1147_);
if (lean_obj_tag(v___x_1151_) == 0)
{
lean_object* v___x_1152_; size_t v___x_1153_; size_t v___x_1154_; lean_object* v___x_1155_; 
lean_dec_ref_known(v___x_1151_, 1);
v___x_1152_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__4_spec__6_spec__10___closed__0));
v___x_1153_ = ((size_t)1ULL);
v___x_1154_ = lean_usize_add(v_i_1138_, v___x_1153_);
v___x_1155_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__4_spec__6_spec__10(v_as_1136_, v_sz_1137_, v___x_1154_, v___x_1152_, v___y_1140_, v___y_1141_);
return v___x_1155_;
}
else
{
lean_object* v_a_1156_; lean_object* v___x_1158_; uint8_t v_isShared_1159_; uint8_t v_isSharedCheck_1163_; 
v_a_1156_ = lean_ctor_get(v___x_1151_, 0);
v_isSharedCheck_1163_ = !lean_is_exclusive(v___x_1151_);
if (v_isSharedCheck_1163_ == 0)
{
v___x_1158_ = v___x_1151_;
v_isShared_1159_ = v_isSharedCheck_1163_;
goto v_resetjp_1157_;
}
else
{
lean_inc(v_a_1156_);
lean_dec(v___x_1151_);
v___x_1158_ = lean_box(0);
v_isShared_1159_ = v_isSharedCheck_1163_;
goto v_resetjp_1157_;
}
v_resetjp_1157_:
{
lean_object* v___x_1161_; 
if (v_isShared_1159_ == 0)
{
v___x_1161_ = v___x_1158_;
goto v_reusejp_1160_;
}
else
{
lean_object* v_reuseFailAlloc_1162_; 
v_reuseFailAlloc_1162_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1162_, 0, v_a_1156_);
v___x_1161_ = v_reuseFailAlloc_1162_;
goto v_reusejp_1160_;
}
v_reusejp_1160_:
{
return v___x_1161_;
}
}
}
}
else
{
lean_object* v_a_1164_; lean_object* v___x_1166_; uint8_t v_isShared_1167_; uint8_t v_isSharedCheck_1171_; 
v_a_1164_ = lean_ctor_get(v___x_1146_, 0);
v_isSharedCheck_1171_ = !lean_is_exclusive(v___x_1146_);
if (v_isSharedCheck_1171_ == 0)
{
v___x_1166_ = v___x_1146_;
v_isShared_1167_ = v_isSharedCheck_1171_;
goto v_resetjp_1165_;
}
else
{
lean_inc(v_a_1164_);
lean_dec(v___x_1146_);
v___x_1166_ = lean_box(0);
v_isShared_1167_ = v_isSharedCheck_1171_;
goto v_resetjp_1165_;
}
v_resetjp_1165_:
{
lean_object* v___x_1169_; 
if (v_isShared_1167_ == 0)
{
v___x_1169_ = v___x_1166_;
goto v_reusejp_1168_;
}
else
{
lean_object* v_reuseFailAlloc_1170_; 
v_reuseFailAlloc_1170_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1170_, 0, v_a_1164_);
v___x_1169_ = v_reuseFailAlloc_1170_;
goto v_reusejp_1168_;
}
v_reusejp_1168_:
{
return v___x_1169_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__4_spec__6___boxed(lean_object* v_as_1172_, lean_object* v_sz_1173_, lean_object* v_i_1174_, lean_object* v_b_1175_, lean_object* v___y_1176_, lean_object* v___y_1177_, lean_object* v___y_1178_){
_start:
{
size_t v_sz_boxed_1179_; size_t v_i_boxed_1180_; lean_object* v_res_1181_; 
v_sz_boxed_1179_ = lean_unbox_usize(v_sz_1173_);
lean_dec(v_sz_1173_);
v_i_boxed_1180_ = lean_unbox_usize(v_i_1174_);
lean_dec(v_i_1174_);
v_res_1181_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__4_spec__6(v_as_1172_, v_sz_boxed_1179_, v_i_boxed_1180_, v_b_1175_, v___y_1176_, v___y_1177_);
lean_dec(v___y_1177_);
lean_dec_ref(v___y_1176_);
lean_dec_ref(v_as_1172_);
return v_res_1181_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__4_spec__5_spec__8_spec__11(lean_object* v_as_1185_, size_t v_sz_1186_, size_t v_i_1187_, lean_object* v_b_1188_, lean_object* v___y_1189_, lean_object* v___y_1190_){
_start:
{
uint8_t v___x_1192_; 
v___x_1192_ = lean_usize_dec_lt(v_i_1187_, v_sz_1186_);
if (v___x_1192_ == 0)
{
lean_object* v___x_1193_; 
v___x_1193_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1193_, 0, v_b_1188_);
return v___x_1193_;
}
else
{
lean_object* v_a_1194_; lean_object* v___x_1195_; 
lean_dec_ref(v_b_1188_);
v_a_1194_ = lean_array_uget_borrowed(v_as_1185_, v_i_1187_);
lean_inc(v_a_1194_);
v___x_1195_ = lp_mathlib_Mathlib_Linter_haveLet_nonPropHaves(v_a_1194_, v___y_1189_, v___y_1190_);
if (lean_obj_tag(v___x_1195_) == 0)
{
lean_object* v_a_1196_; lean_object* v___x_1197_; size_t v_sz_1198_; size_t v___x_1199_; lean_object* v___x_1200_; 
v_a_1196_ = lean_ctor_get(v___x_1195_, 0);
lean_inc(v_a_1196_);
lean_dec_ref_known(v___x_1195_, 1);
v___x_1197_ = lean_box(0);
v_sz_1198_ = lean_array_size(v_a_1196_);
v___x_1199_ = ((size_t)0ULL);
v___x_1200_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__3(v_a_1196_, v_sz_1198_, v___x_1199_, v___x_1197_, v___y_1189_, v___y_1190_);
lean_dec(v_a_1196_);
if (lean_obj_tag(v___x_1200_) == 0)
{
lean_object* v___x_1201_; size_t v___x_1202_; size_t v___x_1203_; 
lean_dec_ref_known(v___x_1200_, 1);
v___x_1201_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__4_spec__5_spec__8_spec__11___closed__0));
v___x_1202_ = ((size_t)1ULL);
v___x_1203_ = lean_usize_add(v_i_1187_, v___x_1202_);
v_i_1187_ = v___x_1203_;
v_b_1188_ = v___x_1201_;
goto _start;
}
else
{
lean_object* v_a_1205_; lean_object* v___x_1207_; uint8_t v_isShared_1208_; uint8_t v_isSharedCheck_1212_; 
v_a_1205_ = lean_ctor_get(v___x_1200_, 0);
v_isSharedCheck_1212_ = !lean_is_exclusive(v___x_1200_);
if (v_isSharedCheck_1212_ == 0)
{
v___x_1207_ = v___x_1200_;
v_isShared_1208_ = v_isSharedCheck_1212_;
goto v_resetjp_1206_;
}
else
{
lean_inc(v_a_1205_);
lean_dec(v___x_1200_);
v___x_1207_ = lean_box(0);
v_isShared_1208_ = v_isSharedCheck_1212_;
goto v_resetjp_1206_;
}
v_resetjp_1206_:
{
lean_object* v___x_1210_; 
if (v_isShared_1208_ == 0)
{
v___x_1210_ = v___x_1207_;
goto v_reusejp_1209_;
}
else
{
lean_object* v_reuseFailAlloc_1211_; 
v_reuseFailAlloc_1211_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1211_, 0, v_a_1205_);
v___x_1210_ = v_reuseFailAlloc_1211_;
goto v_reusejp_1209_;
}
v_reusejp_1209_:
{
return v___x_1210_;
}
}
}
}
else
{
lean_object* v_a_1213_; lean_object* v___x_1215_; uint8_t v_isShared_1216_; uint8_t v_isSharedCheck_1220_; 
v_a_1213_ = lean_ctor_get(v___x_1195_, 0);
v_isSharedCheck_1220_ = !lean_is_exclusive(v___x_1195_);
if (v_isSharedCheck_1220_ == 0)
{
v___x_1215_ = v___x_1195_;
v_isShared_1216_ = v_isSharedCheck_1220_;
goto v_resetjp_1214_;
}
else
{
lean_inc(v_a_1213_);
lean_dec(v___x_1195_);
v___x_1215_ = lean_box(0);
v_isShared_1216_ = v_isSharedCheck_1220_;
goto v_resetjp_1214_;
}
v_resetjp_1214_:
{
lean_object* v___x_1218_; 
if (v_isShared_1216_ == 0)
{
v___x_1218_ = v___x_1215_;
goto v_reusejp_1217_;
}
else
{
lean_object* v_reuseFailAlloc_1219_; 
v_reuseFailAlloc_1219_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1219_, 0, v_a_1213_);
v___x_1218_ = v_reuseFailAlloc_1219_;
goto v_reusejp_1217_;
}
v_reusejp_1217_:
{
return v___x_1218_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__4_spec__5_spec__8_spec__11___boxed(lean_object* v_as_1221_, lean_object* v_sz_1222_, lean_object* v_i_1223_, lean_object* v_b_1224_, lean_object* v___y_1225_, lean_object* v___y_1226_, lean_object* v___y_1227_){
_start:
{
size_t v_sz_boxed_1228_; size_t v_i_boxed_1229_; lean_object* v_res_1230_; 
v_sz_boxed_1228_ = lean_unbox_usize(v_sz_1222_);
lean_dec(v_sz_1222_);
v_i_boxed_1229_ = lean_unbox_usize(v_i_1223_);
lean_dec(v_i_1223_);
v_res_1230_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__4_spec__5_spec__8_spec__11(v_as_1221_, v_sz_boxed_1228_, v_i_boxed_1229_, v_b_1224_, v___y_1225_, v___y_1226_);
lean_dec(v___y_1226_);
lean_dec_ref(v___y_1225_);
lean_dec_ref(v_as_1221_);
return v_res_1230_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__4_spec__5_spec__8(lean_object* v_as_1231_, size_t v_sz_1232_, size_t v_i_1233_, lean_object* v_b_1234_, lean_object* v___y_1235_, lean_object* v___y_1236_){
_start:
{
uint8_t v___x_1238_; 
v___x_1238_ = lean_usize_dec_lt(v_i_1233_, v_sz_1232_);
if (v___x_1238_ == 0)
{
lean_object* v___x_1239_; 
v___x_1239_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1239_, 0, v_b_1234_);
return v___x_1239_;
}
else
{
lean_object* v_a_1240_; lean_object* v___x_1241_; 
lean_dec_ref(v_b_1234_);
v_a_1240_ = lean_array_uget_borrowed(v_as_1231_, v_i_1233_);
lean_inc(v_a_1240_);
v___x_1241_ = lp_mathlib_Mathlib_Linter_haveLet_nonPropHaves(v_a_1240_, v___y_1235_, v___y_1236_);
if (lean_obj_tag(v___x_1241_) == 0)
{
lean_object* v_a_1242_; lean_object* v___x_1243_; size_t v_sz_1244_; size_t v___x_1245_; lean_object* v___x_1246_; 
v_a_1242_ = lean_ctor_get(v___x_1241_, 0);
lean_inc(v_a_1242_);
lean_dec_ref_known(v___x_1241_, 1);
v___x_1243_ = lean_box(0);
v_sz_1244_ = lean_array_size(v_a_1242_);
v___x_1245_ = ((size_t)0ULL);
v___x_1246_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__3(v_a_1242_, v_sz_1244_, v___x_1245_, v___x_1243_, v___y_1235_, v___y_1236_);
lean_dec(v_a_1242_);
if (lean_obj_tag(v___x_1246_) == 0)
{
lean_object* v___x_1247_; size_t v___x_1248_; size_t v___x_1249_; lean_object* v___x_1250_; 
lean_dec_ref_known(v___x_1246_, 1);
v___x_1247_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__4_spec__5_spec__8_spec__11___closed__0));
v___x_1248_ = ((size_t)1ULL);
v___x_1249_ = lean_usize_add(v_i_1233_, v___x_1248_);
v___x_1250_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__4_spec__5_spec__8_spec__11(v_as_1231_, v_sz_1232_, v___x_1249_, v___x_1247_, v___y_1235_, v___y_1236_);
return v___x_1250_;
}
else
{
lean_object* v_a_1251_; lean_object* v___x_1253_; uint8_t v_isShared_1254_; uint8_t v_isSharedCheck_1258_; 
v_a_1251_ = lean_ctor_get(v___x_1246_, 0);
v_isSharedCheck_1258_ = !lean_is_exclusive(v___x_1246_);
if (v_isSharedCheck_1258_ == 0)
{
v___x_1253_ = v___x_1246_;
v_isShared_1254_ = v_isSharedCheck_1258_;
goto v_resetjp_1252_;
}
else
{
lean_inc(v_a_1251_);
lean_dec(v___x_1246_);
v___x_1253_ = lean_box(0);
v_isShared_1254_ = v_isSharedCheck_1258_;
goto v_resetjp_1252_;
}
v_resetjp_1252_:
{
lean_object* v___x_1256_; 
if (v_isShared_1254_ == 0)
{
v___x_1256_ = v___x_1253_;
goto v_reusejp_1255_;
}
else
{
lean_object* v_reuseFailAlloc_1257_; 
v_reuseFailAlloc_1257_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1257_, 0, v_a_1251_);
v___x_1256_ = v_reuseFailAlloc_1257_;
goto v_reusejp_1255_;
}
v_reusejp_1255_:
{
return v___x_1256_;
}
}
}
}
else
{
lean_object* v_a_1259_; lean_object* v___x_1261_; uint8_t v_isShared_1262_; uint8_t v_isSharedCheck_1266_; 
v_a_1259_ = lean_ctor_get(v___x_1241_, 0);
v_isSharedCheck_1266_ = !lean_is_exclusive(v___x_1241_);
if (v_isSharedCheck_1266_ == 0)
{
v___x_1261_ = v___x_1241_;
v_isShared_1262_ = v_isSharedCheck_1266_;
goto v_resetjp_1260_;
}
else
{
lean_inc(v_a_1259_);
lean_dec(v___x_1241_);
v___x_1261_ = lean_box(0);
v_isShared_1262_ = v_isSharedCheck_1266_;
goto v_resetjp_1260_;
}
v_resetjp_1260_:
{
lean_object* v___x_1264_; 
if (v_isShared_1262_ == 0)
{
v___x_1264_ = v___x_1261_;
goto v_reusejp_1263_;
}
else
{
lean_object* v_reuseFailAlloc_1265_; 
v_reuseFailAlloc_1265_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1265_, 0, v_a_1259_);
v___x_1264_ = v_reuseFailAlloc_1265_;
goto v_reusejp_1263_;
}
v_reusejp_1263_:
{
return v___x_1264_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__4_spec__5_spec__8___boxed(lean_object* v_as_1267_, lean_object* v_sz_1268_, lean_object* v_i_1269_, lean_object* v_b_1270_, lean_object* v___y_1271_, lean_object* v___y_1272_, lean_object* v___y_1273_){
_start:
{
size_t v_sz_boxed_1274_; size_t v_i_boxed_1275_; lean_object* v_res_1276_; 
v_sz_boxed_1274_ = lean_unbox_usize(v_sz_1268_);
lean_dec(v_sz_1268_);
v_i_boxed_1275_ = lean_unbox_usize(v_i_1269_);
lean_dec(v_i_1269_);
v_res_1276_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__4_spec__5_spec__8(v_as_1267_, v_sz_boxed_1274_, v_i_boxed_1275_, v_b_1270_, v___y_1271_, v___y_1272_);
lean_dec(v___y_1272_);
lean_dec_ref(v___y_1271_);
lean_dec_ref(v_as_1267_);
return v_res_1276_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__4_spec__5(lean_object* v_init_1277_, lean_object* v_n_1278_, lean_object* v_b_1279_, lean_object* v___y_1280_, lean_object* v___y_1281_){
_start:
{
if (lean_obj_tag(v_n_1278_) == 0)
{
lean_object* v_cs_1283_; lean_object* v___x_1284_; lean_object* v___x_1285_; size_t v_sz_1286_; size_t v___x_1287_; lean_object* v___x_1288_; 
v_cs_1283_ = lean_ctor_get(v_n_1278_, 0);
v___x_1284_ = lean_box(0);
v___x_1285_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1285_, 0, v___x_1284_);
lean_ctor_set(v___x_1285_, 1, v_b_1279_);
v_sz_1286_ = lean_array_size(v_cs_1283_);
v___x_1287_ = ((size_t)0ULL);
v___x_1288_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__4_spec__5_spec__7(v_init_1277_, v_cs_1283_, v_sz_1286_, v___x_1287_, v___x_1285_, v___y_1280_, v___y_1281_);
if (lean_obj_tag(v___x_1288_) == 0)
{
lean_object* v_a_1289_; lean_object* v___x_1291_; uint8_t v_isShared_1292_; uint8_t v_isSharedCheck_1303_; 
v_a_1289_ = lean_ctor_get(v___x_1288_, 0);
v_isSharedCheck_1303_ = !lean_is_exclusive(v___x_1288_);
if (v_isSharedCheck_1303_ == 0)
{
v___x_1291_ = v___x_1288_;
v_isShared_1292_ = v_isSharedCheck_1303_;
goto v_resetjp_1290_;
}
else
{
lean_inc(v_a_1289_);
lean_dec(v___x_1288_);
v___x_1291_ = lean_box(0);
v_isShared_1292_ = v_isSharedCheck_1303_;
goto v_resetjp_1290_;
}
v_resetjp_1290_:
{
lean_object* v_fst_1293_; 
v_fst_1293_ = lean_ctor_get(v_a_1289_, 0);
if (lean_obj_tag(v_fst_1293_) == 0)
{
lean_object* v_snd_1294_; lean_object* v___x_1295_; lean_object* v___x_1297_; 
v_snd_1294_ = lean_ctor_get(v_a_1289_, 1);
lean_inc(v_snd_1294_);
lean_dec(v_a_1289_);
v___x_1295_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1295_, 0, v_snd_1294_);
if (v_isShared_1292_ == 0)
{
lean_ctor_set(v___x_1291_, 0, v___x_1295_);
v___x_1297_ = v___x_1291_;
goto v_reusejp_1296_;
}
else
{
lean_object* v_reuseFailAlloc_1298_; 
v_reuseFailAlloc_1298_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1298_, 0, v___x_1295_);
v___x_1297_ = v_reuseFailAlloc_1298_;
goto v_reusejp_1296_;
}
v_reusejp_1296_:
{
return v___x_1297_;
}
}
else
{
lean_object* v_val_1299_; lean_object* v___x_1301_; 
lean_inc_ref(v_fst_1293_);
lean_dec(v_a_1289_);
v_val_1299_ = lean_ctor_get(v_fst_1293_, 0);
lean_inc(v_val_1299_);
lean_dec_ref_known(v_fst_1293_, 1);
if (v_isShared_1292_ == 0)
{
lean_ctor_set(v___x_1291_, 0, v_val_1299_);
v___x_1301_ = v___x_1291_;
goto v_reusejp_1300_;
}
else
{
lean_object* v_reuseFailAlloc_1302_; 
v_reuseFailAlloc_1302_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1302_, 0, v_val_1299_);
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
else
{
lean_object* v_a_1304_; lean_object* v___x_1306_; uint8_t v_isShared_1307_; uint8_t v_isSharedCheck_1311_; 
v_a_1304_ = lean_ctor_get(v___x_1288_, 0);
v_isSharedCheck_1311_ = !lean_is_exclusive(v___x_1288_);
if (v_isSharedCheck_1311_ == 0)
{
v___x_1306_ = v___x_1288_;
v_isShared_1307_ = v_isSharedCheck_1311_;
goto v_resetjp_1305_;
}
else
{
lean_inc(v_a_1304_);
lean_dec(v___x_1288_);
v___x_1306_ = lean_box(0);
v_isShared_1307_ = v_isSharedCheck_1311_;
goto v_resetjp_1305_;
}
v_resetjp_1305_:
{
lean_object* v___x_1309_; 
if (v_isShared_1307_ == 0)
{
v___x_1309_ = v___x_1306_;
goto v_reusejp_1308_;
}
else
{
lean_object* v_reuseFailAlloc_1310_; 
v_reuseFailAlloc_1310_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1310_, 0, v_a_1304_);
v___x_1309_ = v_reuseFailAlloc_1310_;
goto v_reusejp_1308_;
}
v_reusejp_1308_:
{
return v___x_1309_;
}
}
}
}
else
{
lean_object* v_vs_1312_; lean_object* v___x_1313_; lean_object* v___x_1314_; size_t v_sz_1315_; size_t v___x_1316_; lean_object* v___x_1317_; 
v_vs_1312_ = lean_ctor_get(v_n_1278_, 0);
v___x_1313_ = lean_box(0);
v___x_1314_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1314_, 0, v___x_1313_);
lean_ctor_set(v___x_1314_, 1, v_b_1279_);
v_sz_1315_ = lean_array_size(v_vs_1312_);
v___x_1316_ = ((size_t)0ULL);
v___x_1317_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__4_spec__5_spec__8(v_vs_1312_, v_sz_1315_, v___x_1316_, v___x_1314_, v___y_1280_, v___y_1281_);
if (lean_obj_tag(v___x_1317_) == 0)
{
lean_object* v_a_1318_; lean_object* v___x_1320_; uint8_t v_isShared_1321_; uint8_t v_isSharedCheck_1332_; 
v_a_1318_ = lean_ctor_get(v___x_1317_, 0);
v_isSharedCheck_1332_ = !lean_is_exclusive(v___x_1317_);
if (v_isSharedCheck_1332_ == 0)
{
v___x_1320_ = v___x_1317_;
v_isShared_1321_ = v_isSharedCheck_1332_;
goto v_resetjp_1319_;
}
else
{
lean_inc(v_a_1318_);
lean_dec(v___x_1317_);
v___x_1320_ = lean_box(0);
v_isShared_1321_ = v_isSharedCheck_1332_;
goto v_resetjp_1319_;
}
v_resetjp_1319_:
{
lean_object* v_fst_1322_; 
v_fst_1322_ = lean_ctor_get(v_a_1318_, 0);
if (lean_obj_tag(v_fst_1322_) == 0)
{
lean_object* v_snd_1323_; lean_object* v___x_1324_; lean_object* v___x_1326_; 
v_snd_1323_ = lean_ctor_get(v_a_1318_, 1);
lean_inc(v_snd_1323_);
lean_dec(v_a_1318_);
v___x_1324_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1324_, 0, v_snd_1323_);
if (v_isShared_1321_ == 0)
{
lean_ctor_set(v___x_1320_, 0, v___x_1324_);
v___x_1326_ = v___x_1320_;
goto v_reusejp_1325_;
}
else
{
lean_object* v_reuseFailAlloc_1327_; 
v_reuseFailAlloc_1327_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1327_, 0, v___x_1324_);
v___x_1326_ = v_reuseFailAlloc_1327_;
goto v_reusejp_1325_;
}
v_reusejp_1325_:
{
return v___x_1326_;
}
}
else
{
lean_object* v_val_1328_; lean_object* v___x_1330_; 
lean_inc_ref(v_fst_1322_);
lean_dec(v_a_1318_);
v_val_1328_ = lean_ctor_get(v_fst_1322_, 0);
lean_inc(v_val_1328_);
lean_dec_ref_known(v_fst_1322_, 1);
if (v_isShared_1321_ == 0)
{
lean_ctor_set(v___x_1320_, 0, v_val_1328_);
v___x_1330_ = v___x_1320_;
goto v_reusejp_1329_;
}
else
{
lean_object* v_reuseFailAlloc_1331_; 
v_reuseFailAlloc_1331_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1331_, 0, v_val_1328_);
v___x_1330_ = v_reuseFailAlloc_1331_;
goto v_reusejp_1329_;
}
v_reusejp_1329_:
{
return v___x_1330_;
}
}
}
}
else
{
lean_object* v_a_1333_; lean_object* v___x_1335_; uint8_t v_isShared_1336_; uint8_t v_isSharedCheck_1340_; 
v_a_1333_ = lean_ctor_get(v___x_1317_, 0);
v_isSharedCheck_1340_ = !lean_is_exclusive(v___x_1317_);
if (v_isSharedCheck_1340_ == 0)
{
v___x_1335_ = v___x_1317_;
v_isShared_1336_ = v_isSharedCheck_1340_;
goto v_resetjp_1334_;
}
else
{
lean_inc(v_a_1333_);
lean_dec(v___x_1317_);
v___x_1335_ = lean_box(0);
v_isShared_1336_ = v_isSharedCheck_1340_;
goto v_resetjp_1334_;
}
v_resetjp_1334_:
{
lean_object* v___x_1338_; 
if (v_isShared_1336_ == 0)
{
v___x_1338_ = v___x_1335_;
goto v_reusejp_1337_;
}
else
{
lean_object* v_reuseFailAlloc_1339_; 
v_reuseFailAlloc_1339_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1339_, 0, v_a_1333_);
v___x_1338_ = v_reuseFailAlloc_1339_;
goto v_reusejp_1337_;
}
v_reusejp_1337_:
{
return v___x_1338_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__4_spec__5_spec__7(lean_object* v_init_1341_, lean_object* v_as_1342_, size_t v_sz_1343_, size_t v_i_1344_, lean_object* v_b_1345_, lean_object* v___y_1346_, lean_object* v___y_1347_){
_start:
{
uint8_t v___x_1349_; 
v___x_1349_ = lean_usize_dec_lt(v_i_1344_, v_sz_1343_);
if (v___x_1349_ == 0)
{
lean_object* v___x_1350_; 
v___x_1350_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1350_, 0, v_b_1345_);
return v___x_1350_;
}
else
{
lean_object* v_snd_1351_; lean_object* v___x_1353_; uint8_t v_isShared_1354_; uint8_t v_isSharedCheck_1385_; 
v_snd_1351_ = lean_ctor_get(v_b_1345_, 1);
v_isSharedCheck_1385_ = !lean_is_exclusive(v_b_1345_);
if (v_isSharedCheck_1385_ == 0)
{
lean_object* v_unused_1386_; 
v_unused_1386_ = lean_ctor_get(v_b_1345_, 0);
lean_dec(v_unused_1386_);
v___x_1353_ = v_b_1345_;
v_isShared_1354_ = v_isSharedCheck_1385_;
goto v_resetjp_1352_;
}
else
{
lean_inc(v_snd_1351_);
lean_dec(v_b_1345_);
v___x_1353_ = lean_box(0);
v_isShared_1354_ = v_isSharedCheck_1385_;
goto v_resetjp_1352_;
}
v_resetjp_1352_:
{
lean_object* v_a_1355_; lean_object* v___x_1356_; 
v_a_1355_ = lean_array_uget_borrowed(v_as_1342_, v_i_1344_);
lean_inc(v_snd_1351_);
v___x_1356_ = lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__4_spec__5(v_init_1341_, v_a_1355_, v_snd_1351_, v___y_1346_, v___y_1347_);
if (lean_obj_tag(v___x_1356_) == 0)
{
lean_object* v_a_1357_; lean_object* v___x_1359_; uint8_t v_isShared_1360_; uint8_t v_isSharedCheck_1376_; 
v_a_1357_ = lean_ctor_get(v___x_1356_, 0);
v_isSharedCheck_1376_ = !lean_is_exclusive(v___x_1356_);
if (v_isSharedCheck_1376_ == 0)
{
v___x_1359_ = v___x_1356_;
v_isShared_1360_ = v_isSharedCheck_1376_;
goto v_resetjp_1358_;
}
else
{
lean_inc(v_a_1357_);
lean_dec(v___x_1356_);
v___x_1359_ = lean_box(0);
v_isShared_1360_ = v_isSharedCheck_1376_;
goto v_resetjp_1358_;
}
v_resetjp_1358_:
{
if (lean_obj_tag(v_a_1357_) == 0)
{
lean_object* v___x_1361_; lean_object* v___x_1363_; 
v___x_1361_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1361_, 0, v_a_1357_);
if (v_isShared_1354_ == 0)
{
lean_ctor_set(v___x_1353_, 0, v___x_1361_);
v___x_1363_ = v___x_1353_;
goto v_reusejp_1362_;
}
else
{
lean_object* v_reuseFailAlloc_1367_; 
v_reuseFailAlloc_1367_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1367_, 0, v___x_1361_);
lean_ctor_set(v_reuseFailAlloc_1367_, 1, v_snd_1351_);
v___x_1363_ = v_reuseFailAlloc_1367_;
goto v_reusejp_1362_;
}
v_reusejp_1362_:
{
lean_object* v___x_1365_; 
if (v_isShared_1360_ == 0)
{
lean_ctor_set(v___x_1359_, 0, v___x_1363_);
v___x_1365_ = v___x_1359_;
goto v_reusejp_1364_;
}
else
{
lean_object* v_reuseFailAlloc_1366_; 
v_reuseFailAlloc_1366_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1366_, 0, v___x_1363_);
v___x_1365_ = v_reuseFailAlloc_1366_;
goto v_reusejp_1364_;
}
v_reusejp_1364_:
{
return v___x_1365_;
}
}
}
else
{
lean_object* v_a_1368_; lean_object* v___x_1369_; lean_object* v___x_1371_; 
lean_del_object(v___x_1359_);
lean_dec(v_snd_1351_);
v_a_1368_ = lean_ctor_get(v_a_1357_, 0);
lean_inc(v_a_1368_);
lean_dec_ref_known(v_a_1357_, 1);
v___x_1369_ = lean_box(0);
if (v_isShared_1354_ == 0)
{
lean_ctor_set(v___x_1353_, 1, v_a_1368_);
lean_ctor_set(v___x_1353_, 0, v___x_1369_);
v___x_1371_ = v___x_1353_;
goto v_reusejp_1370_;
}
else
{
lean_object* v_reuseFailAlloc_1375_; 
v_reuseFailAlloc_1375_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1375_, 0, v___x_1369_);
lean_ctor_set(v_reuseFailAlloc_1375_, 1, v_a_1368_);
v___x_1371_ = v_reuseFailAlloc_1375_;
goto v_reusejp_1370_;
}
v_reusejp_1370_:
{
size_t v___x_1372_; size_t v___x_1373_; 
v___x_1372_ = ((size_t)1ULL);
v___x_1373_ = lean_usize_add(v_i_1344_, v___x_1372_);
v_i_1344_ = v___x_1373_;
v_b_1345_ = v___x_1371_;
goto _start;
}
}
}
}
else
{
lean_object* v_a_1377_; lean_object* v___x_1379_; uint8_t v_isShared_1380_; uint8_t v_isSharedCheck_1384_; 
lean_del_object(v___x_1353_);
lean_dec(v_snd_1351_);
v_a_1377_ = lean_ctor_get(v___x_1356_, 0);
v_isSharedCheck_1384_ = !lean_is_exclusive(v___x_1356_);
if (v_isSharedCheck_1384_ == 0)
{
v___x_1379_ = v___x_1356_;
v_isShared_1380_ = v_isSharedCheck_1384_;
goto v_resetjp_1378_;
}
else
{
lean_inc(v_a_1377_);
lean_dec(v___x_1356_);
v___x_1379_ = lean_box(0);
v_isShared_1380_ = v_isSharedCheck_1384_;
goto v_resetjp_1378_;
}
v_resetjp_1378_:
{
lean_object* v___x_1382_; 
if (v_isShared_1380_ == 0)
{
v___x_1382_ = v___x_1379_;
goto v_reusejp_1381_;
}
else
{
lean_object* v_reuseFailAlloc_1383_; 
v_reuseFailAlloc_1383_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1383_, 0, v_a_1377_);
v___x_1382_ = v_reuseFailAlloc_1383_;
goto v_reusejp_1381_;
}
v_reusejp_1381_:
{
return v___x_1382_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__4_spec__5_spec__7___boxed(lean_object* v_init_1387_, lean_object* v_as_1388_, lean_object* v_sz_1389_, lean_object* v_i_1390_, lean_object* v_b_1391_, lean_object* v___y_1392_, lean_object* v___y_1393_, lean_object* v___y_1394_){
_start:
{
size_t v_sz_boxed_1395_; size_t v_i_boxed_1396_; lean_object* v_res_1397_; 
v_sz_boxed_1395_ = lean_unbox_usize(v_sz_1389_);
lean_dec(v_sz_1389_);
v_i_boxed_1396_ = lean_unbox_usize(v_i_1390_);
lean_dec(v_i_1390_);
v_res_1397_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__4_spec__5_spec__7(v_init_1387_, v_as_1388_, v_sz_boxed_1395_, v_i_boxed_1396_, v_b_1391_, v___y_1392_, v___y_1393_);
lean_dec(v___y_1393_);
lean_dec_ref(v___y_1392_);
lean_dec_ref(v_as_1388_);
return v_res_1397_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__4_spec__5___boxed(lean_object* v_init_1398_, lean_object* v_n_1399_, lean_object* v_b_1400_, lean_object* v___y_1401_, lean_object* v___y_1402_, lean_object* v___y_1403_){
_start:
{
lean_object* v_res_1404_; 
v_res_1404_ = lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__4_spec__5(v_init_1398_, v_n_1399_, v_b_1400_, v___y_1401_, v___y_1402_);
lean_dec(v___y_1402_);
lean_dec_ref(v___y_1401_);
lean_dec_ref(v_n_1399_);
return v_res_1404_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__4(lean_object* v_t_1405_, lean_object* v_init_1406_, lean_object* v___y_1407_, lean_object* v___y_1408_){
_start:
{
lean_object* v_root_1410_; lean_object* v_tail_1411_; lean_object* v___x_1412_; 
v_root_1410_ = lean_ctor_get(v_t_1405_, 0);
v_tail_1411_ = lean_ctor_get(v_t_1405_, 1);
v___x_1412_ = lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__4_spec__5(v_init_1406_, v_root_1410_, v_init_1406_, v___y_1407_, v___y_1408_);
if (lean_obj_tag(v___x_1412_) == 0)
{
lean_object* v_a_1413_; lean_object* v___x_1415_; uint8_t v_isShared_1416_; uint8_t v_isSharedCheck_1449_; 
v_a_1413_ = lean_ctor_get(v___x_1412_, 0);
v_isSharedCheck_1449_ = !lean_is_exclusive(v___x_1412_);
if (v_isSharedCheck_1449_ == 0)
{
v___x_1415_ = v___x_1412_;
v_isShared_1416_ = v_isSharedCheck_1449_;
goto v_resetjp_1414_;
}
else
{
lean_inc(v_a_1413_);
lean_dec(v___x_1412_);
v___x_1415_ = lean_box(0);
v_isShared_1416_ = v_isSharedCheck_1449_;
goto v_resetjp_1414_;
}
v_resetjp_1414_:
{
if (lean_obj_tag(v_a_1413_) == 0)
{
lean_object* v_a_1417_; lean_object* v___x_1419_; 
v_a_1417_ = lean_ctor_get(v_a_1413_, 0);
lean_inc(v_a_1417_);
lean_dec_ref_known(v_a_1413_, 1);
if (v_isShared_1416_ == 0)
{
lean_ctor_set(v___x_1415_, 0, v_a_1417_);
v___x_1419_ = v___x_1415_;
goto v_reusejp_1418_;
}
else
{
lean_object* v_reuseFailAlloc_1420_; 
v_reuseFailAlloc_1420_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1420_, 0, v_a_1417_);
v___x_1419_ = v_reuseFailAlloc_1420_;
goto v_reusejp_1418_;
}
v_reusejp_1418_:
{
return v___x_1419_;
}
}
else
{
lean_object* v_a_1421_; lean_object* v___x_1422_; lean_object* v___x_1423_; size_t v_sz_1424_; size_t v___x_1425_; lean_object* v___x_1426_; 
lean_del_object(v___x_1415_);
v_a_1421_ = lean_ctor_get(v_a_1413_, 0);
lean_inc(v_a_1421_);
lean_dec_ref_known(v_a_1413_, 1);
v___x_1422_ = lean_box(0);
v___x_1423_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1423_, 0, v___x_1422_);
lean_ctor_set(v___x_1423_, 1, v_a_1421_);
v_sz_1424_ = lean_array_size(v_tail_1411_);
v___x_1425_ = ((size_t)0ULL);
v___x_1426_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__4_spec__6(v_tail_1411_, v_sz_1424_, v___x_1425_, v___x_1423_, v___y_1407_, v___y_1408_);
if (lean_obj_tag(v___x_1426_) == 0)
{
lean_object* v_a_1427_; lean_object* v___x_1429_; uint8_t v_isShared_1430_; uint8_t v_isSharedCheck_1440_; 
v_a_1427_ = lean_ctor_get(v___x_1426_, 0);
v_isSharedCheck_1440_ = !lean_is_exclusive(v___x_1426_);
if (v_isSharedCheck_1440_ == 0)
{
v___x_1429_ = v___x_1426_;
v_isShared_1430_ = v_isSharedCheck_1440_;
goto v_resetjp_1428_;
}
else
{
lean_inc(v_a_1427_);
lean_dec(v___x_1426_);
v___x_1429_ = lean_box(0);
v_isShared_1430_ = v_isSharedCheck_1440_;
goto v_resetjp_1428_;
}
v_resetjp_1428_:
{
lean_object* v_fst_1431_; 
v_fst_1431_ = lean_ctor_get(v_a_1427_, 0);
if (lean_obj_tag(v_fst_1431_) == 0)
{
lean_object* v_snd_1432_; lean_object* v___x_1434_; 
v_snd_1432_ = lean_ctor_get(v_a_1427_, 1);
lean_inc(v_snd_1432_);
lean_dec(v_a_1427_);
if (v_isShared_1430_ == 0)
{
lean_ctor_set(v___x_1429_, 0, v_snd_1432_);
v___x_1434_ = v___x_1429_;
goto v_reusejp_1433_;
}
else
{
lean_object* v_reuseFailAlloc_1435_; 
v_reuseFailAlloc_1435_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1435_, 0, v_snd_1432_);
v___x_1434_ = v_reuseFailAlloc_1435_;
goto v_reusejp_1433_;
}
v_reusejp_1433_:
{
return v___x_1434_;
}
}
else
{
lean_object* v_val_1436_; lean_object* v___x_1438_; 
lean_inc_ref(v_fst_1431_);
lean_dec(v_a_1427_);
v_val_1436_ = lean_ctor_get(v_fst_1431_, 0);
lean_inc(v_val_1436_);
lean_dec_ref_known(v_fst_1431_, 1);
if (v_isShared_1430_ == 0)
{
lean_ctor_set(v___x_1429_, 0, v_val_1436_);
v___x_1438_ = v___x_1429_;
goto v_reusejp_1437_;
}
else
{
lean_object* v_reuseFailAlloc_1439_; 
v_reuseFailAlloc_1439_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1439_, 0, v_val_1436_);
v___x_1438_ = v_reuseFailAlloc_1439_;
goto v_reusejp_1437_;
}
v_reusejp_1437_:
{
return v___x_1438_;
}
}
}
}
else
{
lean_object* v_a_1441_; lean_object* v___x_1443_; uint8_t v_isShared_1444_; uint8_t v_isSharedCheck_1448_; 
v_a_1441_ = lean_ctor_get(v___x_1426_, 0);
v_isSharedCheck_1448_ = !lean_is_exclusive(v___x_1426_);
if (v_isSharedCheck_1448_ == 0)
{
v___x_1443_ = v___x_1426_;
v_isShared_1444_ = v_isSharedCheck_1448_;
goto v_resetjp_1442_;
}
else
{
lean_inc(v_a_1441_);
lean_dec(v___x_1426_);
v___x_1443_ = lean_box(0);
v_isShared_1444_ = v_isSharedCheck_1448_;
goto v_resetjp_1442_;
}
v_resetjp_1442_:
{
lean_object* v___x_1446_; 
if (v_isShared_1444_ == 0)
{
v___x_1446_ = v___x_1443_;
goto v_reusejp_1445_;
}
else
{
lean_object* v_reuseFailAlloc_1447_; 
v_reuseFailAlloc_1447_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1447_, 0, v_a_1441_);
v___x_1446_ = v_reuseFailAlloc_1447_;
goto v_reusejp_1445_;
}
v_reusejp_1445_:
{
return v___x_1446_;
}
}
}
}
}
}
else
{
lean_object* v_a_1450_; lean_object* v___x_1452_; uint8_t v_isShared_1453_; uint8_t v_isSharedCheck_1457_; 
v_a_1450_ = lean_ctor_get(v___x_1412_, 0);
v_isSharedCheck_1457_ = !lean_is_exclusive(v___x_1412_);
if (v_isSharedCheck_1457_ == 0)
{
v___x_1452_ = v___x_1412_;
v_isShared_1453_ = v_isSharedCheck_1457_;
goto v_resetjp_1451_;
}
else
{
lean_inc(v_a_1450_);
lean_dec(v___x_1412_);
v___x_1452_ = lean_box(0);
v_isShared_1453_ = v_isSharedCheck_1457_;
goto v_resetjp_1451_;
}
v_resetjp_1451_:
{
lean_object* v___x_1455_; 
if (v_isShared_1453_ == 0)
{
v___x_1455_ = v___x_1452_;
goto v_reusejp_1454_;
}
else
{
lean_object* v_reuseFailAlloc_1456_; 
v_reuseFailAlloc_1456_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1456_, 0, v_a_1450_);
v___x_1455_ = v_reuseFailAlloc_1456_;
goto v_reusejp_1454_;
}
v_reusejp_1454_:
{
return v___x_1455_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__4___boxed(lean_object* v_t_1458_, lean_object* v_init_1459_, lean_object* v___y_1460_, lean_object* v___y_1461_, lean_object* v___y_1462_){
_start:
{
lean_object* v_res_1463_; 
v_res_1463_ = lp_mathlib_Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__4(v_t_1458_, v_init_1459_, v___y_1460_, v___y_1461_);
lean_dec(v___y_1461_);
lean_dec_ref(v___y_1460_);
lean_dec_ref(v_t_1458_);
return v_res_1463_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___lam__0(lean_object* v___stx_1464_, lean_object* v___y_1465_, lean_object* v___y_1466_){
_start:
{
lean_object* v___x_1468_; lean_object* v___x_1469_; lean_object* v_scopes_1473_; lean_object* v___x_1474_; lean_object* v___x_1475_; lean_object* v_opts_1476_; lean_object* v_infoState_1477_; lean_object* v___x_1478_; lean_object* v___x_1479_; lean_object* v___x_1480_; uint8_t v___x_1481_; 
v___x_1468_ = lean_st_ref_get(v___y_1466_);
v___x_1469_ = lean_st_ref_get(v___y_1466_);
v_scopes_1473_ = lean_ctor_get(v___x_1468_, 2);
lean_inc(v_scopes_1473_);
lean_dec(v___x_1468_);
v___x_1474_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_1475_ = l_List_head_x21___redArg(v___x_1474_, v_scopes_1473_);
lean_dec(v_scopes_1473_);
v_opts_1476_ = lean_ctor_get(v___x_1475_, 1);
lean_inc_ref(v_opts_1476_);
lean_dec(v___x_1475_);
v_infoState_1477_ = lean_ctor_get(v___x_1469_, 8);
lean_inc_ref(v_infoState_1477_);
lean_dec(v___x_1469_);
v___x_1478_ = lp_mathlib_Mathlib_Linter_linter_haveLet;
v___x_1479_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__0(v_opts_1476_, v___x_1478_);
lean_dec_ref(v_opts_1476_);
v___x_1480_ = lean_unsigned_to_nat(0u);
v___x_1481_ = lean_nat_dec_eq(v___x_1479_, v___x_1480_);
if (v___x_1481_ == 0)
{
uint8_t v_enabled_1482_; 
v_enabled_1482_ = lean_ctor_get_uint8(v_infoState_1477_, sizeof(void*)*3);
lean_dec_ref(v_infoState_1477_);
if (v_enabled_1482_ == 0)
{
lean_dec(v___x_1479_);
goto v___jp_1470_;
}
else
{
lean_object* v___x_1483_; uint8_t v___y_1485_; lean_object* v___x_1500_; uint8_t v___x_1501_; 
v___x_1483_ = lean_st_ref_get(v___y_1466_);
v___x_1500_ = lean_unsigned_to_nat(1u);
v___x_1501_ = lean_nat_dec_eq(v___x_1479_, v___x_1500_);
lean_dec(v___x_1479_);
if (v___x_1501_ == 0)
{
lean_dec(v___x_1483_);
v___y_1485_ = v___x_1501_;
goto v___jp_1484_;
}
else
{
lean_object* v_messages_1502_; lean_object* v_unreported_1503_; uint8_t v___x_1504_; 
v_messages_1502_ = lean_ctor_get(v___x_1483_, 1);
lean_inc_ref(v_messages_1502_);
lean_dec(v___x_1483_);
v_unreported_1503_ = lean_ctor_get(v_messages_1502_, 1);
lean_inc_ref(v_unreported_1503_);
lean_dec_ref(v_messages_1502_);
v___x_1504_ = l_Lean_PersistentArray_isEmpty___redArg(v_unreported_1503_);
lean_dec_ref(v_unreported_1503_);
v___y_1485_ = v___x_1504_;
goto v___jp_1484_;
}
v___jp_1484_:
{
if (v___y_1485_ == 0)
{
lean_object* v___x_1486_; lean_object* v_a_1487_; lean_object* v___x_1488_; lean_object* v___x_1489_; 
v___x_1486_ = lp_mathlib_Lean_Elab_getInfoTrees___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__1___redArg(v___y_1466_);
v_a_1487_ = lean_ctor_get(v___x_1486_, 0);
lean_inc(v_a_1487_);
lean_dec_ref(v___x_1486_);
v___x_1488_ = lean_box(0);
v___x_1489_ = lp_mathlib_Lean_PersistentArray_forIn___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__4(v_a_1487_, v___x_1488_, v___y_1465_, v___y_1466_);
lean_dec(v_a_1487_);
if (lean_obj_tag(v___x_1489_) == 0)
{
lean_object* v___x_1491_; uint8_t v_isShared_1492_; uint8_t v_isSharedCheck_1496_; 
v_isSharedCheck_1496_ = !lean_is_exclusive(v___x_1489_);
if (v_isSharedCheck_1496_ == 0)
{
lean_object* v_unused_1497_; 
v_unused_1497_ = lean_ctor_get(v___x_1489_, 0);
lean_dec(v_unused_1497_);
v___x_1491_ = v___x_1489_;
v_isShared_1492_ = v_isSharedCheck_1496_;
goto v_resetjp_1490_;
}
else
{
lean_dec(v___x_1489_);
v___x_1491_ = lean_box(0);
v_isShared_1492_ = v_isSharedCheck_1496_;
goto v_resetjp_1490_;
}
v_resetjp_1490_:
{
lean_object* v___x_1494_; 
if (v_isShared_1492_ == 0)
{
lean_ctor_set(v___x_1491_, 0, v___x_1488_);
v___x_1494_ = v___x_1491_;
goto v_reusejp_1493_;
}
else
{
lean_object* v_reuseFailAlloc_1495_; 
v_reuseFailAlloc_1495_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1495_, 0, v___x_1488_);
v___x_1494_ = v_reuseFailAlloc_1495_;
goto v_reusejp_1493_;
}
v_reusejp_1493_:
{
return v___x_1494_;
}
}
}
else
{
return v___x_1489_;
}
}
else
{
lean_object* v___x_1498_; lean_object* v___x_1499_; 
v___x_1498_ = lean_box(0);
v___x_1499_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1499_, 0, v___x_1498_);
return v___x_1499_;
}
}
}
}
else
{
lean_dec(v___x_1479_);
lean_dec_ref(v_infoState_1477_);
goto v___jp_1470_;
}
v___jp_1470_:
{
lean_object* v___x_1471_; lean_object* v___x_1472_; 
v___x_1471_ = lean_box(0);
v___x_1472_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1472_, 0, v___x_1471_);
return v___x_1472_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___lam__0___boxed(lean_object* v___stx_1505_, lean_object* v___y_1506_, lean_object* v___y_1507_, lean_object* v___y_1508_){
_start:
{
lean_object* v_res_1509_; 
v_res_1509_ = lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter___lam__0(v___stx_1505_, v___y_1506_, v___y_1507_);
lean_dec(v___y_1507_);
lean_dec_ref(v___y_1506_);
lean_dec(v___stx_1505_);
return v_res_1509_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3_spec__6(lean_object* v_msgData_1550_, lean_object* v___y_1551_, lean_object* v___y_1552_){
_start:
{
lean_object* v___x_1554_; 
v___x_1554_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3_spec__6___redArg(v_msgData_1550_, v___y_1552_);
return v___x_1554_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3_spec__6___boxed(lean_object* v_msgData_1555_, lean_object* v___y_1556_, lean_object* v___y_1557_, lean_object* v___y_1558_){
_start:
{
lean_object* v_res_1559_; 
v_res_1559_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Linter_logLint0Disable___at___00__private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter_spec__2_spec__2_spec__3_spec__6(v_msgData_1555_, v___y_1556_, v___y_1557_);
lean_dec(v___y_1557_);
lean_dec_ref(v___y_1556_);
return v_res_1559_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_initFn_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_3906617390____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_1561_; lean_object* v___x_1562_; 
v___x_1561_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_haveLetLinter));
v___x_1562_ = l_Lean_Elab_Command_addLinter(v___x_1561_);
return v___x_1562_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_initFn_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_3906617390____hygCtx___hyg_2____boxed(lean_object* v_a_1563_){
_start:
{
lean_object* v_res_1564_; 
v_res_1564_ = lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_initFn_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_3906617390____hygCtx___hyg_2_();
return v_res_1564_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_HaveLetLinter(uint8_t builtin) {
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
lean_object* runtime_initialize_Lean_Elab_Command(uint8_t builtin);
lean_object* runtime_initialize_Lean_Server_InfoUtils(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Linter_HaveLetLinter(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Command(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Server_InfoUtils(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_4193727637____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Linter_linter_haveLet = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Linter_linter_haveLet);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linter_HaveLetLinter_0__Mathlib_Linter_haveLet_initFn_00___x40_Mathlib_Tactic_Linter_HaveLetLinter_3906617390____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Command(uint8_t builtin);
lean_object* initialize_Lean_Server_InfoUtils(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Linter_HaveLetLinter(uint8_t builtin) {
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
res = initialize_Lean_Elab_Command(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Server_InfoUtils(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_HaveLetLinter(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Linter_HaveLetLinter(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Linter_HaveLetLinter(builtin);
}
#ifdef __cplusplus
}
#endif
