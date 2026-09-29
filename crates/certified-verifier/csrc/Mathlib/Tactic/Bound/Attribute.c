// Lean compiler output
// Module: Mathlib.Tactic.Bound.Attribute
// Imports: public import Init public meta import Init public import Aesop public import Mathlib.Tactic.Bound.Init public import Qq
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
extern lean_object* lp_aesop_Aesop_instInhabitedBaseRuleSet_default;
lean_object* l_Lean_ScopedEnvExtension_getState___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Meta_instInhabitedSimpTheorems_default;
extern lean_object* l_Lean_Meta_Simp_instInhabitedSimprocs_default;
lean_object* l_Lean_Name_mkStr1(lean_object*);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_usize_to_nat(size_t);
lean_object* lean_array_set(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* l_Lean_Meta_Origin_key(lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
lean_object* l_Lean_PersistentHashMap_isUnaryNode___redArg(lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* l_Array_eraseIdx___redArg(lean_object*, lean_object*);
uint64_t lean_uint64_mix_hash(uint64_t, uint64_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* l_Lean_Expr_lit___override(lean_object*);
lean_object* l_Lean_Meta_mkFreshExprMVar(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_sub(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lp_aesop_Aesop_getDeclaredRuleSets();
lean_object* l_Lean_Meta_getSimpExtension_x3f(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_Simp_getSimprocExtension_x3f(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_io_error_to_string(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_ScopedEnvExtension_modifyState___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkFreshLevelMVar(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Level_succ___override(lean_object*);
lean_object* l_Lean_Expr_sort___override(lean_object*);
lean_object* lean_instantiate_level_mvars(lean_object*, lean_object*);
lean_object* l_Lean_Name_toString(lean_object*, uint8_t);
lean_object* lean_expr_dbg_to_string(lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingAuxAux(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Environment_find_x3f(lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_ConstantInfo_type(lean_object*);
extern lean_object* lp_aesop_Aesop_RuleSetNameFilter_all;
lean_object* lp_aesop_Aesop_RuleSetNameFilter_matchedRuleSetNames(lean_object*);
lean_object* lp_aesop_Aesop_GlobalRuleSet_erase(lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(lean_object*, uint8_t);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_Lean_MessageData_ofList(lean_object*);
uint64_t l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_TermElabM_run___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkIdent(lean_object*);
lean_object* lean_nat_to_int(lean_object*);
extern lean_object* lp_aesop_Aesop_RuleBuilderOptions_default;
lean_object* lean_array_push(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_Frontend_RuleConfig_buildGlobalRule___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_ElabM_run___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_ScopedEnvExtension_addCore___redArg(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*);
lean_object* lp_aesop_Aesop_GlobalRuleSetMember_name(lean_object*);
uint8_t lp_aesop_Aesop_GlobalRuleSet_contains(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
uint8_t l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(lean_object*, lean_object*, lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
double lean_float_of_nat(lean_object*);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
lean_object* l_Lean_registerBuiltinAttribute(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_registerTraceClass(lean_object*, uint8_t, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__0_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "bound"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__0_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__0_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__1_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "attribute"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__1_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__1_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__2_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__0_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(67, 51, 108, 71, 20, 161, 164, 139)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__2_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__2_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__1_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(62, 173, 194, 242, 84, 155, 79, 147)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__2_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__2_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__3_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__3_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__3_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__4_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__3_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__4_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__4_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__5_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__5_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__5_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__6_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__4_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__5_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__6_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__6_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__7_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__7_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__7_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__8_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__6_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__7_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(191, 70, 156, 159, 11, 54, 216, 94)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__8_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__8_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__9_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Bound"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__9_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__9_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__10_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__8_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__9_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(76, 119, 122, 238, 92, 147, 196, 138)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__10_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__10_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__11_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Attribute"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__11_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__11_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__12_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__10_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__11_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(79, 145, 108, 163, 255, 240, 169, 20)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__12_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__12_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__13_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__12_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(226, 192, 63, 97, 164, 24, 3, 249)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__13_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__13_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__14_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__13_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__5_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(83, 128, 207, 225, 48, 51, 39, 171)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__14_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__14_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__15_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__14_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__7_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(66, 8, 234, 225, 210, 163, 76, 37)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__15_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__15_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__16_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__15_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__9_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(245, 208, 61, 75, 58, 111, 203, 213)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__16_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__16_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__17_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "initFn"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__17_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__17_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__18_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__16_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__17_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(28, 74, 223, 177, 221, 116, 12, 133)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__18_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__18_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__19_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_@"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__19_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__19_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__20_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__18_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__19_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(149, 168, 5, 102, 55, 146, 148, 183)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__20_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__20_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__21_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__20_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__5_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(240, 59, 178, 168, 141, 74, 253, 252)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__21_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__21_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__22_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__21_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__7_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(141, 84, 122, 125, 38, 219, 191, 250)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__22_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__22_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__23_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__22_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__9_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(198, 105, 6, 122, 124, 133, 36, 125)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__23_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__23_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__24_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__23_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__11_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(77, 95, 225, 110, 108, 42, 151, 218)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__24_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__24_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__25_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__24_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value),((lean_object*)(((size_t)(1868565755) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(132, 36, 172, 51, 233, 159, 108, 144)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__25_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__25_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__26_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "_hygCtx"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__26_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__26_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__27_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__25_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__26_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(171, 141, 109, 76, 137, 163, 233, 53)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__27_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__27_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__28_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "_hyg"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__28_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__28_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__29_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__27_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__28_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(107, 167, 198, 98, 38, 212, 177, 224)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__29_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__29_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__30_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__29_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value),((lean_object*)(((size_t)(2) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(190, 113, 236, 153, 64, 239, 86, 194)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__30_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__30_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Bound_isZero_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Bound_isZero_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Bound_isZero_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Bound_isZero_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Bound_isZero_spec__1___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Bound_isZero_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Bound_isZero_spec__1(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Bound_isZero_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Bound_isZero___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ofNat"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound_isZero___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_isZero___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound_isZero___lam__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound_isZero___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Bound_isZero___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "OfNat"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound_isZero___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_isZero___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound_isZero___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_isZero___closed__0_value),LEAN_SCALAR_PTR_LITERAL(135, 241, 166, 108, 243, 216, 193, 244)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound_isZero___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_isZero___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound_isZero___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound_isZero___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_isZero___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Bound_isZero___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Bound_isZero___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound_isZero(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound_isZero___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound_ineqPriority(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound_ineqPriority___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00Mathlib_Tactic_Bound_hypPriority_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00Mathlib_Tactic_Bound_hypPriority_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00Mathlib_Tactic_Bound_hypPriority_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00Mathlib_Tactic_Bound_hypPriority_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "And"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(49, 220, 212, 156, 122, 214, 55, 135)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__0___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__0___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__0(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Or"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(34, 237, 162, 225, 217, 98, 205, 196)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__1___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__1___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__1___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__1(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "LE"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__2___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__2___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__2___closed__0_value),LEAN_SCALAR_PTR_LITERAL(216, 149, 183, 186, 191, 145, 216, 115)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__2___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__2___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "le"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__2___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__2___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__2___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__2___closed__0_value),LEAN_SCALAR_PTR_LITERAL(216, 149, 183, 186, 191, 145, 216, 115)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__2___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__2___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__2___closed__2_value),LEAN_SCALAR_PTR_LITERAL(109, 14, 90, 172, 72, 170, 136, 101)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__2___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__2___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__2(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "LT"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__3___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__3___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__3___closed__0_value),LEAN_SCALAR_PTR_LITERAL(71, 235, 154, 184, 62, 135, 30, 248)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__3___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__3___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__3___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "lt"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__3___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__3___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__3___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__3___closed__0_value),LEAN_SCALAR_PTR_LITERAL(71, 235, 154, 184, 62, 135, 30, 248)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__3___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__3___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__3___closed__2_value),LEAN_SCALAR_PTR_LITERAL(54, 235, 251, 9, 4, 74, 57, 164)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__3___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__3___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__3(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "GE"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__4___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__4___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__4___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "ge"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__4___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__4___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__4___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__4___closed__0_value),LEAN_SCALAR_PTR_LITERAL(74, 169, 4, 72, 62, 21, 91, 24)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__4___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__4___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__4___closed__1_value),LEAN_SCALAR_PTR_LITERAL(71, 88, 92, 156, 129, 215, 23, 77)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__4___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__4___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__4(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__5___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "GT"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__5___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__5___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__5___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "gt"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__5___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__5___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__5___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__5___closed__0_value),LEAN_SCALAR_PTR_LITERAL(240, 16, 15, 58, 66, 186, 138, 31)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__5___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__5___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__5___closed__1_value),LEAN_SCALAR_PTR_LITERAL(239, 75, 137, 103, 59, 22, 209, 130)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__5___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__5___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Bound_hypPriority___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Bound_hypPriority___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Bound_hypPriority___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Bound_hypPriority___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound_hypPriority(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound_hypPriority___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_argPriority(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_argPriority___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult___lam__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult___lam__1(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult___lam__2(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "`"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult___closed__0_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "` has invalid type `"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult___closed__2_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult___closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 49, .m_capacity = 49, .m_length = 48, .m_data = "` as a 'bound' lemma: it should be an inequality"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult___closed__4_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00Mathlib_Tactic_Bound_typePriority_spec__1___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00Mathlib_Tactic_Bound_typePriority_spec__1___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00Mathlib_Tactic_Bound_typePriority_spec__1___redArg(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00Mathlib_Tactic_Bound_typePriority_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00Mathlib_Tactic_Bound_typePriority_spec__1(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00Mathlib_Tactic_Bound_typePriority_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Bound_typePriority_spec__0(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Bound_typePriority_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound_typePriority___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound_typePriority___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound_typePriority(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound_typePriority___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Bound_declPriority___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "unknown declaration "};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound_declPriority___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_declPriority___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Bound_declPriority___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Bound_declPriority___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound_declPriority(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound_declPriority___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Mathlib_Tactic_Bound_scoreToConfig_spec__0(lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound_scoreToConfig___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*0 + 8, .m_other = 0, .m_tag = 0}, .m_objs = {LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound_scoreToConfig___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_scoreToConfig___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound_scoreToConfig___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_scoreToConfig___closed__0_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound_scoreToConfig___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_scoreToConfig___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound_scoreToConfig___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__9_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(248, 144, 81, 165, 73, 52, 205, 25)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound_scoreToConfig___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_scoreToConfig___closed__2_value;
static const lean_array_object lp_mathlib_Mathlib_Tactic_Bound_scoreToConfig___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 246}, .m_size = 1, .m_capacity = 1, .m_data = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_scoreToConfig___closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound_scoreToConfig___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_scoreToConfig___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound_scoreToConfig(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1___lam__0(lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1_spec__2___redArg___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1_spec__2___redArg___lam__1___boxed(lean_object*);
static lean_once_cell_t lp_mathlib_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__11___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__11___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__11___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__11___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__11___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__11___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__11___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__11___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1_spec__2___redArg___lam__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1_spec__2___redArg___lam__3___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3_spec__16___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3_spec__16___closed__0;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3_spec__16___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3_spec__16___closed__1;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3_spec__16___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3_spec__16___closed__2;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3_spec__16___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3_spec__16___closed__3;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3_spec__16___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3_spec__16___closed__4;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3_spec__16___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3_spec__16___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3_spec__16(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3_spec__16___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8_spec__10_spec__17___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8_spec__10_spec__17___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8_spec__10___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8_spec__10___redArg___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "internal error: expected '"};
static const lean_object* lp_mathlib_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8___closed__0 = (const lean_object*)&lp_mathlib_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8___closed__0_value;
static lean_once_cell_t lp_mathlib_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8___closed__1;
static const lean_string_object lp_mathlib_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "' to be a declared simp extension"};
static const lean_object* lp_mathlib_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8___closed__2 = (const lean_object*)&lp_mathlib_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8___closed__2_value;
static lean_once_cell_t lp_mathlib_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8___closed__3;
static const lean_string_object lp_mathlib_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "no such rule set: '"};
static const lean_object* lp_mathlib_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8___closed__4 = (const lean_object*)&lp_mathlib_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8___closed__4_value;
static lean_once_cell_t lp_mathlib_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8___closed__5;
static const lean_string_object lp_mathlib_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 177, .m_capacity = 177, .m_length = 176, .m_data = "'\n  (Use 'declare_aesop_rule_set' to declare rule sets.\n   Declared rule sets are not visible in the current file; they only become visible once you import the declaring file.)"};
static const lean_object* lp_mathlib_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8___closed__6 = (const lean_object*)&lp_mathlib_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8___closed__6_value;
static lean_once_cell_t lp_mathlib_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8___closed__7;
LEAN_EXPORT lean_object* lp_mathlib_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1_spec__2___redArg___lam__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1_spec__2___redArg___lam__4___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1_spec__2___redArg___lam__2(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1_spec__2___redArg___lam__2___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1_spec__2___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1_spec__2___redArg___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1_spec__2___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1_spec__2___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1_spec__2___redArg___closed__0 = (const lean_object*)&lp_mathlib_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1_spec__2___redArg___closed__0_value;
static const lean_closure_object lp_mathlib_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1_spec__2___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1_spec__2___redArg___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1_spec__2___redArg___closed__1 = (const lean_object*)&lp_mathlib_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1_spec__2___redArg___closed__1_value;
static const lean_closure_object lp_mathlib_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1_spec__2___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1_spec__2___redArg___lam__2___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1_spec__2___redArg___closed__2 = (const lean_object*)&lp_mathlib_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1_spec__2___redArg___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__6(lean_object*, lean_object*, size_t, size_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__2(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__3(lean_object*, lean_object*, size_t, size_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__4(uint8_t, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "'"};
static const lean_object* lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__0 = (const lean_object*)&lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__0_value;
static lean_once_cell_t lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__1;
static const lean_string_object lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 63, .m_capacity = 63, .m_length = 62, .m_data = "' is not registered (with the given features) in any rule set."};
static const lean_object* lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__2 = (const lean_object*)&lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__2_value;
static lean_once_cell_t lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__3;
static const lean_string_object lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 71, .m_capacity = 71, .m_length = 70, .m_data = "' is not registered (with the given features) in any of the rule sets "};
static const lean_object* lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__4 = (const lean_object*)&lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__4_value;
static lean_once_cell_t lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__5;
static const lean_string_object lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "."};
static const lean_object* lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__6 = (const lean_object*)&lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__6_value;
static lean_once_cell_t lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__7;
LEAN_EXPORT lean_object* lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__0___closed__0_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__0___closed__0_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__0___closed__0_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__0_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__0_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__1_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__1_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__3_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__3_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Aesop_Frontend_getGlobalRuleSetFromData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__13___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Aesop_Frontend_getGlobalRuleSetFromData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__13___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_ScopedEnvExtension_add___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__9___redArg(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_ScopedEnvExtension_add___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__9___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__10_spec__13_spec__20_spec__22(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__10_spec__13_spec__20_spec__22___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__10_spec__13_spec__20(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__10_spec__13_spec__20___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__10_spec__13___redArg(lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__10_spec__13___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__10___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__10___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__12___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__12___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__12(lean_object*, uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__12___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "aesop: rule '"};
static const lean_object* lp_mathlib_Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1___closed__0 = (const lean_object*)&lp_mathlib_Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1___closed__0_value;
static lean_once_cell_t lp_mathlib_Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1___closed__1;
static const lean_string_object lp_mathlib_Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 38, .m_capacity = 38, .m_length = 37, .m_data = "' is already registered in rule set '"};
static const lean_object* lp_mathlib_Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1___closed__2 = (const lean_object*)&lp_mathlib_Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1___closed__2_value;
static lean_once_cell_t lp_mathlib_Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__2(lean_object*, uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3___closed__0;
static const lean_string_object lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3___closed__1 = (const lean_object*)&lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3___closed__1_value;
static const lean_array_object lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3___closed__2 = (const lean_object*)&lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__0_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*0 + 24, .m_other = 0, .m_tag = 0}, .m_objs = {LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 1, 1, 0),LEAN_SCALAR_PTR_LITERAL(1, 1, 0, 1, 1, 1, 2, 1),LEAN_SCALAR_PTR_LITERAL(1, 1, 1, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__0_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__0_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__1_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static uint64_t lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__1_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__2_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__2_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__3_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__3_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__4_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__4_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__5_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__5_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__6_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__6_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__7_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*7 + 0, .m_other = 7, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__7_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__7_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__8_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__8_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__8_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__9_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__8_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__9_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__9_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__10_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "' has score '"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__10_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__10_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__11_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__11_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__0_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__0_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2____boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__0_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__0_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__1_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__1_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2____boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__1_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__1_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__2_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__2_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__3_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__3_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__4_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__4_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__5_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__5_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__6_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*4, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2____boxed, .m_arity = 10, .m_num_fixed = 4, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__1_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__1_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__0_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value)} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__6_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__6_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__7_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__0_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(67, 51, 108, 71, 20, 161, 164, 139)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__7_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__7_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__8_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 60, .m_capacity = 60, .m_length = 59, .m_data = "Register a theorem as an apply rule for the `bound` tactic."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__8_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__8_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__9_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__9_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__10_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__10_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_ScopedEnvExtension_add___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_ScopedEnvExtension_add___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__11(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Aesop_Frontend_getGlobalRuleSetFromData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__13(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Aesop_Frontend_getGlobalRuleSetFromData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__13___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__10(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__10___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8_spec__10(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8_spec__10___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__10_spec__13(lean_object*, lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__10_spec__13___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8_spec__10_spec__17(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8_spec__10_spec__17___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Bound_attrBound__forward___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "attrBound_forward"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound_attrBound__forward___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_attrBound__forward___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound_attrBound__forward___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__5_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound_attrBound__forward___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_attrBound__forward___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__7_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound_attrBound__forward___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_attrBound__forward___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__9_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(72, 242, 195, 86, 141, 25, 214, 168)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound_attrBound__forward___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_attrBound__forward___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_attrBound__forward___closed__0_value),LEAN_SCALAR_PTR_LITERAL(220, 179, 210, 210, 129, 190, 143, 61)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound_attrBound__forward___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_attrBound__forward___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Bound_attrBound__forward___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "bound_forward"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound_attrBound__forward___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_attrBound__forward___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound_attrBound__forward___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_attrBound__forward___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound_attrBound__forward___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_attrBound__forward___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound_attrBound__forward___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_attrBound__forward___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_attrBound__forward___closed__3_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound_attrBound__forward___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_attrBound__forward___closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_Bound_attrBound__forward = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound_attrBound__forward___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Aesop"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Frontend"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "aesop"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(212, 61, 86, 210, 90, 71, 197, 0)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(13, 69, 189, 207, 58, 198, 186, 160)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(203, 146, 252, 150, 212, 106, 105, 173)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "attr_rules_"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__6_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(212, 61, 86, 210, 90, 71, 197, 0)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__6_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(13, 69, 189, 207, 58, 198, 186, 160)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__6_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(117, 200, 0, 99, 172, 255, 207, 57)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "rule_expr___"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(212, 61, 86, 210, 90, 71, 197, 0)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__8_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(13, 69, 189, 207, 58, 198, 186, 160)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__8_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(65, 164, 1, 162, 58, 43, 108, 12)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "choice"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__9_value),LEAN_SCALAR_PTR_LITERAL(59, 66, 148, 42, 181, 100, 85, 166)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "feature_"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__12_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__12_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__12_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(212, 61, 86, 210, 90, 71, 197, 0)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__12_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__12_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(13, 69, 189, 207, 58, 198, 186, 160)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__12_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(199, 138, 252, 41, 221, 82, 223, 77)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "phaseSafe"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__14_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__14_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(212, 61, 86, 210, 90, 71, 197, 0)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__14_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__14_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(13, 69, 189, 207, 58, 198, 186, 160)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__14_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__13_value),LEAN_SCALAR_PTR_LITERAL(227, 182, 27, 143, 126, 198, 52, 83)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "safe"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__15_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "featIdent"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__17_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__17_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__17_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(212, 61, 86, 210, 90, 71, 197, 0)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__17_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__17_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(13, 69, 189, 207, 58, 198, 186, 160)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__17_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__16_value),LEAN_SCALAR_PTR_LITERAL(224, 77, 15, 252, 144, 160, 175, 23)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__17_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__18;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__15_value),LEAN_SCALAR_PTR_LITERAL(38, 58, 162, 249, 229, 252, 65, 54)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__19_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "feature__2"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__20_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__21_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__21_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__21_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(212, 61, 86, 210, 90, 71, 197, 0)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__21_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__21_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(13, 69, 189, 207, 58, 198, 186, 160)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__21_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__20_value),LEAN_SCALAR_PTR_LITERAL(104, 166, 179, 245, 16, 140, 238, 209)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__21_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "builder_nameForward"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__22_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__23_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__23_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__23_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(212, 61, 86, 210, 90, 71, 197, 0)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__23_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__23_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(13, 69, 189, 207, 58, 198, 186, 160)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__23_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__22_value),LEAN_SCALAR_PTR_LITERAL(1, 110, 197, 236, 233, 87, 139, 91)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__23_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "forward"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__24_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__25_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__25;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__24_value),LEAN_SCALAR_PTR_LITERAL(71, 99, 59, 33, 27, 114, 107, 49)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__26_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "rule_expr_"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__27_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__28_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__28_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__28_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(212, 61, 86, 210, 90, 71, 197, 0)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__28_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__28_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(13, 69, 189, 207, 58, 198, 186, 160)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__28_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__27_value),LEAN_SCALAR_PTR_LITERAL(185, 88, 72, 227, 167, 159, 138, 146)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__28 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__28_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "feature__4"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__29_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__30_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__30_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__30_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(212, 61, 86, 210, 90, 71, 197, 0)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__30_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__30_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(13, 69, 189, 207, 58, 198, 186, 160)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__30_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__29_value),LEAN_SCALAR_PTR_LITERAL(129, 92, 181, 200, 21, 218, 115, 127)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__30 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__30_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "ruleSetsFeature"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__31 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__31_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__32_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__32_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__32_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(212, 61, 86, 210, 90, 71, 197, 0)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__32_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__32_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(13, 69, 189, 207, 58, 198, 186, 160)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__32_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__31_value),LEAN_SCALAR_PTR_LITERAL(44, 55, 27, 5, 41, 2, 63, 105)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__32 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__32_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__33 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__33_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "rule_sets"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__34 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__34_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ":="};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__35 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__35_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__36 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__36_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__37 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__37_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__37_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__38 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__38_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__39_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__39;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__40 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__40_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__41 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__41_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_73_; uint8_t v___x_74_; lean_object* v___x_75_; lean_object* v___x_76_; 
v___x_73_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__2_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2_));
v___x_74_ = 0;
v___x_75_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__30_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2_));
v___x_76_ = l_Lean_registerTraceClass(v___x_73_, v___x_74_, v___x_75_);
return v___x_76_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2____boxed(lean_object* v_a_77_){
_start:
{
lean_object* v_res_78_; 
v_res_78_ = lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2_();
return v_res_78_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Bound_isZero_spec__0___redArg(lean_object* v_e_79_, lean_object* v___y_80_){
_start:
{
uint8_t v___x_82_; 
v___x_82_ = l_Lean_Expr_hasMVar(v_e_79_);
if (v___x_82_ == 0)
{
lean_object* v___x_83_; 
v___x_83_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_83_, 0, v_e_79_);
return v___x_83_;
}
else
{
lean_object* v___x_84_; lean_object* v_mctx_85_; lean_object* v___x_86_; lean_object* v_fst_87_; lean_object* v_snd_88_; lean_object* v___x_89_; lean_object* v_cache_90_; lean_object* v_zetaDeltaFVarIds_91_; lean_object* v_postponed_92_; lean_object* v_diag_93_; lean_object* v___x_95_; uint8_t v_isShared_96_; uint8_t v_isSharedCheck_102_; 
v___x_84_ = lean_st_ref_get(v___y_80_);
v_mctx_85_ = lean_ctor_get(v___x_84_, 0);
lean_inc_ref(v_mctx_85_);
lean_dec(v___x_84_);
v___x_86_ = l_Lean_instantiateMVarsCore(v_mctx_85_, v_e_79_);
v_fst_87_ = lean_ctor_get(v___x_86_, 0);
lean_inc(v_fst_87_);
v_snd_88_ = lean_ctor_get(v___x_86_, 1);
lean_inc(v_snd_88_);
lean_dec_ref(v___x_86_);
v___x_89_ = lean_st_ref_take(v___y_80_);
v_cache_90_ = lean_ctor_get(v___x_89_, 1);
v_zetaDeltaFVarIds_91_ = lean_ctor_get(v___x_89_, 2);
v_postponed_92_ = lean_ctor_get(v___x_89_, 3);
v_diag_93_ = lean_ctor_get(v___x_89_, 4);
v_isSharedCheck_102_ = !lean_is_exclusive(v___x_89_);
if (v_isSharedCheck_102_ == 0)
{
lean_object* v_unused_103_; 
v_unused_103_ = lean_ctor_get(v___x_89_, 0);
lean_dec(v_unused_103_);
v___x_95_ = v___x_89_;
v_isShared_96_ = v_isSharedCheck_102_;
goto v_resetjp_94_;
}
else
{
lean_inc(v_diag_93_);
lean_inc(v_postponed_92_);
lean_inc(v_zetaDeltaFVarIds_91_);
lean_inc(v_cache_90_);
lean_dec(v___x_89_);
v___x_95_ = lean_box(0);
v_isShared_96_ = v_isSharedCheck_102_;
goto v_resetjp_94_;
}
v_resetjp_94_:
{
lean_object* v___x_98_; 
if (v_isShared_96_ == 0)
{
lean_ctor_set(v___x_95_, 0, v_snd_88_);
v___x_98_ = v___x_95_;
goto v_reusejp_97_;
}
else
{
lean_object* v_reuseFailAlloc_101_; 
v_reuseFailAlloc_101_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_101_, 0, v_snd_88_);
lean_ctor_set(v_reuseFailAlloc_101_, 1, v_cache_90_);
lean_ctor_set(v_reuseFailAlloc_101_, 2, v_zetaDeltaFVarIds_91_);
lean_ctor_set(v_reuseFailAlloc_101_, 3, v_postponed_92_);
lean_ctor_set(v_reuseFailAlloc_101_, 4, v_diag_93_);
v___x_98_ = v_reuseFailAlloc_101_;
goto v_reusejp_97_;
}
v_reusejp_97_:
{
lean_object* v___x_99_; lean_object* v___x_100_; 
v___x_99_ = lean_st_ref_set(v___y_80_, v___x_98_);
v___x_100_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_100_, 0, v_fst_87_);
return v___x_100_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Bound_isZero_spec__0___redArg___boxed(lean_object* v_e_104_, lean_object* v___y_105_, lean_object* v___y_106_){
_start:
{
lean_object* v_res_107_; 
v_res_107_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Bound_isZero_spec__0___redArg(v_e_104_, v___y_105_);
lean_dec(v___y_105_);
return v_res_107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Bound_isZero_spec__0(lean_object* v_e_108_, lean_object* v___y_109_, lean_object* v___y_110_, lean_object* v___y_111_, lean_object* v___y_112_){
_start:
{
lean_object* v___x_114_; 
v___x_114_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Bound_isZero_spec__0___redArg(v_e_108_, v___y_110_);
return v___x_114_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Bound_isZero_spec__0___boxed(lean_object* v_e_115_, lean_object* v___y_116_, lean_object* v___y_117_, lean_object* v___y_118_, lean_object* v___y_119_, lean_object* v___y_120_){
_start:
{
lean_object* v_res_121_; 
v_res_121_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Bound_isZero_spec__0(v_e_115_, v___y_116_, v___y_117_, v___y_118_, v___y_119_);
lean_dec(v___y_119_);
lean_dec_ref(v___y_118_);
lean_dec(v___y_117_);
lean_dec_ref(v___y_116_);
return v_res_121_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Bound_isZero_spec__1___redArg(lean_object* v_k_122_, uint8_t v_allowLevelAssignments_123_, lean_object* v___y_124_, lean_object* v___y_125_, lean_object* v___y_126_, lean_object* v___y_127_){
_start:
{
lean_object* v___x_129_; 
v___x_129_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_box(0), v_allowLevelAssignments_123_, v_k_122_, v___y_124_, v___y_125_, v___y_126_, v___y_127_);
if (lean_obj_tag(v___x_129_) == 0)
{
lean_object* v_a_130_; lean_object* v___x_132_; uint8_t v_isShared_133_; uint8_t v_isSharedCheck_137_; 
v_a_130_ = lean_ctor_get(v___x_129_, 0);
v_isSharedCheck_137_ = !lean_is_exclusive(v___x_129_);
if (v_isSharedCheck_137_ == 0)
{
v___x_132_ = v___x_129_;
v_isShared_133_ = v_isSharedCheck_137_;
goto v_resetjp_131_;
}
else
{
lean_inc(v_a_130_);
lean_dec(v___x_129_);
v___x_132_ = lean_box(0);
v_isShared_133_ = v_isSharedCheck_137_;
goto v_resetjp_131_;
}
v_resetjp_131_:
{
lean_object* v___x_135_; 
if (v_isShared_133_ == 0)
{
v___x_135_ = v___x_132_;
goto v_reusejp_134_;
}
else
{
lean_object* v_reuseFailAlloc_136_; 
v_reuseFailAlloc_136_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_136_, 0, v_a_130_);
v___x_135_ = v_reuseFailAlloc_136_;
goto v_reusejp_134_;
}
v_reusejp_134_:
{
return v___x_135_;
}
}
}
else
{
lean_object* v_a_138_; lean_object* v___x_140_; uint8_t v_isShared_141_; uint8_t v_isSharedCheck_145_; 
v_a_138_ = lean_ctor_get(v___x_129_, 0);
v_isSharedCheck_145_ = !lean_is_exclusive(v___x_129_);
if (v_isSharedCheck_145_ == 0)
{
v___x_140_ = v___x_129_;
v_isShared_141_ = v_isSharedCheck_145_;
goto v_resetjp_139_;
}
else
{
lean_inc(v_a_138_);
lean_dec(v___x_129_);
v___x_140_ = lean_box(0);
v_isShared_141_ = v_isSharedCheck_145_;
goto v_resetjp_139_;
}
v_resetjp_139_:
{
lean_object* v___x_143_; 
if (v_isShared_141_ == 0)
{
v___x_143_ = v___x_140_;
goto v_reusejp_142_;
}
else
{
lean_object* v_reuseFailAlloc_144_; 
v_reuseFailAlloc_144_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_144_, 0, v_a_138_);
v___x_143_ = v_reuseFailAlloc_144_;
goto v_reusejp_142_;
}
v_reusejp_142_:
{
return v___x_143_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Bound_isZero_spec__1___redArg___boxed(lean_object* v_k_146_, lean_object* v_allowLevelAssignments_147_, lean_object* v___y_148_, lean_object* v___y_149_, lean_object* v___y_150_, lean_object* v___y_151_, lean_object* v___y_152_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_153_; lean_object* v_res_154_; 
v_allowLevelAssignments_boxed_153_ = lean_unbox(v_allowLevelAssignments_147_);
v_res_154_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Bound_isZero_spec__1___redArg(v_k_146_, v_allowLevelAssignments_boxed_153_, v___y_148_, v___y_149_, v___y_150_, v___y_151_);
lean_dec(v___y_151_);
lean_dec_ref(v___y_150_);
lean_dec(v___y_149_);
lean_dec_ref(v___y_148_);
return v_res_154_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Bound_isZero_spec__1(lean_object* v_00_u03b1_155_, lean_object* v_k_156_, uint8_t v_allowLevelAssignments_157_, lean_object* v___y_158_, lean_object* v___y_159_, lean_object* v___y_160_, lean_object* v___y_161_){
_start:
{
lean_object* v___x_163_; 
v___x_163_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Bound_isZero_spec__1___redArg(v_k_156_, v_allowLevelAssignments_157_, v___y_158_, v___y_159_, v___y_160_, v___y_161_);
return v___x_163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Bound_isZero_spec__1___boxed(lean_object* v_00_u03b1_164_, lean_object* v_k_165_, lean_object* v_allowLevelAssignments_166_, lean_object* v___y_167_, lean_object* v___y_168_, lean_object* v___y_169_, lean_object* v___y_170_, lean_object* v___y_171_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_172_; lean_object* v_res_173_; 
v_allowLevelAssignments_boxed_172_ = lean_unbox(v_allowLevelAssignments_166_);
v_res_173_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Bound_isZero_spec__1(v_00_u03b1_164_, v_k_165_, v_allowLevelAssignments_boxed_172_, v___y_167_, v___y_168_, v___y_169_, v___y_170_);
lean_dec(v___y_170_);
lean_dec_ref(v___y_169_);
lean_dec(v___y_168_);
lean_dec_ref(v___y_167_);
return v_res_173_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound_isZero___lam__0(lean_object* v___x_175_, uint8_t v___x_176_, lean_object* v___x_177_, lean_object* v___x_178_, lean_object* v___x_179_, lean_object* v_00_u03b1_180_, lean_object* v___x_181_, lean_object* v_e_182_, uint8_t v___x_183_, lean_object* v___y_184_, lean_object* v___y_185_, lean_object* v___y_186_, lean_object* v___y_187_){
_start:
{
lean_object* v___x_189_; 
v___x_189_ = l_Lean_Meta_mkFreshExprMVar(v___x_175_, v___x_176_, v___x_177_, v___y_184_, v___y_185_, v___y_186_, v___y_187_);
if (lean_obj_tag(v___x_189_) == 0)
{
lean_object* v_a_190_; lean_object* v_keyedConfig_191_; uint8_t v_trackZetaDelta_192_; lean_object* v_zetaDeltaSet_193_; lean_object* v_lctx_194_; lean_object* v_localInstances_195_; lean_object* v_defEqCtx_x3f_196_; lean_object* v_synthPendingDepth_197_; lean_object* v_customCanUnfoldPredicate_x3f_198_; uint8_t v_univApprox_199_; uint8_t v_inTypeClassResolution_200_; uint8_t v_cacheInferType_201_; lean_object* v___x_203_; uint8_t v_isShared_204_; uint8_t v_isSharedCheck_246_; 
v_a_190_ = lean_ctor_get(v___x_189_, 0);
lean_inc(v_a_190_);
lean_dec_ref_known(v___x_189_, 1);
v_keyedConfig_191_ = lean_ctor_get(v___y_184_, 0);
v_trackZetaDelta_192_ = lean_ctor_get_uint8(v___y_184_, sizeof(void*)*7);
v_zetaDeltaSet_193_ = lean_ctor_get(v___y_184_, 1);
v_lctx_194_ = lean_ctor_get(v___y_184_, 2);
v_localInstances_195_ = lean_ctor_get(v___y_184_, 3);
v_defEqCtx_x3f_196_ = lean_ctor_get(v___y_184_, 4);
v_synthPendingDepth_197_ = lean_ctor_get(v___y_184_, 5);
v_customCanUnfoldPredicate_x3f_198_ = lean_ctor_get(v___y_184_, 6);
v_univApprox_199_ = lean_ctor_get_uint8(v___y_184_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_200_ = lean_ctor_get_uint8(v___y_184_, sizeof(void*)*7 + 2);
v_cacheInferType_201_ = lean_ctor_get_uint8(v___y_184_, sizeof(void*)*7 + 3);
v_isSharedCheck_246_ = !lean_is_exclusive(v___y_184_);
if (v_isSharedCheck_246_ == 0)
{
v___x_203_ = v___y_184_;
v_isShared_204_ = v_isSharedCheck_246_;
goto v_resetjp_202_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_198_);
lean_inc(v_synthPendingDepth_197_);
lean_inc(v_defEqCtx_x3f_196_);
lean_inc(v_localInstances_195_);
lean_inc(v_lctx_194_);
lean_inc(v_zetaDeltaSet_193_);
lean_inc(v_keyedConfig_191_);
lean_dec(v___y_184_);
v___x_203_ = lean_box(0);
v_isShared_204_ = v_isSharedCheck_246_;
goto v_resetjp_202_;
}
v_resetjp_202_:
{
lean_object* v___x_205_; lean_object* v___x_206_; lean_object* v___x_207_; lean_object* v___x_208_; lean_object* v___x_209_; lean_object* v___x_210_; uint8_t v___x_211_; lean_object* v___x_212_; lean_object* v___x_214_; 
v___x_205_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Bound_isZero___lam__0___closed__0));
v___x_206_ = l_Lean_Name_mkStr2(v___x_178_, v___x_205_);
v___x_207_ = l_Lean_Expr_const___override(v___x_206_, v___x_179_);
v___x_208_ = l_Lean_Expr_app___override(v___x_207_, v_00_u03b1_180_);
v___x_209_ = l_Lean_Expr_app___override(v___x_208_, v___x_181_);
lean_inc(v_a_190_);
v___x_210_ = l_Lean_Expr_app___override(v___x_209_, v_a_190_);
v___x_211_ = 2;
v___x_212_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_211_, v_keyedConfig_191_);
if (v_isShared_204_ == 0)
{
lean_ctor_set(v___x_203_, 0, v___x_212_);
v___x_214_ = v___x_203_;
goto v_reusejp_213_;
}
else
{
lean_object* v_reuseFailAlloc_245_; 
v_reuseFailAlloc_245_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_245_, 0, v___x_212_);
lean_ctor_set(v_reuseFailAlloc_245_, 1, v_zetaDeltaSet_193_);
lean_ctor_set(v_reuseFailAlloc_245_, 2, v_lctx_194_);
lean_ctor_set(v_reuseFailAlloc_245_, 3, v_localInstances_195_);
lean_ctor_set(v_reuseFailAlloc_245_, 4, v_defEqCtx_x3f_196_);
lean_ctor_set(v_reuseFailAlloc_245_, 5, v_synthPendingDepth_197_);
lean_ctor_set(v_reuseFailAlloc_245_, 6, v_customCanUnfoldPredicate_x3f_198_);
lean_ctor_set_uint8(v_reuseFailAlloc_245_, sizeof(void*)*7, v_trackZetaDelta_192_);
lean_ctor_set_uint8(v_reuseFailAlloc_245_, sizeof(void*)*7 + 1, v_univApprox_199_);
lean_ctor_set_uint8(v_reuseFailAlloc_245_, sizeof(void*)*7 + 2, v_inTypeClassResolution_200_);
lean_ctor_set_uint8(v_reuseFailAlloc_245_, sizeof(void*)*7 + 3, v_cacheInferType_201_);
v___x_214_ = v_reuseFailAlloc_245_;
goto v_reusejp_213_;
}
v_reusejp_213_:
{
lean_object* v___x_215_; 
v___x_215_ = l_Lean_Meta_isExprDefEq(v___x_210_, v_e_182_, v___x_214_, v___y_185_, v___y_186_, v___y_187_);
lean_dec_ref(v___x_214_);
if (lean_obj_tag(v___x_215_) == 0)
{
lean_object* v_a_216_; lean_object* v___x_218_; uint8_t v_isShared_219_; uint8_t v_isSharedCheck_236_; 
v_a_216_ = lean_ctor_get(v___x_215_, 0);
v_isSharedCheck_236_ = !lean_is_exclusive(v___x_215_);
if (v_isSharedCheck_236_ == 0)
{
v___x_218_ = v___x_215_;
v_isShared_219_ = v_isSharedCheck_236_;
goto v_resetjp_217_;
}
else
{
lean_inc(v_a_216_);
lean_dec(v___x_215_);
v___x_218_ = lean_box(0);
v_isShared_219_ = v_isSharedCheck_236_;
goto v_resetjp_217_;
}
v_resetjp_217_:
{
uint8_t v___x_220_; 
v___x_220_ = lean_unbox(v_a_216_);
if (v___x_220_ == 0)
{
lean_object* v___x_221_; lean_object* v___x_222_; lean_object* v___x_224_; 
lean_dec(v_a_216_);
v___x_221_ = lean_box(v___x_183_);
v___x_222_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_222_, 0, v_a_190_);
lean_ctor_set(v___x_222_, 1, v___x_221_);
if (v_isShared_219_ == 0)
{
lean_ctor_set(v___x_218_, 0, v___x_222_);
v___x_224_ = v___x_218_;
goto v_reusejp_223_;
}
else
{
lean_object* v_reuseFailAlloc_225_; 
v_reuseFailAlloc_225_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_225_, 0, v___x_222_);
v___x_224_ = v_reuseFailAlloc_225_;
goto v_reusejp_223_;
}
v_reusejp_223_:
{
return v___x_224_;
}
}
else
{
lean_object* v___x_226_; lean_object* v_a_227_; lean_object* v___x_229_; uint8_t v_isShared_230_; uint8_t v_isSharedCheck_235_; 
lean_del_object(v___x_218_);
v___x_226_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Bound_isZero_spec__0___redArg(v_a_190_, v___y_185_);
v_a_227_ = lean_ctor_get(v___x_226_, 0);
v_isSharedCheck_235_ = !lean_is_exclusive(v___x_226_);
if (v_isSharedCheck_235_ == 0)
{
v___x_229_ = v___x_226_;
v_isShared_230_ = v_isSharedCheck_235_;
goto v_resetjp_228_;
}
else
{
lean_inc(v_a_227_);
lean_dec(v___x_226_);
v___x_229_ = lean_box(0);
v_isShared_230_ = v_isSharedCheck_235_;
goto v_resetjp_228_;
}
v_resetjp_228_:
{
lean_object* v___x_231_; lean_object* v___x_233_; 
v___x_231_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_231_, 0, v_a_227_);
lean_ctor_set(v___x_231_, 1, v_a_216_);
if (v_isShared_230_ == 0)
{
lean_ctor_set(v___x_229_, 0, v___x_231_);
v___x_233_ = v___x_229_;
goto v_reusejp_232_;
}
else
{
lean_object* v_reuseFailAlloc_234_; 
v_reuseFailAlloc_234_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_234_, 0, v___x_231_);
v___x_233_ = v_reuseFailAlloc_234_;
goto v_reusejp_232_;
}
v_reusejp_232_:
{
return v___x_233_;
}
}
}
}
}
else
{
lean_object* v_a_237_; lean_object* v___x_239_; uint8_t v_isShared_240_; uint8_t v_isSharedCheck_244_; 
lean_dec(v_a_190_);
v_a_237_ = lean_ctor_get(v___x_215_, 0);
v_isSharedCheck_244_ = !lean_is_exclusive(v___x_215_);
if (v_isSharedCheck_244_ == 0)
{
v___x_239_ = v___x_215_;
v_isShared_240_ = v_isSharedCheck_244_;
goto v_resetjp_238_;
}
else
{
lean_inc(v_a_237_);
lean_dec(v___x_215_);
v___x_239_ = lean_box(0);
v_isShared_240_ = v_isSharedCheck_244_;
goto v_resetjp_238_;
}
v_resetjp_238_:
{
lean_object* v___x_242_; 
if (v_isShared_240_ == 0)
{
v___x_242_ = v___x_239_;
goto v_reusejp_241_;
}
else
{
lean_object* v_reuseFailAlloc_243_; 
v_reuseFailAlloc_243_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_243_, 0, v_a_237_);
v___x_242_ = v_reuseFailAlloc_243_;
goto v_reusejp_241_;
}
v_reusejp_241_:
{
return v___x_242_;
}
}
}
}
}
}
else
{
lean_object* v_a_247_; lean_object* v___x_249_; uint8_t v_isShared_250_; uint8_t v_isSharedCheck_254_; 
lean_dec_ref(v___y_184_);
lean_dec_ref(v_e_182_);
lean_dec_ref(v___x_181_);
lean_dec_ref(v_00_u03b1_180_);
lean_dec(v___x_179_);
lean_dec_ref(v___x_178_);
v_a_247_ = lean_ctor_get(v___x_189_, 0);
v_isSharedCheck_254_ = !lean_is_exclusive(v___x_189_);
if (v_isSharedCheck_254_ == 0)
{
v___x_249_ = v___x_189_;
v_isShared_250_ = v_isSharedCheck_254_;
goto v_resetjp_248_;
}
else
{
lean_inc(v_a_247_);
lean_dec(v___x_189_);
v___x_249_ = lean_box(0);
v_isShared_250_ = v_isSharedCheck_254_;
goto v_resetjp_248_;
}
v_resetjp_248_:
{
lean_object* v___x_252_; 
if (v_isShared_250_ == 0)
{
v___x_252_ = v___x_249_;
goto v_reusejp_251_;
}
else
{
lean_object* v_reuseFailAlloc_253_; 
v_reuseFailAlloc_253_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_253_, 0, v_a_247_);
v___x_252_ = v_reuseFailAlloc_253_;
goto v_reusejp_251_;
}
v_reusejp_251_:
{
return v___x_252_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound_isZero___lam__0___boxed(lean_object* v___x_255_, lean_object* v___x_256_, lean_object* v___x_257_, lean_object* v___x_258_, lean_object* v___x_259_, lean_object* v_00_u03b1_260_, lean_object* v___x_261_, lean_object* v_e_262_, lean_object* v___x_263_, lean_object* v___y_264_, lean_object* v___y_265_, lean_object* v___y_266_, lean_object* v___y_267_, lean_object* v___y_268_){
_start:
{
uint8_t v___x_1872__boxed_269_; uint8_t v___x_1877__boxed_270_; lean_object* v_res_271_; 
v___x_1872__boxed_269_ = lean_unbox(v___x_256_);
v___x_1877__boxed_270_ = lean_unbox(v___x_263_);
v_res_271_ = lp_mathlib_Mathlib_Tactic_Bound_isZero___lam__0(v___x_255_, v___x_1872__boxed_269_, v___x_257_, v___x_258_, v___x_259_, v_00_u03b1_260_, v___x_261_, v_e_262_, v___x_1877__boxed_270_, v___y_264_, v___y_265_, v___y_266_, v___y_267_);
lean_dec(v___y_267_);
lean_dec_ref(v___y_266_);
lean_dec(v___y_265_);
return v_res_271_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Bound_isZero___closed__3(void){
_start:
{
lean_object* v___x_277_; lean_object* v___x_278_; 
v___x_277_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Bound_isZero___closed__2));
v___x_278_ = l_Lean_Expr_lit___override(v___x_277_);
return v___x_278_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound_isZero(lean_object* v_u_279_, lean_object* v_00_u03b1_280_, lean_object* v_e_281_, lean_object* v_a_282_, lean_object* v_a_283_, lean_object* v_a_284_, lean_object* v_a_285_){
_start:
{
uint8_t v___x_287_; lean_object* v___x_288_; lean_object* v___x_289_; lean_object* v___x_290_; lean_object* v___x_291_; lean_object* v___x_292_; lean_object* v___x_293_; lean_object* v___x_294_; lean_object* v___x_295_; lean_object* v___x_296_; uint8_t v___x_297_; lean_object* v___x_298_; lean_object* v___x_299_; lean_object* v___x_300_; lean_object* v___f_301_; lean_object* v___x_302_; 
v___x_287_ = 0;
v___x_288_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Bound_isZero___closed__0));
v___x_289_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Bound_isZero___closed__1));
v___x_290_ = lean_box(0);
v___x_291_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_291_, 0, v_u_279_);
lean_ctor_set(v___x_291_, 1, v___x_290_);
lean_inc_ref(v___x_291_);
v___x_292_ = l_Lean_Expr_const___override(v___x_289_, v___x_291_);
lean_inc_ref(v_00_u03b1_280_);
v___x_293_ = l_Lean_Expr_app___override(v___x_292_, v_00_u03b1_280_);
v___x_294_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Bound_isZero___closed__3, &lp_mathlib_Mathlib_Tactic_Bound_isZero___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_Bound_isZero___closed__3);
v___x_295_ = l_Lean_Expr_app___override(v___x_293_, v___x_294_);
v___x_296_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_296_, 0, v___x_295_);
v___x_297_ = 0;
v___x_298_ = lean_box(0);
v___x_299_ = lean_box(v___x_297_);
v___x_300_ = lean_box(v___x_287_);
v___f_301_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Bound_isZero___lam__0___boxed), 14, 9);
lean_closure_set(v___f_301_, 0, v___x_296_);
lean_closure_set(v___f_301_, 1, v___x_299_);
lean_closure_set(v___f_301_, 2, v___x_298_);
lean_closure_set(v___f_301_, 3, v___x_288_);
lean_closure_set(v___f_301_, 4, v___x_291_);
lean_closure_set(v___f_301_, 5, v_00_u03b1_280_);
lean_closure_set(v___f_301_, 6, v___x_294_);
lean_closure_set(v___f_301_, 7, v_e_281_);
lean_closure_set(v___f_301_, 8, v___x_300_);
v___x_302_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Bound_isZero_spec__1___redArg(v___f_301_, v___x_287_, v_a_282_, v_a_283_, v_a_284_, v_a_285_);
if (lean_obj_tag(v___x_302_) == 0)
{
lean_object* v_a_303_; lean_object* v___x_305_; uint8_t v_isShared_306_; uint8_t v_isSharedCheck_316_; 
v_a_303_ = lean_ctor_get(v___x_302_, 0);
v_isSharedCheck_316_ = !lean_is_exclusive(v___x_302_);
if (v_isSharedCheck_316_ == 0)
{
v___x_305_ = v___x_302_;
v_isShared_306_ = v_isSharedCheck_316_;
goto v_resetjp_304_;
}
else
{
lean_inc(v_a_303_);
lean_dec(v___x_302_);
v___x_305_ = lean_box(0);
v_isShared_306_ = v_isSharedCheck_316_;
goto v_resetjp_304_;
}
v_resetjp_304_:
{
lean_object* v_snd_307_; uint8_t v___x_308_; 
v_snd_307_ = lean_ctor_get(v_a_303_, 1);
lean_inc(v_snd_307_);
lean_dec(v_a_303_);
v___x_308_ = lean_unbox(v_snd_307_);
if (v___x_308_ == 0)
{
lean_object* v___x_309_; lean_object* v___x_311_; 
lean_dec(v_snd_307_);
v___x_309_ = lean_box(v___x_287_);
if (v_isShared_306_ == 0)
{
lean_ctor_set(v___x_305_, 0, v___x_309_);
v___x_311_ = v___x_305_;
goto v_reusejp_310_;
}
else
{
lean_object* v_reuseFailAlloc_312_; 
v_reuseFailAlloc_312_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_312_, 0, v___x_309_);
v___x_311_ = v_reuseFailAlloc_312_;
goto v_reusejp_310_;
}
v_reusejp_310_:
{
return v___x_311_;
}
}
else
{
lean_object* v___x_314_; 
if (v_isShared_306_ == 0)
{
lean_ctor_set(v___x_305_, 0, v_snd_307_);
v___x_314_ = v___x_305_;
goto v_reusejp_313_;
}
else
{
lean_object* v_reuseFailAlloc_315_; 
v_reuseFailAlloc_315_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_315_, 0, v_snd_307_);
v___x_314_ = v_reuseFailAlloc_315_;
goto v_reusejp_313_;
}
v_reusejp_313_:
{
return v___x_314_;
}
}
}
}
else
{
lean_object* v_a_317_; lean_object* v___x_319_; uint8_t v_isShared_320_; uint8_t v_isSharedCheck_324_; 
v_a_317_ = lean_ctor_get(v___x_302_, 0);
v_isSharedCheck_324_ = !lean_is_exclusive(v___x_302_);
if (v_isSharedCheck_324_ == 0)
{
v___x_319_ = v___x_302_;
v_isShared_320_ = v_isSharedCheck_324_;
goto v_resetjp_318_;
}
else
{
lean_inc(v_a_317_);
lean_dec(v___x_302_);
v___x_319_ = lean_box(0);
v_isShared_320_ = v_isSharedCheck_324_;
goto v_resetjp_318_;
}
v_resetjp_318_:
{
lean_object* v___x_322_; 
if (v_isShared_320_ == 0)
{
v___x_322_ = v___x_319_;
goto v_reusejp_321_;
}
else
{
lean_object* v_reuseFailAlloc_323_; 
v_reuseFailAlloc_323_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_323_, 0, v_a_317_);
v___x_322_ = v_reuseFailAlloc_323_;
goto v_reusejp_321_;
}
v_reusejp_321_:
{
return v___x_322_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound_isZero___boxed(lean_object* v_u_325_, lean_object* v_00_u03b1_326_, lean_object* v_e_327_, lean_object* v_a_328_, lean_object* v_a_329_, lean_object* v_a_330_, lean_object* v_a_331_, lean_object* v_a_332_){
_start:
{
lean_object* v_res_333_; 
v_res_333_ = lp_mathlib_Mathlib_Tactic_Bound_isZero(v_u_325_, v_00_u03b1_326_, v_e_327_, v_a_328_, v_a_329_, v_a_330_, v_a_331_);
lean_dec(v_a_331_);
lean_dec_ref(v_a_330_);
lean_dec(v_a_329_);
lean_dec_ref(v_a_328_);
return v_res_333_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound_ineqPriority(lean_object* v_u_334_, lean_object* v_00_u03b1_335_, lean_object* v_a_336_, lean_object* v_b_337_, lean_object* v_a_338_, lean_object* v_a_339_, lean_object* v_a_340_, lean_object* v_a_341_){
_start:
{
lean_object* v___x_343_; 
lean_inc_ref(v_00_u03b1_335_);
lean_inc(v_u_334_);
v___x_343_ = lp_mathlib_Mathlib_Tactic_Bound_isZero(v_u_334_, v_00_u03b1_335_, v_a_336_, v_a_338_, v_a_339_, v_a_340_, v_a_341_);
if (lean_obj_tag(v___x_343_) == 0)
{
lean_object* v_a_344_; lean_object* v___x_346_; uint8_t v_isShared_347_; uint8_t v_isSharedCheck_373_; 
v_a_344_ = lean_ctor_get(v___x_343_, 0);
v_isSharedCheck_373_ = !lean_is_exclusive(v___x_343_);
if (v_isSharedCheck_373_ == 0)
{
v___x_346_ = v___x_343_;
v_isShared_347_ = v_isSharedCheck_373_;
goto v_resetjp_345_;
}
else
{
lean_inc(v_a_344_);
lean_dec(v___x_343_);
v___x_346_ = lean_box(0);
v_isShared_347_ = v_isSharedCheck_373_;
goto v_resetjp_345_;
}
v_resetjp_345_:
{
lean_object* v___x_348_; 
v___x_348_ = lp_mathlib_Mathlib_Tactic_Bound_isZero(v_u_334_, v_00_u03b1_335_, v_b_337_, v_a_338_, v_a_339_, v_a_340_, v_a_341_);
if (lean_obj_tag(v___x_348_) == 0)
{
lean_object* v_a_349_; lean_object* v___x_351_; uint8_t v_isShared_352_; uint8_t v_isSharedCheck_364_; 
v_a_349_ = lean_ctor_get(v___x_348_, 0);
v_isSharedCheck_364_ = !lean_is_exclusive(v___x_348_);
if (v_isSharedCheck_364_ == 0)
{
v___x_351_ = v___x_348_;
v_isShared_352_ = v_isSharedCheck_364_;
goto v_resetjp_350_;
}
else
{
lean_inc(v_a_349_);
lean_dec(v___x_348_);
v___x_351_ = lean_box(0);
v_isShared_352_ = v_isSharedCheck_364_;
goto v_resetjp_350_;
}
v_resetjp_350_:
{
uint8_t v___x_358_; 
v___x_358_ = lean_unbox(v_a_344_);
lean_dec(v_a_344_);
if (v___x_358_ == 0)
{
uint8_t v___x_359_; 
v___x_359_ = lean_unbox(v_a_349_);
lean_dec(v_a_349_);
if (v___x_359_ == 0)
{
lean_object* v___x_360_; lean_object* v___x_362_; 
lean_del_object(v___x_351_);
v___x_360_ = lean_unsigned_to_nat(10u);
if (v_isShared_347_ == 0)
{
lean_ctor_set(v___x_346_, 0, v___x_360_);
v___x_362_ = v___x_346_;
goto v_reusejp_361_;
}
else
{
lean_object* v_reuseFailAlloc_363_; 
v_reuseFailAlloc_363_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_363_, 0, v___x_360_);
v___x_362_ = v_reuseFailAlloc_363_;
goto v_reusejp_361_;
}
v_reusejp_361_:
{
return v___x_362_;
}
}
else
{
lean_del_object(v___x_346_);
goto v___jp_353_;
}
}
else
{
lean_dec(v_a_349_);
lean_del_object(v___x_346_);
goto v___jp_353_;
}
v___jp_353_:
{
lean_object* v___x_354_; lean_object* v___x_356_; 
v___x_354_ = lean_unsigned_to_nat(1u);
if (v_isShared_352_ == 0)
{
lean_ctor_set(v___x_351_, 0, v___x_354_);
v___x_356_ = v___x_351_;
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
}
else
{
lean_object* v_a_365_; lean_object* v___x_367_; uint8_t v_isShared_368_; uint8_t v_isSharedCheck_372_; 
lean_del_object(v___x_346_);
lean_dec(v_a_344_);
v_a_365_ = lean_ctor_get(v___x_348_, 0);
v_isSharedCheck_372_ = !lean_is_exclusive(v___x_348_);
if (v_isSharedCheck_372_ == 0)
{
v___x_367_ = v___x_348_;
v_isShared_368_ = v_isSharedCheck_372_;
goto v_resetjp_366_;
}
else
{
lean_inc(v_a_365_);
lean_dec(v___x_348_);
v___x_367_ = lean_box(0);
v_isShared_368_ = v_isSharedCheck_372_;
goto v_resetjp_366_;
}
v_resetjp_366_:
{
lean_object* v___x_370_; 
if (v_isShared_368_ == 0)
{
v___x_370_ = v___x_367_;
goto v_reusejp_369_;
}
else
{
lean_object* v_reuseFailAlloc_371_; 
v_reuseFailAlloc_371_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_371_, 0, v_a_365_);
v___x_370_ = v_reuseFailAlloc_371_;
goto v_reusejp_369_;
}
v_reusejp_369_:
{
return v___x_370_;
}
}
}
}
}
else
{
lean_object* v_a_374_; lean_object* v___x_376_; uint8_t v_isShared_377_; uint8_t v_isSharedCheck_381_; 
lean_dec_ref(v_b_337_);
lean_dec_ref(v_00_u03b1_335_);
lean_dec(v_u_334_);
v_a_374_ = lean_ctor_get(v___x_343_, 0);
v_isSharedCheck_381_ = !lean_is_exclusive(v___x_343_);
if (v_isSharedCheck_381_ == 0)
{
v___x_376_ = v___x_343_;
v_isShared_377_ = v_isSharedCheck_381_;
goto v_resetjp_375_;
}
else
{
lean_inc(v_a_374_);
lean_dec(v___x_343_);
v___x_376_ = lean_box(0);
v_isShared_377_ = v_isSharedCheck_381_;
goto v_resetjp_375_;
}
v_resetjp_375_:
{
lean_object* v___x_379_; 
if (v_isShared_377_ == 0)
{
v___x_379_ = v___x_376_;
goto v_reusejp_378_;
}
else
{
lean_object* v_reuseFailAlloc_380_; 
v_reuseFailAlloc_380_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_380_, 0, v_a_374_);
v___x_379_ = v_reuseFailAlloc_380_;
goto v_reusejp_378_;
}
v_reusejp_378_:
{
return v___x_379_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound_ineqPriority___boxed(lean_object* v_u_382_, lean_object* v_00_u03b1_383_, lean_object* v_a_384_, lean_object* v_b_385_, lean_object* v_a_386_, lean_object* v_a_387_, lean_object* v_a_388_, lean_object* v_a_389_, lean_object* v_a_390_){
_start:
{
lean_object* v_res_391_; 
v_res_391_ = lp_mathlib_Mathlib_Tactic_Bound_ineqPriority(v_u_382_, v_00_u03b1_383_, v_a_384_, v_b_385_, v_a_386_, v_a_387_, v_a_388_, v_a_389_);
lean_dec(v_a_389_);
lean_dec_ref(v_a_388_);
lean_dec(v_a_387_);
lean_dec_ref(v_a_386_);
return v_res_391_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00Mathlib_Tactic_Bound_hypPriority_spec__0___redArg(lean_object* v_l_392_, lean_object* v___y_393_){
_start:
{
lean_object* v___x_395_; lean_object* v_mctx_396_; lean_object* v___x_397_; lean_object* v_fst_398_; lean_object* v_snd_399_; lean_object* v___x_400_; lean_object* v_cache_401_; lean_object* v_zetaDeltaFVarIds_402_; lean_object* v_postponed_403_; lean_object* v_diag_404_; lean_object* v___x_406_; uint8_t v_isShared_407_; uint8_t v_isSharedCheck_413_; 
v___x_395_ = lean_st_ref_get(v___y_393_);
v_mctx_396_ = lean_ctor_get(v___x_395_, 0);
lean_inc_ref(v_mctx_396_);
lean_dec(v___x_395_);
v___x_397_ = lean_instantiate_level_mvars(v_mctx_396_, v_l_392_);
v_fst_398_ = lean_ctor_get(v___x_397_, 0);
lean_inc(v_fst_398_);
v_snd_399_ = lean_ctor_get(v___x_397_, 1);
lean_inc(v_snd_399_);
lean_dec_ref(v___x_397_);
v___x_400_ = lean_st_ref_take(v___y_393_);
v_cache_401_ = lean_ctor_get(v___x_400_, 1);
v_zetaDeltaFVarIds_402_ = lean_ctor_get(v___x_400_, 2);
v_postponed_403_ = lean_ctor_get(v___x_400_, 3);
v_diag_404_ = lean_ctor_get(v___x_400_, 4);
v_isSharedCheck_413_ = !lean_is_exclusive(v___x_400_);
if (v_isSharedCheck_413_ == 0)
{
lean_object* v_unused_414_; 
v_unused_414_ = lean_ctor_get(v___x_400_, 0);
lean_dec(v_unused_414_);
v___x_406_ = v___x_400_;
v_isShared_407_ = v_isSharedCheck_413_;
goto v_resetjp_405_;
}
else
{
lean_inc(v_diag_404_);
lean_inc(v_postponed_403_);
lean_inc(v_zetaDeltaFVarIds_402_);
lean_inc(v_cache_401_);
lean_dec(v___x_400_);
v___x_406_ = lean_box(0);
v_isShared_407_ = v_isSharedCheck_413_;
goto v_resetjp_405_;
}
v_resetjp_405_:
{
lean_object* v___x_409_; 
if (v_isShared_407_ == 0)
{
lean_ctor_set(v___x_406_, 0, v_fst_398_);
v___x_409_ = v___x_406_;
goto v_reusejp_408_;
}
else
{
lean_object* v_reuseFailAlloc_412_; 
v_reuseFailAlloc_412_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_412_, 0, v_fst_398_);
lean_ctor_set(v_reuseFailAlloc_412_, 1, v_cache_401_);
lean_ctor_set(v_reuseFailAlloc_412_, 2, v_zetaDeltaFVarIds_402_);
lean_ctor_set(v_reuseFailAlloc_412_, 3, v_postponed_403_);
lean_ctor_set(v_reuseFailAlloc_412_, 4, v_diag_404_);
v___x_409_ = v_reuseFailAlloc_412_;
goto v_reusejp_408_;
}
v_reusejp_408_:
{
lean_object* v___x_410_; lean_object* v___x_411_; 
v___x_410_ = lean_st_ref_set(v___y_393_, v___x_409_);
v___x_411_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_411_, 0, v_snd_399_);
return v___x_411_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00Mathlib_Tactic_Bound_hypPriority_spec__0___redArg___boxed(lean_object* v_l_415_, lean_object* v___y_416_, lean_object* v___y_417_){
_start:
{
lean_object* v_res_418_; 
v_res_418_ = lp_mathlib_Lean_instantiateLevelMVars___at___00Mathlib_Tactic_Bound_hypPriority_spec__0___redArg(v_l_415_, v___y_416_);
lean_dec(v___y_416_);
return v_res_418_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00Mathlib_Tactic_Bound_hypPriority_spec__0(lean_object* v_l_419_, lean_object* v___y_420_, lean_object* v___y_421_, lean_object* v___y_422_, lean_object* v___y_423_){
_start:
{
lean_object* v___x_425_; 
v___x_425_ = lp_mathlib_Lean_instantiateLevelMVars___at___00Mathlib_Tactic_Bound_hypPriority_spec__0___redArg(v_l_419_, v___y_421_);
return v___x_425_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00Mathlib_Tactic_Bound_hypPriority_spec__0___boxed(lean_object* v_l_426_, lean_object* v___y_427_, lean_object* v___y_428_, lean_object* v___y_429_, lean_object* v___y_430_, lean_object* v___y_431_){
_start:
{
lean_object* v_res_432_; 
v_res_432_ = lp_mathlib_Lean_instantiateLevelMVars___at___00Mathlib_Tactic_Bound_hypPriority_spec__0(v_l_426_, v___y_427_, v___y_428_, v___y_429_, v___y_430_);
lean_dec(v___y_430_);
lean_dec_ref(v___y_429_);
lean_dec(v___y_428_);
lean_dec_ref(v___y_427_);
return v_res_432_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__0___closed__2(void){
_start:
{
lean_object* v___x_436_; lean_object* v___x_437_; lean_object* v___x_438_; 
v___x_436_ = lean_box(0);
v___x_437_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__0___closed__1));
v___x_438_ = l_Lean_Expr_const___override(v___x_437_, v___x_436_);
return v___x_438_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__0(lean_object* v___x_439_, uint8_t v___x_440_, lean_object* v___x_441_, lean_object* v_hyp_442_, uint8_t v___x_443_, lean_object* v___y_444_, lean_object* v___y_445_, lean_object* v___y_446_, lean_object* v___y_447_){
_start:
{
lean_object* v___x_449_; 
lean_inc(v___x_441_);
lean_inc(v___x_439_);
v___x_449_ = l_Lean_Meta_mkFreshExprMVar(v___x_439_, v___x_440_, v___x_441_, v___y_444_, v___y_445_, v___y_446_, v___y_447_);
if (lean_obj_tag(v___x_449_) == 0)
{
lean_object* v_a_450_; lean_object* v___x_451_; 
v_a_450_ = lean_ctor_get(v___x_449_, 0);
lean_inc(v_a_450_);
lean_dec_ref_known(v___x_449_, 1);
v___x_451_ = l_Lean_Meta_mkFreshExprMVar(v___x_439_, v___x_440_, v___x_441_, v___y_444_, v___y_445_, v___y_446_, v___y_447_);
if (lean_obj_tag(v___x_451_) == 0)
{
lean_object* v_a_452_; lean_object* v_keyedConfig_453_; uint8_t v_trackZetaDelta_454_; lean_object* v_zetaDeltaSet_455_; lean_object* v_lctx_456_; lean_object* v_localInstances_457_; lean_object* v_defEqCtx_x3f_458_; lean_object* v_synthPendingDepth_459_; lean_object* v_customCanUnfoldPredicate_x3f_460_; uint8_t v_univApprox_461_; uint8_t v_inTypeClassResolution_462_; uint8_t v_cacheInferType_463_; lean_object* v___x_465_; uint8_t v_isShared_466_; uint8_t v_isSharedCheck_525_; 
v_a_452_ = lean_ctor_get(v___x_451_, 0);
lean_inc(v_a_452_);
lean_dec_ref_known(v___x_451_, 1);
v_keyedConfig_453_ = lean_ctor_get(v___y_444_, 0);
v_trackZetaDelta_454_ = lean_ctor_get_uint8(v___y_444_, sizeof(void*)*7);
v_zetaDeltaSet_455_ = lean_ctor_get(v___y_444_, 1);
v_lctx_456_ = lean_ctor_get(v___y_444_, 2);
v_localInstances_457_ = lean_ctor_get(v___y_444_, 3);
v_defEqCtx_x3f_458_ = lean_ctor_get(v___y_444_, 4);
v_synthPendingDepth_459_ = lean_ctor_get(v___y_444_, 5);
v_customCanUnfoldPredicate_x3f_460_ = lean_ctor_get(v___y_444_, 6);
v_univApprox_461_ = lean_ctor_get_uint8(v___y_444_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_462_ = lean_ctor_get_uint8(v___y_444_, sizeof(void*)*7 + 2);
v_cacheInferType_463_ = lean_ctor_get_uint8(v___y_444_, sizeof(void*)*7 + 3);
v_isSharedCheck_525_ = !lean_is_exclusive(v___y_444_);
if (v_isSharedCheck_525_ == 0)
{
v___x_465_ = v___y_444_;
v_isShared_466_ = v_isSharedCheck_525_;
goto v_resetjp_464_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_460_);
lean_inc(v_synthPendingDepth_459_);
lean_inc(v_defEqCtx_x3f_458_);
lean_inc(v_localInstances_457_);
lean_inc(v_lctx_456_);
lean_inc(v_zetaDeltaSet_455_);
lean_inc(v_keyedConfig_453_);
lean_dec(v___y_444_);
v___x_465_ = lean_box(0);
v_isShared_466_ = v_isSharedCheck_525_;
goto v_resetjp_464_;
}
v_resetjp_464_:
{
lean_object* v___x_467_; lean_object* v___x_468_; lean_object* v___x_469_; uint8_t v___x_470_; lean_object* v___x_471_; lean_object* v___x_473_; 
v___x_467_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__0___closed__2, &lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__0___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__0___closed__2);
lean_inc(v_a_450_);
v___x_468_ = l_Lean_Expr_app___override(v___x_467_, v_a_450_);
lean_inc(v_a_452_);
v___x_469_ = l_Lean_Expr_app___override(v___x_468_, v_a_452_);
v___x_470_ = 2;
v___x_471_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_470_, v_keyedConfig_453_);
if (v_isShared_466_ == 0)
{
lean_ctor_set(v___x_465_, 0, v___x_471_);
v___x_473_ = v___x_465_;
goto v_reusejp_472_;
}
else
{
lean_object* v_reuseFailAlloc_524_; 
v_reuseFailAlloc_524_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_524_, 0, v___x_471_);
lean_ctor_set(v_reuseFailAlloc_524_, 1, v_zetaDeltaSet_455_);
lean_ctor_set(v_reuseFailAlloc_524_, 2, v_lctx_456_);
lean_ctor_set(v_reuseFailAlloc_524_, 3, v_localInstances_457_);
lean_ctor_set(v_reuseFailAlloc_524_, 4, v_defEqCtx_x3f_458_);
lean_ctor_set(v_reuseFailAlloc_524_, 5, v_synthPendingDepth_459_);
lean_ctor_set(v_reuseFailAlloc_524_, 6, v_customCanUnfoldPredicate_x3f_460_);
lean_ctor_set_uint8(v_reuseFailAlloc_524_, sizeof(void*)*7, v_trackZetaDelta_454_);
lean_ctor_set_uint8(v_reuseFailAlloc_524_, sizeof(void*)*7 + 1, v_univApprox_461_);
lean_ctor_set_uint8(v_reuseFailAlloc_524_, sizeof(void*)*7 + 2, v_inTypeClassResolution_462_);
lean_ctor_set_uint8(v_reuseFailAlloc_524_, sizeof(void*)*7 + 3, v_cacheInferType_463_);
v___x_473_ = v_reuseFailAlloc_524_;
goto v_reusejp_472_;
}
v_reusejp_472_:
{
lean_object* v___x_474_; 
v___x_474_ = l_Lean_Meta_isExprDefEq(v___x_469_, v_hyp_442_, v___x_473_, v___y_445_, v___y_446_, v___y_447_);
lean_dec_ref(v___x_473_);
if (lean_obj_tag(v___x_474_) == 0)
{
lean_object* v_a_475_; lean_object* v___x_477_; uint8_t v_isShared_478_; uint8_t v_isSharedCheck_515_; 
v_a_475_ = lean_ctor_get(v___x_474_, 0);
v_isSharedCheck_515_ = !lean_is_exclusive(v___x_474_);
if (v_isSharedCheck_515_ == 0)
{
v___x_477_ = v___x_474_;
v_isShared_478_ = v_isSharedCheck_515_;
goto v_resetjp_476_;
}
else
{
lean_inc(v_a_475_);
lean_dec(v___x_474_);
v___x_477_ = lean_box(0);
v_isShared_478_ = v_isSharedCheck_515_;
goto v_resetjp_476_;
}
v_resetjp_476_:
{
uint8_t v___x_479_; 
v___x_479_ = lean_unbox(v_a_475_);
if (v___x_479_ == 0)
{
lean_object* v___x_480_; lean_object* v___x_481_; lean_object* v___x_482_; lean_object* v___x_484_; 
lean_dec(v_a_475_);
v___x_480_ = lean_box(v___x_443_);
v___x_481_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_481_, 0, v_a_452_);
lean_ctor_set(v___x_481_, 1, v___x_480_);
v___x_482_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_482_, 0, v_a_450_);
lean_ctor_set(v___x_482_, 1, v___x_481_);
if (v_isShared_478_ == 0)
{
lean_ctor_set(v___x_477_, 0, v___x_482_);
v___x_484_ = v___x_477_;
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
else
{
lean_object* v___x_486_; 
lean_del_object(v___x_477_);
v___x_486_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Bound_isZero_spec__0___redArg(v_a_450_, v___y_445_);
if (lean_obj_tag(v___x_486_) == 0)
{
lean_object* v_a_487_; lean_object* v___x_488_; 
v_a_487_ = lean_ctor_get(v___x_486_, 0);
lean_inc(v_a_487_);
lean_dec_ref_known(v___x_486_, 1);
v___x_488_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Bound_isZero_spec__0___redArg(v_a_452_, v___y_445_);
if (lean_obj_tag(v___x_488_) == 0)
{
lean_object* v_a_489_; lean_object* v___x_491_; uint8_t v_isShared_492_; uint8_t v_isSharedCheck_498_; 
v_a_489_ = lean_ctor_get(v___x_488_, 0);
v_isSharedCheck_498_ = !lean_is_exclusive(v___x_488_);
if (v_isSharedCheck_498_ == 0)
{
v___x_491_ = v___x_488_;
v_isShared_492_ = v_isSharedCheck_498_;
goto v_resetjp_490_;
}
else
{
lean_inc(v_a_489_);
lean_dec(v___x_488_);
v___x_491_ = lean_box(0);
v_isShared_492_ = v_isSharedCheck_498_;
goto v_resetjp_490_;
}
v_resetjp_490_:
{
lean_object* v___x_493_; lean_object* v___x_494_; lean_object* v___x_496_; 
v___x_493_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_493_, 0, v_a_489_);
lean_ctor_set(v___x_493_, 1, v_a_475_);
v___x_494_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_494_, 0, v_a_487_);
lean_ctor_set(v___x_494_, 1, v___x_493_);
if (v_isShared_492_ == 0)
{
lean_ctor_set(v___x_491_, 0, v___x_494_);
v___x_496_ = v___x_491_;
goto v_reusejp_495_;
}
else
{
lean_object* v_reuseFailAlloc_497_; 
v_reuseFailAlloc_497_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_497_, 0, v___x_494_);
v___x_496_ = v_reuseFailAlloc_497_;
goto v_reusejp_495_;
}
v_reusejp_495_:
{
return v___x_496_;
}
}
}
else
{
lean_object* v_a_499_; lean_object* v___x_501_; uint8_t v_isShared_502_; uint8_t v_isSharedCheck_506_; 
lean_dec(v_a_487_);
lean_dec(v_a_475_);
v_a_499_ = lean_ctor_get(v___x_488_, 0);
v_isSharedCheck_506_ = !lean_is_exclusive(v___x_488_);
if (v_isSharedCheck_506_ == 0)
{
v___x_501_ = v___x_488_;
v_isShared_502_ = v_isSharedCheck_506_;
goto v_resetjp_500_;
}
else
{
lean_inc(v_a_499_);
lean_dec(v___x_488_);
v___x_501_ = lean_box(0);
v_isShared_502_ = v_isSharedCheck_506_;
goto v_resetjp_500_;
}
v_resetjp_500_:
{
lean_object* v___x_504_; 
if (v_isShared_502_ == 0)
{
v___x_504_ = v___x_501_;
goto v_reusejp_503_;
}
else
{
lean_object* v_reuseFailAlloc_505_; 
v_reuseFailAlloc_505_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_505_, 0, v_a_499_);
v___x_504_ = v_reuseFailAlloc_505_;
goto v_reusejp_503_;
}
v_reusejp_503_:
{
return v___x_504_;
}
}
}
}
else
{
lean_object* v_a_507_; lean_object* v___x_509_; uint8_t v_isShared_510_; uint8_t v_isSharedCheck_514_; 
lean_dec(v_a_475_);
lean_dec(v_a_452_);
v_a_507_ = lean_ctor_get(v___x_486_, 0);
v_isSharedCheck_514_ = !lean_is_exclusive(v___x_486_);
if (v_isSharedCheck_514_ == 0)
{
v___x_509_ = v___x_486_;
v_isShared_510_ = v_isSharedCheck_514_;
goto v_resetjp_508_;
}
else
{
lean_inc(v_a_507_);
lean_dec(v___x_486_);
v___x_509_ = lean_box(0);
v_isShared_510_ = v_isSharedCheck_514_;
goto v_resetjp_508_;
}
v_resetjp_508_:
{
lean_object* v___x_512_; 
if (v_isShared_510_ == 0)
{
v___x_512_ = v___x_509_;
goto v_reusejp_511_;
}
else
{
lean_object* v_reuseFailAlloc_513_; 
v_reuseFailAlloc_513_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_513_, 0, v_a_507_);
v___x_512_ = v_reuseFailAlloc_513_;
goto v_reusejp_511_;
}
v_reusejp_511_:
{
return v___x_512_;
}
}
}
}
}
}
else
{
lean_object* v_a_516_; lean_object* v___x_518_; uint8_t v_isShared_519_; uint8_t v_isSharedCheck_523_; 
lean_dec(v_a_452_);
lean_dec(v_a_450_);
v_a_516_ = lean_ctor_get(v___x_474_, 0);
v_isSharedCheck_523_ = !lean_is_exclusive(v___x_474_);
if (v_isSharedCheck_523_ == 0)
{
v___x_518_ = v___x_474_;
v_isShared_519_ = v_isSharedCheck_523_;
goto v_resetjp_517_;
}
else
{
lean_inc(v_a_516_);
lean_dec(v___x_474_);
v___x_518_ = lean_box(0);
v_isShared_519_ = v_isSharedCheck_523_;
goto v_resetjp_517_;
}
v_resetjp_517_:
{
lean_object* v___x_521_; 
if (v_isShared_519_ == 0)
{
v___x_521_ = v___x_518_;
goto v_reusejp_520_;
}
else
{
lean_object* v_reuseFailAlloc_522_; 
v_reuseFailAlloc_522_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_522_, 0, v_a_516_);
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
}
else
{
lean_object* v_a_526_; lean_object* v___x_528_; uint8_t v_isShared_529_; uint8_t v_isSharedCheck_533_; 
lean_dec(v_a_450_);
lean_dec_ref(v___y_444_);
lean_dec_ref(v_hyp_442_);
v_a_526_ = lean_ctor_get(v___x_451_, 0);
v_isSharedCheck_533_ = !lean_is_exclusive(v___x_451_);
if (v_isSharedCheck_533_ == 0)
{
v___x_528_ = v___x_451_;
v_isShared_529_ = v_isSharedCheck_533_;
goto v_resetjp_527_;
}
else
{
lean_inc(v_a_526_);
lean_dec(v___x_451_);
v___x_528_ = lean_box(0);
v_isShared_529_ = v_isSharedCheck_533_;
goto v_resetjp_527_;
}
v_resetjp_527_:
{
lean_object* v___x_531_; 
if (v_isShared_529_ == 0)
{
v___x_531_ = v___x_528_;
goto v_reusejp_530_;
}
else
{
lean_object* v_reuseFailAlloc_532_; 
v_reuseFailAlloc_532_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_532_, 0, v_a_526_);
v___x_531_ = v_reuseFailAlloc_532_;
goto v_reusejp_530_;
}
v_reusejp_530_:
{
return v___x_531_;
}
}
}
}
else
{
lean_object* v_a_534_; lean_object* v___x_536_; uint8_t v_isShared_537_; uint8_t v_isSharedCheck_541_; 
lean_dec_ref(v___y_444_);
lean_dec_ref(v_hyp_442_);
lean_dec(v___x_441_);
lean_dec(v___x_439_);
v_a_534_ = lean_ctor_get(v___x_449_, 0);
v_isSharedCheck_541_ = !lean_is_exclusive(v___x_449_);
if (v_isSharedCheck_541_ == 0)
{
v___x_536_ = v___x_449_;
v_isShared_537_ = v_isSharedCheck_541_;
goto v_resetjp_535_;
}
else
{
lean_inc(v_a_534_);
lean_dec(v___x_449_);
v___x_536_ = lean_box(0);
v_isShared_537_ = v_isSharedCheck_541_;
goto v_resetjp_535_;
}
v_resetjp_535_:
{
lean_object* v___x_539_; 
if (v_isShared_537_ == 0)
{
v___x_539_ = v___x_536_;
goto v_reusejp_538_;
}
else
{
lean_object* v_reuseFailAlloc_540_; 
v_reuseFailAlloc_540_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_540_, 0, v_a_534_);
v___x_539_ = v_reuseFailAlloc_540_;
goto v_reusejp_538_;
}
v_reusejp_538_:
{
return v___x_539_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__0___boxed(lean_object* v___x_542_, lean_object* v___x_543_, lean_object* v___x_544_, lean_object* v_hyp_545_, lean_object* v___x_546_, lean_object* v___y_547_, lean_object* v___y_548_, lean_object* v___y_549_, lean_object* v___y_550_, lean_object* v___y_551_){
_start:
{
uint8_t v___x_14134__boxed_552_; uint8_t v___x_14136__boxed_553_; lean_object* v_res_554_; 
v___x_14134__boxed_552_ = lean_unbox(v___x_543_);
v___x_14136__boxed_553_ = lean_unbox(v___x_546_);
v_res_554_ = lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__0(v___x_542_, v___x_14134__boxed_552_, v___x_544_, v_hyp_545_, v___x_14136__boxed_553_, v___y_547_, v___y_548_, v___y_549_, v___y_550_);
lean_dec(v___y_550_);
lean_dec_ref(v___y_549_);
lean_dec(v___y_548_);
return v_res_554_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__1___closed__2(void){
_start:
{
lean_object* v___x_558_; lean_object* v___x_559_; lean_object* v___x_560_; 
v___x_558_ = lean_box(0);
v___x_559_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__1___closed__1));
v___x_560_ = l_Lean_Expr_const___override(v___x_559_, v___x_558_);
return v___x_560_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__1(lean_object* v___x_561_, uint8_t v___x_562_, lean_object* v___x_563_, lean_object* v_hyp_564_, uint8_t v___x_565_, lean_object* v___y_566_, lean_object* v___y_567_, lean_object* v___y_568_, lean_object* v___y_569_){
_start:
{
lean_object* v___x_571_; 
lean_inc(v___x_563_);
lean_inc(v___x_561_);
v___x_571_ = l_Lean_Meta_mkFreshExprMVar(v___x_561_, v___x_562_, v___x_563_, v___y_566_, v___y_567_, v___y_568_, v___y_569_);
if (lean_obj_tag(v___x_571_) == 0)
{
lean_object* v_a_572_; lean_object* v___x_573_; 
v_a_572_ = lean_ctor_get(v___x_571_, 0);
lean_inc(v_a_572_);
lean_dec_ref_known(v___x_571_, 1);
v___x_573_ = l_Lean_Meta_mkFreshExprMVar(v___x_561_, v___x_562_, v___x_563_, v___y_566_, v___y_567_, v___y_568_, v___y_569_);
if (lean_obj_tag(v___x_573_) == 0)
{
lean_object* v_a_574_; lean_object* v_keyedConfig_575_; uint8_t v_trackZetaDelta_576_; lean_object* v_zetaDeltaSet_577_; lean_object* v_lctx_578_; lean_object* v_localInstances_579_; lean_object* v_defEqCtx_x3f_580_; lean_object* v_synthPendingDepth_581_; lean_object* v_customCanUnfoldPredicate_x3f_582_; uint8_t v_univApprox_583_; uint8_t v_inTypeClassResolution_584_; uint8_t v_cacheInferType_585_; lean_object* v___x_587_; uint8_t v_isShared_588_; uint8_t v_isSharedCheck_647_; 
v_a_574_ = lean_ctor_get(v___x_573_, 0);
lean_inc(v_a_574_);
lean_dec_ref_known(v___x_573_, 1);
v_keyedConfig_575_ = lean_ctor_get(v___y_566_, 0);
v_trackZetaDelta_576_ = lean_ctor_get_uint8(v___y_566_, sizeof(void*)*7);
v_zetaDeltaSet_577_ = lean_ctor_get(v___y_566_, 1);
v_lctx_578_ = lean_ctor_get(v___y_566_, 2);
v_localInstances_579_ = lean_ctor_get(v___y_566_, 3);
v_defEqCtx_x3f_580_ = lean_ctor_get(v___y_566_, 4);
v_synthPendingDepth_581_ = lean_ctor_get(v___y_566_, 5);
v_customCanUnfoldPredicate_x3f_582_ = lean_ctor_get(v___y_566_, 6);
v_univApprox_583_ = lean_ctor_get_uint8(v___y_566_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_584_ = lean_ctor_get_uint8(v___y_566_, sizeof(void*)*7 + 2);
v_cacheInferType_585_ = lean_ctor_get_uint8(v___y_566_, sizeof(void*)*7 + 3);
v_isSharedCheck_647_ = !lean_is_exclusive(v___y_566_);
if (v_isSharedCheck_647_ == 0)
{
v___x_587_ = v___y_566_;
v_isShared_588_ = v_isSharedCheck_647_;
goto v_resetjp_586_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_582_);
lean_inc(v_synthPendingDepth_581_);
lean_inc(v_defEqCtx_x3f_580_);
lean_inc(v_localInstances_579_);
lean_inc(v_lctx_578_);
lean_inc(v_zetaDeltaSet_577_);
lean_inc(v_keyedConfig_575_);
lean_dec(v___y_566_);
v___x_587_ = lean_box(0);
v_isShared_588_ = v_isSharedCheck_647_;
goto v_resetjp_586_;
}
v_resetjp_586_:
{
lean_object* v___x_589_; lean_object* v___x_590_; lean_object* v___x_591_; uint8_t v___x_592_; lean_object* v___x_593_; lean_object* v___x_595_; 
v___x_589_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__1___closed__2, &lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__1___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__1___closed__2);
lean_inc(v_a_572_);
v___x_590_ = l_Lean_Expr_app___override(v___x_589_, v_a_572_);
lean_inc(v_a_574_);
v___x_591_ = l_Lean_Expr_app___override(v___x_590_, v_a_574_);
v___x_592_ = 2;
v___x_593_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_592_, v_keyedConfig_575_);
if (v_isShared_588_ == 0)
{
lean_ctor_set(v___x_587_, 0, v___x_593_);
v___x_595_ = v___x_587_;
goto v_reusejp_594_;
}
else
{
lean_object* v_reuseFailAlloc_646_; 
v_reuseFailAlloc_646_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_646_, 0, v___x_593_);
lean_ctor_set(v_reuseFailAlloc_646_, 1, v_zetaDeltaSet_577_);
lean_ctor_set(v_reuseFailAlloc_646_, 2, v_lctx_578_);
lean_ctor_set(v_reuseFailAlloc_646_, 3, v_localInstances_579_);
lean_ctor_set(v_reuseFailAlloc_646_, 4, v_defEqCtx_x3f_580_);
lean_ctor_set(v_reuseFailAlloc_646_, 5, v_synthPendingDepth_581_);
lean_ctor_set(v_reuseFailAlloc_646_, 6, v_customCanUnfoldPredicate_x3f_582_);
lean_ctor_set_uint8(v_reuseFailAlloc_646_, sizeof(void*)*7, v_trackZetaDelta_576_);
lean_ctor_set_uint8(v_reuseFailAlloc_646_, sizeof(void*)*7 + 1, v_univApprox_583_);
lean_ctor_set_uint8(v_reuseFailAlloc_646_, sizeof(void*)*7 + 2, v_inTypeClassResolution_584_);
lean_ctor_set_uint8(v_reuseFailAlloc_646_, sizeof(void*)*7 + 3, v_cacheInferType_585_);
v___x_595_ = v_reuseFailAlloc_646_;
goto v_reusejp_594_;
}
v_reusejp_594_:
{
lean_object* v___x_596_; 
v___x_596_ = l_Lean_Meta_isExprDefEq(v___x_591_, v_hyp_564_, v___x_595_, v___y_567_, v___y_568_, v___y_569_);
lean_dec_ref(v___x_595_);
if (lean_obj_tag(v___x_596_) == 0)
{
lean_object* v_a_597_; lean_object* v___x_599_; uint8_t v_isShared_600_; uint8_t v_isSharedCheck_637_; 
v_a_597_ = lean_ctor_get(v___x_596_, 0);
v_isSharedCheck_637_ = !lean_is_exclusive(v___x_596_);
if (v_isSharedCheck_637_ == 0)
{
v___x_599_ = v___x_596_;
v_isShared_600_ = v_isSharedCheck_637_;
goto v_resetjp_598_;
}
else
{
lean_inc(v_a_597_);
lean_dec(v___x_596_);
v___x_599_ = lean_box(0);
v_isShared_600_ = v_isSharedCheck_637_;
goto v_resetjp_598_;
}
v_resetjp_598_:
{
uint8_t v___x_601_; 
v___x_601_ = lean_unbox(v_a_597_);
if (v___x_601_ == 0)
{
lean_object* v___x_602_; lean_object* v___x_603_; lean_object* v___x_604_; lean_object* v___x_606_; 
lean_dec(v_a_597_);
v___x_602_ = lean_box(v___x_565_);
v___x_603_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_603_, 0, v_a_574_);
lean_ctor_set(v___x_603_, 1, v___x_602_);
v___x_604_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_604_, 0, v_a_572_);
lean_ctor_set(v___x_604_, 1, v___x_603_);
if (v_isShared_600_ == 0)
{
lean_ctor_set(v___x_599_, 0, v___x_604_);
v___x_606_ = v___x_599_;
goto v_reusejp_605_;
}
else
{
lean_object* v_reuseFailAlloc_607_; 
v_reuseFailAlloc_607_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_607_, 0, v___x_604_);
v___x_606_ = v_reuseFailAlloc_607_;
goto v_reusejp_605_;
}
v_reusejp_605_:
{
return v___x_606_;
}
}
else
{
lean_object* v___x_608_; 
lean_del_object(v___x_599_);
v___x_608_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Bound_isZero_spec__0___redArg(v_a_572_, v___y_567_);
if (lean_obj_tag(v___x_608_) == 0)
{
lean_object* v_a_609_; lean_object* v___x_610_; 
v_a_609_ = lean_ctor_get(v___x_608_, 0);
lean_inc(v_a_609_);
lean_dec_ref_known(v___x_608_, 1);
v___x_610_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Bound_isZero_spec__0___redArg(v_a_574_, v___y_567_);
if (lean_obj_tag(v___x_610_) == 0)
{
lean_object* v_a_611_; lean_object* v___x_613_; uint8_t v_isShared_614_; uint8_t v_isSharedCheck_620_; 
v_a_611_ = lean_ctor_get(v___x_610_, 0);
v_isSharedCheck_620_ = !lean_is_exclusive(v___x_610_);
if (v_isSharedCheck_620_ == 0)
{
v___x_613_ = v___x_610_;
v_isShared_614_ = v_isSharedCheck_620_;
goto v_resetjp_612_;
}
else
{
lean_inc(v_a_611_);
lean_dec(v___x_610_);
v___x_613_ = lean_box(0);
v_isShared_614_ = v_isSharedCheck_620_;
goto v_resetjp_612_;
}
v_resetjp_612_:
{
lean_object* v___x_615_; lean_object* v___x_616_; lean_object* v___x_618_; 
v___x_615_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_615_, 0, v_a_611_);
lean_ctor_set(v___x_615_, 1, v_a_597_);
v___x_616_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_616_, 0, v_a_609_);
lean_ctor_set(v___x_616_, 1, v___x_615_);
if (v_isShared_614_ == 0)
{
lean_ctor_set(v___x_613_, 0, v___x_616_);
v___x_618_ = v___x_613_;
goto v_reusejp_617_;
}
else
{
lean_object* v_reuseFailAlloc_619_; 
v_reuseFailAlloc_619_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_619_, 0, v___x_616_);
v___x_618_ = v_reuseFailAlloc_619_;
goto v_reusejp_617_;
}
v_reusejp_617_:
{
return v___x_618_;
}
}
}
else
{
lean_object* v_a_621_; lean_object* v___x_623_; uint8_t v_isShared_624_; uint8_t v_isSharedCheck_628_; 
lean_dec(v_a_609_);
lean_dec(v_a_597_);
v_a_621_ = lean_ctor_get(v___x_610_, 0);
v_isSharedCheck_628_ = !lean_is_exclusive(v___x_610_);
if (v_isSharedCheck_628_ == 0)
{
v___x_623_ = v___x_610_;
v_isShared_624_ = v_isSharedCheck_628_;
goto v_resetjp_622_;
}
else
{
lean_inc(v_a_621_);
lean_dec(v___x_610_);
v___x_623_ = lean_box(0);
v_isShared_624_ = v_isSharedCheck_628_;
goto v_resetjp_622_;
}
v_resetjp_622_:
{
lean_object* v___x_626_; 
if (v_isShared_624_ == 0)
{
v___x_626_ = v___x_623_;
goto v_reusejp_625_;
}
else
{
lean_object* v_reuseFailAlloc_627_; 
v_reuseFailAlloc_627_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_627_, 0, v_a_621_);
v___x_626_ = v_reuseFailAlloc_627_;
goto v_reusejp_625_;
}
v_reusejp_625_:
{
return v___x_626_;
}
}
}
}
else
{
lean_object* v_a_629_; lean_object* v___x_631_; uint8_t v_isShared_632_; uint8_t v_isSharedCheck_636_; 
lean_dec(v_a_597_);
lean_dec(v_a_574_);
v_a_629_ = lean_ctor_get(v___x_608_, 0);
v_isSharedCheck_636_ = !lean_is_exclusive(v___x_608_);
if (v_isSharedCheck_636_ == 0)
{
v___x_631_ = v___x_608_;
v_isShared_632_ = v_isSharedCheck_636_;
goto v_resetjp_630_;
}
else
{
lean_inc(v_a_629_);
lean_dec(v___x_608_);
v___x_631_ = lean_box(0);
v_isShared_632_ = v_isSharedCheck_636_;
goto v_resetjp_630_;
}
v_resetjp_630_:
{
lean_object* v___x_634_; 
if (v_isShared_632_ == 0)
{
v___x_634_ = v___x_631_;
goto v_reusejp_633_;
}
else
{
lean_object* v_reuseFailAlloc_635_; 
v_reuseFailAlloc_635_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_635_, 0, v_a_629_);
v___x_634_ = v_reuseFailAlloc_635_;
goto v_reusejp_633_;
}
v_reusejp_633_:
{
return v___x_634_;
}
}
}
}
}
}
else
{
lean_object* v_a_638_; lean_object* v___x_640_; uint8_t v_isShared_641_; uint8_t v_isSharedCheck_645_; 
lean_dec(v_a_574_);
lean_dec(v_a_572_);
v_a_638_ = lean_ctor_get(v___x_596_, 0);
v_isSharedCheck_645_ = !lean_is_exclusive(v___x_596_);
if (v_isSharedCheck_645_ == 0)
{
v___x_640_ = v___x_596_;
v_isShared_641_ = v_isSharedCheck_645_;
goto v_resetjp_639_;
}
else
{
lean_inc(v_a_638_);
lean_dec(v___x_596_);
v___x_640_ = lean_box(0);
v_isShared_641_ = v_isSharedCheck_645_;
goto v_resetjp_639_;
}
v_resetjp_639_:
{
lean_object* v___x_643_; 
if (v_isShared_641_ == 0)
{
v___x_643_ = v___x_640_;
goto v_reusejp_642_;
}
else
{
lean_object* v_reuseFailAlloc_644_; 
v_reuseFailAlloc_644_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_644_, 0, v_a_638_);
v___x_643_ = v_reuseFailAlloc_644_;
goto v_reusejp_642_;
}
v_reusejp_642_:
{
return v___x_643_;
}
}
}
}
}
}
else
{
lean_object* v_a_648_; lean_object* v___x_650_; uint8_t v_isShared_651_; uint8_t v_isSharedCheck_655_; 
lean_dec(v_a_572_);
lean_dec_ref(v___y_566_);
lean_dec_ref(v_hyp_564_);
v_a_648_ = lean_ctor_get(v___x_573_, 0);
v_isSharedCheck_655_ = !lean_is_exclusive(v___x_573_);
if (v_isSharedCheck_655_ == 0)
{
v___x_650_ = v___x_573_;
v_isShared_651_ = v_isSharedCheck_655_;
goto v_resetjp_649_;
}
else
{
lean_inc(v_a_648_);
lean_dec(v___x_573_);
v___x_650_ = lean_box(0);
v_isShared_651_ = v_isSharedCheck_655_;
goto v_resetjp_649_;
}
v_resetjp_649_:
{
lean_object* v___x_653_; 
if (v_isShared_651_ == 0)
{
v___x_653_ = v___x_650_;
goto v_reusejp_652_;
}
else
{
lean_object* v_reuseFailAlloc_654_; 
v_reuseFailAlloc_654_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_654_, 0, v_a_648_);
v___x_653_ = v_reuseFailAlloc_654_;
goto v_reusejp_652_;
}
v_reusejp_652_:
{
return v___x_653_;
}
}
}
}
else
{
lean_object* v_a_656_; lean_object* v___x_658_; uint8_t v_isShared_659_; uint8_t v_isSharedCheck_663_; 
lean_dec_ref(v___y_566_);
lean_dec_ref(v_hyp_564_);
lean_dec(v___x_563_);
lean_dec(v___x_561_);
v_a_656_ = lean_ctor_get(v___x_571_, 0);
v_isSharedCheck_663_ = !lean_is_exclusive(v___x_571_);
if (v_isSharedCheck_663_ == 0)
{
v___x_658_ = v___x_571_;
v_isShared_659_ = v_isSharedCheck_663_;
goto v_resetjp_657_;
}
else
{
lean_inc(v_a_656_);
lean_dec(v___x_571_);
v___x_658_ = lean_box(0);
v_isShared_659_ = v_isSharedCheck_663_;
goto v_resetjp_657_;
}
v_resetjp_657_:
{
lean_object* v___x_661_; 
if (v_isShared_659_ == 0)
{
v___x_661_ = v___x_658_;
goto v_reusejp_660_;
}
else
{
lean_object* v_reuseFailAlloc_662_; 
v_reuseFailAlloc_662_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_662_, 0, v_a_656_);
v___x_661_ = v_reuseFailAlloc_662_;
goto v_reusejp_660_;
}
v_reusejp_660_:
{
return v___x_661_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__1___boxed(lean_object* v___x_664_, lean_object* v___x_665_, lean_object* v___x_666_, lean_object* v_hyp_667_, lean_object* v___x_668_, lean_object* v___y_669_, lean_object* v___y_670_, lean_object* v___y_671_, lean_object* v___y_672_, lean_object* v___y_673_){
_start:
{
uint8_t v___x_14343__boxed_674_; uint8_t v___x_14345__boxed_675_; lean_object* v_res_676_; 
v___x_14343__boxed_674_ = lean_unbox(v___x_665_);
v___x_14345__boxed_675_ = lean_unbox(v___x_668_);
v_res_676_ = lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__1(v___x_664_, v___x_14343__boxed_674_, v___x_666_, v_hyp_667_, v___x_14345__boxed_675_, v___y_669_, v___y_670_, v___y_671_, v___y_672_);
lean_dec(v___y_672_);
lean_dec_ref(v___y_671_);
lean_dec(v___y_670_);
return v_res_676_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__2(lean_object* v_hyp_684_, uint8_t v___x_685_, lean_object* v___y_686_, lean_object* v___y_687_, lean_object* v___y_688_, lean_object* v___y_689_){
_start:
{
lean_object* v___x_691_; 
v___x_691_ = l_Lean_Meta_mkFreshLevelMVar(v___y_686_, v___y_687_, v___y_688_, v___y_689_);
if (lean_obj_tag(v___x_691_) == 0)
{
lean_object* v_a_692_; lean_object* v___x_693_; lean_object* v___x_694_; lean_object* v___x_695_; uint8_t v___x_696_; lean_object* v___x_697_; lean_object* v___x_698_; 
v_a_692_ = lean_ctor_get(v___x_691_, 0);
lean_inc_n(v_a_692_, 2);
lean_dec_ref_known(v___x_691_, 1);
v___x_693_ = l_Lean_Level_succ___override(v_a_692_);
v___x_694_ = l_Lean_Expr_sort___override(v___x_693_);
v___x_695_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_695_, 0, v___x_694_);
v___x_696_ = 0;
v___x_697_ = lean_box(0);
v___x_698_ = l_Lean_Meta_mkFreshExprMVar(v___x_695_, v___x_696_, v___x_697_, v___y_686_, v___y_687_, v___y_688_, v___y_689_);
if (lean_obj_tag(v___x_698_) == 0)
{
lean_object* v_a_699_; lean_object* v___x_700_; lean_object* v___x_701_; lean_object* v___x_702_; lean_object* v___x_703_; lean_object* v___x_704_; lean_object* v___x_705_; lean_object* v___x_706_; 
v_a_699_ = lean_ctor_get(v___x_698_, 0);
lean_inc_n(v_a_699_, 2);
lean_dec_ref_known(v___x_698_, 1);
v___x_700_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__2___closed__1));
v___x_701_ = lean_box(0);
lean_inc(v_a_692_);
v___x_702_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_702_, 0, v_a_692_);
lean_ctor_set(v___x_702_, 1, v___x_701_);
lean_inc_ref(v___x_702_);
v___x_703_ = l_Lean_Expr_const___override(v___x_700_, v___x_702_);
v___x_704_ = l_Lean_Expr_app___override(v___x_703_, v_a_699_);
v___x_705_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_705_, 0, v___x_704_);
v___x_706_ = l_Lean_Meta_mkFreshExprMVar(v___x_705_, v___x_696_, v___x_697_, v___y_686_, v___y_687_, v___y_688_, v___y_689_);
if (lean_obj_tag(v___x_706_) == 0)
{
lean_object* v_a_707_; lean_object* v___x_708_; lean_object* v___x_709_; 
v_a_707_ = lean_ctor_get(v___x_706_, 0);
lean_inc(v_a_707_);
lean_dec_ref_known(v___x_706_, 1);
lean_inc(v_a_699_);
v___x_708_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_708_, 0, v_a_699_);
lean_inc_ref(v___x_708_);
v___x_709_ = l_Lean_Meta_mkFreshExprMVar(v___x_708_, v___x_696_, v___x_697_, v___y_686_, v___y_687_, v___y_688_, v___y_689_);
if (lean_obj_tag(v___x_709_) == 0)
{
lean_object* v_a_710_; lean_object* v___x_711_; 
v_a_710_ = lean_ctor_get(v___x_709_, 0);
lean_inc(v_a_710_);
lean_dec_ref_known(v___x_709_, 1);
v___x_711_ = l_Lean_Meta_mkFreshExprMVar(v___x_708_, v___x_696_, v___x_697_, v___y_686_, v___y_687_, v___y_688_, v___y_689_);
if (lean_obj_tag(v___x_711_) == 0)
{
lean_object* v_a_712_; lean_object* v_keyedConfig_713_; uint8_t v_trackZetaDelta_714_; lean_object* v_zetaDeltaSet_715_; lean_object* v_lctx_716_; lean_object* v_localInstances_717_; lean_object* v_defEqCtx_x3f_718_; lean_object* v_synthPendingDepth_719_; lean_object* v_customCanUnfoldPredicate_x3f_720_; uint8_t v_univApprox_721_; uint8_t v_inTypeClassResolution_722_; uint8_t v_cacheInferType_723_; lean_object* v___x_725_; uint8_t v_isShared_726_; uint8_t v_isSharedCheck_824_; 
v_a_712_ = lean_ctor_get(v___x_711_, 0);
lean_inc(v_a_712_);
lean_dec_ref_known(v___x_711_, 1);
v_keyedConfig_713_ = lean_ctor_get(v___y_686_, 0);
v_trackZetaDelta_714_ = lean_ctor_get_uint8(v___y_686_, sizeof(void*)*7);
v_zetaDeltaSet_715_ = lean_ctor_get(v___y_686_, 1);
v_lctx_716_ = lean_ctor_get(v___y_686_, 2);
v_localInstances_717_ = lean_ctor_get(v___y_686_, 3);
v_defEqCtx_x3f_718_ = lean_ctor_get(v___y_686_, 4);
v_synthPendingDepth_719_ = lean_ctor_get(v___y_686_, 5);
v_customCanUnfoldPredicate_x3f_720_ = lean_ctor_get(v___y_686_, 6);
v_univApprox_721_ = lean_ctor_get_uint8(v___y_686_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_722_ = lean_ctor_get_uint8(v___y_686_, sizeof(void*)*7 + 2);
v_cacheInferType_723_ = lean_ctor_get_uint8(v___y_686_, sizeof(void*)*7 + 3);
v_isSharedCheck_824_ = !lean_is_exclusive(v___y_686_);
if (v_isSharedCheck_824_ == 0)
{
v___x_725_ = v___y_686_;
v_isShared_726_ = v_isSharedCheck_824_;
goto v_resetjp_724_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_720_);
lean_inc(v_synthPendingDepth_719_);
lean_inc(v_defEqCtx_x3f_718_);
lean_inc(v_localInstances_717_);
lean_inc(v_lctx_716_);
lean_inc(v_zetaDeltaSet_715_);
lean_inc(v_keyedConfig_713_);
lean_dec(v___y_686_);
v___x_725_ = lean_box(0);
v_isShared_726_ = v_isSharedCheck_824_;
goto v_resetjp_724_;
}
v_resetjp_724_:
{
lean_object* v___x_727_; lean_object* v___x_728_; lean_object* v___x_729_; lean_object* v___x_730_; lean_object* v___x_731_; lean_object* v___x_732_; uint8_t v___x_733_; lean_object* v___x_734_; lean_object* v___x_736_; 
v___x_727_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__2___closed__3));
v___x_728_ = l_Lean_Expr_const___override(v___x_727_, v___x_702_);
lean_inc(v_a_699_);
v___x_729_ = l_Lean_Expr_app___override(v___x_728_, v_a_699_);
lean_inc(v_a_707_);
v___x_730_ = l_Lean_Expr_app___override(v___x_729_, v_a_707_);
lean_inc(v_a_710_);
v___x_731_ = l_Lean_Expr_app___override(v___x_730_, v_a_710_);
lean_inc(v_a_712_);
v___x_732_ = l_Lean_Expr_app___override(v___x_731_, v_a_712_);
v___x_733_ = 2;
v___x_734_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_733_, v_keyedConfig_713_);
if (v_isShared_726_ == 0)
{
lean_ctor_set(v___x_725_, 0, v___x_734_);
v___x_736_ = v___x_725_;
goto v_reusejp_735_;
}
else
{
lean_object* v_reuseFailAlloc_823_; 
v_reuseFailAlloc_823_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_823_, 0, v___x_734_);
lean_ctor_set(v_reuseFailAlloc_823_, 1, v_zetaDeltaSet_715_);
lean_ctor_set(v_reuseFailAlloc_823_, 2, v_lctx_716_);
lean_ctor_set(v_reuseFailAlloc_823_, 3, v_localInstances_717_);
lean_ctor_set(v_reuseFailAlloc_823_, 4, v_defEqCtx_x3f_718_);
lean_ctor_set(v_reuseFailAlloc_823_, 5, v_synthPendingDepth_719_);
lean_ctor_set(v_reuseFailAlloc_823_, 6, v_customCanUnfoldPredicate_x3f_720_);
lean_ctor_set_uint8(v_reuseFailAlloc_823_, sizeof(void*)*7, v_trackZetaDelta_714_);
lean_ctor_set_uint8(v_reuseFailAlloc_823_, sizeof(void*)*7 + 1, v_univApprox_721_);
lean_ctor_set_uint8(v_reuseFailAlloc_823_, sizeof(void*)*7 + 2, v_inTypeClassResolution_722_);
lean_ctor_set_uint8(v_reuseFailAlloc_823_, sizeof(void*)*7 + 3, v_cacheInferType_723_);
v___x_736_ = v_reuseFailAlloc_823_;
goto v_reusejp_735_;
}
v_reusejp_735_:
{
lean_object* v___x_737_; 
v___x_737_ = l_Lean_Meta_isExprDefEq(v___x_732_, v_hyp_684_, v___x_736_, v___y_687_, v___y_688_, v___y_689_);
lean_dec_ref(v___x_736_);
if (lean_obj_tag(v___x_737_) == 0)
{
lean_object* v_a_738_; lean_object* v___x_740_; uint8_t v_isShared_741_; uint8_t v_isSharedCheck_814_; 
v_a_738_ = lean_ctor_get(v___x_737_, 0);
v_isSharedCheck_814_ = !lean_is_exclusive(v___x_737_);
if (v_isSharedCheck_814_ == 0)
{
v___x_740_ = v___x_737_;
v_isShared_741_ = v_isSharedCheck_814_;
goto v_resetjp_739_;
}
else
{
lean_inc(v_a_738_);
lean_dec(v___x_737_);
v___x_740_ = lean_box(0);
v_isShared_741_ = v_isSharedCheck_814_;
goto v_resetjp_739_;
}
v_resetjp_739_:
{
uint8_t v___x_742_; 
v___x_742_ = lean_unbox(v_a_738_);
if (v___x_742_ == 0)
{
lean_object* v___x_743_; lean_object* v___x_744_; lean_object* v___x_745_; lean_object* v___x_746_; lean_object* v___x_747_; lean_object* v___x_748_; lean_object* v___x_750_; 
lean_dec(v_a_738_);
v___x_743_ = lean_box(v___x_685_);
v___x_744_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_744_, 0, v_a_712_);
lean_ctor_set(v___x_744_, 1, v___x_743_);
v___x_745_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_745_, 0, v_a_710_);
lean_ctor_set(v___x_745_, 1, v___x_744_);
v___x_746_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_746_, 0, v_a_707_);
lean_ctor_set(v___x_746_, 1, v___x_745_);
v___x_747_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_747_, 0, v_a_699_);
lean_ctor_set(v___x_747_, 1, v___x_746_);
v___x_748_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_748_, 0, v_a_692_);
lean_ctor_set(v___x_748_, 1, v___x_747_);
if (v_isShared_741_ == 0)
{
lean_ctor_set(v___x_740_, 0, v___x_748_);
v___x_750_ = v___x_740_;
goto v_reusejp_749_;
}
else
{
lean_object* v_reuseFailAlloc_751_; 
v_reuseFailAlloc_751_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_751_, 0, v___x_748_);
v___x_750_ = v_reuseFailAlloc_751_;
goto v_reusejp_749_;
}
v_reusejp_749_:
{
return v___x_750_;
}
}
else
{
lean_object* v___x_752_; 
lean_del_object(v___x_740_);
v___x_752_ = lp_mathlib_Lean_instantiateLevelMVars___at___00Mathlib_Tactic_Bound_hypPriority_spec__0___redArg(v_a_692_, v___y_687_);
if (lean_obj_tag(v___x_752_) == 0)
{
lean_object* v_a_753_; lean_object* v___x_754_; 
v_a_753_ = lean_ctor_get(v___x_752_, 0);
lean_inc(v_a_753_);
lean_dec_ref_known(v___x_752_, 1);
v___x_754_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Bound_isZero_spec__0___redArg(v_a_699_, v___y_687_);
if (lean_obj_tag(v___x_754_) == 0)
{
lean_object* v_a_755_; lean_object* v___x_756_; 
v_a_755_ = lean_ctor_get(v___x_754_, 0);
lean_inc(v_a_755_);
lean_dec_ref_known(v___x_754_, 1);
v___x_756_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Bound_isZero_spec__0___redArg(v_a_707_, v___y_687_);
if (lean_obj_tag(v___x_756_) == 0)
{
lean_object* v_a_757_; lean_object* v___x_758_; 
v_a_757_ = lean_ctor_get(v___x_756_, 0);
lean_inc(v_a_757_);
lean_dec_ref_known(v___x_756_, 1);
v___x_758_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Bound_isZero_spec__0___redArg(v_a_710_, v___y_687_);
if (lean_obj_tag(v___x_758_) == 0)
{
lean_object* v_a_759_; lean_object* v___x_760_; 
v_a_759_ = lean_ctor_get(v___x_758_, 0);
lean_inc(v_a_759_);
lean_dec_ref_known(v___x_758_, 1);
v___x_760_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Bound_isZero_spec__0___redArg(v_a_712_, v___y_687_);
if (lean_obj_tag(v___x_760_) == 0)
{
lean_object* v_a_761_; lean_object* v___x_763_; uint8_t v_isShared_764_; uint8_t v_isSharedCheck_773_; 
v_a_761_ = lean_ctor_get(v___x_760_, 0);
v_isSharedCheck_773_ = !lean_is_exclusive(v___x_760_);
if (v_isSharedCheck_773_ == 0)
{
v___x_763_ = v___x_760_;
v_isShared_764_ = v_isSharedCheck_773_;
goto v_resetjp_762_;
}
else
{
lean_inc(v_a_761_);
lean_dec(v___x_760_);
v___x_763_ = lean_box(0);
v_isShared_764_ = v_isSharedCheck_773_;
goto v_resetjp_762_;
}
v_resetjp_762_:
{
lean_object* v___x_765_; lean_object* v___x_766_; lean_object* v___x_767_; lean_object* v___x_768_; lean_object* v___x_769_; lean_object* v___x_771_; 
v___x_765_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_765_, 0, v_a_761_);
lean_ctor_set(v___x_765_, 1, v_a_738_);
v___x_766_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_766_, 0, v_a_759_);
lean_ctor_set(v___x_766_, 1, v___x_765_);
v___x_767_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_767_, 0, v_a_757_);
lean_ctor_set(v___x_767_, 1, v___x_766_);
v___x_768_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_768_, 0, v_a_755_);
lean_ctor_set(v___x_768_, 1, v___x_767_);
v___x_769_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_769_, 0, v_a_753_);
lean_ctor_set(v___x_769_, 1, v___x_768_);
if (v_isShared_764_ == 0)
{
lean_ctor_set(v___x_763_, 0, v___x_769_);
v___x_771_ = v___x_763_;
goto v_reusejp_770_;
}
else
{
lean_object* v_reuseFailAlloc_772_; 
v_reuseFailAlloc_772_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_772_, 0, v___x_769_);
v___x_771_ = v_reuseFailAlloc_772_;
goto v_reusejp_770_;
}
v_reusejp_770_:
{
return v___x_771_;
}
}
}
else
{
lean_object* v_a_774_; lean_object* v___x_776_; uint8_t v_isShared_777_; uint8_t v_isSharedCheck_781_; 
lean_dec(v_a_759_);
lean_dec(v_a_757_);
lean_dec(v_a_755_);
lean_dec(v_a_753_);
lean_dec(v_a_738_);
v_a_774_ = lean_ctor_get(v___x_760_, 0);
v_isSharedCheck_781_ = !lean_is_exclusive(v___x_760_);
if (v_isSharedCheck_781_ == 0)
{
v___x_776_ = v___x_760_;
v_isShared_777_ = v_isSharedCheck_781_;
goto v_resetjp_775_;
}
else
{
lean_inc(v_a_774_);
lean_dec(v___x_760_);
v___x_776_ = lean_box(0);
v_isShared_777_ = v_isSharedCheck_781_;
goto v_resetjp_775_;
}
v_resetjp_775_:
{
lean_object* v___x_779_; 
if (v_isShared_777_ == 0)
{
v___x_779_ = v___x_776_;
goto v_reusejp_778_;
}
else
{
lean_object* v_reuseFailAlloc_780_; 
v_reuseFailAlloc_780_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_780_, 0, v_a_774_);
v___x_779_ = v_reuseFailAlloc_780_;
goto v_reusejp_778_;
}
v_reusejp_778_:
{
return v___x_779_;
}
}
}
}
else
{
lean_object* v_a_782_; lean_object* v___x_784_; uint8_t v_isShared_785_; uint8_t v_isSharedCheck_789_; 
lean_dec(v_a_757_);
lean_dec(v_a_755_);
lean_dec(v_a_753_);
lean_dec(v_a_738_);
lean_dec(v_a_712_);
v_a_782_ = lean_ctor_get(v___x_758_, 0);
v_isSharedCheck_789_ = !lean_is_exclusive(v___x_758_);
if (v_isSharedCheck_789_ == 0)
{
v___x_784_ = v___x_758_;
v_isShared_785_ = v_isSharedCheck_789_;
goto v_resetjp_783_;
}
else
{
lean_inc(v_a_782_);
lean_dec(v___x_758_);
v___x_784_ = lean_box(0);
v_isShared_785_ = v_isSharedCheck_789_;
goto v_resetjp_783_;
}
v_resetjp_783_:
{
lean_object* v___x_787_; 
if (v_isShared_785_ == 0)
{
v___x_787_ = v___x_784_;
goto v_reusejp_786_;
}
else
{
lean_object* v_reuseFailAlloc_788_; 
v_reuseFailAlloc_788_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_788_, 0, v_a_782_);
v___x_787_ = v_reuseFailAlloc_788_;
goto v_reusejp_786_;
}
v_reusejp_786_:
{
return v___x_787_;
}
}
}
}
else
{
lean_object* v_a_790_; lean_object* v___x_792_; uint8_t v_isShared_793_; uint8_t v_isSharedCheck_797_; 
lean_dec(v_a_755_);
lean_dec(v_a_753_);
lean_dec(v_a_738_);
lean_dec(v_a_712_);
lean_dec(v_a_710_);
v_a_790_ = lean_ctor_get(v___x_756_, 0);
v_isSharedCheck_797_ = !lean_is_exclusive(v___x_756_);
if (v_isSharedCheck_797_ == 0)
{
v___x_792_ = v___x_756_;
v_isShared_793_ = v_isSharedCheck_797_;
goto v_resetjp_791_;
}
else
{
lean_inc(v_a_790_);
lean_dec(v___x_756_);
v___x_792_ = lean_box(0);
v_isShared_793_ = v_isSharedCheck_797_;
goto v_resetjp_791_;
}
v_resetjp_791_:
{
lean_object* v___x_795_; 
if (v_isShared_793_ == 0)
{
v___x_795_ = v___x_792_;
goto v_reusejp_794_;
}
else
{
lean_object* v_reuseFailAlloc_796_; 
v_reuseFailAlloc_796_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_796_, 0, v_a_790_);
v___x_795_ = v_reuseFailAlloc_796_;
goto v_reusejp_794_;
}
v_reusejp_794_:
{
return v___x_795_;
}
}
}
}
else
{
lean_object* v_a_798_; lean_object* v___x_800_; uint8_t v_isShared_801_; uint8_t v_isSharedCheck_805_; 
lean_dec(v_a_753_);
lean_dec(v_a_738_);
lean_dec(v_a_712_);
lean_dec(v_a_710_);
lean_dec(v_a_707_);
v_a_798_ = lean_ctor_get(v___x_754_, 0);
v_isSharedCheck_805_ = !lean_is_exclusive(v___x_754_);
if (v_isSharedCheck_805_ == 0)
{
v___x_800_ = v___x_754_;
v_isShared_801_ = v_isSharedCheck_805_;
goto v_resetjp_799_;
}
else
{
lean_inc(v_a_798_);
lean_dec(v___x_754_);
v___x_800_ = lean_box(0);
v_isShared_801_ = v_isSharedCheck_805_;
goto v_resetjp_799_;
}
v_resetjp_799_:
{
lean_object* v___x_803_; 
if (v_isShared_801_ == 0)
{
v___x_803_ = v___x_800_;
goto v_reusejp_802_;
}
else
{
lean_object* v_reuseFailAlloc_804_; 
v_reuseFailAlloc_804_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_804_, 0, v_a_798_);
v___x_803_ = v_reuseFailAlloc_804_;
goto v_reusejp_802_;
}
v_reusejp_802_:
{
return v___x_803_;
}
}
}
}
else
{
lean_object* v_a_806_; lean_object* v___x_808_; uint8_t v_isShared_809_; uint8_t v_isSharedCheck_813_; 
lean_dec(v_a_738_);
lean_dec(v_a_712_);
lean_dec(v_a_710_);
lean_dec(v_a_707_);
lean_dec(v_a_699_);
v_a_806_ = lean_ctor_get(v___x_752_, 0);
v_isSharedCheck_813_ = !lean_is_exclusive(v___x_752_);
if (v_isSharedCheck_813_ == 0)
{
v___x_808_ = v___x_752_;
v_isShared_809_ = v_isSharedCheck_813_;
goto v_resetjp_807_;
}
else
{
lean_inc(v_a_806_);
lean_dec(v___x_752_);
v___x_808_ = lean_box(0);
v_isShared_809_ = v_isSharedCheck_813_;
goto v_resetjp_807_;
}
v_resetjp_807_:
{
lean_object* v___x_811_; 
if (v_isShared_809_ == 0)
{
v___x_811_ = v___x_808_;
goto v_reusejp_810_;
}
else
{
lean_object* v_reuseFailAlloc_812_; 
v_reuseFailAlloc_812_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_812_, 0, v_a_806_);
v___x_811_ = v_reuseFailAlloc_812_;
goto v_reusejp_810_;
}
v_reusejp_810_:
{
return v___x_811_;
}
}
}
}
}
}
else
{
lean_object* v_a_815_; lean_object* v___x_817_; uint8_t v_isShared_818_; uint8_t v_isSharedCheck_822_; 
lean_dec(v_a_712_);
lean_dec(v_a_710_);
lean_dec(v_a_707_);
lean_dec(v_a_699_);
lean_dec(v_a_692_);
v_a_815_ = lean_ctor_get(v___x_737_, 0);
v_isSharedCheck_822_ = !lean_is_exclusive(v___x_737_);
if (v_isSharedCheck_822_ == 0)
{
v___x_817_ = v___x_737_;
v_isShared_818_ = v_isSharedCheck_822_;
goto v_resetjp_816_;
}
else
{
lean_inc(v_a_815_);
lean_dec(v___x_737_);
v___x_817_ = lean_box(0);
v_isShared_818_ = v_isSharedCheck_822_;
goto v_resetjp_816_;
}
v_resetjp_816_:
{
lean_object* v___x_820_; 
if (v_isShared_818_ == 0)
{
v___x_820_ = v___x_817_;
goto v_reusejp_819_;
}
else
{
lean_object* v_reuseFailAlloc_821_; 
v_reuseFailAlloc_821_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_821_, 0, v_a_815_);
v___x_820_ = v_reuseFailAlloc_821_;
goto v_reusejp_819_;
}
v_reusejp_819_:
{
return v___x_820_;
}
}
}
}
}
}
else
{
lean_object* v_a_825_; lean_object* v___x_827_; uint8_t v_isShared_828_; uint8_t v_isSharedCheck_832_; 
lean_dec(v_a_710_);
lean_dec(v_a_707_);
lean_dec_ref_known(v___x_702_, 2);
lean_dec(v_a_699_);
lean_dec(v_a_692_);
lean_dec_ref(v___y_686_);
lean_dec_ref(v_hyp_684_);
v_a_825_ = lean_ctor_get(v___x_711_, 0);
v_isSharedCheck_832_ = !lean_is_exclusive(v___x_711_);
if (v_isSharedCheck_832_ == 0)
{
v___x_827_ = v___x_711_;
v_isShared_828_ = v_isSharedCheck_832_;
goto v_resetjp_826_;
}
else
{
lean_inc(v_a_825_);
lean_dec(v___x_711_);
v___x_827_ = lean_box(0);
v_isShared_828_ = v_isSharedCheck_832_;
goto v_resetjp_826_;
}
v_resetjp_826_:
{
lean_object* v___x_830_; 
if (v_isShared_828_ == 0)
{
v___x_830_ = v___x_827_;
goto v_reusejp_829_;
}
else
{
lean_object* v_reuseFailAlloc_831_; 
v_reuseFailAlloc_831_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_831_, 0, v_a_825_);
v___x_830_ = v_reuseFailAlloc_831_;
goto v_reusejp_829_;
}
v_reusejp_829_:
{
return v___x_830_;
}
}
}
}
else
{
lean_object* v_a_833_; lean_object* v___x_835_; uint8_t v_isShared_836_; uint8_t v_isSharedCheck_840_; 
lean_dec_ref_known(v___x_708_, 1);
lean_dec(v_a_707_);
lean_dec_ref_known(v___x_702_, 2);
lean_dec(v_a_699_);
lean_dec(v_a_692_);
lean_dec_ref(v___y_686_);
lean_dec_ref(v_hyp_684_);
v_a_833_ = lean_ctor_get(v___x_709_, 0);
v_isSharedCheck_840_ = !lean_is_exclusive(v___x_709_);
if (v_isSharedCheck_840_ == 0)
{
v___x_835_ = v___x_709_;
v_isShared_836_ = v_isSharedCheck_840_;
goto v_resetjp_834_;
}
else
{
lean_inc(v_a_833_);
lean_dec(v___x_709_);
v___x_835_ = lean_box(0);
v_isShared_836_ = v_isSharedCheck_840_;
goto v_resetjp_834_;
}
v_resetjp_834_:
{
lean_object* v___x_838_; 
if (v_isShared_836_ == 0)
{
v___x_838_ = v___x_835_;
goto v_reusejp_837_;
}
else
{
lean_object* v_reuseFailAlloc_839_; 
v_reuseFailAlloc_839_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_839_, 0, v_a_833_);
v___x_838_ = v_reuseFailAlloc_839_;
goto v_reusejp_837_;
}
v_reusejp_837_:
{
return v___x_838_;
}
}
}
}
else
{
lean_object* v_a_841_; lean_object* v___x_843_; uint8_t v_isShared_844_; uint8_t v_isSharedCheck_848_; 
lean_dec_ref_known(v___x_702_, 2);
lean_dec(v_a_699_);
lean_dec(v_a_692_);
lean_dec_ref(v___y_686_);
lean_dec_ref(v_hyp_684_);
v_a_841_ = lean_ctor_get(v___x_706_, 0);
v_isSharedCheck_848_ = !lean_is_exclusive(v___x_706_);
if (v_isSharedCheck_848_ == 0)
{
v___x_843_ = v___x_706_;
v_isShared_844_ = v_isSharedCheck_848_;
goto v_resetjp_842_;
}
else
{
lean_inc(v_a_841_);
lean_dec(v___x_706_);
v___x_843_ = lean_box(0);
v_isShared_844_ = v_isSharedCheck_848_;
goto v_resetjp_842_;
}
v_resetjp_842_:
{
lean_object* v___x_846_; 
if (v_isShared_844_ == 0)
{
v___x_846_ = v___x_843_;
goto v_reusejp_845_;
}
else
{
lean_object* v_reuseFailAlloc_847_; 
v_reuseFailAlloc_847_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_847_, 0, v_a_841_);
v___x_846_ = v_reuseFailAlloc_847_;
goto v_reusejp_845_;
}
v_reusejp_845_:
{
return v___x_846_;
}
}
}
}
else
{
lean_object* v_a_849_; lean_object* v___x_851_; uint8_t v_isShared_852_; uint8_t v_isSharedCheck_856_; 
lean_dec(v_a_692_);
lean_dec_ref(v___y_686_);
lean_dec_ref(v_hyp_684_);
v_a_849_ = lean_ctor_get(v___x_698_, 0);
v_isSharedCheck_856_ = !lean_is_exclusive(v___x_698_);
if (v_isSharedCheck_856_ == 0)
{
v___x_851_ = v___x_698_;
v_isShared_852_ = v_isSharedCheck_856_;
goto v_resetjp_850_;
}
else
{
lean_inc(v_a_849_);
lean_dec(v___x_698_);
v___x_851_ = lean_box(0);
v_isShared_852_ = v_isSharedCheck_856_;
goto v_resetjp_850_;
}
v_resetjp_850_:
{
lean_object* v___x_854_; 
if (v_isShared_852_ == 0)
{
v___x_854_ = v___x_851_;
goto v_reusejp_853_;
}
else
{
lean_object* v_reuseFailAlloc_855_; 
v_reuseFailAlloc_855_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_855_, 0, v_a_849_);
v___x_854_ = v_reuseFailAlloc_855_;
goto v_reusejp_853_;
}
v_reusejp_853_:
{
return v___x_854_;
}
}
}
}
else
{
lean_object* v_a_857_; lean_object* v___x_859_; uint8_t v_isShared_860_; uint8_t v_isSharedCheck_864_; 
lean_dec_ref(v___y_686_);
lean_dec_ref(v_hyp_684_);
v_a_857_ = lean_ctor_get(v___x_691_, 0);
v_isSharedCheck_864_ = !lean_is_exclusive(v___x_691_);
if (v_isSharedCheck_864_ == 0)
{
v___x_859_ = v___x_691_;
v_isShared_860_ = v_isSharedCheck_864_;
goto v_resetjp_858_;
}
else
{
lean_inc(v_a_857_);
lean_dec(v___x_691_);
v___x_859_ = lean_box(0);
v_isShared_860_ = v_isSharedCheck_864_;
goto v_resetjp_858_;
}
v_resetjp_858_:
{
lean_object* v___x_862_; 
if (v_isShared_860_ == 0)
{
v___x_862_ = v___x_859_;
goto v_reusejp_861_;
}
else
{
lean_object* v_reuseFailAlloc_863_; 
v_reuseFailAlloc_863_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_863_, 0, v_a_857_);
v___x_862_ = v_reuseFailAlloc_863_;
goto v_reusejp_861_;
}
v_reusejp_861_:
{
return v___x_862_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__2___boxed(lean_object* v_hyp_865_, lean_object* v___x_866_, lean_object* v___y_867_, lean_object* v___y_868_, lean_object* v___y_869_, lean_object* v___y_870_, lean_object* v___y_871_){
_start:
{
uint8_t v___x_14553__boxed_872_; lean_object* v_res_873_; 
v___x_14553__boxed_872_ = lean_unbox(v___x_866_);
v_res_873_ = lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__2(v_hyp_865_, v___x_14553__boxed_872_, v___y_867_, v___y_868_, v___y_869_, v___y_870_);
lean_dec(v___y_870_);
lean_dec_ref(v___y_869_);
lean_dec(v___y_868_);
return v_res_873_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__3(lean_object* v_hyp_881_, uint8_t v___x_882_, lean_object* v___y_883_, lean_object* v___y_884_, lean_object* v___y_885_, lean_object* v___y_886_){
_start:
{
lean_object* v___x_888_; 
v___x_888_ = l_Lean_Meta_mkFreshLevelMVar(v___y_883_, v___y_884_, v___y_885_, v___y_886_);
if (lean_obj_tag(v___x_888_) == 0)
{
lean_object* v_a_889_; lean_object* v___x_890_; lean_object* v___x_891_; lean_object* v___x_892_; uint8_t v___x_893_; lean_object* v___x_894_; lean_object* v___x_895_; 
v_a_889_ = lean_ctor_get(v___x_888_, 0);
lean_inc_n(v_a_889_, 2);
lean_dec_ref_known(v___x_888_, 1);
v___x_890_ = l_Lean_Level_succ___override(v_a_889_);
v___x_891_ = l_Lean_Expr_sort___override(v___x_890_);
v___x_892_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_892_, 0, v___x_891_);
v___x_893_ = 0;
v___x_894_ = lean_box(0);
v___x_895_ = l_Lean_Meta_mkFreshExprMVar(v___x_892_, v___x_893_, v___x_894_, v___y_883_, v___y_884_, v___y_885_, v___y_886_);
if (lean_obj_tag(v___x_895_) == 0)
{
lean_object* v_a_896_; lean_object* v___x_897_; lean_object* v___x_898_; lean_object* v___x_899_; lean_object* v___x_900_; lean_object* v___x_901_; lean_object* v___x_902_; lean_object* v___x_903_; 
v_a_896_ = lean_ctor_get(v___x_895_, 0);
lean_inc_n(v_a_896_, 2);
lean_dec_ref_known(v___x_895_, 1);
v___x_897_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__3___closed__1));
v___x_898_ = lean_box(0);
lean_inc(v_a_889_);
v___x_899_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_899_, 0, v_a_889_);
lean_ctor_set(v___x_899_, 1, v___x_898_);
lean_inc_ref(v___x_899_);
v___x_900_ = l_Lean_Expr_const___override(v___x_897_, v___x_899_);
v___x_901_ = l_Lean_Expr_app___override(v___x_900_, v_a_896_);
v___x_902_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_902_, 0, v___x_901_);
v___x_903_ = l_Lean_Meta_mkFreshExprMVar(v___x_902_, v___x_893_, v___x_894_, v___y_883_, v___y_884_, v___y_885_, v___y_886_);
if (lean_obj_tag(v___x_903_) == 0)
{
lean_object* v_a_904_; lean_object* v___x_905_; lean_object* v___x_906_; 
v_a_904_ = lean_ctor_get(v___x_903_, 0);
lean_inc(v_a_904_);
lean_dec_ref_known(v___x_903_, 1);
lean_inc(v_a_896_);
v___x_905_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_905_, 0, v_a_896_);
lean_inc_ref(v___x_905_);
v___x_906_ = l_Lean_Meta_mkFreshExprMVar(v___x_905_, v___x_893_, v___x_894_, v___y_883_, v___y_884_, v___y_885_, v___y_886_);
if (lean_obj_tag(v___x_906_) == 0)
{
lean_object* v_a_907_; lean_object* v___x_908_; 
v_a_907_ = lean_ctor_get(v___x_906_, 0);
lean_inc(v_a_907_);
lean_dec_ref_known(v___x_906_, 1);
v___x_908_ = l_Lean_Meta_mkFreshExprMVar(v___x_905_, v___x_893_, v___x_894_, v___y_883_, v___y_884_, v___y_885_, v___y_886_);
if (lean_obj_tag(v___x_908_) == 0)
{
lean_object* v_a_909_; lean_object* v_keyedConfig_910_; uint8_t v_trackZetaDelta_911_; lean_object* v_zetaDeltaSet_912_; lean_object* v_lctx_913_; lean_object* v_localInstances_914_; lean_object* v_defEqCtx_x3f_915_; lean_object* v_synthPendingDepth_916_; lean_object* v_customCanUnfoldPredicate_x3f_917_; uint8_t v_univApprox_918_; uint8_t v_inTypeClassResolution_919_; uint8_t v_cacheInferType_920_; lean_object* v___x_922_; uint8_t v_isShared_923_; uint8_t v_isSharedCheck_1021_; 
v_a_909_ = lean_ctor_get(v___x_908_, 0);
lean_inc(v_a_909_);
lean_dec_ref_known(v___x_908_, 1);
v_keyedConfig_910_ = lean_ctor_get(v___y_883_, 0);
v_trackZetaDelta_911_ = lean_ctor_get_uint8(v___y_883_, sizeof(void*)*7);
v_zetaDeltaSet_912_ = lean_ctor_get(v___y_883_, 1);
v_lctx_913_ = lean_ctor_get(v___y_883_, 2);
v_localInstances_914_ = lean_ctor_get(v___y_883_, 3);
v_defEqCtx_x3f_915_ = lean_ctor_get(v___y_883_, 4);
v_synthPendingDepth_916_ = lean_ctor_get(v___y_883_, 5);
v_customCanUnfoldPredicate_x3f_917_ = lean_ctor_get(v___y_883_, 6);
v_univApprox_918_ = lean_ctor_get_uint8(v___y_883_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_919_ = lean_ctor_get_uint8(v___y_883_, sizeof(void*)*7 + 2);
v_cacheInferType_920_ = lean_ctor_get_uint8(v___y_883_, sizeof(void*)*7 + 3);
v_isSharedCheck_1021_ = !lean_is_exclusive(v___y_883_);
if (v_isSharedCheck_1021_ == 0)
{
v___x_922_ = v___y_883_;
v_isShared_923_ = v_isSharedCheck_1021_;
goto v_resetjp_921_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_917_);
lean_inc(v_synthPendingDepth_916_);
lean_inc(v_defEqCtx_x3f_915_);
lean_inc(v_localInstances_914_);
lean_inc(v_lctx_913_);
lean_inc(v_zetaDeltaSet_912_);
lean_inc(v_keyedConfig_910_);
lean_dec(v___y_883_);
v___x_922_ = lean_box(0);
v_isShared_923_ = v_isSharedCheck_1021_;
goto v_resetjp_921_;
}
v_resetjp_921_:
{
lean_object* v___x_924_; lean_object* v___x_925_; lean_object* v___x_926_; lean_object* v___x_927_; lean_object* v___x_928_; lean_object* v___x_929_; uint8_t v___x_930_; lean_object* v___x_931_; lean_object* v___x_933_; 
v___x_924_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__3___closed__3));
v___x_925_ = l_Lean_Expr_const___override(v___x_924_, v___x_899_);
lean_inc(v_a_896_);
v___x_926_ = l_Lean_Expr_app___override(v___x_925_, v_a_896_);
lean_inc(v_a_904_);
v___x_927_ = l_Lean_Expr_app___override(v___x_926_, v_a_904_);
lean_inc(v_a_907_);
v___x_928_ = l_Lean_Expr_app___override(v___x_927_, v_a_907_);
lean_inc(v_a_909_);
v___x_929_ = l_Lean_Expr_app___override(v___x_928_, v_a_909_);
v___x_930_ = 2;
v___x_931_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_930_, v_keyedConfig_910_);
if (v_isShared_923_ == 0)
{
lean_ctor_set(v___x_922_, 0, v___x_931_);
v___x_933_ = v___x_922_;
goto v_reusejp_932_;
}
else
{
lean_object* v_reuseFailAlloc_1020_; 
v_reuseFailAlloc_1020_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_1020_, 0, v___x_931_);
lean_ctor_set(v_reuseFailAlloc_1020_, 1, v_zetaDeltaSet_912_);
lean_ctor_set(v_reuseFailAlloc_1020_, 2, v_lctx_913_);
lean_ctor_set(v_reuseFailAlloc_1020_, 3, v_localInstances_914_);
lean_ctor_set(v_reuseFailAlloc_1020_, 4, v_defEqCtx_x3f_915_);
lean_ctor_set(v_reuseFailAlloc_1020_, 5, v_synthPendingDepth_916_);
lean_ctor_set(v_reuseFailAlloc_1020_, 6, v_customCanUnfoldPredicate_x3f_917_);
lean_ctor_set_uint8(v_reuseFailAlloc_1020_, sizeof(void*)*7, v_trackZetaDelta_911_);
lean_ctor_set_uint8(v_reuseFailAlloc_1020_, sizeof(void*)*7 + 1, v_univApprox_918_);
lean_ctor_set_uint8(v_reuseFailAlloc_1020_, sizeof(void*)*7 + 2, v_inTypeClassResolution_919_);
lean_ctor_set_uint8(v_reuseFailAlloc_1020_, sizeof(void*)*7 + 3, v_cacheInferType_920_);
v___x_933_ = v_reuseFailAlloc_1020_;
goto v_reusejp_932_;
}
v_reusejp_932_:
{
lean_object* v___x_934_; 
v___x_934_ = l_Lean_Meta_isExprDefEq(v___x_929_, v_hyp_881_, v___x_933_, v___y_884_, v___y_885_, v___y_886_);
lean_dec_ref(v___x_933_);
if (lean_obj_tag(v___x_934_) == 0)
{
lean_object* v_a_935_; lean_object* v___x_937_; uint8_t v_isShared_938_; uint8_t v_isSharedCheck_1011_; 
v_a_935_ = lean_ctor_get(v___x_934_, 0);
v_isSharedCheck_1011_ = !lean_is_exclusive(v___x_934_);
if (v_isSharedCheck_1011_ == 0)
{
v___x_937_ = v___x_934_;
v_isShared_938_ = v_isSharedCheck_1011_;
goto v_resetjp_936_;
}
else
{
lean_inc(v_a_935_);
lean_dec(v___x_934_);
v___x_937_ = lean_box(0);
v_isShared_938_ = v_isSharedCheck_1011_;
goto v_resetjp_936_;
}
v_resetjp_936_:
{
uint8_t v___x_939_; 
v___x_939_ = lean_unbox(v_a_935_);
if (v___x_939_ == 0)
{
lean_object* v___x_940_; lean_object* v___x_941_; lean_object* v___x_942_; lean_object* v___x_943_; lean_object* v___x_944_; lean_object* v___x_945_; lean_object* v___x_947_; 
lean_dec(v_a_935_);
v___x_940_ = lean_box(v___x_882_);
v___x_941_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_941_, 0, v_a_909_);
lean_ctor_set(v___x_941_, 1, v___x_940_);
v___x_942_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_942_, 0, v_a_907_);
lean_ctor_set(v___x_942_, 1, v___x_941_);
v___x_943_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_943_, 0, v_a_904_);
lean_ctor_set(v___x_943_, 1, v___x_942_);
v___x_944_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_944_, 0, v_a_896_);
lean_ctor_set(v___x_944_, 1, v___x_943_);
v___x_945_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_945_, 0, v_a_889_);
lean_ctor_set(v___x_945_, 1, v___x_944_);
if (v_isShared_938_ == 0)
{
lean_ctor_set(v___x_937_, 0, v___x_945_);
v___x_947_ = v___x_937_;
goto v_reusejp_946_;
}
else
{
lean_object* v_reuseFailAlloc_948_; 
v_reuseFailAlloc_948_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_948_, 0, v___x_945_);
v___x_947_ = v_reuseFailAlloc_948_;
goto v_reusejp_946_;
}
v_reusejp_946_:
{
return v___x_947_;
}
}
else
{
lean_object* v___x_949_; 
lean_del_object(v___x_937_);
v___x_949_ = lp_mathlib_Lean_instantiateLevelMVars___at___00Mathlib_Tactic_Bound_hypPriority_spec__0___redArg(v_a_889_, v___y_884_);
if (lean_obj_tag(v___x_949_) == 0)
{
lean_object* v_a_950_; lean_object* v___x_951_; 
v_a_950_ = lean_ctor_get(v___x_949_, 0);
lean_inc(v_a_950_);
lean_dec_ref_known(v___x_949_, 1);
v___x_951_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Bound_isZero_spec__0___redArg(v_a_896_, v___y_884_);
if (lean_obj_tag(v___x_951_) == 0)
{
lean_object* v_a_952_; lean_object* v___x_953_; 
v_a_952_ = lean_ctor_get(v___x_951_, 0);
lean_inc(v_a_952_);
lean_dec_ref_known(v___x_951_, 1);
v___x_953_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Bound_isZero_spec__0___redArg(v_a_904_, v___y_884_);
if (lean_obj_tag(v___x_953_) == 0)
{
lean_object* v_a_954_; lean_object* v___x_955_; 
v_a_954_ = lean_ctor_get(v___x_953_, 0);
lean_inc(v_a_954_);
lean_dec_ref_known(v___x_953_, 1);
v___x_955_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Bound_isZero_spec__0___redArg(v_a_907_, v___y_884_);
if (lean_obj_tag(v___x_955_) == 0)
{
lean_object* v_a_956_; lean_object* v___x_957_; 
v_a_956_ = lean_ctor_get(v___x_955_, 0);
lean_inc(v_a_956_);
lean_dec_ref_known(v___x_955_, 1);
v___x_957_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Bound_isZero_spec__0___redArg(v_a_909_, v___y_884_);
if (lean_obj_tag(v___x_957_) == 0)
{
lean_object* v_a_958_; lean_object* v___x_960_; uint8_t v_isShared_961_; uint8_t v_isSharedCheck_970_; 
v_a_958_ = lean_ctor_get(v___x_957_, 0);
v_isSharedCheck_970_ = !lean_is_exclusive(v___x_957_);
if (v_isSharedCheck_970_ == 0)
{
v___x_960_ = v___x_957_;
v_isShared_961_ = v_isSharedCheck_970_;
goto v_resetjp_959_;
}
else
{
lean_inc(v_a_958_);
lean_dec(v___x_957_);
v___x_960_ = lean_box(0);
v_isShared_961_ = v_isSharedCheck_970_;
goto v_resetjp_959_;
}
v_resetjp_959_:
{
lean_object* v___x_962_; lean_object* v___x_963_; lean_object* v___x_964_; lean_object* v___x_965_; lean_object* v___x_966_; lean_object* v___x_968_; 
v___x_962_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_962_, 0, v_a_958_);
lean_ctor_set(v___x_962_, 1, v_a_935_);
v___x_963_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_963_, 0, v_a_956_);
lean_ctor_set(v___x_963_, 1, v___x_962_);
v___x_964_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_964_, 0, v_a_954_);
lean_ctor_set(v___x_964_, 1, v___x_963_);
v___x_965_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_965_, 0, v_a_952_);
lean_ctor_set(v___x_965_, 1, v___x_964_);
v___x_966_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_966_, 0, v_a_950_);
lean_ctor_set(v___x_966_, 1, v___x_965_);
if (v_isShared_961_ == 0)
{
lean_ctor_set(v___x_960_, 0, v___x_966_);
v___x_968_ = v___x_960_;
goto v_reusejp_967_;
}
else
{
lean_object* v_reuseFailAlloc_969_; 
v_reuseFailAlloc_969_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_969_, 0, v___x_966_);
v___x_968_ = v_reuseFailAlloc_969_;
goto v_reusejp_967_;
}
v_reusejp_967_:
{
return v___x_968_;
}
}
}
else
{
lean_object* v_a_971_; lean_object* v___x_973_; uint8_t v_isShared_974_; uint8_t v_isSharedCheck_978_; 
lean_dec(v_a_956_);
lean_dec(v_a_954_);
lean_dec(v_a_952_);
lean_dec(v_a_950_);
lean_dec(v_a_935_);
v_a_971_ = lean_ctor_get(v___x_957_, 0);
v_isSharedCheck_978_ = !lean_is_exclusive(v___x_957_);
if (v_isSharedCheck_978_ == 0)
{
v___x_973_ = v___x_957_;
v_isShared_974_ = v_isSharedCheck_978_;
goto v_resetjp_972_;
}
else
{
lean_inc(v_a_971_);
lean_dec(v___x_957_);
v___x_973_ = lean_box(0);
v_isShared_974_ = v_isSharedCheck_978_;
goto v_resetjp_972_;
}
v_resetjp_972_:
{
lean_object* v___x_976_; 
if (v_isShared_974_ == 0)
{
v___x_976_ = v___x_973_;
goto v_reusejp_975_;
}
else
{
lean_object* v_reuseFailAlloc_977_; 
v_reuseFailAlloc_977_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_977_, 0, v_a_971_);
v___x_976_ = v_reuseFailAlloc_977_;
goto v_reusejp_975_;
}
v_reusejp_975_:
{
return v___x_976_;
}
}
}
}
else
{
lean_object* v_a_979_; lean_object* v___x_981_; uint8_t v_isShared_982_; uint8_t v_isSharedCheck_986_; 
lean_dec(v_a_954_);
lean_dec(v_a_952_);
lean_dec(v_a_950_);
lean_dec(v_a_935_);
lean_dec(v_a_909_);
v_a_979_ = lean_ctor_get(v___x_955_, 0);
v_isSharedCheck_986_ = !lean_is_exclusive(v___x_955_);
if (v_isSharedCheck_986_ == 0)
{
v___x_981_ = v___x_955_;
v_isShared_982_ = v_isSharedCheck_986_;
goto v_resetjp_980_;
}
else
{
lean_inc(v_a_979_);
lean_dec(v___x_955_);
v___x_981_ = lean_box(0);
v_isShared_982_ = v_isSharedCheck_986_;
goto v_resetjp_980_;
}
v_resetjp_980_:
{
lean_object* v___x_984_; 
if (v_isShared_982_ == 0)
{
v___x_984_ = v___x_981_;
goto v_reusejp_983_;
}
else
{
lean_object* v_reuseFailAlloc_985_; 
v_reuseFailAlloc_985_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_985_, 0, v_a_979_);
v___x_984_ = v_reuseFailAlloc_985_;
goto v_reusejp_983_;
}
v_reusejp_983_:
{
return v___x_984_;
}
}
}
}
else
{
lean_object* v_a_987_; lean_object* v___x_989_; uint8_t v_isShared_990_; uint8_t v_isSharedCheck_994_; 
lean_dec(v_a_952_);
lean_dec(v_a_950_);
lean_dec(v_a_935_);
lean_dec(v_a_909_);
lean_dec(v_a_907_);
v_a_987_ = lean_ctor_get(v___x_953_, 0);
v_isSharedCheck_994_ = !lean_is_exclusive(v___x_953_);
if (v_isSharedCheck_994_ == 0)
{
v___x_989_ = v___x_953_;
v_isShared_990_ = v_isSharedCheck_994_;
goto v_resetjp_988_;
}
else
{
lean_inc(v_a_987_);
lean_dec(v___x_953_);
v___x_989_ = lean_box(0);
v_isShared_990_ = v_isSharedCheck_994_;
goto v_resetjp_988_;
}
v_resetjp_988_:
{
lean_object* v___x_992_; 
if (v_isShared_990_ == 0)
{
v___x_992_ = v___x_989_;
goto v_reusejp_991_;
}
else
{
lean_object* v_reuseFailAlloc_993_; 
v_reuseFailAlloc_993_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_993_, 0, v_a_987_);
v___x_992_ = v_reuseFailAlloc_993_;
goto v_reusejp_991_;
}
v_reusejp_991_:
{
return v___x_992_;
}
}
}
}
else
{
lean_object* v_a_995_; lean_object* v___x_997_; uint8_t v_isShared_998_; uint8_t v_isSharedCheck_1002_; 
lean_dec(v_a_950_);
lean_dec(v_a_935_);
lean_dec(v_a_909_);
lean_dec(v_a_907_);
lean_dec(v_a_904_);
v_a_995_ = lean_ctor_get(v___x_951_, 0);
v_isSharedCheck_1002_ = !lean_is_exclusive(v___x_951_);
if (v_isSharedCheck_1002_ == 0)
{
v___x_997_ = v___x_951_;
v_isShared_998_ = v_isSharedCheck_1002_;
goto v_resetjp_996_;
}
else
{
lean_inc(v_a_995_);
lean_dec(v___x_951_);
v___x_997_ = lean_box(0);
v_isShared_998_ = v_isSharedCheck_1002_;
goto v_resetjp_996_;
}
v_resetjp_996_:
{
lean_object* v___x_1000_; 
if (v_isShared_998_ == 0)
{
v___x_1000_ = v___x_997_;
goto v_reusejp_999_;
}
else
{
lean_object* v_reuseFailAlloc_1001_; 
v_reuseFailAlloc_1001_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1001_, 0, v_a_995_);
v___x_1000_ = v_reuseFailAlloc_1001_;
goto v_reusejp_999_;
}
v_reusejp_999_:
{
return v___x_1000_;
}
}
}
}
else
{
lean_object* v_a_1003_; lean_object* v___x_1005_; uint8_t v_isShared_1006_; uint8_t v_isSharedCheck_1010_; 
lean_dec(v_a_935_);
lean_dec(v_a_909_);
lean_dec(v_a_907_);
lean_dec(v_a_904_);
lean_dec(v_a_896_);
v_a_1003_ = lean_ctor_get(v___x_949_, 0);
v_isSharedCheck_1010_ = !lean_is_exclusive(v___x_949_);
if (v_isSharedCheck_1010_ == 0)
{
v___x_1005_ = v___x_949_;
v_isShared_1006_ = v_isSharedCheck_1010_;
goto v_resetjp_1004_;
}
else
{
lean_inc(v_a_1003_);
lean_dec(v___x_949_);
v___x_1005_ = lean_box(0);
v_isShared_1006_ = v_isSharedCheck_1010_;
goto v_resetjp_1004_;
}
v_resetjp_1004_:
{
lean_object* v___x_1008_; 
if (v_isShared_1006_ == 0)
{
v___x_1008_ = v___x_1005_;
goto v_reusejp_1007_;
}
else
{
lean_object* v_reuseFailAlloc_1009_; 
v_reuseFailAlloc_1009_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1009_, 0, v_a_1003_);
v___x_1008_ = v_reuseFailAlloc_1009_;
goto v_reusejp_1007_;
}
v_reusejp_1007_:
{
return v___x_1008_;
}
}
}
}
}
}
else
{
lean_object* v_a_1012_; lean_object* v___x_1014_; uint8_t v_isShared_1015_; uint8_t v_isSharedCheck_1019_; 
lean_dec(v_a_909_);
lean_dec(v_a_907_);
lean_dec(v_a_904_);
lean_dec(v_a_896_);
lean_dec(v_a_889_);
v_a_1012_ = lean_ctor_get(v___x_934_, 0);
v_isSharedCheck_1019_ = !lean_is_exclusive(v___x_934_);
if (v_isSharedCheck_1019_ == 0)
{
v___x_1014_ = v___x_934_;
v_isShared_1015_ = v_isSharedCheck_1019_;
goto v_resetjp_1013_;
}
else
{
lean_inc(v_a_1012_);
lean_dec(v___x_934_);
v___x_1014_ = lean_box(0);
v_isShared_1015_ = v_isSharedCheck_1019_;
goto v_resetjp_1013_;
}
v_resetjp_1013_:
{
lean_object* v___x_1017_; 
if (v_isShared_1015_ == 0)
{
v___x_1017_ = v___x_1014_;
goto v_reusejp_1016_;
}
else
{
lean_object* v_reuseFailAlloc_1018_; 
v_reuseFailAlloc_1018_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1018_, 0, v_a_1012_);
v___x_1017_ = v_reuseFailAlloc_1018_;
goto v_reusejp_1016_;
}
v_reusejp_1016_:
{
return v___x_1017_;
}
}
}
}
}
}
else
{
lean_object* v_a_1022_; lean_object* v___x_1024_; uint8_t v_isShared_1025_; uint8_t v_isSharedCheck_1029_; 
lean_dec(v_a_907_);
lean_dec(v_a_904_);
lean_dec_ref_known(v___x_899_, 2);
lean_dec(v_a_896_);
lean_dec(v_a_889_);
lean_dec_ref(v___y_883_);
lean_dec_ref(v_hyp_881_);
v_a_1022_ = lean_ctor_get(v___x_908_, 0);
v_isSharedCheck_1029_ = !lean_is_exclusive(v___x_908_);
if (v_isSharedCheck_1029_ == 0)
{
v___x_1024_ = v___x_908_;
v_isShared_1025_ = v_isSharedCheck_1029_;
goto v_resetjp_1023_;
}
else
{
lean_inc(v_a_1022_);
lean_dec(v___x_908_);
v___x_1024_ = lean_box(0);
v_isShared_1025_ = v_isSharedCheck_1029_;
goto v_resetjp_1023_;
}
v_resetjp_1023_:
{
lean_object* v___x_1027_; 
if (v_isShared_1025_ == 0)
{
v___x_1027_ = v___x_1024_;
goto v_reusejp_1026_;
}
else
{
lean_object* v_reuseFailAlloc_1028_; 
v_reuseFailAlloc_1028_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1028_, 0, v_a_1022_);
v___x_1027_ = v_reuseFailAlloc_1028_;
goto v_reusejp_1026_;
}
v_reusejp_1026_:
{
return v___x_1027_;
}
}
}
}
else
{
lean_object* v_a_1030_; lean_object* v___x_1032_; uint8_t v_isShared_1033_; uint8_t v_isSharedCheck_1037_; 
lean_dec_ref_known(v___x_905_, 1);
lean_dec(v_a_904_);
lean_dec_ref_known(v___x_899_, 2);
lean_dec(v_a_896_);
lean_dec(v_a_889_);
lean_dec_ref(v___y_883_);
lean_dec_ref(v_hyp_881_);
v_a_1030_ = lean_ctor_get(v___x_906_, 0);
v_isSharedCheck_1037_ = !lean_is_exclusive(v___x_906_);
if (v_isSharedCheck_1037_ == 0)
{
v___x_1032_ = v___x_906_;
v_isShared_1033_ = v_isSharedCheck_1037_;
goto v_resetjp_1031_;
}
else
{
lean_inc(v_a_1030_);
lean_dec(v___x_906_);
v___x_1032_ = lean_box(0);
v_isShared_1033_ = v_isSharedCheck_1037_;
goto v_resetjp_1031_;
}
v_resetjp_1031_:
{
lean_object* v___x_1035_; 
if (v_isShared_1033_ == 0)
{
v___x_1035_ = v___x_1032_;
goto v_reusejp_1034_;
}
else
{
lean_object* v_reuseFailAlloc_1036_; 
v_reuseFailAlloc_1036_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1036_, 0, v_a_1030_);
v___x_1035_ = v_reuseFailAlloc_1036_;
goto v_reusejp_1034_;
}
v_reusejp_1034_:
{
return v___x_1035_;
}
}
}
}
else
{
lean_object* v_a_1038_; lean_object* v___x_1040_; uint8_t v_isShared_1041_; uint8_t v_isSharedCheck_1045_; 
lean_dec_ref_known(v___x_899_, 2);
lean_dec(v_a_896_);
lean_dec(v_a_889_);
lean_dec_ref(v___y_883_);
lean_dec_ref(v_hyp_881_);
v_a_1038_ = lean_ctor_get(v___x_903_, 0);
v_isSharedCheck_1045_ = !lean_is_exclusive(v___x_903_);
if (v_isSharedCheck_1045_ == 0)
{
v___x_1040_ = v___x_903_;
v_isShared_1041_ = v_isSharedCheck_1045_;
goto v_resetjp_1039_;
}
else
{
lean_inc(v_a_1038_);
lean_dec(v___x_903_);
v___x_1040_ = lean_box(0);
v_isShared_1041_ = v_isSharedCheck_1045_;
goto v_resetjp_1039_;
}
v_resetjp_1039_:
{
lean_object* v___x_1043_; 
if (v_isShared_1041_ == 0)
{
v___x_1043_ = v___x_1040_;
goto v_reusejp_1042_;
}
else
{
lean_object* v_reuseFailAlloc_1044_; 
v_reuseFailAlloc_1044_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1044_, 0, v_a_1038_);
v___x_1043_ = v_reuseFailAlloc_1044_;
goto v_reusejp_1042_;
}
v_reusejp_1042_:
{
return v___x_1043_;
}
}
}
}
else
{
lean_object* v_a_1046_; lean_object* v___x_1048_; uint8_t v_isShared_1049_; uint8_t v_isSharedCheck_1053_; 
lean_dec(v_a_889_);
lean_dec_ref(v___y_883_);
lean_dec_ref(v_hyp_881_);
v_a_1046_ = lean_ctor_get(v___x_895_, 0);
v_isSharedCheck_1053_ = !lean_is_exclusive(v___x_895_);
if (v_isSharedCheck_1053_ == 0)
{
v___x_1048_ = v___x_895_;
v_isShared_1049_ = v_isSharedCheck_1053_;
goto v_resetjp_1047_;
}
else
{
lean_inc(v_a_1046_);
lean_dec(v___x_895_);
v___x_1048_ = lean_box(0);
v_isShared_1049_ = v_isSharedCheck_1053_;
goto v_resetjp_1047_;
}
v_resetjp_1047_:
{
lean_object* v___x_1051_; 
if (v_isShared_1049_ == 0)
{
v___x_1051_ = v___x_1048_;
goto v_reusejp_1050_;
}
else
{
lean_object* v_reuseFailAlloc_1052_; 
v_reuseFailAlloc_1052_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1052_, 0, v_a_1046_);
v___x_1051_ = v_reuseFailAlloc_1052_;
goto v_reusejp_1050_;
}
v_reusejp_1050_:
{
return v___x_1051_;
}
}
}
}
else
{
lean_object* v_a_1054_; lean_object* v___x_1056_; uint8_t v_isShared_1057_; uint8_t v_isSharedCheck_1061_; 
lean_dec_ref(v___y_883_);
lean_dec_ref(v_hyp_881_);
v_a_1054_ = lean_ctor_get(v___x_888_, 0);
v_isSharedCheck_1061_ = !lean_is_exclusive(v___x_888_);
if (v_isSharedCheck_1061_ == 0)
{
v___x_1056_ = v___x_888_;
v_isShared_1057_ = v_isSharedCheck_1061_;
goto v_resetjp_1055_;
}
else
{
lean_inc(v_a_1054_);
lean_dec(v___x_888_);
v___x_1056_ = lean_box(0);
v_isShared_1057_ = v_isSharedCheck_1061_;
goto v_resetjp_1055_;
}
v_resetjp_1055_:
{
lean_object* v___x_1059_; 
if (v_isShared_1057_ == 0)
{
v___x_1059_ = v___x_1056_;
goto v_reusejp_1058_;
}
else
{
lean_object* v_reuseFailAlloc_1060_; 
v_reuseFailAlloc_1060_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1060_, 0, v_a_1054_);
v___x_1059_ = v_reuseFailAlloc_1060_;
goto v_reusejp_1058_;
}
v_reusejp_1058_:
{
return v___x_1059_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__3___boxed(lean_object* v_hyp_1062_, lean_object* v___x_1063_, lean_object* v___y_1064_, lean_object* v___y_1065_, lean_object* v___y_1066_, lean_object* v___y_1067_, lean_object* v___y_1068_){
_start:
{
uint8_t v___x_14918__boxed_1069_; lean_object* v_res_1070_; 
v___x_14918__boxed_1069_ = lean_unbox(v___x_1063_);
v_res_1070_ = lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__3(v_hyp_1062_, v___x_14918__boxed_1069_, v___y_1064_, v___y_1065_, v___y_1066_, v___y_1067_);
lean_dec(v___y_1067_);
lean_dec_ref(v___y_1066_);
lean_dec(v___y_1065_);
return v_res_1070_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__4(lean_object* v_hyp_1076_, uint8_t v___x_1077_, lean_object* v___y_1078_, lean_object* v___y_1079_, lean_object* v___y_1080_, lean_object* v___y_1081_){
_start:
{
lean_object* v___x_1083_; 
v___x_1083_ = l_Lean_Meta_mkFreshLevelMVar(v___y_1078_, v___y_1079_, v___y_1080_, v___y_1081_);
if (lean_obj_tag(v___x_1083_) == 0)
{
lean_object* v_a_1084_; lean_object* v___x_1085_; lean_object* v___x_1086_; lean_object* v___x_1087_; uint8_t v___x_1088_; lean_object* v___x_1089_; lean_object* v___x_1090_; 
v_a_1084_ = lean_ctor_get(v___x_1083_, 0);
lean_inc_n(v_a_1084_, 2);
lean_dec_ref_known(v___x_1083_, 1);
v___x_1085_ = l_Lean_Level_succ___override(v_a_1084_);
v___x_1086_ = l_Lean_Expr_sort___override(v___x_1085_);
v___x_1087_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1087_, 0, v___x_1086_);
v___x_1088_ = 0;
v___x_1089_ = lean_box(0);
v___x_1090_ = l_Lean_Meta_mkFreshExprMVar(v___x_1087_, v___x_1088_, v___x_1089_, v___y_1078_, v___y_1079_, v___y_1080_, v___y_1081_);
if (lean_obj_tag(v___x_1090_) == 0)
{
lean_object* v_a_1091_; lean_object* v___x_1092_; lean_object* v___x_1093_; lean_object* v___x_1094_; lean_object* v___x_1095_; lean_object* v___x_1096_; lean_object* v___x_1097_; lean_object* v___x_1098_; 
v_a_1091_ = lean_ctor_get(v___x_1090_, 0);
lean_inc_n(v_a_1091_, 2);
lean_dec_ref_known(v___x_1090_, 1);
v___x_1092_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__2___closed__1));
v___x_1093_ = lean_box(0);
lean_inc(v_a_1084_);
v___x_1094_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1094_, 0, v_a_1084_);
lean_ctor_set(v___x_1094_, 1, v___x_1093_);
lean_inc_ref(v___x_1094_);
v___x_1095_ = l_Lean_Expr_const___override(v___x_1092_, v___x_1094_);
v___x_1096_ = l_Lean_Expr_app___override(v___x_1095_, v_a_1091_);
v___x_1097_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1097_, 0, v___x_1096_);
v___x_1098_ = l_Lean_Meta_mkFreshExprMVar(v___x_1097_, v___x_1088_, v___x_1089_, v___y_1078_, v___y_1079_, v___y_1080_, v___y_1081_);
if (lean_obj_tag(v___x_1098_) == 0)
{
lean_object* v_a_1099_; lean_object* v___x_1100_; lean_object* v___x_1101_; 
v_a_1099_ = lean_ctor_get(v___x_1098_, 0);
lean_inc(v_a_1099_);
lean_dec_ref_known(v___x_1098_, 1);
lean_inc(v_a_1091_);
v___x_1100_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1100_, 0, v_a_1091_);
lean_inc_ref(v___x_1100_);
v___x_1101_ = l_Lean_Meta_mkFreshExprMVar(v___x_1100_, v___x_1088_, v___x_1089_, v___y_1078_, v___y_1079_, v___y_1080_, v___y_1081_);
if (lean_obj_tag(v___x_1101_) == 0)
{
lean_object* v_a_1102_; lean_object* v___x_1103_; 
v_a_1102_ = lean_ctor_get(v___x_1101_, 0);
lean_inc(v_a_1102_);
lean_dec_ref_known(v___x_1101_, 1);
v___x_1103_ = l_Lean_Meta_mkFreshExprMVar(v___x_1100_, v___x_1088_, v___x_1089_, v___y_1078_, v___y_1079_, v___y_1080_, v___y_1081_);
if (lean_obj_tag(v___x_1103_) == 0)
{
lean_object* v_a_1104_; lean_object* v_keyedConfig_1105_; uint8_t v_trackZetaDelta_1106_; lean_object* v_zetaDeltaSet_1107_; lean_object* v_lctx_1108_; lean_object* v_localInstances_1109_; lean_object* v_defEqCtx_x3f_1110_; lean_object* v_synthPendingDepth_1111_; lean_object* v_customCanUnfoldPredicate_x3f_1112_; uint8_t v_univApprox_1113_; uint8_t v_inTypeClassResolution_1114_; uint8_t v_cacheInferType_1115_; lean_object* v___x_1117_; uint8_t v_isShared_1118_; uint8_t v_isSharedCheck_1216_; 
v_a_1104_ = lean_ctor_get(v___x_1103_, 0);
lean_inc(v_a_1104_);
lean_dec_ref_known(v___x_1103_, 1);
v_keyedConfig_1105_ = lean_ctor_get(v___y_1078_, 0);
v_trackZetaDelta_1106_ = lean_ctor_get_uint8(v___y_1078_, sizeof(void*)*7);
v_zetaDeltaSet_1107_ = lean_ctor_get(v___y_1078_, 1);
v_lctx_1108_ = lean_ctor_get(v___y_1078_, 2);
v_localInstances_1109_ = lean_ctor_get(v___y_1078_, 3);
v_defEqCtx_x3f_1110_ = lean_ctor_get(v___y_1078_, 4);
v_synthPendingDepth_1111_ = lean_ctor_get(v___y_1078_, 5);
v_customCanUnfoldPredicate_x3f_1112_ = lean_ctor_get(v___y_1078_, 6);
v_univApprox_1113_ = lean_ctor_get_uint8(v___y_1078_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_1114_ = lean_ctor_get_uint8(v___y_1078_, sizeof(void*)*7 + 2);
v_cacheInferType_1115_ = lean_ctor_get_uint8(v___y_1078_, sizeof(void*)*7 + 3);
v_isSharedCheck_1216_ = !lean_is_exclusive(v___y_1078_);
if (v_isSharedCheck_1216_ == 0)
{
v___x_1117_ = v___y_1078_;
v_isShared_1118_ = v_isSharedCheck_1216_;
goto v_resetjp_1116_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_1112_);
lean_inc(v_synthPendingDepth_1111_);
lean_inc(v_defEqCtx_x3f_1110_);
lean_inc(v_localInstances_1109_);
lean_inc(v_lctx_1108_);
lean_inc(v_zetaDeltaSet_1107_);
lean_inc(v_keyedConfig_1105_);
lean_dec(v___y_1078_);
v___x_1117_ = lean_box(0);
v_isShared_1118_ = v_isSharedCheck_1216_;
goto v_resetjp_1116_;
}
v_resetjp_1116_:
{
lean_object* v___x_1119_; lean_object* v___x_1120_; lean_object* v___x_1121_; lean_object* v___x_1122_; lean_object* v___x_1123_; lean_object* v___x_1124_; uint8_t v___x_1125_; lean_object* v___x_1126_; lean_object* v___x_1128_; 
v___x_1119_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__4___closed__2));
v___x_1120_ = l_Lean_Expr_const___override(v___x_1119_, v___x_1094_);
lean_inc(v_a_1091_);
v___x_1121_ = l_Lean_Expr_app___override(v___x_1120_, v_a_1091_);
lean_inc(v_a_1099_);
v___x_1122_ = l_Lean_Expr_app___override(v___x_1121_, v_a_1099_);
lean_inc(v_a_1102_);
v___x_1123_ = l_Lean_Expr_app___override(v___x_1122_, v_a_1102_);
lean_inc(v_a_1104_);
v___x_1124_ = l_Lean_Expr_app___override(v___x_1123_, v_a_1104_);
v___x_1125_ = 2;
v___x_1126_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_1125_, v_keyedConfig_1105_);
if (v_isShared_1118_ == 0)
{
lean_ctor_set(v___x_1117_, 0, v___x_1126_);
v___x_1128_ = v___x_1117_;
goto v_reusejp_1127_;
}
else
{
lean_object* v_reuseFailAlloc_1215_; 
v_reuseFailAlloc_1215_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_1215_, 0, v___x_1126_);
lean_ctor_set(v_reuseFailAlloc_1215_, 1, v_zetaDeltaSet_1107_);
lean_ctor_set(v_reuseFailAlloc_1215_, 2, v_lctx_1108_);
lean_ctor_set(v_reuseFailAlloc_1215_, 3, v_localInstances_1109_);
lean_ctor_set(v_reuseFailAlloc_1215_, 4, v_defEqCtx_x3f_1110_);
lean_ctor_set(v_reuseFailAlloc_1215_, 5, v_synthPendingDepth_1111_);
lean_ctor_set(v_reuseFailAlloc_1215_, 6, v_customCanUnfoldPredicate_x3f_1112_);
lean_ctor_set_uint8(v_reuseFailAlloc_1215_, sizeof(void*)*7, v_trackZetaDelta_1106_);
lean_ctor_set_uint8(v_reuseFailAlloc_1215_, sizeof(void*)*7 + 1, v_univApprox_1113_);
lean_ctor_set_uint8(v_reuseFailAlloc_1215_, sizeof(void*)*7 + 2, v_inTypeClassResolution_1114_);
lean_ctor_set_uint8(v_reuseFailAlloc_1215_, sizeof(void*)*7 + 3, v_cacheInferType_1115_);
v___x_1128_ = v_reuseFailAlloc_1215_;
goto v_reusejp_1127_;
}
v_reusejp_1127_:
{
lean_object* v___x_1129_; 
v___x_1129_ = l_Lean_Meta_isExprDefEq(v___x_1124_, v_hyp_1076_, v___x_1128_, v___y_1079_, v___y_1080_, v___y_1081_);
lean_dec_ref(v___x_1128_);
if (lean_obj_tag(v___x_1129_) == 0)
{
lean_object* v_a_1130_; lean_object* v___x_1132_; uint8_t v_isShared_1133_; uint8_t v_isSharedCheck_1206_; 
v_a_1130_ = lean_ctor_get(v___x_1129_, 0);
v_isSharedCheck_1206_ = !lean_is_exclusive(v___x_1129_);
if (v_isSharedCheck_1206_ == 0)
{
v___x_1132_ = v___x_1129_;
v_isShared_1133_ = v_isSharedCheck_1206_;
goto v_resetjp_1131_;
}
else
{
lean_inc(v_a_1130_);
lean_dec(v___x_1129_);
v___x_1132_ = lean_box(0);
v_isShared_1133_ = v_isSharedCheck_1206_;
goto v_resetjp_1131_;
}
v_resetjp_1131_:
{
uint8_t v___x_1134_; 
v___x_1134_ = lean_unbox(v_a_1130_);
if (v___x_1134_ == 0)
{
lean_object* v___x_1135_; lean_object* v___x_1136_; lean_object* v___x_1137_; lean_object* v___x_1138_; lean_object* v___x_1139_; lean_object* v___x_1140_; lean_object* v___x_1142_; 
lean_dec(v_a_1130_);
v___x_1135_ = lean_box(v___x_1077_);
v___x_1136_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1136_, 0, v_a_1104_);
lean_ctor_set(v___x_1136_, 1, v___x_1135_);
v___x_1137_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1137_, 0, v_a_1102_);
lean_ctor_set(v___x_1137_, 1, v___x_1136_);
v___x_1138_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1138_, 0, v_a_1099_);
lean_ctor_set(v___x_1138_, 1, v___x_1137_);
v___x_1139_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1139_, 0, v_a_1091_);
lean_ctor_set(v___x_1139_, 1, v___x_1138_);
v___x_1140_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1140_, 0, v_a_1084_);
lean_ctor_set(v___x_1140_, 1, v___x_1139_);
if (v_isShared_1133_ == 0)
{
lean_ctor_set(v___x_1132_, 0, v___x_1140_);
v___x_1142_ = v___x_1132_;
goto v_reusejp_1141_;
}
else
{
lean_object* v_reuseFailAlloc_1143_; 
v_reuseFailAlloc_1143_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1143_, 0, v___x_1140_);
v___x_1142_ = v_reuseFailAlloc_1143_;
goto v_reusejp_1141_;
}
v_reusejp_1141_:
{
return v___x_1142_;
}
}
else
{
lean_object* v___x_1144_; 
lean_del_object(v___x_1132_);
v___x_1144_ = lp_mathlib_Lean_instantiateLevelMVars___at___00Mathlib_Tactic_Bound_hypPriority_spec__0___redArg(v_a_1084_, v___y_1079_);
if (lean_obj_tag(v___x_1144_) == 0)
{
lean_object* v_a_1145_; lean_object* v___x_1146_; 
v_a_1145_ = lean_ctor_get(v___x_1144_, 0);
lean_inc(v_a_1145_);
lean_dec_ref_known(v___x_1144_, 1);
v___x_1146_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Bound_isZero_spec__0___redArg(v_a_1091_, v___y_1079_);
if (lean_obj_tag(v___x_1146_) == 0)
{
lean_object* v_a_1147_; lean_object* v___x_1148_; 
v_a_1147_ = lean_ctor_get(v___x_1146_, 0);
lean_inc(v_a_1147_);
lean_dec_ref_known(v___x_1146_, 1);
v___x_1148_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Bound_isZero_spec__0___redArg(v_a_1099_, v___y_1079_);
if (lean_obj_tag(v___x_1148_) == 0)
{
lean_object* v_a_1149_; lean_object* v___x_1150_; 
v_a_1149_ = lean_ctor_get(v___x_1148_, 0);
lean_inc(v_a_1149_);
lean_dec_ref_known(v___x_1148_, 1);
v___x_1150_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Bound_isZero_spec__0___redArg(v_a_1102_, v___y_1079_);
if (lean_obj_tag(v___x_1150_) == 0)
{
lean_object* v_a_1151_; lean_object* v___x_1152_; 
v_a_1151_ = lean_ctor_get(v___x_1150_, 0);
lean_inc(v_a_1151_);
lean_dec_ref_known(v___x_1150_, 1);
v___x_1152_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Bound_isZero_spec__0___redArg(v_a_1104_, v___y_1079_);
if (lean_obj_tag(v___x_1152_) == 0)
{
lean_object* v_a_1153_; lean_object* v___x_1155_; uint8_t v_isShared_1156_; uint8_t v_isSharedCheck_1165_; 
v_a_1153_ = lean_ctor_get(v___x_1152_, 0);
v_isSharedCheck_1165_ = !lean_is_exclusive(v___x_1152_);
if (v_isSharedCheck_1165_ == 0)
{
v___x_1155_ = v___x_1152_;
v_isShared_1156_ = v_isSharedCheck_1165_;
goto v_resetjp_1154_;
}
else
{
lean_inc(v_a_1153_);
lean_dec(v___x_1152_);
v___x_1155_ = lean_box(0);
v_isShared_1156_ = v_isSharedCheck_1165_;
goto v_resetjp_1154_;
}
v_resetjp_1154_:
{
lean_object* v___x_1157_; lean_object* v___x_1158_; lean_object* v___x_1159_; lean_object* v___x_1160_; lean_object* v___x_1161_; lean_object* v___x_1163_; 
v___x_1157_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1157_, 0, v_a_1153_);
lean_ctor_set(v___x_1157_, 1, v_a_1130_);
v___x_1158_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1158_, 0, v_a_1151_);
lean_ctor_set(v___x_1158_, 1, v___x_1157_);
v___x_1159_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1159_, 0, v_a_1149_);
lean_ctor_set(v___x_1159_, 1, v___x_1158_);
v___x_1160_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1160_, 0, v_a_1147_);
lean_ctor_set(v___x_1160_, 1, v___x_1159_);
v___x_1161_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1161_, 0, v_a_1145_);
lean_ctor_set(v___x_1161_, 1, v___x_1160_);
if (v_isShared_1156_ == 0)
{
lean_ctor_set(v___x_1155_, 0, v___x_1161_);
v___x_1163_ = v___x_1155_;
goto v_reusejp_1162_;
}
else
{
lean_object* v_reuseFailAlloc_1164_; 
v_reuseFailAlloc_1164_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1164_, 0, v___x_1161_);
v___x_1163_ = v_reuseFailAlloc_1164_;
goto v_reusejp_1162_;
}
v_reusejp_1162_:
{
return v___x_1163_;
}
}
}
else
{
lean_object* v_a_1166_; lean_object* v___x_1168_; uint8_t v_isShared_1169_; uint8_t v_isSharedCheck_1173_; 
lean_dec(v_a_1151_);
lean_dec(v_a_1149_);
lean_dec(v_a_1147_);
lean_dec(v_a_1145_);
lean_dec(v_a_1130_);
v_a_1166_ = lean_ctor_get(v___x_1152_, 0);
v_isSharedCheck_1173_ = !lean_is_exclusive(v___x_1152_);
if (v_isSharedCheck_1173_ == 0)
{
v___x_1168_ = v___x_1152_;
v_isShared_1169_ = v_isSharedCheck_1173_;
goto v_resetjp_1167_;
}
else
{
lean_inc(v_a_1166_);
lean_dec(v___x_1152_);
v___x_1168_ = lean_box(0);
v_isShared_1169_ = v_isSharedCheck_1173_;
goto v_resetjp_1167_;
}
v_resetjp_1167_:
{
lean_object* v___x_1171_; 
if (v_isShared_1169_ == 0)
{
v___x_1171_ = v___x_1168_;
goto v_reusejp_1170_;
}
else
{
lean_object* v_reuseFailAlloc_1172_; 
v_reuseFailAlloc_1172_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1172_, 0, v_a_1166_);
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
else
{
lean_object* v_a_1174_; lean_object* v___x_1176_; uint8_t v_isShared_1177_; uint8_t v_isSharedCheck_1181_; 
lean_dec(v_a_1149_);
lean_dec(v_a_1147_);
lean_dec(v_a_1145_);
lean_dec(v_a_1130_);
lean_dec(v_a_1104_);
v_a_1174_ = lean_ctor_get(v___x_1150_, 0);
v_isSharedCheck_1181_ = !lean_is_exclusive(v___x_1150_);
if (v_isSharedCheck_1181_ == 0)
{
v___x_1176_ = v___x_1150_;
v_isShared_1177_ = v_isSharedCheck_1181_;
goto v_resetjp_1175_;
}
else
{
lean_inc(v_a_1174_);
lean_dec(v___x_1150_);
v___x_1176_ = lean_box(0);
v_isShared_1177_ = v_isSharedCheck_1181_;
goto v_resetjp_1175_;
}
v_resetjp_1175_:
{
lean_object* v___x_1179_; 
if (v_isShared_1177_ == 0)
{
v___x_1179_ = v___x_1176_;
goto v_reusejp_1178_;
}
else
{
lean_object* v_reuseFailAlloc_1180_; 
v_reuseFailAlloc_1180_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1180_, 0, v_a_1174_);
v___x_1179_ = v_reuseFailAlloc_1180_;
goto v_reusejp_1178_;
}
v_reusejp_1178_:
{
return v___x_1179_;
}
}
}
}
else
{
lean_object* v_a_1182_; lean_object* v___x_1184_; uint8_t v_isShared_1185_; uint8_t v_isSharedCheck_1189_; 
lean_dec(v_a_1147_);
lean_dec(v_a_1145_);
lean_dec(v_a_1130_);
lean_dec(v_a_1104_);
lean_dec(v_a_1102_);
v_a_1182_ = lean_ctor_get(v___x_1148_, 0);
v_isSharedCheck_1189_ = !lean_is_exclusive(v___x_1148_);
if (v_isSharedCheck_1189_ == 0)
{
v___x_1184_ = v___x_1148_;
v_isShared_1185_ = v_isSharedCheck_1189_;
goto v_resetjp_1183_;
}
else
{
lean_inc(v_a_1182_);
lean_dec(v___x_1148_);
v___x_1184_ = lean_box(0);
v_isShared_1185_ = v_isSharedCheck_1189_;
goto v_resetjp_1183_;
}
v_resetjp_1183_:
{
lean_object* v___x_1187_; 
if (v_isShared_1185_ == 0)
{
v___x_1187_ = v___x_1184_;
goto v_reusejp_1186_;
}
else
{
lean_object* v_reuseFailAlloc_1188_; 
v_reuseFailAlloc_1188_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1188_, 0, v_a_1182_);
v___x_1187_ = v_reuseFailAlloc_1188_;
goto v_reusejp_1186_;
}
v_reusejp_1186_:
{
return v___x_1187_;
}
}
}
}
else
{
lean_object* v_a_1190_; lean_object* v___x_1192_; uint8_t v_isShared_1193_; uint8_t v_isSharedCheck_1197_; 
lean_dec(v_a_1145_);
lean_dec(v_a_1130_);
lean_dec(v_a_1104_);
lean_dec(v_a_1102_);
lean_dec(v_a_1099_);
v_a_1190_ = lean_ctor_get(v___x_1146_, 0);
v_isSharedCheck_1197_ = !lean_is_exclusive(v___x_1146_);
if (v_isSharedCheck_1197_ == 0)
{
v___x_1192_ = v___x_1146_;
v_isShared_1193_ = v_isSharedCheck_1197_;
goto v_resetjp_1191_;
}
else
{
lean_inc(v_a_1190_);
lean_dec(v___x_1146_);
v___x_1192_ = lean_box(0);
v_isShared_1193_ = v_isSharedCheck_1197_;
goto v_resetjp_1191_;
}
v_resetjp_1191_:
{
lean_object* v___x_1195_; 
if (v_isShared_1193_ == 0)
{
v___x_1195_ = v___x_1192_;
goto v_reusejp_1194_;
}
else
{
lean_object* v_reuseFailAlloc_1196_; 
v_reuseFailAlloc_1196_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1196_, 0, v_a_1190_);
v___x_1195_ = v_reuseFailAlloc_1196_;
goto v_reusejp_1194_;
}
v_reusejp_1194_:
{
return v___x_1195_;
}
}
}
}
else
{
lean_object* v_a_1198_; lean_object* v___x_1200_; uint8_t v_isShared_1201_; uint8_t v_isSharedCheck_1205_; 
lean_dec(v_a_1130_);
lean_dec(v_a_1104_);
lean_dec(v_a_1102_);
lean_dec(v_a_1099_);
lean_dec(v_a_1091_);
v_a_1198_ = lean_ctor_get(v___x_1144_, 0);
v_isSharedCheck_1205_ = !lean_is_exclusive(v___x_1144_);
if (v_isSharedCheck_1205_ == 0)
{
v___x_1200_ = v___x_1144_;
v_isShared_1201_ = v_isSharedCheck_1205_;
goto v_resetjp_1199_;
}
else
{
lean_inc(v_a_1198_);
lean_dec(v___x_1144_);
v___x_1200_ = lean_box(0);
v_isShared_1201_ = v_isSharedCheck_1205_;
goto v_resetjp_1199_;
}
v_resetjp_1199_:
{
lean_object* v___x_1203_; 
if (v_isShared_1201_ == 0)
{
v___x_1203_ = v___x_1200_;
goto v_reusejp_1202_;
}
else
{
lean_object* v_reuseFailAlloc_1204_; 
v_reuseFailAlloc_1204_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1204_, 0, v_a_1198_);
v___x_1203_ = v_reuseFailAlloc_1204_;
goto v_reusejp_1202_;
}
v_reusejp_1202_:
{
return v___x_1203_;
}
}
}
}
}
}
else
{
lean_object* v_a_1207_; lean_object* v___x_1209_; uint8_t v_isShared_1210_; uint8_t v_isSharedCheck_1214_; 
lean_dec(v_a_1104_);
lean_dec(v_a_1102_);
lean_dec(v_a_1099_);
lean_dec(v_a_1091_);
lean_dec(v_a_1084_);
v_a_1207_ = lean_ctor_get(v___x_1129_, 0);
v_isSharedCheck_1214_ = !lean_is_exclusive(v___x_1129_);
if (v_isSharedCheck_1214_ == 0)
{
v___x_1209_ = v___x_1129_;
v_isShared_1210_ = v_isSharedCheck_1214_;
goto v_resetjp_1208_;
}
else
{
lean_inc(v_a_1207_);
lean_dec(v___x_1129_);
v___x_1209_ = lean_box(0);
v_isShared_1210_ = v_isSharedCheck_1214_;
goto v_resetjp_1208_;
}
v_resetjp_1208_:
{
lean_object* v___x_1212_; 
if (v_isShared_1210_ == 0)
{
v___x_1212_ = v___x_1209_;
goto v_reusejp_1211_;
}
else
{
lean_object* v_reuseFailAlloc_1213_; 
v_reuseFailAlloc_1213_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1213_, 0, v_a_1207_);
v___x_1212_ = v_reuseFailAlloc_1213_;
goto v_reusejp_1211_;
}
v_reusejp_1211_:
{
return v___x_1212_;
}
}
}
}
}
}
else
{
lean_object* v_a_1217_; lean_object* v___x_1219_; uint8_t v_isShared_1220_; uint8_t v_isSharedCheck_1224_; 
lean_dec(v_a_1102_);
lean_dec(v_a_1099_);
lean_dec_ref_known(v___x_1094_, 2);
lean_dec(v_a_1091_);
lean_dec(v_a_1084_);
lean_dec_ref(v___y_1078_);
lean_dec_ref(v_hyp_1076_);
v_a_1217_ = lean_ctor_get(v___x_1103_, 0);
v_isSharedCheck_1224_ = !lean_is_exclusive(v___x_1103_);
if (v_isSharedCheck_1224_ == 0)
{
v___x_1219_ = v___x_1103_;
v_isShared_1220_ = v_isSharedCheck_1224_;
goto v_resetjp_1218_;
}
else
{
lean_inc(v_a_1217_);
lean_dec(v___x_1103_);
v___x_1219_ = lean_box(0);
v_isShared_1220_ = v_isSharedCheck_1224_;
goto v_resetjp_1218_;
}
v_resetjp_1218_:
{
lean_object* v___x_1222_; 
if (v_isShared_1220_ == 0)
{
v___x_1222_ = v___x_1219_;
goto v_reusejp_1221_;
}
else
{
lean_object* v_reuseFailAlloc_1223_; 
v_reuseFailAlloc_1223_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1223_, 0, v_a_1217_);
v___x_1222_ = v_reuseFailAlloc_1223_;
goto v_reusejp_1221_;
}
v_reusejp_1221_:
{
return v___x_1222_;
}
}
}
}
else
{
lean_object* v_a_1225_; lean_object* v___x_1227_; uint8_t v_isShared_1228_; uint8_t v_isSharedCheck_1232_; 
lean_dec_ref_known(v___x_1100_, 1);
lean_dec(v_a_1099_);
lean_dec_ref_known(v___x_1094_, 2);
lean_dec(v_a_1091_);
lean_dec(v_a_1084_);
lean_dec_ref(v___y_1078_);
lean_dec_ref(v_hyp_1076_);
v_a_1225_ = lean_ctor_get(v___x_1101_, 0);
v_isSharedCheck_1232_ = !lean_is_exclusive(v___x_1101_);
if (v_isSharedCheck_1232_ == 0)
{
v___x_1227_ = v___x_1101_;
v_isShared_1228_ = v_isSharedCheck_1232_;
goto v_resetjp_1226_;
}
else
{
lean_inc(v_a_1225_);
lean_dec(v___x_1101_);
v___x_1227_ = lean_box(0);
v_isShared_1228_ = v_isSharedCheck_1232_;
goto v_resetjp_1226_;
}
v_resetjp_1226_:
{
lean_object* v___x_1230_; 
if (v_isShared_1228_ == 0)
{
v___x_1230_ = v___x_1227_;
goto v_reusejp_1229_;
}
else
{
lean_object* v_reuseFailAlloc_1231_; 
v_reuseFailAlloc_1231_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1231_, 0, v_a_1225_);
v___x_1230_ = v_reuseFailAlloc_1231_;
goto v_reusejp_1229_;
}
v_reusejp_1229_:
{
return v___x_1230_;
}
}
}
}
else
{
lean_object* v_a_1233_; lean_object* v___x_1235_; uint8_t v_isShared_1236_; uint8_t v_isSharedCheck_1240_; 
lean_dec_ref_known(v___x_1094_, 2);
lean_dec(v_a_1091_);
lean_dec(v_a_1084_);
lean_dec_ref(v___y_1078_);
lean_dec_ref(v_hyp_1076_);
v_a_1233_ = lean_ctor_get(v___x_1098_, 0);
v_isSharedCheck_1240_ = !lean_is_exclusive(v___x_1098_);
if (v_isSharedCheck_1240_ == 0)
{
v___x_1235_ = v___x_1098_;
v_isShared_1236_ = v_isSharedCheck_1240_;
goto v_resetjp_1234_;
}
else
{
lean_inc(v_a_1233_);
lean_dec(v___x_1098_);
v___x_1235_ = lean_box(0);
v_isShared_1236_ = v_isSharedCheck_1240_;
goto v_resetjp_1234_;
}
v_resetjp_1234_:
{
lean_object* v___x_1238_; 
if (v_isShared_1236_ == 0)
{
v___x_1238_ = v___x_1235_;
goto v_reusejp_1237_;
}
else
{
lean_object* v_reuseFailAlloc_1239_; 
v_reuseFailAlloc_1239_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1239_, 0, v_a_1233_);
v___x_1238_ = v_reuseFailAlloc_1239_;
goto v_reusejp_1237_;
}
v_reusejp_1237_:
{
return v___x_1238_;
}
}
}
}
else
{
lean_object* v_a_1241_; lean_object* v___x_1243_; uint8_t v_isShared_1244_; uint8_t v_isSharedCheck_1248_; 
lean_dec(v_a_1084_);
lean_dec_ref(v___y_1078_);
lean_dec_ref(v_hyp_1076_);
v_a_1241_ = lean_ctor_get(v___x_1090_, 0);
v_isSharedCheck_1248_ = !lean_is_exclusive(v___x_1090_);
if (v_isSharedCheck_1248_ == 0)
{
v___x_1243_ = v___x_1090_;
v_isShared_1244_ = v_isSharedCheck_1248_;
goto v_resetjp_1242_;
}
else
{
lean_inc(v_a_1241_);
lean_dec(v___x_1090_);
v___x_1243_ = lean_box(0);
v_isShared_1244_ = v_isSharedCheck_1248_;
goto v_resetjp_1242_;
}
v_resetjp_1242_:
{
lean_object* v___x_1246_; 
if (v_isShared_1244_ == 0)
{
v___x_1246_ = v___x_1243_;
goto v_reusejp_1245_;
}
else
{
lean_object* v_reuseFailAlloc_1247_; 
v_reuseFailAlloc_1247_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1247_, 0, v_a_1241_);
v___x_1246_ = v_reuseFailAlloc_1247_;
goto v_reusejp_1245_;
}
v_reusejp_1245_:
{
return v___x_1246_;
}
}
}
}
else
{
lean_object* v_a_1249_; lean_object* v___x_1251_; uint8_t v_isShared_1252_; uint8_t v_isSharedCheck_1256_; 
lean_dec_ref(v___y_1078_);
lean_dec_ref(v_hyp_1076_);
v_a_1249_ = lean_ctor_get(v___x_1083_, 0);
v_isSharedCheck_1256_ = !lean_is_exclusive(v___x_1083_);
if (v_isSharedCheck_1256_ == 0)
{
v___x_1251_ = v___x_1083_;
v_isShared_1252_ = v_isSharedCheck_1256_;
goto v_resetjp_1250_;
}
else
{
lean_inc(v_a_1249_);
lean_dec(v___x_1083_);
v___x_1251_ = lean_box(0);
v_isShared_1252_ = v_isSharedCheck_1256_;
goto v_resetjp_1250_;
}
v_resetjp_1250_:
{
lean_object* v___x_1254_; 
if (v_isShared_1252_ == 0)
{
v___x_1254_ = v___x_1251_;
goto v_reusejp_1253_;
}
else
{
lean_object* v_reuseFailAlloc_1255_; 
v_reuseFailAlloc_1255_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1255_, 0, v_a_1249_);
v___x_1254_ = v_reuseFailAlloc_1255_;
goto v_reusejp_1253_;
}
v_reusejp_1253_:
{
return v___x_1254_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__4___boxed(lean_object* v_hyp_1257_, lean_object* v___x_1258_, lean_object* v___y_1259_, lean_object* v___y_1260_, lean_object* v___y_1261_, lean_object* v___y_1262_, lean_object* v___y_1263_){
_start:
{
uint8_t v___x_15282__boxed_1264_; lean_object* v_res_1265_; 
v___x_15282__boxed_1264_ = lean_unbox(v___x_1258_);
v_res_1265_ = lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__4(v_hyp_1257_, v___x_15282__boxed_1264_, v___y_1259_, v___y_1260_, v___y_1261_, v___y_1262_);
lean_dec(v___y_1262_);
lean_dec_ref(v___y_1261_);
lean_dec(v___y_1260_);
return v_res_1265_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__5(lean_object* v_hyp_1271_, lean_object* v___y_1272_, lean_object* v___y_1273_, lean_object* v___y_1274_, lean_object* v___y_1275_){
_start:
{
lean_object* v___x_1277_; 
v___x_1277_ = l_Lean_Meta_mkFreshLevelMVar(v___y_1272_, v___y_1273_, v___y_1274_, v___y_1275_);
if (lean_obj_tag(v___x_1277_) == 0)
{
lean_object* v_a_1278_; lean_object* v___x_1279_; lean_object* v___x_1280_; lean_object* v___x_1281_; uint8_t v___x_1282_; lean_object* v___x_1283_; lean_object* v___x_1284_; 
v_a_1278_ = lean_ctor_get(v___x_1277_, 0);
lean_inc_n(v_a_1278_, 2);
lean_dec_ref_known(v___x_1277_, 1);
v___x_1279_ = l_Lean_Level_succ___override(v_a_1278_);
v___x_1280_ = l_Lean_Expr_sort___override(v___x_1279_);
v___x_1281_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1281_, 0, v___x_1280_);
v___x_1282_ = 0;
v___x_1283_ = lean_box(0);
v___x_1284_ = l_Lean_Meta_mkFreshExprMVar(v___x_1281_, v___x_1282_, v___x_1283_, v___y_1272_, v___y_1273_, v___y_1274_, v___y_1275_);
if (lean_obj_tag(v___x_1284_) == 0)
{
lean_object* v_a_1285_; lean_object* v___x_1286_; lean_object* v___x_1287_; lean_object* v___x_1288_; lean_object* v___x_1289_; lean_object* v___x_1290_; lean_object* v___x_1291_; lean_object* v___x_1292_; 
v_a_1285_ = lean_ctor_get(v___x_1284_, 0);
lean_inc_n(v_a_1285_, 2);
lean_dec_ref_known(v___x_1284_, 1);
v___x_1286_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__3___closed__1));
v___x_1287_ = lean_box(0);
lean_inc(v_a_1278_);
v___x_1288_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1288_, 0, v_a_1278_);
lean_ctor_set(v___x_1288_, 1, v___x_1287_);
lean_inc_ref(v___x_1288_);
v___x_1289_ = l_Lean_Expr_const___override(v___x_1286_, v___x_1288_);
v___x_1290_ = l_Lean_Expr_app___override(v___x_1289_, v_a_1285_);
v___x_1291_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1291_, 0, v___x_1290_);
v___x_1292_ = l_Lean_Meta_mkFreshExprMVar(v___x_1291_, v___x_1282_, v___x_1283_, v___y_1272_, v___y_1273_, v___y_1274_, v___y_1275_);
if (lean_obj_tag(v___x_1292_) == 0)
{
lean_object* v_a_1293_; lean_object* v___x_1294_; lean_object* v___x_1295_; 
v_a_1293_ = lean_ctor_get(v___x_1292_, 0);
lean_inc(v_a_1293_);
lean_dec_ref_known(v___x_1292_, 1);
lean_inc(v_a_1285_);
v___x_1294_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1294_, 0, v_a_1285_);
lean_inc_ref(v___x_1294_);
v___x_1295_ = l_Lean_Meta_mkFreshExprMVar(v___x_1294_, v___x_1282_, v___x_1283_, v___y_1272_, v___y_1273_, v___y_1274_, v___y_1275_);
if (lean_obj_tag(v___x_1295_) == 0)
{
lean_object* v_a_1296_; lean_object* v___x_1297_; 
v_a_1296_ = lean_ctor_get(v___x_1295_, 0);
lean_inc(v_a_1296_);
lean_dec_ref_known(v___x_1295_, 1);
v___x_1297_ = l_Lean_Meta_mkFreshExprMVar(v___x_1294_, v___x_1282_, v___x_1283_, v___y_1272_, v___y_1273_, v___y_1274_, v___y_1275_);
if (lean_obj_tag(v___x_1297_) == 0)
{
lean_object* v_a_1298_; lean_object* v_keyedConfig_1299_; uint8_t v_trackZetaDelta_1300_; lean_object* v_zetaDeltaSet_1301_; lean_object* v_lctx_1302_; lean_object* v_localInstances_1303_; lean_object* v_defEqCtx_x3f_1304_; lean_object* v_synthPendingDepth_1305_; lean_object* v_customCanUnfoldPredicate_x3f_1306_; uint8_t v_univApprox_1307_; uint8_t v_inTypeClassResolution_1308_; uint8_t v_cacheInferType_1309_; lean_object* v___x_1311_; uint8_t v_isShared_1312_; uint8_t v_isSharedCheck_1409_; 
v_a_1298_ = lean_ctor_get(v___x_1297_, 0);
lean_inc(v_a_1298_);
lean_dec_ref_known(v___x_1297_, 1);
v_keyedConfig_1299_ = lean_ctor_get(v___y_1272_, 0);
v_trackZetaDelta_1300_ = lean_ctor_get_uint8(v___y_1272_, sizeof(void*)*7);
v_zetaDeltaSet_1301_ = lean_ctor_get(v___y_1272_, 1);
v_lctx_1302_ = lean_ctor_get(v___y_1272_, 2);
v_localInstances_1303_ = lean_ctor_get(v___y_1272_, 3);
v_defEqCtx_x3f_1304_ = lean_ctor_get(v___y_1272_, 4);
v_synthPendingDepth_1305_ = lean_ctor_get(v___y_1272_, 5);
v_customCanUnfoldPredicate_x3f_1306_ = lean_ctor_get(v___y_1272_, 6);
v_univApprox_1307_ = lean_ctor_get_uint8(v___y_1272_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_1308_ = lean_ctor_get_uint8(v___y_1272_, sizeof(void*)*7 + 2);
v_cacheInferType_1309_ = lean_ctor_get_uint8(v___y_1272_, sizeof(void*)*7 + 3);
v_isSharedCheck_1409_ = !lean_is_exclusive(v___y_1272_);
if (v_isSharedCheck_1409_ == 0)
{
v___x_1311_ = v___y_1272_;
v_isShared_1312_ = v_isSharedCheck_1409_;
goto v_resetjp_1310_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_1306_);
lean_inc(v_synthPendingDepth_1305_);
lean_inc(v_defEqCtx_x3f_1304_);
lean_inc(v_localInstances_1303_);
lean_inc(v_lctx_1302_);
lean_inc(v_zetaDeltaSet_1301_);
lean_inc(v_keyedConfig_1299_);
lean_dec(v___y_1272_);
v___x_1311_ = lean_box(0);
v_isShared_1312_ = v_isSharedCheck_1409_;
goto v_resetjp_1310_;
}
v_resetjp_1310_:
{
lean_object* v___x_1313_; lean_object* v___x_1314_; lean_object* v___x_1315_; lean_object* v___x_1316_; lean_object* v___x_1317_; lean_object* v___x_1318_; uint8_t v___x_1319_; lean_object* v___x_1320_; lean_object* v___x_1322_; 
v___x_1313_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__5___closed__2));
v___x_1314_ = l_Lean_Expr_const___override(v___x_1313_, v___x_1288_);
lean_inc(v_a_1285_);
v___x_1315_ = l_Lean_Expr_app___override(v___x_1314_, v_a_1285_);
lean_inc(v_a_1293_);
v___x_1316_ = l_Lean_Expr_app___override(v___x_1315_, v_a_1293_);
lean_inc(v_a_1296_);
v___x_1317_ = l_Lean_Expr_app___override(v___x_1316_, v_a_1296_);
lean_inc(v_a_1298_);
v___x_1318_ = l_Lean_Expr_app___override(v___x_1317_, v_a_1298_);
v___x_1319_ = 2;
v___x_1320_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_1319_, v_keyedConfig_1299_);
if (v_isShared_1312_ == 0)
{
lean_ctor_set(v___x_1311_, 0, v___x_1320_);
v___x_1322_ = v___x_1311_;
goto v_reusejp_1321_;
}
else
{
lean_object* v_reuseFailAlloc_1408_; 
v_reuseFailAlloc_1408_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_1408_, 0, v___x_1320_);
lean_ctor_set(v_reuseFailAlloc_1408_, 1, v_zetaDeltaSet_1301_);
lean_ctor_set(v_reuseFailAlloc_1408_, 2, v_lctx_1302_);
lean_ctor_set(v_reuseFailAlloc_1408_, 3, v_localInstances_1303_);
lean_ctor_set(v_reuseFailAlloc_1408_, 4, v_defEqCtx_x3f_1304_);
lean_ctor_set(v_reuseFailAlloc_1408_, 5, v_synthPendingDepth_1305_);
lean_ctor_set(v_reuseFailAlloc_1408_, 6, v_customCanUnfoldPredicate_x3f_1306_);
lean_ctor_set_uint8(v_reuseFailAlloc_1408_, sizeof(void*)*7, v_trackZetaDelta_1300_);
lean_ctor_set_uint8(v_reuseFailAlloc_1408_, sizeof(void*)*7 + 1, v_univApprox_1307_);
lean_ctor_set_uint8(v_reuseFailAlloc_1408_, sizeof(void*)*7 + 2, v_inTypeClassResolution_1308_);
lean_ctor_set_uint8(v_reuseFailAlloc_1408_, sizeof(void*)*7 + 3, v_cacheInferType_1309_);
v___x_1322_ = v_reuseFailAlloc_1408_;
goto v_reusejp_1321_;
}
v_reusejp_1321_:
{
lean_object* v___x_1323_; 
v___x_1323_ = l_Lean_Meta_isExprDefEq(v___x_1318_, v_hyp_1271_, v___x_1322_, v___y_1273_, v___y_1274_, v___y_1275_);
lean_dec_ref(v___x_1322_);
if (lean_obj_tag(v___x_1323_) == 0)
{
lean_object* v_a_1324_; lean_object* v___x_1326_; uint8_t v_isShared_1327_; uint8_t v_isSharedCheck_1399_; 
v_a_1324_ = lean_ctor_get(v___x_1323_, 0);
v_isSharedCheck_1399_ = !lean_is_exclusive(v___x_1323_);
if (v_isSharedCheck_1399_ == 0)
{
v___x_1326_ = v___x_1323_;
v_isShared_1327_ = v_isSharedCheck_1399_;
goto v_resetjp_1325_;
}
else
{
lean_inc(v_a_1324_);
lean_dec(v___x_1323_);
v___x_1326_ = lean_box(0);
v_isShared_1327_ = v_isSharedCheck_1399_;
goto v_resetjp_1325_;
}
v_resetjp_1325_:
{
uint8_t v___x_1328_; 
v___x_1328_ = lean_unbox(v_a_1324_);
if (v___x_1328_ == 0)
{
lean_object* v___x_1329_; lean_object* v___x_1330_; lean_object* v___x_1331_; lean_object* v___x_1332_; lean_object* v___x_1333_; lean_object* v___x_1335_; 
v___x_1329_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1329_, 0, v_a_1298_);
lean_ctor_set(v___x_1329_, 1, v_a_1324_);
v___x_1330_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1330_, 0, v_a_1296_);
lean_ctor_set(v___x_1330_, 1, v___x_1329_);
v___x_1331_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1331_, 0, v_a_1293_);
lean_ctor_set(v___x_1331_, 1, v___x_1330_);
v___x_1332_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1332_, 0, v_a_1285_);
lean_ctor_set(v___x_1332_, 1, v___x_1331_);
v___x_1333_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1333_, 0, v_a_1278_);
lean_ctor_set(v___x_1333_, 1, v___x_1332_);
if (v_isShared_1327_ == 0)
{
lean_ctor_set(v___x_1326_, 0, v___x_1333_);
v___x_1335_ = v___x_1326_;
goto v_reusejp_1334_;
}
else
{
lean_object* v_reuseFailAlloc_1336_; 
v_reuseFailAlloc_1336_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1336_, 0, v___x_1333_);
v___x_1335_ = v_reuseFailAlloc_1336_;
goto v_reusejp_1334_;
}
v_reusejp_1334_:
{
return v___x_1335_;
}
}
else
{
lean_object* v___x_1337_; 
lean_del_object(v___x_1326_);
v___x_1337_ = lp_mathlib_Lean_instantiateLevelMVars___at___00Mathlib_Tactic_Bound_hypPriority_spec__0___redArg(v_a_1278_, v___y_1273_);
if (lean_obj_tag(v___x_1337_) == 0)
{
lean_object* v_a_1338_; lean_object* v___x_1339_; 
v_a_1338_ = lean_ctor_get(v___x_1337_, 0);
lean_inc(v_a_1338_);
lean_dec_ref_known(v___x_1337_, 1);
v___x_1339_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Bound_isZero_spec__0___redArg(v_a_1285_, v___y_1273_);
if (lean_obj_tag(v___x_1339_) == 0)
{
lean_object* v_a_1340_; lean_object* v___x_1341_; 
v_a_1340_ = lean_ctor_get(v___x_1339_, 0);
lean_inc(v_a_1340_);
lean_dec_ref_known(v___x_1339_, 1);
v___x_1341_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Bound_isZero_spec__0___redArg(v_a_1293_, v___y_1273_);
if (lean_obj_tag(v___x_1341_) == 0)
{
lean_object* v_a_1342_; lean_object* v___x_1343_; 
v_a_1342_ = lean_ctor_get(v___x_1341_, 0);
lean_inc(v_a_1342_);
lean_dec_ref_known(v___x_1341_, 1);
v___x_1343_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Bound_isZero_spec__0___redArg(v_a_1296_, v___y_1273_);
if (lean_obj_tag(v___x_1343_) == 0)
{
lean_object* v_a_1344_; lean_object* v___x_1345_; 
v_a_1344_ = lean_ctor_get(v___x_1343_, 0);
lean_inc(v_a_1344_);
lean_dec_ref_known(v___x_1343_, 1);
v___x_1345_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Bound_isZero_spec__0___redArg(v_a_1298_, v___y_1273_);
if (lean_obj_tag(v___x_1345_) == 0)
{
lean_object* v_a_1346_; lean_object* v___x_1348_; uint8_t v_isShared_1349_; uint8_t v_isSharedCheck_1358_; 
v_a_1346_ = lean_ctor_get(v___x_1345_, 0);
v_isSharedCheck_1358_ = !lean_is_exclusive(v___x_1345_);
if (v_isSharedCheck_1358_ == 0)
{
v___x_1348_ = v___x_1345_;
v_isShared_1349_ = v_isSharedCheck_1358_;
goto v_resetjp_1347_;
}
else
{
lean_inc(v_a_1346_);
lean_dec(v___x_1345_);
v___x_1348_ = lean_box(0);
v_isShared_1349_ = v_isSharedCheck_1358_;
goto v_resetjp_1347_;
}
v_resetjp_1347_:
{
lean_object* v___x_1350_; lean_object* v___x_1351_; lean_object* v___x_1352_; lean_object* v___x_1353_; lean_object* v___x_1354_; lean_object* v___x_1356_; 
v___x_1350_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1350_, 0, v_a_1346_);
lean_ctor_set(v___x_1350_, 1, v_a_1324_);
v___x_1351_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1351_, 0, v_a_1344_);
lean_ctor_set(v___x_1351_, 1, v___x_1350_);
v___x_1352_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1352_, 0, v_a_1342_);
lean_ctor_set(v___x_1352_, 1, v___x_1351_);
v___x_1353_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1353_, 0, v_a_1340_);
lean_ctor_set(v___x_1353_, 1, v___x_1352_);
v___x_1354_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1354_, 0, v_a_1338_);
lean_ctor_set(v___x_1354_, 1, v___x_1353_);
if (v_isShared_1349_ == 0)
{
lean_ctor_set(v___x_1348_, 0, v___x_1354_);
v___x_1356_ = v___x_1348_;
goto v_reusejp_1355_;
}
else
{
lean_object* v_reuseFailAlloc_1357_; 
v_reuseFailAlloc_1357_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1357_, 0, v___x_1354_);
v___x_1356_ = v_reuseFailAlloc_1357_;
goto v_reusejp_1355_;
}
v_reusejp_1355_:
{
return v___x_1356_;
}
}
}
else
{
lean_object* v_a_1359_; lean_object* v___x_1361_; uint8_t v_isShared_1362_; uint8_t v_isSharedCheck_1366_; 
lean_dec(v_a_1344_);
lean_dec(v_a_1342_);
lean_dec(v_a_1340_);
lean_dec(v_a_1338_);
lean_dec(v_a_1324_);
v_a_1359_ = lean_ctor_get(v___x_1345_, 0);
v_isSharedCheck_1366_ = !lean_is_exclusive(v___x_1345_);
if (v_isSharedCheck_1366_ == 0)
{
v___x_1361_ = v___x_1345_;
v_isShared_1362_ = v_isSharedCheck_1366_;
goto v_resetjp_1360_;
}
else
{
lean_inc(v_a_1359_);
lean_dec(v___x_1345_);
v___x_1361_ = lean_box(0);
v_isShared_1362_ = v_isSharedCheck_1366_;
goto v_resetjp_1360_;
}
v_resetjp_1360_:
{
lean_object* v___x_1364_; 
if (v_isShared_1362_ == 0)
{
v___x_1364_ = v___x_1361_;
goto v_reusejp_1363_;
}
else
{
lean_object* v_reuseFailAlloc_1365_; 
v_reuseFailAlloc_1365_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1365_, 0, v_a_1359_);
v___x_1364_ = v_reuseFailAlloc_1365_;
goto v_reusejp_1363_;
}
v_reusejp_1363_:
{
return v___x_1364_;
}
}
}
}
else
{
lean_object* v_a_1367_; lean_object* v___x_1369_; uint8_t v_isShared_1370_; uint8_t v_isSharedCheck_1374_; 
lean_dec(v_a_1342_);
lean_dec(v_a_1340_);
lean_dec(v_a_1338_);
lean_dec(v_a_1324_);
lean_dec(v_a_1298_);
v_a_1367_ = lean_ctor_get(v___x_1343_, 0);
v_isSharedCheck_1374_ = !lean_is_exclusive(v___x_1343_);
if (v_isSharedCheck_1374_ == 0)
{
v___x_1369_ = v___x_1343_;
v_isShared_1370_ = v_isSharedCheck_1374_;
goto v_resetjp_1368_;
}
else
{
lean_inc(v_a_1367_);
lean_dec(v___x_1343_);
v___x_1369_ = lean_box(0);
v_isShared_1370_ = v_isSharedCheck_1374_;
goto v_resetjp_1368_;
}
v_resetjp_1368_:
{
lean_object* v___x_1372_; 
if (v_isShared_1370_ == 0)
{
v___x_1372_ = v___x_1369_;
goto v_reusejp_1371_;
}
else
{
lean_object* v_reuseFailAlloc_1373_; 
v_reuseFailAlloc_1373_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1373_, 0, v_a_1367_);
v___x_1372_ = v_reuseFailAlloc_1373_;
goto v_reusejp_1371_;
}
v_reusejp_1371_:
{
return v___x_1372_;
}
}
}
}
else
{
lean_object* v_a_1375_; lean_object* v___x_1377_; uint8_t v_isShared_1378_; uint8_t v_isSharedCheck_1382_; 
lean_dec(v_a_1340_);
lean_dec(v_a_1338_);
lean_dec(v_a_1324_);
lean_dec(v_a_1298_);
lean_dec(v_a_1296_);
v_a_1375_ = lean_ctor_get(v___x_1341_, 0);
v_isSharedCheck_1382_ = !lean_is_exclusive(v___x_1341_);
if (v_isSharedCheck_1382_ == 0)
{
v___x_1377_ = v___x_1341_;
v_isShared_1378_ = v_isSharedCheck_1382_;
goto v_resetjp_1376_;
}
else
{
lean_inc(v_a_1375_);
lean_dec(v___x_1341_);
v___x_1377_ = lean_box(0);
v_isShared_1378_ = v_isSharedCheck_1382_;
goto v_resetjp_1376_;
}
v_resetjp_1376_:
{
lean_object* v___x_1380_; 
if (v_isShared_1378_ == 0)
{
v___x_1380_ = v___x_1377_;
goto v_reusejp_1379_;
}
else
{
lean_object* v_reuseFailAlloc_1381_; 
v_reuseFailAlloc_1381_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1381_, 0, v_a_1375_);
v___x_1380_ = v_reuseFailAlloc_1381_;
goto v_reusejp_1379_;
}
v_reusejp_1379_:
{
return v___x_1380_;
}
}
}
}
else
{
lean_object* v_a_1383_; lean_object* v___x_1385_; uint8_t v_isShared_1386_; uint8_t v_isSharedCheck_1390_; 
lean_dec(v_a_1338_);
lean_dec(v_a_1324_);
lean_dec(v_a_1298_);
lean_dec(v_a_1296_);
lean_dec(v_a_1293_);
v_a_1383_ = lean_ctor_get(v___x_1339_, 0);
v_isSharedCheck_1390_ = !lean_is_exclusive(v___x_1339_);
if (v_isSharedCheck_1390_ == 0)
{
v___x_1385_ = v___x_1339_;
v_isShared_1386_ = v_isSharedCheck_1390_;
goto v_resetjp_1384_;
}
else
{
lean_inc(v_a_1383_);
lean_dec(v___x_1339_);
v___x_1385_ = lean_box(0);
v_isShared_1386_ = v_isSharedCheck_1390_;
goto v_resetjp_1384_;
}
v_resetjp_1384_:
{
lean_object* v___x_1388_; 
if (v_isShared_1386_ == 0)
{
v___x_1388_ = v___x_1385_;
goto v_reusejp_1387_;
}
else
{
lean_object* v_reuseFailAlloc_1389_; 
v_reuseFailAlloc_1389_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1389_, 0, v_a_1383_);
v___x_1388_ = v_reuseFailAlloc_1389_;
goto v_reusejp_1387_;
}
v_reusejp_1387_:
{
return v___x_1388_;
}
}
}
}
else
{
lean_object* v_a_1391_; lean_object* v___x_1393_; uint8_t v_isShared_1394_; uint8_t v_isSharedCheck_1398_; 
lean_dec(v_a_1324_);
lean_dec(v_a_1298_);
lean_dec(v_a_1296_);
lean_dec(v_a_1293_);
lean_dec(v_a_1285_);
v_a_1391_ = lean_ctor_get(v___x_1337_, 0);
v_isSharedCheck_1398_ = !lean_is_exclusive(v___x_1337_);
if (v_isSharedCheck_1398_ == 0)
{
v___x_1393_ = v___x_1337_;
v_isShared_1394_ = v_isSharedCheck_1398_;
goto v_resetjp_1392_;
}
else
{
lean_inc(v_a_1391_);
lean_dec(v___x_1337_);
v___x_1393_ = lean_box(0);
v_isShared_1394_ = v_isSharedCheck_1398_;
goto v_resetjp_1392_;
}
v_resetjp_1392_:
{
lean_object* v___x_1396_; 
if (v_isShared_1394_ == 0)
{
v___x_1396_ = v___x_1393_;
goto v_reusejp_1395_;
}
else
{
lean_object* v_reuseFailAlloc_1397_; 
v_reuseFailAlloc_1397_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1397_, 0, v_a_1391_);
v___x_1396_ = v_reuseFailAlloc_1397_;
goto v_reusejp_1395_;
}
v_reusejp_1395_:
{
return v___x_1396_;
}
}
}
}
}
}
else
{
lean_object* v_a_1400_; lean_object* v___x_1402_; uint8_t v_isShared_1403_; uint8_t v_isSharedCheck_1407_; 
lean_dec(v_a_1298_);
lean_dec(v_a_1296_);
lean_dec(v_a_1293_);
lean_dec(v_a_1285_);
lean_dec(v_a_1278_);
v_a_1400_ = lean_ctor_get(v___x_1323_, 0);
v_isSharedCheck_1407_ = !lean_is_exclusive(v___x_1323_);
if (v_isSharedCheck_1407_ == 0)
{
v___x_1402_ = v___x_1323_;
v_isShared_1403_ = v_isSharedCheck_1407_;
goto v_resetjp_1401_;
}
else
{
lean_inc(v_a_1400_);
lean_dec(v___x_1323_);
v___x_1402_ = lean_box(0);
v_isShared_1403_ = v_isSharedCheck_1407_;
goto v_resetjp_1401_;
}
v_resetjp_1401_:
{
lean_object* v___x_1405_; 
if (v_isShared_1403_ == 0)
{
v___x_1405_ = v___x_1402_;
goto v_reusejp_1404_;
}
else
{
lean_object* v_reuseFailAlloc_1406_; 
v_reuseFailAlloc_1406_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1406_, 0, v_a_1400_);
v___x_1405_ = v_reuseFailAlloc_1406_;
goto v_reusejp_1404_;
}
v_reusejp_1404_:
{
return v___x_1405_;
}
}
}
}
}
}
else
{
lean_object* v_a_1410_; lean_object* v___x_1412_; uint8_t v_isShared_1413_; uint8_t v_isSharedCheck_1417_; 
lean_dec(v_a_1296_);
lean_dec(v_a_1293_);
lean_dec_ref_known(v___x_1288_, 2);
lean_dec(v_a_1285_);
lean_dec(v_a_1278_);
lean_dec_ref(v___y_1272_);
lean_dec_ref(v_hyp_1271_);
v_a_1410_ = lean_ctor_get(v___x_1297_, 0);
v_isSharedCheck_1417_ = !lean_is_exclusive(v___x_1297_);
if (v_isSharedCheck_1417_ == 0)
{
v___x_1412_ = v___x_1297_;
v_isShared_1413_ = v_isSharedCheck_1417_;
goto v_resetjp_1411_;
}
else
{
lean_inc(v_a_1410_);
lean_dec(v___x_1297_);
v___x_1412_ = lean_box(0);
v_isShared_1413_ = v_isSharedCheck_1417_;
goto v_resetjp_1411_;
}
v_resetjp_1411_:
{
lean_object* v___x_1415_; 
if (v_isShared_1413_ == 0)
{
v___x_1415_ = v___x_1412_;
goto v_reusejp_1414_;
}
else
{
lean_object* v_reuseFailAlloc_1416_; 
v_reuseFailAlloc_1416_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1416_, 0, v_a_1410_);
v___x_1415_ = v_reuseFailAlloc_1416_;
goto v_reusejp_1414_;
}
v_reusejp_1414_:
{
return v___x_1415_;
}
}
}
}
else
{
lean_object* v_a_1418_; lean_object* v___x_1420_; uint8_t v_isShared_1421_; uint8_t v_isSharedCheck_1425_; 
lean_dec_ref_known(v___x_1294_, 1);
lean_dec(v_a_1293_);
lean_dec_ref_known(v___x_1288_, 2);
lean_dec(v_a_1285_);
lean_dec(v_a_1278_);
lean_dec_ref(v___y_1272_);
lean_dec_ref(v_hyp_1271_);
v_a_1418_ = lean_ctor_get(v___x_1295_, 0);
v_isSharedCheck_1425_ = !lean_is_exclusive(v___x_1295_);
if (v_isSharedCheck_1425_ == 0)
{
v___x_1420_ = v___x_1295_;
v_isShared_1421_ = v_isSharedCheck_1425_;
goto v_resetjp_1419_;
}
else
{
lean_inc(v_a_1418_);
lean_dec(v___x_1295_);
v___x_1420_ = lean_box(0);
v_isShared_1421_ = v_isSharedCheck_1425_;
goto v_resetjp_1419_;
}
v_resetjp_1419_:
{
lean_object* v___x_1423_; 
if (v_isShared_1421_ == 0)
{
v___x_1423_ = v___x_1420_;
goto v_reusejp_1422_;
}
else
{
lean_object* v_reuseFailAlloc_1424_; 
v_reuseFailAlloc_1424_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1424_, 0, v_a_1418_);
v___x_1423_ = v_reuseFailAlloc_1424_;
goto v_reusejp_1422_;
}
v_reusejp_1422_:
{
return v___x_1423_;
}
}
}
}
else
{
lean_object* v_a_1426_; lean_object* v___x_1428_; uint8_t v_isShared_1429_; uint8_t v_isSharedCheck_1433_; 
lean_dec_ref_known(v___x_1288_, 2);
lean_dec(v_a_1285_);
lean_dec(v_a_1278_);
lean_dec_ref(v___y_1272_);
lean_dec_ref(v_hyp_1271_);
v_a_1426_ = lean_ctor_get(v___x_1292_, 0);
v_isSharedCheck_1433_ = !lean_is_exclusive(v___x_1292_);
if (v_isSharedCheck_1433_ == 0)
{
v___x_1428_ = v___x_1292_;
v_isShared_1429_ = v_isSharedCheck_1433_;
goto v_resetjp_1427_;
}
else
{
lean_inc(v_a_1426_);
lean_dec(v___x_1292_);
v___x_1428_ = lean_box(0);
v_isShared_1429_ = v_isSharedCheck_1433_;
goto v_resetjp_1427_;
}
v_resetjp_1427_:
{
lean_object* v___x_1431_; 
if (v_isShared_1429_ == 0)
{
v___x_1431_ = v___x_1428_;
goto v_reusejp_1430_;
}
else
{
lean_object* v_reuseFailAlloc_1432_; 
v_reuseFailAlloc_1432_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1432_, 0, v_a_1426_);
v___x_1431_ = v_reuseFailAlloc_1432_;
goto v_reusejp_1430_;
}
v_reusejp_1430_:
{
return v___x_1431_;
}
}
}
}
else
{
lean_object* v_a_1434_; lean_object* v___x_1436_; uint8_t v_isShared_1437_; uint8_t v_isSharedCheck_1441_; 
lean_dec(v_a_1278_);
lean_dec_ref(v___y_1272_);
lean_dec_ref(v_hyp_1271_);
v_a_1434_ = lean_ctor_get(v___x_1284_, 0);
v_isSharedCheck_1441_ = !lean_is_exclusive(v___x_1284_);
if (v_isSharedCheck_1441_ == 0)
{
v___x_1436_ = v___x_1284_;
v_isShared_1437_ = v_isSharedCheck_1441_;
goto v_resetjp_1435_;
}
else
{
lean_inc(v_a_1434_);
lean_dec(v___x_1284_);
v___x_1436_ = lean_box(0);
v_isShared_1437_ = v_isSharedCheck_1441_;
goto v_resetjp_1435_;
}
v_resetjp_1435_:
{
lean_object* v___x_1439_; 
if (v_isShared_1437_ == 0)
{
v___x_1439_ = v___x_1436_;
goto v_reusejp_1438_;
}
else
{
lean_object* v_reuseFailAlloc_1440_; 
v_reuseFailAlloc_1440_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1440_, 0, v_a_1434_);
v___x_1439_ = v_reuseFailAlloc_1440_;
goto v_reusejp_1438_;
}
v_reusejp_1438_:
{
return v___x_1439_;
}
}
}
}
else
{
lean_object* v_a_1442_; lean_object* v___x_1444_; uint8_t v_isShared_1445_; uint8_t v_isSharedCheck_1449_; 
lean_dec_ref(v___y_1272_);
lean_dec_ref(v_hyp_1271_);
v_a_1442_ = lean_ctor_get(v___x_1277_, 0);
v_isSharedCheck_1449_ = !lean_is_exclusive(v___x_1277_);
if (v_isSharedCheck_1449_ == 0)
{
v___x_1444_ = v___x_1277_;
v_isShared_1445_ = v_isSharedCheck_1449_;
goto v_resetjp_1443_;
}
else
{
lean_inc(v_a_1442_);
lean_dec(v___x_1277_);
v___x_1444_ = lean_box(0);
v_isShared_1445_ = v_isSharedCheck_1449_;
goto v_resetjp_1443_;
}
v_resetjp_1443_:
{
lean_object* v___x_1447_; 
if (v_isShared_1445_ == 0)
{
v___x_1447_ = v___x_1444_;
goto v_reusejp_1446_;
}
else
{
lean_object* v_reuseFailAlloc_1448_; 
v_reuseFailAlloc_1448_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1448_, 0, v_a_1442_);
v___x_1447_ = v_reuseFailAlloc_1448_;
goto v_reusejp_1446_;
}
v_reusejp_1446_:
{
return v___x_1447_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__5___boxed(lean_object* v_hyp_1450_, lean_object* v___y_1451_, lean_object* v___y_1452_, lean_object* v___y_1453_, lean_object* v___y_1454_, lean_object* v___y_1455_){
_start:
{
lean_object* v_res_1456_; 
v_res_1456_ = lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__5(v_hyp_1450_, v___y_1451_, v___y_1452_, v___y_1453_, v___y_1454_);
lean_dec(v___y_1454_);
lean_dec_ref(v___y_1453_);
lean_dec(v___y_1452_);
return v_res_1456_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Bound_hypPriority___closed__0(void){
_start:
{
lean_object* v___x_1457_; lean_object* v___x_1458_; 
v___x_1457_ = lean_box(0);
v___x_1458_ = l_Lean_Expr_sort___override(v___x_1457_);
return v___x_1458_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Bound_hypPriority___closed__1(void){
_start:
{
lean_object* v___x_1459_; lean_object* v___x_1460_; 
v___x_1459_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Bound_hypPriority___closed__0, &lp_mathlib_Mathlib_Tactic_Bound_hypPriority___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Bound_hypPriority___closed__0);
v___x_1460_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1460_, 0, v___x_1459_);
return v___x_1460_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound_hypPriority(lean_object* v_hyp_1461_, lean_object* v_a_1462_, lean_object* v_a_1463_, lean_object* v_a_1464_, lean_object* v_a_1465_){
_start:
{
uint8_t v___x_1467_; lean_object* v___x_1468_; uint8_t v___x_1469_; lean_object* v___x_1470_; lean_object* v___x_1471_; lean_object* v___x_1472_; lean_object* v___f_1473_; lean_object* v___x_1474_; 
v___x_1467_ = 0;
v___x_1468_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Bound_hypPriority___closed__1, &lp_mathlib_Mathlib_Tactic_Bound_hypPriority___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Bound_hypPriority___closed__1);
v___x_1469_ = 0;
v___x_1470_ = lean_box(0);
v___x_1471_ = lean_box(v___x_1469_);
v___x_1472_ = lean_box(v___x_1467_);
lean_inc_ref(v_hyp_1461_);
v___f_1473_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__0___boxed), 10, 5);
lean_closure_set(v___f_1473_, 0, v___x_1468_);
lean_closure_set(v___f_1473_, 1, v___x_1471_);
lean_closure_set(v___f_1473_, 2, v___x_1470_);
lean_closure_set(v___f_1473_, 3, v_hyp_1461_);
lean_closure_set(v___f_1473_, 4, v___x_1472_);
v___x_1474_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Bound_isZero_spec__1___redArg(v___f_1473_, v___x_1467_, v_a_1462_, v_a_1463_, v_a_1464_, v_a_1465_);
if (lean_obj_tag(v___x_1474_) == 0)
{
lean_object* v_a_1475_; lean_object* v_snd_1476_; lean_object* v_snd_1477_; uint8_t v___x_1478_; 
v_a_1475_ = lean_ctor_get(v___x_1474_, 0);
lean_inc(v_a_1475_);
lean_dec_ref_known(v___x_1474_, 1);
v_snd_1476_ = lean_ctor_get(v_a_1475_, 1);
lean_inc(v_snd_1476_);
v_snd_1477_ = lean_ctor_get(v_snd_1476_, 1);
v___x_1478_ = lean_unbox(v_snd_1477_);
if (v___x_1478_ == 0)
{
lean_object* v___x_1479_; lean_object* v___x_1480_; lean_object* v___f_1481_; lean_object* v___x_1482_; 
lean_dec(v_snd_1476_);
lean_dec(v_a_1475_);
v___x_1479_ = lean_box(v___x_1469_);
v___x_1480_ = lean_box(v___x_1467_);
lean_inc_ref(v_hyp_1461_);
v___f_1481_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__1___boxed), 10, 5);
lean_closure_set(v___f_1481_, 0, v___x_1468_);
lean_closure_set(v___f_1481_, 1, v___x_1479_);
lean_closure_set(v___f_1481_, 2, v___x_1470_);
lean_closure_set(v___f_1481_, 3, v_hyp_1461_);
lean_closure_set(v___f_1481_, 4, v___x_1480_);
v___x_1482_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Bound_isZero_spec__1___redArg(v___f_1481_, v___x_1467_, v_a_1462_, v_a_1463_, v_a_1464_, v_a_1465_);
if (lean_obj_tag(v___x_1482_) == 0)
{
lean_object* v_a_1483_; lean_object* v_snd_1484_; lean_object* v_snd_1485_; uint8_t v___x_1486_; 
v_a_1483_ = lean_ctor_get(v___x_1482_, 0);
lean_inc(v_a_1483_);
lean_dec_ref_known(v___x_1482_, 1);
v_snd_1484_ = lean_ctor_get(v_a_1483_, 1);
lean_inc(v_snd_1484_);
v_snd_1485_ = lean_ctor_get(v_snd_1484_, 1);
v___x_1486_ = lean_unbox(v_snd_1485_);
if (v___x_1486_ == 0)
{
lean_object* v___x_1487_; lean_object* v___f_1488_; lean_object* v___x_1489_; 
lean_dec(v_snd_1484_);
lean_dec(v_a_1483_);
v___x_1487_ = lean_box(v___x_1467_);
lean_inc_ref(v_hyp_1461_);
v___f_1488_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__2___boxed), 7, 2);
lean_closure_set(v___f_1488_, 0, v_hyp_1461_);
lean_closure_set(v___f_1488_, 1, v___x_1487_);
v___x_1489_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Bound_isZero_spec__1___redArg(v___f_1488_, v___x_1467_, v_a_1462_, v_a_1463_, v_a_1464_, v_a_1465_);
if (lean_obj_tag(v___x_1489_) == 0)
{
lean_object* v_a_1490_; lean_object* v_snd_1491_; lean_object* v_snd_1492_; lean_object* v_snd_1493_; lean_object* v_snd_1494_; lean_object* v_snd_1495_; uint8_t v___x_1496_; 
v_a_1490_ = lean_ctor_get(v___x_1489_, 0);
lean_inc(v_a_1490_);
lean_dec_ref_known(v___x_1489_, 1);
v_snd_1491_ = lean_ctor_get(v_a_1490_, 1);
lean_inc(v_snd_1491_);
v_snd_1492_ = lean_ctor_get(v_snd_1491_, 1);
v_snd_1493_ = lean_ctor_get(v_snd_1492_, 1);
lean_inc(v_snd_1493_);
v_snd_1494_ = lean_ctor_get(v_snd_1493_, 1);
lean_inc(v_snd_1494_);
v_snd_1495_ = lean_ctor_get(v_snd_1494_, 1);
v___x_1496_ = lean_unbox(v_snd_1495_);
if (v___x_1496_ == 0)
{
lean_object* v___x_1497_; lean_object* v___f_1498_; lean_object* v___x_1499_; 
lean_dec(v_snd_1494_);
lean_dec(v_snd_1493_);
lean_dec(v_snd_1491_);
lean_dec(v_a_1490_);
v___x_1497_ = lean_box(v___x_1467_);
lean_inc_ref(v_hyp_1461_);
v___f_1498_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__3___boxed), 7, 2);
lean_closure_set(v___f_1498_, 0, v_hyp_1461_);
lean_closure_set(v___f_1498_, 1, v___x_1497_);
v___x_1499_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Bound_isZero_spec__1___redArg(v___f_1498_, v___x_1467_, v_a_1462_, v_a_1463_, v_a_1464_, v_a_1465_);
if (lean_obj_tag(v___x_1499_) == 0)
{
lean_object* v_a_1500_; lean_object* v_snd_1501_; lean_object* v_snd_1502_; lean_object* v_snd_1503_; lean_object* v_snd_1504_; lean_object* v_snd_1505_; uint8_t v___x_1506_; 
v_a_1500_ = lean_ctor_get(v___x_1499_, 0);
lean_inc(v_a_1500_);
lean_dec_ref_known(v___x_1499_, 1);
v_snd_1501_ = lean_ctor_get(v_a_1500_, 1);
lean_inc(v_snd_1501_);
v_snd_1502_ = lean_ctor_get(v_snd_1501_, 1);
v_snd_1503_ = lean_ctor_get(v_snd_1502_, 1);
lean_inc(v_snd_1503_);
v_snd_1504_ = lean_ctor_get(v_snd_1503_, 1);
lean_inc(v_snd_1504_);
v_snd_1505_ = lean_ctor_get(v_snd_1504_, 1);
v___x_1506_ = lean_unbox(v_snd_1505_);
if (v___x_1506_ == 0)
{
lean_object* v___x_1507_; lean_object* v___f_1508_; lean_object* v___x_1509_; 
lean_dec(v_snd_1504_);
lean_dec(v_snd_1503_);
lean_dec(v_snd_1501_);
lean_dec(v_a_1500_);
v___x_1507_ = lean_box(v___x_1467_);
lean_inc_ref(v_hyp_1461_);
v___f_1508_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__4___boxed), 7, 2);
lean_closure_set(v___f_1508_, 0, v_hyp_1461_);
lean_closure_set(v___f_1508_, 1, v___x_1507_);
v___x_1509_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Bound_isZero_spec__1___redArg(v___f_1508_, v___x_1467_, v_a_1462_, v_a_1463_, v_a_1464_, v_a_1465_);
if (lean_obj_tag(v___x_1509_) == 0)
{
lean_object* v_a_1510_; lean_object* v_snd_1511_; lean_object* v_snd_1512_; lean_object* v_snd_1513_; lean_object* v_snd_1514_; lean_object* v_snd_1515_; uint8_t v___x_1516_; 
v_a_1510_ = lean_ctor_get(v___x_1509_, 0);
lean_inc(v_a_1510_);
lean_dec_ref_known(v___x_1509_, 1);
v_snd_1511_ = lean_ctor_get(v_a_1510_, 1);
lean_inc(v_snd_1511_);
v_snd_1512_ = lean_ctor_get(v_snd_1511_, 1);
v_snd_1513_ = lean_ctor_get(v_snd_1512_, 1);
lean_inc(v_snd_1513_);
v_snd_1514_ = lean_ctor_get(v_snd_1513_, 1);
lean_inc(v_snd_1514_);
v_snd_1515_ = lean_ctor_get(v_snd_1514_, 1);
v___x_1516_ = lean_unbox(v_snd_1515_);
if (v___x_1516_ == 0)
{
lean_object* v___f_1517_; lean_object* v___x_1518_; 
lean_dec(v_snd_1514_);
lean_dec(v_snd_1513_);
lean_dec(v_snd_1511_);
lean_dec(v_a_1510_);
v___f_1517_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__5___boxed), 6, 1);
lean_closure_set(v___f_1517_, 0, v_hyp_1461_);
v___x_1518_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Bound_isZero_spec__1___redArg(v___f_1517_, v___x_1467_, v_a_1462_, v_a_1463_, v_a_1464_, v_a_1465_);
if (lean_obj_tag(v___x_1518_) == 0)
{
lean_object* v_a_1519_; lean_object* v___x_1521_; uint8_t v_isShared_1522_; uint8_t v_isSharedCheck_1538_; 
v_a_1519_ = lean_ctor_get(v___x_1518_, 0);
v_isSharedCheck_1538_ = !lean_is_exclusive(v___x_1518_);
if (v_isSharedCheck_1538_ == 0)
{
v___x_1521_ = v___x_1518_;
v_isShared_1522_ = v_isSharedCheck_1538_;
goto v_resetjp_1520_;
}
else
{
lean_inc(v_a_1519_);
lean_dec(v___x_1518_);
v___x_1521_ = lean_box(0);
v_isShared_1522_ = v_isSharedCheck_1538_;
goto v_resetjp_1520_;
}
v_resetjp_1520_:
{
lean_object* v_snd_1523_; lean_object* v_snd_1524_; lean_object* v_snd_1525_; lean_object* v_snd_1526_; lean_object* v_snd_1527_; uint8_t v___x_1528_; 
v_snd_1523_ = lean_ctor_get(v_a_1519_, 1);
lean_inc(v_snd_1523_);
v_snd_1524_ = lean_ctor_get(v_snd_1523_, 1);
v_snd_1525_ = lean_ctor_get(v_snd_1524_, 1);
lean_inc(v_snd_1525_);
v_snd_1526_ = lean_ctor_get(v_snd_1525_, 1);
lean_inc(v_snd_1526_);
v_snd_1527_ = lean_ctor_get(v_snd_1526_, 1);
v___x_1528_ = lean_unbox(v_snd_1527_);
if (v___x_1528_ == 0)
{
lean_object* v___x_1529_; lean_object* v___x_1531_; 
lean_dec(v_snd_1526_);
lean_dec(v_snd_1525_);
lean_dec(v_snd_1523_);
lean_dec(v_a_1519_);
v___x_1529_ = lean_unsigned_to_nat(0u);
if (v_isShared_1522_ == 0)
{
lean_ctor_set(v___x_1521_, 0, v___x_1529_);
v___x_1531_ = v___x_1521_;
goto v_reusejp_1530_;
}
else
{
lean_object* v_reuseFailAlloc_1532_; 
v_reuseFailAlloc_1532_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1532_, 0, v___x_1529_);
v___x_1531_ = v_reuseFailAlloc_1532_;
goto v_reusejp_1530_;
}
v_reusejp_1530_:
{
return v___x_1531_;
}
}
else
{
lean_object* v_fst_1533_; lean_object* v_fst_1534_; lean_object* v_fst_1535_; lean_object* v_fst_1536_; lean_object* v___x_1537_; 
lean_del_object(v___x_1521_);
v_fst_1533_ = lean_ctor_get(v_a_1519_, 0);
lean_inc(v_fst_1533_);
lean_dec(v_a_1519_);
v_fst_1534_ = lean_ctor_get(v_snd_1523_, 0);
lean_inc(v_fst_1534_);
lean_dec(v_snd_1523_);
v_fst_1535_ = lean_ctor_get(v_snd_1525_, 0);
lean_inc(v_fst_1535_);
lean_dec(v_snd_1525_);
v_fst_1536_ = lean_ctor_get(v_snd_1526_, 0);
lean_inc(v_fst_1536_);
lean_dec(v_snd_1526_);
v___x_1537_ = lp_mathlib_Mathlib_Tactic_Bound_ineqPriority(v_fst_1533_, v_fst_1534_, v_fst_1536_, v_fst_1535_, v_a_1462_, v_a_1463_, v_a_1464_, v_a_1465_);
return v___x_1537_;
}
}
}
else
{
lean_object* v_a_1539_; lean_object* v___x_1541_; uint8_t v_isShared_1542_; uint8_t v_isSharedCheck_1546_; 
v_a_1539_ = lean_ctor_get(v___x_1518_, 0);
v_isSharedCheck_1546_ = !lean_is_exclusive(v___x_1518_);
if (v_isSharedCheck_1546_ == 0)
{
v___x_1541_ = v___x_1518_;
v_isShared_1542_ = v_isSharedCheck_1546_;
goto v_resetjp_1540_;
}
else
{
lean_inc(v_a_1539_);
lean_dec(v___x_1518_);
v___x_1541_ = lean_box(0);
v_isShared_1542_ = v_isSharedCheck_1546_;
goto v_resetjp_1540_;
}
v_resetjp_1540_:
{
lean_object* v___x_1544_; 
if (v_isShared_1542_ == 0)
{
v___x_1544_ = v___x_1541_;
goto v_reusejp_1543_;
}
else
{
lean_object* v_reuseFailAlloc_1545_; 
v_reuseFailAlloc_1545_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1545_, 0, v_a_1539_);
v___x_1544_ = v_reuseFailAlloc_1545_;
goto v_reusejp_1543_;
}
v_reusejp_1543_:
{
return v___x_1544_;
}
}
}
}
else
{
lean_object* v_fst_1547_; lean_object* v_fst_1548_; lean_object* v_fst_1549_; lean_object* v_fst_1550_; lean_object* v___x_1551_; 
lean_dec_ref(v_hyp_1461_);
v_fst_1547_ = lean_ctor_get(v_a_1510_, 0);
lean_inc(v_fst_1547_);
lean_dec(v_a_1510_);
v_fst_1548_ = lean_ctor_get(v_snd_1511_, 0);
lean_inc(v_fst_1548_);
lean_dec(v_snd_1511_);
v_fst_1549_ = lean_ctor_get(v_snd_1513_, 0);
lean_inc(v_fst_1549_);
lean_dec(v_snd_1513_);
v_fst_1550_ = lean_ctor_get(v_snd_1514_, 0);
lean_inc(v_fst_1550_);
lean_dec(v_snd_1514_);
v___x_1551_ = lp_mathlib_Mathlib_Tactic_Bound_ineqPriority(v_fst_1547_, v_fst_1548_, v_fst_1550_, v_fst_1549_, v_a_1462_, v_a_1463_, v_a_1464_, v_a_1465_);
return v___x_1551_;
}
}
else
{
lean_object* v_a_1552_; lean_object* v___x_1554_; uint8_t v_isShared_1555_; uint8_t v_isSharedCheck_1559_; 
lean_dec_ref(v_hyp_1461_);
v_a_1552_ = lean_ctor_get(v___x_1509_, 0);
v_isSharedCheck_1559_ = !lean_is_exclusive(v___x_1509_);
if (v_isSharedCheck_1559_ == 0)
{
v___x_1554_ = v___x_1509_;
v_isShared_1555_ = v_isSharedCheck_1559_;
goto v_resetjp_1553_;
}
else
{
lean_inc(v_a_1552_);
lean_dec(v___x_1509_);
v___x_1554_ = lean_box(0);
v_isShared_1555_ = v_isSharedCheck_1559_;
goto v_resetjp_1553_;
}
v_resetjp_1553_:
{
lean_object* v___x_1557_; 
if (v_isShared_1555_ == 0)
{
v___x_1557_ = v___x_1554_;
goto v_reusejp_1556_;
}
else
{
lean_object* v_reuseFailAlloc_1558_; 
v_reuseFailAlloc_1558_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1558_, 0, v_a_1552_);
v___x_1557_ = v_reuseFailAlloc_1558_;
goto v_reusejp_1556_;
}
v_reusejp_1556_:
{
return v___x_1557_;
}
}
}
}
else
{
lean_object* v_fst_1560_; lean_object* v_fst_1561_; lean_object* v_fst_1562_; lean_object* v_fst_1563_; lean_object* v___x_1564_; 
lean_dec_ref(v_hyp_1461_);
v_fst_1560_ = lean_ctor_get(v_a_1500_, 0);
lean_inc(v_fst_1560_);
lean_dec(v_a_1500_);
v_fst_1561_ = lean_ctor_get(v_snd_1501_, 0);
lean_inc(v_fst_1561_);
lean_dec(v_snd_1501_);
v_fst_1562_ = lean_ctor_get(v_snd_1503_, 0);
lean_inc(v_fst_1562_);
lean_dec(v_snd_1503_);
v_fst_1563_ = lean_ctor_get(v_snd_1504_, 0);
lean_inc(v_fst_1563_);
lean_dec(v_snd_1504_);
v___x_1564_ = lp_mathlib_Mathlib_Tactic_Bound_ineqPriority(v_fst_1560_, v_fst_1561_, v_fst_1562_, v_fst_1563_, v_a_1462_, v_a_1463_, v_a_1464_, v_a_1465_);
return v___x_1564_;
}
}
else
{
lean_object* v_a_1565_; lean_object* v___x_1567_; uint8_t v_isShared_1568_; uint8_t v_isSharedCheck_1572_; 
lean_dec_ref(v_hyp_1461_);
v_a_1565_ = lean_ctor_get(v___x_1499_, 0);
v_isSharedCheck_1572_ = !lean_is_exclusive(v___x_1499_);
if (v_isSharedCheck_1572_ == 0)
{
v___x_1567_ = v___x_1499_;
v_isShared_1568_ = v_isSharedCheck_1572_;
goto v_resetjp_1566_;
}
else
{
lean_inc(v_a_1565_);
lean_dec(v___x_1499_);
v___x_1567_ = lean_box(0);
v_isShared_1568_ = v_isSharedCheck_1572_;
goto v_resetjp_1566_;
}
v_resetjp_1566_:
{
lean_object* v___x_1570_; 
if (v_isShared_1568_ == 0)
{
v___x_1570_ = v___x_1567_;
goto v_reusejp_1569_;
}
else
{
lean_object* v_reuseFailAlloc_1571_; 
v_reuseFailAlloc_1571_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1571_, 0, v_a_1565_);
v___x_1570_ = v_reuseFailAlloc_1571_;
goto v_reusejp_1569_;
}
v_reusejp_1569_:
{
return v___x_1570_;
}
}
}
}
else
{
lean_object* v_fst_1573_; lean_object* v_fst_1574_; lean_object* v_fst_1575_; lean_object* v_fst_1576_; lean_object* v___x_1577_; 
lean_dec_ref(v_hyp_1461_);
v_fst_1573_ = lean_ctor_get(v_a_1490_, 0);
lean_inc(v_fst_1573_);
lean_dec(v_a_1490_);
v_fst_1574_ = lean_ctor_get(v_snd_1491_, 0);
lean_inc(v_fst_1574_);
lean_dec(v_snd_1491_);
v_fst_1575_ = lean_ctor_get(v_snd_1493_, 0);
lean_inc(v_fst_1575_);
lean_dec(v_snd_1493_);
v_fst_1576_ = lean_ctor_get(v_snd_1494_, 0);
lean_inc(v_fst_1576_);
lean_dec(v_snd_1494_);
v___x_1577_ = lp_mathlib_Mathlib_Tactic_Bound_ineqPriority(v_fst_1573_, v_fst_1574_, v_fst_1575_, v_fst_1576_, v_a_1462_, v_a_1463_, v_a_1464_, v_a_1465_);
return v___x_1577_;
}
}
else
{
lean_object* v_a_1578_; lean_object* v___x_1580_; uint8_t v_isShared_1581_; uint8_t v_isSharedCheck_1585_; 
lean_dec_ref(v_hyp_1461_);
v_a_1578_ = lean_ctor_get(v___x_1489_, 0);
v_isSharedCheck_1585_ = !lean_is_exclusive(v___x_1489_);
if (v_isSharedCheck_1585_ == 0)
{
v___x_1580_ = v___x_1489_;
v_isShared_1581_ = v_isSharedCheck_1585_;
goto v_resetjp_1579_;
}
else
{
lean_inc(v_a_1578_);
lean_dec(v___x_1489_);
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
else
{
lean_object* v_fst_1586_; lean_object* v_fst_1587_; lean_object* v___x_1588_; 
lean_dec_ref(v_hyp_1461_);
v_fst_1586_ = lean_ctor_get(v_a_1483_, 0);
lean_inc(v_fst_1586_);
lean_dec(v_a_1483_);
v_fst_1587_ = lean_ctor_get(v_snd_1484_, 0);
lean_inc(v_fst_1587_);
lean_dec(v_snd_1484_);
v___x_1588_ = lp_mathlib_Mathlib_Tactic_Bound_hypPriority(v_fst_1586_, v_a_1462_, v_a_1463_, v_a_1464_, v_a_1465_);
if (lean_obj_tag(v___x_1588_) == 0)
{
lean_object* v_a_1589_; lean_object* v___x_1590_; 
v_a_1589_ = lean_ctor_get(v___x_1588_, 0);
lean_inc(v_a_1589_);
lean_dec_ref_known(v___x_1588_, 1);
v___x_1590_ = lp_mathlib_Mathlib_Tactic_Bound_hypPriority(v_fst_1587_, v_a_1462_, v_a_1463_, v_a_1464_, v_a_1465_);
if (lean_obj_tag(v___x_1590_) == 0)
{
lean_object* v_a_1591_; lean_object* v___x_1593_; uint8_t v_isShared_1594_; uint8_t v_isSharedCheck_1601_; 
v_a_1591_ = lean_ctor_get(v___x_1590_, 0);
v_isSharedCheck_1601_ = !lean_is_exclusive(v___x_1590_);
if (v_isSharedCheck_1601_ == 0)
{
v___x_1593_ = v___x_1590_;
v_isShared_1594_ = v_isSharedCheck_1601_;
goto v_resetjp_1592_;
}
else
{
lean_inc(v_a_1591_);
lean_dec(v___x_1590_);
v___x_1593_ = lean_box(0);
v_isShared_1594_ = v_isSharedCheck_1601_;
goto v_resetjp_1592_;
}
v_resetjp_1592_:
{
lean_object* v___x_1595_; lean_object* v___x_1596_; lean_object* v___x_1597_; lean_object* v___x_1599_; 
v___x_1595_ = lean_unsigned_to_nat(100u);
v___x_1596_ = lean_nat_add(v___x_1595_, v_a_1589_);
lean_dec(v_a_1589_);
v___x_1597_ = lean_nat_add(v___x_1596_, v_a_1591_);
lean_dec(v_a_1591_);
lean_dec(v___x_1596_);
if (v_isShared_1594_ == 0)
{
lean_ctor_set(v___x_1593_, 0, v___x_1597_);
v___x_1599_ = v___x_1593_;
goto v_reusejp_1598_;
}
else
{
lean_object* v_reuseFailAlloc_1600_; 
v_reuseFailAlloc_1600_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1600_, 0, v___x_1597_);
v___x_1599_ = v_reuseFailAlloc_1600_;
goto v_reusejp_1598_;
}
v_reusejp_1598_:
{
return v___x_1599_;
}
}
}
else
{
lean_dec(v_a_1589_);
return v___x_1590_;
}
}
else
{
lean_dec(v_fst_1587_);
return v___x_1588_;
}
}
}
else
{
lean_object* v_a_1602_; lean_object* v___x_1604_; uint8_t v_isShared_1605_; uint8_t v_isSharedCheck_1609_; 
lean_dec_ref(v_hyp_1461_);
v_a_1602_ = lean_ctor_get(v___x_1482_, 0);
v_isSharedCheck_1609_ = !lean_is_exclusive(v___x_1482_);
if (v_isSharedCheck_1609_ == 0)
{
v___x_1604_ = v___x_1482_;
v_isShared_1605_ = v_isSharedCheck_1609_;
goto v_resetjp_1603_;
}
else
{
lean_inc(v_a_1602_);
lean_dec(v___x_1482_);
v___x_1604_ = lean_box(0);
v_isShared_1605_ = v_isSharedCheck_1609_;
goto v_resetjp_1603_;
}
v_resetjp_1603_:
{
lean_object* v___x_1607_; 
if (v_isShared_1605_ == 0)
{
v___x_1607_ = v___x_1604_;
goto v_reusejp_1606_;
}
else
{
lean_object* v_reuseFailAlloc_1608_; 
v_reuseFailAlloc_1608_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1608_, 0, v_a_1602_);
v___x_1607_ = v_reuseFailAlloc_1608_;
goto v_reusejp_1606_;
}
v_reusejp_1606_:
{
return v___x_1607_;
}
}
}
}
else
{
lean_object* v_fst_1610_; lean_object* v_fst_1611_; lean_object* v___x_1612_; 
lean_dec_ref(v_hyp_1461_);
v_fst_1610_ = lean_ctor_get(v_a_1475_, 0);
lean_inc(v_fst_1610_);
lean_dec(v_a_1475_);
v_fst_1611_ = lean_ctor_get(v_snd_1476_, 0);
lean_inc(v_fst_1611_);
lean_dec(v_snd_1476_);
v___x_1612_ = lp_mathlib_Mathlib_Tactic_Bound_hypPriority(v_fst_1610_, v_a_1462_, v_a_1463_, v_a_1464_, v_a_1465_);
if (lean_obj_tag(v___x_1612_) == 0)
{
lean_object* v_a_1613_; lean_object* v___x_1614_; 
v_a_1613_ = lean_ctor_get(v___x_1612_, 0);
lean_inc(v_a_1613_);
lean_dec_ref_known(v___x_1612_, 1);
v___x_1614_ = lp_mathlib_Mathlib_Tactic_Bound_hypPriority(v_fst_1611_, v_a_1462_, v_a_1463_, v_a_1464_, v_a_1465_);
if (lean_obj_tag(v___x_1614_) == 0)
{
lean_object* v_a_1615_; lean_object* v___x_1617_; uint8_t v_isShared_1618_; uint8_t v_isSharedCheck_1623_; 
v_a_1615_ = lean_ctor_get(v___x_1614_, 0);
v_isSharedCheck_1623_ = !lean_is_exclusive(v___x_1614_);
if (v_isSharedCheck_1623_ == 0)
{
v___x_1617_ = v___x_1614_;
v_isShared_1618_ = v_isSharedCheck_1623_;
goto v_resetjp_1616_;
}
else
{
lean_inc(v_a_1615_);
lean_dec(v___x_1614_);
v___x_1617_ = lean_box(0);
v_isShared_1618_ = v_isSharedCheck_1623_;
goto v_resetjp_1616_;
}
v_resetjp_1616_:
{
lean_object* v___x_1619_; lean_object* v___x_1621_; 
v___x_1619_ = lean_nat_add(v_a_1613_, v_a_1615_);
lean_dec(v_a_1615_);
lean_dec(v_a_1613_);
if (v_isShared_1618_ == 0)
{
lean_ctor_set(v___x_1617_, 0, v___x_1619_);
v___x_1621_ = v___x_1617_;
goto v_reusejp_1620_;
}
else
{
lean_object* v_reuseFailAlloc_1622_; 
v_reuseFailAlloc_1622_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1622_, 0, v___x_1619_);
v___x_1621_ = v_reuseFailAlloc_1622_;
goto v_reusejp_1620_;
}
v_reusejp_1620_:
{
return v___x_1621_;
}
}
}
else
{
lean_dec(v_a_1613_);
return v___x_1614_;
}
}
else
{
lean_dec(v_fst_1611_);
return v___x_1612_;
}
}
}
else
{
lean_object* v_a_1624_; lean_object* v___x_1626_; uint8_t v_isShared_1627_; uint8_t v_isSharedCheck_1631_; 
lean_dec_ref(v_hyp_1461_);
v_a_1624_ = lean_ctor_get(v___x_1474_, 0);
v_isSharedCheck_1631_ = !lean_is_exclusive(v___x_1474_);
if (v_isSharedCheck_1631_ == 0)
{
v___x_1626_ = v___x_1474_;
v_isShared_1627_ = v_isSharedCheck_1631_;
goto v_resetjp_1625_;
}
else
{
lean_inc(v_a_1624_);
lean_dec(v___x_1474_);
v___x_1626_ = lean_box(0);
v_isShared_1627_ = v_isSharedCheck_1631_;
goto v_resetjp_1625_;
}
v_resetjp_1625_:
{
lean_object* v___x_1629_; 
if (v_isShared_1627_ == 0)
{
v___x_1629_ = v___x_1626_;
goto v_reusejp_1628_;
}
else
{
lean_object* v_reuseFailAlloc_1630_; 
v_reuseFailAlloc_1630_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1630_, 0, v_a_1624_);
v___x_1629_ = v_reuseFailAlloc_1630_;
goto v_reusejp_1628_;
}
v_reusejp_1628_:
{
return v___x_1629_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound_hypPriority___boxed(lean_object* v_hyp_1632_, lean_object* v_a_1633_, lean_object* v_a_1634_, lean_object* v_a_1635_, lean_object* v_a_1636_, lean_object* v_a_1637_){
_start:
{
lean_object* v_res_1638_; 
v_res_1638_ = lp_mathlib_Mathlib_Tactic_Bound_hypPriority(v_hyp_1632_, v_a_1633_, v_a_1634_, v_a_1635_, v_a_1636_);
lean_dec(v_a_1636_);
lean_dec_ref(v_a_1635_);
lean_dec(v_a_1634_);
lean_dec_ref(v_a_1633_);
return v_res_1638_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_argPriority(lean_object* v_x_1639_, lean_object* v_a_1640_, lean_object* v_a_1641_, lean_object* v_a_1642_, lean_object* v_a_1643_){
_start:
{
lean_object* v___x_1645_; 
lean_inc(v_a_1643_);
lean_inc_ref(v_a_1642_);
lean_inc(v_a_1641_);
lean_inc_ref(v_a_1640_);
v___x_1645_ = lean_infer_type(v_x_1639_, v_a_1640_, v_a_1641_, v_a_1642_, v_a_1643_);
if (lean_obj_tag(v___x_1645_) == 0)
{
lean_object* v_a_1646_; lean_object* v___x_1647_; 
v_a_1646_ = lean_ctor_get(v___x_1645_, 0);
lean_inc(v_a_1646_);
lean_dec_ref_known(v___x_1645_, 1);
v___x_1647_ = lp_mathlib_Mathlib_Tactic_Bound_hypPriority(v_a_1646_, v_a_1640_, v_a_1641_, v_a_1642_, v_a_1643_);
return v___x_1647_;
}
else
{
lean_object* v_a_1648_; lean_object* v___x_1650_; uint8_t v_isShared_1651_; uint8_t v_isSharedCheck_1655_; 
v_a_1648_ = lean_ctor_get(v___x_1645_, 0);
v_isSharedCheck_1655_ = !lean_is_exclusive(v___x_1645_);
if (v_isSharedCheck_1655_ == 0)
{
v___x_1650_ = v___x_1645_;
v_isShared_1651_ = v_isSharedCheck_1655_;
goto v_resetjp_1649_;
}
else
{
lean_inc(v_a_1648_);
lean_dec(v___x_1645_);
v___x_1650_ = lean_box(0);
v_isShared_1651_ = v_isSharedCheck_1655_;
goto v_resetjp_1649_;
}
v_resetjp_1649_:
{
lean_object* v___x_1653_; 
if (v_isShared_1651_ == 0)
{
v___x_1653_ = v___x_1650_;
goto v_reusejp_1652_;
}
else
{
lean_object* v_reuseFailAlloc_1654_; 
v_reuseFailAlloc_1654_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1654_, 0, v_a_1648_);
v___x_1653_ = v_reuseFailAlloc_1654_;
goto v_reusejp_1652_;
}
v_reusejp_1652_:
{
return v___x_1653_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_argPriority___boxed(lean_object* v_x_1656_, lean_object* v_a_1657_, lean_object* v_a_1658_, lean_object* v_a_1659_, lean_object* v_a_1660_, lean_object* v_a_1661_){
_start:
{
lean_object* v_res_1662_; 
v_res_1662_ = lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_argPriority(v_x_1656_, v_a_1657_, v_a_1658_, v_a_1659_, v_a_1660_);
lean_dec(v_a_1660_);
lean_dec_ref(v_a_1659_);
lean_dec(v_a_1658_);
lean_dec_ref(v_a_1657_);
return v_res_1662_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult___lam__0(lean_object* v_t_1663_, uint8_t v___x_1664_, lean_object* v___y_1665_, lean_object* v___y_1666_, lean_object* v___y_1667_, lean_object* v___y_1668_){
_start:
{
lean_object* v___x_1670_; 
v___x_1670_ = l_Lean_Meta_mkFreshLevelMVar(v___y_1665_, v___y_1666_, v___y_1667_, v___y_1668_);
if (lean_obj_tag(v___x_1670_) == 0)
{
lean_object* v_a_1671_; lean_object* v___x_1672_; lean_object* v___x_1673_; lean_object* v___x_1674_; uint8_t v___x_1675_; lean_object* v___x_1676_; lean_object* v___x_1677_; 
v_a_1671_ = lean_ctor_get(v___x_1670_, 0);
lean_inc_n(v_a_1671_, 2);
lean_dec_ref_known(v___x_1670_, 1);
v___x_1672_ = l_Lean_Level_succ___override(v_a_1671_);
v___x_1673_ = l_Lean_Expr_sort___override(v___x_1672_);
v___x_1674_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1674_, 0, v___x_1673_);
v___x_1675_ = 0;
v___x_1676_ = lean_box(0);
v___x_1677_ = l_Lean_Meta_mkFreshExprMVar(v___x_1674_, v___x_1675_, v___x_1676_, v___y_1665_, v___y_1666_, v___y_1667_, v___y_1668_);
if (lean_obj_tag(v___x_1677_) == 0)
{
lean_object* v_a_1678_; lean_object* v___x_1679_; lean_object* v___x_1680_; lean_object* v___x_1681_; lean_object* v___x_1682_; lean_object* v___x_1683_; lean_object* v___x_1684_; lean_object* v___x_1685_; 
v_a_1678_ = lean_ctor_get(v___x_1677_, 0);
lean_inc_n(v_a_1678_, 2);
lean_dec_ref_known(v___x_1677_, 1);
v___x_1679_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__2___closed__1));
v___x_1680_ = lean_box(0);
lean_inc(v_a_1671_);
v___x_1681_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1681_, 0, v_a_1671_);
lean_ctor_set(v___x_1681_, 1, v___x_1680_);
lean_inc_ref(v___x_1681_);
v___x_1682_ = l_Lean_Expr_const___override(v___x_1679_, v___x_1681_);
v___x_1683_ = l_Lean_Expr_app___override(v___x_1682_, v_a_1678_);
v___x_1684_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1684_, 0, v___x_1683_);
v___x_1685_ = l_Lean_Meta_mkFreshExprMVar(v___x_1684_, v___x_1675_, v___x_1676_, v___y_1665_, v___y_1666_, v___y_1667_, v___y_1668_);
if (lean_obj_tag(v___x_1685_) == 0)
{
lean_object* v_a_1686_; lean_object* v___x_1687_; lean_object* v___x_1688_; 
v_a_1686_ = lean_ctor_get(v___x_1685_, 0);
lean_inc(v_a_1686_);
lean_dec_ref_known(v___x_1685_, 1);
lean_inc(v_a_1678_);
v___x_1687_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1687_, 0, v_a_1678_);
lean_inc_ref(v___x_1687_);
v___x_1688_ = l_Lean_Meta_mkFreshExprMVar(v___x_1687_, v___x_1675_, v___x_1676_, v___y_1665_, v___y_1666_, v___y_1667_, v___y_1668_);
if (lean_obj_tag(v___x_1688_) == 0)
{
lean_object* v_a_1689_; lean_object* v___x_1690_; 
v_a_1689_ = lean_ctor_get(v___x_1688_, 0);
lean_inc(v_a_1689_);
lean_dec_ref_known(v___x_1688_, 1);
v___x_1690_ = l_Lean_Meta_mkFreshExprMVar(v___x_1687_, v___x_1675_, v___x_1676_, v___y_1665_, v___y_1666_, v___y_1667_, v___y_1668_);
if (lean_obj_tag(v___x_1690_) == 0)
{
lean_object* v_a_1691_; lean_object* v_keyedConfig_1692_; uint8_t v_trackZetaDelta_1693_; lean_object* v_zetaDeltaSet_1694_; lean_object* v_lctx_1695_; lean_object* v_localInstances_1696_; lean_object* v_defEqCtx_x3f_1697_; lean_object* v_synthPendingDepth_1698_; lean_object* v_customCanUnfoldPredicate_x3f_1699_; uint8_t v_univApprox_1700_; uint8_t v_inTypeClassResolution_1701_; uint8_t v_cacheInferType_1702_; lean_object* v___x_1704_; uint8_t v_isShared_1705_; uint8_t v_isSharedCheck_1763_; 
v_a_1691_ = lean_ctor_get(v___x_1690_, 0);
lean_inc(v_a_1691_);
lean_dec_ref_known(v___x_1690_, 1);
v_keyedConfig_1692_ = lean_ctor_get(v___y_1665_, 0);
v_trackZetaDelta_1693_ = lean_ctor_get_uint8(v___y_1665_, sizeof(void*)*7);
v_zetaDeltaSet_1694_ = lean_ctor_get(v___y_1665_, 1);
v_lctx_1695_ = lean_ctor_get(v___y_1665_, 2);
v_localInstances_1696_ = lean_ctor_get(v___y_1665_, 3);
v_defEqCtx_x3f_1697_ = lean_ctor_get(v___y_1665_, 4);
v_synthPendingDepth_1698_ = lean_ctor_get(v___y_1665_, 5);
v_customCanUnfoldPredicate_x3f_1699_ = lean_ctor_get(v___y_1665_, 6);
v_univApprox_1700_ = lean_ctor_get_uint8(v___y_1665_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_1701_ = lean_ctor_get_uint8(v___y_1665_, sizeof(void*)*7 + 2);
v_cacheInferType_1702_ = lean_ctor_get_uint8(v___y_1665_, sizeof(void*)*7 + 3);
v_isSharedCheck_1763_ = !lean_is_exclusive(v___y_1665_);
if (v_isSharedCheck_1763_ == 0)
{
v___x_1704_ = v___y_1665_;
v_isShared_1705_ = v_isSharedCheck_1763_;
goto v_resetjp_1703_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_1699_);
lean_inc(v_synthPendingDepth_1698_);
lean_inc(v_defEqCtx_x3f_1697_);
lean_inc(v_localInstances_1696_);
lean_inc(v_lctx_1695_);
lean_inc(v_zetaDeltaSet_1694_);
lean_inc(v_keyedConfig_1692_);
lean_dec(v___y_1665_);
v___x_1704_ = lean_box(0);
v_isShared_1705_ = v_isSharedCheck_1763_;
goto v_resetjp_1703_;
}
v_resetjp_1703_:
{
lean_object* v___x_1706_; lean_object* v___x_1707_; lean_object* v___x_1708_; lean_object* v___x_1709_; lean_object* v___x_1710_; lean_object* v___x_1711_; uint8_t v___x_1712_; lean_object* v___x_1713_; lean_object* v___x_1715_; 
v___x_1706_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__2___closed__3));
v___x_1707_ = l_Lean_Expr_const___override(v___x_1706_, v___x_1681_);
lean_inc(v_a_1678_);
v___x_1708_ = l_Lean_Expr_app___override(v___x_1707_, v_a_1678_);
lean_inc(v_a_1686_);
v___x_1709_ = l_Lean_Expr_app___override(v___x_1708_, v_a_1686_);
lean_inc(v_a_1689_);
v___x_1710_ = l_Lean_Expr_app___override(v___x_1709_, v_a_1689_);
lean_inc(v_a_1691_);
v___x_1711_ = l_Lean_Expr_app___override(v___x_1710_, v_a_1691_);
v___x_1712_ = 2;
v___x_1713_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_1712_, v_keyedConfig_1692_);
if (v_isShared_1705_ == 0)
{
lean_ctor_set(v___x_1704_, 0, v___x_1713_);
v___x_1715_ = v___x_1704_;
goto v_reusejp_1714_;
}
else
{
lean_object* v_reuseFailAlloc_1762_; 
v_reuseFailAlloc_1762_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_1762_, 0, v___x_1713_);
lean_ctor_set(v_reuseFailAlloc_1762_, 1, v_zetaDeltaSet_1694_);
lean_ctor_set(v_reuseFailAlloc_1762_, 2, v_lctx_1695_);
lean_ctor_set(v_reuseFailAlloc_1762_, 3, v_localInstances_1696_);
lean_ctor_set(v_reuseFailAlloc_1762_, 4, v_defEqCtx_x3f_1697_);
lean_ctor_set(v_reuseFailAlloc_1762_, 5, v_synthPendingDepth_1698_);
lean_ctor_set(v_reuseFailAlloc_1762_, 6, v_customCanUnfoldPredicate_x3f_1699_);
lean_ctor_set_uint8(v_reuseFailAlloc_1762_, sizeof(void*)*7, v_trackZetaDelta_1693_);
lean_ctor_set_uint8(v_reuseFailAlloc_1762_, sizeof(void*)*7 + 1, v_univApprox_1700_);
lean_ctor_set_uint8(v_reuseFailAlloc_1762_, sizeof(void*)*7 + 2, v_inTypeClassResolution_1701_);
lean_ctor_set_uint8(v_reuseFailAlloc_1762_, sizeof(void*)*7 + 3, v_cacheInferType_1702_);
v___x_1715_ = v_reuseFailAlloc_1762_;
goto v_reusejp_1714_;
}
v_reusejp_1714_:
{
lean_object* v___x_1716_; 
v___x_1716_ = l_Lean_Meta_isExprDefEq(v___x_1711_, v_t_1663_, v___x_1715_, v___y_1666_, v___y_1667_, v___y_1668_);
lean_dec_ref(v___x_1715_);
if (lean_obj_tag(v___x_1716_) == 0)
{
lean_object* v_a_1717_; lean_object* v___x_1719_; uint8_t v_isShared_1720_; uint8_t v_isSharedCheck_1753_; 
v_a_1717_ = lean_ctor_get(v___x_1716_, 0);
v_isSharedCheck_1753_ = !lean_is_exclusive(v___x_1716_);
if (v_isSharedCheck_1753_ == 0)
{
v___x_1719_ = v___x_1716_;
v_isShared_1720_ = v_isSharedCheck_1753_;
goto v_resetjp_1718_;
}
else
{
lean_inc(v_a_1717_);
lean_dec(v___x_1716_);
v___x_1719_ = lean_box(0);
v_isShared_1720_ = v_isSharedCheck_1753_;
goto v_resetjp_1718_;
}
v_resetjp_1718_:
{
uint8_t v___x_1721_; 
v___x_1721_ = lean_unbox(v_a_1717_);
if (v___x_1721_ == 0)
{
lean_object* v___x_1722_; lean_object* v___x_1723_; lean_object* v___x_1724_; lean_object* v___x_1725_; lean_object* v___x_1726_; lean_object* v___x_1727_; lean_object* v___x_1729_; 
lean_dec(v_a_1717_);
v___x_1722_ = lean_box(v___x_1664_);
v___x_1723_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1723_, 0, v_a_1691_);
lean_ctor_set(v___x_1723_, 1, v___x_1722_);
v___x_1724_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1724_, 0, v_a_1689_);
lean_ctor_set(v___x_1724_, 1, v___x_1723_);
v___x_1725_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1725_, 0, v_a_1686_);
lean_ctor_set(v___x_1725_, 1, v___x_1724_);
v___x_1726_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1726_, 0, v_a_1678_);
lean_ctor_set(v___x_1726_, 1, v___x_1725_);
v___x_1727_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1727_, 0, v_a_1671_);
lean_ctor_set(v___x_1727_, 1, v___x_1726_);
if (v_isShared_1720_ == 0)
{
lean_ctor_set(v___x_1719_, 0, v___x_1727_);
v___x_1729_ = v___x_1719_;
goto v_reusejp_1728_;
}
else
{
lean_object* v_reuseFailAlloc_1730_; 
v_reuseFailAlloc_1730_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1730_, 0, v___x_1727_);
v___x_1729_ = v_reuseFailAlloc_1730_;
goto v_reusejp_1728_;
}
v_reusejp_1728_:
{
return v___x_1729_;
}
}
else
{
lean_object* v___x_1731_; lean_object* v_a_1732_; lean_object* v___x_1733_; lean_object* v_a_1734_; lean_object* v___x_1735_; lean_object* v_a_1736_; lean_object* v___x_1737_; lean_object* v_a_1738_; lean_object* v___x_1739_; lean_object* v_a_1740_; lean_object* v___x_1742_; uint8_t v_isShared_1743_; uint8_t v_isSharedCheck_1752_; 
lean_del_object(v___x_1719_);
v___x_1731_ = lp_mathlib_Lean_instantiateLevelMVars___at___00Mathlib_Tactic_Bound_hypPriority_spec__0___redArg(v_a_1671_, v___y_1666_);
v_a_1732_ = lean_ctor_get(v___x_1731_, 0);
lean_inc(v_a_1732_);
lean_dec_ref(v___x_1731_);
v___x_1733_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Bound_isZero_spec__0___redArg(v_a_1678_, v___y_1666_);
v_a_1734_ = lean_ctor_get(v___x_1733_, 0);
lean_inc(v_a_1734_);
lean_dec_ref(v___x_1733_);
v___x_1735_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Bound_isZero_spec__0___redArg(v_a_1686_, v___y_1666_);
v_a_1736_ = lean_ctor_get(v___x_1735_, 0);
lean_inc(v_a_1736_);
lean_dec_ref(v___x_1735_);
v___x_1737_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Bound_isZero_spec__0___redArg(v_a_1689_, v___y_1666_);
v_a_1738_ = lean_ctor_get(v___x_1737_, 0);
lean_inc(v_a_1738_);
lean_dec_ref(v___x_1737_);
v___x_1739_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Bound_isZero_spec__0___redArg(v_a_1691_, v___y_1666_);
v_a_1740_ = lean_ctor_get(v___x_1739_, 0);
v_isSharedCheck_1752_ = !lean_is_exclusive(v___x_1739_);
if (v_isSharedCheck_1752_ == 0)
{
v___x_1742_ = v___x_1739_;
v_isShared_1743_ = v_isSharedCheck_1752_;
goto v_resetjp_1741_;
}
else
{
lean_inc(v_a_1740_);
lean_dec(v___x_1739_);
v___x_1742_ = lean_box(0);
v_isShared_1743_ = v_isSharedCheck_1752_;
goto v_resetjp_1741_;
}
v_resetjp_1741_:
{
lean_object* v___x_1744_; lean_object* v___x_1745_; lean_object* v___x_1746_; lean_object* v___x_1747_; lean_object* v___x_1748_; lean_object* v___x_1750_; 
v___x_1744_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1744_, 0, v_a_1740_);
lean_ctor_set(v___x_1744_, 1, v_a_1717_);
v___x_1745_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1745_, 0, v_a_1738_);
lean_ctor_set(v___x_1745_, 1, v___x_1744_);
v___x_1746_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1746_, 0, v_a_1736_);
lean_ctor_set(v___x_1746_, 1, v___x_1745_);
v___x_1747_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1747_, 0, v_a_1734_);
lean_ctor_set(v___x_1747_, 1, v___x_1746_);
v___x_1748_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1748_, 0, v_a_1732_);
lean_ctor_set(v___x_1748_, 1, v___x_1747_);
if (v_isShared_1743_ == 0)
{
lean_ctor_set(v___x_1742_, 0, v___x_1748_);
v___x_1750_ = v___x_1742_;
goto v_reusejp_1749_;
}
else
{
lean_object* v_reuseFailAlloc_1751_; 
v_reuseFailAlloc_1751_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1751_, 0, v___x_1748_);
v___x_1750_ = v_reuseFailAlloc_1751_;
goto v_reusejp_1749_;
}
v_reusejp_1749_:
{
return v___x_1750_;
}
}
}
}
}
else
{
lean_object* v_a_1754_; lean_object* v___x_1756_; uint8_t v_isShared_1757_; uint8_t v_isSharedCheck_1761_; 
lean_dec(v_a_1691_);
lean_dec(v_a_1689_);
lean_dec(v_a_1686_);
lean_dec(v_a_1678_);
lean_dec(v_a_1671_);
v_a_1754_ = lean_ctor_get(v___x_1716_, 0);
v_isSharedCheck_1761_ = !lean_is_exclusive(v___x_1716_);
if (v_isSharedCheck_1761_ == 0)
{
v___x_1756_ = v___x_1716_;
v_isShared_1757_ = v_isSharedCheck_1761_;
goto v_resetjp_1755_;
}
else
{
lean_inc(v_a_1754_);
lean_dec(v___x_1716_);
v___x_1756_ = lean_box(0);
v_isShared_1757_ = v_isSharedCheck_1761_;
goto v_resetjp_1755_;
}
v_resetjp_1755_:
{
lean_object* v___x_1759_; 
if (v_isShared_1757_ == 0)
{
v___x_1759_ = v___x_1756_;
goto v_reusejp_1758_;
}
else
{
lean_object* v_reuseFailAlloc_1760_; 
v_reuseFailAlloc_1760_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1760_, 0, v_a_1754_);
v___x_1759_ = v_reuseFailAlloc_1760_;
goto v_reusejp_1758_;
}
v_reusejp_1758_:
{
return v___x_1759_;
}
}
}
}
}
}
else
{
lean_object* v_a_1764_; lean_object* v___x_1766_; uint8_t v_isShared_1767_; uint8_t v_isSharedCheck_1771_; 
lean_dec(v_a_1689_);
lean_dec(v_a_1686_);
lean_dec_ref_known(v___x_1681_, 2);
lean_dec(v_a_1678_);
lean_dec(v_a_1671_);
lean_dec_ref(v___y_1665_);
lean_dec_ref(v_t_1663_);
v_a_1764_ = lean_ctor_get(v___x_1690_, 0);
v_isSharedCheck_1771_ = !lean_is_exclusive(v___x_1690_);
if (v_isSharedCheck_1771_ == 0)
{
v___x_1766_ = v___x_1690_;
v_isShared_1767_ = v_isSharedCheck_1771_;
goto v_resetjp_1765_;
}
else
{
lean_inc(v_a_1764_);
lean_dec(v___x_1690_);
v___x_1766_ = lean_box(0);
v_isShared_1767_ = v_isSharedCheck_1771_;
goto v_resetjp_1765_;
}
v_resetjp_1765_:
{
lean_object* v___x_1769_; 
if (v_isShared_1767_ == 0)
{
v___x_1769_ = v___x_1766_;
goto v_reusejp_1768_;
}
else
{
lean_object* v_reuseFailAlloc_1770_; 
v_reuseFailAlloc_1770_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1770_, 0, v_a_1764_);
v___x_1769_ = v_reuseFailAlloc_1770_;
goto v_reusejp_1768_;
}
v_reusejp_1768_:
{
return v___x_1769_;
}
}
}
}
else
{
lean_object* v_a_1772_; lean_object* v___x_1774_; uint8_t v_isShared_1775_; uint8_t v_isSharedCheck_1779_; 
lean_dec_ref_known(v___x_1687_, 1);
lean_dec(v_a_1686_);
lean_dec_ref_known(v___x_1681_, 2);
lean_dec(v_a_1678_);
lean_dec(v_a_1671_);
lean_dec_ref(v___y_1665_);
lean_dec_ref(v_t_1663_);
v_a_1772_ = lean_ctor_get(v___x_1688_, 0);
v_isSharedCheck_1779_ = !lean_is_exclusive(v___x_1688_);
if (v_isSharedCheck_1779_ == 0)
{
v___x_1774_ = v___x_1688_;
v_isShared_1775_ = v_isSharedCheck_1779_;
goto v_resetjp_1773_;
}
else
{
lean_inc(v_a_1772_);
lean_dec(v___x_1688_);
v___x_1774_ = lean_box(0);
v_isShared_1775_ = v_isSharedCheck_1779_;
goto v_resetjp_1773_;
}
v_resetjp_1773_:
{
lean_object* v___x_1777_; 
if (v_isShared_1775_ == 0)
{
v___x_1777_ = v___x_1774_;
goto v_reusejp_1776_;
}
else
{
lean_object* v_reuseFailAlloc_1778_; 
v_reuseFailAlloc_1778_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1778_, 0, v_a_1772_);
v___x_1777_ = v_reuseFailAlloc_1778_;
goto v_reusejp_1776_;
}
v_reusejp_1776_:
{
return v___x_1777_;
}
}
}
}
else
{
lean_object* v_a_1780_; lean_object* v___x_1782_; uint8_t v_isShared_1783_; uint8_t v_isSharedCheck_1787_; 
lean_dec_ref_known(v___x_1681_, 2);
lean_dec(v_a_1678_);
lean_dec(v_a_1671_);
lean_dec_ref(v___y_1665_);
lean_dec_ref(v_t_1663_);
v_a_1780_ = lean_ctor_get(v___x_1685_, 0);
v_isSharedCheck_1787_ = !lean_is_exclusive(v___x_1685_);
if (v_isSharedCheck_1787_ == 0)
{
v___x_1782_ = v___x_1685_;
v_isShared_1783_ = v_isSharedCheck_1787_;
goto v_resetjp_1781_;
}
else
{
lean_inc(v_a_1780_);
lean_dec(v___x_1685_);
v___x_1782_ = lean_box(0);
v_isShared_1783_ = v_isSharedCheck_1787_;
goto v_resetjp_1781_;
}
v_resetjp_1781_:
{
lean_object* v___x_1785_; 
if (v_isShared_1783_ == 0)
{
v___x_1785_ = v___x_1782_;
goto v_reusejp_1784_;
}
else
{
lean_object* v_reuseFailAlloc_1786_; 
v_reuseFailAlloc_1786_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1786_, 0, v_a_1780_);
v___x_1785_ = v_reuseFailAlloc_1786_;
goto v_reusejp_1784_;
}
v_reusejp_1784_:
{
return v___x_1785_;
}
}
}
}
else
{
lean_object* v_a_1788_; lean_object* v___x_1790_; uint8_t v_isShared_1791_; uint8_t v_isSharedCheck_1795_; 
lean_dec(v_a_1671_);
lean_dec_ref(v___y_1665_);
lean_dec_ref(v_t_1663_);
v_a_1788_ = lean_ctor_get(v___x_1677_, 0);
v_isSharedCheck_1795_ = !lean_is_exclusive(v___x_1677_);
if (v_isSharedCheck_1795_ == 0)
{
v___x_1790_ = v___x_1677_;
v_isShared_1791_ = v_isSharedCheck_1795_;
goto v_resetjp_1789_;
}
else
{
lean_inc(v_a_1788_);
lean_dec(v___x_1677_);
v___x_1790_ = lean_box(0);
v_isShared_1791_ = v_isSharedCheck_1795_;
goto v_resetjp_1789_;
}
v_resetjp_1789_:
{
lean_object* v___x_1793_; 
if (v_isShared_1791_ == 0)
{
v___x_1793_ = v___x_1790_;
goto v_reusejp_1792_;
}
else
{
lean_object* v_reuseFailAlloc_1794_; 
v_reuseFailAlloc_1794_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1794_, 0, v_a_1788_);
v___x_1793_ = v_reuseFailAlloc_1794_;
goto v_reusejp_1792_;
}
v_reusejp_1792_:
{
return v___x_1793_;
}
}
}
}
else
{
lean_object* v_a_1796_; lean_object* v___x_1798_; uint8_t v_isShared_1799_; uint8_t v_isSharedCheck_1803_; 
lean_dec_ref(v___y_1665_);
lean_dec_ref(v_t_1663_);
v_a_1796_ = lean_ctor_get(v___x_1670_, 0);
v_isSharedCheck_1803_ = !lean_is_exclusive(v___x_1670_);
if (v_isSharedCheck_1803_ == 0)
{
v___x_1798_ = v___x_1670_;
v_isShared_1799_ = v_isSharedCheck_1803_;
goto v_resetjp_1797_;
}
else
{
lean_inc(v_a_1796_);
lean_dec(v___x_1670_);
v___x_1798_ = lean_box(0);
v_isShared_1799_ = v_isSharedCheck_1803_;
goto v_resetjp_1797_;
}
v_resetjp_1797_:
{
lean_object* v___x_1801_; 
if (v_isShared_1799_ == 0)
{
v___x_1801_ = v___x_1798_;
goto v_reusejp_1800_;
}
else
{
lean_object* v_reuseFailAlloc_1802_; 
v_reuseFailAlloc_1802_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1802_, 0, v_a_1796_);
v___x_1801_ = v_reuseFailAlloc_1802_;
goto v_reusejp_1800_;
}
v_reusejp_1800_:
{
return v___x_1801_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult___lam__0___boxed(lean_object* v_t_1804_, lean_object* v___x_1805_, lean_object* v___y_1806_, lean_object* v___y_1807_, lean_object* v___y_1808_, lean_object* v___y_1809_, lean_object* v___y_1810_){
_start:
{
uint8_t v___x_11694__boxed_1811_; lean_object* v_res_1812_; 
v___x_11694__boxed_1811_ = lean_unbox(v___x_1805_);
v_res_1812_ = lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult___lam__0(v_t_1804_, v___x_11694__boxed_1811_, v___y_1806_, v___y_1807_, v___y_1808_, v___y_1809_);
lean_dec(v___y_1809_);
lean_dec_ref(v___y_1808_);
lean_dec(v___y_1807_);
return v_res_1812_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult___lam__1(lean_object* v_t_1813_, uint8_t v___x_1814_, lean_object* v___y_1815_, lean_object* v___y_1816_, lean_object* v___y_1817_, lean_object* v___y_1818_){
_start:
{
lean_object* v___x_1820_; 
v___x_1820_ = l_Lean_Meta_mkFreshLevelMVar(v___y_1815_, v___y_1816_, v___y_1817_, v___y_1818_);
if (lean_obj_tag(v___x_1820_) == 0)
{
lean_object* v_a_1821_; lean_object* v___x_1822_; lean_object* v___x_1823_; lean_object* v___x_1824_; uint8_t v___x_1825_; lean_object* v___x_1826_; lean_object* v___x_1827_; 
v_a_1821_ = lean_ctor_get(v___x_1820_, 0);
lean_inc_n(v_a_1821_, 2);
lean_dec_ref_known(v___x_1820_, 1);
v___x_1822_ = l_Lean_Level_succ___override(v_a_1821_);
v___x_1823_ = l_Lean_Expr_sort___override(v___x_1822_);
v___x_1824_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1824_, 0, v___x_1823_);
v___x_1825_ = 0;
v___x_1826_ = lean_box(0);
v___x_1827_ = l_Lean_Meta_mkFreshExprMVar(v___x_1824_, v___x_1825_, v___x_1826_, v___y_1815_, v___y_1816_, v___y_1817_, v___y_1818_);
if (lean_obj_tag(v___x_1827_) == 0)
{
lean_object* v_a_1828_; lean_object* v___x_1829_; lean_object* v___x_1830_; lean_object* v___x_1831_; lean_object* v___x_1832_; lean_object* v___x_1833_; lean_object* v___x_1834_; lean_object* v___x_1835_; 
v_a_1828_ = lean_ctor_get(v___x_1827_, 0);
lean_inc_n(v_a_1828_, 2);
lean_dec_ref_known(v___x_1827_, 1);
v___x_1829_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__3___closed__1));
v___x_1830_ = lean_box(0);
lean_inc(v_a_1821_);
v___x_1831_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1831_, 0, v_a_1821_);
lean_ctor_set(v___x_1831_, 1, v___x_1830_);
lean_inc_ref(v___x_1831_);
v___x_1832_ = l_Lean_Expr_const___override(v___x_1829_, v___x_1831_);
v___x_1833_ = l_Lean_Expr_app___override(v___x_1832_, v_a_1828_);
v___x_1834_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1834_, 0, v___x_1833_);
v___x_1835_ = l_Lean_Meta_mkFreshExprMVar(v___x_1834_, v___x_1825_, v___x_1826_, v___y_1815_, v___y_1816_, v___y_1817_, v___y_1818_);
if (lean_obj_tag(v___x_1835_) == 0)
{
lean_object* v_a_1836_; lean_object* v___x_1837_; lean_object* v___x_1838_; 
v_a_1836_ = lean_ctor_get(v___x_1835_, 0);
lean_inc(v_a_1836_);
lean_dec_ref_known(v___x_1835_, 1);
lean_inc(v_a_1828_);
v___x_1837_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1837_, 0, v_a_1828_);
lean_inc_ref(v___x_1837_);
v___x_1838_ = l_Lean_Meta_mkFreshExprMVar(v___x_1837_, v___x_1825_, v___x_1826_, v___y_1815_, v___y_1816_, v___y_1817_, v___y_1818_);
if (lean_obj_tag(v___x_1838_) == 0)
{
lean_object* v_a_1839_; lean_object* v___x_1840_; 
v_a_1839_ = lean_ctor_get(v___x_1838_, 0);
lean_inc(v_a_1839_);
lean_dec_ref_known(v___x_1838_, 1);
v___x_1840_ = l_Lean_Meta_mkFreshExprMVar(v___x_1837_, v___x_1825_, v___x_1826_, v___y_1815_, v___y_1816_, v___y_1817_, v___y_1818_);
if (lean_obj_tag(v___x_1840_) == 0)
{
lean_object* v_a_1841_; lean_object* v_keyedConfig_1842_; uint8_t v_trackZetaDelta_1843_; lean_object* v_zetaDeltaSet_1844_; lean_object* v_lctx_1845_; lean_object* v_localInstances_1846_; lean_object* v_defEqCtx_x3f_1847_; lean_object* v_synthPendingDepth_1848_; lean_object* v_customCanUnfoldPredicate_x3f_1849_; uint8_t v_univApprox_1850_; uint8_t v_inTypeClassResolution_1851_; uint8_t v_cacheInferType_1852_; lean_object* v___x_1854_; uint8_t v_isShared_1855_; uint8_t v_isSharedCheck_1913_; 
v_a_1841_ = lean_ctor_get(v___x_1840_, 0);
lean_inc(v_a_1841_);
lean_dec_ref_known(v___x_1840_, 1);
v_keyedConfig_1842_ = lean_ctor_get(v___y_1815_, 0);
v_trackZetaDelta_1843_ = lean_ctor_get_uint8(v___y_1815_, sizeof(void*)*7);
v_zetaDeltaSet_1844_ = lean_ctor_get(v___y_1815_, 1);
v_lctx_1845_ = lean_ctor_get(v___y_1815_, 2);
v_localInstances_1846_ = lean_ctor_get(v___y_1815_, 3);
v_defEqCtx_x3f_1847_ = lean_ctor_get(v___y_1815_, 4);
v_synthPendingDepth_1848_ = lean_ctor_get(v___y_1815_, 5);
v_customCanUnfoldPredicate_x3f_1849_ = lean_ctor_get(v___y_1815_, 6);
v_univApprox_1850_ = lean_ctor_get_uint8(v___y_1815_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_1851_ = lean_ctor_get_uint8(v___y_1815_, sizeof(void*)*7 + 2);
v_cacheInferType_1852_ = lean_ctor_get_uint8(v___y_1815_, sizeof(void*)*7 + 3);
v_isSharedCheck_1913_ = !lean_is_exclusive(v___y_1815_);
if (v_isSharedCheck_1913_ == 0)
{
v___x_1854_ = v___y_1815_;
v_isShared_1855_ = v_isSharedCheck_1913_;
goto v_resetjp_1853_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_1849_);
lean_inc(v_synthPendingDepth_1848_);
lean_inc(v_defEqCtx_x3f_1847_);
lean_inc(v_localInstances_1846_);
lean_inc(v_lctx_1845_);
lean_inc(v_zetaDeltaSet_1844_);
lean_inc(v_keyedConfig_1842_);
lean_dec(v___y_1815_);
v___x_1854_ = lean_box(0);
v_isShared_1855_ = v_isSharedCheck_1913_;
goto v_resetjp_1853_;
}
v_resetjp_1853_:
{
lean_object* v___x_1856_; lean_object* v___x_1857_; lean_object* v___x_1858_; lean_object* v___x_1859_; lean_object* v___x_1860_; lean_object* v___x_1861_; uint8_t v___x_1862_; lean_object* v___x_1863_; lean_object* v___x_1865_; 
v___x_1856_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__3___closed__3));
v___x_1857_ = l_Lean_Expr_const___override(v___x_1856_, v___x_1831_);
lean_inc(v_a_1828_);
v___x_1858_ = l_Lean_Expr_app___override(v___x_1857_, v_a_1828_);
lean_inc(v_a_1836_);
v___x_1859_ = l_Lean_Expr_app___override(v___x_1858_, v_a_1836_);
lean_inc(v_a_1839_);
v___x_1860_ = l_Lean_Expr_app___override(v___x_1859_, v_a_1839_);
lean_inc(v_a_1841_);
v___x_1861_ = l_Lean_Expr_app___override(v___x_1860_, v_a_1841_);
v___x_1862_ = 2;
v___x_1863_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_1862_, v_keyedConfig_1842_);
if (v_isShared_1855_ == 0)
{
lean_ctor_set(v___x_1854_, 0, v___x_1863_);
v___x_1865_ = v___x_1854_;
goto v_reusejp_1864_;
}
else
{
lean_object* v_reuseFailAlloc_1912_; 
v_reuseFailAlloc_1912_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_1912_, 0, v___x_1863_);
lean_ctor_set(v_reuseFailAlloc_1912_, 1, v_zetaDeltaSet_1844_);
lean_ctor_set(v_reuseFailAlloc_1912_, 2, v_lctx_1845_);
lean_ctor_set(v_reuseFailAlloc_1912_, 3, v_localInstances_1846_);
lean_ctor_set(v_reuseFailAlloc_1912_, 4, v_defEqCtx_x3f_1847_);
lean_ctor_set(v_reuseFailAlloc_1912_, 5, v_synthPendingDepth_1848_);
lean_ctor_set(v_reuseFailAlloc_1912_, 6, v_customCanUnfoldPredicate_x3f_1849_);
lean_ctor_set_uint8(v_reuseFailAlloc_1912_, sizeof(void*)*7, v_trackZetaDelta_1843_);
lean_ctor_set_uint8(v_reuseFailAlloc_1912_, sizeof(void*)*7 + 1, v_univApprox_1850_);
lean_ctor_set_uint8(v_reuseFailAlloc_1912_, sizeof(void*)*7 + 2, v_inTypeClassResolution_1851_);
lean_ctor_set_uint8(v_reuseFailAlloc_1912_, sizeof(void*)*7 + 3, v_cacheInferType_1852_);
v___x_1865_ = v_reuseFailAlloc_1912_;
goto v_reusejp_1864_;
}
v_reusejp_1864_:
{
lean_object* v___x_1866_; 
v___x_1866_ = l_Lean_Meta_isExprDefEq(v___x_1861_, v_t_1813_, v___x_1865_, v___y_1816_, v___y_1817_, v___y_1818_);
lean_dec_ref(v___x_1865_);
if (lean_obj_tag(v___x_1866_) == 0)
{
lean_object* v_a_1867_; lean_object* v___x_1869_; uint8_t v_isShared_1870_; uint8_t v_isSharedCheck_1903_; 
v_a_1867_ = lean_ctor_get(v___x_1866_, 0);
v_isSharedCheck_1903_ = !lean_is_exclusive(v___x_1866_);
if (v_isSharedCheck_1903_ == 0)
{
v___x_1869_ = v___x_1866_;
v_isShared_1870_ = v_isSharedCheck_1903_;
goto v_resetjp_1868_;
}
else
{
lean_inc(v_a_1867_);
lean_dec(v___x_1866_);
v___x_1869_ = lean_box(0);
v_isShared_1870_ = v_isSharedCheck_1903_;
goto v_resetjp_1868_;
}
v_resetjp_1868_:
{
uint8_t v___x_1871_; 
v___x_1871_ = lean_unbox(v_a_1867_);
if (v___x_1871_ == 0)
{
lean_object* v___x_1872_; lean_object* v___x_1873_; lean_object* v___x_1874_; lean_object* v___x_1875_; lean_object* v___x_1876_; lean_object* v___x_1877_; lean_object* v___x_1879_; 
lean_dec(v_a_1867_);
v___x_1872_ = lean_box(v___x_1814_);
v___x_1873_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1873_, 0, v_a_1841_);
lean_ctor_set(v___x_1873_, 1, v___x_1872_);
v___x_1874_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1874_, 0, v_a_1839_);
lean_ctor_set(v___x_1874_, 1, v___x_1873_);
v___x_1875_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1875_, 0, v_a_1836_);
lean_ctor_set(v___x_1875_, 1, v___x_1874_);
v___x_1876_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1876_, 0, v_a_1828_);
lean_ctor_set(v___x_1876_, 1, v___x_1875_);
v___x_1877_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1877_, 0, v_a_1821_);
lean_ctor_set(v___x_1877_, 1, v___x_1876_);
if (v_isShared_1870_ == 0)
{
lean_ctor_set(v___x_1869_, 0, v___x_1877_);
v___x_1879_ = v___x_1869_;
goto v_reusejp_1878_;
}
else
{
lean_object* v_reuseFailAlloc_1880_; 
v_reuseFailAlloc_1880_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1880_, 0, v___x_1877_);
v___x_1879_ = v_reuseFailAlloc_1880_;
goto v_reusejp_1878_;
}
v_reusejp_1878_:
{
return v___x_1879_;
}
}
else
{
lean_object* v___x_1881_; lean_object* v_a_1882_; lean_object* v___x_1883_; lean_object* v_a_1884_; lean_object* v___x_1885_; lean_object* v_a_1886_; lean_object* v___x_1887_; lean_object* v_a_1888_; lean_object* v___x_1889_; lean_object* v_a_1890_; lean_object* v___x_1892_; uint8_t v_isShared_1893_; uint8_t v_isSharedCheck_1902_; 
lean_del_object(v___x_1869_);
v___x_1881_ = lp_mathlib_Lean_instantiateLevelMVars___at___00Mathlib_Tactic_Bound_hypPriority_spec__0___redArg(v_a_1821_, v___y_1816_);
v_a_1882_ = lean_ctor_get(v___x_1881_, 0);
lean_inc(v_a_1882_);
lean_dec_ref(v___x_1881_);
v___x_1883_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Bound_isZero_spec__0___redArg(v_a_1828_, v___y_1816_);
v_a_1884_ = lean_ctor_get(v___x_1883_, 0);
lean_inc(v_a_1884_);
lean_dec_ref(v___x_1883_);
v___x_1885_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Bound_isZero_spec__0___redArg(v_a_1836_, v___y_1816_);
v_a_1886_ = lean_ctor_get(v___x_1885_, 0);
lean_inc(v_a_1886_);
lean_dec_ref(v___x_1885_);
v___x_1887_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Bound_isZero_spec__0___redArg(v_a_1839_, v___y_1816_);
v_a_1888_ = lean_ctor_get(v___x_1887_, 0);
lean_inc(v_a_1888_);
lean_dec_ref(v___x_1887_);
v___x_1889_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Bound_isZero_spec__0___redArg(v_a_1841_, v___y_1816_);
v_a_1890_ = lean_ctor_get(v___x_1889_, 0);
v_isSharedCheck_1902_ = !lean_is_exclusive(v___x_1889_);
if (v_isSharedCheck_1902_ == 0)
{
v___x_1892_ = v___x_1889_;
v_isShared_1893_ = v_isSharedCheck_1902_;
goto v_resetjp_1891_;
}
else
{
lean_inc(v_a_1890_);
lean_dec(v___x_1889_);
v___x_1892_ = lean_box(0);
v_isShared_1893_ = v_isSharedCheck_1902_;
goto v_resetjp_1891_;
}
v_resetjp_1891_:
{
lean_object* v___x_1894_; lean_object* v___x_1895_; lean_object* v___x_1896_; lean_object* v___x_1897_; lean_object* v___x_1898_; lean_object* v___x_1900_; 
v___x_1894_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1894_, 0, v_a_1890_);
lean_ctor_set(v___x_1894_, 1, v_a_1867_);
v___x_1895_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1895_, 0, v_a_1888_);
lean_ctor_set(v___x_1895_, 1, v___x_1894_);
v___x_1896_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1896_, 0, v_a_1886_);
lean_ctor_set(v___x_1896_, 1, v___x_1895_);
v___x_1897_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1897_, 0, v_a_1884_);
lean_ctor_set(v___x_1897_, 1, v___x_1896_);
v___x_1898_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1898_, 0, v_a_1882_);
lean_ctor_set(v___x_1898_, 1, v___x_1897_);
if (v_isShared_1893_ == 0)
{
lean_ctor_set(v___x_1892_, 0, v___x_1898_);
v___x_1900_ = v___x_1892_;
goto v_reusejp_1899_;
}
else
{
lean_object* v_reuseFailAlloc_1901_; 
v_reuseFailAlloc_1901_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1901_, 0, v___x_1898_);
v___x_1900_ = v_reuseFailAlloc_1901_;
goto v_reusejp_1899_;
}
v_reusejp_1899_:
{
return v___x_1900_;
}
}
}
}
}
else
{
lean_object* v_a_1904_; lean_object* v___x_1906_; uint8_t v_isShared_1907_; uint8_t v_isSharedCheck_1911_; 
lean_dec(v_a_1841_);
lean_dec(v_a_1839_);
lean_dec(v_a_1836_);
lean_dec(v_a_1828_);
lean_dec(v_a_1821_);
v_a_1904_ = lean_ctor_get(v___x_1866_, 0);
v_isSharedCheck_1911_ = !lean_is_exclusive(v___x_1866_);
if (v_isSharedCheck_1911_ == 0)
{
v___x_1906_ = v___x_1866_;
v_isShared_1907_ = v_isSharedCheck_1911_;
goto v_resetjp_1905_;
}
else
{
lean_inc(v_a_1904_);
lean_dec(v___x_1866_);
v___x_1906_ = lean_box(0);
v_isShared_1907_ = v_isSharedCheck_1911_;
goto v_resetjp_1905_;
}
v_resetjp_1905_:
{
lean_object* v___x_1909_; 
if (v_isShared_1907_ == 0)
{
v___x_1909_ = v___x_1906_;
goto v_reusejp_1908_;
}
else
{
lean_object* v_reuseFailAlloc_1910_; 
v_reuseFailAlloc_1910_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1910_, 0, v_a_1904_);
v___x_1909_ = v_reuseFailAlloc_1910_;
goto v_reusejp_1908_;
}
v_reusejp_1908_:
{
return v___x_1909_;
}
}
}
}
}
}
else
{
lean_object* v_a_1914_; lean_object* v___x_1916_; uint8_t v_isShared_1917_; uint8_t v_isSharedCheck_1921_; 
lean_dec(v_a_1839_);
lean_dec(v_a_1836_);
lean_dec_ref_known(v___x_1831_, 2);
lean_dec(v_a_1828_);
lean_dec(v_a_1821_);
lean_dec_ref(v___y_1815_);
lean_dec_ref(v_t_1813_);
v_a_1914_ = lean_ctor_get(v___x_1840_, 0);
v_isSharedCheck_1921_ = !lean_is_exclusive(v___x_1840_);
if (v_isSharedCheck_1921_ == 0)
{
v___x_1916_ = v___x_1840_;
v_isShared_1917_ = v_isSharedCheck_1921_;
goto v_resetjp_1915_;
}
else
{
lean_inc(v_a_1914_);
lean_dec(v___x_1840_);
v___x_1916_ = lean_box(0);
v_isShared_1917_ = v_isSharedCheck_1921_;
goto v_resetjp_1915_;
}
v_resetjp_1915_:
{
lean_object* v___x_1919_; 
if (v_isShared_1917_ == 0)
{
v___x_1919_ = v___x_1916_;
goto v_reusejp_1918_;
}
else
{
lean_object* v_reuseFailAlloc_1920_; 
v_reuseFailAlloc_1920_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1920_, 0, v_a_1914_);
v___x_1919_ = v_reuseFailAlloc_1920_;
goto v_reusejp_1918_;
}
v_reusejp_1918_:
{
return v___x_1919_;
}
}
}
}
else
{
lean_object* v_a_1922_; lean_object* v___x_1924_; uint8_t v_isShared_1925_; uint8_t v_isSharedCheck_1929_; 
lean_dec_ref_known(v___x_1837_, 1);
lean_dec(v_a_1836_);
lean_dec_ref_known(v___x_1831_, 2);
lean_dec(v_a_1828_);
lean_dec(v_a_1821_);
lean_dec_ref(v___y_1815_);
lean_dec_ref(v_t_1813_);
v_a_1922_ = lean_ctor_get(v___x_1838_, 0);
v_isSharedCheck_1929_ = !lean_is_exclusive(v___x_1838_);
if (v_isSharedCheck_1929_ == 0)
{
v___x_1924_ = v___x_1838_;
v_isShared_1925_ = v_isSharedCheck_1929_;
goto v_resetjp_1923_;
}
else
{
lean_inc(v_a_1922_);
lean_dec(v___x_1838_);
v___x_1924_ = lean_box(0);
v_isShared_1925_ = v_isSharedCheck_1929_;
goto v_resetjp_1923_;
}
v_resetjp_1923_:
{
lean_object* v___x_1927_; 
if (v_isShared_1925_ == 0)
{
v___x_1927_ = v___x_1924_;
goto v_reusejp_1926_;
}
else
{
lean_object* v_reuseFailAlloc_1928_; 
v_reuseFailAlloc_1928_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1928_, 0, v_a_1922_);
v___x_1927_ = v_reuseFailAlloc_1928_;
goto v_reusejp_1926_;
}
v_reusejp_1926_:
{
return v___x_1927_;
}
}
}
}
else
{
lean_object* v_a_1930_; lean_object* v___x_1932_; uint8_t v_isShared_1933_; uint8_t v_isSharedCheck_1937_; 
lean_dec_ref_known(v___x_1831_, 2);
lean_dec(v_a_1828_);
lean_dec(v_a_1821_);
lean_dec_ref(v___y_1815_);
lean_dec_ref(v_t_1813_);
v_a_1930_ = lean_ctor_get(v___x_1835_, 0);
v_isSharedCheck_1937_ = !lean_is_exclusive(v___x_1835_);
if (v_isSharedCheck_1937_ == 0)
{
v___x_1932_ = v___x_1835_;
v_isShared_1933_ = v_isSharedCheck_1937_;
goto v_resetjp_1931_;
}
else
{
lean_inc(v_a_1930_);
lean_dec(v___x_1835_);
v___x_1932_ = lean_box(0);
v_isShared_1933_ = v_isSharedCheck_1937_;
goto v_resetjp_1931_;
}
v_resetjp_1931_:
{
lean_object* v___x_1935_; 
if (v_isShared_1933_ == 0)
{
v___x_1935_ = v___x_1932_;
goto v_reusejp_1934_;
}
else
{
lean_object* v_reuseFailAlloc_1936_; 
v_reuseFailAlloc_1936_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1936_, 0, v_a_1930_);
v___x_1935_ = v_reuseFailAlloc_1936_;
goto v_reusejp_1934_;
}
v_reusejp_1934_:
{
return v___x_1935_;
}
}
}
}
else
{
lean_object* v_a_1938_; lean_object* v___x_1940_; uint8_t v_isShared_1941_; uint8_t v_isSharedCheck_1945_; 
lean_dec(v_a_1821_);
lean_dec_ref(v___y_1815_);
lean_dec_ref(v_t_1813_);
v_a_1938_ = lean_ctor_get(v___x_1827_, 0);
v_isSharedCheck_1945_ = !lean_is_exclusive(v___x_1827_);
if (v_isSharedCheck_1945_ == 0)
{
v___x_1940_ = v___x_1827_;
v_isShared_1941_ = v_isSharedCheck_1945_;
goto v_resetjp_1939_;
}
else
{
lean_inc(v_a_1938_);
lean_dec(v___x_1827_);
v___x_1940_ = lean_box(0);
v_isShared_1941_ = v_isSharedCheck_1945_;
goto v_resetjp_1939_;
}
v_resetjp_1939_:
{
lean_object* v___x_1943_; 
if (v_isShared_1941_ == 0)
{
v___x_1943_ = v___x_1940_;
goto v_reusejp_1942_;
}
else
{
lean_object* v_reuseFailAlloc_1944_; 
v_reuseFailAlloc_1944_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1944_, 0, v_a_1938_);
v___x_1943_ = v_reuseFailAlloc_1944_;
goto v_reusejp_1942_;
}
v_reusejp_1942_:
{
return v___x_1943_;
}
}
}
}
else
{
lean_object* v_a_1946_; lean_object* v___x_1948_; uint8_t v_isShared_1949_; uint8_t v_isSharedCheck_1953_; 
lean_dec_ref(v___y_1815_);
lean_dec_ref(v_t_1813_);
v_a_1946_ = lean_ctor_get(v___x_1820_, 0);
v_isSharedCheck_1953_ = !lean_is_exclusive(v___x_1820_);
if (v_isSharedCheck_1953_ == 0)
{
v___x_1948_ = v___x_1820_;
v_isShared_1949_ = v_isSharedCheck_1953_;
goto v_resetjp_1947_;
}
else
{
lean_inc(v_a_1946_);
lean_dec(v___x_1820_);
v___x_1948_ = lean_box(0);
v_isShared_1949_ = v_isSharedCheck_1953_;
goto v_resetjp_1947_;
}
v_resetjp_1947_:
{
lean_object* v___x_1951_; 
if (v_isShared_1949_ == 0)
{
v___x_1951_ = v___x_1948_;
goto v_reusejp_1950_;
}
else
{
lean_object* v_reuseFailAlloc_1952_; 
v_reuseFailAlloc_1952_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1952_, 0, v_a_1946_);
v___x_1951_ = v_reuseFailAlloc_1952_;
goto v_reusejp_1950_;
}
v_reusejp_1950_:
{
return v___x_1951_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult___lam__1___boxed(lean_object* v_t_1954_, lean_object* v___x_1955_, lean_object* v___y_1956_, lean_object* v___y_1957_, lean_object* v___y_1958_, lean_object* v___y_1959_, lean_object* v___y_1960_){
_start:
{
uint8_t v___x_11972__boxed_1961_; lean_object* v_res_1962_; 
v___x_11972__boxed_1961_ = lean_unbox(v___x_1955_);
v_res_1962_ = lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult___lam__1(v_t_1954_, v___x_11972__boxed_1961_, v___y_1956_, v___y_1957_, v___y_1958_, v___y_1959_);
lean_dec(v___y_1959_);
lean_dec_ref(v___y_1958_);
lean_dec(v___y_1957_);
return v_res_1962_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult___lam__2(lean_object* v_t_1963_, uint8_t v___x_1964_, lean_object* v___y_1965_, lean_object* v___y_1966_, lean_object* v___y_1967_, lean_object* v___y_1968_){
_start:
{
lean_object* v___x_1970_; 
v___x_1970_ = l_Lean_Meta_mkFreshLevelMVar(v___y_1965_, v___y_1966_, v___y_1967_, v___y_1968_);
if (lean_obj_tag(v___x_1970_) == 0)
{
lean_object* v_a_1971_; lean_object* v___x_1972_; lean_object* v___x_1973_; lean_object* v___x_1974_; uint8_t v___x_1975_; lean_object* v___x_1976_; lean_object* v___x_1977_; 
v_a_1971_ = lean_ctor_get(v___x_1970_, 0);
lean_inc_n(v_a_1971_, 2);
lean_dec_ref_known(v___x_1970_, 1);
v___x_1972_ = l_Lean_Level_succ___override(v_a_1971_);
v___x_1973_ = l_Lean_Expr_sort___override(v___x_1972_);
v___x_1974_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1974_, 0, v___x_1973_);
v___x_1975_ = 0;
v___x_1976_ = lean_box(0);
v___x_1977_ = l_Lean_Meta_mkFreshExprMVar(v___x_1974_, v___x_1975_, v___x_1976_, v___y_1965_, v___y_1966_, v___y_1967_, v___y_1968_);
if (lean_obj_tag(v___x_1977_) == 0)
{
lean_object* v_a_1978_; lean_object* v___x_1979_; lean_object* v___x_1980_; lean_object* v___x_1981_; lean_object* v___x_1982_; lean_object* v___x_1983_; lean_object* v___x_1984_; lean_object* v___x_1985_; 
v_a_1978_ = lean_ctor_get(v___x_1977_, 0);
lean_inc_n(v_a_1978_, 2);
lean_dec_ref_known(v___x_1977_, 1);
v___x_1979_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__2___closed__1));
v___x_1980_ = lean_box(0);
lean_inc(v_a_1971_);
v___x_1981_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1981_, 0, v_a_1971_);
lean_ctor_set(v___x_1981_, 1, v___x_1980_);
lean_inc_ref(v___x_1981_);
v___x_1982_ = l_Lean_Expr_const___override(v___x_1979_, v___x_1981_);
v___x_1983_ = l_Lean_Expr_app___override(v___x_1982_, v_a_1978_);
v___x_1984_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1984_, 0, v___x_1983_);
v___x_1985_ = l_Lean_Meta_mkFreshExprMVar(v___x_1984_, v___x_1975_, v___x_1976_, v___y_1965_, v___y_1966_, v___y_1967_, v___y_1968_);
if (lean_obj_tag(v___x_1985_) == 0)
{
lean_object* v_a_1986_; lean_object* v___x_1987_; lean_object* v___x_1988_; 
v_a_1986_ = lean_ctor_get(v___x_1985_, 0);
lean_inc(v_a_1986_);
lean_dec_ref_known(v___x_1985_, 1);
lean_inc(v_a_1978_);
v___x_1987_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1987_, 0, v_a_1978_);
lean_inc_ref(v___x_1987_);
v___x_1988_ = l_Lean_Meta_mkFreshExprMVar(v___x_1987_, v___x_1975_, v___x_1976_, v___y_1965_, v___y_1966_, v___y_1967_, v___y_1968_);
if (lean_obj_tag(v___x_1988_) == 0)
{
lean_object* v_a_1989_; lean_object* v___x_1990_; 
v_a_1989_ = lean_ctor_get(v___x_1988_, 0);
lean_inc(v_a_1989_);
lean_dec_ref_known(v___x_1988_, 1);
v___x_1990_ = l_Lean_Meta_mkFreshExprMVar(v___x_1987_, v___x_1975_, v___x_1976_, v___y_1965_, v___y_1966_, v___y_1967_, v___y_1968_);
if (lean_obj_tag(v___x_1990_) == 0)
{
lean_object* v_a_1991_; lean_object* v_keyedConfig_1992_; uint8_t v_trackZetaDelta_1993_; lean_object* v_zetaDeltaSet_1994_; lean_object* v_lctx_1995_; lean_object* v_localInstances_1996_; lean_object* v_defEqCtx_x3f_1997_; lean_object* v_synthPendingDepth_1998_; lean_object* v_customCanUnfoldPredicate_x3f_1999_; uint8_t v_univApprox_2000_; uint8_t v_inTypeClassResolution_2001_; uint8_t v_cacheInferType_2002_; lean_object* v___x_2004_; uint8_t v_isShared_2005_; uint8_t v_isSharedCheck_2063_; 
v_a_1991_ = lean_ctor_get(v___x_1990_, 0);
lean_inc(v_a_1991_);
lean_dec_ref_known(v___x_1990_, 1);
v_keyedConfig_1992_ = lean_ctor_get(v___y_1965_, 0);
v_trackZetaDelta_1993_ = lean_ctor_get_uint8(v___y_1965_, sizeof(void*)*7);
v_zetaDeltaSet_1994_ = lean_ctor_get(v___y_1965_, 1);
v_lctx_1995_ = lean_ctor_get(v___y_1965_, 2);
v_localInstances_1996_ = lean_ctor_get(v___y_1965_, 3);
v_defEqCtx_x3f_1997_ = lean_ctor_get(v___y_1965_, 4);
v_synthPendingDepth_1998_ = lean_ctor_get(v___y_1965_, 5);
v_customCanUnfoldPredicate_x3f_1999_ = lean_ctor_get(v___y_1965_, 6);
v_univApprox_2000_ = lean_ctor_get_uint8(v___y_1965_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_2001_ = lean_ctor_get_uint8(v___y_1965_, sizeof(void*)*7 + 2);
v_cacheInferType_2002_ = lean_ctor_get_uint8(v___y_1965_, sizeof(void*)*7 + 3);
v_isSharedCheck_2063_ = !lean_is_exclusive(v___y_1965_);
if (v_isSharedCheck_2063_ == 0)
{
v___x_2004_ = v___y_1965_;
v_isShared_2005_ = v_isSharedCheck_2063_;
goto v_resetjp_2003_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_1999_);
lean_inc(v_synthPendingDepth_1998_);
lean_inc(v_defEqCtx_x3f_1997_);
lean_inc(v_localInstances_1996_);
lean_inc(v_lctx_1995_);
lean_inc(v_zetaDeltaSet_1994_);
lean_inc(v_keyedConfig_1992_);
lean_dec(v___y_1965_);
v___x_2004_ = lean_box(0);
v_isShared_2005_ = v_isSharedCheck_2063_;
goto v_resetjp_2003_;
}
v_resetjp_2003_:
{
lean_object* v___x_2006_; lean_object* v___x_2007_; lean_object* v___x_2008_; lean_object* v___x_2009_; lean_object* v___x_2010_; lean_object* v___x_2011_; uint8_t v___x_2012_; lean_object* v___x_2013_; lean_object* v___x_2015_; 
v___x_2006_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__4___closed__2));
v___x_2007_ = l_Lean_Expr_const___override(v___x_2006_, v___x_1981_);
lean_inc(v_a_1978_);
v___x_2008_ = l_Lean_Expr_app___override(v___x_2007_, v_a_1978_);
lean_inc(v_a_1986_);
v___x_2009_ = l_Lean_Expr_app___override(v___x_2008_, v_a_1986_);
lean_inc(v_a_1989_);
v___x_2010_ = l_Lean_Expr_app___override(v___x_2009_, v_a_1989_);
lean_inc(v_a_1991_);
v___x_2011_ = l_Lean_Expr_app___override(v___x_2010_, v_a_1991_);
v___x_2012_ = 2;
v___x_2013_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_2012_, v_keyedConfig_1992_);
if (v_isShared_2005_ == 0)
{
lean_ctor_set(v___x_2004_, 0, v___x_2013_);
v___x_2015_ = v___x_2004_;
goto v_reusejp_2014_;
}
else
{
lean_object* v_reuseFailAlloc_2062_; 
v_reuseFailAlloc_2062_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_2062_, 0, v___x_2013_);
lean_ctor_set(v_reuseFailAlloc_2062_, 1, v_zetaDeltaSet_1994_);
lean_ctor_set(v_reuseFailAlloc_2062_, 2, v_lctx_1995_);
lean_ctor_set(v_reuseFailAlloc_2062_, 3, v_localInstances_1996_);
lean_ctor_set(v_reuseFailAlloc_2062_, 4, v_defEqCtx_x3f_1997_);
lean_ctor_set(v_reuseFailAlloc_2062_, 5, v_synthPendingDepth_1998_);
lean_ctor_set(v_reuseFailAlloc_2062_, 6, v_customCanUnfoldPredicate_x3f_1999_);
lean_ctor_set_uint8(v_reuseFailAlloc_2062_, sizeof(void*)*7, v_trackZetaDelta_1993_);
lean_ctor_set_uint8(v_reuseFailAlloc_2062_, sizeof(void*)*7 + 1, v_univApprox_2000_);
lean_ctor_set_uint8(v_reuseFailAlloc_2062_, sizeof(void*)*7 + 2, v_inTypeClassResolution_2001_);
lean_ctor_set_uint8(v_reuseFailAlloc_2062_, sizeof(void*)*7 + 3, v_cacheInferType_2002_);
v___x_2015_ = v_reuseFailAlloc_2062_;
goto v_reusejp_2014_;
}
v_reusejp_2014_:
{
lean_object* v___x_2016_; 
v___x_2016_ = l_Lean_Meta_isExprDefEq(v___x_2011_, v_t_1963_, v___x_2015_, v___y_1966_, v___y_1967_, v___y_1968_);
lean_dec_ref(v___x_2015_);
if (lean_obj_tag(v___x_2016_) == 0)
{
lean_object* v_a_2017_; lean_object* v___x_2019_; uint8_t v_isShared_2020_; uint8_t v_isSharedCheck_2053_; 
v_a_2017_ = lean_ctor_get(v___x_2016_, 0);
v_isSharedCheck_2053_ = !lean_is_exclusive(v___x_2016_);
if (v_isSharedCheck_2053_ == 0)
{
v___x_2019_ = v___x_2016_;
v_isShared_2020_ = v_isSharedCheck_2053_;
goto v_resetjp_2018_;
}
else
{
lean_inc(v_a_2017_);
lean_dec(v___x_2016_);
v___x_2019_ = lean_box(0);
v_isShared_2020_ = v_isSharedCheck_2053_;
goto v_resetjp_2018_;
}
v_resetjp_2018_:
{
uint8_t v___x_2021_; 
v___x_2021_ = lean_unbox(v_a_2017_);
if (v___x_2021_ == 0)
{
lean_object* v___x_2022_; lean_object* v___x_2023_; lean_object* v___x_2024_; lean_object* v___x_2025_; lean_object* v___x_2026_; lean_object* v___x_2027_; lean_object* v___x_2029_; 
lean_dec(v_a_2017_);
v___x_2022_ = lean_box(v___x_1964_);
v___x_2023_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2023_, 0, v_a_1991_);
lean_ctor_set(v___x_2023_, 1, v___x_2022_);
v___x_2024_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2024_, 0, v_a_1989_);
lean_ctor_set(v___x_2024_, 1, v___x_2023_);
v___x_2025_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2025_, 0, v_a_1986_);
lean_ctor_set(v___x_2025_, 1, v___x_2024_);
v___x_2026_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2026_, 0, v_a_1978_);
lean_ctor_set(v___x_2026_, 1, v___x_2025_);
v___x_2027_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2027_, 0, v_a_1971_);
lean_ctor_set(v___x_2027_, 1, v___x_2026_);
if (v_isShared_2020_ == 0)
{
lean_ctor_set(v___x_2019_, 0, v___x_2027_);
v___x_2029_ = v___x_2019_;
goto v_reusejp_2028_;
}
else
{
lean_object* v_reuseFailAlloc_2030_; 
v_reuseFailAlloc_2030_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2030_, 0, v___x_2027_);
v___x_2029_ = v_reuseFailAlloc_2030_;
goto v_reusejp_2028_;
}
v_reusejp_2028_:
{
return v___x_2029_;
}
}
else
{
lean_object* v___x_2031_; lean_object* v_a_2032_; lean_object* v___x_2033_; lean_object* v_a_2034_; lean_object* v___x_2035_; lean_object* v_a_2036_; lean_object* v___x_2037_; lean_object* v_a_2038_; lean_object* v___x_2039_; lean_object* v_a_2040_; lean_object* v___x_2042_; uint8_t v_isShared_2043_; uint8_t v_isSharedCheck_2052_; 
lean_del_object(v___x_2019_);
v___x_2031_ = lp_mathlib_Lean_instantiateLevelMVars___at___00Mathlib_Tactic_Bound_hypPriority_spec__0___redArg(v_a_1971_, v___y_1966_);
v_a_2032_ = lean_ctor_get(v___x_2031_, 0);
lean_inc(v_a_2032_);
lean_dec_ref(v___x_2031_);
v___x_2033_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Bound_isZero_spec__0___redArg(v_a_1978_, v___y_1966_);
v_a_2034_ = lean_ctor_get(v___x_2033_, 0);
lean_inc(v_a_2034_);
lean_dec_ref(v___x_2033_);
v___x_2035_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Bound_isZero_spec__0___redArg(v_a_1986_, v___y_1966_);
v_a_2036_ = lean_ctor_get(v___x_2035_, 0);
lean_inc(v_a_2036_);
lean_dec_ref(v___x_2035_);
v___x_2037_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Bound_isZero_spec__0___redArg(v_a_1989_, v___y_1966_);
v_a_2038_ = lean_ctor_get(v___x_2037_, 0);
lean_inc(v_a_2038_);
lean_dec_ref(v___x_2037_);
v___x_2039_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Bound_isZero_spec__0___redArg(v_a_1991_, v___y_1966_);
v_a_2040_ = lean_ctor_get(v___x_2039_, 0);
v_isSharedCheck_2052_ = !lean_is_exclusive(v___x_2039_);
if (v_isSharedCheck_2052_ == 0)
{
v___x_2042_ = v___x_2039_;
v_isShared_2043_ = v_isSharedCheck_2052_;
goto v_resetjp_2041_;
}
else
{
lean_inc(v_a_2040_);
lean_dec(v___x_2039_);
v___x_2042_ = lean_box(0);
v_isShared_2043_ = v_isSharedCheck_2052_;
goto v_resetjp_2041_;
}
v_resetjp_2041_:
{
lean_object* v___x_2044_; lean_object* v___x_2045_; lean_object* v___x_2046_; lean_object* v___x_2047_; lean_object* v___x_2048_; lean_object* v___x_2050_; 
v___x_2044_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2044_, 0, v_a_2040_);
lean_ctor_set(v___x_2044_, 1, v_a_2017_);
v___x_2045_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2045_, 0, v_a_2038_);
lean_ctor_set(v___x_2045_, 1, v___x_2044_);
v___x_2046_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2046_, 0, v_a_2036_);
lean_ctor_set(v___x_2046_, 1, v___x_2045_);
v___x_2047_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2047_, 0, v_a_2034_);
lean_ctor_set(v___x_2047_, 1, v___x_2046_);
v___x_2048_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2048_, 0, v_a_2032_);
lean_ctor_set(v___x_2048_, 1, v___x_2047_);
if (v_isShared_2043_ == 0)
{
lean_ctor_set(v___x_2042_, 0, v___x_2048_);
v___x_2050_ = v___x_2042_;
goto v_reusejp_2049_;
}
else
{
lean_object* v_reuseFailAlloc_2051_; 
v_reuseFailAlloc_2051_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2051_, 0, v___x_2048_);
v___x_2050_ = v_reuseFailAlloc_2051_;
goto v_reusejp_2049_;
}
v_reusejp_2049_:
{
return v___x_2050_;
}
}
}
}
}
else
{
lean_object* v_a_2054_; lean_object* v___x_2056_; uint8_t v_isShared_2057_; uint8_t v_isSharedCheck_2061_; 
lean_dec(v_a_1991_);
lean_dec(v_a_1989_);
lean_dec(v_a_1986_);
lean_dec(v_a_1978_);
lean_dec(v_a_1971_);
v_a_2054_ = lean_ctor_get(v___x_2016_, 0);
v_isSharedCheck_2061_ = !lean_is_exclusive(v___x_2016_);
if (v_isSharedCheck_2061_ == 0)
{
v___x_2056_ = v___x_2016_;
v_isShared_2057_ = v_isSharedCheck_2061_;
goto v_resetjp_2055_;
}
else
{
lean_inc(v_a_2054_);
lean_dec(v___x_2016_);
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
}
else
{
lean_object* v_a_2064_; lean_object* v___x_2066_; uint8_t v_isShared_2067_; uint8_t v_isSharedCheck_2071_; 
lean_dec(v_a_1989_);
lean_dec(v_a_1986_);
lean_dec_ref_known(v___x_1981_, 2);
lean_dec(v_a_1978_);
lean_dec(v_a_1971_);
lean_dec_ref(v___y_1965_);
lean_dec_ref(v_t_1963_);
v_a_2064_ = lean_ctor_get(v___x_1990_, 0);
v_isSharedCheck_2071_ = !lean_is_exclusive(v___x_1990_);
if (v_isSharedCheck_2071_ == 0)
{
v___x_2066_ = v___x_1990_;
v_isShared_2067_ = v_isSharedCheck_2071_;
goto v_resetjp_2065_;
}
else
{
lean_inc(v_a_2064_);
lean_dec(v___x_1990_);
v___x_2066_ = lean_box(0);
v_isShared_2067_ = v_isSharedCheck_2071_;
goto v_resetjp_2065_;
}
v_resetjp_2065_:
{
lean_object* v___x_2069_; 
if (v_isShared_2067_ == 0)
{
v___x_2069_ = v___x_2066_;
goto v_reusejp_2068_;
}
else
{
lean_object* v_reuseFailAlloc_2070_; 
v_reuseFailAlloc_2070_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2070_, 0, v_a_2064_);
v___x_2069_ = v_reuseFailAlloc_2070_;
goto v_reusejp_2068_;
}
v_reusejp_2068_:
{
return v___x_2069_;
}
}
}
}
else
{
lean_object* v_a_2072_; lean_object* v___x_2074_; uint8_t v_isShared_2075_; uint8_t v_isSharedCheck_2079_; 
lean_dec_ref_known(v___x_1987_, 1);
lean_dec(v_a_1986_);
lean_dec_ref_known(v___x_1981_, 2);
lean_dec(v_a_1978_);
lean_dec(v_a_1971_);
lean_dec_ref(v___y_1965_);
lean_dec_ref(v_t_1963_);
v_a_2072_ = lean_ctor_get(v___x_1988_, 0);
v_isSharedCheck_2079_ = !lean_is_exclusive(v___x_1988_);
if (v_isSharedCheck_2079_ == 0)
{
v___x_2074_ = v___x_1988_;
v_isShared_2075_ = v_isSharedCheck_2079_;
goto v_resetjp_2073_;
}
else
{
lean_inc(v_a_2072_);
lean_dec(v___x_1988_);
v___x_2074_ = lean_box(0);
v_isShared_2075_ = v_isSharedCheck_2079_;
goto v_resetjp_2073_;
}
v_resetjp_2073_:
{
lean_object* v___x_2077_; 
if (v_isShared_2075_ == 0)
{
v___x_2077_ = v___x_2074_;
goto v_reusejp_2076_;
}
else
{
lean_object* v_reuseFailAlloc_2078_; 
v_reuseFailAlloc_2078_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2078_, 0, v_a_2072_);
v___x_2077_ = v_reuseFailAlloc_2078_;
goto v_reusejp_2076_;
}
v_reusejp_2076_:
{
return v___x_2077_;
}
}
}
}
else
{
lean_object* v_a_2080_; lean_object* v___x_2082_; uint8_t v_isShared_2083_; uint8_t v_isSharedCheck_2087_; 
lean_dec_ref_known(v___x_1981_, 2);
lean_dec(v_a_1978_);
lean_dec(v_a_1971_);
lean_dec_ref(v___y_1965_);
lean_dec_ref(v_t_1963_);
v_a_2080_ = lean_ctor_get(v___x_1985_, 0);
v_isSharedCheck_2087_ = !lean_is_exclusive(v___x_1985_);
if (v_isSharedCheck_2087_ == 0)
{
v___x_2082_ = v___x_1985_;
v_isShared_2083_ = v_isSharedCheck_2087_;
goto v_resetjp_2081_;
}
else
{
lean_inc(v_a_2080_);
lean_dec(v___x_1985_);
v___x_2082_ = lean_box(0);
v_isShared_2083_ = v_isSharedCheck_2087_;
goto v_resetjp_2081_;
}
v_resetjp_2081_:
{
lean_object* v___x_2085_; 
if (v_isShared_2083_ == 0)
{
v___x_2085_ = v___x_2082_;
goto v_reusejp_2084_;
}
else
{
lean_object* v_reuseFailAlloc_2086_; 
v_reuseFailAlloc_2086_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2086_, 0, v_a_2080_);
v___x_2085_ = v_reuseFailAlloc_2086_;
goto v_reusejp_2084_;
}
v_reusejp_2084_:
{
return v___x_2085_;
}
}
}
}
else
{
lean_object* v_a_2088_; lean_object* v___x_2090_; uint8_t v_isShared_2091_; uint8_t v_isSharedCheck_2095_; 
lean_dec(v_a_1971_);
lean_dec_ref(v___y_1965_);
lean_dec_ref(v_t_1963_);
v_a_2088_ = lean_ctor_get(v___x_1977_, 0);
v_isSharedCheck_2095_ = !lean_is_exclusive(v___x_1977_);
if (v_isSharedCheck_2095_ == 0)
{
v___x_2090_ = v___x_1977_;
v_isShared_2091_ = v_isSharedCheck_2095_;
goto v_resetjp_2089_;
}
else
{
lean_inc(v_a_2088_);
lean_dec(v___x_1977_);
v___x_2090_ = lean_box(0);
v_isShared_2091_ = v_isSharedCheck_2095_;
goto v_resetjp_2089_;
}
v_resetjp_2089_:
{
lean_object* v___x_2093_; 
if (v_isShared_2091_ == 0)
{
v___x_2093_ = v___x_2090_;
goto v_reusejp_2092_;
}
else
{
lean_object* v_reuseFailAlloc_2094_; 
v_reuseFailAlloc_2094_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2094_, 0, v_a_2088_);
v___x_2093_ = v_reuseFailAlloc_2094_;
goto v_reusejp_2092_;
}
v_reusejp_2092_:
{
return v___x_2093_;
}
}
}
}
else
{
lean_object* v_a_2096_; lean_object* v___x_2098_; uint8_t v_isShared_2099_; uint8_t v_isSharedCheck_2103_; 
lean_dec_ref(v___y_1965_);
lean_dec_ref(v_t_1963_);
v_a_2096_ = lean_ctor_get(v___x_1970_, 0);
v_isSharedCheck_2103_ = !lean_is_exclusive(v___x_1970_);
if (v_isSharedCheck_2103_ == 0)
{
v___x_2098_ = v___x_1970_;
v_isShared_2099_ = v_isSharedCheck_2103_;
goto v_resetjp_2097_;
}
else
{
lean_inc(v_a_2096_);
lean_dec(v___x_1970_);
v___x_2098_ = lean_box(0);
v_isShared_2099_ = v_isSharedCheck_2103_;
goto v_resetjp_2097_;
}
v_resetjp_2097_:
{
lean_object* v___x_2101_; 
if (v_isShared_2099_ == 0)
{
v___x_2101_ = v___x_2098_;
goto v_reusejp_2100_;
}
else
{
lean_object* v_reuseFailAlloc_2102_; 
v_reuseFailAlloc_2102_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2102_, 0, v_a_2096_);
v___x_2101_ = v_reuseFailAlloc_2102_;
goto v_reusejp_2100_;
}
v_reusejp_2100_:
{
return v___x_2101_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult___lam__2___boxed(lean_object* v_t_2104_, lean_object* v___x_2105_, lean_object* v___y_2106_, lean_object* v___y_2107_, lean_object* v___y_2108_, lean_object* v___y_2109_, lean_object* v___y_2110_){
_start:
{
uint8_t v___x_12251__boxed_2111_; lean_object* v_res_2112_; 
v___x_12251__boxed_2111_ = lean_unbox(v___x_2105_);
v_res_2112_ = lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult___lam__2(v_t_2104_, v___x_12251__boxed_2111_, v___y_2106_, v___y_2107_, v___y_2108_, v___y_2109_);
lean_dec(v___y_2109_);
lean_dec_ref(v___y_2108_);
lean_dec(v___y_2107_);
return v_res_2112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult___lam__3(lean_object* v_t_2113_, lean_object* v___y_2114_, lean_object* v___y_2115_, lean_object* v___y_2116_, lean_object* v___y_2117_){
_start:
{
lean_object* v___x_2119_; 
v___x_2119_ = l_Lean_Meta_mkFreshLevelMVar(v___y_2114_, v___y_2115_, v___y_2116_, v___y_2117_);
if (lean_obj_tag(v___x_2119_) == 0)
{
lean_object* v_a_2120_; lean_object* v___x_2121_; lean_object* v___x_2122_; lean_object* v___x_2123_; uint8_t v___x_2124_; lean_object* v___x_2125_; lean_object* v___x_2126_; 
v_a_2120_ = lean_ctor_get(v___x_2119_, 0);
lean_inc_n(v_a_2120_, 2);
lean_dec_ref_known(v___x_2119_, 1);
v___x_2121_ = l_Lean_Level_succ___override(v_a_2120_);
v___x_2122_ = l_Lean_Expr_sort___override(v___x_2121_);
v___x_2123_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2123_, 0, v___x_2122_);
v___x_2124_ = 0;
v___x_2125_ = lean_box(0);
v___x_2126_ = l_Lean_Meta_mkFreshExprMVar(v___x_2123_, v___x_2124_, v___x_2125_, v___y_2114_, v___y_2115_, v___y_2116_, v___y_2117_);
if (lean_obj_tag(v___x_2126_) == 0)
{
lean_object* v_a_2127_; lean_object* v___x_2128_; lean_object* v___x_2129_; lean_object* v___x_2130_; lean_object* v___x_2131_; lean_object* v___x_2132_; lean_object* v___x_2133_; lean_object* v___x_2134_; 
v_a_2127_ = lean_ctor_get(v___x_2126_, 0);
lean_inc_n(v_a_2127_, 2);
lean_dec_ref_known(v___x_2126_, 1);
v___x_2128_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__3___closed__1));
v___x_2129_ = lean_box(0);
lean_inc(v_a_2120_);
v___x_2130_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2130_, 0, v_a_2120_);
lean_ctor_set(v___x_2130_, 1, v___x_2129_);
lean_inc_ref(v___x_2130_);
v___x_2131_ = l_Lean_Expr_const___override(v___x_2128_, v___x_2130_);
v___x_2132_ = l_Lean_Expr_app___override(v___x_2131_, v_a_2127_);
v___x_2133_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2133_, 0, v___x_2132_);
v___x_2134_ = l_Lean_Meta_mkFreshExprMVar(v___x_2133_, v___x_2124_, v___x_2125_, v___y_2114_, v___y_2115_, v___y_2116_, v___y_2117_);
if (lean_obj_tag(v___x_2134_) == 0)
{
lean_object* v_a_2135_; lean_object* v___x_2136_; lean_object* v___x_2137_; 
v_a_2135_ = lean_ctor_get(v___x_2134_, 0);
lean_inc(v_a_2135_);
lean_dec_ref_known(v___x_2134_, 1);
lean_inc(v_a_2127_);
v___x_2136_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2136_, 0, v_a_2127_);
lean_inc_ref(v___x_2136_);
v___x_2137_ = l_Lean_Meta_mkFreshExprMVar(v___x_2136_, v___x_2124_, v___x_2125_, v___y_2114_, v___y_2115_, v___y_2116_, v___y_2117_);
if (lean_obj_tag(v___x_2137_) == 0)
{
lean_object* v_a_2138_; lean_object* v___x_2139_; 
v_a_2138_ = lean_ctor_get(v___x_2137_, 0);
lean_inc(v_a_2138_);
lean_dec_ref_known(v___x_2137_, 1);
v___x_2139_ = l_Lean_Meta_mkFreshExprMVar(v___x_2136_, v___x_2124_, v___x_2125_, v___y_2114_, v___y_2115_, v___y_2116_, v___y_2117_);
if (lean_obj_tag(v___x_2139_) == 0)
{
lean_object* v_a_2140_; lean_object* v_keyedConfig_2141_; uint8_t v_trackZetaDelta_2142_; lean_object* v_zetaDeltaSet_2143_; lean_object* v_lctx_2144_; lean_object* v_localInstances_2145_; lean_object* v_defEqCtx_x3f_2146_; lean_object* v_synthPendingDepth_2147_; lean_object* v_customCanUnfoldPredicate_x3f_2148_; uint8_t v_univApprox_2149_; uint8_t v_inTypeClassResolution_2150_; uint8_t v_cacheInferType_2151_; lean_object* v___x_2153_; uint8_t v_isShared_2154_; uint8_t v_isSharedCheck_2211_; 
v_a_2140_ = lean_ctor_get(v___x_2139_, 0);
lean_inc(v_a_2140_);
lean_dec_ref_known(v___x_2139_, 1);
v_keyedConfig_2141_ = lean_ctor_get(v___y_2114_, 0);
v_trackZetaDelta_2142_ = lean_ctor_get_uint8(v___y_2114_, sizeof(void*)*7);
v_zetaDeltaSet_2143_ = lean_ctor_get(v___y_2114_, 1);
v_lctx_2144_ = lean_ctor_get(v___y_2114_, 2);
v_localInstances_2145_ = lean_ctor_get(v___y_2114_, 3);
v_defEqCtx_x3f_2146_ = lean_ctor_get(v___y_2114_, 4);
v_synthPendingDepth_2147_ = lean_ctor_get(v___y_2114_, 5);
v_customCanUnfoldPredicate_x3f_2148_ = lean_ctor_get(v___y_2114_, 6);
v_univApprox_2149_ = lean_ctor_get_uint8(v___y_2114_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_2150_ = lean_ctor_get_uint8(v___y_2114_, sizeof(void*)*7 + 2);
v_cacheInferType_2151_ = lean_ctor_get_uint8(v___y_2114_, sizeof(void*)*7 + 3);
v_isSharedCheck_2211_ = !lean_is_exclusive(v___y_2114_);
if (v_isSharedCheck_2211_ == 0)
{
v___x_2153_ = v___y_2114_;
v_isShared_2154_ = v_isSharedCheck_2211_;
goto v_resetjp_2152_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_2148_);
lean_inc(v_synthPendingDepth_2147_);
lean_inc(v_defEqCtx_x3f_2146_);
lean_inc(v_localInstances_2145_);
lean_inc(v_lctx_2144_);
lean_inc(v_zetaDeltaSet_2143_);
lean_inc(v_keyedConfig_2141_);
lean_dec(v___y_2114_);
v___x_2153_ = lean_box(0);
v_isShared_2154_ = v_isSharedCheck_2211_;
goto v_resetjp_2152_;
}
v_resetjp_2152_:
{
lean_object* v___x_2155_; lean_object* v___x_2156_; lean_object* v___x_2157_; lean_object* v___x_2158_; lean_object* v___x_2159_; lean_object* v___x_2160_; uint8_t v___x_2161_; lean_object* v___x_2162_; lean_object* v___x_2164_; 
v___x_2155_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Bound_hypPriority___lam__5___closed__2));
v___x_2156_ = l_Lean_Expr_const___override(v___x_2155_, v___x_2130_);
lean_inc(v_a_2127_);
v___x_2157_ = l_Lean_Expr_app___override(v___x_2156_, v_a_2127_);
lean_inc(v_a_2135_);
v___x_2158_ = l_Lean_Expr_app___override(v___x_2157_, v_a_2135_);
lean_inc(v_a_2138_);
v___x_2159_ = l_Lean_Expr_app___override(v___x_2158_, v_a_2138_);
lean_inc(v_a_2140_);
v___x_2160_ = l_Lean_Expr_app___override(v___x_2159_, v_a_2140_);
v___x_2161_ = 2;
v___x_2162_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_2161_, v_keyedConfig_2141_);
if (v_isShared_2154_ == 0)
{
lean_ctor_set(v___x_2153_, 0, v___x_2162_);
v___x_2164_ = v___x_2153_;
goto v_reusejp_2163_;
}
else
{
lean_object* v_reuseFailAlloc_2210_; 
v_reuseFailAlloc_2210_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_2210_, 0, v___x_2162_);
lean_ctor_set(v_reuseFailAlloc_2210_, 1, v_zetaDeltaSet_2143_);
lean_ctor_set(v_reuseFailAlloc_2210_, 2, v_lctx_2144_);
lean_ctor_set(v_reuseFailAlloc_2210_, 3, v_localInstances_2145_);
lean_ctor_set(v_reuseFailAlloc_2210_, 4, v_defEqCtx_x3f_2146_);
lean_ctor_set(v_reuseFailAlloc_2210_, 5, v_synthPendingDepth_2147_);
lean_ctor_set(v_reuseFailAlloc_2210_, 6, v_customCanUnfoldPredicate_x3f_2148_);
lean_ctor_set_uint8(v_reuseFailAlloc_2210_, sizeof(void*)*7, v_trackZetaDelta_2142_);
lean_ctor_set_uint8(v_reuseFailAlloc_2210_, sizeof(void*)*7 + 1, v_univApprox_2149_);
lean_ctor_set_uint8(v_reuseFailAlloc_2210_, sizeof(void*)*7 + 2, v_inTypeClassResolution_2150_);
lean_ctor_set_uint8(v_reuseFailAlloc_2210_, sizeof(void*)*7 + 3, v_cacheInferType_2151_);
v___x_2164_ = v_reuseFailAlloc_2210_;
goto v_reusejp_2163_;
}
v_reusejp_2163_:
{
lean_object* v___x_2165_; 
v___x_2165_ = l_Lean_Meta_isExprDefEq(v___x_2160_, v_t_2113_, v___x_2164_, v___y_2115_, v___y_2116_, v___y_2117_);
lean_dec_ref(v___x_2164_);
if (lean_obj_tag(v___x_2165_) == 0)
{
lean_object* v_a_2166_; lean_object* v___x_2168_; uint8_t v_isShared_2169_; uint8_t v_isSharedCheck_2201_; 
v_a_2166_ = lean_ctor_get(v___x_2165_, 0);
v_isSharedCheck_2201_ = !lean_is_exclusive(v___x_2165_);
if (v_isSharedCheck_2201_ == 0)
{
v___x_2168_ = v___x_2165_;
v_isShared_2169_ = v_isSharedCheck_2201_;
goto v_resetjp_2167_;
}
else
{
lean_inc(v_a_2166_);
lean_dec(v___x_2165_);
v___x_2168_ = lean_box(0);
v_isShared_2169_ = v_isSharedCheck_2201_;
goto v_resetjp_2167_;
}
v_resetjp_2167_:
{
uint8_t v___x_2170_; 
v___x_2170_ = lean_unbox(v_a_2166_);
if (v___x_2170_ == 0)
{
lean_object* v___x_2171_; lean_object* v___x_2172_; lean_object* v___x_2173_; lean_object* v___x_2174_; lean_object* v___x_2175_; lean_object* v___x_2177_; 
v___x_2171_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2171_, 0, v_a_2140_);
lean_ctor_set(v___x_2171_, 1, v_a_2166_);
v___x_2172_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2172_, 0, v_a_2138_);
lean_ctor_set(v___x_2172_, 1, v___x_2171_);
v___x_2173_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2173_, 0, v_a_2135_);
lean_ctor_set(v___x_2173_, 1, v___x_2172_);
v___x_2174_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2174_, 0, v_a_2127_);
lean_ctor_set(v___x_2174_, 1, v___x_2173_);
v___x_2175_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2175_, 0, v_a_2120_);
lean_ctor_set(v___x_2175_, 1, v___x_2174_);
if (v_isShared_2169_ == 0)
{
lean_ctor_set(v___x_2168_, 0, v___x_2175_);
v___x_2177_ = v___x_2168_;
goto v_reusejp_2176_;
}
else
{
lean_object* v_reuseFailAlloc_2178_; 
v_reuseFailAlloc_2178_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2178_, 0, v___x_2175_);
v___x_2177_ = v_reuseFailAlloc_2178_;
goto v_reusejp_2176_;
}
v_reusejp_2176_:
{
return v___x_2177_;
}
}
else
{
lean_object* v___x_2179_; lean_object* v_a_2180_; lean_object* v___x_2181_; lean_object* v_a_2182_; lean_object* v___x_2183_; lean_object* v_a_2184_; lean_object* v___x_2185_; lean_object* v_a_2186_; lean_object* v___x_2187_; lean_object* v_a_2188_; lean_object* v___x_2190_; uint8_t v_isShared_2191_; uint8_t v_isSharedCheck_2200_; 
lean_del_object(v___x_2168_);
v___x_2179_ = lp_mathlib_Lean_instantiateLevelMVars___at___00Mathlib_Tactic_Bound_hypPriority_spec__0___redArg(v_a_2120_, v___y_2115_);
v_a_2180_ = lean_ctor_get(v___x_2179_, 0);
lean_inc(v_a_2180_);
lean_dec_ref(v___x_2179_);
v___x_2181_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Bound_isZero_spec__0___redArg(v_a_2127_, v___y_2115_);
v_a_2182_ = lean_ctor_get(v___x_2181_, 0);
lean_inc(v_a_2182_);
lean_dec_ref(v___x_2181_);
v___x_2183_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Bound_isZero_spec__0___redArg(v_a_2135_, v___y_2115_);
v_a_2184_ = lean_ctor_get(v___x_2183_, 0);
lean_inc(v_a_2184_);
lean_dec_ref(v___x_2183_);
v___x_2185_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Bound_isZero_spec__0___redArg(v_a_2138_, v___y_2115_);
v_a_2186_ = lean_ctor_get(v___x_2185_, 0);
lean_inc(v_a_2186_);
lean_dec_ref(v___x_2185_);
v___x_2187_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Bound_isZero_spec__0___redArg(v_a_2140_, v___y_2115_);
v_a_2188_ = lean_ctor_get(v___x_2187_, 0);
v_isSharedCheck_2200_ = !lean_is_exclusive(v___x_2187_);
if (v_isSharedCheck_2200_ == 0)
{
v___x_2190_ = v___x_2187_;
v_isShared_2191_ = v_isSharedCheck_2200_;
goto v_resetjp_2189_;
}
else
{
lean_inc(v_a_2188_);
lean_dec(v___x_2187_);
v___x_2190_ = lean_box(0);
v_isShared_2191_ = v_isSharedCheck_2200_;
goto v_resetjp_2189_;
}
v_resetjp_2189_:
{
lean_object* v___x_2192_; lean_object* v___x_2193_; lean_object* v___x_2194_; lean_object* v___x_2195_; lean_object* v___x_2196_; lean_object* v___x_2198_; 
v___x_2192_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2192_, 0, v_a_2188_);
lean_ctor_set(v___x_2192_, 1, v_a_2166_);
v___x_2193_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2193_, 0, v_a_2186_);
lean_ctor_set(v___x_2193_, 1, v___x_2192_);
v___x_2194_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2194_, 0, v_a_2184_);
lean_ctor_set(v___x_2194_, 1, v___x_2193_);
v___x_2195_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2195_, 0, v_a_2182_);
lean_ctor_set(v___x_2195_, 1, v___x_2194_);
v___x_2196_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2196_, 0, v_a_2180_);
lean_ctor_set(v___x_2196_, 1, v___x_2195_);
if (v_isShared_2191_ == 0)
{
lean_ctor_set(v___x_2190_, 0, v___x_2196_);
v___x_2198_ = v___x_2190_;
goto v_reusejp_2197_;
}
else
{
lean_object* v_reuseFailAlloc_2199_; 
v_reuseFailAlloc_2199_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2199_, 0, v___x_2196_);
v___x_2198_ = v_reuseFailAlloc_2199_;
goto v_reusejp_2197_;
}
v_reusejp_2197_:
{
return v___x_2198_;
}
}
}
}
}
else
{
lean_object* v_a_2202_; lean_object* v___x_2204_; uint8_t v_isShared_2205_; uint8_t v_isSharedCheck_2209_; 
lean_dec(v_a_2140_);
lean_dec(v_a_2138_);
lean_dec(v_a_2135_);
lean_dec(v_a_2127_);
lean_dec(v_a_2120_);
v_a_2202_ = lean_ctor_get(v___x_2165_, 0);
v_isSharedCheck_2209_ = !lean_is_exclusive(v___x_2165_);
if (v_isSharedCheck_2209_ == 0)
{
v___x_2204_ = v___x_2165_;
v_isShared_2205_ = v_isSharedCheck_2209_;
goto v_resetjp_2203_;
}
else
{
lean_inc(v_a_2202_);
lean_dec(v___x_2165_);
v___x_2204_ = lean_box(0);
v_isShared_2205_ = v_isSharedCheck_2209_;
goto v_resetjp_2203_;
}
v_resetjp_2203_:
{
lean_object* v___x_2207_; 
if (v_isShared_2205_ == 0)
{
v___x_2207_ = v___x_2204_;
goto v_reusejp_2206_;
}
else
{
lean_object* v_reuseFailAlloc_2208_; 
v_reuseFailAlloc_2208_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2208_, 0, v_a_2202_);
v___x_2207_ = v_reuseFailAlloc_2208_;
goto v_reusejp_2206_;
}
v_reusejp_2206_:
{
return v___x_2207_;
}
}
}
}
}
}
else
{
lean_object* v_a_2212_; lean_object* v___x_2214_; uint8_t v_isShared_2215_; uint8_t v_isSharedCheck_2219_; 
lean_dec(v_a_2138_);
lean_dec(v_a_2135_);
lean_dec_ref_known(v___x_2130_, 2);
lean_dec(v_a_2127_);
lean_dec(v_a_2120_);
lean_dec_ref(v___y_2114_);
lean_dec_ref(v_t_2113_);
v_a_2212_ = lean_ctor_get(v___x_2139_, 0);
v_isSharedCheck_2219_ = !lean_is_exclusive(v___x_2139_);
if (v_isSharedCheck_2219_ == 0)
{
v___x_2214_ = v___x_2139_;
v_isShared_2215_ = v_isSharedCheck_2219_;
goto v_resetjp_2213_;
}
else
{
lean_inc(v_a_2212_);
lean_dec(v___x_2139_);
v___x_2214_ = lean_box(0);
v_isShared_2215_ = v_isSharedCheck_2219_;
goto v_resetjp_2213_;
}
v_resetjp_2213_:
{
lean_object* v___x_2217_; 
if (v_isShared_2215_ == 0)
{
v___x_2217_ = v___x_2214_;
goto v_reusejp_2216_;
}
else
{
lean_object* v_reuseFailAlloc_2218_; 
v_reuseFailAlloc_2218_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2218_, 0, v_a_2212_);
v___x_2217_ = v_reuseFailAlloc_2218_;
goto v_reusejp_2216_;
}
v_reusejp_2216_:
{
return v___x_2217_;
}
}
}
}
else
{
lean_object* v_a_2220_; lean_object* v___x_2222_; uint8_t v_isShared_2223_; uint8_t v_isSharedCheck_2227_; 
lean_dec_ref_known(v___x_2136_, 1);
lean_dec(v_a_2135_);
lean_dec_ref_known(v___x_2130_, 2);
lean_dec(v_a_2127_);
lean_dec(v_a_2120_);
lean_dec_ref(v___y_2114_);
lean_dec_ref(v_t_2113_);
v_a_2220_ = lean_ctor_get(v___x_2137_, 0);
v_isSharedCheck_2227_ = !lean_is_exclusive(v___x_2137_);
if (v_isSharedCheck_2227_ == 0)
{
v___x_2222_ = v___x_2137_;
v_isShared_2223_ = v_isSharedCheck_2227_;
goto v_resetjp_2221_;
}
else
{
lean_inc(v_a_2220_);
lean_dec(v___x_2137_);
v___x_2222_ = lean_box(0);
v_isShared_2223_ = v_isSharedCheck_2227_;
goto v_resetjp_2221_;
}
v_resetjp_2221_:
{
lean_object* v___x_2225_; 
if (v_isShared_2223_ == 0)
{
v___x_2225_ = v___x_2222_;
goto v_reusejp_2224_;
}
else
{
lean_object* v_reuseFailAlloc_2226_; 
v_reuseFailAlloc_2226_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2226_, 0, v_a_2220_);
v___x_2225_ = v_reuseFailAlloc_2226_;
goto v_reusejp_2224_;
}
v_reusejp_2224_:
{
return v___x_2225_;
}
}
}
}
else
{
lean_object* v_a_2228_; lean_object* v___x_2230_; uint8_t v_isShared_2231_; uint8_t v_isSharedCheck_2235_; 
lean_dec_ref_known(v___x_2130_, 2);
lean_dec(v_a_2127_);
lean_dec(v_a_2120_);
lean_dec_ref(v___y_2114_);
lean_dec_ref(v_t_2113_);
v_a_2228_ = lean_ctor_get(v___x_2134_, 0);
v_isSharedCheck_2235_ = !lean_is_exclusive(v___x_2134_);
if (v_isSharedCheck_2235_ == 0)
{
v___x_2230_ = v___x_2134_;
v_isShared_2231_ = v_isSharedCheck_2235_;
goto v_resetjp_2229_;
}
else
{
lean_inc(v_a_2228_);
lean_dec(v___x_2134_);
v___x_2230_ = lean_box(0);
v_isShared_2231_ = v_isSharedCheck_2235_;
goto v_resetjp_2229_;
}
v_resetjp_2229_:
{
lean_object* v___x_2233_; 
if (v_isShared_2231_ == 0)
{
v___x_2233_ = v___x_2230_;
goto v_reusejp_2232_;
}
else
{
lean_object* v_reuseFailAlloc_2234_; 
v_reuseFailAlloc_2234_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2234_, 0, v_a_2228_);
v___x_2233_ = v_reuseFailAlloc_2234_;
goto v_reusejp_2232_;
}
v_reusejp_2232_:
{
return v___x_2233_;
}
}
}
}
else
{
lean_object* v_a_2236_; lean_object* v___x_2238_; uint8_t v_isShared_2239_; uint8_t v_isSharedCheck_2243_; 
lean_dec(v_a_2120_);
lean_dec_ref(v___y_2114_);
lean_dec_ref(v_t_2113_);
v_a_2236_ = lean_ctor_get(v___x_2126_, 0);
v_isSharedCheck_2243_ = !lean_is_exclusive(v___x_2126_);
if (v_isSharedCheck_2243_ == 0)
{
v___x_2238_ = v___x_2126_;
v_isShared_2239_ = v_isSharedCheck_2243_;
goto v_resetjp_2237_;
}
else
{
lean_inc(v_a_2236_);
lean_dec(v___x_2126_);
v___x_2238_ = lean_box(0);
v_isShared_2239_ = v_isSharedCheck_2243_;
goto v_resetjp_2237_;
}
v_resetjp_2237_:
{
lean_object* v___x_2241_; 
if (v_isShared_2239_ == 0)
{
v___x_2241_ = v___x_2238_;
goto v_reusejp_2240_;
}
else
{
lean_object* v_reuseFailAlloc_2242_; 
v_reuseFailAlloc_2242_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2242_, 0, v_a_2236_);
v___x_2241_ = v_reuseFailAlloc_2242_;
goto v_reusejp_2240_;
}
v_reusejp_2240_:
{
return v___x_2241_;
}
}
}
}
else
{
lean_object* v_a_2244_; lean_object* v___x_2246_; uint8_t v_isShared_2247_; uint8_t v_isSharedCheck_2251_; 
lean_dec_ref(v___y_2114_);
lean_dec_ref(v_t_2113_);
v_a_2244_ = lean_ctor_get(v___x_2119_, 0);
v_isSharedCheck_2251_ = !lean_is_exclusive(v___x_2119_);
if (v_isSharedCheck_2251_ == 0)
{
v___x_2246_ = v___x_2119_;
v_isShared_2247_ = v_isSharedCheck_2251_;
goto v_resetjp_2245_;
}
else
{
lean_inc(v_a_2244_);
lean_dec(v___x_2119_);
v___x_2246_ = lean_box(0);
v_isShared_2247_ = v_isSharedCheck_2251_;
goto v_resetjp_2245_;
}
v_resetjp_2245_:
{
lean_object* v___x_2249_; 
if (v_isShared_2247_ == 0)
{
v___x_2249_ = v___x_2246_;
goto v_reusejp_2248_;
}
else
{
lean_object* v_reuseFailAlloc_2250_; 
v_reuseFailAlloc_2250_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2250_, 0, v_a_2244_);
v___x_2249_ = v_reuseFailAlloc_2250_;
goto v_reusejp_2248_;
}
v_reusejp_2248_:
{
return v___x_2249_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult___lam__3___boxed(lean_object* v_t_2252_, lean_object* v___y_2253_, lean_object* v___y_2254_, lean_object* v___y_2255_, lean_object* v___y_2256_, lean_object* v___y_2257_){
_start:
{
lean_object* v_res_2258_; 
v_res_2258_ = lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult___lam__3(v_t_2252_, v___y_2253_, v___y_2254_, v___y_2255_, v___y_2256_);
lean_dec(v___y_2256_);
lean_dec_ref(v___y_2255_);
lean_dec(v___y_2254_);
return v_res_2258_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult_spec__0_spec__0(lean_object* v_msgData_2259_, lean_object* v___y_2260_, lean_object* v___y_2261_, lean_object* v___y_2262_, lean_object* v___y_2263_){
_start:
{
lean_object* v___x_2265_; lean_object* v_env_2266_; lean_object* v___x_2267_; lean_object* v_mctx_2268_; lean_object* v_lctx_2269_; lean_object* v_options_2270_; lean_object* v___x_2271_; lean_object* v___x_2272_; lean_object* v___x_2273_; 
v___x_2265_ = lean_st_ref_get(v___y_2263_);
v_env_2266_ = lean_ctor_get(v___x_2265_, 0);
lean_inc_ref(v_env_2266_);
lean_dec(v___x_2265_);
v___x_2267_ = lean_st_ref_get(v___y_2261_);
v_mctx_2268_ = lean_ctor_get(v___x_2267_, 0);
lean_inc_ref(v_mctx_2268_);
lean_dec(v___x_2267_);
v_lctx_2269_ = lean_ctor_get(v___y_2260_, 2);
v_options_2270_ = lean_ctor_get(v___y_2262_, 2);
lean_inc_ref(v_options_2270_);
lean_inc_ref(v_lctx_2269_);
v___x_2271_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_2271_, 0, v_env_2266_);
lean_ctor_set(v___x_2271_, 1, v_mctx_2268_);
lean_ctor_set(v___x_2271_, 2, v_lctx_2269_);
lean_ctor_set(v___x_2271_, 3, v_options_2270_);
v___x_2272_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_2272_, 0, v___x_2271_);
lean_ctor_set(v___x_2272_, 1, v_msgData_2259_);
v___x_2273_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2273_, 0, v___x_2272_);
return v___x_2273_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult_spec__0_spec__0___boxed(lean_object* v_msgData_2274_, lean_object* v___y_2275_, lean_object* v___y_2276_, lean_object* v___y_2277_, lean_object* v___y_2278_, lean_object* v___y_2279_){
_start:
{
lean_object* v_res_2280_; 
v_res_2280_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult_spec__0_spec__0(v_msgData_2274_, v___y_2275_, v___y_2276_, v___y_2277_, v___y_2278_);
lean_dec(v___y_2278_);
lean_dec_ref(v___y_2277_);
lean_dec(v___y_2276_);
lean_dec_ref(v___y_2275_);
return v_res_2280_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult_spec__0___redArg(lean_object* v_msg_2281_, lean_object* v___y_2282_, lean_object* v___y_2283_, lean_object* v___y_2284_, lean_object* v___y_2285_){
_start:
{
lean_object* v_ref_2287_; lean_object* v___x_2288_; lean_object* v_a_2289_; lean_object* v___x_2291_; uint8_t v_isShared_2292_; uint8_t v_isSharedCheck_2297_; 
v_ref_2287_ = lean_ctor_get(v___y_2284_, 5);
v___x_2288_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult_spec__0_spec__0(v_msg_2281_, v___y_2282_, v___y_2283_, v___y_2284_, v___y_2285_);
v_a_2289_ = lean_ctor_get(v___x_2288_, 0);
v_isSharedCheck_2297_ = !lean_is_exclusive(v___x_2288_);
if (v_isSharedCheck_2297_ == 0)
{
v___x_2291_ = v___x_2288_;
v_isShared_2292_ = v_isSharedCheck_2297_;
goto v_resetjp_2290_;
}
else
{
lean_inc(v_a_2289_);
lean_dec(v___x_2288_);
v___x_2291_ = lean_box(0);
v_isShared_2292_ = v_isSharedCheck_2297_;
goto v_resetjp_2290_;
}
v_resetjp_2290_:
{
lean_object* v___x_2293_; lean_object* v___x_2295_; 
lean_inc(v_ref_2287_);
v___x_2293_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2293_, 0, v_ref_2287_);
lean_ctor_set(v___x_2293_, 1, v_a_2289_);
if (v_isShared_2292_ == 0)
{
lean_ctor_set_tag(v___x_2291_, 1);
lean_ctor_set(v___x_2291_, 0, v___x_2293_);
v___x_2295_ = v___x_2291_;
goto v_reusejp_2294_;
}
else
{
lean_object* v_reuseFailAlloc_2296_; 
v_reuseFailAlloc_2296_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2296_, 0, v___x_2293_);
v___x_2295_ = v_reuseFailAlloc_2296_;
goto v_reusejp_2294_;
}
v_reusejp_2294_:
{
return v___x_2295_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult_spec__0___redArg___boxed(lean_object* v_msg_2298_, lean_object* v___y_2299_, lean_object* v___y_2300_, lean_object* v___y_2301_, lean_object* v___y_2302_, lean_object* v___y_2303_){
_start:
{
lean_object* v_res_2304_; 
v_res_2304_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult_spec__0___redArg(v_msg_2298_, v___y_2299_, v___y_2300_, v___y_2301_, v___y_2302_);
lean_dec(v___y_2302_);
lean_dec_ref(v___y_2301_);
lean_dec(v___y_2300_);
lean_dec_ref(v___y_2299_);
return v_res_2304_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult(lean_object* v_decl_2314_, lean_object* v_type_2315_, lean_object* v_t_2316_, lean_object* v_a_2317_, lean_object* v_a_2318_, lean_object* v_a_2319_, lean_object* v_a_2320_){
_start:
{
uint8_t v___x_2322_; lean_object* v___x_2323_; lean_object* v___f_2324_; lean_object* v___x_2325_; 
v___x_2322_ = 0;
v___x_2323_ = lean_box(v___x_2322_);
lean_inc_ref(v_t_2316_);
v___f_2324_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult___lam__0___boxed), 7, 2);
lean_closure_set(v___f_2324_, 0, v_t_2316_);
lean_closure_set(v___f_2324_, 1, v___x_2323_);
v___x_2325_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Bound_isZero_spec__1___redArg(v___f_2324_, v___x_2322_, v_a_2317_, v_a_2318_, v_a_2319_, v_a_2320_);
if (lean_obj_tag(v___x_2325_) == 0)
{
lean_object* v_a_2326_; lean_object* v___x_2328_; uint8_t v_isShared_2329_; uint8_t v_isSharedCheck_2459_; 
v_a_2326_ = lean_ctor_get(v___x_2325_, 0);
v_isSharedCheck_2459_ = !lean_is_exclusive(v___x_2325_);
if (v_isSharedCheck_2459_ == 0)
{
v___x_2328_ = v___x_2325_;
v_isShared_2329_ = v_isSharedCheck_2459_;
goto v_resetjp_2327_;
}
else
{
lean_inc(v_a_2326_);
lean_dec(v___x_2325_);
v___x_2328_ = lean_box(0);
v_isShared_2329_ = v_isSharedCheck_2459_;
goto v_resetjp_2327_;
}
v_resetjp_2327_:
{
lean_object* v_snd_2330_; lean_object* v_snd_2331_; lean_object* v_snd_2332_; lean_object* v_snd_2333_; lean_object* v_snd_2334_; uint8_t v___x_2335_; 
v_snd_2330_ = lean_ctor_get(v_a_2326_, 1);
lean_inc(v_snd_2330_);
lean_dec(v_a_2326_);
v_snd_2331_ = lean_ctor_get(v_snd_2330_, 1);
lean_inc(v_snd_2331_);
lean_dec(v_snd_2330_);
v_snd_2332_ = lean_ctor_get(v_snd_2331_, 1);
lean_inc(v_snd_2332_);
lean_dec(v_snd_2331_);
v_snd_2333_ = lean_ctor_get(v_snd_2332_, 1);
lean_inc(v_snd_2333_);
lean_dec(v_snd_2332_);
v_snd_2334_ = lean_ctor_get(v_snd_2333_, 1);
lean_inc(v_snd_2334_);
lean_dec(v_snd_2333_);
v___x_2335_ = lean_unbox(v_snd_2334_);
lean_dec(v_snd_2334_);
if (v___x_2335_ == 0)
{
lean_object* v___x_2336_; lean_object* v___f_2337_; lean_object* v___x_2338_; 
lean_del_object(v___x_2328_);
v___x_2336_ = lean_box(v___x_2322_);
lean_inc_ref(v_t_2316_);
v___f_2337_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult___lam__1___boxed), 7, 2);
lean_closure_set(v___f_2337_, 0, v_t_2316_);
lean_closure_set(v___f_2337_, 1, v___x_2336_);
v___x_2338_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Bound_isZero_spec__1___redArg(v___f_2337_, v___x_2322_, v_a_2317_, v_a_2318_, v_a_2319_, v_a_2320_);
if (lean_obj_tag(v___x_2338_) == 0)
{
lean_object* v_a_2339_; lean_object* v___x_2341_; uint8_t v_isShared_2342_; uint8_t v_isSharedCheck_2446_; 
v_a_2339_ = lean_ctor_get(v___x_2338_, 0);
v_isSharedCheck_2446_ = !lean_is_exclusive(v___x_2338_);
if (v_isSharedCheck_2446_ == 0)
{
v___x_2341_ = v___x_2338_;
v_isShared_2342_ = v_isSharedCheck_2446_;
goto v_resetjp_2340_;
}
else
{
lean_inc(v_a_2339_);
lean_dec(v___x_2338_);
v___x_2341_ = lean_box(0);
v_isShared_2342_ = v_isSharedCheck_2446_;
goto v_resetjp_2340_;
}
v_resetjp_2340_:
{
lean_object* v_snd_2343_; lean_object* v_snd_2344_; lean_object* v_snd_2345_; lean_object* v_snd_2346_; lean_object* v_snd_2347_; uint8_t v___x_2348_; 
v_snd_2343_ = lean_ctor_get(v_a_2339_, 1);
lean_inc(v_snd_2343_);
lean_dec(v_a_2339_);
v_snd_2344_ = lean_ctor_get(v_snd_2343_, 1);
lean_inc(v_snd_2344_);
lean_dec(v_snd_2343_);
v_snd_2345_ = lean_ctor_get(v_snd_2344_, 1);
lean_inc(v_snd_2345_);
lean_dec(v_snd_2344_);
v_snd_2346_ = lean_ctor_get(v_snd_2345_, 1);
lean_inc(v_snd_2346_);
lean_dec(v_snd_2345_);
v_snd_2347_ = lean_ctor_get(v_snd_2346_, 1);
lean_inc(v_snd_2347_);
lean_dec(v_snd_2346_);
v___x_2348_ = lean_unbox(v_snd_2347_);
lean_dec(v_snd_2347_);
if (v___x_2348_ == 0)
{
lean_object* v___x_2349_; lean_object* v___f_2350_; lean_object* v___x_2351_; 
lean_del_object(v___x_2341_);
v___x_2349_ = lean_box(v___x_2322_);
lean_inc_ref(v_t_2316_);
v___f_2350_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult___lam__2___boxed), 7, 2);
lean_closure_set(v___f_2350_, 0, v_t_2316_);
lean_closure_set(v___f_2350_, 1, v___x_2349_);
v___x_2351_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Bound_isZero_spec__1___redArg(v___f_2350_, v___x_2322_, v_a_2317_, v_a_2318_, v_a_2319_, v_a_2320_);
if (lean_obj_tag(v___x_2351_) == 0)
{
lean_object* v_a_2352_; lean_object* v___x_2354_; uint8_t v_isShared_2355_; uint8_t v_isSharedCheck_2433_; 
v_a_2352_ = lean_ctor_get(v___x_2351_, 0);
v_isSharedCheck_2433_ = !lean_is_exclusive(v___x_2351_);
if (v_isSharedCheck_2433_ == 0)
{
v___x_2354_ = v___x_2351_;
v_isShared_2355_ = v_isSharedCheck_2433_;
goto v_resetjp_2353_;
}
else
{
lean_inc(v_a_2352_);
lean_dec(v___x_2351_);
v___x_2354_ = lean_box(0);
v_isShared_2355_ = v_isSharedCheck_2433_;
goto v_resetjp_2353_;
}
v_resetjp_2353_:
{
lean_object* v_snd_2356_; lean_object* v_snd_2357_; lean_object* v_snd_2358_; lean_object* v_snd_2359_; lean_object* v_snd_2360_; uint8_t v___x_2361_; 
v_snd_2356_ = lean_ctor_get(v_a_2352_, 1);
lean_inc(v_snd_2356_);
lean_dec(v_a_2352_);
v_snd_2357_ = lean_ctor_get(v_snd_2356_, 1);
lean_inc(v_snd_2357_);
lean_dec(v_snd_2356_);
v_snd_2358_ = lean_ctor_get(v_snd_2357_, 1);
lean_inc(v_snd_2358_);
lean_dec(v_snd_2357_);
v_snd_2359_ = lean_ctor_get(v_snd_2358_, 1);
lean_inc(v_snd_2359_);
lean_dec(v_snd_2358_);
v_snd_2360_ = lean_ctor_get(v_snd_2359_, 1);
lean_inc(v_snd_2360_);
lean_dec(v_snd_2359_);
v___x_2361_ = lean_unbox(v_snd_2360_);
lean_dec(v_snd_2360_);
if (v___x_2361_ == 0)
{
lean_object* v___f_2362_; lean_object* v___x_2363_; 
lean_del_object(v___x_2354_);
v___f_2362_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult___lam__3___boxed), 6, 1);
lean_closure_set(v___f_2362_, 0, v_t_2316_);
v___x_2363_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Bound_isZero_spec__1___redArg(v___f_2362_, v___x_2322_, v_a_2317_, v_a_2318_, v_a_2319_, v_a_2320_);
if (lean_obj_tag(v___x_2363_) == 0)
{
lean_object* v_a_2364_; lean_object* v___x_2366_; uint8_t v_isShared_2367_; uint8_t v_isSharedCheck_2420_; 
v_a_2364_ = lean_ctor_get(v___x_2363_, 0);
v_isSharedCheck_2420_ = !lean_is_exclusive(v___x_2363_);
if (v_isSharedCheck_2420_ == 0)
{
v___x_2366_ = v___x_2363_;
v_isShared_2367_ = v_isSharedCheck_2420_;
goto v_resetjp_2365_;
}
else
{
lean_inc(v_a_2364_);
lean_dec(v___x_2363_);
v___x_2366_ = lean_box(0);
v_isShared_2367_ = v_isSharedCheck_2420_;
goto v_resetjp_2365_;
}
v_resetjp_2365_:
{
lean_object* v_snd_2368_; lean_object* v_snd_2369_; lean_object* v___x_2371_; uint8_t v_isShared_2372_; uint8_t v_isSharedCheck_2418_; 
v_snd_2368_ = lean_ctor_get(v_a_2364_, 1);
lean_inc(v_snd_2368_);
lean_dec(v_a_2364_);
v_snd_2369_ = lean_ctor_get(v_snd_2368_, 1);
v_isSharedCheck_2418_ = !lean_is_exclusive(v_snd_2368_);
if (v_isSharedCheck_2418_ == 0)
{
lean_object* v_unused_2419_; 
v_unused_2419_ = lean_ctor_get(v_snd_2368_, 0);
lean_dec(v_unused_2419_);
v___x_2371_ = v_snd_2368_;
v_isShared_2372_ = v_isSharedCheck_2418_;
goto v_resetjp_2370_;
}
else
{
lean_inc(v_snd_2369_);
lean_dec(v_snd_2368_);
v___x_2371_ = lean_box(0);
v_isShared_2372_ = v_isSharedCheck_2418_;
goto v_resetjp_2370_;
}
v_resetjp_2370_:
{
lean_object* v_snd_2373_; lean_object* v___x_2375_; uint8_t v_isShared_2376_; uint8_t v_isSharedCheck_2416_; 
v_snd_2373_ = lean_ctor_get(v_snd_2369_, 1);
v_isSharedCheck_2416_ = !lean_is_exclusive(v_snd_2369_);
if (v_isSharedCheck_2416_ == 0)
{
lean_object* v_unused_2417_; 
v_unused_2417_ = lean_ctor_get(v_snd_2369_, 0);
lean_dec(v_unused_2417_);
v___x_2375_ = v_snd_2369_;
v_isShared_2376_ = v_isSharedCheck_2416_;
goto v_resetjp_2374_;
}
else
{
lean_inc(v_snd_2373_);
lean_dec(v_snd_2369_);
v___x_2375_ = lean_box(0);
v_isShared_2376_ = v_isSharedCheck_2416_;
goto v_resetjp_2374_;
}
v_resetjp_2374_:
{
lean_object* v_snd_2377_; lean_object* v___x_2379_; uint8_t v_isShared_2380_; uint8_t v_isSharedCheck_2414_; 
v_snd_2377_ = lean_ctor_get(v_snd_2373_, 1);
v_isSharedCheck_2414_ = !lean_is_exclusive(v_snd_2373_);
if (v_isSharedCheck_2414_ == 0)
{
lean_object* v_unused_2415_; 
v_unused_2415_ = lean_ctor_get(v_snd_2373_, 0);
lean_dec(v_unused_2415_);
v___x_2379_ = v_snd_2373_;
v_isShared_2380_ = v_isSharedCheck_2414_;
goto v_resetjp_2378_;
}
else
{
lean_inc(v_snd_2377_);
lean_dec(v_snd_2373_);
v___x_2379_ = lean_box(0);
v_isShared_2380_ = v_isSharedCheck_2414_;
goto v_resetjp_2378_;
}
v_resetjp_2378_:
{
lean_object* v_snd_2381_; lean_object* v___x_2383_; uint8_t v_isShared_2384_; uint8_t v_isSharedCheck_2412_; 
v_snd_2381_ = lean_ctor_get(v_snd_2377_, 1);
v_isSharedCheck_2412_ = !lean_is_exclusive(v_snd_2377_);
if (v_isSharedCheck_2412_ == 0)
{
lean_object* v_unused_2413_; 
v_unused_2413_ = lean_ctor_get(v_snd_2377_, 0);
lean_dec(v_unused_2413_);
v___x_2383_ = v_snd_2377_;
v_isShared_2384_ = v_isSharedCheck_2412_;
goto v_resetjp_2382_;
}
else
{
lean_inc(v_snd_2381_);
lean_dec(v_snd_2377_);
v___x_2383_ = lean_box(0);
v_isShared_2384_ = v_isSharedCheck_2412_;
goto v_resetjp_2382_;
}
v_resetjp_2382_:
{
uint8_t v___x_2385_; 
v___x_2385_ = lean_unbox(v_snd_2381_);
lean_dec(v_snd_2381_);
if (v___x_2385_ == 0)
{
lean_object* v___x_2386_; uint8_t v___x_2387_; lean_object* v___x_2388_; lean_object* v___x_2389_; lean_object* v___x_2391_; 
lean_del_object(v___x_2366_);
v___x_2386_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult___closed__1));
v___x_2387_ = 1;
v___x_2388_ = l_Lean_Name_toString(v_decl_2314_, v___x_2387_);
v___x_2389_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2389_, 0, v___x_2388_);
if (v_isShared_2384_ == 0)
{
lean_ctor_set_tag(v___x_2383_, 5);
lean_ctor_set(v___x_2383_, 1, v___x_2389_);
lean_ctor_set(v___x_2383_, 0, v___x_2386_);
v___x_2391_ = v___x_2383_;
goto v_reusejp_2390_;
}
else
{
lean_object* v_reuseFailAlloc_2407_; 
v_reuseFailAlloc_2407_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2407_, 0, v___x_2386_);
lean_ctor_set(v_reuseFailAlloc_2407_, 1, v___x_2389_);
v___x_2391_ = v_reuseFailAlloc_2407_;
goto v_reusejp_2390_;
}
v_reusejp_2390_:
{
lean_object* v___x_2392_; lean_object* v___x_2394_; 
v___x_2392_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult___closed__3));
if (v_isShared_2380_ == 0)
{
lean_ctor_set_tag(v___x_2379_, 5);
lean_ctor_set(v___x_2379_, 1, v___x_2392_);
lean_ctor_set(v___x_2379_, 0, v___x_2391_);
v___x_2394_ = v___x_2379_;
goto v_reusejp_2393_;
}
else
{
lean_object* v_reuseFailAlloc_2406_; 
v_reuseFailAlloc_2406_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2406_, 0, v___x_2391_);
lean_ctor_set(v_reuseFailAlloc_2406_, 1, v___x_2392_);
v___x_2394_ = v_reuseFailAlloc_2406_;
goto v_reusejp_2393_;
}
v_reusejp_2393_:
{
lean_object* v___x_2395_; lean_object* v___x_2396_; lean_object* v___x_2398_; 
v___x_2395_ = lean_expr_dbg_to_string(v_type_2315_);
v___x_2396_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2396_, 0, v___x_2395_);
if (v_isShared_2376_ == 0)
{
lean_ctor_set_tag(v___x_2375_, 5);
lean_ctor_set(v___x_2375_, 1, v___x_2396_);
lean_ctor_set(v___x_2375_, 0, v___x_2394_);
v___x_2398_ = v___x_2375_;
goto v_reusejp_2397_;
}
else
{
lean_object* v_reuseFailAlloc_2405_; 
v_reuseFailAlloc_2405_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2405_, 0, v___x_2394_);
lean_ctor_set(v_reuseFailAlloc_2405_, 1, v___x_2396_);
v___x_2398_ = v_reuseFailAlloc_2405_;
goto v_reusejp_2397_;
}
v_reusejp_2397_:
{
lean_object* v___x_2399_; lean_object* v___x_2401_; 
v___x_2399_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult___closed__5));
if (v_isShared_2372_ == 0)
{
lean_ctor_set_tag(v___x_2371_, 5);
lean_ctor_set(v___x_2371_, 1, v___x_2399_);
lean_ctor_set(v___x_2371_, 0, v___x_2398_);
v___x_2401_ = v___x_2371_;
goto v_reusejp_2400_;
}
else
{
lean_object* v_reuseFailAlloc_2404_; 
v_reuseFailAlloc_2404_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2404_, 0, v___x_2398_);
lean_ctor_set(v_reuseFailAlloc_2404_, 1, v___x_2399_);
v___x_2401_ = v_reuseFailAlloc_2404_;
goto v_reusejp_2400_;
}
v_reusejp_2400_:
{
lean_object* v___x_2402_; lean_object* v___x_2403_; 
v___x_2402_ = l_Lean_MessageData_ofFormat(v___x_2401_);
v___x_2403_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult_spec__0___redArg(v___x_2402_, v_a_2317_, v_a_2318_, v_a_2319_, v_a_2320_);
return v___x_2403_;
}
}
}
}
}
else
{
lean_object* v___x_2408_; lean_object* v___x_2410_; 
lean_del_object(v___x_2383_);
lean_del_object(v___x_2379_);
lean_del_object(v___x_2375_);
lean_del_object(v___x_2371_);
lean_dec(v_decl_2314_);
v___x_2408_ = lean_box(0);
if (v_isShared_2367_ == 0)
{
lean_ctor_set(v___x_2366_, 0, v___x_2408_);
v___x_2410_ = v___x_2366_;
goto v_reusejp_2409_;
}
else
{
lean_object* v_reuseFailAlloc_2411_; 
v_reuseFailAlloc_2411_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2411_, 0, v___x_2408_);
v___x_2410_ = v_reuseFailAlloc_2411_;
goto v_reusejp_2409_;
}
v_reusejp_2409_:
{
return v___x_2410_;
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
lean_object* v_a_2421_; lean_object* v___x_2423_; uint8_t v_isShared_2424_; uint8_t v_isSharedCheck_2428_; 
lean_dec(v_decl_2314_);
v_a_2421_ = lean_ctor_get(v___x_2363_, 0);
v_isSharedCheck_2428_ = !lean_is_exclusive(v___x_2363_);
if (v_isSharedCheck_2428_ == 0)
{
v___x_2423_ = v___x_2363_;
v_isShared_2424_ = v_isSharedCheck_2428_;
goto v_resetjp_2422_;
}
else
{
lean_inc(v_a_2421_);
lean_dec(v___x_2363_);
v___x_2423_ = lean_box(0);
v_isShared_2424_ = v_isSharedCheck_2428_;
goto v_resetjp_2422_;
}
v_resetjp_2422_:
{
lean_object* v___x_2426_; 
if (v_isShared_2424_ == 0)
{
v___x_2426_ = v___x_2423_;
goto v_reusejp_2425_;
}
else
{
lean_object* v_reuseFailAlloc_2427_; 
v_reuseFailAlloc_2427_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2427_, 0, v_a_2421_);
v___x_2426_ = v_reuseFailAlloc_2427_;
goto v_reusejp_2425_;
}
v_reusejp_2425_:
{
return v___x_2426_;
}
}
}
}
else
{
lean_object* v___x_2429_; lean_object* v___x_2431_; 
lean_dec_ref(v_t_2316_);
lean_dec(v_decl_2314_);
v___x_2429_ = lean_box(0);
if (v_isShared_2355_ == 0)
{
lean_ctor_set(v___x_2354_, 0, v___x_2429_);
v___x_2431_ = v___x_2354_;
goto v_reusejp_2430_;
}
else
{
lean_object* v_reuseFailAlloc_2432_; 
v_reuseFailAlloc_2432_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2432_, 0, v___x_2429_);
v___x_2431_ = v_reuseFailAlloc_2432_;
goto v_reusejp_2430_;
}
v_reusejp_2430_:
{
return v___x_2431_;
}
}
}
}
else
{
lean_object* v_a_2434_; lean_object* v___x_2436_; uint8_t v_isShared_2437_; uint8_t v_isSharedCheck_2441_; 
lean_dec_ref(v_t_2316_);
lean_dec(v_decl_2314_);
v_a_2434_ = lean_ctor_get(v___x_2351_, 0);
v_isSharedCheck_2441_ = !lean_is_exclusive(v___x_2351_);
if (v_isSharedCheck_2441_ == 0)
{
v___x_2436_ = v___x_2351_;
v_isShared_2437_ = v_isSharedCheck_2441_;
goto v_resetjp_2435_;
}
else
{
lean_inc(v_a_2434_);
lean_dec(v___x_2351_);
v___x_2436_ = lean_box(0);
v_isShared_2437_ = v_isSharedCheck_2441_;
goto v_resetjp_2435_;
}
v_resetjp_2435_:
{
lean_object* v___x_2439_; 
if (v_isShared_2437_ == 0)
{
v___x_2439_ = v___x_2436_;
goto v_reusejp_2438_;
}
else
{
lean_object* v_reuseFailAlloc_2440_; 
v_reuseFailAlloc_2440_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2440_, 0, v_a_2434_);
v___x_2439_ = v_reuseFailAlloc_2440_;
goto v_reusejp_2438_;
}
v_reusejp_2438_:
{
return v___x_2439_;
}
}
}
}
else
{
lean_object* v___x_2442_; lean_object* v___x_2444_; 
lean_dec_ref(v_t_2316_);
lean_dec(v_decl_2314_);
v___x_2442_ = lean_box(0);
if (v_isShared_2342_ == 0)
{
lean_ctor_set(v___x_2341_, 0, v___x_2442_);
v___x_2444_ = v___x_2341_;
goto v_reusejp_2443_;
}
else
{
lean_object* v_reuseFailAlloc_2445_; 
v_reuseFailAlloc_2445_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2445_, 0, v___x_2442_);
v___x_2444_ = v_reuseFailAlloc_2445_;
goto v_reusejp_2443_;
}
v_reusejp_2443_:
{
return v___x_2444_;
}
}
}
}
else
{
lean_object* v_a_2447_; lean_object* v___x_2449_; uint8_t v_isShared_2450_; uint8_t v_isSharedCheck_2454_; 
lean_dec_ref(v_t_2316_);
lean_dec(v_decl_2314_);
v_a_2447_ = lean_ctor_get(v___x_2338_, 0);
v_isSharedCheck_2454_ = !lean_is_exclusive(v___x_2338_);
if (v_isSharedCheck_2454_ == 0)
{
v___x_2449_ = v___x_2338_;
v_isShared_2450_ = v_isSharedCheck_2454_;
goto v_resetjp_2448_;
}
else
{
lean_inc(v_a_2447_);
lean_dec(v___x_2338_);
v___x_2449_ = lean_box(0);
v_isShared_2450_ = v_isSharedCheck_2454_;
goto v_resetjp_2448_;
}
v_resetjp_2448_:
{
lean_object* v___x_2452_; 
if (v_isShared_2450_ == 0)
{
v___x_2452_ = v___x_2449_;
goto v_reusejp_2451_;
}
else
{
lean_object* v_reuseFailAlloc_2453_; 
v_reuseFailAlloc_2453_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2453_, 0, v_a_2447_);
v___x_2452_ = v_reuseFailAlloc_2453_;
goto v_reusejp_2451_;
}
v_reusejp_2451_:
{
return v___x_2452_;
}
}
}
}
else
{
lean_object* v___x_2455_; lean_object* v___x_2457_; 
lean_dec_ref(v_t_2316_);
lean_dec(v_decl_2314_);
v___x_2455_ = lean_box(0);
if (v_isShared_2329_ == 0)
{
lean_ctor_set(v___x_2328_, 0, v___x_2455_);
v___x_2457_ = v___x_2328_;
goto v_reusejp_2456_;
}
else
{
lean_object* v_reuseFailAlloc_2458_; 
v_reuseFailAlloc_2458_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2458_, 0, v___x_2455_);
v___x_2457_ = v_reuseFailAlloc_2458_;
goto v_reusejp_2456_;
}
v_reusejp_2456_:
{
return v___x_2457_;
}
}
}
}
else
{
lean_object* v_a_2460_; lean_object* v___x_2462_; uint8_t v_isShared_2463_; uint8_t v_isSharedCheck_2467_; 
lean_dec_ref(v_t_2316_);
lean_dec(v_decl_2314_);
v_a_2460_ = lean_ctor_get(v___x_2325_, 0);
v_isSharedCheck_2467_ = !lean_is_exclusive(v___x_2325_);
if (v_isSharedCheck_2467_ == 0)
{
v___x_2462_ = v___x_2325_;
v_isShared_2463_ = v_isSharedCheck_2467_;
goto v_resetjp_2461_;
}
else
{
lean_inc(v_a_2460_);
lean_dec(v___x_2325_);
v___x_2462_ = lean_box(0);
v_isShared_2463_ = v_isSharedCheck_2467_;
goto v_resetjp_2461_;
}
v_resetjp_2461_:
{
lean_object* v___x_2465_; 
if (v_isShared_2463_ == 0)
{
v___x_2465_ = v___x_2462_;
goto v_reusejp_2464_;
}
else
{
lean_object* v_reuseFailAlloc_2466_; 
v_reuseFailAlloc_2466_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2466_, 0, v_a_2460_);
v___x_2465_ = v_reuseFailAlloc_2466_;
goto v_reusejp_2464_;
}
v_reusejp_2464_:
{
return v___x_2465_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult___boxed(lean_object* v_decl_2468_, lean_object* v_type_2469_, lean_object* v_t_2470_, lean_object* v_a_2471_, lean_object* v_a_2472_, lean_object* v_a_2473_, lean_object* v_a_2474_, lean_object* v_a_2475_){
_start:
{
lean_object* v_res_2476_; 
v_res_2476_ = lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult(v_decl_2468_, v_type_2469_, v_t_2470_, v_a_2471_, v_a_2472_, v_a_2473_, v_a_2474_);
lean_dec(v_a_2474_);
lean_dec_ref(v_a_2473_);
lean_dec(v_a_2472_);
lean_dec_ref(v_a_2471_);
lean_dec_ref(v_type_2469_);
return v_res_2476_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult_spec__0(lean_object* v_00_u03b1_2477_, lean_object* v_msg_2478_, lean_object* v___y_2479_, lean_object* v___y_2480_, lean_object* v___y_2481_, lean_object* v___y_2482_){
_start:
{
lean_object* v___x_2484_; 
v___x_2484_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult_spec__0___redArg(v_msg_2478_, v___y_2479_, v___y_2480_, v___y_2481_, v___y_2482_);
return v___x_2484_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult_spec__0___boxed(lean_object* v_00_u03b1_2485_, lean_object* v_msg_2486_, lean_object* v___y_2487_, lean_object* v___y_2488_, lean_object* v___y_2489_, lean_object* v___y_2490_, lean_object* v___y_2491_){
_start:
{
lean_object* v_res_2492_; 
v_res_2492_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult_spec__0(v_00_u03b1_2485_, v_msg_2486_, v___y_2487_, v___y_2488_, v___y_2489_, v___y_2490_);
lean_dec(v___y_2490_);
lean_dec_ref(v___y_2489_);
lean_dec(v___y_2488_);
lean_dec_ref(v___y_2487_);
return v_res_2492_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00Mathlib_Tactic_Bound_typePriority_spec__1___redArg___lam__0(lean_object* v_k_2493_, lean_object* v_b_2494_, lean_object* v_c_2495_, lean_object* v___y_2496_, lean_object* v___y_2497_, lean_object* v___y_2498_, lean_object* v___y_2499_){
_start:
{
lean_object* v___x_2501_; 
lean_inc(v___y_2499_);
lean_inc_ref(v___y_2498_);
lean_inc(v___y_2497_);
lean_inc_ref(v___y_2496_);
v___x_2501_ = lean_apply_7(v_k_2493_, v_b_2494_, v_c_2495_, v___y_2496_, v___y_2497_, v___y_2498_, v___y_2499_, lean_box(0));
return v___x_2501_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00Mathlib_Tactic_Bound_typePriority_spec__1___redArg___lam__0___boxed(lean_object* v_k_2502_, lean_object* v_b_2503_, lean_object* v_c_2504_, lean_object* v___y_2505_, lean_object* v___y_2506_, lean_object* v___y_2507_, lean_object* v___y_2508_, lean_object* v___y_2509_){
_start:
{
lean_object* v_res_2510_; 
v_res_2510_ = lp_mathlib_Lean_Meta_forallTelescope___at___00Mathlib_Tactic_Bound_typePriority_spec__1___redArg___lam__0(v_k_2502_, v_b_2503_, v_c_2504_, v___y_2505_, v___y_2506_, v___y_2507_, v___y_2508_);
lean_dec(v___y_2508_);
lean_dec_ref(v___y_2507_);
lean_dec(v___y_2506_);
lean_dec_ref(v___y_2505_);
return v_res_2510_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00Mathlib_Tactic_Bound_typePriority_spec__1___redArg(lean_object* v_type_2511_, lean_object* v_k_2512_, uint8_t v_cleanupAnnotations_2513_, lean_object* v___y_2514_, lean_object* v___y_2515_, lean_object* v___y_2516_, lean_object* v___y_2517_){
_start:
{
lean_object* v___f_2519_; uint8_t v___x_2520_; lean_object* v___x_2521_; lean_object* v___x_2522_; 
v___f_2519_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_forallTelescope___at___00Mathlib_Tactic_Bound_typePriority_spec__1___redArg___lam__0___boxed), 8, 1);
lean_closure_set(v___f_2519_, 0, v_k_2512_);
v___x_2520_ = 0;
v___x_2521_ = lean_box(0);
v___x_2522_ = l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingAuxAux(lean_box(0), v___x_2520_, v___x_2521_, v_type_2511_, v___f_2519_, v_cleanupAnnotations_2513_, v___x_2520_, v___y_2514_, v___y_2515_, v___y_2516_, v___y_2517_);
if (lean_obj_tag(v___x_2522_) == 0)
{
lean_object* v_a_2523_; lean_object* v___x_2525_; uint8_t v_isShared_2526_; uint8_t v_isSharedCheck_2530_; 
v_a_2523_ = lean_ctor_get(v___x_2522_, 0);
v_isSharedCheck_2530_ = !lean_is_exclusive(v___x_2522_);
if (v_isSharedCheck_2530_ == 0)
{
v___x_2525_ = v___x_2522_;
v_isShared_2526_ = v_isSharedCheck_2530_;
goto v_resetjp_2524_;
}
else
{
lean_inc(v_a_2523_);
lean_dec(v___x_2522_);
v___x_2525_ = lean_box(0);
v_isShared_2526_ = v_isSharedCheck_2530_;
goto v_resetjp_2524_;
}
v_resetjp_2524_:
{
lean_object* v___x_2528_; 
if (v_isShared_2526_ == 0)
{
v___x_2528_ = v___x_2525_;
goto v_reusejp_2527_;
}
else
{
lean_object* v_reuseFailAlloc_2529_; 
v_reuseFailAlloc_2529_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2529_, 0, v_a_2523_);
v___x_2528_ = v_reuseFailAlloc_2529_;
goto v_reusejp_2527_;
}
v_reusejp_2527_:
{
return v___x_2528_;
}
}
}
else
{
lean_object* v_a_2531_; lean_object* v___x_2533_; uint8_t v_isShared_2534_; uint8_t v_isSharedCheck_2538_; 
v_a_2531_ = lean_ctor_get(v___x_2522_, 0);
v_isSharedCheck_2538_ = !lean_is_exclusive(v___x_2522_);
if (v_isSharedCheck_2538_ == 0)
{
v___x_2533_ = v___x_2522_;
v_isShared_2534_ = v_isSharedCheck_2538_;
goto v_resetjp_2532_;
}
else
{
lean_inc(v_a_2531_);
lean_dec(v___x_2522_);
v___x_2533_ = lean_box(0);
v_isShared_2534_ = v_isSharedCheck_2538_;
goto v_resetjp_2532_;
}
v_resetjp_2532_:
{
lean_object* v___x_2536_; 
if (v_isShared_2534_ == 0)
{
v___x_2536_ = v___x_2533_;
goto v_reusejp_2535_;
}
else
{
lean_object* v_reuseFailAlloc_2537_; 
v_reuseFailAlloc_2537_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2537_, 0, v_a_2531_);
v___x_2536_ = v_reuseFailAlloc_2537_;
goto v_reusejp_2535_;
}
v_reusejp_2535_:
{
return v___x_2536_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00Mathlib_Tactic_Bound_typePriority_spec__1___redArg___boxed(lean_object* v_type_2539_, lean_object* v_k_2540_, lean_object* v_cleanupAnnotations_2541_, lean_object* v___y_2542_, lean_object* v___y_2543_, lean_object* v___y_2544_, lean_object* v___y_2545_, lean_object* v___y_2546_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_2547_; lean_object* v_res_2548_; 
v_cleanupAnnotations_boxed_2547_ = lean_unbox(v_cleanupAnnotations_2541_);
v_res_2548_ = lp_mathlib_Lean_Meta_forallTelescope___at___00Mathlib_Tactic_Bound_typePriority_spec__1___redArg(v_type_2539_, v_k_2540_, v_cleanupAnnotations_boxed_2547_, v___y_2542_, v___y_2543_, v___y_2544_, v___y_2545_);
lean_dec(v___y_2545_);
lean_dec_ref(v___y_2544_);
lean_dec(v___y_2543_);
lean_dec_ref(v___y_2542_);
return v_res_2548_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00Mathlib_Tactic_Bound_typePriority_spec__1(lean_object* v_00_u03b1_2549_, lean_object* v_type_2550_, lean_object* v_k_2551_, uint8_t v_cleanupAnnotations_2552_, lean_object* v___y_2553_, lean_object* v___y_2554_, lean_object* v___y_2555_, lean_object* v___y_2556_){
_start:
{
lean_object* v___x_2558_; 
v___x_2558_ = lp_mathlib_Lean_Meta_forallTelescope___at___00Mathlib_Tactic_Bound_typePriority_spec__1___redArg(v_type_2550_, v_k_2551_, v_cleanupAnnotations_2552_, v___y_2553_, v___y_2554_, v___y_2555_, v___y_2556_);
return v___x_2558_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00Mathlib_Tactic_Bound_typePriority_spec__1___boxed(lean_object* v_00_u03b1_2559_, lean_object* v_type_2560_, lean_object* v_k_2561_, lean_object* v_cleanupAnnotations_2562_, lean_object* v___y_2563_, lean_object* v___y_2564_, lean_object* v___y_2565_, lean_object* v___y_2566_, lean_object* v___y_2567_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_2568_; lean_object* v_res_2569_; 
v_cleanupAnnotations_boxed_2568_ = lean_unbox(v_cleanupAnnotations_2562_);
v_res_2569_ = lp_mathlib_Lean_Meta_forallTelescope___at___00Mathlib_Tactic_Bound_typePriority_spec__1(v_00_u03b1_2559_, v_type_2560_, v_k_2561_, v_cleanupAnnotations_boxed_2568_, v___y_2563_, v___y_2564_, v___y_2565_, v___y_2566_);
lean_dec(v___y_2566_);
lean_dec_ref(v___y_2565_);
lean_dec(v___y_2564_);
lean_dec_ref(v___y_2563_);
return v_res_2569_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Bound_typePriority_spec__0(lean_object* v_as_2570_, size_t v_i_2571_, size_t v_stop_2572_, lean_object* v_b_2573_, lean_object* v___y_2574_, lean_object* v___y_2575_, lean_object* v___y_2576_, lean_object* v___y_2577_){
_start:
{
lean_object* v_a_2580_; uint8_t v___x_2584_; 
v___x_2584_ = lean_usize_dec_eq(v_i_2571_, v_stop_2572_);
if (v___x_2584_ == 0)
{
lean_object* v___x_2585_; lean_object* v___x_2586_; 
v___x_2585_ = lean_array_uget_borrowed(v_as_2570_, v_i_2571_);
lean_inc(v___x_2585_);
v___x_2586_ = lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_argPriority(v___x_2585_, v___y_2574_, v___y_2575_, v___y_2576_, v___y_2577_);
if (lean_obj_tag(v___x_2586_) == 0)
{
lean_object* v_a_2587_; lean_object* v___x_2588_; 
v_a_2587_ = lean_ctor_get(v___x_2586_, 0);
lean_inc(v_a_2587_);
lean_dec_ref_known(v___x_2586_, 1);
v___x_2588_ = lean_nat_add(v_b_2573_, v_a_2587_);
lean_dec(v_a_2587_);
lean_dec(v_b_2573_);
v_a_2580_ = v___x_2588_;
goto v___jp_2579_;
}
else
{
lean_dec(v_b_2573_);
if (lean_obj_tag(v___x_2586_) == 0)
{
lean_object* v_a_2589_; 
v_a_2589_ = lean_ctor_get(v___x_2586_, 0);
lean_inc(v_a_2589_);
lean_dec_ref_known(v___x_2586_, 1);
v_a_2580_ = v_a_2589_;
goto v___jp_2579_;
}
else
{
return v___x_2586_;
}
}
}
else
{
lean_object* v___x_2590_; 
v___x_2590_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2590_, 0, v_b_2573_);
return v___x_2590_;
}
v___jp_2579_:
{
size_t v___x_2581_; size_t v___x_2582_; 
v___x_2581_ = ((size_t)1ULL);
v___x_2582_ = lean_usize_add(v_i_2571_, v___x_2581_);
v_i_2571_ = v___x_2582_;
v_b_2573_ = v_a_2580_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Bound_typePriority_spec__0___boxed(lean_object* v_as_2591_, lean_object* v_i_2592_, lean_object* v_stop_2593_, lean_object* v_b_2594_, lean_object* v___y_2595_, lean_object* v___y_2596_, lean_object* v___y_2597_, lean_object* v___y_2598_, lean_object* v___y_2599_){
_start:
{
size_t v_i_boxed_2600_; size_t v_stop_boxed_2601_; lean_object* v_res_2602_; 
v_i_boxed_2600_ = lean_unbox_usize(v_i_2592_);
lean_dec(v_i_2592_);
v_stop_boxed_2601_ = lean_unbox_usize(v_stop_2593_);
lean_dec(v_stop_2593_);
v_res_2602_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Bound_typePriority_spec__0(v_as_2591_, v_i_boxed_2600_, v_stop_boxed_2601_, v_b_2594_, v___y_2595_, v___y_2596_, v___y_2597_, v___y_2598_);
lean_dec(v___y_2598_);
lean_dec_ref(v___y_2597_);
lean_dec(v___y_2596_);
lean_dec_ref(v___y_2595_);
lean_dec_ref(v_as_2591_);
return v_res_2602_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound_typePriority___lam__0(lean_object* v_decl_2603_, lean_object* v_type_2604_, lean_object* v_xs_2605_, lean_object* v_t_2606_, lean_object* v___y_2607_, lean_object* v___y_2608_, lean_object* v___y_2609_, lean_object* v___y_2610_){
_start:
{
lean_object* v___x_2612_; 
v___x_2612_ = lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult(v_decl_2603_, v_type_2604_, v_t_2606_, v___y_2607_, v___y_2608_, v___y_2609_, v___y_2610_);
if (lean_obj_tag(v___x_2612_) == 0)
{
lean_object* v___x_2614_; uint8_t v_isShared_2615_; uint8_t v_isSharedCheck_2632_; 
v_isSharedCheck_2632_ = !lean_is_exclusive(v___x_2612_);
if (v_isSharedCheck_2632_ == 0)
{
lean_object* v_unused_2633_; 
v_unused_2633_ = lean_ctor_get(v___x_2612_, 0);
lean_dec(v_unused_2633_);
v___x_2614_ = v___x_2612_;
v_isShared_2615_ = v_isSharedCheck_2632_;
goto v_resetjp_2613_;
}
else
{
lean_dec(v___x_2612_);
v___x_2614_ = lean_box(0);
v_isShared_2615_ = v_isSharedCheck_2632_;
goto v_resetjp_2613_;
}
v_resetjp_2613_:
{
lean_object* v___x_2616_; lean_object* v___x_2617_; uint8_t v___x_2618_; 
v___x_2616_ = lean_unsigned_to_nat(0u);
v___x_2617_ = lean_array_get_size(v_xs_2605_);
v___x_2618_ = lean_nat_dec_lt(v___x_2616_, v___x_2617_);
if (v___x_2618_ == 0)
{
lean_object* v___x_2620_; 
if (v_isShared_2615_ == 0)
{
lean_ctor_set(v___x_2614_, 0, v___x_2616_);
v___x_2620_ = v___x_2614_;
goto v_reusejp_2619_;
}
else
{
lean_object* v_reuseFailAlloc_2621_; 
v_reuseFailAlloc_2621_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2621_, 0, v___x_2616_);
v___x_2620_ = v_reuseFailAlloc_2621_;
goto v_reusejp_2619_;
}
v_reusejp_2619_:
{
return v___x_2620_;
}
}
else
{
uint8_t v___x_2622_; 
v___x_2622_ = lean_nat_dec_le(v___x_2617_, v___x_2617_);
if (v___x_2622_ == 0)
{
if (v___x_2618_ == 0)
{
lean_object* v___x_2624_; 
if (v_isShared_2615_ == 0)
{
lean_ctor_set(v___x_2614_, 0, v___x_2616_);
v___x_2624_ = v___x_2614_;
goto v_reusejp_2623_;
}
else
{
lean_object* v_reuseFailAlloc_2625_; 
v_reuseFailAlloc_2625_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2625_, 0, v___x_2616_);
v___x_2624_ = v_reuseFailAlloc_2625_;
goto v_reusejp_2623_;
}
v_reusejp_2623_:
{
return v___x_2624_;
}
}
else
{
size_t v___x_2626_; size_t v___x_2627_; lean_object* v___x_2628_; 
lean_del_object(v___x_2614_);
v___x_2626_ = ((size_t)0ULL);
v___x_2627_ = lean_usize_of_nat(v___x_2617_);
v___x_2628_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Bound_typePriority_spec__0(v_xs_2605_, v___x_2626_, v___x_2627_, v___x_2616_, v___y_2607_, v___y_2608_, v___y_2609_, v___y_2610_);
return v___x_2628_;
}
}
else
{
size_t v___x_2629_; size_t v___x_2630_; lean_object* v___x_2631_; 
lean_del_object(v___x_2614_);
v___x_2629_ = ((size_t)0ULL);
v___x_2630_ = lean_usize_of_nat(v___x_2617_);
v___x_2631_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Bound_typePriority_spec__0(v_xs_2605_, v___x_2629_, v___x_2630_, v___x_2616_, v___y_2607_, v___y_2608_, v___y_2609_, v___y_2610_);
return v___x_2631_;
}
}
}
}
else
{
lean_object* v_a_2634_; lean_object* v___x_2636_; uint8_t v_isShared_2637_; uint8_t v_isSharedCheck_2641_; 
v_a_2634_ = lean_ctor_get(v___x_2612_, 0);
v_isSharedCheck_2641_ = !lean_is_exclusive(v___x_2612_);
if (v_isSharedCheck_2641_ == 0)
{
v___x_2636_ = v___x_2612_;
v_isShared_2637_ = v_isSharedCheck_2641_;
goto v_resetjp_2635_;
}
else
{
lean_inc(v_a_2634_);
lean_dec(v___x_2612_);
v___x_2636_ = lean_box(0);
v_isShared_2637_ = v_isSharedCheck_2641_;
goto v_resetjp_2635_;
}
v_resetjp_2635_:
{
lean_object* v___x_2639_; 
if (v_isShared_2637_ == 0)
{
v___x_2639_ = v___x_2636_;
goto v_reusejp_2638_;
}
else
{
lean_object* v_reuseFailAlloc_2640_; 
v_reuseFailAlloc_2640_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2640_, 0, v_a_2634_);
v___x_2639_ = v_reuseFailAlloc_2640_;
goto v_reusejp_2638_;
}
v_reusejp_2638_:
{
return v___x_2639_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound_typePriority___lam__0___boxed(lean_object* v_decl_2642_, lean_object* v_type_2643_, lean_object* v_xs_2644_, lean_object* v_t_2645_, lean_object* v___y_2646_, lean_object* v___y_2647_, lean_object* v___y_2648_, lean_object* v___y_2649_, lean_object* v___y_2650_){
_start:
{
lean_object* v_res_2651_; 
v_res_2651_ = lp_mathlib_Mathlib_Tactic_Bound_typePriority___lam__0(v_decl_2642_, v_type_2643_, v_xs_2644_, v_t_2645_, v___y_2646_, v___y_2647_, v___y_2648_, v___y_2649_);
lean_dec(v___y_2649_);
lean_dec_ref(v___y_2648_);
lean_dec(v___y_2647_);
lean_dec_ref(v___y_2646_);
lean_dec_ref(v_xs_2644_);
lean_dec_ref(v_type_2643_);
return v_res_2651_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound_typePriority(lean_object* v_decl_2652_, lean_object* v_type_2653_, lean_object* v_a_2654_, lean_object* v_a_2655_, lean_object* v_a_2656_, lean_object* v_a_2657_){
_start:
{
lean_object* v___f_2659_; uint8_t v___x_2660_; lean_object* v___x_2661_; 
lean_inc_ref(v_type_2653_);
v___f_2659_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Bound_typePriority___lam__0___boxed), 9, 2);
lean_closure_set(v___f_2659_, 0, v_decl_2652_);
lean_closure_set(v___f_2659_, 1, v_type_2653_);
v___x_2660_ = 0;
v___x_2661_ = lp_mathlib_Lean_Meta_forallTelescope___at___00Mathlib_Tactic_Bound_typePriority_spec__1___redArg(v_type_2653_, v___f_2659_, v___x_2660_, v_a_2654_, v_a_2655_, v_a_2656_, v_a_2657_);
return v___x_2661_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound_typePriority___boxed(lean_object* v_decl_2662_, lean_object* v_type_2663_, lean_object* v_a_2664_, lean_object* v_a_2665_, lean_object* v_a_2666_, lean_object* v_a_2667_, lean_object* v_a_2668_){
_start:
{
lean_object* v_res_2669_; 
v_res_2669_ = lp_mathlib_Mathlib_Tactic_Bound_typePriority(v_decl_2662_, v_type_2663_, v_a_2664_, v_a_2665_, v_a_2666_, v_a_2667_);
lean_dec(v_a_2667_);
lean_dec_ref(v_a_2666_);
lean_dec(v_a_2665_);
lean_dec_ref(v_a_2664_);
return v_res_2669_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Bound_declPriority___closed__1(void){
_start:
{
lean_object* v___x_2671_; lean_object* v___x_2672_; 
v___x_2671_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Bound_declPriority___closed__0));
v___x_2672_ = l_Lean_stringToMessageData(v___x_2671_);
return v___x_2672_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound_declPriority(lean_object* v_decl_2673_, lean_object* v_a_2674_, lean_object* v_a_2675_, lean_object* v_a_2676_, lean_object* v_a_2677_){
_start:
{
lean_object* v___x_2679_; lean_object* v_env_2680_; uint8_t v___x_2681_; lean_object* v___x_2682_; 
v___x_2679_ = lean_st_ref_get(v_a_2677_);
v_env_2680_ = lean_ctor_get(v___x_2679_, 0);
lean_inc_ref(v_env_2680_);
lean_dec(v___x_2679_);
v___x_2681_ = 0;
lean_inc(v_decl_2673_);
v___x_2682_ = l_Lean_Environment_find_x3f(v_env_2680_, v_decl_2673_, v___x_2681_);
if (lean_obj_tag(v___x_2682_) == 0)
{
lean_object* v___x_2683_; lean_object* v___x_2684_; lean_object* v___x_2685_; lean_object* v___x_2686_; 
v___x_2683_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Bound_declPriority___closed__1, &lp_mathlib_Mathlib_Tactic_Bound_declPriority___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Bound_declPriority___closed__1);
v___x_2684_ = l_Lean_MessageData_ofName(v_decl_2673_);
v___x_2685_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2685_, 0, v___x_2683_);
lean_ctor_set(v___x_2685_, 1, v___x_2684_);
v___x_2686_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_typePriority_checkResult_spec__0___redArg(v___x_2685_, v_a_2674_, v_a_2675_, v_a_2676_, v_a_2677_);
return v___x_2686_;
}
else
{
lean_object* v_val_2687_; lean_object* v___x_2688_; lean_object* v___x_2689_; 
v_val_2687_ = lean_ctor_get(v___x_2682_, 0);
lean_inc(v_val_2687_);
lean_dec_ref_known(v___x_2682_, 1);
v___x_2688_ = l_Lean_ConstantInfo_type(v_val_2687_);
lean_dec(v_val_2687_);
v___x_2689_ = lp_mathlib_Mathlib_Tactic_Bound_typePriority(v_decl_2673_, v___x_2688_, v_a_2674_, v_a_2675_, v_a_2676_, v_a_2677_);
return v___x_2689_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound_declPriority___boxed(lean_object* v_decl_2690_, lean_object* v_a_2691_, lean_object* v_a_2692_, lean_object* v_a_2693_, lean_object* v_a_2694_, lean_object* v_a_2695_){
_start:
{
lean_object* v_res_2696_; 
v_res_2696_ = lp_mathlib_Mathlib_Tactic_Bound_declPriority(v_decl_2690_, v_a_2691_, v_a_2692_, v_a_2693_, v_a_2694_);
lean_dec(v_a_2694_);
lean_dec_ref(v_a_2693_);
lean_dec(v_a_2692_);
lean_dec_ref(v_a_2691_);
return v_res_2696_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Mathlib_Tactic_Bound_scoreToConfig_spec__0(lean_object* v_a_2697_){
_start:
{
lean_object* v___x_2698_; 
v___x_2698_ = lean_nat_to_int(v_a_2697_);
return v___x_2698_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound_scoreToConfig(lean_object* v_decl_2709_, lean_object* v_score_2710_){
_start:
{
uint8_t v_fst_2712_; lean_object* v_snd_2713_; lean_object* v___x_2725_; uint8_t v___x_2726_; 
v___x_2725_ = lean_unsigned_to_nat(0u);
v___x_2726_ = lean_nat_dec_eq(v_score_2710_, v___x_2725_);
if (v___x_2726_ == 0)
{
uint8_t v___x_2727_; 
v___x_2727_ = 1;
v_fst_2712_ = v___x_2727_;
v_snd_2713_ = v_score_2710_;
goto v___jp_2711_;
}
else
{
uint8_t v___x_2728_; 
lean_dec(v_score_2710_);
v___x_2728_ = 0;
v_fst_2712_ = v___x_2728_;
v_snd_2713_ = v___x_2725_;
goto v___jp_2711_;
}
v___jp_2711_:
{
lean_object* v___x_2714_; lean_object* v___x_2715_; lean_object* v___x_2716_; lean_object* v___x_2717_; lean_object* v___x_2718_; lean_object* v___x_2719_; lean_object* v___x_2720_; lean_object* v___x_2721_; lean_object* v___x_2722_; lean_object* v___x_2723_; lean_object* v___x_2724_; 
v___x_2714_ = l_Lean_mkIdent(v_decl_2709_);
v___x_2715_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2715_, 0, v___x_2714_);
v___x_2716_ = lean_box(v_fst_2712_);
v___x_2717_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2717_, 0, v___x_2716_);
v___x_2718_ = lean_nat_to_int(v_snd_2713_);
v___x_2719_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2719_, 0, v___x_2718_);
v___x_2720_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2720_, 0, v___x_2719_);
v___x_2721_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Bound_scoreToConfig___closed__1));
v___x_2722_ = lp_aesop_Aesop_RuleBuilderOptions_default;
v___x_2723_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Bound_scoreToConfig___closed__3));
v___x_2724_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_2724_, 0, v___x_2715_);
lean_ctor_set(v___x_2724_, 1, v___x_2717_);
lean_ctor_set(v___x_2724_, 2, v___x_2720_);
lean_ctor_set(v___x_2724_, 3, v___x_2721_);
lean_ctor_set(v___x_2724_, 4, v___x_2722_);
lean_ctor_set(v___x_2724_, 5, v___x_2723_);
return v___x_2724_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1___lam__0(lean_object* v_rf_2729_, uint8_t v_anyErased_2730_, lean_object* v_rs_2731_){
_start:
{
lean_object* v___x_2732_; 
v___x_2732_ = lp_aesop_Aesop_GlobalRuleSet_erase(v_rs_2731_, v_rf_2729_);
if (v_anyErased_2730_ == 0)
{
lean_object* v_fst_2733_; lean_object* v_snd_2734_; lean_object* v___x_2736_; uint8_t v_isShared_2737_; uint8_t v_isSharedCheck_2741_; 
v_fst_2733_ = lean_ctor_get(v___x_2732_, 0);
v_snd_2734_ = lean_ctor_get(v___x_2732_, 1);
v_isSharedCheck_2741_ = !lean_is_exclusive(v___x_2732_);
if (v_isSharedCheck_2741_ == 0)
{
v___x_2736_ = v___x_2732_;
v_isShared_2737_ = v_isSharedCheck_2741_;
goto v_resetjp_2735_;
}
else
{
lean_inc(v_snd_2734_);
lean_inc(v_fst_2733_);
lean_dec(v___x_2732_);
v___x_2736_ = lean_box(0);
v_isShared_2737_ = v_isSharedCheck_2741_;
goto v_resetjp_2735_;
}
v_resetjp_2735_:
{
lean_object* v___x_2739_; 
if (v_isShared_2737_ == 0)
{
lean_ctor_set(v___x_2736_, 1, v_fst_2733_);
lean_ctor_set(v___x_2736_, 0, v_snd_2734_);
v___x_2739_ = v___x_2736_;
goto v_reusejp_2738_;
}
else
{
lean_object* v_reuseFailAlloc_2740_; 
v_reuseFailAlloc_2740_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2740_, 0, v_snd_2734_);
lean_ctor_set(v_reuseFailAlloc_2740_, 1, v_fst_2733_);
v___x_2739_ = v_reuseFailAlloc_2740_;
goto v_reusejp_2738_;
}
v_reusejp_2738_:
{
return v___x_2739_;
}
}
}
else
{
lean_object* v_fst_2742_; lean_object* v___x_2744_; uint8_t v_isShared_2745_; uint8_t v_isSharedCheck_2750_; 
v_fst_2742_ = lean_ctor_get(v___x_2732_, 0);
v_isSharedCheck_2750_ = !lean_is_exclusive(v___x_2732_);
if (v_isSharedCheck_2750_ == 0)
{
lean_object* v_unused_2751_; 
v_unused_2751_ = lean_ctor_get(v___x_2732_, 1);
lean_dec(v_unused_2751_);
v___x_2744_ = v___x_2732_;
v_isShared_2745_ = v_isSharedCheck_2750_;
goto v_resetjp_2743_;
}
else
{
lean_inc(v_fst_2742_);
lean_dec(v___x_2732_);
v___x_2744_ = lean_box(0);
v_isShared_2745_ = v_isSharedCheck_2750_;
goto v_resetjp_2743_;
}
v_resetjp_2743_:
{
lean_object* v___x_2746_; lean_object* v___x_2748_; 
v___x_2746_ = lean_box(v_anyErased_2730_);
if (v_isShared_2745_ == 0)
{
lean_ctor_set(v___x_2744_, 1, v_fst_2742_);
lean_ctor_set(v___x_2744_, 0, v___x_2746_);
v___x_2748_ = v___x_2744_;
goto v_reusejp_2747_;
}
else
{
lean_object* v_reuseFailAlloc_2749_; 
v_reuseFailAlloc_2749_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2749_, 0, v___x_2746_);
lean_ctor_set(v_reuseFailAlloc_2749_, 1, v_fst_2742_);
v___x_2748_ = v_reuseFailAlloc_2749_;
goto v_reusejp_2747_;
}
v_reusejp_2747_:
{
return v___x_2748_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1___lam__0___boxed(lean_object* v_rf_2752_, lean_object* v_anyErased_2753_, lean_object* v_rs_2754_){
_start:
{
uint8_t v_anyErased_boxed_2755_; lean_object* v_res_2756_; 
v_anyErased_boxed_2755_ = lean_unbox(v_anyErased_2753_);
v_res_2756_ = lp_mathlib___private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1___lam__0(v_rf_2752_, v_anyErased_boxed_2755_, v_rs_2754_);
return v_res_2756_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1_spec__2___redArg___lam__1(lean_object* v_x_2757_){
_start:
{
lean_object* v___x_2758_; 
v___x_2758_ = l_Lean_Meta_instInhabitedSimpTheorems_default;
return v___x_2758_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1_spec__2___redArg___lam__1___boxed(lean_object* v_x_2759_){
_start:
{
lean_object* v_res_2760_; 
v_res_2760_ = lp_mathlib_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1_spec__2___redArg___lam__1(v_x_2759_);
lean_dec_ref(v_x_2759_);
return v_res_2760_;
}
}
static lean_object* _init_lp_mathlib_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__11___redArg___closed__0(void){
_start:
{
lean_object* v___x_2761_; 
v___x_2761_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_2761_;
}
}
static lean_object* _init_lp_mathlib_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__11___redArg___closed__1(void){
_start:
{
lean_object* v___x_2762_; lean_object* v___x_2763_; 
v___x_2762_ = lean_obj_once(&lp_mathlib_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__11___redArg___closed__0, &lp_mathlib_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__11___redArg___closed__0_once, _init_lp_mathlib_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__11___redArg___closed__0);
v___x_2763_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2763_, 0, v___x_2762_);
return v___x_2763_;
}
}
static lean_object* _init_lp_mathlib_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__11___redArg___closed__2(void){
_start:
{
lean_object* v___x_2764_; lean_object* v___x_2765_; 
v___x_2764_ = lean_obj_once(&lp_mathlib_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__11___redArg___closed__1, &lp_mathlib_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__11___redArg___closed__1_once, _init_lp_mathlib_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__11___redArg___closed__1);
v___x_2765_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2765_, 0, v___x_2764_);
lean_ctor_set(v___x_2765_, 1, v___x_2764_);
return v___x_2765_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__11___redArg(lean_object* v_env_2766_, lean_object* v___y_2767_){
_start:
{
lean_object* v___x_2769_; lean_object* v_nextMacroScope_2770_; lean_object* v_ngen_2771_; lean_object* v_auxDeclNGen_2772_; lean_object* v_traceState_2773_; lean_object* v_messages_2774_; lean_object* v_infoState_2775_; lean_object* v_snapshotTasks_2776_; lean_object* v___x_2778_; uint8_t v_isShared_2779_; uint8_t v_isSharedCheck_2787_; 
v___x_2769_ = lean_st_ref_take(v___y_2767_);
v_nextMacroScope_2770_ = lean_ctor_get(v___x_2769_, 1);
v_ngen_2771_ = lean_ctor_get(v___x_2769_, 2);
v_auxDeclNGen_2772_ = lean_ctor_get(v___x_2769_, 3);
v_traceState_2773_ = lean_ctor_get(v___x_2769_, 4);
v_messages_2774_ = lean_ctor_get(v___x_2769_, 6);
v_infoState_2775_ = lean_ctor_get(v___x_2769_, 7);
v_snapshotTasks_2776_ = lean_ctor_get(v___x_2769_, 8);
v_isSharedCheck_2787_ = !lean_is_exclusive(v___x_2769_);
if (v_isSharedCheck_2787_ == 0)
{
lean_object* v_unused_2788_; lean_object* v_unused_2789_; 
v_unused_2788_ = lean_ctor_get(v___x_2769_, 5);
lean_dec(v_unused_2788_);
v_unused_2789_ = lean_ctor_get(v___x_2769_, 0);
lean_dec(v_unused_2789_);
v___x_2778_ = v___x_2769_;
v_isShared_2779_ = v_isSharedCheck_2787_;
goto v_resetjp_2777_;
}
else
{
lean_inc(v_snapshotTasks_2776_);
lean_inc(v_infoState_2775_);
lean_inc(v_messages_2774_);
lean_inc(v_traceState_2773_);
lean_inc(v_auxDeclNGen_2772_);
lean_inc(v_ngen_2771_);
lean_inc(v_nextMacroScope_2770_);
lean_dec(v___x_2769_);
v___x_2778_ = lean_box(0);
v_isShared_2779_ = v_isSharedCheck_2787_;
goto v_resetjp_2777_;
}
v_resetjp_2777_:
{
lean_object* v___x_2780_; lean_object* v___x_2782_; 
v___x_2780_ = lean_obj_once(&lp_mathlib_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__11___redArg___closed__2, &lp_mathlib_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__11___redArg___closed__2_once, _init_lp_mathlib_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__11___redArg___closed__2);
if (v_isShared_2779_ == 0)
{
lean_ctor_set(v___x_2778_, 5, v___x_2780_);
lean_ctor_set(v___x_2778_, 0, v_env_2766_);
v___x_2782_ = v___x_2778_;
goto v_reusejp_2781_;
}
else
{
lean_object* v_reuseFailAlloc_2786_; 
v_reuseFailAlloc_2786_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_2786_, 0, v_env_2766_);
lean_ctor_set(v_reuseFailAlloc_2786_, 1, v_nextMacroScope_2770_);
lean_ctor_set(v_reuseFailAlloc_2786_, 2, v_ngen_2771_);
lean_ctor_set(v_reuseFailAlloc_2786_, 3, v_auxDeclNGen_2772_);
lean_ctor_set(v_reuseFailAlloc_2786_, 4, v_traceState_2773_);
lean_ctor_set(v_reuseFailAlloc_2786_, 5, v___x_2780_);
lean_ctor_set(v_reuseFailAlloc_2786_, 6, v_messages_2774_);
lean_ctor_set(v_reuseFailAlloc_2786_, 7, v_infoState_2775_);
lean_ctor_set(v_reuseFailAlloc_2786_, 8, v_snapshotTasks_2776_);
v___x_2782_ = v_reuseFailAlloc_2786_;
goto v_reusejp_2781_;
}
v_reusejp_2781_:
{
lean_object* v___x_2783_; lean_object* v___x_2784_; lean_object* v___x_2785_; 
v___x_2783_ = lean_st_ref_set(v___y_2767_, v___x_2782_);
v___x_2784_ = lean_box(0);
v___x_2785_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2785_, 0, v___x_2784_);
return v___x_2785_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__11___redArg___boxed(lean_object* v_env_2790_, lean_object* v___y_2791_, lean_object* v___y_2792_){
_start:
{
lean_object* v_res_2793_; 
v_res_2793_ = lp_mathlib_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__11___redArg(v_env_2790_, v___y_2791_);
lean_dec(v___y_2791_);
return v_res_2793_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1_spec__2___redArg___lam__3(lean_object* v_snd_2794_, lean_object* v_x_2795_){
_start:
{
lean_object* v_simpTheorems_2796_; 
v_simpTheorems_2796_ = lean_ctor_get(v_snd_2794_, 1);
lean_inc_ref(v_simpTheorems_2796_);
return v_simpTheorems_2796_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1_spec__2___redArg___lam__3___boxed(lean_object* v_snd_2797_, lean_object* v_x_2798_){
_start:
{
lean_object* v_res_2799_; 
v_res_2799_ = lp_mathlib_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1_spec__2___redArg___lam__3(v_snd_2797_, v_x_2798_);
lean_dec_ref(v_x_2798_);
lean_dec_ref(v_snd_2797_);
return v_res_2799_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3_spec__16___closed__0(void){
_start:
{
lean_object* v___x_2800_; 
v___x_2800_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_2800_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3_spec__16___closed__1(void){
_start:
{
lean_object* v___x_2801_; lean_object* v___x_2802_; 
v___x_2801_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3_spec__16___closed__0, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3_spec__16___closed__0_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3_spec__16___closed__0);
v___x_2802_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2802_, 0, v___x_2801_);
return v___x_2802_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3_spec__16___closed__2(void){
_start:
{
lean_object* v___x_2803_; lean_object* v___x_2804_; lean_object* v___x_2805_; 
v___x_2803_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3_spec__16___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3_spec__16___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3_spec__16___closed__1);
v___x_2804_ = lean_unsigned_to_nat(0u);
v___x_2805_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_2805_, 0, v___x_2804_);
lean_ctor_set(v___x_2805_, 1, v___x_2804_);
lean_ctor_set(v___x_2805_, 2, v___x_2804_);
lean_ctor_set(v___x_2805_, 3, v___x_2804_);
lean_ctor_set(v___x_2805_, 4, v___x_2803_);
lean_ctor_set(v___x_2805_, 5, v___x_2803_);
lean_ctor_set(v___x_2805_, 6, v___x_2803_);
lean_ctor_set(v___x_2805_, 7, v___x_2803_);
lean_ctor_set(v___x_2805_, 8, v___x_2803_);
lean_ctor_set(v___x_2805_, 9, v___x_2803_);
return v___x_2805_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3_spec__16___closed__3(void){
_start:
{
lean_object* v___x_2806_; lean_object* v___x_2807_; lean_object* v___x_2808_; 
v___x_2806_ = lean_unsigned_to_nat(32u);
v___x_2807_ = lean_mk_empty_array_with_capacity(v___x_2806_);
v___x_2808_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2808_, 0, v___x_2807_);
return v___x_2808_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3_spec__16___closed__4(void){
_start:
{
size_t v___x_2809_; lean_object* v___x_2810_; lean_object* v___x_2811_; lean_object* v___x_2812_; lean_object* v___x_2813_; lean_object* v___x_2814_; 
v___x_2809_ = ((size_t)5ULL);
v___x_2810_ = lean_unsigned_to_nat(0u);
v___x_2811_ = lean_unsigned_to_nat(32u);
v___x_2812_ = lean_mk_empty_array_with_capacity(v___x_2811_);
v___x_2813_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3_spec__16___closed__3, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3_spec__16___closed__3_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3_spec__16___closed__3);
v___x_2814_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_2814_, 0, v___x_2813_);
lean_ctor_set(v___x_2814_, 1, v___x_2812_);
lean_ctor_set(v___x_2814_, 2, v___x_2810_);
lean_ctor_set(v___x_2814_, 3, v___x_2810_);
lean_ctor_set_usize(v___x_2814_, 4, v___x_2809_);
return v___x_2814_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3_spec__16___closed__5(void){
_start:
{
lean_object* v___x_2815_; lean_object* v___x_2816_; lean_object* v___x_2817_; lean_object* v___x_2818_; 
v___x_2815_ = lean_box(1);
v___x_2816_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3_spec__16___closed__4, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3_spec__16___closed__4_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3_spec__16___closed__4);
v___x_2817_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3_spec__16___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3_spec__16___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3_spec__16___closed__1);
v___x_2818_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2818_, 0, v___x_2817_);
lean_ctor_set(v___x_2818_, 1, v___x_2816_);
lean_ctor_set(v___x_2818_, 2, v___x_2815_);
return v___x_2818_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3_spec__16(lean_object* v_msgData_2819_, lean_object* v___y_2820_, lean_object* v___y_2821_){
_start:
{
lean_object* v___x_2823_; lean_object* v_env_2824_; lean_object* v_options_2825_; lean_object* v___x_2826_; lean_object* v___x_2827_; lean_object* v___x_2828_; lean_object* v___x_2829_; lean_object* v___x_2830_; 
v___x_2823_ = lean_st_ref_get(v___y_2821_);
v_env_2824_ = lean_ctor_get(v___x_2823_, 0);
lean_inc_ref(v_env_2824_);
lean_dec(v___x_2823_);
v_options_2825_ = lean_ctor_get(v___y_2820_, 2);
v___x_2826_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3_spec__16___closed__2, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3_spec__16___closed__2_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3_spec__16___closed__2);
v___x_2827_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3_spec__16___closed__5, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3_spec__16___closed__5_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3_spec__16___closed__5);
lean_inc_ref(v_options_2825_);
v___x_2828_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_2828_, 0, v_env_2824_);
lean_ctor_set(v___x_2828_, 1, v___x_2826_);
lean_ctor_set(v___x_2828_, 2, v___x_2827_);
lean_ctor_set(v___x_2828_, 3, v_options_2825_);
v___x_2829_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_2829_, 0, v___x_2828_);
lean_ctor_set(v___x_2829_, 1, v_msgData_2819_);
v___x_2830_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2830_, 0, v___x_2829_);
return v___x_2830_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3_spec__16___boxed(lean_object* v_msgData_2831_, lean_object* v___y_2832_, lean_object* v___y_2833_, lean_object* v___y_2834_){
_start:
{
lean_object* v_res_2835_; 
v_res_2835_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3_spec__16(v_msgData_2831_, v___y_2832_, v___y_2833_);
lean_dec(v___y_2833_);
lean_dec_ref(v___y_2832_);
return v_res_2835_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__0___redArg(lean_object* v_msg_2836_, lean_object* v___y_2837_, lean_object* v___y_2838_){
_start:
{
lean_object* v_ref_2840_; lean_object* v___x_2841_; lean_object* v_a_2842_; lean_object* v___x_2844_; uint8_t v_isShared_2845_; uint8_t v_isSharedCheck_2850_; 
v_ref_2840_ = lean_ctor_get(v___y_2837_, 5);
v___x_2841_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3_spec__16(v_msg_2836_, v___y_2837_, v___y_2838_);
v_a_2842_ = lean_ctor_get(v___x_2841_, 0);
v_isSharedCheck_2850_ = !lean_is_exclusive(v___x_2841_);
if (v_isSharedCheck_2850_ == 0)
{
v___x_2844_ = v___x_2841_;
v_isShared_2845_ = v_isSharedCheck_2850_;
goto v_resetjp_2843_;
}
else
{
lean_inc(v_a_2842_);
lean_dec(v___x_2841_);
v___x_2844_ = lean_box(0);
v_isShared_2845_ = v_isSharedCheck_2850_;
goto v_resetjp_2843_;
}
v_resetjp_2843_:
{
lean_object* v___x_2846_; lean_object* v___x_2848_; 
lean_inc(v_ref_2840_);
v___x_2846_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2846_, 0, v_ref_2840_);
lean_ctor_set(v___x_2846_, 1, v_a_2842_);
if (v_isShared_2845_ == 0)
{
lean_ctor_set_tag(v___x_2844_, 1);
lean_ctor_set(v___x_2844_, 0, v___x_2846_);
v___x_2848_ = v___x_2844_;
goto v_reusejp_2847_;
}
else
{
lean_object* v_reuseFailAlloc_2849_; 
v_reuseFailAlloc_2849_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2849_, 0, v___x_2846_);
v___x_2848_ = v_reuseFailAlloc_2849_;
goto v_reusejp_2847_;
}
v_reusejp_2847_:
{
return v___x_2848_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__0___redArg___boxed(lean_object* v_msg_2851_, lean_object* v___y_2852_, lean_object* v___y_2853_, lean_object* v___y_2854_){
_start:
{
lean_object* v_res_2855_; 
v_res_2855_ = lp_mathlib_Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__0___redArg(v_msg_2851_, v___y_2852_, v___y_2853_);
lean_dec(v___y_2853_);
lean_dec_ref(v___y_2852_);
return v_res_2855_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8_spec__10_spec__17___redArg(lean_object* v_a_2856_, lean_object* v_x_2857_){
_start:
{
if (lean_obj_tag(v_x_2857_) == 0)
{
lean_object* v___x_2858_; 
v___x_2858_ = lean_box(0);
return v___x_2858_;
}
else
{
lean_object* v_key_2859_; lean_object* v_value_2860_; lean_object* v_tail_2861_; uint8_t v___x_2862_; 
v_key_2859_ = lean_ctor_get(v_x_2857_, 0);
v_value_2860_ = lean_ctor_get(v_x_2857_, 1);
v_tail_2861_ = lean_ctor_get(v_x_2857_, 2);
v___x_2862_ = lean_name_eq(v_key_2859_, v_a_2856_);
if (v___x_2862_ == 0)
{
v_x_2857_ = v_tail_2861_;
goto _start;
}
else
{
lean_object* v___x_2864_; 
lean_inc(v_value_2860_);
v___x_2864_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2864_, 0, v_value_2860_);
return v___x_2864_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8_spec__10_spec__17___redArg___boxed(lean_object* v_a_2865_, lean_object* v_x_2866_){
_start:
{
lean_object* v_res_2867_; 
v_res_2867_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8_spec__10_spec__17___redArg(v_a_2865_, v_x_2866_);
lean_dec(v_x_2866_);
lean_dec(v_a_2865_);
return v_res_2867_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8_spec__10___redArg(lean_object* v_m_2868_, lean_object* v_a_2869_){
_start:
{
lean_object* v_buckets_2870_; lean_object* v___x_2871_; uint64_t v___y_2873_; 
v_buckets_2870_ = lean_ctor_get(v_m_2868_, 1);
v___x_2871_ = lean_array_get_size(v_buckets_2870_);
if (lean_obj_tag(v_a_2869_) == 0)
{
uint64_t v___x_2887_; 
v___x_2887_ = 1723ULL;
v___y_2873_ = v___x_2887_;
goto v___jp_2872_;
}
else
{
uint64_t v_hash_2888_; 
v_hash_2888_ = lean_ctor_get_uint64(v_a_2869_, sizeof(void*)*2);
v___y_2873_ = v_hash_2888_;
goto v___jp_2872_;
}
v___jp_2872_:
{
uint64_t v___x_2874_; uint64_t v___x_2875_; uint64_t v_fold_2876_; uint64_t v___x_2877_; uint64_t v___x_2878_; uint64_t v___x_2879_; size_t v___x_2880_; size_t v___x_2881_; size_t v___x_2882_; size_t v___x_2883_; size_t v___x_2884_; lean_object* v___x_2885_; lean_object* v___x_2886_; 
v___x_2874_ = 32ULL;
v___x_2875_ = lean_uint64_shift_right(v___y_2873_, v___x_2874_);
v_fold_2876_ = lean_uint64_xor(v___y_2873_, v___x_2875_);
v___x_2877_ = 16ULL;
v___x_2878_ = lean_uint64_shift_right(v_fold_2876_, v___x_2877_);
v___x_2879_ = lean_uint64_xor(v_fold_2876_, v___x_2878_);
v___x_2880_ = lean_uint64_to_usize(v___x_2879_);
v___x_2881_ = lean_usize_of_nat(v___x_2871_);
v___x_2882_ = ((size_t)1ULL);
v___x_2883_ = lean_usize_sub(v___x_2881_, v___x_2882_);
v___x_2884_ = lean_usize_land(v___x_2880_, v___x_2883_);
v___x_2885_ = lean_array_uget_borrowed(v_buckets_2870_, v___x_2884_);
v___x_2886_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8_spec__10_spec__17___redArg(v_a_2869_, v___x_2885_);
return v___x_2886_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8_spec__10___redArg___boxed(lean_object* v_m_2889_, lean_object* v_a_2890_){
_start:
{
lean_object* v_res_2891_; 
v_res_2891_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8_spec__10___redArg(v_m_2889_, v_a_2890_);
lean_dec(v_a_2890_);
lean_dec_ref(v_m_2889_);
return v_res_2891_;
}
}
static lean_object* _init_lp_mathlib_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8___closed__1(void){
_start:
{
lean_object* v___x_2893_; lean_object* v___x_2894_; 
v___x_2893_ = ((lean_object*)(lp_mathlib_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8___closed__0));
v___x_2894_ = l_Lean_stringToMessageData(v___x_2893_);
return v___x_2894_;
}
}
static lean_object* _init_lp_mathlib_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8___closed__3(void){
_start:
{
lean_object* v___x_2896_; lean_object* v___x_2897_; 
v___x_2896_ = ((lean_object*)(lp_mathlib_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8___closed__2));
v___x_2897_ = l_Lean_stringToMessageData(v___x_2896_);
return v___x_2897_;
}
}
static lean_object* _init_lp_mathlib_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8___closed__5(void){
_start:
{
lean_object* v___x_2899_; lean_object* v___x_2900_; 
v___x_2899_ = ((lean_object*)(lp_mathlib_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8___closed__4));
v___x_2900_ = l_Lean_stringToMessageData(v___x_2899_);
return v___x_2900_;
}
}
static lean_object* _init_lp_mathlib_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8___closed__7(void){
_start:
{
lean_object* v___x_2902_; lean_object* v___x_2903_; 
v___x_2902_ = ((lean_object*)(lp_mathlib_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8___closed__6));
v___x_2903_ = l_Lean_stringToMessageData(v___x_2902_);
return v___x_2903_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8(lean_object* v_rsName_2904_, lean_object* v___y_2905_, lean_object* v___y_2906_){
_start:
{
lean_object* v___x_2908_; 
v___x_2908_ = lp_aesop_Aesop_getDeclaredRuleSets();
if (lean_obj_tag(v___x_2908_) == 0)
{
lean_object* v_a_2909_; lean_object* v___x_2910_; 
v_a_2909_ = lean_ctor_get(v___x_2908_, 0);
lean_inc(v_a_2909_);
lean_dec_ref_known(v___x_2908_, 1);
v___x_2910_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8_spec__10___redArg(v_a_2909_, v_rsName_2904_);
lean_dec(v_a_2909_);
if (lean_obj_tag(v___x_2910_) == 1)
{
lean_object* v_val_2911_; lean_object* v_snd_2912_; lean_object* v_fst_2913_; lean_object* v___x_2915_; uint8_t v_isShared_2916_; uint8_t v_isSharedCheck_2983_; 
lean_dec(v_rsName_2904_);
v_val_2911_ = lean_ctor_get(v___x_2910_, 0);
lean_inc(v_val_2911_);
lean_dec_ref_known(v___x_2910_, 1);
v_snd_2912_ = lean_ctor_get(v_val_2911_, 1);
v_fst_2913_ = lean_ctor_get(v_val_2911_, 0);
v_isSharedCheck_2983_ = !lean_is_exclusive(v_val_2911_);
if (v_isSharedCheck_2983_ == 0)
{
v___x_2915_ = v_val_2911_;
v_isShared_2916_ = v_isSharedCheck_2983_;
goto v_resetjp_2914_;
}
else
{
lean_inc(v_snd_2912_);
lean_inc(v_fst_2913_);
lean_dec(v_val_2911_);
v___x_2915_ = lean_box(0);
v_isShared_2916_ = v_isSharedCheck_2983_;
goto v_resetjp_2914_;
}
v_resetjp_2914_:
{
lean_object* v_fst_2917_; lean_object* v_snd_2918_; lean_object* v___x_2920_; uint8_t v_isShared_2921_; uint8_t v_isSharedCheck_2982_; 
v_fst_2917_ = lean_ctor_get(v_snd_2912_, 0);
v_snd_2918_ = lean_ctor_get(v_snd_2912_, 1);
v_isSharedCheck_2982_ = !lean_is_exclusive(v_snd_2912_);
if (v_isSharedCheck_2982_ == 0)
{
v___x_2920_ = v_snd_2912_;
v_isShared_2921_ = v_isSharedCheck_2982_;
goto v_resetjp_2919_;
}
else
{
lean_inc(v_snd_2918_);
lean_inc(v_fst_2917_);
lean_dec(v_snd_2912_);
v___x_2920_ = lean_box(0);
v_isShared_2921_ = v_isSharedCheck_2982_;
goto v_resetjp_2919_;
}
v_resetjp_2919_:
{
lean_object* v___x_2922_; 
v___x_2922_ = l_Lean_Meta_getSimpExtension_x3f(v_fst_2917_, v___y_2905_, v___y_2906_);
if (lean_obj_tag(v___x_2922_) == 0)
{
lean_object* v_a_2923_; 
v_a_2923_ = lean_ctor_get(v___x_2922_, 0);
lean_inc(v_a_2923_);
lean_dec_ref_known(v___x_2922_, 1);
if (lean_obj_tag(v_a_2923_) == 1)
{
lean_object* v_val_2924_; lean_object* v___x_2926_; uint8_t v_isShared_2927_; uint8_t v_isSharedCheck_2967_; 
v_val_2924_ = lean_ctor_get(v_a_2923_, 0);
v_isSharedCheck_2967_ = !lean_is_exclusive(v_a_2923_);
if (v_isSharedCheck_2967_ == 0)
{
v___x_2926_ = v_a_2923_;
v_isShared_2927_ = v_isSharedCheck_2967_;
goto v_resetjp_2925_;
}
else
{
lean_inc(v_val_2924_);
lean_dec(v_a_2923_);
v___x_2926_ = lean_box(0);
v_isShared_2927_ = v_isSharedCheck_2967_;
goto v_resetjp_2925_;
}
v_resetjp_2925_:
{
lean_object* v___x_2928_; 
lean_inc(v_fst_2917_);
v___x_2928_ = l_Lean_Meta_Simp_getSimprocExtension_x3f(v_fst_2917_);
if (lean_obj_tag(v___x_2928_) == 0)
{
lean_object* v_a_2929_; lean_object* v___x_2931_; uint8_t v_isShared_2932_; uint8_t v_isSharedCheck_2951_; 
lean_del_object(v___x_2926_);
v_a_2929_ = lean_ctor_get(v___x_2928_, 0);
v_isSharedCheck_2951_ = !lean_is_exclusive(v___x_2928_);
if (v_isSharedCheck_2951_ == 0)
{
v___x_2931_ = v___x_2928_;
v_isShared_2932_ = v_isSharedCheck_2951_;
goto v_resetjp_2930_;
}
else
{
lean_inc(v_a_2929_);
lean_dec(v___x_2928_);
v___x_2931_ = lean_box(0);
v_isShared_2932_ = v_isSharedCheck_2951_;
goto v_resetjp_2930_;
}
v_resetjp_2930_:
{
if (lean_obj_tag(v_a_2929_) == 1)
{
lean_object* v_val_2933_; lean_object* v___x_2935_; 
v_val_2933_ = lean_ctor_get(v_a_2929_, 0);
lean_inc(v_val_2933_);
lean_dec_ref_known(v_a_2929_, 1);
if (v_isShared_2921_ == 0)
{
lean_ctor_set(v___x_2920_, 1, v_val_2933_);
lean_ctor_set(v___x_2920_, 0, v_snd_2918_);
v___x_2935_ = v___x_2920_;
goto v_reusejp_2934_;
}
else
{
lean_object* v_reuseFailAlloc_2944_; 
v_reuseFailAlloc_2944_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2944_, 0, v_snd_2918_);
lean_ctor_set(v_reuseFailAlloc_2944_, 1, v_val_2933_);
v___x_2935_ = v_reuseFailAlloc_2944_;
goto v_reusejp_2934_;
}
v_reusejp_2934_:
{
lean_object* v___x_2937_; 
if (v_isShared_2916_ == 0)
{
lean_ctor_set(v___x_2915_, 1, v___x_2935_);
lean_ctor_set(v___x_2915_, 0, v_val_2924_);
v___x_2937_ = v___x_2915_;
goto v_reusejp_2936_;
}
else
{
lean_object* v_reuseFailAlloc_2943_; 
v_reuseFailAlloc_2943_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2943_, 0, v_val_2924_);
lean_ctor_set(v_reuseFailAlloc_2943_, 1, v___x_2935_);
v___x_2937_ = v_reuseFailAlloc_2943_;
goto v_reusejp_2936_;
}
v_reusejp_2936_:
{
lean_object* v___x_2938_; lean_object* v___x_2939_; lean_object* v___x_2941_; 
v___x_2938_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2938_, 0, v_fst_2917_);
lean_ctor_set(v___x_2938_, 1, v___x_2937_);
v___x_2939_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2939_, 0, v_fst_2913_);
lean_ctor_set(v___x_2939_, 1, v___x_2938_);
if (v_isShared_2932_ == 0)
{
lean_ctor_set(v___x_2931_, 0, v___x_2939_);
v___x_2941_ = v___x_2931_;
goto v_reusejp_2940_;
}
else
{
lean_object* v_reuseFailAlloc_2942_; 
v_reuseFailAlloc_2942_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2942_, 0, v___x_2939_);
v___x_2941_ = v_reuseFailAlloc_2942_;
goto v_reusejp_2940_;
}
v_reusejp_2940_:
{
return v___x_2941_;
}
}
}
}
else
{
lean_object* v___x_2945_; lean_object* v___x_2946_; lean_object* v___x_2947_; lean_object* v___x_2948_; lean_object* v___x_2949_; lean_object* v___x_2950_; 
lean_del_object(v___x_2931_);
lean_dec(v_a_2929_);
lean_dec(v_val_2924_);
lean_del_object(v___x_2920_);
lean_dec(v_snd_2918_);
lean_del_object(v___x_2915_);
lean_dec(v_fst_2913_);
v___x_2945_ = lean_obj_once(&lp_mathlib_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8___closed__1, &lp_mathlib_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8___closed__1_once, _init_lp_mathlib_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8___closed__1);
v___x_2946_ = l_Lean_MessageData_ofName(v_fst_2917_);
v___x_2947_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2947_, 0, v___x_2945_);
lean_ctor_set(v___x_2947_, 1, v___x_2946_);
v___x_2948_ = lean_obj_once(&lp_mathlib_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8___closed__3, &lp_mathlib_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8___closed__3_once, _init_lp_mathlib_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8___closed__3);
v___x_2949_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2949_, 0, v___x_2947_);
lean_ctor_set(v___x_2949_, 1, v___x_2948_);
v___x_2950_ = lp_mathlib_Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__0___redArg(v___x_2949_, v___y_2905_, v___y_2906_);
return v___x_2950_;
}
}
}
else
{
lean_object* v_a_2952_; lean_object* v___x_2954_; uint8_t v_isShared_2955_; uint8_t v_isSharedCheck_2966_; 
lean_dec(v_val_2924_);
lean_del_object(v___x_2920_);
lean_dec(v_snd_2918_);
lean_dec(v_fst_2917_);
lean_del_object(v___x_2915_);
lean_dec(v_fst_2913_);
v_a_2952_ = lean_ctor_get(v___x_2928_, 0);
v_isSharedCheck_2966_ = !lean_is_exclusive(v___x_2928_);
if (v_isSharedCheck_2966_ == 0)
{
v___x_2954_ = v___x_2928_;
v_isShared_2955_ = v_isSharedCheck_2966_;
goto v_resetjp_2953_;
}
else
{
lean_inc(v_a_2952_);
lean_dec(v___x_2928_);
v___x_2954_ = lean_box(0);
v_isShared_2955_ = v_isSharedCheck_2966_;
goto v_resetjp_2953_;
}
v_resetjp_2953_:
{
lean_object* v_ref_2956_; lean_object* v___x_2957_; lean_object* v___x_2959_; 
v_ref_2956_ = lean_ctor_get(v___y_2905_, 5);
v___x_2957_ = lean_io_error_to_string(v_a_2952_);
if (v_isShared_2927_ == 0)
{
lean_ctor_set_tag(v___x_2926_, 3);
lean_ctor_set(v___x_2926_, 0, v___x_2957_);
v___x_2959_ = v___x_2926_;
goto v_reusejp_2958_;
}
else
{
lean_object* v_reuseFailAlloc_2965_; 
v_reuseFailAlloc_2965_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2965_, 0, v___x_2957_);
v___x_2959_ = v_reuseFailAlloc_2965_;
goto v_reusejp_2958_;
}
v_reusejp_2958_:
{
lean_object* v___x_2960_; lean_object* v___x_2961_; lean_object* v___x_2963_; 
v___x_2960_ = l_Lean_MessageData_ofFormat(v___x_2959_);
lean_inc(v_ref_2956_);
v___x_2961_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2961_, 0, v_ref_2956_);
lean_ctor_set(v___x_2961_, 1, v___x_2960_);
if (v_isShared_2955_ == 0)
{
lean_ctor_set(v___x_2954_, 0, v___x_2961_);
v___x_2963_ = v___x_2954_;
goto v_reusejp_2962_;
}
else
{
lean_object* v_reuseFailAlloc_2964_; 
v_reuseFailAlloc_2964_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2964_, 0, v___x_2961_);
v___x_2963_ = v_reuseFailAlloc_2964_;
goto v_reusejp_2962_;
}
v_reusejp_2962_:
{
return v___x_2963_;
}
}
}
}
}
}
else
{
lean_object* v___x_2968_; lean_object* v___x_2969_; lean_object* v___x_2970_; lean_object* v___x_2971_; lean_object* v___x_2972_; lean_object* v___x_2973_; 
lean_dec(v_a_2923_);
lean_del_object(v___x_2920_);
lean_dec(v_snd_2918_);
lean_del_object(v___x_2915_);
lean_dec(v_fst_2913_);
v___x_2968_ = lean_obj_once(&lp_mathlib_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8___closed__1, &lp_mathlib_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8___closed__1_once, _init_lp_mathlib_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8___closed__1);
v___x_2969_ = l_Lean_MessageData_ofName(v_fst_2917_);
v___x_2970_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2970_, 0, v___x_2968_);
lean_ctor_set(v___x_2970_, 1, v___x_2969_);
v___x_2971_ = lean_obj_once(&lp_mathlib_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8___closed__3, &lp_mathlib_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8___closed__3_once, _init_lp_mathlib_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8___closed__3);
v___x_2972_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2972_, 0, v___x_2970_);
lean_ctor_set(v___x_2972_, 1, v___x_2971_);
v___x_2973_ = lp_mathlib_Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__0___redArg(v___x_2972_, v___y_2905_, v___y_2906_);
return v___x_2973_;
}
}
else
{
lean_object* v_a_2974_; lean_object* v___x_2976_; uint8_t v_isShared_2977_; uint8_t v_isSharedCheck_2981_; 
lean_del_object(v___x_2920_);
lean_dec(v_snd_2918_);
lean_dec(v_fst_2917_);
lean_del_object(v___x_2915_);
lean_dec(v_fst_2913_);
v_a_2974_ = lean_ctor_get(v___x_2922_, 0);
v_isSharedCheck_2981_ = !lean_is_exclusive(v___x_2922_);
if (v_isSharedCheck_2981_ == 0)
{
v___x_2976_ = v___x_2922_;
v_isShared_2977_ = v_isSharedCheck_2981_;
goto v_resetjp_2975_;
}
else
{
lean_inc(v_a_2974_);
lean_dec(v___x_2922_);
v___x_2976_ = lean_box(0);
v_isShared_2977_ = v_isSharedCheck_2981_;
goto v_resetjp_2975_;
}
v_resetjp_2975_:
{
lean_object* v___x_2979_; 
if (v_isShared_2977_ == 0)
{
v___x_2979_ = v___x_2976_;
goto v_reusejp_2978_;
}
else
{
lean_object* v_reuseFailAlloc_2980_; 
v_reuseFailAlloc_2980_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2980_, 0, v_a_2974_);
v___x_2979_ = v_reuseFailAlloc_2980_;
goto v_reusejp_2978_;
}
v_reusejp_2978_:
{
return v___x_2979_;
}
}
}
}
}
}
else
{
lean_object* v___x_2984_; lean_object* v___x_2985_; lean_object* v___x_2986_; lean_object* v___x_2987_; lean_object* v___x_2988_; lean_object* v___x_2989_; 
lean_dec(v___x_2910_);
v___x_2984_ = lean_obj_once(&lp_mathlib_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8___closed__5, &lp_mathlib_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8___closed__5_once, _init_lp_mathlib_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8___closed__5);
v___x_2985_ = l_Lean_MessageData_ofName(v_rsName_2904_);
v___x_2986_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2986_, 0, v___x_2984_);
lean_ctor_set(v___x_2986_, 1, v___x_2985_);
v___x_2987_ = lean_obj_once(&lp_mathlib_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8___closed__7, &lp_mathlib_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8___closed__7_once, _init_lp_mathlib_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8___closed__7);
v___x_2988_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2988_, 0, v___x_2986_);
lean_ctor_set(v___x_2988_, 1, v___x_2987_);
v___x_2989_ = lp_mathlib_Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__0___redArg(v___x_2988_, v___y_2905_, v___y_2906_);
return v___x_2989_;
}
}
else
{
lean_object* v_a_2990_; lean_object* v___x_2992_; uint8_t v_isShared_2993_; uint8_t v_isSharedCheck_3002_; 
lean_dec(v_rsName_2904_);
v_a_2990_ = lean_ctor_get(v___x_2908_, 0);
v_isSharedCheck_3002_ = !lean_is_exclusive(v___x_2908_);
if (v_isSharedCheck_3002_ == 0)
{
v___x_2992_ = v___x_2908_;
v_isShared_2993_ = v_isSharedCheck_3002_;
goto v_resetjp_2991_;
}
else
{
lean_inc(v_a_2990_);
lean_dec(v___x_2908_);
v___x_2992_ = lean_box(0);
v_isShared_2993_ = v_isSharedCheck_3002_;
goto v_resetjp_2991_;
}
v_resetjp_2991_:
{
lean_object* v_ref_2994_; lean_object* v___x_2995_; lean_object* v___x_2996_; lean_object* v___x_2997_; lean_object* v___x_2998_; lean_object* v___x_3000_; 
v_ref_2994_ = lean_ctor_get(v___y_2905_, 5);
v___x_2995_ = lean_io_error_to_string(v_a_2990_);
v___x_2996_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2996_, 0, v___x_2995_);
v___x_2997_ = l_Lean_MessageData_ofFormat(v___x_2996_);
lean_inc(v_ref_2994_);
v___x_2998_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2998_, 0, v_ref_2994_);
lean_ctor_set(v___x_2998_, 1, v___x_2997_);
if (v_isShared_2993_ == 0)
{
lean_ctor_set(v___x_2992_, 0, v___x_2998_);
v___x_3000_ = v___x_2992_;
goto v_reusejp_2999_;
}
else
{
lean_object* v_reuseFailAlloc_3001_; 
v_reuseFailAlloc_3001_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3001_, 0, v___x_2998_);
v___x_3000_ = v_reuseFailAlloc_3001_;
goto v_reusejp_2999_;
}
v_reusejp_2999_:
{
return v___x_3000_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8___boxed(lean_object* v_rsName_3003_, lean_object* v___y_3004_, lean_object* v___y_3005_, lean_object* v___y_3006_){
_start:
{
lean_object* v_res_3007_; 
v_res_3007_ = lp_mathlib_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8(v_rsName_3003_, v___y_3004_, v___y_3005_);
lean_dec(v___y_3005_);
lean_dec_ref(v___y_3004_);
return v_res_3007_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1_spec__2___redArg___lam__4(lean_object* v_snd_3008_, lean_object* v_x_3009_){
_start:
{
lean_object* v_toBaseRuleSet_3010_; 
v_toBaseRuleSet_3010_ = lean_ctor_get(v_snd_3008_, 0);
lean_inc_ref(v_toBaseRuleSet_3010_);
return v_toBaseRuleSet_3010_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1_spec__2___redArg___lam__4___boxed(lean_object* v_snd_3011_, lean_object* v_x_3012_){
_start:
{
lean_object* v_res_3013_; 
v_res_3013_ = lp_mathlib_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1_spec__2___redArg___lam__4(v_snd_3011_, v_x_3012_);
lean_dec_ref(v_x_3012_);
lean_dec_ref(v_snd_3011_);
return v_res_3013_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1_spec__2___redArg___lam__2(lean_object* v_x_3014_){
_start:
{
lean_object* v___x_3015_; 
v___x_3015_ = lp_aesop_Aesop_instInhabitedBaseRuleSet_default;
return v___x_3015_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1_spec__2___redArg___lam__2___boxed(lean_object* v_x_3016_){
_start:
{
lean_object* v_res_3017_; 
v_res_3017_ = lp_mathlib_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1_spec__2___redArg___lam__2(v_x_3016_);
lean_dec_ref(v_x_3016_);
return v_res_3017_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1_spec__2___redArg___lam__0(lean_object* v_x_3018_){
_start:
{
lean_object* v___x_3019_; 
v___x_3019_ = l_Lean_Meta_Simp_instInhabitedSimprocs_default;
return v___x_3019_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1_spec__2___redArg___lam__0___boxed(lean_object* v_x_3020_){
_start:
{
lean_object* v_res_3021_; 
v_res_3021_ = lp_mathlib_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1_spec__2___redArg___lam__0(v_x_3020_);
lean_dec_ref(v_x_3020_);
return v_res_3021_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1_spec__2___redArg(lean_object* v_rsName_3025_, lean_object* v_f_3026_, lean_object* v___y_3027_, lean_object* v___y_3028_){
_start:
{
lean_object* v___x_3030_; 
v___x_3030_ = lp_mathlib_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8(v_rsName_3025_, v___y_3027_, v___y_3028_);
if (lean_obj_tag(v___x_3030_) == 0)
{
lean_object* v_a_3031_; lean_object* v_snd_3032_; lean_object* v_snd_3033_; lean_object* v_snd_3034_; lean_object* v_fst_3035_; lean_object* v_fst_3036_; lean_object* v_snd_3037_; lean_object* v___x_3038_; lean_object* v_ext_3039_; lean_object* v_toEnvExtension_3040_; lean_object* v_ext_3041_; lean_object* v_toEnvExtension_3042_; lean_object* v_ext_3043_; lean_object* v_toEnvExtension_3044_; lean_object* v_env_3045_; lean_object* v_asyncMode_3046_; lean_object* v_asyncMode_3047_; lean_object* v_asyncMode_3048_; lean_object* v___x_3049_; lean_object* v_base_3050_; lean_object* v___x_3051_; lean_object* v___x_3052_; lean_object* v_simpTheorems_3053_; lean_object* v_simprocs_3054_; lean_object* v_rs_3055_; lean_object* v___x_3056_; lean_object* v_fst_3057_; lean_object* v_snd_3058_; lean_object* v___f_3059_; lean_object* v___f_3060_; lean_object* v___f_3061_; lean_object* v___f_3062_; lean_object* v___f_3063_; lean_object* v_env_3064_; lean_object* v_env_3065_; lean_object* v_env_3066_; lean_object* v_env_3067_; lean_object* v_env_3068_; lean_object* v___x_3069_; lean_object* v___x_3071_; uint8_t v_isShared_3072_; uint8_t v_isSharedCheck_3076_; 
v_a_3031_ = lean_ctor_get(v___x_3030_, 0);
lean_inc(v_a_3031_);
lean_dec_ref_known(v___x_3030_, 1);
v_snd_3032_ = lean_ctor_get(v_a_3031_, 1);
v_snd_3033_ = lean_ctor_get(v_snd_3032_, 1);
lean_inc(v_snd_3033_);
v_snd_3034_ = lean_ctor_get(v_snd_3033_, 1);
lean_inc(v_snd_3034_);
v_fst_3035_ = lean_ctor_get(v_a_3031_, 0);
lean_inc_n(v_fst_3035_, 2);
lean_dec(v_a_3031_);
v_fst_3036_ = lean_ctor_get(v_snd_3033_, 0);
lean_inc_n(v_fst_3036_, 2);
lean_dec(v_snd_3033_);
v_snd_3037_ = lean_ctor_get(v_snd_3034_, 1);
lean_inc(v_snd_3037_);
lean_dec(v_snd_3034_);
v___x_3038_ = lean_st_ref_get(v___y_3028_);
v_ext_3039_ = lean_ctor_get(v_fst_3035_, 1);
v_toEnvExtension_3040_ = lean_ctor_get(v_ext_3039_, 0);
v_ext_3041_ = lean_ctor_get(v_fst_3036_, 1);
v_toEnvExtension_3042_ = lean_ctor_get(v_ext_3041_, 0);
v_ext_3043_ = lean_ctor_get(v_snd_3037_, 1);
v_toEnvExtension_3044_ = lean_ctor_get(v_ext_3043_, 0);
v_env_3045_ = lean_ctor_get(v___x_3038_, 0);
lean_inc_ref_n(v_env_3045_, 4);
lean_dec(v___x_3038_);
v_asyncMode_3046_ = lean_ctor_get(v_toEnvExtension_3040_, 2);
v_asyncMode_3047_ = lean_ctor_get(v_toEnvExtension_3042_, 2);
v_asyncMode_3048_ = lean_ctor_get(v_toEnvExtension_3044_, 2);
v___x_3049_ = lp_aesop_Aesop_instInhabitedBaseRuleSet_default;
v_base_3050_ = l_Lean_ScopedEnvExtension_getState___redArg(v___x_3049_, v_fst_3035_, v_env_3045_, v_asyncMode_3046_);
v___x_3051_ = l_Lean_Meta_instInhabitedSimpTheorems_default;
v___x_3052_ = l_Lean_Meta_Simp_instInhabitedSimprocs_default;
v_simpTheorems_3053_ = l_Lean_ScopedEnvExtension_getState___redArg(v___x_3051_, v_fst_3036_, v_env_3045_, v_asyncMode_3047_);
v_simprocs_3054_ = l_Lean_ScopedEnvExtension_getState___redArg(v___x_3052_, v_snd_3037_, v_env_3045_, v_asyncMode_3048_);
v_rs_3055_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_rs_3055_, 0, v_base_3050_);
lean_ctor_set(v_rs_3055_, 1, v_simpTheorems_3053_);
lean_ctor_set(v_rs_3055_, 2, v_simprocs_3054_);
v___x_3056_ = lean_apply_1(v_f_3026_, v_rs_3055_);
v_fst_3057_ = lean_ctor_get(v___x_3056_, 0);
lean_inc(v_fst_3057_);
v_snd_3058_ = lean_ctor_get(v___x_3056_, 1);
lean_inc_n(v_snd_3058_, 2);
lean_dec_ref(v___x_3056_);
v___f_3059_ = ((lean_object*)(lp_mathlib_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1_spec__2___redArg___closed__0));
v___f_3060_ = ((lean_object*)(lp_mathlib_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1_spec__2___redArg___closed__1));
v___f_3061_ = ((lean_object*)(lp_mathlib_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1_spec__2___redArg___closed__2));
v___f_3062_ = lean_alloc_closure((void*)(lp_mathlib_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1_spec__2___redArg___lam__3___boxed), 2, 1);
lean_closure_set(v___f_3062_, 0, v_snd_3058_);
v___f_3063_ = lean_alloc_closure((void*)(lp_mathlib_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1_spec__2___redArg___lam__4___boxed), 2, 1);
lean_closure_set(v___f_3063_, 0, v_snd_3058_);
v_env_3064_ = l_Lean_ScopedEnvExtension_modifyState___redArg(v_fst_3035_, v_env_3045_, v___f_3061_);
v_env_3065_ = l_Lean_ScopedEnvExtension_modifyState___redArg(v_fst_3036_, v_env_3064_, v___f_3060_);
v_env_3066_ = l_Lean_ScopedEnvExtension_modifyState___redArg(v_snd_3037_, v_env_3065_, v___f_3059_);
v_env_3067_ = l_Lean_ScopedEnvExtension_modifyState___redArg(v_fst_3035_, v_env_3066_, v___f_3063_);
v_env_3068_ = l_Lean_ScopedEnvExtension_modifyState___redArg(v_fst_3036_, v_env_3067_, v___f_3062_);
v___x_3069_ = lp_mathlib_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__11___redArg(v_env_3068_, v___y_3028_);
v_isSharedCheck_3076_ = !lean_is_exclusive(v___x_3069_);
if (v_isSharedCheck_3076_ == 0)
{
lean_object* v_unused_3077_; 
v_unused_3077_ = lean_ctor_get(v___x_3069_, 0);
lean_dec(v_unused_3077_);
v___x_3071_ = v___x_3069_;
v_isShared_3072_ = v_isSharedCheck_3076_;
goto v_resetjp_3070_;
}
else
{
lean_dec(v___x_3069_);
v___x_3071_ = lean_box(0);
v_isShared_3072_ = v_isSharedCheck_3076_;
goto v_resetjp_3070_;
}
v_resetjp_3070_:
{
lean_object* v___x_3074_; 
if (v_isShared_3072_ == 0)
{
lean_ctor_set(v___x_3071_, 0, v_fst_3057_);
v___x_3074_ = v___x_3071_;
goto v_reusejp_3073_;
}
else
{
lean_object* v_reuseFailAlloc_3075_; 
v_reuseFailAlloc_3075_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3075_, 0, v_fst_3057_);
v___x_3074_ = v_reuseFailAlloc_3075_;
goto v_reusejp_3073_;
}
v_reusejp_3073_:
{
return v___x_3074_;
}
}
}
else
{
lean_object* v_a_3078_; lean_object* v___x_3080_; uint8_t v_isShared_3081_; uint8_t v_isSharedCheck_3085_; 
lean_dec_ref(v_f_3026_);
v_a_3078_ = lean_ctor_get(v___x_3030_, 0);
v_isSharedCheck_3085_ = !lean_is_exclusive(v___x_3030_);
if (v_isSharedCheck_3085_ == 0)
{
v___x_3080_ = v___x_3030_;
v_isShared_3081_ = v_isSharedCheck_3085_;
goto v_resetjp_3079_;
}
else
{
lean_inc(v_a_3078_);
lean_dec(v___x_3030_);
v___x_3080_ = lean_box(0);
v_isShared_3081_ = v_isSharedCheck_3085_;
goto v_resetjp_3079_;
}
v_resetjp_3079_:
{
lean_object* v___x_3083_; 
if (v_isShared_3081_ == 0)
{
v___x_3083_ = v___x_3080_;
goto v_reusejp_3082_;
}
else
{
lean_object* v_reuseFailAlloc_3084_; 
v_reuseFailAlloc_3084_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3084_, 0, v_a_3078_);
v___x_3083_ = v_reuseFailAlloc_3084_;
goto v_reusejp_3082_;
}
v_reusejp_3082_:
{
return v___x_3083_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1_spec__2___redArg___boxed(lean_object* v_rsName_3086_, lean_object* v_f_3087_, lean_object* v___y_3088_, lean_object* v___y_3089_, lean_object* v___y_3090_){
_start:
{
lean_object* v_res_3091_; 
v_res_3091_ = lp_mathlib_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1_spec__2___redArg(v_rsName_3086_, v_f_3087_, v___y_3088_, v___y_3089_);
lean_dec(v___y_3089_);
lean_dec_ref(v___y_3088_);
return v_res_3091_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1(lean_object* v_rf_3092_, uint8_t v_anyErased_3093_, lean_object* v_rsName_3094_, lean_object* v___y_3095_, lean_object* v___y_3096_){
_start:
{
lean_object* v___x_3098_; lean_object* v___f_3099_; lean_object* v___x_3100_; 
v___x_3098_ = lean_box(v_anyErased_3093_);
v___f_3099_ = lean_alloc_closure((void*)(lp_mathlib___private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1___lam__0___boxed), 3, 2);
lean_closure_set(v___f_3099_, 0, v_rf_3092_);
lean_closure_set(v___f_3099_, 1, v___x_3098_);
v___x_3100_ = lp_mathlib_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1_spec__2___redArg(v_rsName_3094_, v___f_3099_, v___y_3095_, v___y_3096_);
return v___x_3100_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1___boxed(lean_object* v_rf_3101_, lean_object* v_anyErased_3102_, lean_object* v_rsName_3103_, lean_object* v___y_3104_, lean_object* v___y_3105_, lean_object* v___y_3106_){
_start:
{
uint8_t v_anyErased_boxed_3107_; lean_object* v_res_3108_; 
v_anyErased_boxed_3107_ = lean_unbox(v_anyErased_3102_);
v_res_3108_ = lp_mathlib___private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1(v_rf_3101_, v_anyErased_boxed_3107_, v_rsName_3103_, v___y_3104_, v___y_3105_);
lean_dec(v___y_3105_);
lean_dec_ref(v___y_3104_);
return v_res_3108_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__6(lean_object* v_rf_3109_, lean_object* v_as_3110_, size_t v_i_3111_, size_t v_stop_3112_, uint8_t v_b_3113_, lean_object* v___y_3114_, lean_object* v___y_3115_){
_start:
{
uint8_t v___x_3117_; 
v___x_3117_ = lean_usize_dec_eq(v_i_3111_, v_stop_3112_);
if (v___x_3117_ == 0)
{
lean_object* v___x_3118_; lean_object* v___x_3119_; 
v___x_3118_ = lean_array_uget_borrowed(v_as_3110_, v_i_3111_);
lean_inc(v___x_3118_);
lean_inc_ref(v_rf_3109_);
v___x_3119_ = lp_mathlib___private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1(v_rf_3109_, v_b_3113_, v___x_3118_, v___y_3114_, v___y_3115_);
if (lean_obj_tag(v___x_3119_) == 0)
{
lean_object* v_a_3120_; size_t v___x_3121_; size_t v___x_3122_; uint8_t v___x_3123_; 
v_a_3120_ = lean_ctor_get(v___x_3119_, 0);
lean_inc(v_a_3120_);
lean_dec_ref_known(v___x_3119_, 1);
v___x_3121_ = ((size_t)1ULL);
v___x_3122_ = lean_usize_add(v_i_3111_, v___x_3121_);
v___x_3123_ = lean_unbox(v_a_3120_);
lean_dec(v_a_3120_);
v_i_3111_ = v___x_3122_;
v_b_3113_ = v___x_3123_;
goto _start;
}
else
{
lean_dec_ref(v_rf_3109_);
return v___x_3119_;
}
}
else
{
lean_object* v___x_3125_; lean_object* v___x_3126_; 
lean_dec_ref(v_rf_3109_);
v___x_3125_ = lean_box(v_b_3113_);
v___x_3126_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3126_, 0, v___x_3125_);
return v___x_3126_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__6___boxed(lean_object* v_rf_3127_, lean_object* v_as_3128_, lean_object* v_i_3129_, lean_object* v_stop_3130_, lean_object* v_b_3131_, lean_object* v___y_3132_, lean_object* v___y_3133_, lean_object* v___y_3134_){
_start:
{
size_t v_i_boxed_3135_; size_t v_stop_boxed_3136_; uint8_t v_b_boxed_3137_; lean_object* v_res_3138_; 
v_i_boxed_3135_ = lean_unbox_usize(v_i_3129_);
lean_dec(v_i_3129_);
v_stop_boxed_3136_ = lean_unbox_usize(v_stop_3130_);
lean_dec(v_stop_3130_);
v_b_boxed_3137_ = lean_unbox(v_b_3131_);
v_res_3138_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__6(v_rf_3127_, v_as_3128_, v_i_boxed_3135_, v_stop_boxed_3136_, v_b_boxed_3137_, v___y_3132_, v___y_3133_);
lean_dec(v___y_3133_);
lean_dec_ref(v___y_3132_);
lean_dec_ref(v_as_3128_);
return v_res_3138_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__5(lean_object* v_a_3139_, lean_object* v_a_3140_){
_start:
{
if (lean_obj_tag(v_a_3139_) == 0)
{
lean_object* v___x_3141_; 
v___x_3141_ = l_List_reverse___redArg(v_a_3140_);
return v___x_3141_;
}
else
{
lean_object* v_head_3142_; lean_object* v_tail_3143_; lean_object* v___x_3145_; uint8_t v_isShared_3146_; uint8_t v_isSharedCheck_3152_; 
v_head_3142_ = lean_ctor_get(v_a_3139_, 0);
v_tail_3143_ = lean_ctor_get(v_a_3139_, 1);
v_isSharedCheck_3152_ = !lean_is_exclusive(v_a_3139_);
if (v_isSharedCheck_3152_ == 0)
{
v___x_3145_ = v_a_3139_;
v_isShared_3146_ = v_isSharedCheck_3152_;
goto v_resetjp_3144_;
}
else
{
lean_inc(v_tail_3143_);
lean_inc(v_head_3142_);
lean_dec(v_a_3139_);
v___x_3145_ = lean_box(0);
v_isShared_3146_ = v_isSharedCheck_3152_;
goto v_resetjp_3144_;
}
v_resetjp_3144_:
{
lean_object* v___x_3147_; lean_object* v___x_3149_; 
v___x_3147_ = l_Lean_stringToMessageData(v_head_3142_);
if (v_isShared_3146_ == 0)
{
lean_ctor_set(v___x_3145_, 1, v_a_3140_);
lean_ctor_set(v___x_3145_, 0, v___x_3147_);
v___x_3149_ = v___x_3145_;
goto v_reusejp_3148_;
}
else
{
lean_object* v_reuseFailAlloc_3151_; 
v_reuseFailAlloc_3151_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3151_, 0, v___x_3147_);
lean_ctor_set(v_reuseFailAlloc_3151_, 1, v_a_3140_);
v___x_3149_ = v_reuseFailAlloc_3151_;
goto v_reusejp_3148_;
}
v_reusejp_3148_:
{
v_a_3139_ = v_tail_3143_;
v_a_3140_ = v___x_3149_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__2(lean_object* v_rf_3153_, uint8_t v_x_3154_, lean_object* v_x_3155_, lean_object* v___y_3156_, lean_object* v___y_3157_){
_start:
{
if (lean_obj_tag(v_x_3155_) == 0)
{
lean_object* v___x_3159_; lean_object* v___x_3160_; 
lean_dec_ref(v_rf_3153_);
v___x_3159_ = lean_box(v_x_3154_);
v___x_3160_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3160_, 0, v___x_3159_);
return v___x_3160_;
}
else
{
lean_object* v_key_3161_; lean_object* v_tail_3162_; lean_object* v___x_3163_; 
v_key_3161_ = lean_ctor_get(v_x_3155_, 0);
lean_inc(v_key_3161_);
v_tail_3162_ = lean_ctor_get(v_x_3155_, 2);
lean_inc(v_tail_3162_);
lean_dec_ref_known(v_x_3155_, 3);
lean_inc_ref(v_rf_3153_);
v___x_3163_ = lp_mathlib___private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1(v_rf_3153_, v_x_3154_, v_key_3161_, v___y_3156_, v___y_3157_);
if (lean_obj_tag(v___x_3163_) == 0)
{
lean_object* v_a_3164_; uint8_t v___x_3165_; 
v_a_3164_ = lean_ctor_get(v___x_3163_, 0);
lean_inc(v_a_3164_);
lean_dec_ref_known(v___x_3163_, 1);
v___x_3165_ = lean_unbox(v_a_3164_);
lean_dec(v_a_3164_);
v_x_3154_ = v___x_3165_;
v_x_3155_ = v_tail_3162_;
goto _start;
}
else
{
lean_dec(v_tail_3162_);
lean_dec_ref(v_rf_3153_);
return v___x_3163_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__2___boxed(lean_object* v_rf_3167_, lean_object* v_x_3168_, lean_object* v_x_3169_, lean_object* v___y_3170_, lean_object* v___y_3171_, lean_object* v___y_3172_){
_start:
{
uint8_t v_x_14338__boxed_3173_; lean_object* v_res_3174_; 
v_x_14338__boxed_3173_ = lean_unbox(v_x_3168_);
v_res_3174_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__2(v_rf_3167_, v_x_14338__boxed_3173_, v_x_3169_, v___y_3170_, v___y_3171_);
lean_dec(v___y_3171_);
lean_dec_ref(v___y_3170_);
return v_res_3174_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__3(lean_object* v_rf_3175_, lean_object* v_as_3176_, size_t v_i_3177_, size_t v_stop_3178_, uint8_t v_b_3179_, lean_object* v___y_3180_, lean_object* v___y_3181_){
_start:
{
uint8_t v___x_3183_; 
v___x_3183_ = lean_usize_dec_eq(v_i_3177_, v_stop_3178_);
if (v___x_3183_ == 0)
{
lean_object* v___x_3184_; lean_object* v___x_3185_; 
v___x_3184_ = lean_array_uget_borrowed(v_as_3176_, v_i_3177_);
lean_inc(v___x_3184_);
lean_inc_ref(v_rf_3175_);
v___x_3185_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__2(v_rf_3175_, v_b_3179_, v___x_3184_, v___y_3180_, v___y_3181_);
if (lean_obj_tag(v___x_3185_) == 0)
{
lean_object* v_a_3186_; size_t v___x_3187_; size_t v___x_3188_; uint8_t v___x_3189_; 
v_a_3186_ = lean_ctor_get(v___x_3185_, 0);
lean_inc(v_a_3186_);
lean_dec_ref_known(v___x_3185_, 1);
v___x_3187_ = ((size_t)1ULL);
v___x_3188_ = lean_usize_add(v_i_3177_, v___x_3187_);
v___x_3189_ = lean_unbox(v_a_3186_);
lean_dec(v_a_3186_);
v_i_3177_ = v___x_3188_;
v_b_3179_ = v___x_3189_;
goto _start;
}
else
{
lean_dec_ref(v_rf_3175_);
return v___x_3185_;
}
}
else
{
lean_object* v___x_3191_; lean_object* v___x_3192_; 
lean_dec_ref(v_rf_3175_);
v___x_3191_ = lean_box(v_b_3179_);
v___x_3192_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3192_, 0, v___x_3191_);
return v___x_3192_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__3___boxed(lean_object* v_rf_3193_, lean_object* v_as_3194_, lean_object* v_i_3195_, lean_object* v_stop_3196_, lean_object* v_b_3197_, lean_object* v___y_3198_, lean_object* v___y_3199_, lean_object* v___y_3200_){
_start:
{
size_t v_i_boxed_3201_; size_t v_stop_boxed_3202_; uint8_t v_b_boxed_3203_; lean_object* v_res_3204_; 
v_i_boxed_3201_ = lean_unbox_usize(v_i_3195_);
lean_dec(v_i_3195_);
v_stop_boxed_3202_ = lean_unbox_usize(v_stop_3196_);
lean_dec(v_stop_3196_);
v_b_boxed_3203_ = lean_unbox(v_b_3197_);
v_res_3204_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__3(v_rf_3193_, v_as_3194_, v_i_boxed_3201_, v_stop_boxed_3202_, v_b_boxed_3203_, v___y_3198_, v___y_3199_);
lean_dec(v___y_3199_);
lean_dec_ref(v___y_3198_);
lean_dec_ref(v_as_3194_);
return v_res_3204_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__4(uint8_t v_checkExists_3205_, size_t v_sz_3206_, size_t v_i_3207_, lean_object* v_bs_3208_){
_start:
{
uint8_t v___x_3209_; 
v___x_3209_ = lean_usize_dec_lt(v_i_3207_, v_sz_3206_);
if (v___x_3209_ == 0)
{
return v_bs_3208_;
}
else
{
lean_object* v_v_3210_; lean_object* v___x_3211_; lean_object* v_bs_x27_3212_; lean_object* v___x_3213_; size_t v___x_3214_; size_t v___x_3215_; lean_object* v___x_3216_; 
v_v_3210_ = lean_array_uget(v_bs_3208_, v_i_3207_);
v___x_3211_ = lean_unsigned_to_nat(0u);
v_bs_x27_3212_ = lean_array_uset(v_bs_3208_, v_i_3207_, v___x_3211_);
v___x_3213_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_v_3210_, v_checkExists_3205_);
v___x_3214_ = ((size_t)1ULL);
v___x_3215_ = lean_usize_add(v_i_3207_, v___x_3214_);
v___x_3216_ = lean_array_uset(v_bs_x27_3212_, v_i_3207_, v___x_3213_);
v_i_3207_ = v___x_3215_;
v_bs_3208_ = v___x_3216_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__4___boxed(lean_object* v_checkExists_3218_, lean_object* v_sz_3219_, lean_object* v_i_3220_, lean_object* v_bs_3221_){
_start:
{
uint8_t v_checkExists_boxed_3222_; size_t v_sz_boxed_3223_; size_t v_i_boxed_3224_; lean_object* v_res_3225_; 
v_checkExists_boxed_3222_ = lean_unbox(v_checkExists_3218_);
v_sz_boxed_3223_ = lean_unbox_usize(v_sz_3219_);
lean_dec(v_sz_3219_);
v_i_boxed_3224_ = lean_unbox_usize(v_i_3220_);
lean_dec(v_i_3220_);
v_res_3225_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__4(v_checkExists_boxed_3222_, v_sz_boxed_3223_, v_i_boxed_3224_, v_bs_3221_);
return v_res_3225_;
}
}
static lean_object* _init_lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__1(void){
_start:
{
lean_object* v___x_3227_; lean_object* v___x_3228_; 
v___x_3227_ = ((lean_object*)(lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__0));
v___x_3228_ = l_Lean_stringToMessageData(v___x_3227_);
return v___x_3228_;
}
}
static lean_object* _init_lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__3(void){
_start:
{
lean_object* v___x_3230_; lean_object* v___x_3231_; 
v___x_3230_ = ((lean_object*)(lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__2));
v___x_3231_ = l_Lean_stringToMessageData(v___x_3230_);
return v___x_3231_;
}
}
static lean_object* _init_lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__5(void){
_start:
{
lean_object* v___x_3233_; lean_object* v___x_3234_; 
v___x_3233_ = ((lean_object*)(lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__4));
v___x_3234_ = l_Lean_stringToMessageData(v___x_3233_);
return v___x_3234_;
}
}
static lean_object* _init_lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__7(void){
_start:
{
lean_object* v___x_3236_; lean_object* v___x_3237_; 
v___x_3236_ = ((lean_object*)(lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__6));
v___x_3237_ = l_Lean_stringToMessageData(v___x_3236_);
return v___x_3237_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0(lean_object* v_rsf_3238_, lean_object* v_rf_3239_, uint8_t v_checkExists_3240_, lean_object* v___y_3241_, lean_object* v___y_3242_){
_start:
{
uint8_t v_anyErased_3251_; lean_object* v___y_3252_; lean_object* v___y_3253_; lean_object* v___x_3261_; 
v___x_3261_ = lp_aesop_Aesop_RuleSetNameFilter_matchedRuleSetNames(v_rsf_3238_);
if (lean_obj_tag(v___x_3261_) == 0)
{
lean_object* v___x_3262_; 
v___x_3262_ = lp_aesop_Aesop_getDeclaredRuleSets();
if (lean_obj_tag(v___x_3262_) == 0)
{
lean_object* v_a_3263_; lean_object* v___x_3265_; uint8_t v_isShared_3266_; uint8_t v_isSharedCheck_3317_; 
v_a_3263_ = lean_ctor_get(v___x_3262_, 0);
v_isSharedCheck_3317_ = !lean_is_exclusive(v___x_3262_);
if (v_isSharedCheck_3317_ == 0)
{
v___x_3265_ = v___x_3262_;
v_isShared_3266_ = v_isSharedCheck_3317_;
goto v_resetjp_3264_;
}
else
{
lean_inc(v_a_3263_);
lean_dec(v___x_3262_);
v___x_3265_ = lean_box(0);
v_isShared_3266_ = v_isSharedCheck_3317_;
goto v_resetjp_3264_;
}
v_resetjp_3264_:
{
lean_object* v_buckets_3267_; lean_object* v___x_3269_; uint8_t v_isShared_3270_; uint8_t v_isSharedCheck_3315_; 
v_buckets_3267_ = lean_ctor_get(v_a_3263_, 1);
v_isSharedCheck_3315_ = !lean_is_exclusive(v_a_3263_);
if (v_isSharedCheck_3315_ == 0)
{
lean_object* v_unused_3316_; 
v_unused_3316_ = lean_ctor_get(v_a_3263_, 0);
lean_dec(v_unused_3316_);
v___x_3269_ = v_a_3263_;
v_isShared_3270_ = v_isSharedCheck_3315_;
goto v_resetjp_3268_;
}
else
{
lean_inc(v_buckets_3267_);
lean_dec(v_a_3263_);
v___x_3269_ = lean_box(0);
v_isShared_3270_ = v_isSharedCheck_3315_;
goto v_resetjp_3268_;
}
v_resetjp_3268_:
{
lean_object* v___x_3271_; lean_object* v___x_3272_; uint8_t v___x_3273_; 
v___x_3271_ = lean_unsigned_to_nat(0u);
v___x_3272_ = lean_array_get_size(v_buckets_3267_);
v___x_3273_ = lean_nat_dec_lt(v___x_3271_, v___x_3272_);
if (v___x_3273_ == 0)
{
lean_dec_ref(v_buckets_3267_);
if (v_checkExists_3240_ == 0)
{
lean_object* v___x_3274_; lean_object* v___x_3276_; 
lean_del_object(v___x_3269_);
lean_dec_ref(v_rf_3239_);
v___x_3274_ = lean_box(0);
if (v_isShared_3266_ == 0)
{
lean_ctor_set(v___x_3265_, 0, v___x_3274_);
v___x_3276_ = v___x_3265_;
goto v_reusejp_3275_;
}
else
{
lean_object* v_reuseFailAlloc_3277_; 
v_reuseFailAlloc_3277_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3277_, 0, v___x_3274_);
v___x_3276_ = v_reuseFailAlloc_3277_;
goto v_reusejp_3275_;
}
v_reusejp_3275_:
{
return v___x_3276_;
}
}
else
{
lean_object* v_name_3278_; lean_object* v___x_3279_; lean_object* v___x_3280_; lean_object* v___x_3282_; 
lean_del_object(v___x_3265_);
v_name_3278_ = lean_ctor_get(v_rf_3239_, 0);
lean_inc(v_name_3278_);
lean_dec_ref(v_rf_3239_);
v___x_3279_ = lean_obj_once(&lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__1, &lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__1_once, _init_lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__1);
v___x_3280_ = l_Lean_MessageData_ofName(v_name_3278_);
if (v_isShared_3270_ == 0)
{
lean_ctor_set_tag(v___x_3269_, 7);
lean_ctor_set(v___x_3269_, 1, v___x_3280_);
lean_ctor_set(v___x_3269_, 0, v___x_3279_);
v___x_3282_ = v___x_3269_;
goto v_reusejp_3281_;
}
else
{
lean_object* v_reuseFailAlloc_3286_; 
v_reuseFailAlloc_3286_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3286_, 0, v___x_3279_);
lean_ctor_set(v_reuseFailAlloc_3286_, 1, v___x_3280_);
v___x_3282_ = v_reuseFailAlloc_3286_;
goto v_reusejp_3281_;
}
v_reusejp_3281_:
{
lean_object* v___x_3283_; lean_object* v___x_3284_; lean_object* v___x_3285_; 
v___x_3283_ = lean_obj_once(&lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__3, &lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__3_once, _init_lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__3);
v___x_3284_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3284_, 0, v___x_3282_);
lean_ctor_set(v___x_3284_, 1, v___x_3283_);
v___x_3285_ = lp_mathlib_Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__0___redArg(v___x_3284_, v___y_3241_, v___y_3242_);
return v___x_3285_;
}
}
}
else
{
uint8_t v___x_3287_; uint8_t v___x_3288_; 
lean_del_object(v___x_3269_);
lean_del_object(v___x_3265_);
v___x_3287_ = 0;
v___x_3288_ = lean_nat_dec_le(v___x_3272_, v___x_3272_);
if (v___x_3288_ == 0)
{
if (v___x_3273_ == 0)
{
lean_dec_ref(v_buckets_3267_);
v_anyErased_3251_ = v___x_3287_;
v___y_3252_ = v___y_3241_;
v___y_3253_ = v___y_3242_;
goto v___jp_3250_;
}
else
{
size_t v___x_3289_; size_t v___x_3290_; lean_object* v___x_3291_; 
v___x_3289_ = ((size_t)0ULL);
v___x_3290_ = lean_usize_of_nat(v___x_3272_);
lean_inc_ref(v_rf_3239_);
v___x_3291_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__3(v_rf_3239_, v_buckets_3267_, v___x_3289_, v___x_3290_, v___x_3287_, v___y_3241_, v___y_3242_);
lean_dec_ref(v_buckets_3267_);
if (lean_obj_tag(v___x_3291_) == 0)
{
lean_object* v_a_3292_; uint8_t v___x_3293_; 
v_a_3292_ = lean_ctor_get(v___x_3291_, 0);
lean_inc(v_a_3292_);
lean_dec_ref_known(v___x_3291_, 1);
v___x_3293_ = lean_unbox(v_a_3292_);
lean_dec(v_a_3292_);
v_anyErased_3251_ = v___x_3293_;
v___y_3252_ = v___y_3241_;
v___y_3253_ = v___y_3242_;
goto v___jp_3250_;
}
else
{
lean_object* v_a_3294_; lean_object* v___x_3296_; uint8_t v_isShared_3297_; uint8_t v_isSharedCheck_3301_; 
lean_dec_ref(v_rf_3239_);
v_a_3294_ = lean_ctor_get(v___x_3291_, 0);
v_isSharedCheck_3301_ = !lean_is_exclusive(v___x_3291_);
if (v_isSharedCheck_3301_ == 0)
{
v___x_3296_ = v___x_3291_;
v_isShared_3297_ = v_isSharedCheck_3301_;
goto v_resetjp_3295_;
}
else
{
lean_inc(v_a_3294_);
lean_dec(v___x_3291_);
v___x_3296_ = lean_box(0);
v_isShared_3297_ = v_isSharedCheck_3301_;
goto v_resetjp_3295_;
}
v_resetjp_3295_:
{
lean_object* v___x_3299_; 
if (v_isShared_3297_ == 0)
{
v___x_3299_ = v___x_3296_;
goto v_reusejp_3298_;
}
else
{
lean_object* v_reuseFailAlloc_3300_; 
v_reuseFailAlloc_3300_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3300_, 0, v_a_3294_);
v___x_3299_ = v_reuseFailAlloc_3300_;
goto v_reusejp_3298_;
}
v_reusejp_3298_:
{
return v___x_3299_;
}
}
}
}
}
else
{
size_t v___x_3302_; size_t v___x_3303_; lean_object* v___x_3304_; 
v___x_3302_ = ((size_t)0ULL);
v___x_3303_ = lean_usize_of_nat(v___x_3272_);
lean_inc_ref(v_rf_3239_);
v___x_3304_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__3(v_rf_3239_, v_buckets_3267_, v___x_3302_, v___x_3303_, v___x_3287_, v___y_3241_, v___y_3242_);
lean_dec_ref(v_buckets_3267_);
if (lean_obj_tag(v___x_3304_) == 0)
{
lean_object* v_a_3305_; uint8_t v___x_3306_; 
v_a_3305_ = lean_ctor_get(v___x_3304_, 0);
lean_inc(v_a_3305_);
lean_dec_ref_known(v___x_3304_, 1);
v___x_3306_ = lean_unbox(v_a_3305_);
lean_dec(v_a_3305_);
v_anyErased_3251_ = v___x_3306_;
v___y_3252_ = v___y_3241_;
v___y_3253_ = v___y_3242_;
goto v___jp_3250_;
}
else
{
lean_object* v_a_3307_; lean_object* v___x_3309_; uint8_t v_isShared_3310_; uint8_t v_isSharedCheck_3314_; 
lean_dec_ref(v_rf_3239_);
v_a_3307_ = lean_ctor_get(v___x_3304_, 0);
v_isSharedCheck_3314_ = !lean_is_exclusive(v___x_3304_);
if (v_isSharedCheck_3314_ == 0)
{
v___x_3309_ = v___x_3304_;
v_isShared_3310_ = v_isSharedCheck_3314_;
goto v_resetjp_3308_;
}
else
{
lean_inc(v_a_3307_);
lean_dec(v___x_3304_);
v___x_3309_ = lean_box(0);
v_isShared_3310_ = v_isSharedCheck_3314_;
goto v_resetjp_3308_;
}
v_resetjp_3308_:
{
lean_object* v___x_3312_; 
if (v_isShared_3310_ == 0)
{
v___x_3312_ = v___x_3309_;
goto v_reusejp_3311_;
}
else
{
lean_object* v_reuseFailAlloc_3313_; 
v_reuseFailAlloc_3313_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3313_, 0, v_a_3307_);
v___x_3312_ = v_reuseFailAlloc_3313_;
goto v_reusejp_3311_;
}
v_reusejp_3311_:
{
return v___x_3312_;
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
lean_object* v_a_3318_; lean_object* v___x_3320_; uint8_t v_isShared_3321_; uint8_t v_isSharedCheck_3330_; 
lean_dec_ref(v_rf_3239_);
v_a_3318_ = lean_ctor_get(v___x_3262_, 0);
v_isSharedCheck_3330_ = !lean_is_exclusive(v___x_3262_);
if (v_isSharedCheck_3330_ == 0)
{
v___x_3320_ = v___x_3262_;
v_isShared_3321_ = v_isSharedCheck_3330_;
goto v_resetjp_3319_;
}
else
{
lean_inc(v_a_3318_);
lean_dec(v___x_3262_);
v___x_3320_ = lean_box(0);
v_isShared_3321_ = v_isSharedCheck_3330_;
goto v_resetjp_3319_;
}
v_resetjp_3319_:
{
lean_object* v_ref_3322_; lean_object* v___x_3323_; lean_object* v___x_3324_; lean_object* v___x_3325_; lean_object* v___x_3326_; lean_object* v___x_3328_; 
v_ref_3322_ = lean_ctor_get(v___y_3241_, 5);
v___x_3323_ = lean_io_error_to_string(v_a_3318_);
v___x_3324_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_3324_, 0, v___x_3323_);
v___x_3325_ = l_Lean_MessageData_ofFormat(v___x_3324_);
lean_inc(v_ref_3322_);
v___x_3326_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3326_, 0, v_ref_3322_);
lean_ctor_set(v___x_3326_, 1, v___x_3325_);
if (v_isShared_3321_ == 0)
{
lean_ctor_set(v___x_3320_, 0, v___x_3326_);
v___x_3328_ = v___x_3320_;
goto v_reusejp_3327_;
}
else
{
lean_object* v_reuseFailAlloc_3329_; 
v_reuseFailAlloc_3329_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3329_, 0, v___x_3326_);
v___x_3328_ = v_reuseFailAlloc_3329_;
goto v_reusejp_3327_;
}
v_reusejp_3327_:
{
return v___x_3328_;
}
}
}
}
else
{
lean_object* v_val_3331_; lean_object* v___x_3333_; uint8_t v_isShared_3334_; uint8_t v_isSharedCheck_3408_; 
v_val_3331_ = lean_ctor_get(v___x_3261_, 0);
v_isSharedCheck_3408_ = !lean_is_exclusive(v___x_3261_);
if (v_isSharedCheck_3408_ == 0)
{
v___x_3333_ = v___x_3261_;
v_isShared_3334_ = v_isSharedCheck_3408_;
goto v_resetjp_3332_;
}
else
{
lean_inc(v_val_3331_);
lean_dec(v___x_3261_);
v___x_3333_ = lean_box(0);
v_isShared_3334_ = v_isSharedCheck_3408_;
goto v_resetjp_3332_;
}
v_resetjp_3332_:
{
uint8_t v_anyErased_3336_; lean_object* v___y_3337_; lean_object* v___y_3338_; lean_object* v___x_3356_; lean_object* v___x_3357_; uint8_t v___x_3358_; 
v___x_3356_ = lean_unsigned_to_nat(0u);
v___x_3357_ = lean_array_get_size(v_val_3331_);
v___x_3358_ = lean_nat_dec_lt(v___x_3356_, v___x_3357_);
if (v___x_3358_ == 0)
{
if (v_checkExists_3240_ == 0)
{
lean_object* v___x_3359_; lean_object* v___x_3361_; 
lean_dec(v_val_3331_);
lean_dec_ref(v_rf_3239_);
v___x_3359_ = lean_box(0);
if (v_isShared_3334_ == 0)
{
lean_ctor_set_tag(v___x_3333_, 0);
lean_ctor_set(v___x_3333_, 0, v___x_3359_);
v___x_3361_ = v___x_3333_;
goto v_reusejp_3360_;
}
else
{
lean_object* v_reuseFailAlloc_3362_; 
v_reuseFailAlloc_3362_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3362_, 0, v___x_3359_);
v___x_3361_ = v_reuseFailAlloc_3362_;
goto v_reusejp_3360_;
}
v_reusejp_3360_:
{
return v___x_3361_;
}
}
else
{
lean_object* v_name_3363_; lean_object* v___x_3364_; lean_object* v___x_3365_; lean_object* v___x_3366_; lean_object* v___x_3367_; lean_object* v___x_3368_; size_t v_sz_3369_; size_t v___x_3370_; lean_object* v___x_3371_; lean_object* v___x_3372_; lean_object* v___x_3373_; lean_object* v___x_3374_; lean_object* v___x_3375_; lean_object* v___x_3376_; lean_object* v___x_3377_; lean_object* v___x_3378_; lean_object* v___x_3379_; 
lean_del_object(v___x_3333_);
v_name_3363_ = lean_ctor_get(v_rf_3239_, 0);
lean_inc(v_name_3363_);
lean_dec_ref(v_rf_3239_);
v___x_3364_ = lean_obj_once(&lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__1, &lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__1_once, _init_lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__1);
v___x_3365_ = l_Lean_MessageData_ofName(v_name_3363_);
v___x_3366_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3366_, 0, v___x_3364_);
lean_ctor_set(v___x_3366_, 1, v___x_3365_);
v___x_3367_ = lean_obj_once(&lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__5, &lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__5_once, _init_lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__5);
v___x_3368_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3368_, 0, v___x_3366_);
lean_ctor_set(v___x_3368_, 1, v___x_3367_);
v_sz_3369_ = lean_array_size(v_val_3331_);
v___x_3370_ = ((size_t)0ULL);
v___x_3371_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__4(v_checkExists_3240_, v_sz_3369_, v___x_3370_, v_val_3331_);
v___x_3372_ = lean_array_to_list(v___x_3371_);
v___x_3373_ = lean_box(0);
v___x_3374_ = lp_mathlib_List_mapTR_loop___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__5(v___x_3372_, v___x_3373_);
v___x_3375_ = l_Lean_MessageData_ofList(v___x_3374_);
v___x_3376_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3376_, 0, v___x_3368_);
lean_ctor_set(v___x_3376_, 1, v___x_3375_);
v___x_3377_ = lean_obj_once(&lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__7, &lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__7_once, _init_lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__7);
v___x_3378_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3378_, 0, v___x_3376_);
lean_ctor_set(v___x_3378_, 1, v___x_3377_);
v___x_3379_ = lp_mathlib_Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__0___redArg(v___x_3378_, v___y_3241_, v___y_3242_);
return v___x_3379_;
}
}
else
{
uint8_t v___x_3380_; uint8_t v___x_3381_; 
lean_del_object(v___x_3333_);
v___x_3380_ = 0;
v___x_3381_ = lean_nat_dec_le(v___x_3357_, v___x_3357_);
if (v___x_3381_ == 0)
{
if (v___x_3358_ == 0)
{
v_anyErased_3336_ = v___x_3380_;
v___y_3337_ = v___y_3241_;
v___y_3338_ = v___y_3242_;
goto v___jp_3335_;
}
else
{
size_t v___x_3382_; size_t v___x_3383_; lean_object* v___x_3384_; 
v___x_3382_ = ((size_t)0ULL);
v___x_3383_ = lean_usize_of_nat(v___x_3357_);
lean_inc_ref(v_rf_3239_);
v___x_3384_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__6(v_rf_3239_, v_val_3331_, v___x_3382_, v___x_3383_, v___x_3380_, v___y_3241_, v___y_3242_);
if (lean_obj_tag(v___x_3384_) == 0)
{
lean_object* v_a_3385_; uint8_t v___x_3386_; 
v_a_3385_ = lean_ctor_get(v___x_3384_, 0);
lean_inc(v_a_3385_);
lean_dec_ref_known(v___x_3384_, 1);
v___x_3386_ = lean_unbox(v_a_3385_);
lean_dec(v_a_3385_);
v_anyErased_3336_ = v___x_3386_;
v___y_3337_ = v___y_3241_;
v___y_3338_ = v___y_3242_;
goto v___jp_3335_;
}
else
{
lean_object* v_a_3387_; lean_object* v___x_3389_; uint8_t v_isShared_3390_; uint8_t v_isSharedCheck_3394_; 
lean_dec(v_val_3331_);
lean_dec_ref(v_rf_3239_);
v_a_3387_ = lean_ctor_get(v___x_3384_, 0);
v_isSharedCheck_3394_ = !lean_is_exclusive(v___x_3384_);
if (v_isSharedCheck_3394_ == 0)
{
v___x_3389_ = v___x_3384_;
v_isShared_3390_ = v_isSharedCheck_3394_;
goto v_resetjp_3388_;
}
else
{
lean_inc(v_a_3387_);
lean_dec(v___x_3384_);
v___x_3389_ = lean_box(0);
v_isShared_3390_ = v_isSharedCheck_3394_;
goto v_resetjp_3388_;
}
v_resetjp_3388_:
{
lean_object* v___x_3392_; 
if (v_isShared_3390_ == 0)
{
v___x_3392_ = v___x_3389_;
goto v_reusejp_3391_;
}
else
{
lean_object* v_reuseFailAlloc_3393_; 
v_reuseFailAlloc_3393_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3393_, 0, v_a_3387_);
v___x_3392_ = v_reuseFailAlloc_3393_;
goto v_reusejp_3391_;
}
v_reusejp_3391_:
{
return v___x_3392_;
}
}
}
}
}
else
{
size_t v___x_3395_; size_t v___x_3396_; lean_object* v___x_3397_; 
v___x_3395_ = ((size_t)0ULL);
v___x_3396_ = lean_usize_of_nat(v___x_3357_);
lean_inc_ref(v_rf_3239_);
v___x_3397_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__6(v_rf_3239_, v_val_3331_, v___x_3395_, v___x_3396_, v___x_3380_, v___y_3241_, v___y_3242_);
if (lean_obj_tag(v___x_3397_) == 0)
{
lean_object* v_a_3398_; uint8_t v___x_3399_; 
v_a_3398_ = lean_ctor_get(v___x_3397_, 0);
lean_inc(v_a_3398_);
lean_dec_ref_known(v___x_3397_, 1);
v___x_3399_ = lean_unbox(v_a_3398_);
lean_dec(v_a_3398_);
v_anyErased_3336_ = v___x_3399_;
v___y_3337_ = v___y_3241_;
v___y_3338_ = v___y_3242_;
goto v___jp_3335_;
}
else
{
lean_object* v_a_3400_; lean_object* v___x_3402_; uint8_t v_isShared_3403_; uint8_t v_isSharedCheck_3407_; 
lean_dec(v_val_3331_);
lean_dec_ref(v_rf_3239_);
v_a_3400_ = lean_ctor_get(v___x_3397_, 0);
v_isSharedCheck_3407_ = !lean_is_exclusive(v___x_3397_);
if (v_isSharedCheck_3407_ == 0)
{
v___x_3402_ = v___x_3397_;
v_isShared_3403_ = v_isSharedCheck_3407_;
goto v_resetjp_3401_;
}
else
{
lean_inc(v_a_3400_);
lean_dec(v___x_3397_);
v___x_3402_ = lean_box(0);
v_isShared_3403_ = v_isSharedCheck_3407_;
goto v_resetjp_3401_;
}
v_resetjp_3401_:
{
lean_object* v___x_3405_; 
if (v_isShared_3403_ == 0)
{
v___x_3405_ = v___x_3402_;
goto v_reusejp_3404_;
}
else
{
lean_object* v_reuseFailAlloc_3406_; 
v_reuseFailAlloc_3406_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3406_, 0, v_a_3400_);
v___x_3405_ = v_reuseFailAlloc_3406_;
goto v_reusejp_3404_;
}
v_reusejp_3404_:
{
return v___x_3405_;
}
}
}
}
}
v___jp_3335_:
{
if (v_checkExists_3240_ == 0)
{
lean_dec(v_val_3331_);
lean_dec_ref(v_rf_3239_);
goto v___jp_3244_;
}
else
{
if (v_anyErased_3336_ == 0)
{
lean_object* v_name_3339_; lean_object* v___x_3340_; lean_object* v___x_3341_; lean_object* v___x_3342_; lean_object* v___x_3343_; lean_object* v___x_3344_; size_t v_sz_3345_; size_t v___x_3346_; lean_object* v___x_3347_; lean_object* v___x_3348_; lean_object* v___x_3349_; lean_object* v___x_3350_; lean_object* v___x_3351_; lean_object* v___x_3352_; lean_object* v___x_3353_; lean_object* v___x_3354_; lean_object* v___x_3355_; 
v_name_3339_ = lean_ctor_get(v_rf_3239_, 0);
lean_inc(v_name_3339_);
lean_dec_ref(v_rf_3239_);
v___x_3340_ = lean_obj_once(&lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__1, &lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__1_once, _init_lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__1);
v___x_3341_ = l_Lean_MessageData_ofName(v_name_3339_);
v___x_3342_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3342_, 0, v___x_3340_);
lean_ctor_set(v___x_3342_, 1, v___x_3341_);
v___x_3343_ = lean_obj_once(&lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__5, &lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__5_once, _init_lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__5);
v___x_3344_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3344_, 0, v___x_3342_);
lean_ctor_set(v___x_3344_, 1, v___x_3343_);
v_sz_3345_ = lean_array_size(v_val_3331_);
v___x_3346_ = ((size_t)0ULL);
v___x_3347_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__4(v_checkExists_3240_, v_sz_3345_, v___x_3346_, v_val_3331_);
v___x_3348_ = lean_array_to_list(v___x_3347_);
v___x_3349_ = lean_box(0);
v___x_3350_ = lp_mathlib_List_mapTR_loop___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__5(v___x_3348_, v___x_3349_);
v___x_3351_ = l_Lean_MessageData_ofList(v___x_3350_);
v___x_3352_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3352_, 0, v___x_3344_);
lean_ctor_set(v___x_3352_, 1, v___x_3351_);
v___x_3353_ = lean_obj_once(&lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__7, &lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__7_once, _init_lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__7);
v___x_3354_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3354_, 0, v___x_3352_);
lean_ctor_set(v___x_3354_, 1, v___x_3353_);
v___x_3355_ = lp_mathlib_Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__0___redArg(v___x_3354_, v___y_3337_, v___y_3338_);
return v___x_3355_;
}
else
{
lean_dec(v_val_3331_);
lean_dec_ref(v_rf_3239_);
goto v___jp_3244_;
}
}
}
}
}
v___jp_3244_:
{
lean_object* v___x_3245_; lean_object* v___x_3246_; 
v___x_3245_ = lean_box(0);
v___x_3246_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3246_, 0, v___x_3245_);
return v___x_3246_;
}
v___jp_3247_:
{
lean_object* v___x_3248_; lean_object* v___x_3249_; 
v___x_3248_ = lean_box(0);
v___x_3249_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3249_, 0, v___x_3248_);
return v___x_3249_;
}
v___jp_3250_:
{
if (v_checkExists_3240_ == 0)
{
lean_dec_ref(v_rf_3239_);
goto v___jp_3247_;
}
else
{
if (v_anyErased_3251_ == 0)
{
lean_object* v_name_3254_; lean_object* v___x_3255_; lean_object* v___x_3256_; lean_object* v___x_3257_; lean_object* v___x_3258_; lean_object* v___x_3259_; lean_object* v___x_3260_; 
v_name_3254_ = lean_ctor_get(v_rf_3239_, 0);
lean_inc(v_name_3254_);
lean_dec_ref(v_rf_3239_);
v___x_3255_ = lean_obj_once(&lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__1, &lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__1_once, _init_lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__1);
v___x_3256_ = l_Lean_MessageData_ofName(v_name_3254_);
v___x_3257_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3257_, 0, v___x_3255_);
lean_ctor_set(v___x_3257_, 1, v___x_3256_);
v___x_3258_ = lean_obj_once(&lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__3, &lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__3_once, _init_lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__3);
v___x_3259_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3259_, 0, v___x_3257_);
lean_ctor_set(v___x_3259_, 1, v___x_3258_);
v___x_3260_ = lp_mathlib_Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__0___redArg(v___x_3259_, v___y_3252_, v___y_3253_);
return v___x_3260_;
}
else
{
lean_dec_ref(v_rf_3239_);
goto v___jp_3247_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___boxed(lean_object* v_rsf_3409_, lean_object* v_rf_3410_, lean_object* v_checkExists_3411_, lean_object* v___y_3412_, lean_object* v___y_3413_, lean_object* v___y_3414_){
_start:
{
uint8_t v_checkExists_boxed_3415_; lean_object* v_res_3416_; 
v_checkExists_boxed_3415_ = lean_unbox(v_checkExists_3411_);
v_res_3416_ = lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0(v_rsf_3409_, v_rf_3410_, v_checkExists_boxed_3415_, v___y_3412_, v___y_3413_);
lean_dec(v___y_3413_);
lean_dec_ref(v___y_3412_);
return v_res_3416_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__0_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_(lean_object* v_decl_3419_, lean_object* v___y_3420_, lean_object* v___y_3421_){
_start:
{
uint8_t v___x_3423_; lean_object* v___x_3424_; lean_object* v_ruleFilter_3425_; lean_object* v___x_3426_; uint8_t v___x_3427_; lean_object* v___x_3428_; 
v___x_3423_ = 0;
v___x_3424_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__0___closed__0_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_));
v_ruleFilter_3425_ = lean_alloc_ctor(0, 3, 1);
lean_ctor_set(v_ruleFilter_3425_, 0, v_decl_3419_);
lean_ctor_set(v_ruleFilter_3425_, 1, v___x_3424_);
lean_ctor_set(v_ruleFilter_3425_, 2, v___x_3424_);
lean_ctor_set_uint8(v_ruleFilter_3425_, sizeof(void*)*3, v___x_3423_);
v___x_3426_ = lp_aesop_Aesop_RuleSetNameFilter_all;
v___x_3427_ = 1;
v___x_3428_ = lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0(v___x_3426_, v_ruleFilter_3425_, v___x_3427_, v___y_3420_, v___y_3421_);
return v___x_3428_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__0_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2____boxed(lean_object* v_decl_3429_, lean_object* v___y_3430_, lean_object* v___y_3431_, lean_object* v___y_3432_){
_start:
{
lean_object* v_res_3433_; 
v_res_3433_ = lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__0_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_(v_decl_3429_, v___y_3430_, v___y_3431_);
lean_dec(v___y_3431_);
lean_dec_ref(v___y_3430_);
return v_res_3433_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__1_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_(lean_object* v_x_3434_){
_start:
{
uint8_t v___x_3435_; 
v___x_3435_ = 0;
return v___x_3435_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__1_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2____boxed(lean_object* v_x_3436_){
_start:
{
uint8_t v_res_3437_; lean_object* v_r_3438_; 
v_res_3437_ = lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__1_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_(v_x_3436_);
lean_dec(v_x_3436_);
v_r_3438_ = lean_box(v_res_3437_);
return v_r_3438_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__3_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_(lean_object* v_decl_3439_, lean_object* v___y_3440_, lean_object* v___y_3441_, lean_object* v___y_3442_, lean_object* v___y_3443_, lean_object* v___y_3444_, lean_object* v___y_3445_){
_start:
{
lean_object* v___x_3447_; 
v___x_3447_ = lp_mathlib_Mathlib_Tactic_Bound_declPriority(v_decl_3439_, v___y_3442_, v___y_3443_, v___y_3444_, v___y_3445_);
return v___x_3447_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__3_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2____boxed(lean_object* v_decl_3448_, lean_object* v___y_3449_, lean_object* v___y_3450_, lean_object* v___y_3451_, lean_object* v___y_3452_, lean_object* v___y_3453_, lean_object* v___y_3454_, lean_object* v___y_3455_){
_start:
{
lean_object* v_res_3456_; 
v_res_3456_ = lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__3_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_(v_decl_3448_, v___y_3449_, v___y_3450_, v___y_3451_, v___y_3452_, v___y_3453_, v___y_3454_);
lean_dec(v___y_3454_);
lean_dec_ref(v___y_3453_);
lean_dec(v___y_3452_);
lean_dec_ref(v___y_3451_);
lean_dec(v___y_3450_);
lean_dec_ref(v___y_3449_);
return v_res_3456_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Aesop_Frontend_getGlobalRuleSetFromData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__13___redArg(lean_object* v_ext_3457_, lean_object* v_simpExt_3458_, lean_object* v_simprocExt_3459_, lean_object* v___y_3460_){
_start:
{
lean_object* v___x_3462_; lean_object* v_ext_3463_; lean_object* v_toEnvExtension_3464_; lean_object* v_ext_3465_; lean_object* v_toEnvExtension_3466_; lean_object* v_ext_3467_; lean_object* v_toEnvExtension_3468_; lean_object* v_env_3469_; lean_object* v_asyncMode_3470_; lean_object* v_asyncMode_3471_; lean_object* v_asyncMode_3472_; lean_object* v___x_3473_; lean_object* v_base_3474_; lean_object* v___x_3475_; lean_object* v___x_3476_; lean_object* v_simpTheorems_3477_; lean_object* v_simprocs_3478_; lean_object* v___x_3479_; lean_object* v___x_3480_; 
v___x_3462_ = lean_st_ref_get(v___y_3460_);
v_ext_3463_ = lean_ctor_get(v_ext_3457_, 1);
v_toEnvExtension_3464_ = lean_ctor_get(v_ext_3463_, 0);
v_ext_3465_ = lean_ctor_get(v_simpExt_3458_, 1);
v_toEnvExtension_3466_ = lean_ctor_get(v_ext_3465_, 0);
v_ext_3467_ = lean_ctor_get(v_simprocExt_3459_, 1);
v_toEnvExtension_3468_ = lean_ctor_get(v_ext_3467_, 0);
v_env_3469_ = lean_ctor_get(v___x_3462_, 0);
lean_inc_ref_n(v_env_3469_, 3);
lean_dec(v___x_3462_);
v_asyncMode_3470_ = lean_ctor_get(v_toEnvExtension_3464_, 2);
v_asyncMode_3471_ = lean_ctor_get(v_toEnvExtension_3466_, 2);
v_asyncMode_3472_ = lean_ctor_get(v_toEnvExtension_3468_, 2);
v___x_3473_ = lp_aesop_Aesop_instInhabitedBaseRuleSet_default;
v_base_3474_ = l_Lean_ScopedEnvExtension_getState___redArg(v___x_3473_, v_ext_3457_, v_env_3469_, v_asyncMode_3470_);
v___x_3475_ = l_Lean_Meta_instInhabitedSimpTheorems_default;
v___x_3476_ = l_Lean_Meta_Simp_instInhabitedSimprocs_default;
v_simpTheorems_3477_ = l_Lean_ScopedEnvExtension_getState___redArg(v___x_3475_, v_simpExt_3458_, v_env_3469_, v_asyncMode_3471_);
v_simprocs_3478_ = l_Lean_ScopedEnvExtension_getState___redArg(v___x_3476_, v_simprocExt_3459_, v_env_3469_, v_asyncMode_3472_);
v___x_3479_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_3479_, 0, v_base_3474_);
lean_ctor_set(v___x_3479_, 1, v_simpTheorems_3477_);
lean_ctor_set(v___x_3479_, 2, v_simprocs_3478_);
v___x_3480_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3480_, 0, v___x_3479_);
return v___x_3480_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Aesop_Frontend_getGlobalRuleSetFromData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__13___redArg___boxed(lean_object* v_ext_3481_, lean_object* v_simpExt_3482_, lean_object* v_simprocExt_3483_, lean_object* v___y_3484_, lean_object* v___y_3485_){
_start:
{
lean_object* v_res_3486_; 
v_res_3486_ = lp_mathlib_Aesop_Frontend_getGlobalRuleSetFromData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__13___redArg(v_ext_3481_, v_simpExt_3482_, v_simprocExt_3483_, v___y_3484_);
lean_dec(v___y_3484_);
lean_dec_ref(v_simprocExt_3483_);
lean_dec_ref(v_simpExt_3482_);
lean_dec_ref(v_ext_3481_);
return v_res_3486_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_ScopedEnvExtension_add___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__9___redArg(lean_object* v_ext_3487_, lean_object* v_b_3488_, uint8_t v_kind_3489_, lean_object* v___y_3490_, lean_object* v___y_3491_){
_start:
{
lean_object* v_currNamespace_3493_; lean_object* v___x_3494_; lean_object* v_env_3495_; lean_object* v_nextMacroScope_3496_; lean_object* v_ngen_3497_; lean_object* v_auxDeclNGen_3498_; lean_object* v_traceState_3499_; lean_object* v_messages_3500_; lean_object* v_infoState_3501_; lean_object* v_snapshotTasks_3502_; lean_object* v___x_3504_; uint8_t v_isShared_3505_; uint8_t v_isSharedCheck_3514_; 
v_currNamespace_3493_ = lean_ctor_get(v___y_3490_, 6);
v___x_3494_ = lean_st_ref_take(v___y_3491_);
v_env_3495_ = lean_ctor_get(v___x_3494_, 0);
v_nextMacroScope_3496_ = lean_ctor_get(v___x_3494_, 1);
v_ngen_3497_ = lean_ctor_get(v___x_3494_, 2);
v_auxDeclNGen_3498_ = lean_ctor_get(v___x_3494_, 3);
v_traceState_3499_ = lean_ctor_get(v___x_3494_, 4);
v_messages_3500_ = lean_ctor_get(v___x_3494_, 6);
v_infoState_3501_ = lean_ctor_get(v___x_3494_, 7);
v_snapshotTasks_3502_ = lean_ctor_get(v___x_3494_, 8);
v_isSharedCheck_3514_ = !lean_is_exclusive(v___x_3494_);
if (v_isSharedCheck_3514_ == 0)
{
lean_object* v_unused_3515_; 
v_unused_3515_ = lean_ctor_get(v___x_3494_, 5);
lean_dec(v_unused_3515_);
v___x_3504_ = v___x_3494_;
v_isShared_3505_ = v_isSharedCheck_3514_;
goto v_resetjp_3503_;
}
else
{
lean_inc(v_snapshotTasks_3502_);
lean_inc(v_infoState_3501_);
lean_inc(v_messages_3500_);
lean_inc(v_traceState_3499_);
lean_inc(v_auxDeclNGen_3498_);
lean_inc(v_ngen_3497_);
lean_inc(v_nextMacroScope_3496_);
lean_inc(v_env_3495_);
lean_dec(v___x_3494_);
v___x_3504_ = lean_box(0);
v_isShared_3505_ = v_isSharedCheck_3514_;
goto v_resetjp_3503_;
}
v_resetjp_3503_:
{
lean_object* v___x_3506_; lean_object* v___x_3507_; lean_object* v___x_3509_; 
lean_inc(v_currNamespace_3493_);
v___x_3506_ = l_Lean_ScopedEnvExtension_addCore___redArg(v_env_3495_, v_ext_3487_, v_b_3488_, v_kind_3489_, v_currNamespace_3493_);
v___x_3507_ = lean_obj_once(&lp_mathlib_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__11___redArg___closed__2, &lp_mathlib_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__11___redArg___closed__2_once, _init_lp_mathlib_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__11___redArg___closed__2);
if (v_isShared_3505_ == 0)
{
lean_ctor_set(v___x_3504_, 5, v___x_3507_);
lean_ctor_set(v___x_3504_, 0, v___x_3506_);
v___x_3509_ = v___x_3504_;
goto v_reusejp_3508_;
}
else
{
lean_object* v_reuseFailAlloc_3513_; 
v_reuseFailAlloc_3513_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_3513_, 0, v___x_3506_);
lean_ctor_set(v_reuseFailAlloc_3513_, 1, v_nextMacroScope_3496_);
lean_ctor_set(v_reuseFailAlloc_3513_, 2, v_ngen_3497_);
lean_ctor_set(v_reuseFailAlloc_3513_, 3, v_auxDeclNGen_3498_);
lean_ctor_set(v_reuseFailAlloc_3513_, 4, v_traceState_3499_);
lean_ctor_set(v_reuseFailAlloc_3513_, 5, v___x_3507_);
lean_ctor_set(v_reuseFailAlloc_3513_, 6, v_messages_3500_);
lean_ctor_set(v_reuseFailAlloc_3513_, 7, v_infoState_3501_);
lean_ctor_set(v_reuseFailAlloc_3513_, 8, v_snapshotTasks_3502_);
v___x_3509_ = v_reuseFailAlloc_3513_;
goto v_reusejp_3508_;
}
v_reusejp_3508_:
{
lean_object* v___x_3510_; lean_object* v___x_3511_; lean_object* v___x_3512_; 
v___x_3510_ = lean_st_ref_set(v___y_3491_, v___x_3509_);
v___x_3511_ = lean_box(0);
v___x_3512_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3512_, 0, v___x_3511_);
return v___x_3512_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_ScopedEnvExtension_add___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__9___redArg___boxed(lean_object* v_ext_3516_, lean_object* v_b_3517_, lean_object* v_kind_3518_, lean_object* v___y_3519_, lean_object* v___y_3520_, lean_object* v___y_3521_){
_start:
{
uint8_t v_kind_boxed_3522_; lean_object* v_res_3523_; 
v_kind_boxed_3522_ = lean_unbox(v_kind_3518_);
v_res_3523_ = lp_mathlib_Lean_ScopedEnvExtension_add___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__9___redArg(v_ext_3516_, v_b_3517_, v_kind_boxed_3522_, v___y_3519_, v___y_3520_);
lean_dec(v___y_3520_);
lean_dec_ref(v___y_3519_);
return v_res_3523_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__10_spec__13_spec__20_spec__22(lean_object* v_xs_3524_, lean_object* v_v_3525_, lean_object* v_i_3526_){
_start:
{
uint8_t v___y_3532_; lean_object* v___x_3534_; uint8_t v___x_3535_; 
v___x_3534_ = lean_array_get_size(v_xs_3524_);
v___x_3535_ = lean_nat_dec_lt(v_i_3526_, v___x_3534_);
if (v___x_3535_ == 0)
{
lean_object* v___x_3536_; 
lean_dec(v_i_3526_);
v___x_3536_ = lean_box(0);
return v___x_3536_;
}
else
{
lean_object* v___x_3537_; 
v___x_3537_ = lean_array_fget_borrowed(v_xs_3524_, v_i_3526_);
if (lean_obj_tag(v___x_3537_) == 0)
{
if (lean_obj_tag(v_v_3525_) == 0)
{
lean_object* v_declName_3538_; uint8_t v_inv_3539_; lean_object* v_declName_3540_; uint8_t v_inv_3541_; uint8_t v___x_3542_; 
v_declName_3538_ = lean_ctor_get(v___x_3537_, 0);
v_inv_3539_ = lean_ctor_get_uint8(v___x_3537_, sizeof(void*)*1 + 1);
v_declName_3540_ = lean_ctor_get(v_v_3525_, 0);
v_inv_3541_ = lean_ctor_get_uint8(v_v_3525_, sizeof(void*)*1 + 1);
v___x_3542_ = lean_name_eq(v_declName_3538_, v_declName_3540_);
if (v___x_3542_ == 0)
{
v___y_3532_ = v___x_3542_;
goto v___jp_3531_;
}
else
{
if (v_inv_3539_ == 0)
{
if (v_inv_3541_ == 0)
{
v___y_3532_ = v___x_3542_;
goto v___jp_3531_;
}
else
{
goto v___jp_3527_;
}
}
else
{
v___y_3532_ = v_inv_3541_;
goto v___jp_3531_;
}
}
}
else
{
goto v___jp_3527_;
}
}
else
{
if (lean_obj_tag(v_v_3525_) == 0)
{
goto v___jp_3527_;
}
else
{
lean_object* v___x_3543_; lean_object* v___x_3544_; uint8_t v___x_3545_; 
v___x_3543_ = l_Lean_Meta_Origin_key(v___x_3537_);
v___x_3544_ = l_Lean_Meta_Origin_key(v_v_3525_);
v___x_3545_ = lean_name_eq(v___x_3543_, v___x_3544_);
lean_dec(v___x_3544_);
lean_dec(v___x_3543_);
v___y_3532_ = v___x_3545_;
goto v___jp_3531_;
}
}
}
v___jp_3527_:
{
lean_object* v___x_3528_; lean_object* v___x_3529_; 
v___x_3528_ = lean_unsigned_to_nat(1u);
v___x_3529_ = lean_nat_add(v_i_3526_, v___x_3528_);
lean_dec(v_i_3526_);
v_i_3526_ = v___x_3529_;
goto _start;
}
v___jp_3531_:
{
if (v___y_3532_ == 0)
{
goto v___jp_3527_;
}
else
{
lean_object* v___x_3533_; 
v___x_3533_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3533_, 0, v_i_3526_);
return v___x_3533_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__10_spec__13_spec__20_spec__22___boxed(lean_object* v_xs_3546_, lean_object* v_v_3547_, lean_object* v_i_3548_){
_start:
{
lean_object* v_res_3549_; 
v_res_3549_ = lp_mathlib_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__10_spec__13_spec__20_spec__22(v_xs_3546_, v_v_3547_, v_i_3548_);
lean_dec_ref(v_v_3547_);
lean_dec_ref(v_xs_3546_);
return v_res_3549_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__10_spec__13_spec__20(lean_object* v_xs_3550_, lean_object* v_v_3551_){
_start:
{
lean_object* v___x_3552_; lean_object* v___x_3553_; 
v___x_3552_ = lean_unsigned_to_nat(0u);
v___x_3553_ = lp_mathlib_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__10_spec__13_spec__20_spec__22(v_xs_3550_, v_v_3551_, v___x_3552_);
return v___x_3553_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__10_spec__13_spec__20___boxed(lean_object* v_xs_3554_, lean_object* v_v_3555_){
_start:
{
lean_object* v_res_3556_; 
v_res_3556_ = lp_mathlib_Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__10_spec__13_spec__20(v_xs_3554_, v_v_3555_);
lean_dec_ref(v_v_3555_);
lean_dec_ref(v_xs_3554_);
return v_res_3556_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__10_spec__13___redArg(lean_object* v_x_3557_, size_t v_x_3558_, lean_object* v_x_3559_){
_start:
{
if (lean_obj_tag(v_x_3557_) == 0)
{
lean_object* v_es_3560_; lean_object* v___x_3561_; size_t v___x_3562_; size_t v___x_3563_; lean_object* v_j_3564_; uint8_t v___y_3566_; lean_object* v_entry_3576_; 
v_es_3560_ = lean_ctor_get(v_x_3557_, 0);
v___x_3561_ = lean_box(2);
v___x_3562_ = ((size_t)31ULL);
v___x_3563_ = lean_usize_land(v_x_3558_, v___x_3562_);
v_j_3564_ = lean_usize_to_nat(v___x_3563_);
v_entry_3576_ = lean_array_get(v___x_3561_, v_es_3560_, v_j_3564_);
switch(lean_obj_tag(v_entry_3576_))
{
case 0:
{
if (lean_obj_tag(v_x_3559_) == 0)
{
lean_object* v_key_3577_; 
v_key_3577_ = lean_ctor_get(v_entry_3576_, 0);
lean_inc(v_key_3577_);
lean_dec_ref_known(v_entry_3576_, 2);
if (lean_obj_tag(v_key_3577_) == 0)
{
lean_object* v_declName_3578_; uint8_t v_inv_3579_; lean_object* v_declName_3580_; uint8_t v_inv_3581_; uint8_t v___x_3582_; 
v_declName_3578_ = lean_ctor_get(v_x_3559_, 0);
v_inv_3579_ = lean_ctor_get_uint8(v_x_3559_, sizeof(void*)*1 + 1);
v_declName_3580_ = lean_ctor_get(v_key_3577_, 0);
lean_inc(v_declName_3580_);
v_inv_3581_ = lean_ctor_get_uint8(v_key_3577_, sizeof(void*)*1 + 1);
lean_dec_ref_known(v_key_3577_, 1);
v___x_3582_ = lean_name_eq(v_declName_3578_, v_declName_3580_);
lean_dec(v_declName_3580_);
if (v___x_3582_ == 0)
{
v___y_3566_ = v___x_3582_;
goto v___jp_3565_;
}
else
{
if (v_inv_3579_ == 0)
{
if (v_inv_3581_ == 0)
{
v___y_3566_ = v___x_3582_;
goto v___jp_3565_;
}
else
{
lean_dec(v_j_3564_);
return v_x_3557_;
}
}
else
{
v___y_3566_ = v_inv_3581_;
goto v___jp_3565_;
}
}
}
else
{
lean_dec(v_key_3577_);
lean_dec(v_j_3564_);
return v_x_3557_;
}
}
else
{
lean_object* v_key_3583_; 
v_key_3583_ = lean_ctor_get(v_entry_3576_, 0);
lean_inc(v_key_3583_);
lean_dec_ref_known(v_entry_3576_, 2);
if (lean_obj_tag(v_key_3583_) == 0)
{
lean_dec_ref_known(v_key_3583_, 1);
lean_dec(v_j_3564_);
return v_x_3557_;
}
else
{
lean_object* v___x_3584_; lean_object* v___x_3585_; uint8_t v___x_3586_; 
v___x_3584_ = l_Lean_Meta_Origin_key(v_x_3559_);
v___x_3585_ = l_Lean_Meta_Origin_key(v_key_3583_);
lean_dec(v_key_3583_);
v___x_3586_ = lean_name_eq(v___x_3584_, v___x_3585_);
lean_dec(v___x_3585_);
lean_dec(v___x_3584_);
v___y_3566_ = v___x_3586_;
goto v___jp_3565_;
}
}
}
case 1:
{
lean_object* v___x_3588_; uint8_t v_isShared_3589_; uint8_t v_isSharedCheck_3621_; 
lean_inc_ref(v_es_3560_);
v_isSharedCheck_3621_ = !lean_is_exclusive(v_x_3557_);
if (v_isSharedCheck_3621_ == 0)
{
lean_object* v_unused_3622_; 
v_unused_3622_ = lean_ctor_get(v_x_3557_, 0);
lean_dec(v_unused_3622_);
v___x_3588_ = v_x_3557_;
v_isShared_3589_ = v_isSharedCheck_3621_;
goto v_resetjp_3587_;
}
else
{
lean_dec(v_x_3557_);
v___x_3588_ = lean_box(0);
v_isShared_3589_ = v_isSharedCheck_3621_;
goto v_resetjp_3587_;
}
v_resetjp_3587_:
{
lean_object* v_node_3590_; lean_object* v___x_3592_; uint8_t v_isShared_3593_; uint8_t v_isSharedCheck_3620_; 
v_node_3590_ = lean_ctor_get(v_entry_3576_, 0);
v_isSharedCheck_3620_ = !lean_is_exclusive(v_entry_3576_);
if (v_isSharedCheck_3620_ == 0)
{
v___x_3592_ = v_entry_3576_;
v_isShared_3593_ = v_isSharedCheck_3620_;
goto v_resetjp_3591_;
}
else
{
lean_inc(v_node_3590_);
lean_dec(v_entry_3576_);
v___x_3592_ = lean_box(0);
v_isShared_3593_ = v_isSharedCheck_3620_;
goto v_resetjp_3591_;
}
v_resetjp_3591_:
{
size_t v___x_3594_; lean_object* v_entries_3595_; size_t v___x_3596_; lean_object* v_newNode_3597_; lean_object* v___x_3598_; 
v___x_3594_ = ((size_t)5ULL);
v_entries_3595_ = lean_array_set(v_es_3560_, v_j_3564_, v___x_3561_);
v___x_3596_ = lean_usize_shift_right(v_x_3558_, v___x_3594_);
v_newNode_3597_ = lp_mathlib_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__10_spec__13___redArg(v_node_3590_, v___x_3596_, v_x_3559_);
lean_inc_ref(v_newNode_3597_);
v___x_3598_ = l_Lean_PersistentHashMap_isUnaryNode___redArg(v_newNode_3597_);
if (lean_obj_tag(v___x_3598_) == 0)
{
lean_object* v___x_3600_; 
if (v_isShared_3593_ == 0)
{
lean_ctor_set(v___x_3592_, 0, v_newNode_3597_);
v___x_3600_ = v___x_3592_;
goto v_reusejp_3599_;
}
else
{
lean_object* v_reuseFailAlloc_3605_; 
v_reuseFailAlloc_3605_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3605_, 0, v_newNode_3597_);
v___x_3600_ = v_reuseFailAlloc_3605_;
goto v_reusejp_3599_;
}
v_reusejp_3599_:
{
lean_object* v___x_3601_; lean_object* v___x_3603_; 
v___x_3601_ = lean_array_set(v_entries_3595_, v_j_3564_, v___x_3600_);
lean_dec(v_j_3564_);
if (v_isShared_3589_ == 0)
{
lean_ctor_set(v___x_3588_, 0, v___x_3601_);
v___x_3603_ = v___x_3588_;
goto v_reusejp_3602_;
}
else
{
lean_object* v_reuseFailAlloc_3604_; 
v_reuseFailAlloc_3604_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3604_, 0, v___x_3601_);
v___x_3603_ = v_reuseFailAlloc_3604_;
goto v_reusejp_3602_;
}
v_reusejp_3602_:
{
return v___x_3603_;
}
}
}
else
{
lean_object* v_val_3606_; lean_object* v_fst_3607_; lean_object* v_snd_3608_; lean_object* v___x_3610_; uint8_t v_isShared_3611_; uint8_t v_isSharedCheck_3619_; 
lean_dec_ref(v_newNode_3597_);
lean_del_object(v___x_3592_);
v_val_3606_ = lean_ctor_get(v___x_3598_, 0);
lean_inc(v_val_3606_);
lean_dec_ref_known(v___x_3598_, 1);
v_fst_3607_ = lean_ctor_get(v_val_3606_, 0);
v_snd_3608_ = lean_ctor_get(v_val_3606_, 1);
v_isSharedCheck_3619_ = !lean_is_exclusive(v_val_3606_);
if (v_isSharedCheck_3619_ == 0)
{
v___x_3610_ = v_val_3606_;
v_isShared_3611_ = v_isSharedCheck_3619_;
goto v_resetjp_3609_;
}
else
{
lean_inc(v_snd_3608_);
lean_inc(v_fst_3607_);
lean_dec(v_val_3606_);
v___x_3610_ = lean_box(0);
v_isShared_3611_ = v_isSharedCheck_3619_;
goto v_resetjp_3609_;
}
v_resetjp_3609_:
{
lean_object* v___x_3613_; 
if (v_isShared_3611_ == 0)
{
v___x_3613_ = v___x_3610_;
goto v_reusejp_3612_;
}
else
{
lean_object* v_reuseFailAlloc_3618_; 
v_reuseFailAlloc_3618_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3618_, 0, v_fst_3607_);
lean_ctor_set(v_reuseFailAlloc_3618_, 1, v_snd_3608_);
v___x_3613_ = v_reuseFailAlloc_3618_;
goto v_reusejp_3612_;
}
v_reusejp_3612_:
{
lean_object* v___x_3614_; lean_object* v___x_3616_; 
v___x_3614_ = lean_array_set(v_entries_3595_, v_j_3564_, v___x_3613_);
lean_dec(v_j_3564_);
if (v_isShared_3589_ == 0)
{
lean_ctor_set(v___x_3588_, 0, v___x_3614_);
v___x_3616_ = v___x_3588_;
goto v_reusejp_3615_;
}
else
{
lean_object* v_reuseFailAlloc_3617_; 
v_reuseFailAlloc_3617_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3617_, 0, v___x_3614_);
v___x_3616_ = v_reuseFailAlloc_3617_;
goto v_reusejp_3615_;
}
v_reusejp_3615_:
{
return v___x_3616_;
}
}
}
}
}
}
}
default: 
{
lean_dec(v_j_3564_);
return v_x_3557_;
}
}
v___jp_3565_:
{
if (v___y_3566_ == 0)
{
lean_dec(v_j_3564_);
return v_x_3557_;
}
else
{
lean_object* v___x_3568_; uint8_t v_isShared_3569_; uint8_t v_isSharedCheck_3574_; 
lean_inc_ref(v_es_3560_);
v_isSharedCheck_3574_ = !lean_is_exclusive(v_x_3557_);
if (v_isSharedCheck_3574_ == 0)
{
lean_object* v_unused_3575_; 
v_unused_3575_ = lean_ctor_get(v_x_3557_, 0);
lean_dec(v_unused_3575_);
v___x_3568_ = v_x_3557_;
v_isShared_3569_ = v_isSharedCheck_3574_;
goto v_resetjp_3567_;
}
else
{
lean_dec(v_x_3557_);
v___x_3568_ = lean_box(0);
v_isShared_3569_ = v_isSharedCheck_3574_;
goto v_resetjp_3567_;
}
v_resetjp_3567_:
{
lean_object* v___x_3570_; lean_object* v___x_3572_; 
v___x_3570_ = lean_array_set(v_es_3560_, v_j_3564_, v___x_3561_);
lean_dec(v_j_3564_);
if (v_isShared_3569_ == 0)
{
lean_ctor_set(v___x_3568_, 0, v___x_3570_);
v___x_3572_ = v___x_3568_;
goto v_reusejp_3571_;
}
else
{
lean_object* v_reuseFailAlloc_3573_; 
v_reuseFailAlloc_3573_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3573_, 0, v___x_3570_);
v___x_3572_ = v_reuseFailAlloc_3573_;
goto v_reusejp_3571_;
}
v_reusejp_3571_:
{
return v___x_3572_;
}
}
}
}
}
else
{
lean_object* v_ks_3623_; lean_object* v_vs_3624_; lean_object* v___x_3626_; uint8_t v_isShared_3627_; uint8_t v_isSharedCheck_3638_; 
v_ks_3623_ = lean_ctor_get(v_x_3557_, 0);
v_vs_3624_ = lean_ctor_get(v_x_3557_, 1);
v_isSharedCheck_3638_ = !lean_is_exclusive(v_x_3557_);
if (v_isSharedCheck_3638_ == 0)
{
v___x_3626_ = v_x_3557_;
v_isShared_3627_ = v_isSharedCheck_3638_;
goto v_resetjp_3625_;
}
else
{
lean_inc(v_vs_3624_);
lean_inc(v_ks_3623_);
lean_dec(v_x_3557_);
v___x_3626_ = lean_box(0);
v_isShared_3627_ = v_isSharedCheck_3638_;
goto v_resetjp_3625_;
}
v_resetjp_3625_:
{
lean_object* v___x_3628_; 
v___x_3628_ = lp_mathlib_Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__10_spec__13_spec__20(v_ks_3623_, v_x_3559_);
if (lean_obj_tag(v___x_3628_) == 0)
{
lean_object* v___x_3630_; 
if (v_isShared_3627_ == 0)
{
v___x_3630_ = v___x_3626_;
goto v_reusejp_3629_;
}
else
{
lean_object* v_reuseFailAlloc_3631_; 
v_reuseFailAlloc_3631_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3631_, 0, v_ks_3623_);
lean_ctor_set(v_reuseFailAlloc_3631_, 1, v_vs_3624_);
v___x_3630_ = v_reuseFailAlloc_3631_;
goto v_reusejp_3629_;
}
v_reusejp_3629_:
{
return v___x_3630_;
}
}
else
{
lean_object* v_val_3632_; lean_object* v_keys_x27_3633_; lean_object* v_vals_x27_3634_; lean_object* v___x_3636_; 
v_val_3632_ = lean_ctor_get(v___x_3628_, 0);
lean_inc_n(v_val_3632_, 2);
lean_dec_ref_known(v___x_3628_, 1);
v_keys_x27_3633_ = l_Array_eraseIdx___redArg(v_ks_3623_, v_val_3632_);
v_vals_x27_3634_ = l_Array_eraseIdx___redArg(v_vs_3624_, v_val_3632_);
if (v_isShared_3627_ == 0)
{
lean_ctor_set(v___x_3626_, 1, v_vals_x27_3634_);
lean_ctor_set(v___x_3626_, 0, v_keys_x27_3633_);
v___x_3636_ = v___x_3626_;
goto v_reusejp_3635_;
}
else
{
lean_object* v_reuseFailAlloc_3637_; 
v_reuseFailAlloc_3637_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3637_, 0, v_keys_x27_3633_);
lean_ctor_set(v_reuseFailAlloc_3637_, 1, v_vals_x27_3634_);
v___x_3636_ = v_reuseFailAlloc_3637_;
goto v_reusejp_3635_;
}
v_reusejp_3635_:
{
return v___x_3636_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__10_spec__13___redArg___boxed(lean_object* v_x_3639_, lean_object* v_x_3640_, lean_object* v_x_3641_){
_start:
{
size_t v_x_14951__boxed_3642_; lean_object* v_res_3643_; 
v_x_14951__boxed_3642_ = lean_unbox_usize(v_x_3640_);
lean_dec(v_x_3640_);
v_res_3643_ = lp_mathlib_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__10_spec__13___redArg(v_x_3639_, v_x_14951__boxed_3642_, v_x_3641_);
lean_dec_ref(v_x_3641_);
return v_res_3643_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__10___redArg(lean_object* v_x_3644_, lean_object* v_x_3645_){
_start:
{
uint64_t v___y_3647_; uint64_t v___y_3651_; uint64_t v___y_3655_; 
if (lean_obj_tag(v_x_3645_) == 0)
{
uint8_t v_inv_3658_; 
v_inv_3658_ = lean_ctor_get_uint8(v_x_3645_, sizeof(void*)*1 + 1);
if (v_inv_3658_ == 0)
{
lean_object* v_declName_3659_; 
v_declName_3659_ = lean_ctor_get(v_x_3645_, 0);
if (lean_obj_tag(v_declName_3659_) == 0)
{
uint64_t v___x_3660_; 
v___x_3660_ = 1723ULL;
v___y_3651_ = v___x_3660_;
goto v___jp_3650_;
}
else
{
uint64_t v_hash_3661_; 
v_hash_3661_ = lean_ctor_get_uint64(v_declName_3659_, sizeof(void*)*2);
v___y_3651_ = v_hash_3661_;
goto v___jp_3650_;
}
}
else
{
lean_object* v_declName_3662_; 
v_declName_3662_ = lean_ctor_get(v_x_3645_, 0);
if (lean_obj_tag(v_declName_3662_) == 0)
{
uint64_t v___x_3663_; 
v___x_3663_ = 1723ULL;
v___y_3655_ = v___x_3663_;
goto v___jp_3654_;
}
else
{
uint64_t v_hash_3664_; 
v_hash_3664_ = lean_ctor_get_uint64(v_declName_3662_, sizeof(void*)*2);
v___y_3655_ = v_hash_3664_;
goto v___jp_3654_;
}
}
}
else
{
lean_object* v___x_3665_; 
v___x_3665_ = l_Lean_Meta_Origin_key(v_x_3645_);
if (lean_obj_tag(v___x_3665_) == 0)
{
uint64_t v___x_3666_; 
v___x_3666_ = 1723ULL;
v___y_3647_ = v___x_3666_;
goto v___jp_3646_;
}
else
{
uint64_t v_hash_3667_; 
v_hash_3667_ = lean_ctor_get_uint64(v___x_3665_, sizeof(void*)*2);
lean_dec(v___x_3665_);
v___y_3647_ = v_hash_3667_;
goto v___jp_3646_;
}
}
v___jp_3646_:
{
size_t v_h_3648_; lean_object* v___x_3649_; 
v_h_3648_ = lean_uint64_to_usize(v___y_3647_);
v___x_3649_ = lp_mathlib_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__10_spec__13___redArg(v_x_3644_, v_h_3648_, v_x_3645_);
return v___x_3649_;
}
v___jp_3650_:
{
uint64_t v___x_3652_; uint64_t v___x_3653_; 
v___x_3652_ = 13ULL;
v___x_3653_ = lean_uint64_mix_hash(v___y_3651_, v___x_3652_);
v___y_3647_ = v___x_3653_;
goto v___jp_3646_;
}
v___jp_3654_:
{
uint64_t v___x_3656_; uint64_t v___x_3657_; 
v___x_3656_ = 11ULL;
v___x_3657_ = lean_uint64_mix_hash(v___y_3655_, v___x_3656_);
v___y_3647_ = v___x_3657_;
goto v___jp_3646_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__10___redArg___boxed(lean_object* v_x_3668_, lean_object* v_x_3669_){
_start:
{
lean_object* v_res_3670_; 
v_res_3670_ = lp_mathlib_Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__10___redArg(v_x_3668_, v_x_3669_);
lean_dec_ref(v_x_3669_);
return v_res_3670_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__12___lam__0(lean_object* v_a_3671_, lean_object* v_simpTheorems_3672_){
_start:
{
lean_object* v_pre_3673_; lean_object* v_post_3674_; lean_object* v_lemmaNames_3675_; lean_object* v_toUnfold_3676_; lean_object* v_erased_3677_; lean_object* v_toUnfoldThms_3678_; lean_object* v___x_3680_; uint8_t v_isShared_3681_; uint8_t v_isSharedCheck_3687_; 
v_pre_3673_ = lean_ctor_get(v_simpTheorems_3672_, 0);
v_post_3674_ = lean_ctor_get(v_simpTheorems_3672_, 1);
v_lemmaNames_3675_ = lean_ctor_get(v_simpTheorems_3672_, 2);
v_toUnfold_3676_ = lean_ctor_get(v_simpTheorems_3672_, 3);
v_erased_3677_ = lean_ctor_get(v_simpTheorems_3672_, 4);
v_toUnfoldThms_3678_ = lean_ctor_get(v_simpTheorems_3672_, 5);
v_isSharedCheck_3687_ = !lean_is_exclusive(v_simpTheorems_3672_);
if (v_isSharedCheck_3687_ == 0)
{
v___x_3680_ = v_simpTheorems_3672_;
v_isShared_3681_ = v_isSharedCheck_3687_;
goto v_resetjp_3679_;
}
else
{
lean_inc(v_toUnfoldThms_3678_);
lean_inc(v_erased_3677_);
lean_inc(v_toUnfold_3676_);
lean_inc(v_lemmaNames_3675_);
lean_inc(v_post_3674_);
lean_inc(v_pre_3673_);
lean_dec(v_simpTheorems_3672_);
v___x_3680_ = lean_box(0);
v_isShared_3681_ = v_isSharedCheck_3687_;
goto v_resetjp_3679_;
}
v_resetjp_3679_:
{
lean_object* v_origin_3682_; lean_object* v___x_3683_; lean_object* v___x_3685_; 
v_origin_3682_ = lean_ctor_get(v_a_3671_, 4);
v___x_3683_ = lp_mathlib_Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__10___redArg(v_erased_3677_, v_origin_3682_);
if (v_isShared_3681_ == 0)
{
lean_ctor_set(v___x_3680_, 4, v___x_3683_);
v___x_3685_ = v___x_3680_;
goto v_reusejp_3684_;
}
else
{
lean_object* v_reuseFailAlloc_3686_; 
v_reuseFailAlloc_3686_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v_reuseFailAlloc_3686_, 0, v_pre_3673_);
lean_ctor_set(v_reuseFailAlloc_3686_, 1, v_post_3674_);
lean_ctor_set(v_reuseFailAlloc_3686_, 2, v_lemmaNames_3675_);
lean_ctor_set(v_reuseFailAlloc_3686_, 3, v_toUnfold_3676_);
lean_ctor_set(v_reuseFailAlloc_3686_, 4, v___x_3683_);
lean_ctor_set(v_reuseFailAlloc_3686_, 5, v_toUnfoldThms_3678_);
v___x_3685_ = v_reuseFailAlloc_3686_;
goto v_reusejp_3684_;
}
v_reusejp_3684_:
{
return v___x_3685_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__12___lam__0___boxed(lean_object* v_a_3688_, lean_object* v_simpTheorems_3689_){
_start:
{
lean_object* v_res_3690_; 
v_res_3690_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__12___lam__0(v_a_3688_, v_simpTheorems_3689_);
lean_dec_ref(v_a_3688_);
return v_res_3690_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__12(lean_object* v_fst_3691_, uint8_t v_kind_3692_, lean_object* v_as_3693_, size_t v_sz_3694_, size_t v_i_3695_, lean_object* v_b_3696_, lean_object* v___y_3697_, lean_object* v___y_3698_){
_start:
{
lean_object* v_a_3701_; uint8_t v___x_3705_; 
v___x_3705_ = lean_usize_dec_lt(v_i_3695_, v_sz_3694_);
if (v___x_3705_ == 0)
{
lean_object* v___x_3706_; 
lean_dec_ref(v_fst_3691_);
v___x_3706_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3706_, 0, v_b_3696_);
return v___x_3706_;
}
else
{
lean_object* v_a_3707_; lean_object* v___x_3708_; 
v_a_3707_ = lean_array_uget_borrowed(v_as_3693_, v_i_3695_);
lean_inc(v_a_3707_);
lean_inc_ref(v_fst_3691_);
v___x_3708_ = lp_mathlib_Lean_ScopedEnvExtension_add___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__9___redArg(v_fst_3691_, v_a_3707_, v_kind_3692_, v___y_3697_, v___y_3698_);
if (lean_obj_tag(v___x_3708_) == 0)
{
lean_object* v___x_3709_; 
lean_dec_ref_known(v___x_3708_, 1);
v___x_3709_ = lean_box(0);
if (lean_obj_tag(v_a_3707_) == 0)
{
lean_object* v_a_3710_; lean_object* v___x_3711_; lean_object* v_env_3712_; lean_object* v___f_3713_; lean_object* v___x_3714_; lean_object* v___x_3715_; 
v_a_3710_ = lean_ctor_get(v_a_3707_, 0);
v___x_3711_ = lean_st_ref_get(v___y_3698_);
v_env_3712_ = lean_ctor_get(v___x_3711_, 0);
lean_inc_ref(v_env_3712_);
lean_dec(v___x_3711_);
lean_inc_ref(v_a_3710_);
v___f_3713_ = lean_alloc_closure((void*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__12___lam__0___boxed), 2, 1);
lean_closure_set(v___f_3713_, 0, v_a_3710_);
lean_inc_ref(v_fst_3691_);
v___x_3714_ = l_Lean_ScopedEnvExtension_modifyState___redArg(v_fst_3691_, v_env_3712_, v___f_3713_);
v___x_3715_ = lp_mathlib_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__11___redArg(v___x_3714_, v___y_3698_);
if (lean_obj_tag(v___x_3715_) == 0)
{
lean_dec_ref_known(v___x_3715_, 1);
v_a_3701_ = v___x_3709_;
goto v___jp_3700_;
}
else
{
lean_dec_ref(v_fst_3691_);
return v___x_3715_;
}
}
else
{
v_a_3701_ = v___x_3709_;
goto v___jp_3700_;
}
}
else
{
lean_dec_ref(v_fst_3691_);
return v___x_3708_;
}
}
v___jp_3700_:
{
size_t v___x_3702_; size_t v___x_3703_; 
v___x_3702_ = ((size_t)1ULL);
v___x_3703_ = lean_usize_add(v_i_3695_, v___x_3702_);
v_i_3695_ = v___x_3703_;
v_b_3696_ = v_a_3701_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__12___boxed(lean_object* v_fst_3716_, lean_object* v_kind_3717_, lean_object* v_as_3718_, lean_object* v_sz_3719_, lean_object* v_i_3720_, lean_object* v_b_3721_, lean_object* v___y_3722_, lean_object* v___y_3723_, lean_object* v___y_3724_){
_start:
{
uint8_t v_kind_boxed_3725_; size_t v_sz_boxed_3726_; size_t v_i_boxed_3727_; lean_object* v_res_3728_; 
v_kind_boxed_3725_ = lean_unbox(v_kind_3717_);
v_sz_boxed_3726_ = lean_unbox_usize(v_sz_3719_);
lean_dec(v_sz_3719_);
v_i_boxed_3727_ = lean_unbox_usize(v_i_3720_);
lean_dec(v_i_3720_);
v_res_3728_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__12(v_fst_3716_, v_kind_boxed_3725_, v_as_3718_, v_sz_boxed_3726_, v_i_boxed_3727_, v_b_3721_, v___y_3722_, v___y_3723_);
lean_dec(v___y_3723_);
lean_dec_ref(v___y_3722_);
lean_dec_ref(v_as_3718_);
return v_res_3728_;
}
}
static lean_object* _init_lp_mathlib_Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1___closed__1(void){
_start:
{
lean_object* v___x_3730_; lean_object* v___x_3731_; 
v___x_3730_ = ((lean_object*)(lp_mathlib_Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1___closed__0));
v___x_3731_ = l_Lean_stringToMessageData(v___x_3730_);
return v___x_3731_;
}
}
static lean_object* _init_lp_mathlib_Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1___closed__3(void){
_start:
{
lean_object* v___x_3733_; lean_object* v___x_3734_; 
v___x_3733_ = ((lean_object*)(lp_mathlib_Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1___closed__2));
v___x_3734_ = l_Lean_stringToMessageData(v___x_3733_);
return v___x_3734_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1(lean_object* v_rsName_3735_, lean_object* v_r_3736_, uint8_t v_kind_3737_, uint8_t v_checkNotExists_3738_, lean_object* v___y_3739_, lean_object* v___y_3740_){
_start:
{
lean_object* v___x_3742_; 
lean_inc(v_rsName_3735_);
v___x_3742_ = lp_mathlib_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8(v_rsName_3735_, v___y_3739_, v___y_3740_);
if (lean_obj_tag(v___x_3742_) == 0)
{
lean_object* v_a_3743_; lean_object* v_snd_3744_; lean_object* v_snd_3745_; lean_object* v___x_3747_; uint8_t v_isShared_3748_; uint8_t v_isSharedCheck_3809_; 
v_a_3743_ = lean_ctor_get(v___x_3742_, 0);
lean_inc(v_a_3743_);
lean_dec_ref_known(v___x_3742_, 1);
v_snd_3744_ = lean_ctor_get(v_a_3743_, 1);
lean_inc(v_snd_3744_);
v_snd_3745_ = lean_ctor_get(v_snd_3744_, 1);
v_isSharedCheck_3809_ = !lean_is_exclusive(v_snd_3744_);
if (v_isSharedCheck_3809_ == 0)
{
lean_object* v_unused_3810_; 
v_unused_3810_ = lean_ctor_get(v_snd_3744_, 0);
lean_dec(v_unused_3810_);
v___x_3747_ = v_snd_3744_;
v_isShared_3748_ = v_isSharedCheck_3809_;
goto v_resetjp_3746_;
}
else
{
lean_inc(v_snd_3745_);
lean_dec(v_snd_3744_);
v___x_3747_ = lean_box(0);
v_isShared_3748_ = v_isSharedCheck_3809_;
goto v_resetjp_3746_;
}
v_resetjp_3746_:
{
lean_object* v_fst_3749_; lean_object* v___x_3751_; uint8_t v_isShared_3752_; uint8_t v_isSharedCheck_3807_; 
v_fst_3749_ = lean_ctor_get(v_a_3743_, 0);
v_isSharedCheck_3807_ = !lean_is_exclusive(v_a_3743_);
if (v_isSharedCheck_3807_ == 0)
{
lean_object* v_unused_3808_; 
v_unused_3808_ = lean_ctor_get(v_a_3743_, 1);
lean_dec(v_unused_3808_);
v___x_3751_ = v_a_3743_;
v_isShared_3752_ = v_isSharedCheck_3807_;
goto v_resetjp_3750_;
}
else
{
lean_inc(v_fst_3749_);
lean_dec(v_a_3743_);
v___x_3751_ = lean_box(0);
v_isShared_3752_ = v_isSharedCheck_3807_;
goto v_resetjp_3750_;
}
v_resetjp_3750_:
{
lean_object* v_fst_3753_; lean_object* v_snd_3754_; lean_object* v___x_3756_; uint8_t v_isShared_3757_; uint8_t v_isSharedCheck_3806_; 
v_fst_3753_ = lean_ctor_get(v_snd_3745_, 0);
v_snd_3754_ = lean_ctor_get(v_snd_3745_, 1);
v_isSharedCheck_3806_ = !lean_is_exclusive(v_snd_3745_);
if (v_isSharedCheck_3806_ == 0)
{
v___x_3756_ = v_snd_3745_;
v_isShared_3757_ = v_isSharedCheck_3806_;
goto v_resetjp_3755_;
}
else
{
lean_inc(v_snd_3754_);
lean_inc(v_fst_3753_);
lean_dec(v_snd_3745_);
v___x_3756_ = lean_box(0);
v_isShared_3757_ = v_isSharedCheck_3806_;
goto v_resetjp_3755_;
}
v_resetjp_3755_:
{
lean_object* v___y_3759_; lean_object* v___y_3760_; 
if (v_checkNotExists_3738_ == 0)
{
lean_del_object(v___x_3756_);
lean_dec(v_snd_3754_);
lean_del_object(v___x_3751_);
lean_del_object(v___x_3747_);
lean_dec(v_rsName_3735_);
v___y_3759_ = v___y_3739_;
v___y_3760_ = v___y_3740_;
goto v___jp_3758_;
}
else
{
lean_object* v_snd_3777_; lean_object* v___x_3779_; uint8_t v_isShared_3780_; uint8_t v_isSharedCheck_3804_; 
v_snd_3777_ = lean_ctor_get(v_snd_3754_, 1);
v_isSharedCheck_3804_ = !lean_is_exclusive(v_snd_3754_);
if (v_isSharedCheck_3804_ == 0)
{
lean_object* v_unused_3805_; 
v_unused_3805_ = lean_ctor_get(v_snd_3754_, 0);
lean_dec(v_unused_3805_);
v___x_3779_ = v_snd_3754_;
v_isShared_3780_ = v_isSharedCheck_3804_;
goto v_resetjp_3778_;
}
else
{
lean_inc(v_snd_3777_);
lean_dec(v_snd_3754_);
v___x_3779_ = lean_box(0);
v_isShared_3780_ = v_isSharedCheck_3804_;
goto v_resetjp_3778_;
}
v_resetjp_3778_:
{
lean_object* v___x_3781_; lean_object* v_a_3782_; lean_object* v___x_3783_; uint8_t v___x_3784_; 
v___x_3781_ = lp_mathlib_Aesop_Frontend_getGlobalRuleSetFromData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__13___redArg(v_fst_3749_, v_fst_3753_, v_snd_3777_, v___y_3740_);
lean_dec(v_snd_3777_);
v_a_3782_ = lean_ctor_get(v___x_3781_, 0);
lean_inc(v_a_3782_);
lean_dec_ref(v___x_3781_);
v___x_3783_ = lp_aesop_Aesop_GlobalRuleSetMember_name(v_r_3736_);
lean_inc_ref(v___x_3783_);
v___x_3784_ = lp_aesop_Aesop_GlobalRuleSet_contains(v_a_3782_, v___x_3783_);
lean_dec(v_a_3782_);
if (v___x_3784_ == 0)
{
lean_dec_ref(v___x_3783_);
lean_del_object(v___x_3779_);
lean_del_object(v___x_3756_);
lean_del_object(v___x_3751_);
lean_del_object(v___x_3747_);
lean_dec(v_rsName_3735_);
v___y_3759_ = v___y_3739_;
v___y_3760_ = v___y_3740_;
goto v___jp_3758_;
}
else
{
lean_object* v_name_3785_; lean_object* v___x_3786_; lean_object* v___x_3787_; lean_object* v___x_3789_; 
lean_dec(v_fst_3753_);
lean_dec(v_fst_3749_);
lean_dec_ref(v_r_3736_);
v_name_3785_ = lean_ctor_get(v___x_3783_, 0);
lean_inc(v_name_3785_);
lean_dec_ref(v___x_3783_);
v___x_3786_ = lean_obj_once(&lp_mathlib_Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1___closed__1, &lp_mathlib_Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1___closed__1_once, _init_lp_mathlib_Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1___closed__1);
v___x_3787_ = l_Lean_MessageData_ofName(v_name_3785_);
if (v_isShared_3780_ == 0)
{
lean_ctor_set_tag(v___x_3779_, 7);
lean_ctor_set(v___x_3779_, 1, v___x_3787_);
lean_ctor_set(v___x_3779_, 0, v___x_3786_);
v___x_3789_ = v___x_3779_;
goto v_reusejp_3788_;
}
else
{
lean_object* v_reuseFailAlloc_3803_; 
v_reuseFailAlloc_3803_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3803_, 0, v___x_3786_);
lean_ctor_set(v_reuseFailAlloc_3803_, 1, v___x_3787_);
v___x_3789_ = v_reuseFailAlloc_3803_;
goto v_reusejp_3788_;
}
v_reusejp_3788_:
{
lean_object* v___x_3790_; lean_object* v___x_3792_; 
v___x_3790_ = lean_obj_once(&lp_mathlib_Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1___closed__3, &lp_mathlib_Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1___closed__3_once, _init_lp_mathlib_Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1___closed__3);
if (v_isShared_3757_ == 0)
{
lean_ctor_set_tag(v___x_3756_, 7);
lean_ctor_set(v___x_3756_, 1, v___x_3790_);
lean_ctor_set(v___x_3756_, 0, v___x_3789_);
v___x_3792_ = v___x_3756_;
goto v_reusejp_3791_;
}
else
{
lean_object* v_reuseFailAlloc_3802_; 
v_reuseFailAlloc_3802_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3802_, 0, v___x_3789_);
lean_ctor_set(v_reuseFailAlloc_3802_, 1, v___x_3790_);
v___x_3792_ = v_reuseFailAlloc_3802_;
goto v_reusejp_3791_;
}
v_reusejp_3791_:
{
lean_object* v___x_3793_; lean_object* v___x_3795_; 
v___x_3793_ = l_Lean_MessageData_ofName(v_rsName_3735_);
if (v_isShared_3748_ == 0)
{
lean_ctor_set_tag(v___x_3747_, 7);
lean_ctor_set(v___x_3747_, 1, v___x_3793_);
lean_ctor_set(v___x_3747_, 0, v___x_3792_);
v___x_3795_ = v___x_3747_;
goto v_reusejp_3794_;
}
else
{
lean_object* v_reuseFailAlloc_3801_; 
v_reuseFailAlloc_3801_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3801_, 0, v___x_3792_);
lean_ctor_set(v_reuseFailAlloc_3801_, 1, v___x_3793_);
v___x_3795_ = v_reuseFailAlloc_3801_;
goto v_reusejp_3794_;
}
v_reusejp_3794_:
{
lean_object* v___x_3796_; lean_object* v___x_3798_; 
v___x_3796_ = lean_obj_once(&lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__1, &lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__1_once, _init_lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__1);
if (v_isShared_3752_ == 0)
{
lean_ctor_set_tag(v___x_3751_, 7);
lean_ctor_set(v___x_3751_, 1, v___x_3796_);
lean_ctor_set(v___x_3751_, 0, v___x_3795_);
v___x_3798_ = v___x_3751_;
goto v_reusejp_3797_;
}
else
{
lean_object* v_reuseFailAlloc_3800_; 
v_reuseFailAlloc_3800_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3800_, 0, v___x_3795_);
lean_ctor_set(v_reuseFailAlloc_3800_, 1, v___x_3796_);
v___x_3798_ = v_reuseFailAlloc_3800_;
goto v_reusejp_3797_;
}
v_reusejp_3797_:
{
lean_object* v___x_3799_; 
v___x_3799_ = lp_mathlib_Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__0___redArg(v___x_3798_, v___y_3739_, v___y_3740_);
return v___x_3799_;
}
}
}
}
}
}
}
v___jp_3758_:
{
if (lean_obj_tag(v_r_3736_) == 0)
{
lean_object* v_m_3761_; lean_object* v___x_3762_; 
lean_dec(v_fst_3753_);
v_m_3761_ = lean_ctor_get(v_r_3736_, 0);
lean_inc_ref(v_m_3761_);
lean_dec_ref_known(v_r_3736_, 1);
v___x_3762_ = lp_mathlib_Lean_ScopedEnvExtension_add___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__9___redArg(v_fst_3749_, v_m_3761_, v_kind_3737_, v___y_3759_, v___y_3760_);
return v___x_3762_;
}
else
{
lean_object* v_e_3763_; lean_object* v_entries_3764_; lean_object* v___x_3765_; size_t v_sz_3766_; size_t v___x_3767_; lean_object* v___x_3768_; 
lean_dec(v_fst_3749_);
v_e_3763_ = lean_ctor_get(v_r_3736_, 0);
lean_inc_ref(v_e_3763_);
lean_dec_ref_known(v_r_3736_, 1);
v_entries_3764_ = lean_ctor_get(v_e_3763_, 1);
lean_inc_ref(v_entries_3764_);
lean_dec_ref(v_e_3763_);
v___x_3765_ = lean_box(0);
v_sz_3766_ = lean_array_size(v_entries_3764_);
v___x_3767_ = ((size_t)0ULL);
v___x_3768_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__12(v_fst_3753_, v_kind_3737_, v_entries_3764_, v_sz_3766_, v___x_3767_, v___x_3765_, v___y_3759_, v___y_3760_);
lean_dec_ref(v_entries_3764_);
if (lean_obj_tag(v___x_3768_) == 0)
{
lean_object* v___x_3770_; uint8_t v_isShared_3771_; uint8_t v_isSharedCheck_3775_; 
v_isSharedCheck_3775_ = !lean_is_exclusive(v___x_3768_);
if (v_isSharedCheck_3775_ == 0)
{
lean_object* v_unused_3776_; 
v_unused_3776_ = lean_ctor_get(v___x_3768_, 0);
lean_dec(v_unused_3776_);
v___x_3770_ = v___x_3768_;
v_isShared_3771_ = v_isSharedCheck_3775_;
goto v_resetjp_3769_;
}
else
{
lean_dec(v___x_3768_);
v___x_3770_ = lean_box(0);
v_isShared_3771_ = v_isSharedCheck_3775_;
goto v_resetjp_3769_;
}
v_resetjp_3769_:
{
lean_object* v___x_3773_; 
if (v_isShared_3771_ == 0)
{
lean_ctor_set(v___x_3770_, 0, v___x_3765_);
v___x_3773_ = v___x_3770_;
goto v_reusejp_3772_;
}
else
{
lean_object* v_reuseFailAlloc_3774_; 
v_reuseFailAlloc_3774_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3774_, 0, v___x_3765_);
v___x_3773_ = v_reuseFailAlloc_3774_;
goto v_reusejp_3772_;
}
v_reusejp_3772_:
{
return v___x_3773_;
}
}
}
else
{
return v___x_3768_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_3811_; lean_object* v___x_3813_; uint8_t v_isShared_3814_; uint8_t v_isSharedCheck_3818_; 
lean_dec_ref(v_r_3736_);
lean_dec(v_rsName_3735_);
v_a_3811_ = lean_ctor_get(v___x_3742_, 0);
v_isSharedCheck_3818_ = !lean_is_exclusive(v___x_3742_);
if (v_isSharedCheck_3818_ == 0)
{
v___x_3813_ = v___x_3742_;
v_isShared_3814_ = v_isSharedCheck_3818_;
goto v_resetjp_3812_;
}
else
{
lean_inc(v_a_3811_);
lean_dec(v___x_3742_);
v___x_3813_ = lean_box(0);
v_isShared_3814_ = v_isSharedCheck_3818_;
goto v_resetjp_3812_;
}
v_resetjp_3812_:
{
lean_object* v___x_3816_; 
if (v_isShared_3814_ == 0)
{
v___x_3816_ = v___x_3813_;
goto v_reusejp_3815_;
}
else
{
lean_object* v_reuseFailAlloc_3817_; 
v_reuseFailAlloc_3817_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3817_, 0, v_a_3811_);
v___x_3816_ = v_reuseFailAlloc_3817_;
goto v_reusejp_3815_;
}
v_reusejp_3815_:
{
return v___x_3816_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1___boxed(lean_object* v_rsName_3819_, lean_object* v_r_3820_, lean_object* v_kind_3821_, lean_object* v_checkNotExists_3822_, lean_object* v___y_3823_, lean_object* v___y_3824_, lean_object* v___y_3825_){
_start:
{
uint8_t v_kind_boxed_3826_; uint8_t v_checkNotExists_boxed_3827_; lean_object* v_res_3828_; 
v_kind_boxed_3826_ = lean_unbox(v_kind_3821_);
v_checkNotExists_boxed_3827_ = lean_unbox(v_checkNotExists_3822_);
v_res_3828_ = lp_mathlib_Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1(v_rsName_3819_, v_r_3820_, v_kind_boxed_3826_, v_checkNotExists_boxed_3827_, v___y_3823_, v___y_3824_);
lean_dec(v___y_3824_);
lean_dec_ref(v___y_3823_);
return v_res_3828_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__2(lean_object* v_fst_3829_, uint8_t v_attrKind_3830_, lean_object* v_as_3831_, size_t v_sz_3832_, size_t v_i_3833_, lean_object* v_b_3834_, lean_object* v___y_3835_, lean_object* v___y_3836_){
_start:
{
uint8_t v___x_3838_; 
v___x_3838_ = lean_usize_dec_lt(v_i_3833_, v_sz_3832_);
if (v___x_3838_ == 0)
{
lean_object* v___x_3839_; 
lean_dec_ref(v_fst_3829_);
v___x_3839_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3839_, 0, v_b_3834_);
return v___x_3839_;
}
else
{
lean_object* v_a_3840_; lean_object* v___x_3841_; 
v_a_3840_ = lean_array_uget_borrowed(v_as_3831_, v_i_3833_);
lean_inc_ref(v_fst_3829_);
lean_inc(v_a_3840_);
v___x_3841_ = lp_mathlib_Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1(v_a_3840_, v_fst_3829_, v_attrKind_3830_, v___x_3838_, v___y_3835_, v___y_3836_);
if (lean_obj_tag(v___x_3841_) == 0)
{
lean_object* v___x_3842_; size_t v___x_3843_; size_t v___x_3844_; 
lean_dec_ref_known(v___x_3841_, 1);
v___x_3842_ = lean_box(0);
v___x_3843_ = ((size_t)1ULL);
v___x_3844_ = lean_usize_add(v_i_3833_, v___x_3843_);
v_i_3833_ = v___x_3844_;
v_b_3834_ = v___x_3842_;
goto _start;
}
else
{
lean_dec_ref(v_fst_3829_);
return v___x_3841_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__2___boxed(lean_object* v_fst_3846_, lean_object* v_attrKind_3847_, lean_object* v_as_3848_, lean_object* v_sz_3849_, lean_object* v_i_3850_, lean_object* v_b_3851_, lean_object* v___y_3852_, lean_object* v___y_3853_, lean_object* v___y_3854_){
_start:
{
uint8_t v_attrKind_boxed_3855_; size_t v_sz_boxed_3856_; size_t v_i_boxed_3857_; lean_object* v_res_3858_; 
v_attrKind_boxed_3855_ = lean_unbox(v_attrKind_3847_);
v_sz_boxed_3856_ = lean_unbox_usize(v_sz_3849_);
lean_dec(v_sz_3849_);
v_i_boxed_3857_ = lean_unbox_usize(v_i_3850_);
lean_dec(v_i_3850_);
v_res_3858_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__2(v_fst_3846_, v_attrKind_boxed_3855_, v_as_3848_, v_sz_boxed_3856_, v_i_boxed_3857_, v_b_3851_, v___y_3852_, v___y_3853_);
lean_dec(v___y_3853_);
lean_dec_ref(v___y_3852_);
lean_dec_ref(v_as_3848_);
return v_res_3858_;
}
}
static double _init_lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3___closed__0(void){
_start:
{
lean_object* v___x_3859_; double v___x_3860_; 
v___x_3859_ = lean_unsigned_to_nat(0u);
v___x_3860_ = lean_float_of_nat(v___x_3859_);
return v___x_3860_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3(lean_object* v_cls_3864_, lean_object* v_msg_3865_, lean_object* v___y_3866_, lean_object* v___y_3867_){
_start:
{
lean_object* v_ref_3869_; lean_object* v___x_3870_; lean_object* v_a_3871_; lean_object* v___x_3873_; uint8_t v_isShared_3874_; uint8_t v_isSharedCheck_3915_; 
v_ref_3869_ = lean_ctor_get(v___y_3866_, 5);
v___x_3870_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3_spec__16(v_msg_3865_, v___y_3866_, v___y_3867_);
v_a_3871_ = lean_ctor_get(v___x_3870_, 0);
v_isSharedCheck_3915_ = !lean_is_exclusive(v___x_3870_);
if (v_isSharedCheck_3915_ == 0)
{
v___x_3873_ = v___x_3870_;
v_isShared_3874_ = v_isSharedCheck_3915_;
goto v_resetjp_3872_;
}
else
{
lean_inc(v_a_3871_);
lean_dec(v___x_3870_);
v___x_3873_ = lean_box(0);
v_isShared_3874_ = v_isSharedCheck_3915_;
goto v_resetjp_3872_;
}
v_resetjp_3872_:
{
lean_object* v___x_3875_; lean_object* v_traceState_3876_; lean_object* v_env_3877_; lean_object* v_nextMacroScope_3878_; lean_object* v_ngen_3879_; lean_object* v_auxDeclNGen_3880_; lean_object* v_cache_3881_; lean_object* v_messages_3882_; lean_object* v_infoState_3883_; lean_object* v_snapshotTasks_3884_; lean_object* v___x_3886_; uint8_t v_isShared_3887_; uint8_t v_isSharedCheck_3914_; 
v___x_3875_ = lean_st_ref_take(v___y_3867_);
v_traceState_3876_ = lean_ctor_get(v___x_3875_, 4);
v_env_3877_ = lean_ctor_get(v___x_3875_, 0);
v_nextMacroScope_3878_ = lean_ctor_get(v___x_3875_, 1);
v_ngen_3879_ = lean_ctor_get(v___x_3875_, 2);
v_auxDeclNGen_3880_ = lean_ctor_get(v___x_3875_, 3);
v_cache_3881_ = lean_ctor_get(v___x_3875_, 5);
v_messages_3882_ = lean_ctor_get(v___x_3875_, 6);
v_infoState_3883_ = lean_ctor_get(v___x_3875_, 7);
v_snapshotTasks_3884_ = lean_ctor_get(v___x_3875_, 8);
v_isSharedCheck_3914_ = !lean_is_exclusive(v___x_3875_);
if (v_isSharedCheck_3914_ == 0)
{
v___x_3886_ = v___x_3875_;
v_isShared_3887_ = v_isSharedCheck_3914_;
goto v_resetjp_3885_;
}
else
{
lean_inc(v_snapshotTasks_3884_);
lean_inc(v_infoState_3883_);
lean_inc(v_messages_3882_);
lean_inc(v_cache_3881_);
lean_inc(v_traceState_3876_);
lean_inc(v_auxDeclNGen_3880_);
lean_inc(v_ngen_3879_);
lean_inc(v_nextMacroScope_3878_);
lean_inc(v_env_3877_);
lean_dec(v___x_3875_);
v___x_3886_ = lean_box(0);
v_isShared_3887_ = v_isSharedCheck_3914_;
goto v_resetjp_3885_;
}
v_resetjp_3885_:
{
uint64_t v_tid_3888_; lean_object* v_traces_3889_; lean_object* v___x_3891_; uint8_t v_isShared_3892_; uint8_t v_isSharedCheck_3913_; 
v_tid_3888_ = lean_ctor_get_uint64(v_traceState_3876_, sizeof(void*)*1);
v_traces_3889_ = lean_ctor_get(v_traceState_3876_, 0);
v_isSharedCheck_3913_ = !lean_is_exclusive(v_traceState_3876_);
if (v_isSharedCheck_3913_ == 0)
{
v___x_3891_ = v_traceState_3876_;
v_isShared_3892_ = v_isSharedCheck_3913_;
goto v_resetjp_3890_;
}
else
{
lean_inc(v_traces_3889_);
lean_dec(v_traceState_3876_);
v___x_3891_ = lean_box(0);
v_isShared_3892_ = v_isSharedCheck_3913_;
goto v_resetjp_3890_;
}
v_resetjp_3890_:
{
lean_object* v___x_3893_; double v___x_3894_; uint8_t v___x_3895_; lean_object* v___x_3896_; lean_object* v___x_3897_; lean_object* v___x_3898_; lean_object* v___x_3899_; lean_object* v___x_3900_; lean_object* v___x_3901_; lean_object* v___x_3903_; 
v___x_3893_ = lean_box(0);
v___x_3894_ = lean_float_once(&lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3___closed__0, &lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3___closed__0_once, _init_lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3___closed__0);
v___x_3895_ = 0;
v___x_3896_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3___closed__1));
v___x_3897_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v___x_3897_, 0, v_cls_3864_);
lean_ctor_set(v___x_3897_, 1, v___x_3893_);
lean_ctor_set(v___x_3897_, 2, v___x_3896_);
lean_ctor_set_float(v___x_3897_, sizeof(void*)*3, v___x_3894_);
lean_ctor_set_float(v___x_3897_, sizeof(void*)*3 + 8, v___x_3894_);
lean_ctor_set_uint8(v___x_3897_, sizeof(void*)*3 + 16, v___x_3895_);
v___x_3898_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3___closed__2));
v___x_3899_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_3899_, 0, v___x_3897_);
lean_ctor_set(v___x_3899_, 1, v_a_3871_);
lean_ctor_set(v___x_3899_, 2, v___x_3898_);
lean_inc(v_ref_3869_);
v___x_3900_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3900_, 0, v_ref_3869_);
lean_ctor_set(v___x_3900_, 1, v___x_3899_);
v___x_3901_ = l_Lean_PersistentArray_push___redArg(v_traces_3889_, v___x_3900_);
if (v_isShared_3892_ == 0)
{
lean_ctor_set(v___x_3891_, 0, v___x_3901_);
v___x_3903_ = v___x_3891_;
goto v_reusejp_3902_;
}
else
{
lean_object* v_reuseFailAlloc_3912_; 
v_reuseFailAlloc_3912_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_3912_, 0, v___x_3901_);
lean_ctor_set_uint64(v_reuseFailAlloc_3912_, sizeof(void*)*1, v_tid_3888_);
v___x_3903_ = v_reuseFailAlloc_3912_;
goto v_reusejp_3902_;
}
v_reusejp_3902_:
{
lean_object* v___x_3905_; 
if (v_isShared_3887_ == 0)
{
lean_ctor_set(v___x_3886_, 4, v___x_3903_);
v___x_3905_ = v___x_3886_;
goto v_reusejp_3904_;
}
else
{
lean_object* v_reuseFailAlloc_3911_; 
v_reuseFailAlloc_3911_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_3911_, 0, v_env_3877_);
lean_ctor_set(v_reuseFailAlloc_3911_, 1, v_nextMacroScope_3878_);
lean_ctor_set(v_reuseFailAlloc_3911_, 2, v_ngen_3879_);
lean_ctor_set(v_reuseFailAlloc_3911_, 3, v_auxDeclNGen_3880_);
lean_ctor_set(v_reuseFailAlloc_3911_, 4, v___x_3903_);
lean_ctor_set(v_reuseFailAlloc_3911_, 5, v_cache_3881_);
lean_ctor_set(v_reuseFailAlloc_3911_, 6, v_messages_3882_);
lean_ctor_set(v_reuseFailAlloc_3911_, 7, v_infoState_3883_);
lean_ctor_set(v_reuseFailAlloc_3911_, 8, v_snapshotTasks_3884_);
v___x_3905_ = v_reuseFailAlloc_3911_;
goto v_reusejp_3904_;
}
v_reusejp_3904_:
{
lean_object* v___x_3906_; lean_object* v___x_3907_; lean_object* v___x_3909_; 
v___x_3906_ = lean_st_ref_set(v___y_3867_, v___x_3905_);
v___x_3907_ = lean_box(0);
if (v_isShared_3874_ == 0)
{
lean_ctor_set(v___x_3873_, 0, v___x_3907_);
v___x_3909_ = v___x_3873_;
goto v_reusejp_3908_;
}
else
{
lean_object* v_reuseFailAlloc_3910_; 
v_reuseFailAlloc_3910_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3910_, 0, v___x_3907_);
v___x_3909_ = v_reuseFailAlloc_3910_;
goto v_reusejp_3908_;
}
v_reusejp_3908_:
{
return v___x_3909_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3___boxed(lean_object* v_cls_3916_, lean_object* v_msg_3917_, lean_object* v___y_3918_, lean_object* v___y_3919_, lean_object* v___y_3920_){
_start:
{
lean_object* v_res_3921_; 
v_res_3921_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3(v_cls_3916_, v_msg_3917_, v___y_3918_, v___y_3919_);
lean_dec(v___y_3919_);
lean_dec_ref(v___y_3918_);
return v_res_3921_;
}
}
static uint64_t _init_lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__1_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_3928_; uint64_t v___x_3929_; 
v___x_3928_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__0_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_));
v___x_3929_ = l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(v___x_3928_);
return v___x_3929_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__2_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_(void){
_start:
{
uint64_t v___x_3930_; lean_object* v___x_3931_; lean_object* v___x_3932_; 
v___x_3930_ = lean_uint64_once(&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__1_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__1_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__1_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_);
v___x_3931_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__0_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_));
v___x_3932_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v___x_3932_, 0, v___x_3931_);
lean_ctor_set_uint64(v___x_3932_, sizeof(void*)*1, v___x_3930_);
return v___x_3932_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__3_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_3933_; 
v___x_3933_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_3933_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__4_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_3934_; lean_object* v___x_3935_; 
v___x_3934_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__3_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__3_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__3_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_);
v___x_3935_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3935_, 0, v___x_3934_);
return v___x_3935_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__5_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_3936_; lean_object* v___x_3937_; 
v___x_3936_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__4_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__4_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__4_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_);
v___x_3937_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_3937_, 0, v___x_3936_);
lean_ctor_set(v___x_3937_, 1, v___x_3936_);
lean_ctor_set(v___x_3937_, 2, v___x_3936_);
lean_ctor_set(v___x_3937_, 3, v___x_3936_);
lean_ctor_set(v___x_3937_, 4, v___x_3936_);
lean_ctor_set(v___x_3937_, 5, v___x_3936_);
return v___x_3937_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__6_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_3938_; lean_object* v___x_3939_; 
v___x_3938_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__4_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__4_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__4_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_);
v___x_3939_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_3939_, 0, v___x_3938_);
lean_ctor_set(v___x_3939_, 1, v___x_3938_);
lean_ctor_set(v___x_3939_, 2, v___x_3938_);
lean_ctor_set(v___x_3939_, 3, v___x_3938_);
lean_ctor_set(v___x_3939_, 4, v___x_3938_);
return v___x_3939_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__11_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_3947_; lean_object* v___x_3948_; 
v___x_3947_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__10_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_));
v___x_3948_ = l_Lean_stringToMessageData(v___x_3947_);
return v___x_3948_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_(lean_object* v___x_3949_, lean_object* v___f_3950_, lean_object* v___f_3951_, lean_object* v___x_3952_, lean_object* v_decl_3953_, lean_object* v_stx_3954_, uint8_t v_attrKind_3955_, lean_object* v___y_3956_, lean_object* v___y_3957_){
_start:
{
lean_object* v___x_3959_; lean_object* v___x_3960_; uint8_t v___x_3961_; lean_object* v___x_3962_; uint8_t v___x_3963_; lean_object* v___x_3964_; lean_object* v___x_3965_; lean_object* v___x_3966_; lean_object* v___x_3967_; lean_object* v___x_3968_; lean_object* v___x_3969_; lean_object* v___x_3970_; size_t v___x_3971_; lean_object* v___x_3972_; lean_object* v___x_3973_; lean_object* v___x_3974_; lean_object* v___x_3975_; lean_object* v___x_3976_; lean_object* v___x_3977_; lean_object* v___x_3978_; lean_object* v___x_3979_; lean_object* v_fileName_3980_; lean_object* v_fileMap_3981_; lean_object* v_options_3982_; lean_object* v_currRecDepth_3983_; lean_object* v_maxRecDepth_3984_; lean_object* v_ref_3985_; lean_object* v_currNamespace_3986_; lean_object* v_openDecls_3987_; lean_object* v_initHeartbeats_3988_; lean_object* v_maxHeartbeats_3989_; lean_object* v_quotContext_3990_; lean_object* v_currMacroScope_3991_; uint8_t v_diag_3992_; lean_object* v_cancelTk_x3f_3993_; uint8_t v_suppressElabErrors_3994_; lean_object* v_inheritedTraceOptions_3995_; lean_object* v___f_3996_; lean_object* v_ref_3997_; lean_object* v___x_3998_; lean_object* v___x_3999_; lean_object* v___x_4000_; 
v___x_3959_ = lean_box(0);
v___x_3960_ = lean_box(0);
v___x_3961_ = 1;
v___x_3962_ = lean_box(1);
v___x_3963_ = 0;
v___x_3964_ = lean_mk_empty_array_with_capacity(v___x_3949_);
lean_inc_ref_n(v___x_3964_, 2);
v___x_3965_ = lean_alloc_ctor(0, 8, 11);
lean_ctor_set(v___x_3965_, 0, v___x_3959_);
lean_ctor_set(v___x_3965_, 1, v___x_3960_);
lean_ctor_set(v___x_3965_, 2, v___x_3959_);
lean_ctor_set(v___x_3965_, 3, v___f_3950_);
lean_ctor_set(v___x_3965_, 4, v___x_3962_);
lean_ctor_set(v___x_3965_, 5, v___x_3962_);
lean_ctor_set(v___x_3965_, 6, v___x_3959_);
lean_ctor_set(v___x_3965_, 7, v___x_3964_);
lean_ctor_set_uint8(v___x_3965_, sizeof(void*)*8, v___x_3961_);
lean_ctor_set_uint8(v___x_3965_, sizeof(void*)*8 + 1, v___x_3961_);
lean_ctor_set_uint8(v___x_3965_, sizeof(void*)*8 + 2, v___x_3961_);
lean_ctor_set_uint8(v___x_3965_, sizeof(void*)*8 + 3, v___x_3961_);
lean_ctor_set_uint8(v___x_3965_, sizeof(void*)*8 + 4, v___x_3963_);
lean_ctor_set_uint8(v___x_3965_, sizeof(void*)*8 + 5, v___x_3963_);
lean_ctor_set_uint8(v___x_3965_, sizeof(void*)*8 + 6, v___x_3963_);
lean_ctor_set_uint8(v___x_3965_, sizeof(void*)*8 + 7, v___x_3963_);
lean_ctor_set_uint8(v___x_3965_, sizeof(void*)*8 + 8, v___x_3961_);
lean_ctor_set_uint8(v___x_3965_, sizeof(void*)*8 + 9, v___x_3963_);
lean_ctor_set_uint8(v___x_3965_, sizeof(void*)*8 + 10, v___x_3961_);
v___x_3966_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__2_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__2_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__2_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_);
v___x_3967_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__4_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__4_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__4_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_);
v___x_3968_ = lean_unsigned_to_nat(32u);
v___x_3969_ = lean_mk_empty_array_with_capacity(v___x_3968_);
v___x_3970_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3_spec__16___closed__3, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3_spec__16___closed__3_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3_spec__16___closed__3);
v___x_3971_ = ((size_t)5ULL);
lean_inc_n(v___x_3949_, 6);
v___x_3972_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_3972_, 0, v___x_3970_);
lean_ctor_set(v___x_3972_, 1, v___x_3969_);
lean_ctor_set(v___x_3972_, 2, v___x_3949_);
lean_ctor_set(v___x_3972_, 3, v___x_3949_);
lean_ctor_set_usize(v___x_3972_, 4, v___x_3971_);
lean_inc_ref(v___x_3972_);
v___x_3973_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_3973_, 0, v___x_3967_);
lean_ctor_set(v___x_3973_, 1, v___x_3972_);
lean_ctor_set(v___x_3973_, 2, v___x_3962_);
v___x_3974_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_3974_, 0, v___x_3966_);
lean_ctor_set(v___x_3974_, 1, v___x_3962_);
lean_ctor_set(v___x_3974_, 2, v___x_3973_);
lean_ctor_set(v___x_3974_, 3, v___x_3964_);
lean_ctor_set(v___x_3974_, 4, v___x_3959_);
lean_ctor_set(v___x_3974_, 5, v___x_3949_);
lean_ctor_set(v___x_3974_, 6, v___x_3959_);
lean_ctor_set_uint8(v___x_3974_, sizeof(void*)*7, v___x_3963_);
lean_ctor_set_uint8(v___x_3974_, sizeof(void*)*7 + 1, v___x_3963_);
lean_ctor_set_uint8(v___x_3974_, sizeof(void*)*7 + 2, v___x_3963_);
lean_ctor_set_uint8(v___x_3974_, sizeof(void*)*7 + 3, v___x_3961_);
v___x_3975_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_3975_, 0, v___x_3949_);
lean_ctor_set(v___x_3975_, 1, v___x_3949_);
lean_ctor_set(v___x_3975_, 2, v___x_3949_);
lean_ctor_set(v___x_3975_, 3, v___x_3949_);
lean_ctor_set(v___x_3975_, 4, v___x_3967_);
lean_ctor_set(v___x_3975_, 5, v___x_3967_);
lean_ctor_set(v___x_3975_, 6, v___x_3967_);
lean_ctor_set(v___x_3975_, 7, v___x_3967_);
lean_ctor_set(v___x_3975_, 8, v___x_3967_);
lean_ctor_set(v___x_3975_, 9, v___x_3967_);
v___x_3976_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__5_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__5_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__5_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_);
v___x_3977_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__6_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__6_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__6_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_);
v___x_3978_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_3978_, 0, v___x_3975_);
lean_ctor_set(v___x_3978_, 1, v___x_3976_);
lean_ctor_set(v___x_3978_, 2, v___x_3962_);
lean_ctor_set(v___x_3978_, 3, v___x_3972_);
lean_ctor_set(v___x_3978_, 4, v___x_3977_);
lean_inc_ref(v___x_3978_);
v___x_3979_ = lean_st_mk_ref(v___x_3978_);
v_fileName_3980_ = lean_ctor_get(v___y_3956_, 0);
v_fileMap_3981_ = lean_ctor_get(v___y_3956_, 1);
v_options_3982_ = lean_ctor_get(v___y_3956_, 2);
v_currRecDepth_3983_ = lean_ctor_get(v___y_3956_, 3);
v_maxRecDepth_3984_ = lean_ctor_get(v___y_3956_, 4);
v_ref_3985_ = lean_ctor_get(v___y_3956_, 5);
v_currNamespace_3986_ = lean_ctor_get(v___y_3956_, 6);
v_openDecls_3987_ = lean_ctor_get(v___y_3956_, 7);
v_initHeartbeats_3988_ = lean_ctor_get(v___y_3956_, 8);
v_maxHeartbeats_3989_ = lean_ctor_get(v___y_3956_, 9);
v_quotContext_3990_ = lean_ctor_get(v___y_3956_, 10);
v_currMacroScope_3991_ = lean_ctor_get(v___y_3956_, 11);
v_diag_3992_ = lean_ctor_get_uint8(v___y_3956_, sizeof(void*)*14);
v_cancelTk_x3f_3993_ = lean_ctor_get(v___y_3956_, 12);
v_suppressElabErrors_3994_ = lean_ctor_get_uint8(v___y_3956_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_3995_ = lean_ctor_get(v___y_3956_, 13);
lean_inc(v_decl_3953_);
v___f_3996_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__3_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2____boxed), 8, 1);
lean_closure_set(v___f_3996_, 0, v_decl_3953_);
v_ref_3997_ = l_Lean_replaceRef(v_stx_3954_, v_ref_3985_);
lean_inc_ref(v_inheritedTraceOptions_3995_);
lean_inc(v_cancelTk_x3f_3993_);
lean_inc(v_currMacroScope_3991_);
lean_inc(v_quotContext_3990_);
lean_inc(v_maxHeartbeats_3989_);
lean_inc(v_initHeartbeats_3988_);
lean_inc(v_openDecls_3987_);
lean_inc(v_currNamespace_3986_);
lean_inc(v_maxRecDepth_3984_);
lean_inc(v_currRecDepth_3983_);
lean_inc_ref(v_options_3982_);
lean_inc_ref(v_fileMap_3981_);
lean_inc_ref(v_fileName_3980_);
v___x_3998_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_3998_, 0, v_fileName_3980_);
lean_ctor_set(v___x_3998_, 1, v_fileMap_3981_);
lean_ctor_set(v___x_3998_, 2, v_options_3982_);
lean_ctor_set(v___x_3998_, 3, v_currRecDepth_3983_);
lean_ctor_set(v___x_3998_, 4, v_maxRecDepth_3984_);
lean_ctor_set(v___x_3998_, 5, v_ref_3997_);
lean_ctor_set(v___x_3998_, 6, v_currNamespace_3986_);
lean_ctor_set(v___x_3998_, 7, v_openDecls_3987_);
lean_ctor_set(v___x_3998_, 8, v_initHeartbeats_3988_);
lean_ctor_set(v___x_3998_, 9, v_maxHeartbeats_3989_);
lean_ctor_set(v___x_3998_, 10, v_quotContext_3990_);
lean_ctor_set(v___x_3998_, 11, v_currMacroScope_3991_);
lean_ctor_set(v___x_3998_, 12, v_cancelTk_x3f_3993_);
lean_ctor_set(v___x_3998_, 13, v_inheritedTraceOptions_3995_);
lean_ctor_set_uint8(v___x_3998_, sizeof(void*)*14, v_diag_3992_);
lean_ctor_set_uint8(v___x_3998_, sizeof(void*)*14 + 1, v_suppressElabErrors_3994_);
v___x_3999_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__7_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_));
v___x_4000_ = l_Lean_Elab_Term_TermElabM_run___redArg(v___f_3996_, v___x_3965_, v___x_3999_, v___x_3974_, v___x_3979_, v___x_3998_, v___y_3957_);
if (lean_obj_tag(v___x_4000_) == 0)
{
lean_object* v_a_4001_; lean_object* v___x_4002_; lean_object* v_fst_4003_; lean_object* v___x_4005_; uint8_t v_isShared_4006_; uint8_t v_isSharedCheck_4077_; 
v_a_4001_ = lean_ctor_get(v___x_4000_, 0);
lean_inc(v_a_4001_);
lean_dec_ref_known(v___x_4000_, 1);
v___x_4002_ = lean_st_ref_get(v___x_3979_);
lean_dec(v___x_3979_);
lean_dec(v___x_4002_);
v_fst_4003_ = lean_ctor_get(v_a_4001_, 0);
v_isSharedCheck_4077_ = !lean_is_exclusive(v_a_4001_);
if (v_isSharedCheck_4077_ == 0)
{
lean_object* v_unused_4078_; 
v_unused_4078_ = lean_ctor_get(v_a_4001_, 1);
lean_dec(v_unused_4078_);
v___x_4005_ = v_a_4001_;
v_isShared_4006_ = v_isSharedCheck_4077_;
goto v_resetjp_4004_;
}
else
{
lean_inc(v_fst_4003_);
lean_dec(v_a_4001_);
v___x_4005_ = lean_box(0);
v_isShared_4006_ = v_isSharedCheck_4077_;
goto v_resetjp_4004_;
}
v_resetjp_4004_:
{
lean_object* v___y_4008_; lean_object* v___y_4009_; lean_object* v_a_4010_; lean_object* v___y_4043_; lean_object* v___y_4044_; uint8_t v_hasTrace_4058_; 
v_hasTrace_4058_ = lean_ctor_get_uint8(v_options_3982_, sizeof(void*)*1);
if (v_hasTrace_4058_ == 0)
{
lean_del_object(v___x_4005_);
lean_dec_ref(v___x_3952_);
v___y_4043_ = v___x_3998_;
v___y_4044_ = v___y_3957_;
goto v___jp_4042_;
}
else
{
lean_object* v___x_4059_; lean_object* v___x_4060_; lean_object* v___x_4061_; lean_object* v___x_4062_; uint8_t v___x_4063_; 
v___x_4059_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__1_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2_));
v___x_4060_ = l_Lean_Name_mkStr2(v___x_3952_, v___x_4059_);
v___x_4061_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__9_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_));
lean_inc(v___x_4060_);
v___x_4062_ = l_Lean_Name_append(v___x_4061_, v___x_4060_);
v___x_4063_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_3995_, v_options_3982_, v___x_4062_);
lean_dec(v___x_4062_);
if (v___x_4063_ == 0)
{
lean_dec(v___x_4060_);
lean_del_object(v___x_4005_);
v___y_4043_ = v___x_3998_;
v___y_4044_ = v___y_3957_;
goto v___jp_4042_;
}
else
{
lean_object* v___x_4064_; lean_object* v___x_4065_; lean_object* v___x_4067_; 
v___x_4064_ = lean_obj_once(&lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__1, &lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__1_once, _init_lp_mathlib_Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0___closed__1);
lean_inc(v_decl_3953_);
v___x_4065_ = l_Lean_MessageData_ofName(v_decl_3953_);
if (v_isShared_4006_ == 0)
{
lean_ctor_set_tag(v___x_4005_, 7);
lean_ctor_set(v___x_4005_, 1, v___x_4065_);
lean_ctor_set(v___x_4005_, 0, v___x_4064_);
v___x_4067_ = v___x_4005_;
goto v_reusejp_4066_;
}
else
{
lean_object* v_reuseFailAlloc_4076_; 
v_reuseFailAlloc_4076_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4076_, 0, v___x_4064_);
lean_ctor_set(v_reuseFailAlloc_4076_, 1, v___x_4065_);
v___x_4067_ = v_reuseFailAlloc_4076_;
goto v_reusejp_4066_;
}
v_reusejp_4066_:
{
lean_object* v___x_4068_; lean_object* v___x_4069_; lean_object* v___x_4070_; lean_object* v___x_4071_; lean_object* v___x_4072_; lean_object* v___x_4073_; lean_object* v___x_4074_; lean_object* v___x_4075_; 
v___x_4068_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__11_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__11_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2___closed__11_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_);
v___x_4069_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4069_, 0, v___x_4067_);
lean_ctor_set(v___x_4069_, 1, v___x_4068_);
lean_inc(v_fst_4003_);
v___x_4070_ = l_Nat_reprFast(v_fst_4003_);
v___x_4071_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_4071_, 0, v___x_4070_);
v___x_4072_ = l_Lean_MessageData_ofFormat(v___x_4071_);
v___x_4073_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4073_, 0, v___x_4069_);
lean_ctor_set(v___x_4073_, 1, v___x_4072_);
v___x_4074_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4074_, 0, v___x_4073_);
lean_ctor_set(v___x_4074_, 1, v___x_4064_);
v___x_4075_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__3(v___x_4060_, v___x_4074_, v___x_3998_, v___y_3957_);
if (lean_obj_tag(v___x_4075_) == 0)
{
lean_dec_ref_known(v___x_4075_, 1);
v___y_4043_ = v___x_3998_;
v___y_4044_ = v___y_3957_;
goto v___jp_4042_;
}
else
{
lean_dec(v_fst_4003_);
lean_dec_ref_known(v___x_3998_, 14);
lean_dec_ref_known(v___x_3978_, 5);
lean_dec_ref_known(v___x_3974_, 7);
lean_dec_ref(v___x_3964_);
lean_dec(v_decl_3953_);
lean_dec_ref(v___f_3951_);
return v___x_4075_;
}
}
}
}
v___jp_4007_:
{
lean_object* v___x_4011_; lean_object* v___x_4012_; lean_object* v___x_4013_; lean_object* v___x_4014_; lean_object* v___x_4015_; lean_object* v___x_4016_; 
v___x_4011_ = lean_alloc_ctor(0, 8, 11);
lean_ctor_set(v___x_4011_, 0, v___x_3959_);
lean_ctor_set(v___x_4011_, 1, v___x_3960_);
lean_ctor_set(v___x_4011_, 2, v___x_3959_);
lean_ctor_set(v___x_4011_, 3, v___f_3951_);
lean_ctor_set(v___x_4011_, 4, v___x_3962_);
lean_ctor_set(v___x_4011_, 5, v___x_3962_);
lean_ctor_set(v___x_4011_, 6, v___x_3959_);
lean_ctor_set(v___x_4011_, 7, v___x_3964_);
lean_ctor_set_uint8(v___x_4011_, sizeof(void*)*8, v___x_3961_);
lean_ctor_set_uint8(v___x_4011_, sizeof(void*)*8 + 1, v___x_3961_);
lean_ctor_set_uint8(v___x_4011_, sizeof(void*)*8 + 2, v___x_3961_);
lean_ctor_set_uint8(v___x_4011_, sizeof(void*)*8 + 3, v___x_3961_);
lean_ctor_set_uint8(v___x_4011_, sizeof(void*)*8 + 4, v___x_3963_);
lean_ctor_set_uint8(v___x_4011_, sizeof(void*)*8 + 5, v___x_3963_);
lean_ctor_set_uint8(v___x_4011_, sizeof(void*)*8 + 6, v___x_3963_);
lean_ctor_set_uint8(v___x_4011_, sizeof(void*)*8 + 7, v___x_3963_);
lean_ctor_set_uint8(v___x_4011_, sizeof(void*)*8 + 8, v___x_3961_);
lean_ctor_set_uint8(v___x_4011_, sizeof(void*)*8 + 9, v___x_3963_);
lean_ctor_set_uint8(v___x_4011_, sizeof(void*)*8 + 10, v___x_3961_);
v___x_4012_ = lean_st_mk_ref(v___x_3978_);
v___x_4013_ = lp_mathlib_Mathlib_Tactic_Bound_scoreToConfig(v_decl_3953_, v_fst_4003_);
v___x_4014_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Frontend_RuleConfig_buildGlobalRule___boxed), 9, 1);
lean_closure_set(v___x_4014_, 0, v___x_4013_);
v___x_4015_ = lean_alloc_closure((void*)(lp_aesop_Aesop_ElabM_run___boxed), 10, 3);
lean_closure_set(v___x_4015_, 0, lean_box(0));
lean_closure_set(v___x_4015_, 1, v_a_4010_);
lean_closure_set(v___x_4015_, 2, v___x_4014_);
v___x_4016_ = l_Lean_Elab_Term_TermElabM_run___redArg(v___x_4015_, v___x_4011_, v___x_3999_, v___x_3974_, v___x_4012_, v___y_4008_, v___y_4009_);
lean_dec_ref_known(v___x_3974_, 7);
if (lean_obj_tag(v___x_4016_) == 0)
{
lean_object* v_a_4017_; lean_object* v___x_4018_; lean_object* v_fst_4019_; lean_object* v_fst_4020_; lean_object* v_snd_4021_; lean_object* v___x_4022_; size_t v_sz_4023_; size_t v___x_4024_; lean_object* v___x_4025_; 
v_a_4017_ = lean_ctor_get(v___x_4016_, 0);
lean_inc(v_a_4017_);
lean_dec_ref_known(v___x_4016_, 1);
v___x_4018_ = lean_st_ref_get(v___x_4012_);
lean_dec(v___x_4012_);
lean_dec(v___x_4018_);
v_fst_4019_ = lean_ctor_get(v_a_4017_, 0);
lean_inc(v_fst_4019_);
lean_dec(v_a_4017_);
v_fst_4020_ = lean_ctor_get(v_fst_4019_, 0);
lean_inc(v_fst_4020_);
v_snd_4021_ = lean_ctor_get(v_fst_4019_, 1);
lean_inc(v_snd_4021_);
lean_dec(v_fst_4019_);
v___x_4022_ = lean_box(0);
v_sz_4023_ = lean_array_size(v_snd_4021_);
v___x_4024_ = ((size_t)0ULL);
v___x_4025_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__2(v_fst_4020_, v_attrKind_3955_, v_snd_4021_, v_sz_4023_, v___x_4024_, v___x_4022_, v___y_4008_, v___y_4009_);
lean_dec_ref(v___y_4008_);
lean_dec(v_snd_4021_);
if (lean_obj_tag(v___x_4025_) == 0)
{
lean_object* v___x_4027_; uint8_t v_isShared_4028_; uint8_t v_isSharedCheck_4032_; 
v_isSharedCheck_4032_ = !lean_is_exclusive(v___x_4025_);
if (v_isSharedCheck_4032_ == 0)
{
lean_object* v_unused_4033_; 
v_unused_4033_ = lean_ctor_get(v___x_4025_, 0);
lean_dec(v_unused_4033_);
v___x_4027_ = v___x_4025_;
v_isShared_4028_ = v_isSharedCheck_4032_;
goto v_resetjp_4026_;
}
else
{
lean_dec(v___x_4025_);
v___x_4027_ = lean_box(0);
v_isShared_4028_ = v_isSharedCheck_4032_;
goto v_resetjp_4026_;
}
v_resetjp_4026_:
{
lean_object* v___x_4030_; 
if (v_isShared_4028_ == 0)
{
lean_ctor_set(v___x_4027_, 0, v___x_4022_);
v___x_4030_ = v___x_4027_;
goto v_reusejp_4029_;
}
else
{
lean_object* v_reuseFailAlloc_4031_; 
v_reuseFailAlloc_4031_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4031_, 0, v___x_4022_);
v___x_4030_ = v_reuseFailAlloc_4031_;
goto v_reusejp_4029_;
}
v_reusejp_4029_:
{
return v___x_4030_;
}
}
}
else
{
return v___x_4025_;
}
}
else
{
lean_object* v_a_4034_; lean_object* v___x_4036_; uint8_t v_isShared_4037_; uint8_t v_isSharedCheck_4041_; 
lean_dec(v___x_4012_);
lean_dec_ref(v___y_4008_);
v_a_4034_ = lean_ctor_get(v___x_4016_, 0);
v_isSharedCheck_4041_ = !lean_is_exclusive(v___x_4016_);
if (v_isSharedCheck_4041_ == 0)
{
v___x_4036_ = v___x_4016_;
v_isShared_4037_ = v_isSharedCheck_4041_;
goto v_resetjp_4035_;
}
else
{
lean_inc(v_a_4034_);
lean_dec(v___x_4016_);
v___x_4036_ = lean_box(0);
v_isShared_4037_ = v_isSharedCheck_4041_;
goto v_resetjp_4035_;
}
v_resetjp_4035_:
{
lean_object* v___x_4039_; 
if (v_isShared_4037_ == 0)
{
v___x_4039_ = v___x_4036_;
goto v_reusejp_4038_;
}
else
{
lean_object* v_reuseFailAlloc_4040_; 
v_reuseFailAlloc_4040_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4040_, 0, v_a_4034_);
v___x_4039_ = v_reuseFailAlloc_4040_;
goto v_reusejp_4038_;
}
v_reusejp_4038_:
{
return v___x_4039_;
}
}
}
}
v___jp_4042_:
{
lean_object* v___x_4045_; lean_object* v___x_4046_; 
lean_inc_ref(v___x_3978_);
v___x_4045_ = lean_st_mk_ref(v___x_3978_);
v___x_4046_ = lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules(v___x_3974_, v___x_4045_, v___y_4043_, v___y_4044_);
if (lean_obj_tag(v___x_4046_) == 0)
{
lean_object* v_a_4047_; lean_object* v___x_4048_; 
v_a_4047_ = lean_ctor_get(v___x_4046_, 0);
lean_inc(v_a_4047_);
lean_dec_ref_known(v___x_4046_, 1);
v___x_4048_ = lean_st_ref_get(v___x_4045_);
lean_dec(v___x_4045_);
lean_dec(v___x_4048_);
v___y_4008_ = v___y_4043_;
v___y_4009_ = v___y_4044_;
v_a_4010_ = v_a_4047_;
goto v___jp_4007_;
}
else
{
lean_dec(v___x_4045_);
if (lean_obj_tag(v___x_4046_) == 0)
{
lean_object* v_a_4049_; 
v_a_4049_ = lean_ctor_get(v___x_4046_, 0);
lean_inc(v_a_4049_);
lean_dec_ref_known(v___x_4046_, 1);
v___y_4008_ = v___y_4043_;
v___y_4009_ = v___y_4044_;
v_a_4010_ = v_a_4049_;
goto v___jp_4007_;
}
else
{
lean_object* v_a_4050_; lean_object* v___x_4052_; uint8_t v_isShared_4053_; uint8_t v_isSharedCheck_4057_; 
lean_dec_ref(v___y_4043_);
lean_dec(v_fst_4003_);
lean_dec_ref_known(v___x_3978_, 5);
lean_dec_ref_known(v___x_3974_, 7);
lean_dec_ref(v___x_3964_);
lean_dec(v_decl_3953_);
lean_dec_ref(v___f_3951_);
v_a_4050_ = lean_ctor_get(v___x_4046_, 0);
v_isSharedCheck_4057_ = !lean_is_exclusive(v___x_4046_);
if (v_isSharedCheck_4057_ == 0)
{
v___x_4052_ = v___x_4046_;
v_isShared_4053_ = v_isSharedCheck_4057_;
goto v_resetjp_4051_;
}
else
{
lean_inc(v_a_4050_);
lean_dec(v___x_4046_);
v___x_4052_ = lean_box(0);
v_isShared_4053_ = v_isSharedCheck_4057_;
goto v_resetjp_4051_;
}
v_resetjp_4051_:
{
lean_object* v___x_4055_; 
if (v_isShared_4053_ == 0)
{
v___x_4055_ = v___x_4052_;
goto v_reusejp_4054_;
}
else
{
lean_object* v_reuseFailAlloc_4056_; 
v_reuseFailAlloc_4056_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4056_, 0, v_a_4050_);
v___x_4055_ = v_reuseFailAlloc_4056_;
goto v_reusejp_4054_;
}
v_reusejp_4054_:
{
return v___x_4055_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_4079_; lean_object* v___x_4081_; uint8_t v_isShared_4082_; uint8_t v_isSharedCheck_4086_; 
lean_dec_ref_known(v___x_3998_, 14);
lean_dec(v___x_3979_);
lean_dec_ref_known(v___x_3978_, 5);
lean_dec_ref_known(v___x_3974_, 7);
lean_dec_ref(v___x_3964_);
lean_dec(v_decl_3953_);
lean_dec_ref(v___x_3952_);
lean_dec_ref(v___f_3951_);
v_a_4079_ = lean_ctor_get(v___x_4000_, 0);
v_isSharedCheck_4086_ = !lean_is_exclusive(v___x_4000_);
if (v_isSharedCheck_4086_ == 0)
{
v___x_4081_ = v___x_4000_;
v_isShared_4082_ = v_isSharedCheck_4086_;
goto v_resetjp_4080_;
}
else
{
lean_inc(v_a_4079_);
lean_dec(v___x_4000_);
v___x_4081_ = lean_box(0);
v_isShared_4082_ = v_isSharedCheck_4086_;
goto v_resetjp_4080_;
}
v_resetjp_4080_:
{
lean_object* v___x_4084_; 
if (v_isShared_4082_ == 0)
{
v___x_4084_ = v___x_4081_;
goto v_reusejp_4083_;
}
else
{
lean_object* v_reuseFailAlloc_4085_; 
v_reuseFailAlloc_4085_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4085_, 0, v_a_4079_);
v___x_4084_ = v_reuseFailAlloc_4085_;
goto v_reusejp_4083_;
}
v_reusejp_4083_:
{
return v___x_4084_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2____boxed(lean_object* v___x_4087_, lean_object* v___f_4088_, lean_object* v___f_4089_, lean_object* v___x_4090_, lean_object* v_decl_4091_, lean_object* v_stx_4092_, lean_object* v_attrKind_4093_, lean_object* v___y_4094_, lean_object* v___y_4095_, lean_object* v___y_4096_){
_start:
{
uint8_t v_attrKind_boxed_4097_; lean_object* v_res_4098_; 
v_attrKind_boxed_4097_ = lean_unbox(v_attrKind_4093_);
v_res_4098_ = lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___lam__2_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_(v___x_4087_, v___f_4088_, v___f_4089_, v___x_4090_, v_decl_4091_, v_stx_4092_, v_attrKind_boxed_4097_, v___y_4094_, v___y_4095_);
lean_dec(v___y_4095_);
lean_dec_ref(v___y_4094_);
lean_dec(v_stx_4092_);
return v_res_4098_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__2_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_4101_; lean_object* v___x_4102_; lean_object* v___x_4103_; 
v___x_4101_ = lean_unsigned_to_nat(2543913104u);
v___x_4102_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__24_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2_));
v___x_4103_ = l_Lean_Name_num___override(v___x_4102_, v___x_4101_);
return v___x_4103_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__3_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_4104_; lean_object* v___x_4105_; lean_object* v___x_4106_; 
v___x_4104_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__26_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2_));
v___x_4105_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__2_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__2_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__2_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_);
v___x_4106_ = l_Lean_Name_str___override(v___x_4105_, v___x_4104_);
return v___x_4106_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__4_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_4107_; lean_object* v___x_4108_; lean_object* v___x_4109_; 
v___x_4107_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__28_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2_));
v___x_4108_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__3_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__3_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__3_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_);
v___x_4109_ = l_Lean_Name_str___override(v___x_4108_, v___x_4107_);
return v___x_4109_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__5_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_4110_; lean_object* v___x_4111_; lean_object* v___x_4112_; 
v___x_4110_ = lean_unsigned_to_nat(2u);
v___x_4111_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__4_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__4_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__4_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_);
v___x_4112_ = l_Lean_Name_num___override(v___x_4111_, v___x_4110_);
return v___x_4112_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__9_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_(void){
_start:
{
uint8_t v___x_4120_; lean_object* v___x_4121_; lean_object* v___x_4122_; lean_object* v___x_4123_; lean_object* v___x_4124_; 
v___x_4120_ = 1;
v___x_4121_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__8_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_));
v___x_4122_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__7_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_));
v___x_4123_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__5_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__5_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__5_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_);
v___x_4124_ = lean_alloc_ctor(0, 3, 1);
lean_ctor_set(v___x_4124_, 0, v___x_4123_);
lean_ctor_set(v___x_4124_, 1, v___x_4122_);
lean_ctor_set(v___x_4124_, 2, v___x_4121_);
lean_ctor_set_uint8(v___x_4124_, sizeof(void*)*3, v___x_4120_);
return v___x_4124_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__10_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___f_4125_; lean_object* v___f_4126_; lean_object* v___x_4127_; lean_object* v___x_4128_; 
v___f_4125_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__0_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_));
v___f_4126_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__6_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_));
v___x_4127_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__9_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__9_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__9_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_);
v___x_4128_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_4128_, 0, v___x_4127_);
lean_ctor_set(v___x_4128_, 1, v___f_4126_);
lean_ctor_set(v___x_4128_, 2, v___f_4125_);
return v___x_4128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_4130_; lean_object* v___x_4131_; 
v___x_4130_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__10_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__10_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn___closed__10_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_);
v___x_4131_ = l_Lean_registerBuiltinAttribute(v___x_4130_);
return v___x_4131_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2____boxed(lean_object* v_a_4132_){
_start:
{
lean_object* v_res_4133_; 
v_res_4133_ = lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_();
return v_res_4133_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_ScopedEnvExtension_add___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__9(lean_object* v_00_u03b1_4134_, lean_object* v_00_u03b2_4135_, lean_object* v_00_u03c3_4136_, lean_object* v_ext_4137_, lean_object* v_b_4138_, uint8_t v_kind_4139_, lean_object* v___y_4140_, lean_object* v___y_4141_){
_start:
{
lean_object* v___x_4143_; 
v___x_4143_ = lp_mathlib_Lean_ScopedEnvExtension_add___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__9___redArg(v_ext_4137_, v_b_4138_, v_kind_4139_, v___y_4140_, v___y_4141_);
return v___x_4143_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_ScopedEnvExtension_add___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__9___boxed(lean_object* v_00_u03b1_4144_, lean_object* v_00_u03b2_4145_, lean_object* v_00_u03c3_4146_, lean_object* v_ext_4147_, lean_object* v_b_4148_, lean_object* v_kind_4149_, lean_object* v___y_4150_, lean_object* v___y_4151_, lean_object* v___y_4152_){
_start:
{
uint8_t v_kind_boxed_4153_; lean_object* v_res_4154_; 
v_kind_boxed_4153_ = lean_unbox(v_kind_4149_);
v_res_4154_ = lp_mathlib_Lean_ScopedEnvExtension_add___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__9(v_00_u03b1_4144_, v_00_u03b2_4145_, v_00_u03c3_4146_, v_ext_4147_, v_b_4148_, v_kind_boxed_4153_, v___y_4150_, v___y_4151_);
lean_dec(v___y_4151_);
lean_dec_ref(v___y_4150_);
return v_res_4154_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__11(lean_object* v_env_4155_, lean_object* v___y_4156_, lean_object* v___y_4157_){
_start:
{
lean_object* v___x_4159_; 
v___x_4159_ = lp_mathlib_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__11___redArg(v_env_4155_, v___y_4157_);
return v___x_4159_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__11___boxed(lean_object* v_env_4160_, lean_object* v___y_4161_, lean_object* v___y_4162_, lean_object* v___y_4163_){
_start:
{
lean_object* v_res_4164_; 
v_res_4164_ = lp_mathlib_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__11(v_env_4160_, v___y_4161_, v___y_4162_);
lean_dec(v___y_4162_);
lean_dec_ref(v___y_4161_);
return v_res_4164_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Aesop_Frontend_getGlobalRuleSetFromData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__13(lean_object* v_ext_4165_, lean_object* v_simpExt_4166_, lean_object* v_simprocExt_4167_, lean_object* v___y_4168_, lean_object* v___y_4169_){
_start:
{
lean_object* v___x_4171_; 
v___x_4171_ = lp_mathlib_Aesop_Frontend_getGlobalRuleSetFromData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__13___redArg(v_ext_4165_, v_simpExt_4166_, v_simprocExt_4167_, v___y_4169_);
return v___x_4171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Aesop_Frontend_getGlobalRuleSetFromData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__13___boxed(lean_object* v_ext_4172_, lean_object* v_simpExt_4173_, lean_object* v_simprocExt_4174_, lean_object* v___y_4175_, lean_object* v___y_4176_, lean_object* v___y_4177_){
_start:
{
lean_object* v_res_4178_; 
v_res_4178_ = lp_mathlib_Aesop_Frontend_getGlobalRuleSetFromData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__13(v_ext_4172_, v_simpExt_4173_, v_simprocExt_4174_, v___y_4175_, v___y_4176_);
lean_dec(v___y_4176_);
lean_dec_ref(v___y_4175_);
lean_dec_ref(v_simprocExt_4174_);
lean_dec_ref(v_simpExt_4173_);
lean_dec_ref(v_ext_4172_);
return v_res_4178_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__0(lean_object* v_00_u03b1_4179_, lean_object* v_msg_4180_, lean_object* v___y_4181_, lean_object* v___y_4182_){
_start:
{
lean_object* v___x_4184_; 
v___x_4184_ = lp_mathlib_Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__0___redArg(v_msg_4180_, v___y_4181_, v___y_4182_);
return v___x_4184_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__0___boxed(lean_object* v_00_u03b1_4185_, lean_object* v_msg_4186_, lean_object* v___y_4187_, lean_object* v___y_4188_, lean_object* v___y_4189_){
_start:
{
lean_object* v_res_4190_; 
v_res_4190_ = lp_mathlib_Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__0(v_00_u03b1_4185_, v_msg_4186_, v___y_4187_, v___y_4188_);
lean_dec(v___y_4188_);
lean_dec_ref(v___y_4187_);
return v_res_4190_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__10(lean_object* v_00_u03b2_4191_, lean_object* v_x_4192_, lean_object* v_x_4193_){
_start:
{
lean_object* v___x_4194_; 
v___x_4194_ = lp_mathlib_Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__10___redArg(v_x_4192_, v_x_4193_);
return v___x_4194_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__10___boxed(lean_object* v_00_u03b2_4195_, lean_object* v_x_4196_, lean_object* v_x_4197_){
_start:
{
lean_object* v_res_4198_; 
v_res_4198_ = lp_mathlib_Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__10(v_00_u03b2_4195_, v_x_4196_, v_x_4197_);
lean_dec_ref(v_x_4197_);
return v_res_4198_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1_spec__2(lean_object* v_00_u03b1_4199_, lean_object* v_rsName_4200_, lean_object* v_f_4201_, lean_object* v___y_4202_, lean_object* v___y_4203_){
_start:
{
lean_object* v___x_4205_; 
v___x_4205_ = lp_mathlib_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1_spec__2___redArg(v_rsName_4200_, v_f_4201_, v___y_4202_, v___y_4203_);
return v___x_4205_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1_spec__2___boxed(lean_object* v_00_u03b1_4206_, lean_object* v_rsName_4207_, lean_object* v_f_4208_, lean_object* v___y_4209_, lean_object* v___y_4210_, lean_object* v___y_4211_){
_start:
{
lean_object* v_res_4212_; 
v_res_4212_ = lp_mathlib_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__0_spec__1_spec__2(v_00_u03b1_4206_, v_rsName_4207_, v_f_4208_, v___y_4209_, v___y_4210_);
lean_dec(v___y_4210_);
lean_dec_ref(v___y_4209_);
return v_res_4212_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8_spec__10(lean_object* v_00_u03b2_4213_, lean_object* v_m_4214_, lean_object* v_a_4215_){
_start:
{
lean_object* v___x_4216_; 
v___x_4216_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8_spec__10___redArg(v_m_4214_, v_a_4215_);
return v___x_4216_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8_spec__10___boxed(lean_object* v_00_u03b2_4217_, lean_object* v_m_4218_, lean_object* v_a_4219_){
_start:
{
lean_object* v_res_4220_; 
v_res_4220_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8_spec__10(v_00_u03b2_4217_, v_m_4218_, v_a_4219_);
lean_dec(v_a_4219_);
lean_dec_ref(v_m_4218_);
return v_res_4220_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__10_spec__13(lean_object* v_00_u03b2_4221_, lean_object* v_x_4222_, size_t v_x_4223_, lean_object* v_x_4224_){
_start:
{
lean_object* v___x_4225_; 
v___x_4225_ = lp_mathlib_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__10_spec__13___redArg(v_x_4222_, v_x_4223_, v_x_4224_);
return v___x_4225_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__10_spec__13___boxed(lean_object* v_00_u03b2_4226_, lean_object* v_x_4227_, lean_object* v_x_4228_, lean_object* v_x_4229_){
_start:
{
size_t v_x_16125__boxed_4230_; lean_object* v_res_4231_; 
v_x_16125__boxed_4230_ = lean_unbox_usize(v_x_4228_);
lean_dec(v_x_4228_);
v_res_4231_ = lp_mathlib_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__10_spec__13(v_00_u03b2_4226_, v_x_4227_, v_x_16125__boxed_4230_, v_x_4229_);
lean_dec_ref(v_x_4229_);
return v_res_4231_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8_spec__10_spec__17(lean_object* v_00_u03b2_4232_, lean_object* v_a_4233_, lean_object* v_x_4234_){
_start:
{
lean_object* v___x_4235_; 
v___x_4235_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8_spec__10_spec__17___redArg(v_a_4233_, v_x_4234_);
return v___x_4235_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8_spec__10_spec__17___boxed(lean_object* v_00_u03b2_4236_, lean_object* v_a_4237_, lean_object* v_x_4238_){
_start:
{
lean_object* v_res_4239_; 
v_res_4239_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2__spec__1_spec__8_spec__10_spec__17(v_00_u03b2_4236_, v_a_4237_, v_x_4238_);
lean_dec(v_x_4238_);
lean_dec(v_a_4237_);
return v_res_4239_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__18(void){
_start:
{
lean_object* v___x_4298_; lean_object* v___x_4299_; 
v___x_4298_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__15));
v___x_4299_ = l_String_toRawSubstring_x27(v___x_4298_);
return v___x_4299_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__25(void){
_start:
{
lean_object* v___x_4315_; lean_object* v___x_4316_; 
v___x_4315_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__24));
v___x_4316_ = l_String_toRawSubstring_x27(v___x_4315_);
return v___x_4316_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__39(void){
_start:
{
lean_object* v___x_4344_; lean_object* v___x_4345_; 
v___x_4344_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Bound_scoreToConfig___closed__2));
v___x_4345_ = l_Lean_mkIdent(v___x_4344_);
return v___x_4345_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1(lean_object* v_x_4348_, lean_object* v_a_4349_, lean_object* v_a_4350_){
_start:
{
lean_object* v___x_4351_; uint8_t v___x_4352_; 
v___x_4351_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Bound_attrBound__forward___closed__1));
v___x_4352_ = l_Lean_Syntax_isOfKind(v_x_4348_, v___x_4351_);
if (v___x_4352_ == 0)
{
lean_object* v___x_4353_; lean_object* v___x_4354_; 
v___x_4353_ = lean_box(1);
v___x_4354_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4354_, 0, v___x_4353_);
lean_ctor_set(v___x_4354_, 1, v_a_4350_);
return v___x_4354_;
}
else
{
lean_object* v_quotContext_4355_; lean_object* v_currMacroScope_4356_; lean_object* v_ref_4357_; uint8_t v___x_4358_; lean_object* v___x_4359_; lean_object* v___x_4360_; lean_object* v___x_4361_; lean_object* v___x_4362_; lean_object* v___x_4363_; lean_object* v___x_4364_; lean_object* v___x_4365_; lean_object* v___x_4366_; lean_object* v___x_4367_; lean_object* v___x_4368_; lean_object* v___x_4369_; lean_object* v___x_4370_; lean_object* v___x_4371_; lean_object* v___x_4372_; lean_object* v___x_4373_; lean_object* v___x_4374_; lean_object* v___x_4375_; lean_object* v___x_4376_; lean_object* v___x_4377_; lean_object* v___x_4378_; lean_object* v___x_4379_; lean_object* v___x_4380_; lean_object* v___x_4381_; lean_object* v___x_4382_; lean_object* v___x_4383_; lean_object* v___x_4384_; lean_object* v___x_4385_; lean_object* v___x_4386_; lean_object* v___x_4387_; lean_object* v___x_4388_; lean_object* v___x_4389_; lean_object* v___x_4390_; lean_object* v___x_4391_; lean_object* v___x_4392_; lean_object* v___x_4393_; lean_object* v___x_4394_; lean_object* v___x_4395_; lean_object* v___x_4396_; lean_object* v___x_4397_; lean_object* v___x_4398_; lean_object* v___x_4399_; lean_object* v___x_4400_; lean_object* v___x_4401_; lean_object* v___x_4402_; lean_object* v___x_4403_; lean_object* v___x_4404_; lean_object* v___x_4405_; lean_object* v___x_4406_; lean_object* v___x_4407_; lean_object* v___x_4408_; lean_object* v___x_4409_; lean_object* v___x_4410_; lean_object* v___x_4411_; lean_object* v___x_4412_; lean_object* v___x_4413_; lean_object* v___x_4414_; lean_object* v___x_4415_; lean_object* v___x_4416_; lean_object* v___x_4417_; 
v_quotContext_4355_ = lean_ctor_get(v_a_4349_, 1);
v_currMacroScope_4356_ = lean_ctor_get(v_a_4349_, 2);
v_ref_4357_ = lean_ctor_get(v_a_4349_, 5);
v___x_4358_ = 0;
v___x_4359_ = l_Lean_SourceInfo_fromRef(v_ref_4357_, v___x_4358_);
v___x_4360_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__3));
v___x_4361_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__4));
lean_inc_n(v___x_4359_, 26);
v___x_4362_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4362_, 0, v___x_4359_);
lean_ctor_set(v___x_4362_, 1, v___x_4360_);
v___x_4363_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__6));
v___x_4364_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__8));
v___x_4365_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__10));
v___x_4366_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__12));
v___x_4367_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__14));
v___x_4368_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__15));
v___x_4369_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4369_, 0, v___x_4359_);
lean_ctor_set(v___x_4369_, 1, v___x_4368_);
v___x_4370_ = l_Lean_Syntax_node1(v___x_4359_, v___x_4367_, v___x_4369_);
v___x_4371_ = l_Lean_Syntax_node1(v___x_4359_, v___x_4366_, v___x_4370_);
v___x_4372_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__17));
v___x_4373_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__18, &lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__18_once, _init_lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__18);
v___x_4374_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__19));
lean_inc_n(v_currMacroScope_4356_, 2);
lean_inc_n(v_quotContext_4355_, 2);
v___x_4375_ = l_Lean_addMacroScope(v_quotContext_4355_, v___x_4374_, v_currMacroScope_4356_);
v___x_4376_ = lean_box(0);
v___x_4377_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_4377_, 0, v___x_4359_);
lean_ctor_set(v___x_4377_, 1, v___x_4373_);
lean_ctor_set(v___x_4377_, 2, v___x_4375_);
lean_ctor_set(v___x_4377_, 3, v___x_4376_);
v___x_4378_ = l_Lean_Syntax_node1(v___x_4359_, v___x_4372_, v___x_4377_);
v___x_4379_ = l_Lean_Syntax_node2(v___x_4359_, v___x_4365_, v___x_4371_, v___x_4378_);
v___x_4380_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__21));
v___x_4381_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__23));
v___x_4382_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__24));
v___x_4383_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4383_, 0, v___x_4359_);
lean_ctor_set(v___x_4383_, 1, v___x_4382_);
v___x_4384_ = l_Lean_Syntax_node1(v___x_4359_, v___x_4381_, v___x_4383_);
v___x_4385_ = l_Lean_Syntax_node1(v___x_4359_, v___x_4380_, v___x_4384_);
v___x_4386_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__25, &lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__25_once, _init_lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__25);
v___x_4387_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__26));
v___x_4388_ = l_Lean_addMacroScope(v_quotContext_4355_, v___x_4387_, v_currMacroScope_4356_);
v___x_4389_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_4389_, 0, v___x_4359_);
lean_ctor_set(v___x_4389_, 1, v___x_4386_);
lean_ctor_set(v___x_4389_, 2, v___x_4388_);
lean_ctor_set(v___x_4389_, 3, v___x_4376_);
v___x_4390_ = l_Lean_Syntax_node1(v___x_4359_, v___x_4372_, v___x_4389_);
v___x_4391_ = l_Lean_Syntax_node2(v___x_4359_, v___x_4365_, v___x_4385_, v___x_4390_);
v___x_4392_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__28));
v___x_4393_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__30));
v___x_4394_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__32));
v___x_4395_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__33));
v___x_4396_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4396_, 0, v___x_4359_);
lean_ctor_set(v___x_4396_, 1, v___x_4395_);
v___x_4397_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__34));
v___x_4398_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4398_, 0, v___x_4359_);
lean_ctor_set(v___x_4398_, 1, v___x_4397_);
v___x_4399_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__35));
v___x_4400_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4400_, 0, v___x_4359_);
lean_ctor_set(v___x_4400_, 1, v___x_4399_);
v___x_4401_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__36));
v___x_4402_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4402_, 0, v___x_4359_);
lean_ctor_set(v___x_4402_, 1, v___x_4401_);
v___x_4403_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__38));
v___x_4404_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__39, &lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__39_once, _init_lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__39);
v___x_4405_ = l_Lean_Syntax_node1(v___x_4359_, v___x_4403_, v___x_4404_);
v___x_4406_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__40));
v___x_4407_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4407_, 0, v___x_4359_);
lean_ctor_set(v___x_4407_, 1, v___x_4406_);
v___x_4408_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___closed__41));
v___x_4409_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4409_, 0, v___x_4359_);
lean_ctor_set(v___x_4409_, 1, v___x_4408_);
v___x_4410_ = l_Lean_Syntax_node7(v___x_4359_, v___x_4394_, v___x_4396_, v___x_4398_, v___x_4400_, v___x_4402_, v___x_4405_, v___x_4407_, v___x_4409_);
v___x_4411_ = l_Lean_Syntax_node1(v___x_4359_, v___x_4393_, v___x_4410_);
v___x_4412_ = l_Lean_Syntax_node1(v___x_4359_, v___x_4392_, v___x_4411_);
v___x_4413_ = l_Lean_Syntax_node2(v___x_4359_, v___x_4364_, v___x_4391_, v___x_4412_);
v___x_4414_ = l_Lean_Syntax_node2(v___x_4359_, v___x_4364_, v___x_4379_, v___x_4413_);
v___x_4415_ = l_Lean_Syntax_node1(v___x_4359_, v___x_4363_, v___x_4414_);
v___x_4416_ = l_Lean_Syntax_node2(v___x_4359_, v___x_4361_, v___x_4362_, v___x_4415_);
v___x_4417_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4417_, 0, v___x_4416_);
lean_ctor_set(v___x_4417_, 1, v_a_4350_);
return v___x_4417_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1___boxed(lean_object* v_x_4418_, lean_object* v_a_4419_, lean_object* v_a_4420_){
_start:
{
lean_object* v_res_4421_; 
v_res_4421_ = lp_mathlib_Mathlib_Tactic_Bound___aux__Mathlib__Tactic__Bound__Attribute______macroRules__Mathlib__Tactic__Bound__attrBound__forward__1(v_x_4418_, v_a_4419_, v_a_4420_);
lean_dec_ref(v_a_4419_);
return v_res_4421_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Bound_Init(uint8_t builtin);
lean_object* runtime_initialize_Qq_Qq(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Bound_Attribute(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Bound_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Qq_Qq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Bound_Attribute(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_1868565755____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Bound_Attribute_0__Mathlib_Tactic_Bound_initFn_00___x40_Mathlib_Tactic_Bound_Attribute_2543913104____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_aesop_Aesop(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Bound_Init(uint8_t builtin);
lean_object* initialize_Qq_Qq(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Bound_Attribute(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Bound_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Qq_Qq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Bound_Attribute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Bound_Attribute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Bound_Attribute(builtin);
}
#ifdef __cplusplus
}
#endif
